from __future__ import annotations

import getpass
import glob
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, cast

import yaml
from jsonschema import validate

from run_orchestrator.graph_generator import graph_results
from run_orchestrator.models import ConfigurationDetails, ExperimentConfigurationGroup, ExperimentConfiguration
from run_orchestrator.recorder.task_database import TaskDatabase

TASK_DATABASE_PATH = Path("run_orchestrator/recorder/tasks.sqlite3")


def _generate_task_metadata(base_task_name: str) -> tuple[str, str]:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    resolved_task_name = f"{base_task_name}-{timestamp}"
    user_name = getpass.getuser()
    log_path = f"/home/ubuntu/mars-arnesen-gh/{user_name}/logs/{resolved_task_name}.log"
    return resolved_task_name, log_path


def _build_subtask_configuration(
    config_group: ExperimentConfigurationGroup,
    configuration: ConfigurationDetails,
    iteration_index: int,
) -> Dict[str, Any]:
    return {
        "group": {
            "instance_ip": config_group.instance_ip,
            "configuration_filepath": config_group.configuration_filepath,
            "extant_debates_directory": config_group.extant_debates_directory,
            "home_dir": config_group.home_dir,
        },
        "configuration": {
            "name": configuration.name,
            "num_iters": configuration.num_iters,
            "count": configuration.count,
            "starting_index": configuration.starting_index,
            "specified_debate_identifiers": configuration.specified_debate_identifiers,
        },
        "iteration": iteration_index,
    }


CommandBuilder = Callable[[ExperimentConfigurationGroup, ConfigurationDetails, str], List[str]]


def _build_debate_command(
    config_group: ExperimentConfigurationGroup,
    configuration: ConfigurationDetails,
    base_task_name: str,
) -> List[str]:
    command: List[str] = [
        "./cli.sh",
        "--ip",
        config_group.instance_ip,
        "bg-task",
        "start",
        "-n",
        base_task_name,
        "--",
        "python",
        "./scripts/run_debate.py",
        f"--configuration_filepath={config_group.configuration_filepath}",
        f"--configuration={configuration.name}",
        f"--num_iters={configuration.num_iters}",
    ]

    if configuration.starting_index is not None:
        command.append(f"--starting_index={configuration.starting_index}")

    if config_group.extant_debates_directory is not None:
        command.append(f"--extant_debates_directory={config_group.extant_debates_directory}")

    if configuration.specified_debate_identifiers:
        command.append("--specified-debate-identifiers")
        command.extend(configuration.specified_debate_identifiers)

    return command


def _build_iterative_dpo_command(
    config_group: ExperimentConfigurationGroup,
    configuration: ConfigurationDetails,
    base_task_name: str,
) -> List[str]:
    return [
        "./cli.sh",
        "--ip",
        config_group.instance_ip,
        "bg-task",
        "start",
        "-n",
        base_task_name,
        "--",
        "python",
        "scripts/run_iterative_dpo.py",
        f"--configuration={configuration.name}",
    ]


def _stream_command_output(command: Sequence[str]) -> None:
    with subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    ) as process:
        stdout = process.stdout
        if stdout is not None:
            for line in stdout:
                print(line, end="")
        stderr = process.stderr
        if stderr is not None:
            for line in stderr:
                print(line, end="")
        process.wait()


def _load_configuration(
    config_name: str,
    *,
    validate_schema: bool,
    config_directory: str,
    schema_path: Optional[str],
    allow_missing_iter_params: bool,
) -> ExperimentConfiguration:
    yaml_path = Path(config_directory) / f"{config_name}.yaml"

    yaml_path_str = str(yaml_path)

    if not os.path.exists(yaml_path_str):
        print(f"Error: {yaml_path_str} not found.")
        sys.exit(1)

    with open(yaml_path_str, "r") as yaml_file:
        data = yaml.safe_load(yaml_file)

    if validate_schema:
        if schema_path is None:
            raise ValueError("Schema path must be provided when validate_schema is True.")
        if not os.path.exists(schema_path):
            print(f"Error: {schema_path} not found.")
            sys.exit(1)
        with open(schema_path, "r") as schema_file:
            schema = json.load(schema_file)
        validate(instance=data, schema=schema)

    return _parse_experiment_configuration(
        cast(Dict[str, Any], data),
        allow_missing_iter_params=allow_missing_iter_params,
    )


def _parse_experiment_configuration(
    data: Mapping[str, Any],
    *,
    allow_missing_iter_params: bool,
) -> ExperimentConfiguration:
    raw_configurations = cast(List[Dict[str, Any]], data["configurations"])
    return ExperimentConfiguration(
        configurations=[
            _parse_experiment_configuration_group(
                raw_group,
                allow_missing_iter_params=allow_missing_iter_params,
            )
            for raw_group in raw_configurations
        ]
    )


def _parse_experiment_configuration_group(
    raw_group: Mapping[str, Any],
    *,
    allow_missing_iter_params: bool,
) -> ExperimentConfigurationGroup:
    instance_ip = cast(str, raw_group["instance_ip"])
    raw_config = cast(Optional[Dict[str, Any]], raw_group.get("configurations"))
    if raw_config is None:
        raise KeyError("Expected `configurations` key within configuration group.")
    configuration = _parse_configuration_details(
        raw_config,
        allow_missing_iter_params=allow_missing_iter_params,
    )
    configuration_filepath = cast(Optional[str], raw_group.get("configuration_filepath"))
    if configuration_filepath is None:
        configuration_filepath = "experiments/configs/standard_experiment.yaml"
    extant_debates_directory = cast(Optional[str], raw_group.get("extant_debates_directory"))
    home_dir = cast(Optional[str], raw_group.get("home_dir"))

    return ExperimentConfigurationGroup(
        instance_ip=instance_ip,
        configurations=configuration,
        configuration_filepath=configuration_filepath,
        extant_debates_directory=extant_debates_directory,
        home_dir=home_dir,
    )


def _parse_configuration_details(
    raw_config: Mapping[str, Any],
    *,
    allow_missing_iter_params: bool,
) -> ConfigurationDetails:
    name = cast(str, raw_config["name"])
    starting_index = cast(Optional[int], raw_config.get("starting_index"))
    raw_specified_debate_identifiers = raw_config.get("specified_debate_identifiers")
    specified_debate_identifiers: Optional[List[str]]
    if raw_specified_debate_identifiers is None:
        specified_debate_identifiers = None
    elif isinstance(raw_specified_debate_identifiers, Sequence) and not isinstance(raw_specified_debate_identifiers, (str, bytes)):
        specified_debate_identifiers = [
            cast(str, identifier) for identifier in raw_specified_debate_identifiers
        ]
    else:
        raise TypeError("specified_debate_identifiers must be a sequence of strings if provided.")
    if allow_missing_iter_params:
        num_iters_value = cast(Optional[int], raw_config.get("num_iters"))
        if num_iters_value is None:
            num_iters_value = 1
        count_value = cast(Optional[int], raw_config.get("count"))
        if count_value is None:
            count_value = 1
    else:
        num_iters_value = cast(int, raw_config["num_iters"])
        count_value = cast(int, raw_config["count"])

    return ConfigurationDetails(
        name=name,
        num_iters=num_iters_value,
        count=count_value,
        starting_index=starting_index,
        specified_debate_identifiers=specified_debate_identifiers,
    )


def _run_configuration_groups(
    config_name: str,
    *,
    command_builder: CommandBuilder,
    dry_run: bool,
    config_directory: str,
    schema_path: Optional[str],
    allow_missing_iter_params: bool,
    validate_schema: bool,
) -> None:
    yaml_path = Path(config_directory) / f"{config_name}.yaml"
    experiment_configuration = _load_configuration(
        config_name,
        validate_schema=validate_schema,
        config_directory=config_directory,
        schema_path=schema_path,
        allow_missing_iter_params=allow_missing_iter_params,
    )

    task_database: Optional[TaskDatabase] = None
    run_task_id: Optional[int] = None
    if not dry_run:
        task_database = TaskDatabase(TASK_DATABASE_PATH)
        run_task_id = task_database.register_run(run_name=config_name, yaml_path=yaml_path)

    for config_group in experiment_configuration.configurations:
        configuration = config_group.configurations
        base_task_name = configuration.name.replace(" ", "")
        command = command_builder(config_group, configuration, base_task_name)

        for iteration_index in range(1, configuration.count + 1):
            print(f"Running command: {' '.join(command)}")
            resolved_task_name, log_path = _generate_task_metadata(base_task_name)
            print(f"Resolved task name: {resolved_task_name}")
            print(f"Log path: {log_path}")
            if dry_run:
                continue
            _stream_command_output(command)
            if task_database is None or run_task_id is None:
                continue
            subtask_configuration: Dict[str, Any] = _build_subtask_configuration(
                config_group,
                configuration,
                iteration_index,
            )
            task_database.record_subtask(
                run_task_id=run_task_id,
                base_task_name=base_task_name,
                resolved_task_name=resolved_task_name,
                ip_address=config_group.instance_ip,
                command=command,
                log_path=log_path,
                configuration=subtask_configuration,
            )


def run_experiments(config_name: str, dry_run: bool = False) -> None:
    _run_configuration_groups(
        config_name,
        command_builder=_build_debate_command,
        dry_run=dry_run,
        config_directory="run_orchestrator/runs",
        schema_path="run_orchestrator/experiment_orchestrator.schema.json",
        allow_missing_iter_params=False,
        validate_schema=True,
    )


def run_iterative_dpo_experiments(config_name: str, dry_run: bool = False) -> None:
    _run_configuration_groups(
        config_name,
        command_builder=_build_iterative_dpo_command,
        dry_run=dry_run,
        config_directory="run_orchestrator/runs_dpo",
        schema_path="run_orchestrator/experiment_orchestrator_dpo.schema.json",
        allow_missing_iter_params=True,
        validate_schema=True,
    )


def download_results(config_name: str) -> None:
    experiment_configuration = _load_configuration(
        config_name,
        validate_schema=False,
        config_directory="run_orchestrator/runs",
        schema_path=None,
        allow_missing_iter_params=False,
    )

    ip_home_dirs: set[tuple[str, str]] = set()
    missing_home_dirs: list[str] = []

    for config_group in experiment_configuration.configurations:
        if config_group.home_dir is None:
            missing_home_dirs.append(config_group.instance_ip)
            continue
        ip_home_dirs.add((config_group.instance_ip, config_group.home_dir))

    if missing_home_dirs:
        raise ValueError(
            "The following instances are missing `home_dir` entries: "
            + ", ".join(sorted(set(missing_home_dirs)))
        )

    for ip, home_dir in ip_home_dirs:
        remote_path = f"{home_dir}/outputs/"
        local_path = "./outputs/"

        command = [
            "./cli.sh",
            "--ip",
            ip,
            "rsync-to-host",
            "--remote-path",
            remote_path,
            "--local-path",
            local_path,
        ]

        print(f"Running command: {' '.join(command)}")

        with subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        ) as process:
            for line in process.stdout:
                print(line, end="")
            for line in process.stderr:
                print(line, end="")


def merge_data(stats_dir: str = "outputs/stats") -> None:
    file_paths = glob.glob(os.path.join(stats_dir, "*.json"))
    # Exclude already merged files
    file_paths = [p for p in file_paths if not os.path.basename(p).startswith("merge-")]

    if not file_paths:
        print(f"No stats files found to merge in {stats_dir}.")
        return

    merged_data: list[Any] = []

    for file_path in file_paths:
        with open(file_path, "r") as f:
            data = json.load(f)
            if not merged_data:
                if not isinstance(data, list):
                    raise ValueError(f"Expected list data in {file_path}.")
                merged_data = data
            else:
                if not isinstance(data, list):
                    raise ValueError(f"Expected list data in {file_path}.")
                for i, item in enumerate(data):
                    if i < len(merged_data):
                        merge_recursively(merged_data[i], item)
                    else:
                        merged_data.append(item)

    recalculate_recursively(merged_data)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f")
    merged_filename = os.path.join(stats_dir, f"merge-{timestamp}.json")

    with open(merged_filename, "w") as f:
        json.dump(merged_data, f, indent=2)

    print(f"Merged stats saved to {merged_filename}")

def merge_recursively(d1: dict[str, Any], d2: dict[str, Any]) -> None:
    for k, v2 in d2.items():
        if k in d1:
            v1 = d1[k]
            if isinstance(v1, dict) and isinstance(v2, dict):
                merge_recursively(v1, v2)
            elif isinstance(v1, list) and isinstance(v2, list):
                v1.extend(v2)
            elif isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                if "average" not in k and "pct" not in k:
                    d1[k] = v1 + v2
        else:
            d1[k] = v2

def recalculate_recursively(data: Any) -> None:
    if isinstance(data, dict):
        for value in data.values():
            recalculate_recursively(value)

        if data.get("matches", 0) > 0:
            matches = data["matches"]
            if "wins" in data:
                data["average_reward"] = data["wins"] / matches
            if "binary_wins" in data:
                data["binary_win_pct"] = data["binary_wins"] / matches
            if (
                "correct_wins" in data
                and "correct_matches" in data
                and data["correct_matches"] > 0
            ):
                data["average_correct_reward"] = data["correct_wins"] / data["correct_matches"]
            if "first_wins" in data and "first_matches" in data and data["first_matches"] > 0:
                data["average_first_wins"] = data["first_wins"] / data["first_matches"]
            if (
                "wins" in data
                and "first_wins" in data
                and "matches" in data
                and "first_matches" in data
                and (matches - data["first_matches"]) > 0
            ):
                data["average_second_wins"] = (data["wins"] - data["first_wins"]) / (
                    matches - data["first_matches"]
                )

    elif isinstance(data, list):
        for item in data:
            recalculate_recursively(item)

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python run_orchestrator/run_experiments.py <start|start-dpo|dry-run|merge-data|graph> [args]")
        sys.exit(1)

    subcommand = sys.argv[1]

    if subcommand == "start":
        if len(sys.argv) < 3:
            print("Usage: python run_orchestrator/run_experiments.py start <config_name>")
            sys.exit(1)
        config_name = sys.argv[2]
        run_experiments(config_name, dry_run=False)
    elif subcommand == "start-dpo":
        if len(sys.argv) < 3:
            print("Usage: python run_orchestrator/run_experiments.py start-dpo <config_name>")
            sys.exit(1)
        config_name = sys.argv[2]
        run_iterative_dpo_experiments(config_name, dry_run=False)
    elif subcommand == "dry-run":
        if len(sys.argv) < 3:
            print("Usage: python run_orchestrator/run_experiments.py dry-run <config_name>")
            sys.exit(1)
        config_name = sys.argv[2]
        run_experiments(config_name, dry_run=True)
    elif subcommand == "merge-data":
        stats_dir = "outputs/stats"
        if len(sys.argv) > 2:
            stats_dir = os.path.join(sys.argv[2], "outputs/stats")
        merge_data(stats_dir)
    elif subcommand == "graph":
        if len(sys.argv) < 3:
            print("Usage: python run_orchestrator/run_experiments.py graph <file_path>")
            sys.exit(1)
        file_path = sys.argv[2]
        graph_results(file_path)
    else:
        print(f"Unknown subcommand: {subcommand}")
        print("Usage: python run_orchestrator/run_experiments.py <start|start-dpo|dry-run|merge-data|graph> [args]")
        sys.exit(1)

if __name__ == "__main__":
    main()
