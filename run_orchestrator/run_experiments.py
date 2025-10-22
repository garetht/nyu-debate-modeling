import glob
import json
import os
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, cast

import yaml
from jsonschema import validate

from run_orchestrator.graph_generator import graph_results
from run_orchestrator.models import ConfigurationDetails, ExperimentConfigurationGroup, ExperimentConfiguration


def _load_configuration(config_name: str, *, validate_schema: bool) -> ExperimentConfiguration:
    yaml_path = f"run_orchestrator/runs/{config_name}.yaml"
    schema_path = "run_orchestrator/experiment_orchestrator.schema.json"

    if not os.path.exists(yaml_path):
        print(f"Error: {yaml_path} not found.")
        sys.exit(1)

    with open(yaml_path, "r") as yaml_file:
        data = yaml.safe_load(yaml_file)

    if validate_schema:
        if not os.path.exists(schema_path):
            print(f"Error: {schema_path} not found.")
            sys.exit(1)
        with open(schema_path, "r") as schema_file:
            schema = json.load(schema_file)
        validate(instance=data, schema=schema)

    return _parse_experiment_configuration(cast(Dict[str, Any], data))


def _parse_experiment_configuration(data: Mapping[str, Any]) -> ExperimentConfiguration:
    raw_configurations = cast(List[Dict[str, Any]], data["configurations"])
    return ExperimentConfiguration(
        configurations=[_parse_experiment_configuration_group(raw_group) for raw_group in raw_configurations]
    )


def _parse_experiment_configuration_group(raw_group: Mapping[str, Any]) -> ExperimentConfigurationGroup:
    instance_ip = cast(str, raw_group["instance_ip"])
    raw_config = cast(Optional[Dict[str, Any]], raw_group.get("configurations"))
    if raw_config is None:
        raise KeyError("Expected `configurations` key within configuration group.")
    configuration = _parse_configuration_details(raw_config)
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


def _parse_configuration_details(raw_config: Mapping[str, Any]) -> ConfigurationDetails:
    name = cast(str, raw_config["name"])
    num_iters = cast(int, raw_config["num_iters"])
    count = cast(int, raw_config["count"])
    starting_index = cast(Optional[int], raw_config.get("starting_index"))

    return ConfigurationDetails(
        name=name,
        num_iters=num_iters,
        count=count,
        starting_index=starting_index,
    )


def run_experiments(config_name: str, dry_run: bool = False) -> None:
    experiment_configuration = _load_configuration(config_name, validate_schema=True)

    for config_group in experiment_configuration.configurations:
        configuration = config_group.configurations
        command = [
            "./cli.sh",
            "--ip",
            config_group.instance_ip,
            "bg-task",
            "start",
            "-n",
            configuration.name.replace(" ", ""),
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

        for _ in range(configuration.count):
            print(f"Running command: {' '.join(command)}")
            if dry_run:
                continue
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


def download_results(config_name: str) -> None:
    experiment_configuration = _load_configuration(config_name, validate_schema=False)

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
        print("Usage: python run_orchestrator/run_experiments.py <start|merge-data|graph> [args]")
        sys.exit(1)

    subcommand = sys.argv[1]

    if subcommand == "start":
        if len(sys.argv) < 3:
            print("Usage: python run_orchestrator/run_experiments.py start <config_name>")
            sys.exit(1)
        config_name = sys.argv[2]
        run_experiments(config_name, dry_run=False)
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
        print("Usage: python run_orchestrator/run_experiments.py <start|dry-run|merge-data|graph> [args]")
        sys.exit(1)

if __name__ == "__main__":
    main()
