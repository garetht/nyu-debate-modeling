from __future__ import annotations

import builtins
import io
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Sequence, Tuple

import pytest
from unittest.mock import MagicMock, mock_open

import run_orchestrator.run_experiments as run_experiments_module


def _build_popen_stub(
    commands: List[Sequence[str]],
    line_supplier: Optional[Callable[[int], Tuple[Sequence[str], Sequence[str]]]] = None,
) -> MagicMock:
    """Create a stub for subprocess.Popen that records commands."""
    invocation_index = 0

    def _popen(command: Sequence[str], **_: Any) -> MagicMock:
        nonlocal invocation_index
        commands.append(tuple(command))
        process_mock = MagicMock()
        if line_supplier is None:
            stdout_lines: Sequence[str] = []
            stderr_lines: Sequence[str] = []
        else:
            stdout_lines, stderr_lines = line_supplier(invocation_index)
        invocation_index += 1
        process_mock.stdout = stdout_lines  # type: ignore[assignment]
        process_mock.stderr = stderr_lines  # type: ignore[assignment]
        process_mock.wait.return_value = None

        context_manager = MagicMock()
        context_manager.__enter__.return_value = process_mock
        context_manager.__exit__.return_value = None
        return context_manager

    return MagicMock(side_effect=_popen)


def test_run_experiments_executes_commands_with_expected_arguments(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_name = "integration"
    expected_yaml_path = f"run_orchestrator/runs/{config_name}.yaml"
    expected_schema_path = "run_orchestrator/experiment_orchestrator.schema.json"

    def _fake_exists(path: str) -> bool:
        return path in {expected_yaml_path, expected_schema_path}

    sample_configuration: Dict[str, Any] = {
        "configurations": [
            {
                "instance_ip": "1.2.3.4",
                "configuration_filepath": "experiments/custom.yaml",
                "extant_debates_directory": "/tmp/debates",
                "configurations": {
                    "name": "Test Config",
                    "num_iters": 5,
                    "count": 2,
                    "starting_index": 3,
                },
            }
        ]
    }
    captured_validate_calls: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []

    monkeypatch.setattr(run_experiments_module.os.path, "exists", _fake_exists)

    yaml_open_mock: MagicMock = mock_open()
    real_open = builtins.open

    def _patched_open(path: str, *args: Any, **kwargs: Any) -> Any:
        if path == expected_yaml_path:
            return yaml_open_mock(path, *args, **kwargs)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", _patched_open)
    monkeypatch.setattr(run_experiments_module.yaml, "safe_load", lambda _: sample_configuration)

    def _fake_validate(*, instance: Dict[str, Any], schema: Dict[str, Any]) -> None:
        captured_validate_calls.append((instance, schema))

    monkeypatch.setattr(run_experiments_module, "validate", _fake_validate)

    generated_metadata: List[Tuple[str, str]] = [
        ("TestConfig-20240101-000001", "/remote/path/0.log"),
        ("TestConfig-20240101-000002", "/remote/path/1.log"),
    ]
    metadata_iter = iter(generated_metadata)

    def _metadata_side_effect(base_task_name: str) -> Tuple[str, str]:
        assert base_task_name == "TestConfig"
        return next(metadata_iter)

    metadata_mock = MagicMock(side_effect=_metadata_side_effect)
    monkeypatch.setattr(run_experiments_module, "_generate_task_metadata", metadata_mock)

    database_instance = MagicMock()
    run_identifier = 17
    database_instance.register_run.return_value = run_identifier
    task_database_ctor = MagicMock(return_value=database_instance)
    monkeypatch.setattr(run_experiments_module, "TaskDatabase", task_database_ctor)

    recorded_commands: List[Sequence[str]] = []
    popen_stub = _build_popen_stub(recorded_commands)
    monkeypatch.setattr(run_experiments_module.subprocess, "Popen", popen_stub)

    run_experiments_module.run_experiments(config_name, dry_run=False)

    out = capsys.readouterr().out
    expected_prefix = "Running command: ./cli.sh --ip 1.2.3.4"
    assert expected_prefix in out
    assert out.count("Running command:") == 2
    assert "Resolved task name: TestConfig-20240101-000001" in out
    assert "Log path: /remote/path/0.log" in out
    assert len(recorded_commands) == 2

    expected_command: Tuple[str, ...] = (
        "./cli.sh",
        "--ip",
        "1.2.3.4",
        "bg-task",
        "start",
        "-n",
        "TestConfig",
        "--",
        "python",
        "./scripts/run_debate.py",
        "--configuration_filepath=experiments/custom.yaml",
        "--configuration=Test Config",
        "--num_iters=5",
        "--starting_index=3",
        "--extant_debates_directory=/tmp/debates",
    )
    assert tuple(recorded_commands[0]) == expected_command
    assert len(captured_validate_calls) == 1
    called_instance, called_schema = captured_validate_calls[0]
    assert called_instance == sample_configuration
    assert isinstance(called_schema, dict)
    assert called_schema.get("title") == "Experiment Orchestrator Configuration Schema"

    task_database_ctor.assert_called_once_with(run_experiments_module.TASK_DATABASE_PATH)
    database_instance.register_run.assert_called_once()
    register_kwargs = database_instance.register_run.call_args.kwargs
    assert register_kwargs == {"run_name": config_name, "yaml_path": Path(expected_yaml_path)}

    assert database_instance.record_subtask.call_count == 2
    first_call_kwargs = database_instance.record_subtask.call_args_list[0].kwargs
    assert first_call_kwargs["run_task_id"] == run_identifier
    assert first_call_kwargs["base_task_name"] == "TestConfig"
    assert first_call_kwargs["resolved_task_name"] == generated_metadata[0][0]
    assert first_call_kwargs["ip_address"] == "1.2.3.4"
    assert tuple(first_call_kwargs["command"]) == expected_command
    assert first_call_kwargs["log_path"] == generated_metadata[0][1]
    assert first_call_kwargs["configuration"]["iteration"] == 1
    assert first_call_kwargs["configuration"]["configuration"]["name"] == "Test Config"

    second_call_kwargs = database_instance.record_subtask.call_args_list[1].kwargs
    assert second_call_kwargs["run_task_id"] == run_identifier
    assert second_call_kwargs["base_task_name"] == "TestConfig"
    assert second_call_kwargs["resolved_task_name"] == generated_metadata[1][0]
    assert second_call_kwargs["ip_address"] == "1.2.3.4"
    assert tuple(second_call_kwargs["command"]) == expected_command
    assert second_call_kwargs["log_path"] == generated_metadata[1][1]
    assert second_call_kwargs["configuration"]["iteration"] == 2
    assert second_call_kwargs["configuration"]["configuration"]["name"] == "Test Config"


def test_run_iterative_dpo_experiments_executes_commands_with_expected_arguments(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_name = "iterative-dpo"
    expected_yaml_path = f"run_orchestrator/runs_dpo/{config_name}.yaml"
    expected_schema_path = "run_orchestrator/experiment_orchestrator_dpo.schema.json"

    def _fake_exists(path: str) -> bool:
        return path in {expected_yaml_path, expected_schema_path}

    sample_configuration: Dict[str, Any] = {
        "configurations": [
            {
                "instance_ip": "5.5.5.5",
                "configuration_filepath": "train/configs/custom_dpo.yaml",
                "configurations": {
                    "name": "DPO Config",
                    "num_iters": 99,
                    "count": 2,
                },
            }
        ]
    }
    captured_validate_calls: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []

    monkeypatch.setattr(run_experiments_module.os.path, "exists", _fake_exists)

    yaml_open_mock: MagicMock = mock_open()
    real_open = builtins.open

    def _patched_open(path: str, *args: Any, **kwargs: Any) -> Any:
        if path == expected_yaml_path:
            return yaml_open_mock(path, *args, **kwargs)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", _patched_open)
    monkeypatch.setattr(run_experiments_module.yaml, "safe_load", lambda _: sample_configuration)

    def _fake_validate(*, instance: Dict[str, Any], schema: Dict[str, Any]) -> None:
        captured_validate_calls.append((instance, schema))

    monkeypatch.setattr(run_experiments_module, "validate", _fake_validate)

    generated_metadata: List[Tuple[str, str]] = [
        ("DPOConfig-20240101-000001", "/remote/path/dpo-0.log"),
        ("DPOConfig-20240101-000002", "/remote/path/dpo-1.log"),
    ]
    metadata_iter = iter(generated_metadata)

    def _metadata_side_effect(base_task_name: str) -> Tuple[str, str]:
        assert base_task_name == "DPOConfig"
        return next(metadata_iter)

    metadata_mock = MagicMock(side_effect=_metadata_side_effect)
    monkeypatch.setattr(run_experiments_module, "_generate_task_metadata", metadata_mock)

    database_instance = MagicMock()
    run_identifier = 23
    database_instance.register_run.return_value = run_identifier
    task_database_ctor = MagicMock(return_value=database_instance)
    monkeypatch.setattr(run_experiments_module, "TaskDatabase", task_database_ctor)

    recorded_commands: List[Sequence[str]] = []
    popen_stub = _build_popen_stub(recorded_commands)
    monkeypatch.setattr(run_experiments_module.subprocess, "Popen", popen_stub)

    run_experiments_module.run_iterative_dpo_experiments(config_name, dry_run=False)

    out = capsys.readouterr().out
    expected_prefix = "Running command: ./cli.sh --ip 5.5.5.5"
    assert expected_prefix in out
    assert out.count("Running command:") == 2
    assert "Resolved task name: DPOConfig-20240101-000001" in out
    assert "Log path: /remote/path/dpo-0.log" in out
    assert len(recorded_commands) == 2

    expected_command: Tuple[str, ...] = (
        "./cli.sh",
        "--ip",
        "5.5.5.5",
        "bg-task",
        "start",
        "-n",
        "DPOConfig",
        "--",
        "python",
        "scripts/run_iterative_dpo.py",
        "--configuration=DPO Config",
    )
    assert tuple(recorded_commands[0]) == expected_command

    assert len(captured_validate_calls) == 1
    called_instance, called_schema = captured_validate_calls[0]
    assert called_instance == sample_configuration
    assert isinstance(called_schema, dict)
    assert called_schema.get("title") == "Experiment Orchestrator DPO Configuration Schema"

    task_database_ctor.assert_called_once_with(run_experiments_module.TASK_DATABASE_PATH)
    database_instance.register_run.assert_called_once()
    register_kwargs = database_instance.register_run.call_args.kwargs
    assert register_kwargs == {"run_name": config_name, "yaml_path": Path(expected_yaml_path)}

    assert database_instance.record_subtask.call_count == 2
    first_call_kwargs = database_instance.record_subtask.call_args_list[0].kwargs
    assert first_call_kwargs["run_task_id"] == run_identifier
    assert first_call_kwargs["base_task_name"] == "DPOConfig"
    assert first_call_kwargs["resolved_task_name"] == generated_metadata[0][0]
    assert first_call_kwargs["ip_address"] == "5.5.5.5"
    assert tuple(first_call_kwargs["command"]) == expected_command
    assert first_call_kwargs["log_path"] == generated_metadata[0][1]
    assert first_call_kwargs["configuration"]["iteration"] == 1
    assert first_call_kwargs["configuration"]["configuration"]["name"] == "DPO Config"

    second_call_kwargs = database_instance.record_subtask.call_args_list[1].kwargs
    assert second_call_kwargs["run_task_id"] == run_identifier
    assert second_call_kwargs["base_task_name"] == "DPOConfig"
    assert second_call_kwargs["resolved_task_name"] == generated_metadata[1][0]
    assert second_call_kwargs["ip_address"] == "5.5.5.5"
    assert tuple(second_call_kwargs["command"]) == expected_command
    assert second_call_kwargs["log_path"] == generated_metadata[1][1]
    assert second_call_kwargs["configuration"]["iteration"] == 2
    assert second_call_kwargs["configuration"]["configuration"]["name"] == "DPO Config"


def test_run_experiments_dry_run_skips_subprocess(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_name = "dry"
    expected_yaml_path = f"run_orchestrator/runs/{config_name}.yaml"
    expected_schema_path = "run_orchestrator/experiment_orchestrator.schema.json"

    def _fake_exists(path: str) -> bool:
        return path in {expected_yaml_path, expected_schema_path}

    sample_configuration: Dict[str, Any] = {
        "configurations": [
            {
                "instance_ip": "9.9.9.9",
                "configurations": {
                    "name": "Dry Config",
                    "num_iters": 1,
                    "count": 1,
                },
            }
        ]
    }

    monkeypatch.setattr(run_experiments_module.os.path, "exists", _fake_exists)

    yaml_open_mock: MagicMock = mock_open()
    real_open = builtins.open

    def _patched_open(path: str, *args: Any, **kwargs: Any) -> Any:
        if path == expected_yaml_path:
            return yaml_open_mock(path, *args, **kwargs)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", _patched_open)
    monkeypatch.setattr(run_experiments_module.yaml, "safe_load", lambda _: sample_configuration)
    monkeypatch.setattr(run_experiments_module, "validate", lambda **_: None)

    metadata_mock = MagicMock(side_effect=[("DryConfig-1", "/logs/log1.log")])
    monkeypatch.setattr(run_experiments_module, "_generate_task_metadata", metadata_mock)

    popen_mock = MagicMock()
    monkeypatch.setattr(run_experiments_module.subprocess, "Popen", popen_mock)
    task_database_ctor = MagicMock()
    monkeypatch.setattr(run_experiments_module, "TaskDatabase", task_database_ctor)

    run_experiments_module.run_experiments(config_name, dry_run=True)

    out = capsys.readouterr().out
    expected_prefix = "Running command: ./cli.sh --ip 9.9.9.9"
    assert expected_prefix in out
    assert "Resolved task name: DryConfig-1" in out
    popen_mock.assert_not_called()
    task_database_ctor.assert_not_called()
    metadata_mock.assert_called_once_with("DryConfig")


def test_download_results_invokes_unique_hosts_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_name = "download"
    expected_yaml_path = f"run_orchestrator/runs/{config_name}.yaml"

    def _fake_exists(path: str) -> bool:
        return path == expected_yaml_path

    sample_configuration: Dict[str, Any] = {
        "configurations": [
            {
                "instance_ip": "1.2.3.4",
                "home_dir": "/home/user/a",
                "configurations": {
                    "name": "Download A",
                    "num_iters": 1,
                    "count": 1,
                },
            },
            {
                "instance_ip": "1.2.3.4",
                "home_dir": "/home/user/a",
                "configurations": {
                    "name": "Download A Replica",
                    "num_iters": 1,
                    "count": 1,
                },
            },
            {
                "instance_ip": "5.6.7.8",
                "home_dir": "/home/user/b",
                "configurations": {
                    "name": "Download B",
                    "num_iters": 1,
                    "count": 1,
                },
            },
        ]
    }

    monkeypatch.setattr(run_experiments_module.os.path, "exists", _fake_exists)

    yaml_open_mock: MagicMock = mock_open()
    real_open = builtins.open

    def _patched_open(path: str, *args: Any, **kwargs: Any) -> Any:
        if path == expected_yaml_path:
            return yaml_open_mock(path, *args, **kwargs)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", _patched_open)
    monkeypatch.setattr(run_experiments_module.yaml, "safe_load", lambda _: sample_configuration)

    recorded_commands: List[Sequence[str]] = []
    popen_stub = _build_popen_stub(recorded_commands)
    monkeypatch.setattr(run_experiments_module.subprocess, "Popen", popen_stub)

    run_experiments_module.download_results(config_name)

    assert len(recorded_commands) == 2
    expected_commands = {
        (
            "./cli.sh",
            "--ip",
            "1.2.3.4",
            "rsync-to-host",
            "--remote-path",
            "/home/user/a/outputs/",
            "--local-path",
            "./outputs/",
        ),
        (
            "./cli.sh",
            "--ip",
            "5.6.7.8",
            "rsync-to-host",
            "--remote-path",
            "/home/user/b/outputs/",
            "--local-path",
            "./outputs/",
        ),
    }
    assert {tuple(command) for command in recorded_commands} == expected_commands
