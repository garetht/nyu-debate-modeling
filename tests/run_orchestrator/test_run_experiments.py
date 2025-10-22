from __future__ import annotations

import builtins
import io
import json
import os
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Sequence, Tuple

import pytest
from unittest.mock import MagicMock, mock_open

import run_orchestrator.run_experiments as run_experiments_module


def _build_popen_stub(commands: List[Sequence[str]]) -> MagicMock:
    """Create a stub for subprocess.Popen that records commands."""
    def _popen(command: Sequence[str], **_: Any) -> MagicMock:
        commands.append(tuple(command))
        process_mock = MagicMock()
        process_mock.stdout = []  # type: ignore[assignment]
        process_mock.stderr = []  # type: ignore[assignment]

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

    recorded_commands: List[Sequence[str]] = []
    popen_stub = _build_popen_stub(recorded_commands)
    monkeypatch.setattr(run_experiments_module.subprocess, "Popen", popen_stub)

    run_experiments_module.run_experiments(config_name, dry_run=False)

    out = capsys.readouterr().out
    expected_prefix = "Running command: ./cli.sh --ip 1.2.3.4"
    assert expected_prefix in out
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

    popen_mock = MagicMock()
    monkeypatch.setattr(run_experiments_module.subprocess, "Popen", popen_mock)

    run_experiments_module.run_experiments(config_name, dry_run=True)

    out = capsys.readouterr().out
    expected_prefix = "Running command: ./cli.sh --ip 9.9.9.9"
    assert expected_prefix in out
    popen_mock.assert_not_called()


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
