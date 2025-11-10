from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

from explorer.explorer_backend import server
from run_orchestrator.recorder.task_database import TaskDatabase
from run_orchestrator.evals_generator.config_spec import ConfigurationType
from run_orchestrator.evals_generator.configuration_name import ConfigurationName
from run_orchestrator.evals_generator.model_definitions import (
    ALL_VALID_DEBATERS,
    ALL_VALID_JUDGES,
)


@pytest.fixture()
def temporary_database(tmp_path: Path) -> TaskDatabase:
    database_path = tmp_path / "tasks.sqlite3"
    database = TaskDatabase(database_path)
    run_id = database.register_run(run_name="demo-run", yaml_path=tmp_path / "demo.yaml")
    database.record_subtask(
        run_task_id=run_id,
        base_task_name="DemoTask",
        resolved_task_name="DemoTask-1",
        ip_address="127.0.0.1",
        command=["python", "script.py", "--flag"],
        log_path="/tmp/demo.log",
        configuration={"iteration": 1},
    )
    return database


@pytest.fixture()
def api_client(temporary_database: TaskDatabase) -> TestClient:
    def _override_database() -> TaskDatabase:
        return temporary_database

    server.app.dependency_overrides[server.get_database] = _override_database
    try:
        yield TestClient(server.app)
    finally:
        server.app.dependency_overrides.pop(server.get_database, None)


def test_health_check(api_client: TestClient) -> None:
    response = api_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_list_runs(api_client: TestClient) -> None:
    response = api_client.get("/runs")
    assert response.status_code == 200
    runs: List[Dict[str, Any]] = response.json()
    assert len(runs) == 1
    run = runs[0]
    assert run["run_name"] == "demo-run"
    assert "subtasks" in run
    subtasks = run["subtasks"]
    assert isinstance(subtasks, list)
    assert len(subtasks) == 1
    subtask = subtasks[0]
    assert subtask["logs_command"] == "./cli.sh --ip 127.0.0.1 bg-task logs -f DemoTask-1"


def test_run_detail_includes_subtasks(api_client: TestClient) -> None:
    run_response = api_client.get("/runs")
    run_id = run_response.json()[0]["id"]

    response = api_client.get(f"/runs/{run_id}")
    assert response.status_code == 200
    payload = response.json()
    assert payload["run"]["id"] == run_id
    assert payload["run"]["run_name"] == "demo-run"
    assert len(payload["subtasks"]) == 1
    subtask = payload["subtasks"][0]
    assert subtask["resolved_task_name"] == "DemoTask-1"
    assert subtask["configuration"]["iteration"] == 1
    assert subtask["logs_command"] == "./cli.sh --ip 127.0.0.1 bg-task logs -f DemoTask-1"


def test_list_subtasks_with_filter(api_client: TestClient) -> None:
    run_response = api_client.get("/runs")
    run_id = run_response.json()[0]["id"]

    filtered_response = api_client.get(f"/subtasks?run_id={run_id}")
    assert filtered_response.status_code == 200
    filtered_subtasks: List[Dict[str, Any]] = filtered_response.json()
    assert len(filtered_subtasks) == 1

    not_found_response = api_client.get("/subtasks?run_id=9999")
    assert not_found_response.status_code == 404


def test_subtask_base_task_configuration_is_structured(
    api_client: TestClient,
    temporary_database: TaskDatabase,
) -> None:
    debater_key = "llama-3-262k"
    judge_key = "gpt-4-turbo-2024-04-09"
    task_type_name = "lojban"
    debater_config = ALL_VALID_DEBATERS[debater_key]
    judge_config = ALL_VALID_JUDGES[judge_key]
    configuration_name = ConfigurationName._create(
        config_type=ConfigurationType.EVAL,
        debater_key=debater_key,
        judge_key=judge_key,
        task_type_name=task_type_name,
    )
    base_task_name = configuration_name.serialize()
    run_id = temporary_database.register_run(
        run_name="config-run",
        yaml_path=Path("/tmp/config.yaml"),
    )
    temporary_database.record_subtask(
        run_task_id=run_id,
        base_task_name=base_task_name,
        resolved_task_name="config-run-subtask",
        ip_address="10.0.0.2",
        command=["python", "task.py"],
        log_path="/tmp/config.log",
        configuration={"iteration": 2},
    )

    runs_response = api_client.get("/runs")
    assert runs_response.status_code == 200
    runs: List[Dict[str, Any]] = runs_response.json()
    config_run = next(run for run in runs if run["run_name"] == "config-run")
    subtask = config_run["subtasks"][0]

    base_task_configuration = subtask["base_task_configuration"]
    assert base_task_configuration is not None
    assert base_task_configuration["config_type"] == "eval"
    assert base_task_configuration["task_type_name"] == task_type_name

    debater_details = base_task_configuration["debater"]
    judge_details = base_task_configuration["judge"]

    def _model_type_name(raw_value: object) -> str:
        name = getattr(raw_value, "name", None)
        return name.lower() if isinstance(name, str) else str(raw_value)

    assert debater_details["key"] == debater_key
    assert debater_details["training_round"] == debater_config.training_round.display_name
    assert debater_details["model_type"] == _model_type_name(debater_config.settings.model_type)
    assert debater_details["model_file_path"] == debater_config.settings.model_file_path

    assert judge_details["key"] == judge_key
    assert judge_details["training_round"] == judge_config.training_round.display_name
    assert judge_details["model_type"] == _model_type_name(judge_config.settings.model_type)
    assert judge_details["model_file_path"] == judge_config.settings.model_file_path

    run_detail_response = api_client.get(f"/runs/{config_run['id']}")
    assert run_detail_response.status_code == 200
    run_detail_payload = run_detail_response.json()
    detail_subtask = run_detail_payload["subtasks"][0]
    assert detail_subtask["base_task_configuration"] == base_task_configuration

    subtasks_response = api_client.get(f"/subtasks?run_id={config_run['id']}")
    assert subtasks_response.status_code == 200
    filtered_subtask = subtasks_response.json()[0]
    assert filtered_subtask["base_task_configuration"] == base_task_configuration


@dataclass
class _StubLookupResult:
    original_command: str
    success: bool
    remote_command: str | None
    pid: int | None
    ps_line: str | None
    error: str | None


class _StubSSHClient:
    def __init__(self, host: str) -> None:
        self.host = host

    def find_process_ids(self, commands: List[str]) -> List[_StubLookupResult]:
        return [
            _StubLookupResult(
                original_command=command,
                success=True,
                remote_command=command.split(" -- ", 1)[1] if " -- " in command else command,
                pid=4321,
                ps_line=f"stubuser 4321  0.0  0.1  1234  5678 ?        S    12:00   0:00 {command}",
                error=None,
            )
            for command in commands
        ]


def test_list_run_processes_returns_ps_data(
    api_client: TestClient,
    temporary_database: TaskDatabase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_response = api_client.get("/runs")
    run_response.raise_for_status()
    run_id = run_response.json()[0]["id"]

    monkeypatch.setattr(server, "_create_ssh_client", lambda host: _StubSSHClient(host))

    response = api_client.get(f"/runs/{run_id}/processes")
    assert response.status_code == 200

    payload: List[Dict[str, Any]] = response.json()
    assert len(payload) == 1
    process_entry = payload[0]
    assert process_entry["pid"] == 4321
    assert "ps_line" in process_entry and "stubuser 4321" in process_entry["ps_line"]
    assert process_entry["success"] is True


def test_hide_run_marks_run_hidden(api_client: TestClient, temporary_database: TaskDatabase) -> None:
    runs_response = api_client.get("/runs")
    runs_response.raise_for_status()
    runs_payload: List[Dict[str, Any]] = runs_response.json()
    run_entry: Dict[str, Any] = runs_payload[0]
    run_id: int = int(run_entry["id"])
    assert run_entry["is_hidden"] is False

    hide_response = api_client.post(f"/runs/{run_id}/hide")
    assert hide_response.status_code == 200
    hide_payload: Dict[str, Any] = hide_response.json()
    assert hide_payload["id"] == run_id
    assert hide_payload["is_hidden"] is True

    refreshed_runs = api_client.get("/runs")
    refreshed_runs.raise_for_status()
    refreshed_payload: List[Dict[str, Any]] = refreshed_runs.json()
    refreshed_entry: Dict[str, Any] = next(run for run in refreshed_payload if run["id"] == run_id)
    assert refreshed_entry["is_hidden"] is True


def test_hide_run_returns_404_for_missing_run(api_client: TestClient) -> None:
    response = api_client.post("/runs/99999/hide")
    assert response.status_code == 404
