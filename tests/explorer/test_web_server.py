from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

from explorer.web import server
from run_orchestrator.recorder.task_database import TaskDatabase


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
