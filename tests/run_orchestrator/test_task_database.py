from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import cast

from pytest import MonkeyPatch

from run_orchestrator.task_database import TaskDatabase


def _run_timestamp() -> str:
    return "2024-01-01T00:00:00+00:00"


def _subtask_timestamp() -> str:
    return "2024-01-02T00:00:00+00:00"


def test_register_run_persists_expected_fields(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    db_path: Path = tmp_path / "tasks.sqlite3"
    database: TaskDatabase = TaskDatabase(db_path)
    monkeypatch.setattr(TaskDatabase, "_timestamp", staticmethod(_run_timestamp))

    yaml_path: Path = tmp_path / "config.yaml"
    run_name: str = "sample-run"
    run_id: int = database.register_run(run_name, yaml_path)

    assert run_id == 1

    connection: sqlite3.Connection
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        cursor: sqlite3.Cursor = connection.execute(
            "SELECT run_name, yaml_path, created_at FROM run_tasks WHERE id = ?",
            (run_id,),
        )
        row = cursor.fetchone()

    assert row is not None
    typed_row_run: sqlite3.Row = cast(sqlite3.Row, row)
    assert typed_row_run["run_name"] == run_name
    assert typed_row_run["yaml_path"] == str(yaml_path)
    assert typed_row_run["created_at"] == _run_timestamp()


def test_record_subtask_stores_joined_command_and_sorted_configuration(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    db_path: Path = tmp_path / "tasks.sqlite3"
    database: TaskDatabase = TaskDatabase(db_path)

    monkeypatch.setattr(TaskDatabase, "_timestamp", staticmethod(_run_timestamp))
    run_id: int = database.register_run("parent-run", tmp_path / "config.yaml")

    monkeypatch.setattr(TaskDatabase, "_timestamp", staticmethod(_subtask_timestamp))
    command: list[str] = ["python", "runner.py", "--flag", "value"]
    configuration: dict[str, int] = {"b": 2, "a": 1}
    database.record_subtask(
        run_task_id=run_id,
        base_task_name="base-task",
        resolved_task_name="resolved-task",
        ip_address="127.0.0.1",
        command=command,
        log_path="/tmp/log.txt",
        configuration=configuration,
    )

    connection: sqlite3.Connection
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        cursor: sqlite3.Cursor = connection.execute(
            """
            SELECT
                run_task_id,
                base_task_name,
                resolved_task_name,
                ip_address,
                command,
                log_path,
                configuration_json,
                created_at
            FROM run_subtasks
            WHERE run_task_id = ?
            """,
            (run_id,),
        )
        row = cursor.fetchone()

    assert row is not None
    typed_row_subtask: sqlite3.Row = cast(sqlite3.Row, row)
    assert typed_row_subtask["run_task_id"] == run_id
    assert typed_row_subtask["base_task_name"] == "base-task"
    assert typed_row_subtask["resolved_task_name"] == "resolved-task"
    assert typed_row_subtask["ip_address"] == "127.0.0.1"
    assert typed_row_subtask["command"] == " ".join(command)
    assert typed_row_subtask["log_path"] == "/tmp/log.txt"
    assert typed_row_subtask["configuration_json"] == json.dumps(configuration, sort_keys=True)
    assert typed_row_subtask["created_at"] == _subtask_timestamp()
