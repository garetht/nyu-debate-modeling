from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel

from run_orchestrator.recorder.task_database import TaskDatabase

DEFAULT_DATABASE_PATH = Path("run_orchestrator/recorder/tasks.sqlite3")


class RunTaskResponse(BaseModel):
    id: int
    run_name: str
    yaml_path: str
    created_at: str


class RunSubtaskResponse(BaseModel):
    id: int
    run_task_id: int
    base_task_name: str
    resolved_task_name: str
    ip_address: str
    command: str
    log_path: str
    configuration: Dict[str, Any]
    created_at: str
    logs_command: str


class RunWithSubtasksResponse(RunTaskResponse):
    subtasks: List[RunSubtaskResponse]


class RunDetailResponse(BaseModel):
    run: RunTaskResponse
    subtasks: List[RunSubtaskResponse]


def _resolve_database_path() -> Path:
    override = os.environ.get("TASK_DATABASE_PATH")
    if override is not None and override.strip() != "":
        return Path(override)
    return DEFAULT_DATABASE_PATH


def get_database() -> TaskDatabase:
    """Return a TaskDatabase instance that points at the configured SQLite file."""
    return TaskDatabase(_resolve_database_path())


app = FastAPI(title="Run Orchestrator Task Database API", version="1.0.0")


def _build_logs_command(ip_address: str, resolved_task_name: str) -> str:
    command_parts: List[str] = [
        "./cli.sh",
        "--ip",
        ip_address,
        "bg-task",
        "logs",
        "-f",
        resolved_task_name,
    ]
    return " ".join(command_parts)


def _row_to_run_task(row: sqlite3.Row) -> RunTaskResponse:
    return RunTaskResponse(
        id=int(row["id"]),
        run_name=str(row["run_name"]),
        yaml_path=str(row["yaml_path"]),
        created_at=str(row["created_at"]),
    )


def _row_to_subtask(row: sqlite3.Row) -> RunSubtaskResponse:
    raw_configuration = row["configuration_json"]
    configuration: Dict[str, Any]
    if isinstance(raw_configuration, str):
        try:
            configuration = json.loads(raw_configuration)
        except json.JSONDecodeError:
            configuration = {"raw": raw_configuration}
    else:
        configuration = {"raw": raw_configuration}
    return RunSubtaskResponse(
        id=int(row["id"]),
        run_task_id=int(row["run_task_id"]),
        base_task_name=str(row["base_task_name"]),
        resolved_task_name=str(row["resolved_task_name"]),
        ip_address=str(row["ip_address"]),
        command=str(row["command"]),
        log_path=str(row["log_path"]),
        configuration=configuration,
        created_at=str(row["created_at"]),
        logs_command=_build_logs_command(
            ip_address=str(row["ip_address"]),
            resolved_task_name=str(row["resolved_task_name"]),
        ),
    )


def _fetch_run(database: TaskDatabase, run_id: int) -> Optional[RunTaskResponse]:
    with database._connect() as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            """
            SELECT id, run_name, yaml_path, created_at
            FROM run_tasks
            WHERE id = ?;
            """,
            (run_id,),
        ).fetchone()
    if row is None:
        return None
    return _row_to_run_task(row)


def _fetch_runs(database: TaskDatabase) -> List[RunTaskResponse]:
    with database._connect() as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT id, run_name, yaml_path, created_at
            FROM run_tasks
            ORDER BY created_at DESC;
            """
        ).fetchall()
    return [_row_to_run_task(row) for row in rows]


def _fetch_subtasks(database: TaskDatabase, run_id: Optional[int] = None) -> List[RunSubtaskResponse]:
    query = """
        SELECT
            id,
            run_task_id,
            base_task_name,
            resolved_task_name,
            ip_address,
            command,
            log_path,
            configuration_json,
            created_at
        FROM run_subtasks
    """
    parameters: tuple[Any, ...] = ()
    if run_id is not None:
        query += " WHERE run_task_id = ?"
        parameters = (run_id,)
    query += " ORDER BY created_at DESC;"

    with database._connect() as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(query, parameters).fetchall()
    return [_row_to_subtask(row) for row in rows]


@app.get("/health", response_model=Dict[str, str])
def health_check() -> Dict[str, str]:
    """Simple health endpoint to confirm the API is responsive."""
    return {"status": "ok"}


@app.get("/runs", response_model=List[RunWithSubtasksResponse])
def list_runs(database: TaskDatabase = Depends(get_database)) -> List[RunWithSubtasksResponse]:
    """Return all recorded runs along with their subtasks ordered by recency."""
    runs = _fetch_runs(database)
    run_summaries: List[RunWithSubtasksResponse] = []
    for run in runs:
        subtasks = _fetch_subtasks(database, run_id=run.id)
        run_summaries.append(
            RunWithSubtasksResponse(
                id=run.id,
                run_name=run.run_name,
                yaml_path=run.yaml_path,
                created_at=run.created_at,
                subtasks=subtasks,
            )
        )
    return run_summaries


@app.get("/runs/{run_id}", response_model=RunDetailResponse)
def get_run(run_id: int, database: TaskDatabase = Depends(get_database)) -> RunDetailResponse:
    """Return a single run and its associated subtasks."""
    run = _fetch_run(database, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run with id {run_id} not found.")
    subtasks = _fetch_subtasks(database, run_id=run_id)
    return RunDetailResponse(run=run, subtasks=subtasks)


@app.get("/runs/{run_id}/subtasks", response_model=List[RunSubtaskResponse])
def list_run_subtasks(run_id: int, database: TaskDatabase = Depends(get_database)) -> List[RunSubtaskResponse]:
    """Return subtasks for a specific run."""
    run = _fetch_run(database, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run with id {run_id} not found.")
    return _fetch_subtasks(database, run_id=run_id)


@app.get("/subtasks", response_model=List[RunSubtaskResponse])
def list_subtasks(run_id: Optional[int] = None, database: TaskDatabase = Depends(get_database)) -> List[RunSubtaskResponse]:
    """Return all subtasks, optionally filtered by run identifier."""
    if run_id is not None:
        run = _fetch_run(database, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run with id {run_id} not found.")
    return _fetch_subtasks(database, run_id=run_id)
