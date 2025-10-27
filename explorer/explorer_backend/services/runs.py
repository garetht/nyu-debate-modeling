from __future__ import annotations

import json
import sqlite3
from typing import Any, Dict, List

from explorer.explorer_backend.models import (
    RunDetailResponse,
    RunSubtaskConfigurationName,
    RunSubtaskModelInfo,
    RunSubtaskResponse,
    RunTaskResponse,
    RunWithSubtasksResponse,
)
from explorer.errors.runs import RunNotFoundError
from models.model import ModelType
from run_orchestrator.evals_generator.configuration_name import ConfigurationName
from run_orchestrator.recorder.task_database import TaskDatabase


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


def _model_type_to_string(model_type: str | ModelType) -> str:
    if isinstance(model_type, ModelType):
        return model_type.name.lower()
    return str(model_type)


def _row_to_subtask(row: sqlite3.Row) -> RunSubtaskResponse:
    raw_configuration: Any = row["configuration_json"]
    configuration: Dict[str, Any]
    if isinstance(raw_configuration, str):
        try:
            configuration = json.loads(raw_configuration)
        except json.JSONDecodeError:
            configuration = {"raw": raw_configuration}
    else:
        configuration = {"raw": raw_configuration}
    raw_base_task_name_value: Any = row["base_task_name"]
    base_task_name: str = str(raw_base_task_name_value)
    base_task_configuration: RunSubtaskConfigurationName | None = None
    try:
        configuration_name: ConfigurationName = ConfigurationName.deserialize(base_task_name)
    except ValueError:
        base_task_configuration = None
    else:
        debater_config = configuration_name.debater_config
        judge_config = configuration_name.judge_config
        base_task_configuration = RunSubtaskConfigurationName(
            config_type=str(configuration_name.config_type.value),
            task_type_name=configuration_name.task_type_name,
            debater=RunSubtaskModelInfo(
                key=configuration_name.debater_key,
                training_round=debater_config.training_round.display_name,
                model_type=_model_type_to_string(debater_config.settings.model_type),
                model_file_path=debater_config.settings.model_file_path,
            ),
            judge=RunSubtaskModelInfo(
                key=configuration_name.judge_key,
                training_round=judge_config.training_round.display_name,
                model_type=_model_type_to_string(judge_config.settings.model_type),
                model_file_path=judge_config.settings.model_file_path,
            ),
        )
    return RunSubtaskResponse(
        id=int(row["id"]),
        run_task_id=int(row["run_task_id"]),
        base_task_name=base_task_name,
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
        base_task_configuration=base_task_configuration,
    )


def _fetch_run(database: TaskDatabase, run_id: int) -> RunTaskResponse | None:
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


def _fetch_subtasks(database: TaskDatabase, run_id: int | None = None) -> List[RunSubtaskResponse]:
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


def get_run_detail(run_id: int, database: TaskDatabase) -> RunDetailResponse:
    """Return a single run and its subtasks, raising if the run is absent."""
    run = _fetch_run(database, run_id)
    if run is None:
        raise RunNotFoundError(run_id)
    subtasks = _fetch_subtasks(database, run_id=run_id)
    return RunDetailResponse(run=run, subtasks=subtasks)


def list_run_subtasks(run_id: int, database: TaskDatabase) -> List[RunSubtaskResponse]:
    """Return subtasks for a specific run, raising if the run is absent."""
    run = _fetch_run(database, run_id)
    if run is None:
        raise RunNotFoundError(run_id)
    return _fetch_subtasks(database, run_id=run_id)


def list_runs_with_subtasks(database: TaskDatabase) -> List[RunWithSubtasksResponse]:
    """Return all runs and their subtasks ordered by recency."""
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


def list_subtasks(run_id: int | None, database: TaskDatabase) -> List[RunSubtaskResponse]:
    """Return all subtasks, optionally filtered by a specific run identifier."""
    if run_id is not None and _fetch_run(database, run_id) is None:
        raise RunNotFoundError(run_id)
    return _fetch_subtasks(database, run_id=run_id)


__all__ = [
    "RunNotFoundError",
    "get_run_detail",
    "list_run_subtasks",
    "list_runs_with_subtasks",
    "list_subtasks",
]
