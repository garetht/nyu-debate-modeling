from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


class TaskDatabase:
    """SQLite-backed recorder for orchestrated runs and individual subtasks."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    def _initialize_schema(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA foreign_keys = ON;")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS run_tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_name TEXT NOT NULL,
                    yaml_path TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    is_hidden INTEGER NOT NULL
                );
                """
            )

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS run_subtasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_task_id INTEGER NOT NULL,
                    base_task_name TEXT NOT NULL,
                    resolved_task_name TEXT NOT NULL,
                    ip_address TEXT NOT NULL,
                    command TEXT NOT NULL,
                    log_path TEXT NOT NULL,
                    configuration_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (run_task_id) REFERENCES run_tasks (id) ON DELETE CASCADE
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def register_run(self, run_name: str, yaml_path: Path, *, is_hidden: bool = False) -> int:
        timestamp = self._timestamp()
        hidden_flag = 1 if is_hidden else 0
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO run_tasks (run_name, yaml_path, created_at, is_hidden)
                VALUES (?, ?, ?, ?);
                """,
                (run_name, str(yaml_path), timestamp, hidden_flag),
            )
            return int(cursor.lastrowid)

    def set_run_hidden(self, run_task_id: int, *, is_hidden: bool) -> None:
        hidden_flag = 1 if is_hidden else 0
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE run_tasks
                SET is_hidden = ?
                WHERE id = ?;
                """,
                (hidden_flag, run_task_id),
            )

    def record_subtask(
        self,
        run_task_id: int,
        base_task_name: str,
        resolved_task_name: str,
        ip_address: str,
        command: Sequence[str],
        log_path: str,
        configuration: Mapping[str, Any],
    ) -> None:
        timestamp = self._timestamp()
        command_text = " ".join(command)
        configuration_json = json.dumps(configuration, sort_keys=True)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO run_subtasks (
                    run_task_id,
                    base_task_name,
                    resolved_task_name,
                    ip_address,
                    command,
                    log_path,
                    configuration_json,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    run_task_id,
                    base_task_name,
                    resolved_task_name,
                    ip_address,
                    command_text,
                    log_path,
                    configuration_json,
                    timestamp,
                ),
            )

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()
