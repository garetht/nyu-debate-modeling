from __future__ import annotations

import os
from pathlib import Path
from typing import Final

from run_orchestrator.recorder.task_database import TaskDatabase

DEFAULT_DATABASE_PATH: Final[Path] = Path("run_orchestrator/recorder/tasks.sqlite3")


def resolve_database_path() -> Path:
    """Return the path to the task database, honoring environment overrides."""
    override: str | None = os.environ.get("TASK_DATABASE_PATH")
    if override is not None and override.strip() != "":
        return Path(override)
    return DEFAULT_DATABASE_PATH


def get_database() -> TaskDatabase:
    """Return a TaskDatabase instance that points at the configured SQLite file."""
    return TaskDatabase(resolve_database_path())


__all__ = ["DEFAULT_DATABASE_PATH", "get_database", "resolve_database_path"]
