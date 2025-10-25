from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel


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


__all__ = [
    "RunTaskResponse",
    "RunSubtaskResponse",
    "RunWithSubtasksResponse",
    "RunDetailResponse",
]
