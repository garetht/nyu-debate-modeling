from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class RunTaskResponse(BaseModel):
    id: int
    run_name: str
    yaml_path: str
    created_at: str


class RunSubtaskModelInfo(BaseModel):
    key: str
    training_round: str
    model_type: str
    model_file_path: Optional[str]


class RunSubtaskConfigurationName(BaseModel):
    config_type: str
    task_type_name: str
    debater: RunSubtaskModelInfo
    judge: RunSubtaskModelInfo


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
    base_task_configuration: Optional[RunSubtaskConfigurationName] = None


class RunProcessResponse(BaseModel):
    subtask_id: int
    ip_address: str
    command: str
    remote_command: Optional[str]
    pid: Optional[int]
    ps_line: Optional[str]
    success: bool
    error: Optional[str]


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
    "RunSubtaskConfigurationName",
    "RunSubtaskModelInfo",
    "RunProcessResponse",
]
