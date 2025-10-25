from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Final, List

from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from run_orchestrator.recorder.task_database import TaskDatabase

from explorer.explorer_backend.models import (
    OutputDetailResponse,
    OutputStatsResponse,
    OutputsListResponse,
    RunDetailResponse,
    RunSubtaskResponse,
    RunWithSubtasksResponse,
)
from explorer.explorer_backend.services import (
    InvalidGroupModeError,
    OutputNotFoundError,
    OutputStatsUnavailableError,
    OutputsDirectoryMissingError,
    RunNotFoundError,
    get_database,
    get_run_detail,
    get_output_detail as fetch_output_detail,
    get_output_stats as fetch_output_stats,
    list_outputs as fetch_outputs,
    list_run_subtasks as fetch_run_subtasks,
    list_runs_with_subtasks,
    list_subtasks as fetch_subtasks,
)
from explorer.explorer_backend.ssh import SSHClientConfig, SSHFileClient, SSHStreamingError, WebSocketSender

SERVER_PORT: Final[int] = 8067

FRONTEND_ORIGINS: List[str] = [
    "http://127.0.0.1:5173",
    "http://localhost:5173",
    "http://0.0.0.0:5173",
]

app = FastAPI(title="Run Orchestrator Task Database API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEFAULT_SSH_HOST: Final[str] = "192.222.57.237"
DEFAULT_SSH_USERNAME: Final[str] = "ubuntu"
DEFAULT_SSH_IDENTITY: Final[Path] = Path("~/.ssh/lambda-labs.pem")
DEFAULT_REMOTE_LOG_DIR: Final[Path] = Path("/home/ubuntu/mars-arnesen-gh/garethtan/logs")
DEFAULT_LAST_LINES: Final[int] = 200
DEFAULT_KEEPALIVE_SECONDS: Final[int] = 30


def get_ssh_client() -> SSHFileClient:
    """Construct a new SSH file client using environment overrides when available."""
    host: str = os.environ.get("EXPLORER_SSH_HOST", DEFAULT_SSH_HOST)
    username: str = os.environ.get("EXPLORER_SSH_USERNAME", DEFAULT_SSH_USERNAME)
    identity_str: str = os.environ.get("EXPLORER_SSH_IDENTITY", str(DEFAULT_SSH_IDENTITY))
    identity_path: str = str(Path(identity_str).expanduser())
    connect_kwargs: Dict[str, object] = {"allow_agent": False}
    config = SSHClientConfig(
        host=host,
        username=username,
        connect_kwargs=connect_kwargs,
        private_key_path=identity_path,
    )
    return SSHFileClient(config=config, keepalive_interval=DEFAULT_KEEPALIVE_SECONDS)


def get_remote_log_dir() -> Path:
    """Return the base directory that stores remote log files."""
    remote_dir = Path(os.environ.get("EXPLORER_REMOTE_LOG_DIR", str(DEFAULT_REMOTE_LOG_DIR)))
    return remote_dir


class FastAPIWebSocketSender(WebSocketSender):
    """Adapter that allows SSH streaming to write directly to a FastAPI WebSocket."""

    def __init__(self, websocket: WebSocket) -> None:
        self._websocket = websocket

    async def send_text(self, data: str) -> None:
        await self._websocket.send_text(data)

    async def close(self, code: int | None = None) -> None:
        await self._websocket.close(code=code)


@app.get("/health", response_model=Dict[str, str])
def health_check() -> Dict[str, str]:
    """Simple health endpoint to confirm the API is responsive."""
    return {"status": "ok"}


@app.get("/api/outputs", response_model=OutputsListResponse)
def list_outputs(group_mode: str | None = None) -> OutputsListResponse:
    """List available output configurations with optional grouping."""
    try:
        return fetch_outputs(group_mode_key=group_mode)
    except OutputsDirectoryMissingError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except InvalidGroupModeError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/outputs/{configuration}", response_model=OutputDetailResponse)
def get_output_configuration(
    configuration: str,
    page: int = 1,
    page_size: int = 100,
) -> OutputDetailResponse:
    """Return detailed information and transcripts for a configuration."""
    try:
        return fetch_output_detail(configuration=configuration, page=page, page_size=page_size)
    except OutputsDirectoryMissingError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except OutputNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/outputs/{configuration}/stats", response_model=OutputStatsResponse)
def get_output_configuration_stats(configuration: str) -> OutputStatsResponse:
    """Return debate statistics for an evaluation configuration."""
    try:
        return fetch_output_stats(configuration=configuration)
    except OutputsDirectoryMissingError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except OutputNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except OutputStatsUnavailableError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/runs", response_model=List[RunWithSubtasksResponse])
def list_runs(database: TaskDatabase = Depends(get_database)) -> List[RunWithSubtasksResponse]:
    """Return all recorded runs along with their subtasks ordered by recency."""
    return list_runs_with_subtasks(database=database)


@app.get("/runs/{run_id}", response_model=RunDetailResponse)
def get_run(run_id: int, database: TaskDatabase = Depends(get_database)) -> RunDetailResponse:
    """Return a single run and its associated subtasks."""
    try:
        return get_run_detail(run_id=run_id, database=database)
    except RunNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.get("/runs/{run_id}/subtasks", response_model=List[RunSubtaskResponse])
def list_run_subtasks(
    run_id: int, database: TaskDatabase = Depends(get_database)
) -> List[RunSubtaskResponse]:
    """Return subtasks for a specific run."""
    try:
        return fetch_run_subtasks(run_id=run_id, database=database)
    except RunNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.get("/subtasks", response_model=List[RunSubtaskResponse])
def list_subtasks(
    run_id: int | None = None, database: TaskDatabase = Depends(get_database)
) -> List[RunSubtaskResponse]:
    """Return all subtasks, optionally filtered by run identifier."""
    try:
        return fetch_subtasks(run_id=run_id, database=database)
    except RunNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.websocket("/subtasks/{subtask_id}/logs")
async def stream_subtask_logs(
    websocket: WebSocket,
    subtask_id: str,
    ssh_client: SSHFileClient = Depends(get_ssh_client),
) -> None:
    """Stream the most recent log lines for a subtask over a WebSocket connection."""
    await websocket.accept()
    try:
        last_lines = _parse_last_lines(websocket.query_params.get("last_lines"))
    except ValueError as error:
        await websocket.send_text(str(error))
        await websocket.close(code=1003)
        return

    try:
        remote_path = _build_remote_log_path(subtask_id)
    except ValueError as error:
        await websocket.send_text(str(error))
        await websocket.close(code=1003)
        return

    sender = FastAPIWebSocketSender(websocket)
    try:
        await ssh_client.stream_last_lines(sender, remote_path, last_lines)
    except WebSocketDisconnect:
        return
    except SSHStreamingError as error:
        await websocket.send_text(f"Streaming error: {error}")
        await websocket.close(code=1011)
    except Exception:
        await websocket.close(code=1011)
        raise


def _parse_last_lines(raw_value: str | None) -> int:
    """Parse and validate the `last_lines` query parameter."""
    if raw_value is None:
        return DEFAULT_LAST_LINES
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ValueError("last_lines must be an integer.") from error
    if value <= 0:
        raise ValueError("last_lines must be greater than zero.")
    return value


def _build_remote_log_path(subtask_id: str) -> str:
    """Translate a subtask identifier into its corresponding remote log path."""
    safe_name = Path(subtask_id).name
    if not safe_name:
        raise ValueError("subtask_id must not be empty.")
    log_dir = get_remote_log_dir()
    return str(log_dir / f"{safe_name}.log")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=SERVER_PORT)
