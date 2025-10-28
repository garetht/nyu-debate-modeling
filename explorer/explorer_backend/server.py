from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Final, List

from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from run_orchestrator.recorder.task_database import TaskDatabase

from explorer.errors import (
    InvalidGroupModeError,
    OutputNotFoundError,
    OutputStatsUnavailableError,
    OutputsDirectoryMissingError,
    RunNotFoundError,
    ExplorerSSHStreamingError,
)

from explorer.explorer_backend.models import (
    OutputDetailResponse,
    OutputStatsResponse,
    OutputsListResponse,
    RunDetailResponse,
    RunProcessResponse,
    RunSubtaskResponse,
    RunWithSubtasksResponse,
)
from explorer.explorer_backend.services import (
    get_database,
    get_run_detail,
    get_output_detail as fetch_output_detail,
    get_output_stats as fetch_output_stats,
    list_outputs as fetch_outputs,
    list_run_subtasks as fetch_run_subtasks,
    list_run_processes as fetch_run_processes,
    list_runs_with_subtasks,
    list_subtasks as fetch_subtasks,
)
from explorer.explorer_backend.ssh import (
    SSHClientConfig,
    ExplorerSSHStreamingError,
    SSHFileClient,
    WebSocketSender,
)

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

DEFAULT_SSH_USERNAME: Final[str] = "ubuntu"
DEFAULT_SSH_IDENTITY: Final[Path] = Path("~/.ssh/lambda-labs.pem")
DEFAULT_LAST_LINES: Final[int] = 200
DEFAULT_KEEPALIVE_SECONDS: Final[int] = 30


@dataclass(frozen=True)
class SubtaskLogLocation:
    ip_address: str
    log_path: str


def _create_ssh_client(host: str) -> SSHFileClient:
    """Construct a new SSH file client for the provided host."""
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


@app.get("/runs/{run_id}/processes", response_model=List[RunProcessResponse])
def list_run_processes(
    run_id: int, database: TaskDatabase = Depends(get_database)
) -> List[RunProcessResponse]:
    """Return remote process metadata for each subtask associated with a run."""
    try:
        return fetch_run_processes(run_id=run_id, database=database, ssh_client_factory=_create_ssh_client)
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
    database: TaskDatabase = Depends(get_database),
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
        log_location = _fetch_subtask_log_location(subtask_id, database=database)
    except ValueError as error:
        await websocket.send_text(str(error))
        await websocket.close(code=1003)
        return

    sender = FastAPIWebSocketSender(websocket)
    ssh_client = _create_ssh_client(host=log_location.ip_address)
    try:
        await ssh_client.stream_last_lines(sender, log_location.log_path, last_lines)
    except WebSocketDisconnect:
        return
    except ExplorerSSHStreamingError as error:
        await websocket.send_text(f"Streaming error: {error}")
        await websocket.close(code=1011)
    except Exception as e:
        await websocket.close(code=1011, reason=e.__class__.__name__)
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


def _fetch_subtask_log_location(subtask_id: str, database: TaskDatabase) -> SubtaskLogLocation:
    """Retrieve connection metadata for the given subtask identifier."""
    try:
        subtask_key = int(subtask_id)
    except ValueError as error:
        raise ValueError("subtask_id must be an integer.") from error

    with database._connect() as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            """
            SELECT ip_address, log_path
            FROM run_subtasks
            WHERE id = ?;
            """,
            (subtask_key,),
        ).fetchone()

    if row is None:
        raise ValueError(f"Subtask {subtask_key} was not found.")

    ip_address = str(row["ip_address"])
    log_path = str(row["log_path"])
    if not ip_address:
        raise ValueError("Subtask is missing an ip_address.")
    if not log_path:
        raise ValueError("Subtask is missing a log_path.")
    return SubtaskLogLocation(ip_address=ip_address, log_path=log_path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=SERVER_PORT)
