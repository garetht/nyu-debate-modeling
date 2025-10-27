from __future__ import annotations

import asyncio
import shlex
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Protocol, Sequence, runtime_checkable

from explorer.errors.ssh import (
    ExplorerSSHProcessAmbiguousError,
    ExplorerSSHProcessError,
    ExplorerSSHProcessNotFoundError,
    ExplorerSSHStreamingError,
)
from fabric import Connection
import paramiko


@dataclass(frozen=True)
class SSHClientConfig:
    """Configuration container for creating Fabric SSH connections."""

    host: str
    username: str
    port: int = 22
    connect_kwargs: Mapping[str, object] | None = None
    private_key_path: str | None = None
    private_key_passphrase: str | None = None

    def build_connection(self) -> Connection:
        """Instantiate a new Fabric Connection from the stored configuration."""
        connect_kwargs_dict: dict[str, object] = (
            {key: value for key, value in self.connect_kwargs.items()} if self.connect_kwargs else {}
        )
        if self.private_key_path:
            connect_kwargs_dict["pkey"] = self._load_private_key(self.private_key_path, self.private_key_passphrase)
        return Connection(
            host=self.host,
            user=self.username,
            port=self.port,
            connect_kwargs=connect_kwargs_dict,
        )

    @staticmethod
    def _load_private_key(private_key_path: str, passphrase: str | None) -> paramiko.PKey:
        expanded_path = Path(private_key_path).expanduser()
        if not expanded_path.is_file():
            raise FileNotFoundError(f"Private key file not found at {expanded_path}")

        candidates: list[type[paramiko.PKey]] = []
        for key_name in ("RSAKey", "ECDSAKey", "Ed25519Key", "DSSKey"):
            key_cls = getattr(paramiko, key_name, None)
            if key_cls is not None:
                candidates.append(key_cls)

        for key_cls in candidates:
            try:
                return key_cls.from_private_key_file(str(expanded_path), password=passphrase)
            except paramiko.PasswordRequiredException:
                raise
            except paramiko.SSHException:
                continue

        raise paramiko.SSHException(f"Unsupported or unreadable private key format: {expanded_path}")


@dataclass(frozen=True)
class ExplorerSSHProcessLookupResult:
    """Structured result detailing the outcome of a bulk process lookup."""

    search_term: str
    success: bool
    pid: int | None = None
    error: str | None = None


@runtime_checkable
class WebSocketSender(Protocol):
    """Protocol describing the websocket interface used for streaming."""

    async def send_text(self, data: str) -> None:  # pragma: no cover - protocol declaration
        ...

    async def close(self, code: int | None = None) -> None:  # pragma: no cover - protocol declaration
        ...


class SSHFileClient:
    """High-level SSH file helper tailored for explorer backend use."""

    def __init__(self, config: SSHClientConfig, *, keepalive_interval: int | None = None) -> None:
        self._config = config
        self._keepalive_interval = keepalive_interval

    def get_last_lines(self, remote_path: str, line_count: int) -> list[str]:
        """
        Fetch the last `line_count` lines from `remote_path`.

        Raises:
            ValueError: If `line_count` is not positive.
        """
        self._validate_line_count(line_count)
        command = f"tail -n {line_count} {shlex.quote(remote_path)}"
        output = self._run_command(command)
        return self._split_lines(output)

    def get_first_lines(self, remote_path: str, line_count: int) -> list[str]:
        """
        Fetch the first `line_count` lines from `remote_path`.

        Raises:
            ValueError: If `line_count` is not positive.
        """
        self._validate_line_count(line_count)
        command = f"head -n {line_count} {shlex.quote(remote_path)}"
        output = self._run_command(command)
        return self._split_lines(output)

    def find_process_id(self, search_term: str) -> int:
        """
        Search for a process whose command line contains `search_term` and return its PID.

        Raises:
            ExplorerSSHProcessNotFoundError: If no matching process is found.
            ExplorerSSHProcessAmbiguousError: If multiple processes match `search_term`.
            ExplorerSSHProcessError: If the process listing output cannot be parsed.
        """
        ps_command: str = "ps aux"
        output: str = self._run_command(ps_command)
        lines: list[str] = self._split_lines(output)
        process_lines: list[str] = lines[1:] if len(lines) > 1 else []
        return self._lookup_pid_from_lines(search_term, process_lines)

    def find_process_ids(self, search_terms: Sequence[str]) -> list[ExplorerSSHProcessLookupResult]:
        """
        Bulk lookup of process identifiers for multiple `search_terms`.

        Returns a list containing the lookup outcome for each search term, including
        success status, PID (if located), and any error message encountered.
        """
        if not search_terms:
            return []

        ps_command: str = "ps aux"
        output: str = self._run_command(ps_command)
        lines: list[str] = self._split_lines(output)
        process_lines: list[str] = lines[1:] if len(lines) > 1 else []

        results: list[ExplorerSSHProcessLookupResult] = []
        for term in search_terms:
            try:
                pid = self._lookup_pid_from_lines(term, process_lines)
            except ExplorerSSHProcessError as error:
                results.append(
                    ExplorerSSHProcessLookupResult(
                        search_term=term,
                        success=False,
                        pid=None,
                        error=str(error),
                    )
                )
            else:
                results.append(
                    ExplorerSSHProcessLookupResult(
                        search_term=term,
                        success=True,
                        pid=pid,
                        error=None,
                    )
                )
        return results

    def _lookup_pid_from_lines(self, search_term: str, process_lines: Sequence[str]) -> int:
        matches: list[str] = [line for line in process_lines if search_term in line]

        if not matches:
            raise ExplorerSSHProcessNotFoundError(f"No process found matching {search_term!r}.")
        if len(matches) > 1:
            raise ExplorerSSHProcessAmbiguousError(
                f"Multiple processes found matching {search_term!r}: {len(matches)} matches."
            )

        match_line: str = matches[0]
        columns: list[str] = match_line.split(maxsplit=10)
        if len(columns) < 2:
            raise ExplorerSSHProcessError(f"Unable to parse process information for match: {match_line!r}")

        pid_str: str = columns[1]
        try:
            return int(pid_str)
        except ValueError as exc:
            raise ExplorerSSHProcessError(f"Failed to parse PID from line: {match_line!r}") from exc

    async def stream_last_lines(
        self,
        websocket: WebSocketSender,
        remote_path: str,
        line_count: int,
        *,
        encoding: str = "utf-8",
    ) -> None:
        """
        Stream the last `line_count` lines of `remote_path` followed by live updates.

        The collected lines are forwarded to `websocket`, which must implement the
        `WebSocketSender` protocol (e.g. FastAPI's WebSocket).
        """
        self._validate_line_count(line_count)
        command = f"tail -n 0 -F {shlex.quote(remote_path)}"
        queue: asyncio.Queue[tuple[Literal["line", "error", "done"], str | BaseException | None]] = asyncio.Queue()
        stop_event = threading.Event()
        loop = asyncio.get_running_loop()
        connection = self._open_connection()
        transport = connection.client.get_transport()
        if transport is None:
            connection.close()
            raise ExplorerSSHStreamingError("Failed to acquire SSH transport.")

        session = transport.open_session()
        session.exec_command(command)

        def reader() -> None:
            """
            Continuously read bytes from the SSH session and forward complete lines.

            The thread uses small fixed-size reads to minimise latency. Partial line
            fragments are accumulated and flushed once a newline arrives or the remote
            command exits.
            """
            buffer = ""
            chunk_size = 1024
            idle_sleep = 0.1
            try:
                while not stop_event.is_set():
                    if session.recv_ready():
                        data = session.recv(chunk_size)
                        if not data:
                            break  # remote closed the stream
                        buffer += data.decode(encoding, errors="replace")
                        buffer = _flush_lines_to_queue(buffer, queue, loop)
                        continue

                    if session.exit_status_ready():
                        break  # command finished; flush residue after loop

                    time.sleep(idle_sleep)  # nothing ready; yield to avoid busy spin

                if buffer:
                    loop.call_soon_threadsafe(queue.put_nowait, ("line", buffer.rstrip("\r\n")))
            except BaseException as exc:  # pragma: no cover - defensive, difficult to simulate
                loop.call_soon_threadsafe(queue.put_nowait, ("error", exc))
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()

        try:
            initial_lines = self.get_last_lines(remote_path, line_count)
            for line in initial_lines:
                await websocket.send_text(line)

            while True:
                item_type, payload = await queue.get()
                if item_type == "line":
                    await websocket.send_text(str(payload))
                elif item_type == "error":
                    raise ExplorerSSHStreamingError("Error while streaming file over SSH.") from payload
                elif item_type == "done":
                    break
        except asyncio.CancelledError:
            stop_event.set()
            raise
        finally:
            stop_event.set()
            session.close()
            connection.close()
            thread.join(timeout=2.0)

    def _run_command(self, command: str) -> str:
        connection = self._open_connection()
        try:
            result = connection.run(command, hide=True)
            return result.stdout
        finally:
            connection.close()

    def _open_connection(self) -> Connection:
        connection = self._config.build_connection()
        connection.open()
        if self._keepalive_interval:
            transport = connection.client.get_transport()
            if transport is not None:
                transport.set_keepalive(self._keepalive_interval)
        return connection

    @staticmethod
    def _split_lines(output: str) -> list[str]:
        return [line.rstrip("\r") for line in output.splitlines()]

    @staticmethod
    def _validate_line_count(line_count: int) -> None:
        if line_count <= 0:
            raise ValueError("line_count must be positive.")


def _flush_lines_to_queue(
    buffer: str,
    queue: asyncio.Queue[tuple[Literal["line", "error", "done"], str | BaseException | None]],
    loop: asyncio.AbstractEventLoop,
) -> str:
    """
    Push complete newline-delimited records from `buffer` onto the queue.

    Returns the residual buffer containing any trailing partial line segment.
    """
    while "\n" in buffer:
        raw_line, buffer = buffer.split("\n", 1)
        cleaned_line = raw_line.rstrip("\r\n")
        loop.call_soon_threadsafe(queue.put_nowait, ("line", cleaned_line))
    return buffer
