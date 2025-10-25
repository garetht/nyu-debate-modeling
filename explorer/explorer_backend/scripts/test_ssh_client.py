from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional

from explorer.explorer_backend.ssh import (
    SSHClientConfig,
    SSHFileClient,
    SSHStreamingError,
    WebSocketSender,
)


class ConsoleWebSocket(WebSocketSender):
    """Minimal WebSocketSender implementation that prints payloads to stdout."""

    def __init__(self) -> None:
        self._closed: bool = False

    async def send_text(self, data: str) -> None:
        print(data, flush=True)

    async def close(self, code: int | None = None) -> None:
        if not self._closed:
            if code is not None:
                print(f"[websocket closed: code={code}]", file=sys.stderr)
            self._closed = True


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Exercise the explorer SSH client helpers. "
            "Defaults emulate `ssh ubuntu@192.222.57.237 -i ~/.ssh/lambda-labs.pem` "
            "and tail the remote log file."
        )
    )
    parser.add_argument(
        "--host",
        default="192.222.57.237",
        help="SSH host (default: %(default)s).",
    )
    parser.add_argument(
        "--username",
        default="ubuntu",
        help="SSH username (default: %(default)s).",
    )
    parser.add_argument(
        "--identity-file",
        type=Path,
        default=Path("~/.ssh/lambda-labs.pem"),
        help="Path to the SSH private key (default: %(default)s).",
    )
    parser.add_argument(
        "--remote-path",
        type=Path,
        default=Path("/home/ubuntu/mars-arnesen-gh/garethtan/logs/llama-trained-for-gpt-41-round-one-20251025-105038.log"),
        help="Remote file to inspect.",
    )
    parser.add_argument(
        "--mode",
        choices=("first", "last", "stream"),
        default="stream",
        help="Operation to perform on the remote file (default: %(default)s).",
    )
    parser.add_argument(
        "--lines",
        type=int,
        default=20,
        help="Number of lines to fetch for first/last modes (default: %(default)s).",
    )
    parser.add_argument(
        "--keepalive",
        type=int,
        default=None,
        help="Optional SSH keepalive interval in seconds.",
    )
    parser.add_argument(
        "--passphrase",
        default=None,
        help="Optional passphrase for the private key.",
    )
    return parser


def _run_non_stream_mode(client: SSHFileClient, mode: str, remote_path: str, line_count: int) -> None:
    if mode == "first":
        lines = client.get_first_lines(remote_path, line_count)
    else:
        lines = client.get_last_lines(remote_path, line_count)
    for line in lines:
        print(line)


async def _run_stream_mode(client: SSHFileClient, remote_path: str, line_count: int) -> None:
    websocket = ConsoleWebSocket()
    try:
        await client.stream_last_lines(websocket, remote_path, line_count)
    except SSHStreamingError as exc:
        print(f"Streaming error: {exc}", file=sys.stderr)
        await websocket.close(code=1011)


def main(argv: Optional[list[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    config = SSHClientConfig(
        host=args.host,
        username=args.username,
        port=22,
        connect_kwargs=None,
        private_key_path=str(args.identity_file),
        private_key_passphrase=args.passphrase,
    )
    client = SSHFileClient(config, keepalive_interval=args.keepalive)
    remote_path = str(args.remote_path)

    if args.mode == "stream":
        try:
            asyncio.run(_run_stream_mode(client, remote_path, args.lines))
        except KeyboardInterrupt:
            print("Cancelled by user.", file=sys.stderr)
    else:
        _run_non_stream_mode(client, args.mode, remote_path, args.lines)


if __name__ == "__main__":
    main()
