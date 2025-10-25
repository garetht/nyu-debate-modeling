from __future__ import annotations

import argparse
import asyncio
from typing import Any, Optional

import websockets


async def stream_logs(endpoint: str, subtask_id: str, last_lines: int) -> None:
    url = f"{endpoint.rstrip('/')}/subtasks/{subtask_id}/logs?last_lines={last_lines}"
    async with websockets.connect(url) as websocket:
        await _consume_messages(websocket)


async def _consume_messages(websocket: Any) -> None:
    try:
        async for message in websocket:
            print(message)
    except websockets.ConnectionClosed as error:
        print(f"Connection closed: code={error.code}, reason={error.reason}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Example script demonstrating how to consume the subtask log WebSocket."
    )
    parser.add_argument(
        "--endpoint",
        default="ws://127.0.0.1:8067",
        help="Base WebSocket endpoint for the Explorer backend (default: %(default)s).",
    )
    parser.add_argument(
        "--last-lines",
        type=int,
        default=200,
        help="Number of trailing log lines to request before streaming updates.",
    )
    parser.add_argument(
        "subtask_id",
        help="Identifier of the subtask whose logs should be streamed.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    asyncio.run(stream_logs(args.endpoint, args.subtask_id, args.last_lines))


if __name__ == "__main__":
    main()
