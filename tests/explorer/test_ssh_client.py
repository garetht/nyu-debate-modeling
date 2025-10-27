from __future__ import annotations

import pytest

from explorer.errors.ssh import ExplorerSSHProcessAmbiguousError, ExplorerSSHProcessNotFoundError
from explorer.explorer_backend.ssh.client import (
    ExplorerSSHProcessLookupResult,
    SSHClientConfig,
    SSHFileClient,
)


def _build_client_with_ps_output(ps_output: str) -> SSHFileClient:
    config = SSHClientConfig(host="example.com", username="tester")
    client = SSHFileClient(config=config)
    client._run_command = lambda _: ps_output  # type: ignore[assignment]
    return client


def test_find_process_id_single_match() -> None:
    ps_output = "\n".join(
        [
            "USER       PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND",
            "tester   12345  0.0  0.1  12345  6789 ?        S    12:00   0:00 python my_process",
        ]
    )
    client = _build_client_with_ps_output(ps_output)

    pid = client.find_process_id("my_process")

    assert pid == 12345


def test_find_process_id_no_match() -> None:
    ps_output = "\n".join(
        [
            "USER       PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND",
            "tester   54321  0.0  0.1  12345  6789 ?        S    12:00   0:00 python other_process",
        ]
    )
    client = _build_client_with_ps_output(ps_output)

    with pytest.raises(ExplorerSSHProcessNotFoundError):
        client.find_process_id("my_process")


def test_find_process_id_multiple_matches() -> None:
    ps_output = "\n".join(
        [
            "USER       PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND",
            "tester   12345  0.0  0.1  12345  6789 ?        S    12:00   0:00 python my_process",
            "tester   67890  0.0  0.1  12345  6789 ?        S    12:01   0:00 python my_process --alt",
        ]
    )
    client = _build_client_with_ps_output(ps_output)

    with pytest.raises(ExplorerSSHProcessAmbiguousError):
        client.find_process_id("my_process")


def test_find_process_ids_mixed_results() -> None:
    ps_output = "\n".join(
        [
            "USER       PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND",
            "tester   12345  0.0  0.1  12345  6789 ?        S    12:00   0:00 python my_process",
            "tester   67890  0.0  0.1  12345  6789 ?        S    12:01   0:00 python other_process",
        ]
    )
    client = _build_client_with_ps_output(ps_output)

    commands = [
        "launcher-script -- python my_process",
        "launcher-script -- missing",
        "launcher-script -- python other_process",
    ]
    results = client.find_process_ids(commands)

    assert results == [
        ExplorerSSHProcessLookupResult(
            original_command="launcher-script -- python my_process",
            success=True,
            remote_command="python my_process",
            pid=12345,
            ps_line="tester   12345  0.0  0.1  12345  6789 ?        S    12:00   0:00 python my_process",
            error=None,
        ),
        ExplorerSSHProcessLookupResult(
            original_command="launcher-script -- missing",
            success=False,
            remote_command="missing",
            pid=None,
            ps_line=None,
            error="No process found matching 'missing'.",
        ),
        ExplorerSSHProcessLookupResult(
            original_command="launcher-script -- python other_process",
            success=True,
            remote_command="python other_process",
            pid=67890,
            ps_line="tester   67890  0.0  0.1  12345  6789 ?        S    12:01   0:00 python other_process",
            error=None,
        ),
    ]


def test_find_process_ids_handles_ambiguous_match() -> None:
    ps_output = "\n".join(
        [
            "USER       PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND",
            "tester   12345  0.0  0.1  12345  6789 ?        S    12:00   0:00 python dup_process",
            "tester   67890  0.0  0.1  12345  6789 ?        S    12:01   0:00 python dup_process --alt",
        ]
    )
    client = _build_client_with_ps_output(ps_output)

    result = client.find_process_ids(["launcher -- python dup_process"])[0]

    assert result.remote_command == "python dup_process"
    assert result.success is False
    assert result.pid is None
    assert "Multiple processes found" in str(result.error)


def test_find_process_ids_empty_input() -> None:
    client = _build_client_with_ps_output(
        "\n".join(["USER       PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND"])
    )

    assert client.find_process_ids([]) == []


def test_find_process_ids_missing_separator() -> None:
    client = _build_client_with_ps_output(
        "\n".join(["USER       PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND"])
    )

    results = client.find_process_ids(["no-remote-command"])

    assert results == [
        ExplorerSSHProcessLookupResult(
            original_command="no-remote-command",
            success=False,
            remote_command=None,
            pid=None,
            ps_line=None,
            error="Command does not contain a remote execution segment after ' -- '.",
        )
    ]


def test_find_process_ids_empty_remote_segment() -> None:
    client = _build_client_with_ps_output(
        "\n".join(["USER       PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND"])
    )

    result = client.find_process_ids(["launcher --    "])[0]

    assert result.success is False
    assert result.remote_command == ""
    assert result.error == "Command does not contain a remote execution segment after ' -- '."


def test_find_process_ids_unparsable_ps_row() -> None:
    ps_output = "\n".join(
        [
            "USER       PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND",
            "invalid-line-without-columns",
        ]
    )
    client = _build_client_with_ps_output(ps_output)

    result = client.find_process_ids(["launcher -- invalid-line-without-columns"])[0]

    assert result.success is False
    assert result.ps_line is None
    assert "Unable to parse process information" in str(result.error)


def test_find_process_ids_non_integer_pid_error() -> None:
    ps_output = "\n".join(
        [
            "USER       PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND",
            "tester   not-a-number  0.0  0.1  12345  6789 ?        S    12:00   0:00 python bad_pid",
        ]
    )
    client = _build_client_with_ps_output(ps_output)

    result = client.find_process_ids(["launcher -- python bad_pid"])[0]

    assert result.success is False
    assert result.error == "Failed to parse PID from line: 'tester   not-a-number  0.0  0.1  12345  6789 ?        S    12:00   0:00 python bad_pid'"
