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

    results = client.find_process_ids(["my_process", "missing", "other_process"])

    assert results == [
        ExplorerSSHProcessLookupResult(search_term="my_process", success=True, pid=12345, error=None),
        ExplorerSSHProcessLookupResult(
            search_term="missing",
            success=False,
            pid=None,
            error="No process found matching 'missing'.",
        ),
        ExplorerSSHProcessLookupResult(search_term="other_process", success=True, pid=67890, error=None),
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

    result = client.find_process_ids(["dup_process"])[0]

    assert result.search_term == "dup_process"
    assert result.success is False
    assert result.pid is None
    assert "Multiple processes found" in str(result.error)


def test_find_process_ids_empty_input() -> None:
    client = _build_client_with_ps_output(
        "\n".join(["USER       PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND"])
    )

    assert client.find_process_ids([]) == []
