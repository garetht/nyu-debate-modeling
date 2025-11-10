from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import pyarrow.parquet as pq
import pytest

from run_orchestrator.analysis.serializers import dataclasses_to_table, write_dataclasses_to_parquet


@dataclass
class SampleRecord:
    identifier: str
    count: int
    path: Path
    tags: List[str]


def test_dataclasses_to_table_converts_records() -> None:
    sample_path: Path = Path("/data/sample.json")
    records: List[SampleRecord] = [
        SampleRecord(identifier="alpha", count=1, path=sample_path, tags=["x", "y"]),
        SampleRecord(identifier="beta", count=2, path=sample_path, tags=["z"]),
    ]

    table = dataclasses_to_table(records)

    assert table.num_rows == 2
    assert table.column("identifier").to_pylist() == ["alpha", "beta"]
    assert table.column("count").to_pylist() == [1, 2]
    assert table.column("path").to_pylist() == [str(sample_path), str(sample_path)]
    assert table.column("tags").to_pylist() == [["x", "y"], ["z"]]


def test_write_dataclasses_to_parquet_round_trip(tmp_path: Path) -> None:
    output_path: Path = tmp_path / "records.parquet"
    sample_path: Path = Path("/tmp/resource.json")
    records: List[SampleRecord] = [
        SampleRecord(identifier="gamma", count=3, path=sample_path, tags=["a"]),
        SampleRecord(identifier="delta", count=4, path=sample_path, tags=["b", "c"]),
    ]

    write_dataclasses_to_parquet(records, output_path)

    assert output_path.exists()
    table = pq.read_table(output_path)
    assert table.num_rows == 2
    assert table.column("identifier").to_pylist() == ["gamma", "delta"]
    assert table.column("tags").to_pylist() == [["a"], ["b", "c"]]


def test_dataclasses_to_table_rejects_non_dataclass() -> None:
    with pytest.raises(TypeError):
        dataclasses_to_table([{"identifier": "oops"}])  # type: ignore[arg-type]
