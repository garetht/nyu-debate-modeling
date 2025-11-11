from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import pyarrow.parquet as pq
import pytest
from pydantic import BaseModel, ConfigDict, Field

from run_orchestrator.analysis.serializers import (
    dataclasses_to_table,
    pydantic_models_to_table,
    pydantic_to_parquet,
    write_dataclasses_to_parquet,
)
from run_orchestrator.analysis.analysis_models.debate_distribution import DebateDistributionAnalysis
from run_orchestrator.analysis.analysis_models.debate_emptiness import DebateEmptinessAnalysis
from run_orchestrator.analysis.analysis_models.debate_lengths import DebateLengthAnalysis
from run_orchestrator.analysis.analysis_models.debate_stats import DebateStats
from run_orchestrator.analysis.analysis_models.evaluation_configuration import EvaluationConfiguration
from run_orchestrator.analysis.analysis_models.full_debate_analysis import FullDebateAnalysis


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
    assert table.column("path").to_pylist() == [str(sample_path), str(sample_path)]
    assert table.column("tags").to_pylist() == [["a"], ["b", "c"]]


def test_dataclasses_to_table_rejects_non_dataclass() -> None:
    with pytest.raises(TypeError):
        dataclasses_to_table([{"identifier": "oops"}])  # type: ignore[arg-type]


class SampleModel(BaseModel):
    identifier: str
    count: int
    path: str
    tags: List[str]


class AliasModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    value: int = Field(serialization_alias="value_alias")


def test_pydantic_models_to_table_converts_records() -> None:
    sample_path: str = "/data/pydantic.json"
    models: List[SampleModel] = [
        SampleModel(identifier="alpha", count=1, path=sample_path, tags=["x", "y"]),
        SampleModel(identifier="beta", count=2, path=sample_path, tags=["z"]),
    ]

    table = pydantic_models_to_table(models)

    assert table.num_rows == 2
    assert table.column("identifier").to_pylist() == ["alpha", "beta"]
    assert table.column("count").to_pylist() == [1, 2]
    assert table.column("path").to_pylist() == [sample_path, sample_path]
    assert table.column("tags").to_pylist() == [["x", "y"], ["z"]]


def test_pydantic_to_parquet_round_trip(tmp_path: Path) -> None:
    output_path: Path = tmp_path / "models.parquet"
    sample_path: str = "/tmp/pydantic.json"
    models: List[SampleModel] = [
        SampleModel(identifier="gamma", count=3, path=sample_path, tags=["a"]),
        SampleModel(identifier="delta", count=4, path=sample_path, tags=["b", "c"]),
    ]

    pydantic_to_parquet(models, output_path)

    assert output_path.exists()
    table = pq.read_table(output_path)
    assert table.num_rows == 2
    assert table.column("identifier").to_pylist() == ["gamma", "delta"]
    assert table.column("tags").to_pylist() == [["a"], ["b", "c"]]


def test_pydantic_models_to_table_by_alias() -> None:
    table = pydantic_models_to_table([AliasModel(value=5)], by_alias=True)

    assert table.column_names == ["value_alias"]
    assert table.column("value_alias").to_pylist() == [5]


def test_pydantic_models_to_table_rejects_mixed_types() -> None:
    with pytest.raises(TypeError):
        pydantic_models_to_table(
            [
                SampleModel(identifier="a", count=1, path="/tmp/a", tags=[]),
                AliasModel(value=1),
            ]
        )  # type: ignore[list-item]


def test_pydantic_to_parquet_preserves_nested_models(tmp_path: Path) -> None:
    output_path: Path = tmp_path / "analysis.parquet"
    emptiness = DebateEmptinessAnalysis(
        empty_speech_counts={"debater": 2},
        debater_a_empty_files=["a.json"],
        debater_b_empty_files=["b.json"],
        unique_empty_files=["c.json"],
        total_debates=3,
    )
    lengths = DebateLengthAnalysis(
        debater_a_lengths=[10, 12],
        debater_b_lengths=[9, 11],
        transcript_count=2,
    )
    distribution = DebateDistributionAnalysis(
        identifier_counts={"Alpha_topic": 2},
        title_counts={"Alpha": 2},
        transcript_count=2,
    )
    stats = DebateStats()
    stats.add_debate(True, False, True, False, 0.6, 0.4)

    analysis = FullDebateAnalysis(
        emptiness=emptiness,
        lengths=lengths,
        distribution=distribution,
        stats=stats,
        configuration=_sample_evaluation_configuration(),
    )

    pydantic_to_parquet([analysis], output_path)

    table_data = pq.read_table(output_path).to_pydict()
    written_distribution = table_data["distribution"][0]

    assert written_distribution["identifier_counts"] == [("Alpha_topic", 2)]
    assert written_distribution["title_counts"] == [("Alpha", 2)]
    assert written_distribution["transcript_count"] == 2
def _sample_evaluation_configuration() -> EvaluationConfiguration:
    return EvaluationConfiguration(
        config_type="eval",
        task_type="task",
        debater_name="debater",
        debater_training_round="round",
        debater_is_reasoning=True,
        debater_model_type="model-a",
        debater_max_new_tokens=1500,
        judge_name="judge",
        judge_training_round="round",
        judge_is_reasoning=False,
        judge_model_type="model-b",
        judge_max_new_tokens=900,
    )

