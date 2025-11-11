from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Set, cast

import pyarrow.parquet as pq
import pytest

from run_orchestrator.analysis.analysis_models.debate_distribution import DebateDistributionAnalysis
from run_orchestrator.analysis.analysis_models.analysis_result import AnalysisResult
from run_orchestrator.analysis.analysis_models.debate_emptiness import DebateEmptinessAnalysis
from run_orchestrator.analysis.analysis_models.debate_lengths import DebateLengthAnalysis
from run_orchestrator.analysis.analysis_models.debate_uniqueness import DebateUniquenessAnalysis
from run_orchestrator.analysis.analysis_models.evaluation_configuration import EvaluationConfiguration
from run_orchestrator.analysis.analysis_models.full_debate_analysis import FullDebateAnalysis
from run_orchestrator.analysis.serializers.dataclass_parquet import dataclasses_to_table, write_dataclasses_to_parquet
from run_orchestrator.analysis.analysis_models.debate_stats import DebateStats


class Color(Enum):
    RED = "red"
    BLUE = "blue"
    GREEN = "green"


@dataclass
class NestedMetadata:
    location: Path
    properties: Dict[str, int]


@dataclass
class ComplexRecord:
    identifier: str
    nested: NestedMetadata
    colors: Sequence[Color]
    tags: Set[str]
    scores: Sequence[int]


def test_dataclasses_to_table_sanitizes_nested_structures() -> None:
    record: ComplexRecord = ComplexRecord(
        identifier="sample",
        nested=NestedMetadata(location=Path("/data/config.json"), properties={"attempts": 3}),
        colors=[Color.RED, Color.GREEN],
        tags={"alpha", "beta"},
        scores=(1, 2, 3),
    )

    table = dataclasses_to_table(record)
    rows: List[Dict[str, Any]] = cast(List[Dict[str, Any]], table.to_pylist())

    assert rows[0]["identifier"] == "sample"
    nested_row = rows[0]["nested"]
    assert isinstance(nested_row, dict)
    assert nested_row["location"] == "/data/config.json"
    assert nested_row["properties"] == {"attempts": 3}
    assert rows[0]["colors"] == ["red", "green"]
    assert set(rows[0]["tags"]) == {"alpha", "beta"}
    assert rows[0]["scores"] == [1, 2, 3]


def test_dataclasses_to_table_rejects_empty_iterable() -> None:
    with pytest.raises(ValueError):
        dataclasses_to_table([])


AnalysisReconstructor = Callable[[Dict[str, Any]], AnalysisResult]


def _rehydrate_counts(raw: Any) -> Dict[str, int]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return {cast(str, key): int(cast(int, value)) for key, value in raw.items()}
    entries = cast(List[Dict[str, Any]], raw)
    reconstructed: Dict[str, int] = {}
    for entry in entries:
        key: str = cast(str, entry["key"])
        value_key: str = "count" if "count" in entry else "value"
        count: int = cast(int, entry[value_key])
        reconstructed[key] = count
    return reconstructed


def _rehydrate_debate_distribution(row: Dict[str, Any]) -> DebateDistributionAnalysis:
    identifier_counts = _rehydrate_counts(row["identifier_counts"])
    title_counts = _rehydrate_counts(row["title_counts"])
    return DebateDistributionAnalysis(
        identifier_counts=identifier_counts,
        title_counts=title_counts,
        transcript_count=cast(int, row["transcript_count"]),
    )


def _rehydrate_debate_lengths(row: Dict[str, Any]) -> DebateLengthAnalysis:
    return DebateLengthAnalysis(
        debater_a_lengths=list(cast(List[int], row["debater_a_lengths"])),
        debater_b_lengths=list(cast(List[int], row["debater_b_lengths"])),
        transcript_count=cast(int, row["transcript_count"]),
    )


def _rehydrate_debate_emptiness(row: Dict[str, Any]) -> DebateEmptinessAnalysis:
    debater_a_files = list(cast(List[str], row["debater_a_empty_files"]))
    debater_b_files = list(cast(List[str], row["debater_b_empty_files"]))
    unique_files = list(cast(List[str], row["unique_empty_files"]))
    return DebateEmptinessAnalysis(
        empty_speech_counts=_rehydrate_counts(row["empty_speech_counts"]),
        debater_a_empty_files=debater_a_files,
        debater_b_empty_files=debater_b_files,
        unique_empty_files=unique_files,
        total_debates=cast(int, row["total_debates"]),
    )


def _rehydrate_debate_uniqueness(row: Dict[str, Any]) -> DebateUniquenessAnalysis:
    duplicate_paths = [Path(path) for path in cast(List[str], row["duplicate_file_paths"])]
    return DebateUniquenessAnalysis(
        unique_identifiers=list(cast(List[str], row["unique_identifiers"])),
        duplicate_file_paths=duplicate_paths,
        total_transcripts=cast(int, row["total_transcripts"]),
    )


def _rehydrate_debate_stats(row: Dict[str, Any]) -> DebateStats:
    return DebateStats(
        total_debates=cast(int, row["total_debates"]),
        debater_a_wins=cast(int, row["debater_a_wins"]),
        debater_b_wins=cast(int, row["debater_b_wins"]),
        judge_correct=cast(int, row["judge_correct"]),
        first_debater_correct=cast(int, row["first_debater_correct"]),
        debater_a_probs=list(cast(List[float], row["debater_a_probs"])),
        debater_b_probs=list(cast(List[float], row["debater_b_probs"])),
    )


def _rehydrate_evaluation_configuration(row: Dict[str, Any]) -> EvaluationConfiguration:
    return EvaluationConfiguration(**row)


def _rehydrate_full_debate_analysis(row: Dict[str, Any]) -> FullDebateAnalysis:
    return FullDebateAnalysis(
        emptiness=_rehydrate_debate_emptiness(cast(Dict[str, Any], row["emptiness"])),
        lengths=_rehydrate_debate_lengths(cast(Dict[str, Any], row["lengths"])),
        distribution=_rehydrate_debate_distribution(cast(Dict[str, Any], row["distribution"])),
        stats=_rehydrate_debate_stats(cast(Dict[str, Any], row["stats"])),
        configuration=_rehydrate_evaluation_configuration(cast(Dict[str, Any], row["configuration"])),
    )


def _sample_debate_distribution() -> DebateDistributionAnalysis:
    return DebateDistributionAnalysis(
        identifier_counts={
            "debate-1": 3,
            "debate-2": 5,
        },
        title_counts={
            "Title 1": 3,
            "Title 2": 5,
        },
        transcript_count=8,
    )


def _sample_debate_lengths() -> DebateLengthAnalysis:
    return DebateLengthAnalysis(
        debater_a_lengths=[120, 130, 110],
        debater_b_lengths=[115, 125, 118],
        transcript_count=3,
    )


def _sample_debate_emptiness() -> DebateEmptinessAnalysis:
    return DebateEmptinessAnalysis(
        empty_speech_counts={"Debater_A": 2, "Debater_B": 1},
        debater_a_empty_files=["/tmp/debate_a1.json", "/tmp/debate_a2.json"],
        debater_b_empty_files=["/tmp/debate_b1.json"],
        unique_empty_files=["/tmp/debate_a1.json", "/tmp/debate_b1.json"],
        total_debates=4,
    )


def _sample_debate_uniqueness() -> DebateUniquenessAnalysis:
    return DebateUniquenessAnalysis(
        unique_identifiers=["debate-unique-1", "debate-unique-2"],
        duplicate_file_paths=[Path("/tmp/dup1.json"), Path("/tmp/dup2.json")],
        total_transcripts=6,
    )


def _sample_debate_stats() -> DebateStats:
    return DebateStats(
        total_debates=10,
        debater_a_wins=6,
        debater_b_wins=4,
        judge_correct=7,
        first_debater_correct=8,
        debater_a_probs=[0.6, 0.7, 0.65],
        debater_b_probs=[0.4, 0.3, 0.35],
    )


def _sample_evaluation_configuration() -> EvaluationConfiguration:
    return EvaluationConfiguration(
        config_type="eval",
        task_type="task",
        debater_name="debater",
        debater_training_round="round",
        debater_is_reasoning=True,
        debater_model_type="model-a",
        debater_max_new_tokens=1024,
        judge_name="judge",
        judge_training_round="round",
        judge_is_reasoning=False,
        judge_model_type="model-b",
        judge_max_new_tokens=768,
    )


def _sample_full_debate_analysis() -> FullDebateAnalysis:
    return FullDebateAnalysis(
        emptiness=_sample_debate_emptiness(),
        lengths=_sample_debate_lengths(),
        distribution=_sample_debate_distribution(),
        stats=_sample_debate_stats(),
        configuration=_sample_evaluation_configuration(),
    )


@pytest.mark.parametrize(
    ("analysis_record", "reconstructor"),
    [
        (_sample_debate_distribution(), _rehydrate_debate_distribution),
        (_sample_debate_lengths(), _rehydrate_debate_lengths),
        (_sample_debate_emptiness(), _rehydrate_debate_emptiness),
        (_sample_debate_uniqueness(), _rehydrate_debate_uniqueness),
        (_sample_full_debate_analysis(), _rehydrate_full_debate_analysis),
    ],
)
def test_analysis_results_round_trip(analysis_record: AnalysisResult, reconstructor: AnalysisReconstructor, tmp_path: Path) -> None:
    output_path: Path = tmp_path / f"{analysis_record.__class__.__name__}.parquet"

    write_dataclasses_to_parquet(analysis_record, output_path)
    assert output_path.exists()

    table = pq.read_table(output_path)
    rows = cast(List[Dict[str, Any]], table.to_pylist())
    restored = reconstructor(rows[0])

    assert restored == analysis_record


def test_write_dataclasses_to_parquet_round_trip(tmp_path: Path) -> None:
    output_path: Path = tmp_path / "complex.parquet"
    records: List[ComplexRecord] = [
        ComplexRecord(
            identifier="row-1",
            nested=NestedMetadata(location=Path("/a/b.json"), properties={"count": 1}),
            colors=[Color.RED, Color.BLUE],
            tags={"x", "y"},
            scores=(10, 20),
        ),
        ComplexRecord(
            identifier="row-2",
            nested=NestedMetadata(location=Path("/c/d.json"), properties={"count": 2}),
            colors=[Color.GREEN],
            tags={"z"},
            scores=(30,),
        ),
    ]

    write_dataclasses_to_parquet(records, output_path)
    assert output_path.exists()

    table = pq.read_table(output_path)
    read_records: List[ComplexRecord] = []
    for row in cast(List[Dict[str, Any]], table.to_pylist()):
        nested_row = cast(Dict[str, Any], row["nested"])
        nested = NestedMetadata(
            location=Path(nested_row["location"]),
            properties=cast(Dict[str, int], nested_row["properties"]),
        )
        color_values: List[str] = cast(List[str], row["colors"])
        tag_values: List[str] = cast(List[str], row["tags"])
        score_values: List[int] = cast(List[int], row["scores"])
        restored: ComplexRecord = ComplexRecord(
            identifier=row["identifier"],
            nested=nested,
            colors=[Color(color) for color in color_values],
            tags=set(tag_values),
            scores=tuple(score_values),
        )
        read_records.append(restored)

    assert read_records == records
