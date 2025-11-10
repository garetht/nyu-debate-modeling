from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence

import pytest


def _ensure_torch_stub() -> None:
    if "torch" in sys.modules:
        return
    torch_stub: types.ModuleType = types.ModuleType("torch")
    torch_stub.Tensor = type("Tensor", (), {})  # type: ignore[attr-defined]
    sys.modules["torch"] = torch_stub


_ensure_torch_stub()

from run_orchestrator.analysis.all_debates_analyzer import (
    PARQUET_FILENAME,
    analyze_all_debates,
    analyze_configuration_directory,
    iter_valid_configuration_directories,
)
from run_orchestrator.analysis.analysis_models.debate_stats import DebateStats
from run_orchestrator.analysis.analysis_models.debate_distribution import (
    DebateDistributionAnalysis,
)
from run_orchestrator.analysis.analysis_models.debate_emptiness import DebateEmptinessAnalysis
from run_orchestrator.analysis.analysis_models.debate_lengths import DebateLengthAnalysis
from run_orchestrator.analysis.analysis_models.full_debate_analysis import FullDebateAnalysis
from run_orchestrator.analysis.transcript_model import Metadata, Speech, Transcript
from run_orchestrator.evals_generator.config_spec import ConfigurationType


def _build_stub_analysis() -> FullDebateAnalysis:
    return FullDebateAnalysis(
        emptiness=DebateEmptinessAnalysis(
            empty_speech_counts={},
            debater_a_empty_files=[],
            debater_b_empty_files=[],
            unique_empty_files=[],
            total_debates=0,
        ),
        lengths=DebateLengthAnalysis(
            debater_a_lengths=[],
            debater_b_lengths=[],
            transcript_count=0,
        ),
        distribution=DebateDistributionAnalysis(
            identifier_counts={},
            title_counts={},
            transcript_count=0,
        ),
        stats=DebateStats(),
    )


@dataclass(frozen=True)
class _FakeConfigurationName:
    config_type: ConfigurationType


def test_iter_valid_configuration_directories_filters_invalid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    valid_directory: Path = tmp_path / "valid_config"
    valid_directory.mkdir()
    non_eval_directory: Path = tmp_path / "non_eval_config"
    non_eval_directory.mkdir()
    invalid_directory: Path = tmp_path / "invalid_config"
    invalid_directory.mkdir()
    extraneous_file: Path = tmp_path / "artifact.txt"
    extraneous_file.write_text("data", encoding="utf-8")

    called_names: List[str] = []

    def fake_deserialize(name: str) -> _FakeConfigurationName:
        called_names.append(name)
        if name == "valid_config":
            return _FakeConfigurationName(ConfigurationType.EVAL)
        if name == "non_eval_config":
            return _FakeConfigurationName(ConfigurationType.DATA_GENERATION)
        raise ValueError("invalid configuration name")

    monkeypatch.setattr(
        "run_orchestrator.analysis.all_debates_analyzer.ConfigurationName.deserialize",
        fake_deserialize,
    )

    directories: List[Path] = list(iter_valid_configuration_directories(tmp_path))

    assert directories == [valid_directory]
    assert "valid_config" in called_names
    assert "non_eval_config" in called_names
    assert "invalid_config" in called_names


def test_analyze_configuration_directory_invokes_full_analysis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configuration_directory: Path = tmp_path / "config_one"
    transcripts_directory: Path = configuration_directory / "outputs" / "transcripts"
    transcripts_directory.mkdir(parents=True)

    metadata: Metadata = Metadata(
        first_debater_correct=True,
        question_idx=1,
        background_text="context",
        question="question",
        first_debater_answer="answer_a",
        second_debater_answer="answer_b",
        debate_identifier="debate-1",
    )
    speech: Speech = Speech(speaker="Judge", content="content")
    transcript: Transcript = Transcript(
        metadata=metadata,
        speeches=[speech],
        file_path=transcripts_directory / "debate-1.json",
    )

    transcripts: List[Transcript] = [transcript]
    consumed_transcripts: List[Transcript] = []

    def fake_iter_transcripts(folder: Path) -> Iterator[Transcript]:
        assert folder == transcripts_directory
        return iter(transcripts)

    def fake_full_debate_analysis(transcripts_iterable: Iterable[Transcript]) -> FullDebateAnalysis:
        consumed_transcripts.extend(list(transcripts_iterable))
        return _build_stub_analysis()

    monkeypatch.setattr(
        "run_orchestrator.analysis.all_debates_analyzer.iter_transcripts_from_folder",
        fake_iter_transcripts,
    )
    monkeypatch.setattr(
        "run_orchestrator.analysis.all_debates_analyzer.full_debate_analysis",
        fake_full_debate_analysis,
    )

    analysis: FullDebateAnalysis = analyze_configuration_directory(configuration_directory)

    assert consumed_transcripts == transcripts
    assert isinstance(analysis, FullDebateAnalysis)


def test_analyze_configuration_directory_missing_transcripts(tmp_path: Path) -> None:
    configuration_directory: Path = tmp_path / "config_missing"
    configuration_directory.mkdir()

    with pytest.raises(FileNotFoundError):
        analyze_configuration_directory(configuration_directory)


def test_analyze_all_debates_processes_valid_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    valid_directory: Path = tmp_path / "config_a"
    valid_directory.mkdir()
    missing_directory: Path = tmp_path / "config_b"
    missing_directory.mkdir()
    invalid_directory: Path = tmp_path / "invalid"
    invalid_directory.mkdir()

    stub_analysis: FullDebateAnalysis = _build_stub_analysis()
    analyzed_directories: List[Path] = []
    write_calls: Dict[Path, FullDebateAnalysis] = {}
    received_descriptions: List[str] = []

    def fake_deserialize(name: str) -> _FakeConfigurationName:
        if name in {"config_a", "config_b"}:
            return _FakeConfigurationName(ConfigurationType.EVAL)
        raise ValueError("invalid")

    class FakeTqdm:
        def __init__(self, sequence: Sequence[Path], *, desc: str, unit: str):
            self._sequence: List[Path] = list(sequence)
            self.last_description: str = desc
            self.unit: str = unit
            self.closed: bool = False

        def __iter__(self) -> Iterator[Path]:
            return iter(self._sequence)

        def set_description(self, description: str) -> None:
            self.last_description = description
            received_descriptions.append(description)

        def close(self) -> None:
            self.closed = True

    def fake_analyze_directory(directory: Path) -> FullDebateAnalysis:
        if directory == valid_directory:
            analyzed_directories.append(directory)
            return stub_analysis
        raise FileNotFoundError("missing transcripts")

    def fake_write_parquet(result: FullDebateAnalysis, destination: Path, *, compression: str | None = "snappy") -> None:
        write_calls[destination] = result

    monkeypatch.setattr(
        "run_orchestrator.analysis.all_debates_analyzer.ConfigurationName.deserialize",
        fake_deserialize,
    )
    monkeypatch.setattr(
        "run_orchestrator.analysis.all_debates_analyzer.tqdm",
        lambda sequence, **kwargs: FakeTqdm(sequence, **kwargs),
    )
    monkeypatch.setattr(
        "run_orchestrator.analysis.all_debates_analyzer.analyze_configuration_directory",
        fake_analyze_directory,
    )
    monkeypatch.setattr(
        "run_orchestrator.analysis.all_debates_analyzer.pydantic_to_parquet",
        fake_write_parquet,
    )

    analyses: Dict[Path, FullDebateAnalysis] = analyze_all_debates(tmp_path)

    assert analyzed_directories == [valid_directory]
    expected_destination: Path = valid_directory / PARQUET_FILENAME
    assert write_calls == {expected_destination: stub_analysis}
    assert analyses == {valid_directory: stub_analysis}
    assert any(description.startswith("Analyzing config") for description in received_descriptions)


def test_analyze_all_debates_returns_empty_when_no_valid_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "candidate_a").mkdir()
    (tmp_path / "candidate_b").mkdir()

    write_calls: List[Path] = []

    def fake_deserialize(name: str) -> _FakeConfigurationName:
        raise ValueError(f"{name} is invalid")

    def fake_write_parquet(
        result: FullDebateAnalysis,
        destination: Path,
        *,
        compression: str | None = "snappy",
    ) -> None:
        write_calls.append(destination)

    monkeypatch.setattr(
        "run_orchestrator.analysis.all_debates_analyzer.ConfigurationName.deserialize",
        fake_deserialize,
    )
    monkeypatch.setattr(
        "run_orchestrator.analysis.all_debates_analyzer.pydantic_to_parquet",
        fake_write_parquet,
    )

    analyses: Dict[Path, FullDebateAnalysis] = analyze_all_debates(tmp_path)

    assert analyses == {}
    assert write_calls == []
