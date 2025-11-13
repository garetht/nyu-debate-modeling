from __future__ import annotations

from pathlib import PosixPath
from typing import Iterable, Iterator, Sequence, TypeAlias

import pytest

from run_orchestrator.analysis.debate_distribution_analyzer import analyze_debate_distribution
from run_orchestrator.analysis.debate_emptiness_analyzer import analyze_debate_emptiness
from run_orchestrator.analysis.debate_lengths_analyzer import analyze_debate_lengths
from run_orchestrator.analysis.debate_stats_analyzer import analyze_debate_statistics
from run_orchestrator.analysis.debate_full_analyzer import (
    full_debate_analysis,
)
from run_orchestrator.analysis.analysis_models.evaluation_configuration import EvaluationConfiguration
from run_orchestrator.analysis.analysis_models.full_debate_analysis import FullDebateAnalysis
from run_orchestrator.analysis.transcript_model import Metadata, ProbabilisticDecision, Speech, Supplemental, Transcript


SpeechSpec: TypeAlias = tuple[str, str, int, str, tuple[float, float] | None]


def build_supplemental(token_count: int, decision: str, probabilities: tuple[float, float] | None) -> Supplemental:
    probabilistic_decision = (
        ProbabilisticDecision(debater_a=probabilities[0], debater_b=probabilities[1])
        if probabilities is not None
        else None
    )
    return Supplemental(
        speech="utterance",
        decision=decision,
        preference=None,
        rejected_responses=[],
        bon_opposing_model_responses=[],
        bon_probabilistic_preferences=[],
        internal_representations="",
        response_tokens=list(range(token_count)),
        prompt_tokens=[42],
        prompt="prompt",
        failed=False,
        probabilistic_decision=probabilistic_decision,
    )


def build_transcript(
    *,
    file_path: str,
    debate_identifier: str,
    first_debater_correct: bool,
    speeches: Sequence[SpeechSpec],
) -> Transcript:
    metadata = Metadata(
        first_debater_correct=first_debater_correct,
        question_idx=1,
        background_text="background",
        question="question",
        first_debater_answer="answer1",
        second_debater_answer="answer2",
        debate_identifier=debate_identifier,
    )
    speech_objects: list[Speech] = [
        Speech(
            speaker=speaker,
            content=content,
            supplemental=build_supplemental(token_count, decision, probabilities),
        )
        for speaker, content, token_count, decision, probabilities in speeches
    ]
    return Transcript(metadata=metadata, speeches=speech_objects, file_path=PosixPath(file_path))


def make_example_transcripts() -> list[Transcript]:
    return [
        build_transcript(
            file_path="/tmp/debate1.json",
            debate_identifier="debate1",
            first_debater_correct=True,
            speeches=(
                ("Debater_A", "", 0, "Debater_A", (0.8, 0.2)),
                ("Debater_B", "argument", 3, "Debater_A", (0.8, 0.2)),
            ),
        ),
        build_transcript(
            file_path="/tmp/debate2.json",
            debate_identifier="debate2",
            first_debater_correct=False,
            speeches=(
                ("Debater_A", "claim", 5, "Debater_B", (0.35, 0.65)),
                ("Debater_B", "", 0, "Debater_B", (0.35, 0.65)),
            ),
        ),
    ]


def _build_stub_configuration() -> EvaluationConfiguration:
    return EvaluationConfiguration(
        raw_name="stub-configuration",
        config_type="eval",
        task_type="task",
        debater_name="debater",
        debater_training_round="round",
        debater_is_reasoning=True,
        debater_model_type="model-a",
        debater_max_new_tokens=1000,
        judge_name="judge",
        judge_training_round="round",
        judge_is_reasoning=False,
        judge_model_type="model-b",
        judge_max_new_tokens=512,
    )


class _StubConfigurationName:
    """Stub object mimicking ConfigurationName for testing purposes."""


def test_full_debate_analysis_matches_component_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transcripts = make_example_transcripts()

    stub_configuration = _StubConfigurationName()

    expected_analysis = FullDebateAnalysis(
        emptiness=analyze_debate_emptiness(transcripts),
        lengths=analyze_debate_lengths(transcripts),
        distribution=analyze_debate_distribution(transcripts),
        stats=analyze_debate_statistics(transcripts),
        configuration=_build_stub_configuration(),
    )

    monkeypatch.setattr(
        "run_orchestrator.analysis.debate_full_analyzer.EvaluationConfiguration.from_configuration_name",
        lambda _: _build_stub_configuration(),
    )

    analysis = full_debate_analysis(
        configuration=stub_configuration,
        transcripts=iter(transcripts),
    )

    assert analysis == expected_analysis


def test_full_debate_analysis_only_consumes_input_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transcripts = make_example_transcripts()
    expected_analysis = FullDebateAnalysis(
        emptiness=analyze_debate_emptiness(transcripts),
        lengths=analyze_debate_lengths(transcripts),
        distribution=analyze_debate_distribution(transcripts),
        stats=analyze_debate_statistics(transcripts),
        configuration=_build_stub_configuration(),
    )

    consumption_log: list[str] = []

    def generator() -> Iterable[Transcript]:
        def iterator() -> Iterator[Transcript]:
            for transcript in transcripts:
                consumption_log.append(transcript.metadata.debate_identifier)
                yield transcript

        return iterator()

    stub_configuration = _StubConfigurationName()

    monkeypatch.setattr(
        "run_orchestrator.analysis.debate_full_analyzer.EvaluationConfiguration.from_configuration_name",
        lambda _: _build_stub_configuration(),
    )

    analysis = full_debate_analysis(
        configuration=stub_configuration,
        transcripts=generator(),
    )

    assert analysis == expected_analysis
    assert consumption_log == [
        transcript.metadata.debate_identifier for transcript in transcripts
    ]
