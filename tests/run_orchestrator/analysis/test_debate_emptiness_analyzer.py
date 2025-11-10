from __future__ import annotations

from collections import Counter
from pathlib import Path, PosixPath
from typing import Sequence

import pytest

from run_orchestrator.analysis.debate_emptiness_analyzer import (
    analyze_debate_emptiness,
    parse_args,
)
from run_orchestrator.analysis.transcript_model import Metadata, Speech, Transcript


def build_transcript(
    *,
    file_path: str,
    debate_identifier: str,
    speeches: Sequence[tuple[str, str]],
) -> Transcript:
    metadata = Metadata(
        first_debater_correct=True,
        question_idx=1,
        background_text="background",
        question="question",
        first_debater_answer="answer1",
        second_debater_answer="answer2",
        debate_identifier=debate_identifier,
    )
    speech_objects: list[Speech] = [
        Speech(speaker=speaker, content=content) for speaker, content in speeches
    ]
    return Transcript(metadata=metadata, speeches=speech_objects, file_path=PosixPath(file_path))


@pytest.mark.parametrize(
    "transcripts, expected_counter, debater_a_files, debater_b_files, unique_files",
    [
        (
            [
                build_transcript(
                    file_path="/tmp/debate1.json",
                    debate_identifier="debate1",
                    speeches=(
                        ("Debater_A", ""),
                        ("Debater_B", "content"),
                    ),
                ),
                build_transcript(
                    file_path="/tmp/debate2.json",
                    debate_identifier="debate2",
                    speeches=(
                        ("Debater_B", ""),
                        ("Judge", "content"),
                    ),
                ),
            ],
            Counter({"debate1": 1, "debate2": 1}),
            (PosixPath("/tmp/debate1.json"),),
            (PosixPath("/tmp/debate2.json"),),
            (PosixPath("/tmp/debate1.json"), PosixPath("/tmp/debate2.json")),
        ),
        (
            [
                build_transcript(
                    file_path="/tmp/debate3.json",
                    debate_identifier="debate3",
                    speeches=(
                        ("Debater_A", ""),
                        ("Debater_B", ""),
                    ),
                ),
            ],
            Counter({"debate3": 2}),
            (PosixPath("/tmp/debate3.json"),),
            (PosixPath("/tmp/debate3.json"),),
            (PosixPath("/tmp/debate3.json"),),
        ),
    ],
)
def test_analyze_debate_emptiness_collects_expected_results(
    transcripts: Sequence[Transcript],
    expected_counter: Counter[str],
    debater_a_files: tuple[PosixPath, ...],
    debater_b_files: tuple[PosixPath, ...],
    unique_files: tuple[PosixPath, ...],
) -> None:
    analysis = analyze_debate_emptiness(transcripts)

    assert analysis.empty_speech_counts == expected_counter
    assert analysis.debater_a_empty_files == debater_a_files
    assert analysis.debater_b_empty_files == debater_b_files
    assert analysis.unique_empty_files == unique_files
    assert analysis.total_debates == len(transcripts)
    assert analysis.total_unique_empty_debates == len(unique_files)


def test_analyze_debate_emptiness_handles_absence_of_empty_speeches() -> None:
    transcripts = [
        build_transcript(
            file_path="/tmp/debate4.json",
            debate_identifier="debate4",
            speeches=(
                ("Debater_A", "content"),
                ("Debater_B", "more"),
            ),
        )
    ]

    analysis = analyze_debate_emptiness(transcripts)

    assert analysis.empty_speech_counts == Counter()
    assert analysis.debater_a_empty_files == ()
    assert analysis.debater_b_empty_files == ()
    assert analysis.unique_empty_files == ()
    assert analysis.total_debates == 1
    assert analysis.total_unique_empty_debates == 0


def test_parse_args_returns_typed_configuration(tmp_path: Path) -> None:
    args = parse_args([str(tmp_path), "--delete"])

    assert args.folder_path == tmp_path.resolve()
    assert args.delete is True
