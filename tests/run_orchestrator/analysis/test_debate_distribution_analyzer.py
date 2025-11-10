from pathlib import Path

from run_orchestrator.analysis.debate_distribution_analyzer import analyze_debate_distribution
from run_orchestrator.analysis.transcript_model import Metadata, Speech, Transcript


def _build_transcript(identifier: str) -> Transcript:
    metadata = Metadata(
        first_debater_correct=True,
        question_idx=1,
        background_text="background",
        question="question",
        first_debater_answer="answer_a",
        second_debater_answer="answer_b",
        debate_identifier=identifier,
    )
    speech = Speech(speaker="Debater_A", content="content")
    return Transcript(metadata=metadata, speeches=[speech], file_path=Path(f"{identifier}.json"))


def test_analyze_debate_distribution_handles_identifiers_without_underscore() -> None:
    transcripts = [
        _build_transcript("Alpha_topic_one"),
        _build_transcript("Alpha_topic_two"),
        _build_transcript("NoUnderscore"),
        _build_transcript("multi_part_topic"),
    ]

    analysis = analyze_debate_distribution(transcripts)

    assert analysis.identifier_counts == {
        "Alpha_topic_one": 1,
        "Alpha_topic_two": 1,
        "NoUnderscore": 1,
        "multi_part_topic": 1,
    }

    assert analysis.title_counts == {
        "Alpha": 2,
        "NoUnderscore": 1,
        "multi": 1,
    }
