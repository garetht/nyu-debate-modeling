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

    identifier_map = {identifier_count.identifier: identifier_count for identifier_count in analysis.identifier_counts}
    assert identifier_map["Alpha_topic_one"].title == "Alpha"
    assert identifier_map["Alpha_topic_one"].topic == "topic_one"

    assert identifier_map["Alpha_topic_two"].title == "Alpha"
    assert identifier_map["Alpha_topic_two"].topic == "topic_two"

    assert identifier_map["NoUnderscore"].title == "NoUnderscore"
    assert identifier_map["NoUnderscore"].topic == ""

    assert identifier_map["multi_part_topic"].title == "multi"
    assert identifier_map["multi_part_topic"].topic == "part_topic"

    title_map = {title_count.title: title_count.count for title_count in analysis.title_counts}
    assert title_map["Alpha"] == 2
    assert title_map["NoUnderscore"] == 1
    assert title_map["multi"] == 1
