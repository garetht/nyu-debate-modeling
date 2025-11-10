from __future__ import annotations

from copy import deepcopy
from pathlib import Path, PosixPath

import json
import pytest
from pydantic import ValidationError

from run_orchestrator.analysis.transcript_model import (
    ProbabilisticDecision,
    Transcript,
    iter_transcripts_from_folder,
)


def _sample_transcript_dict() -> dict[str, object]:
    return {
        "metadata": {
            "first_debater_correct": True,
            "question_idx": 3,
            "background_text": "Background",
            "question": "What is the capital of France?",
            "first_debater_answer": "Paris",
            "second_debater_answer": "Lyon",
            "debate_identifier": "debate-001",
        },
        "speeches": [
            {
                "speaker": "debater_a",
                "content": "Opening argument",
                "supplemental": {
                    "speech": "Detailed speech content",
                    "decision": "undecided",
                    "preference": None,
                    "rejected_responses": [],
                    "bon_opposing_model_responses": [],
                    "bon_probabilistic_preferences": [],
                    "internal_representations": "chain of thought",
                    "response_tokens": [1, 2],
                    "prompt_tokens": [3, 4],
                    "prompt": "Prompt text",
                    "failed": False,
                    "probabilistic_decision": {"Debater_A": 0.7, "Debater_B": 0.3},
                },
            },
            {"speaker": "debater_b", "content": "Rebuttal"},
        ],
    }


def test_transcript_from_dict_sets_file_path_and_serializes() -> None:
    original = _sample_transcript_dict()
    expected = deepcopy(original)
    file_path = PosixPath("/tmp/debate/transcript.json")

    transcript = Transcript.from_dict(original, file_path=file_path)

    assert transcript.file_path == file_path
    assert transcript.metadata.first_debater_correct is True
    assert transcript.speeches[0].supplemental is not None

    serialized = transcript.to_dict()
    expected["file_path"] = str(file_path)
    expected["speeches"][1]["supplemental"] = None
    assert serialized == expected


def test_probabilistic_decision_rejects_boolean_values() -> None:
    with pytest.raises(ValidationError) as exc_info:
        ProbabilisticDecision(**{"Debater_A": True, "Debater_B": 0.5})

    assert "bool" in str(exc_info.value)


def test_iter_transcripts_from_folder_filters_invalid_files(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    valid_data = _sample_transcript_dict()
    valid_path = tmp_path / "valid.json"
    invalid_json_path = tmp_path / "bad.json"
    invalid_structure_path = tmp_path / "invalid_structure.json"

    valid_path.write_text(json.dumps(valid_data), encoding="utf-8")
    invalid_json_path.write_text("{not-json}", encoding="utf-8")
    invalid_structure_path.write_text(json.dumps({"metadata": {}}), encoding="utf-8")

    transcripts = list(iter_transcripts_from_folder(tmp_path))

    captured = capsys.readouterr().out
    assert len(transcripts) == 1
    transcript = transcripts[0]
    assert transcript.file_path == PosixPath(valid_path)
    assert transcript.metadata.debate_identifier == "debate-001"
    assert transcript.speeches[0].supplemental is not None
    assert f"Could not decode JSON from {invalid_json_path}" in captured
    assert "Data structure validation failed" in captured


def test_iter_transcripts_from_folder_non_directory(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    file_path = tmp_path / "not_a_dir.json"
    file_path.write_text("{}", encoding="utf-8")

    result = iter_transcripts_from_folder(file_path)
    captured = capsys.readouterr().out

    assert result is None
    assert f"'{file_path}' is not a directory" in captured
