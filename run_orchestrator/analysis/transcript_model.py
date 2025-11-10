import json
from pathlib import Path, PosixPath
from typing import Any, Iterator, List, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    ValidationError,
    field_validator,
)


class Metadata(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    first_debater_correct: StrictBool
    question_idx: StrictInt
    background_text: StrictStr
    question: StrictStr
    first_debater_answer: StrictStr
    second_debater_answer: StrictStr
    debate_identifier: StrictStr

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class ProbabilisticDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    debater_a: float = Field(alias="Debater_A")
    debater_b: float = Field(alias="Debater_B")

    @field_validator("debater_a", "debater_b", mode="before")
    @classmethod
    def _coerce_numeric(cls, value: Any) -> float:
        if isinstance(value, bool):
            raise ValueError("Expected float-compatible value, received bool")
        if not isinstance(value, (int, float)):
            raise ValueError(f"Expected float-compatible value, received {type(value).__name__}")
        return float(value)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", by_alias=True)


class Supplemental(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    speech: StrictStr
    decision: StrictStr
    preference: None
    rejected_responses: List[Any]
    bon_opposing_model_responses: List[Any]
    bon_probabilistic_preferences: List[Any]
    internal_representations: Optional[StrictStr] = None
    response_tokens: List[StrictInt]
    prompt_tokens: List[StrictInt]
    prompt: StrictStr
    failed: StrictBool
    probabilistic_decision: Optional[ProbabilisticDecision] = None

    @field_validator("preference", mode="before")
    @classmethod
    def _validate_preference(cls, value: Any) -> None:
        if value is not None:
            raise TypeError(f"Expected None for preference, received {type(value).__name__}")
        return None

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", by_alias=True)


class Speech(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    speaker: StrictStr
    content: StrictStr
    supplemental: Optional[Supplemental] = None

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", by_alias=True)


class Transcript(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    metadata: Metadata
    speeches: List[Speech]
    file_path: PosixPath = Field(default_factory=lambda: PosixPath("."))

    @classmethod
    def from_dict(
        cls,
        obj: Any,
        file_path: Optional[PosixPath] = None,
        field_path: Optional[str] = None,
    ) -> "Transcript":
        if not isinstance(obj, dict):
            raise TypeError(f"Expected dict for {field_path or 'transcript'}, received {type(obj).__name__}")
        data: dict[str, Any] = dict(obj)
        if file_path is not None:
            data["file_path"] = file_path
        return cls.model_validate(data)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", by_alias=True)


def transcript_from_dict(s: Any) -> Transcript:
    return Transcript.from_dict(s)


def transcript_to_dict(x: Transcript) -> dict[str, Any]:
    return x.to_dict()


def iter_transcripts_from_folder(folder_path: Path) -> Iterator[Transcript] | None:
    """Recursively reads JSON files in a directory and yields Transcript objects one by one."""
    if not folder_path.is_dir():
        print(f"Error: '{folder_path}' is not a directory.")
        return None

    def _iter_transcripts() -> Iterator[Transcript]:
        for file_path in folder_path.rglob("*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    data = json.load(file)
                yield Transcript.from_dict(data, PosixPath(file_path))
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from {file_path}. File will be skipped.")
            except (TypeError, ValidationError) as error:
                print(
                    f"Warning: Data structure validation failed for {file_path}. "
                    f"File will be skipped. Error: {type(error).__name__}: {error}"
                )
            except Exception as error:
                print(
                    f"Warning: An unexpected error occurred while reading {file_path}. "
                    f"File will be skipped. Error: {type(error).__name__}: {error}"
                )

    return _iter_transcripts()


def read_transcripts_from_folder(folder_path: Path) -> list[Transcript]:
    """Recursively reads all JSON files in a directory and returns a list of Transcript objects."""
    iterator: Iterator[Transcript] | None = iter_transcripts_from_folder(folder_path)
    if iterator is None:
        return []
    return list(iterator)
