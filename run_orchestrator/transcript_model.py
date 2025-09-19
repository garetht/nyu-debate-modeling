import json
from dataclasses import dataclass
from pathlib import Path, PosixPath

from dataclasses import dataclass
from typing import Any, List, Optional, TypeVar, Callable, Type, cast


T = TypeVar("T")


def from_bool(x: Any) -> bool:
    assert isinstance(x, bool)
    return x


def from_int(x: Any) -> int:
    assert isinstance(x, int) and not isinstance(x, bool)
    return x


def from_str(x: Any) -> str:
    assert isinstance(x, str)
    return x


def from_float(x: Any) -> float:
    assert isinstance(x, (float, int)) and not isinstance(x, bool)
    return float(x)


def to_float(x: Any) -> float:
    assert isinstance(x, (int, float))
    return x


def from_none(x: Any) -> Any:
    assert x is None
    return x


def from_list(f: Callable[[Any], T], x: Any) -> List[T]:
    assert isinstance(x, list)
    return [f(y) for y in x]


def from_union(fs, x):
    for f in fs:
        try:
            return f(x)
        except:
            pass
    assert False


def to_class(c: Type[T], x: Any) -> dict:
    assert isinstance(x, c)
    return cast(Any, x).to_dict()


@dataclass
class Metadata:
    first_debater_correct: bool
    question_idx: int
    background_text: str
    question: str
    first_debater_answer: str
    second_debater_answer: str
    debate_identifier: str

    @staticmethod
    def from_dict(obj: Any) -> 'Metadata':
        assert isinstance(obj, dict)
        first_debater_correct = from_bool(obj.get("first_debater_correct"))
        question_idx = from_int(obj.get("question_idx"))
        background_text = from_str(obj.get("background_text"))
        question = from_str(obj.get("question"))
        first_debater_answer = from_str(obj.get("first_debater_answer"))
        second_debater_answer = from_str(obj.get("second_debater_answer"))
        debate_identifier = from_str(obj.get("debate_identifier"))
        return Metadata(first_debater_correct, question_idx, background_text, question, first_debater_answer, second_debater_answer, debate_identifier)

    def to_dict(self) -> dict:
        result: dict = {}
        result["first_debater_correct"] = from_bool(self.first_debater_correct)
        result["question_idx"] = from_int(self.question_idx)
        result["background_text"] = from_str(self.background_text)
        result["question"] = from_str(self.question)
        result["first_debater_answer"] = from_str(self.first_debater_answer)
        result["second_debater_answer"] = from_str(self.second_debater_answer)
        result["debate_identifier"] = from_str(self.debate_identifier)
        return result


@dataclass
class ProbabilisticDecision:
    debater_a: float
    debater_b: float

    @staticmethod
    def from_dict(obj: Any) -> 'ProbabilisticDecision':
        assert isinstance(obj, dict)
        debater_a = from_float(obj.get("Debater_A"))
        debater_b = from_float(obj.get("Debater_B"))
        return ProbabilisticDecision(debater_a, debater_b)

    def to_dict(self) -> dict:
        result: dict = {}
        result["Debater_A"] = to_float(self.debater_a)
        result["Debater_B"] = to_float(self.debater_b)
        return result


@dataclass
class Supplemental:
    speech: str
    decision: str
    preference: None
    rejected_responses: List[Any]
    bon_opposing_model_responses: List[Any]
    bon_probabilistic_preferences: List[Any]
    internal_representations: str
    response_tokens: List[int]
    prompt_tokens: List[int]
    prompt: str
    failed: bool
    probabilistic_decision: Optional[ProbabilisticDecision] = None

    @staticmethod
    def from_dict(obj: Any) -> 'Supplemental':
        assert isinstance(obj, dict)
        speech = from_str(obj.get("speech"))
        decision = from_str(obj.get("decision"))
        preference = from_none(obj.get("preference"))
        rejected_responses = from_list(lambda x: x, obj.get("rejected_responses"))
        bon_opposing_model_responses = from_list(lambda x: x, obj.get("bon_opposing_model_responses"))
        bon_probabilistic_preferences = from_list(lambda x: x, obj.get("bon_probabilistic_preferences"))
        internal_representations = from_str(obj.get("internal_representations"))
        response_tokens = from_list(from_int, obj.get("response_tokens"))
        prompt_tokens = from_list(from_int, obj.get("prompt_tokens"))
        prompt = from_str(obj.get("prompt"))
        failed = from_bool(obj.get("failed"))
        probabilistic_decision = from_union([from_none, ProbabilisticDecision.from_dict], obj.get("probabilistic_decision"))
        return Supplemental(speech, decision, preference, rejected_responses, bon_opposing_model_responses, bon_probabilistic_preferences, internal_representations, response_tokens, prompt_tokens, prompt, failed, probabilistic_decision)

    def to_dict(self) -> dict:
        result: dict = {}
        result["speech"] = from_str(self.speech)
        result["decision"] = from_str(self.decision)
        result["preference"] = from_none(self.preference)
        result["rejected_responses"] = from_list(lambda x: x, self.rejected_responses)
        result["bon_opposing_model_responses"] = from_list(lambda x: x, self.bon_opposing_model_responses)
        result["bon_probabilistic_preferences"] = from_list(lambda x: x, self.bon_probabilistic_preferences)
        result["internal_representations"] = from_str(self.internal_representations)
        result["response_tokens"] = from_list(from_int, self.response_tokens)
        result["prompt_tokens"] = from_list(from_int, self.prompt_tokens)
        result["prompt"] = from_str(self.prompt)
        result["failed"] = from_bool(self.failed)
        result["probabilistic_decision"] = from_union([from_none, lambda x: to_class(ProbabilisticDecision, x)], self.probabilistic_decision)
        return result


@dataclass
class Speech:
    speaker: str
    content: str
    supplemental: Optional[Supplemental] = None

    @staticmethod
    def from_dict(obj: Any) -> 'Speech':
        assert isinstance(obj, dict)
        speaker = from_str(obj.get("speaker"))
        content = from_str(obj.get("content"))
        supplemental = from_union([from_none, Supplemental.from_dict], obj.get("supplemental"))
        return Speech(speaker, content, supplemental)

    def to_dict(self) -> dict:
        result: dict = {}
        result["speaker"] = from_str(self.speaker)
        result["content"] = from_str(self.content)
        result["supplemental"] = from_union([from_none, lambda x: to_class(Supplemental, x)], self.supplemental)
        return result


@dataclass
class Transcript:
    metadata: Metadata
    speeches: List[Speech]
    file_path: PosixPath

    @staticmethod
    def from_dict(obj: Any, file_path: PosixPath) -> 'Transcript':
        assert isinstance(obj, dict)
        metadata = Metadata.from_dict(obj.get("metadata"))
        speeches = from_list(Speech.from_dict, obj.get("speeches"))
        return Transcript(metadata, speeches, file_path)

    def to_dict(self) -> dict:
        result: dict = {}
        result["metadata"] = to_class(Metadata, self.metadata)
        result["speeches"] = from_list(lambda x: to_class(Speech, x), self.speeches)
        result["file_path"] = self.file_path
        return result


def transcript_from_dict(s: Any) -> Transcript:
    return Transcript.from_dict(s)


def transcript_to_dict(x: Transcript) -> Any:
    return to_class(Transcript, x)



def read_transcripts_from_folder(folder_path: Path) -> list[Transcript]:
    """Recursively reads all JSON files in a directory and returns a list of Transcript objects."""
    if not folder_path.is_dir():
        print(f"Error: '{folder_path}' is not a directory.")
        return []

    transcripts = []
    for file_path in folder_path.rglob('*.json'):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                transcripts.append(Transcript.from_dict(data, file_path))
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {file_path}. File will be skipped.")
        except (KeyError, AssertionError, TypeError) as e:
            print(f"Warning: Data structure validation failed for {file_path}. File will be skipped. Error: {type(e).__name__}: {e}")
        except Exception as e:
            print(f"Warning: An unexpected error occurred while reading {file_path}. File will be skipped. Error: {type(e).__name__}: {e}")
    return transcripts
