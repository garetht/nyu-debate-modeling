from dataclasses import dataclass
from typing import List, Any, TypeVar, Callable, Type, cast


T = TypeVar("T")


def from_str(x: Any) -> str:
    assert isinstance(x, str)
    return x


def from_list(f: Callable[[Any], T], x: Any) -> List[T]:
    assert isinstance(x, list)
    return [f(y) for y in x]


def to_class(c: Type[T], x: Any) -> dict:
    assert isinstance(x, c)
    return cast(Any, x).to_dict()


@dataclass
class LojbanTranscript:
    original_id: str
    prompt: str
    original_key: str
    original_explanation: str
    answers: List[str]
    prompt_file_content: str

    @staticmethod
    def from_dict(obj: Any) -> 'LojbanTranscript':
        assert isinstance(obj, dict)
        original_id = from_str(obj.get("original_id"))
        prompt = from_str(obj.get("prompt"))
        original_key = from_str(obj.get("original_key"))
        original_explanation = from_str(obj.get("original_explanation"))
        answers = from_list(from_str, obj.get("answers"))
        prompt_file_content = from_str(obj.get("prompt_file_content"))
        return LojbanTranscript(original_id, prompt, original_key, original_explanation, answers, prompt_file_content)

    def to_dict(self) -> dict:
        result: dict = {}
        result["original_id"] = from_str(self.original_id)
        result["prompt"] = from_str(self.prompt)
        result["original_key"] = from_str(self.original_key)
        result["original_explanation"] = from_str(self.original_explanation)
        result["answers"] = from_list(from_str, self.answers)
        result["prompt_file_content"] = from_str(self.prompt_file_content)
        return result


def lojban_transcript_from_dict(s: Any) -> LojbanTranscript:
    return LojbanTranscript.from_dict(s)


def lojban_transcript_to_dict(x: LojbanTranscript) -> Any:
    return to_class(LojbanTranscript, x)
