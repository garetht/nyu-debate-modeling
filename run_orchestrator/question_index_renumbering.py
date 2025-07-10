# %%

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, TypeVar, Callable, Type, cast

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


def from_none(x: Any) -> Any:
    assert x is None
    return x


def from_float(x: Any) -> float:
    assert isinstance(x, (float, int)) and not isinstance(x, bool)
    return float(x)


def from_list(f: Callable[[Any], T], x: Any) -> List[T]:
    assert isinstance(x, list)
    return [f(y) for y in x]


def to_float(x: Any) -> float:
    assert isinstance(x, (int, float))
    return x


def to_class(c: Type[T], x: Any) -> dict:
    assert isinstance(x, c)
    return cast(Any, x).to_dict()


@dataclass
class Metadatum:
    first_debater_correct: bool
    question_idx: int
    background_text: str
    question: str
    first_debater_answer: str
    second_debater_answer: str
    debate_identifier: str

    @staticmethod
    def from_dict(obj: Any) -> 'Metadatum':
        assert isinstance(obj, dict)
        first_debater_correct = from_bool(obj.get("first_debater_correct"))
        question_idx = from_int(obj.get("question_idx"))
        background_text = from_str(obj.get("background_text"))
        question = from_str(obj.get("question"))
        first_debater_answer = from_str(obj.get("first_debater_answer"))
        second_debater_answer = from_str(obj.get("second_debater_answer"))
        debate_identifier = from_str(obj.get("debate_identifier"))
        return Metadatum(first_debater_correct, question_idx, background_text, question, first_debater_answer,
                         second_debater_answer, debate_identifier)

    def to_dict(self) -> dict:
        result: dict = {"first_debater_correct": from_bool(self.first_debater_correct),
                        "question_idx": from_int(self.question_idx), "background_text": from_str(self.background_text),
                        "question": from_str(self.question),
                        "first_debater_answer": from_str(self.first_debater_answer),
                        "second_debater_answer": from_str(self.second_debater_answer),
                        "debate_identifier": from_str(self.debate_identifier)}
        return result


@dataclass
class Supplemental:
    speech: str
    decision: str
    probabilistic_decision: None
    preference: float
    rejected_responses: List['Supplemental']
    bon_opposing_model_responses: List[Any]
    bon_probabilistic_preferences: List[Any]
    internal_representations: str
    response_tokens: List[Any]
    prompt_tokens: List[Any]
    prompt: str
    failed: bool

    @staticmethod
    def from_dict(obj: Any) -> 'Supplemental':
        assert isinstance(obj, dict)
        speech = from_str(obj.get("speech"))
        decision = from_str(obj.get("decision"))
        probabilistic_decision = from_none(obj.get("probabilistic_decision"))
        preference = from_float(obj.get("preference"))
        rejected_responses = from_list(Supplemental.from_dict, obj.get("rejected_responses"))
        bon_opposing_model_responses = from_list(lambda x: x, obj.get("bon_opposing_model_responses"))
        bon_probabilistic_preferences = from_list(lambda x: x, obj.get("bon_probabilistic_preferences"))
        internal_representations = from_str(obj.get("internal_representations"))
        response_tokens = from_list(lambda x: x, obj.get("response_tokens"))
        prompt_tokens = from_list(lambda x: x, obj.get("prompt_tokens"))
        prompt = from_str(obj.get("prompt"))
        failed = from_bool(obj.get("failed"))
        return Supplemental(speech, decision, probabilistic_decision, preference, rejected_responses,
                            bon_opposing_model_responses, bon_probabilistic_preferences, internal_representations,
                            response_tokens, prompt_tokens, prompt, failed)

    def to_dict(self) -> dict:
        result: dict = {}
        result["speech"] = from_str(self.speech)
        result["decision"] = from_str(self.decision)
        result["probabilistic_decision"] = from_none(self.probabilistic_decision)
        result["preference"] = to_float(self.preference)
        result["rejected_responses"] = from_list(lambda x: to_class(Supplemental, x), self.rejected_responses)
        result["bon_opposing_model_responses"] = from_list(lambda x: x, self.bon_opposing_model_responses)
        result["bon_probabilistic_preferences"] = from_list(lambda x: x, self.bon_probabilistic_preferences)
        result["internal_representations"] = from_str(self.internal_representations)
        result["response_tokens"] = from_list(lambda x: x, self.response_tokens)
        result["prompt_tokens"] = from_list(lambda x: x, self.prompt_tokens)
        result["prompt"] = from_str(self.prompt)
        result["failed"] = from_bool(self.failed)
        return result


@dataclass
class Speech:
    speaker: str
    content: str
    supplemental: Supplemental

    @staticmethod
    def from_dict(obj: Any) -> 'Speech':
        assert isinstance(obj, dict)
        speaker = from_str(obj.get("speaker"))
        content = from_str(obj.get("content"))
        supplemental = Supplemental.from_dict(obj.get("supplemental"))
        return Speech(speaker, content, supplemental)

    def to_dict(self) -> dict:
        result: dict = {}
        result["speaker"] = from_str(self.speaker)
        result["content"] = from_str(self.content)
        result["supplemental"] = to_class(Supplemental, self.supplemental)
        return result


@dataclass
class Transcript:
    metadata: List[Metadatum]
    speeches: List[Speech]

    @staticmethod
    def from_dict(obj: Any) -> 'Transcript':
        assert isinstance(obj, dict)
        metadata = from_list(Metadatum.from_dict, obj.get("metadata"))
        speeches = from_list(Speech.from_dict, obj.get("speeches"))
        return Transcript(metadata, speeches)

    def to_dict(self) -> dict:
        result: dict = {}
        result["metadata"] = from_list(lambda x: to_class(Metadatum, x), self.metadata)
        result["speeches"] = from_list(lambda x: to_class(Speech, x), self.speeches)
        return result


def transcript_from_dict(s: Any) -> Transcript:
    return Transcript.from_dict(s)


def transcript_to_dict(x: Transcript) -> Any:
    return to_class(Transcript, x)


def renumber_question_indices(transcripts: list[Transcript]) -> list[Transcript]:
    grouped_metadata: dict[str, list[Metadatum]] = {}
    print(f"{len(transcripts)=}")
    for transcript in transcripts:
        for metadatum in transcript.metadata:
            if metadatum.debate_identifier not in grouped_metadata:
                grouped_metadata[metadatum.debate_identifier] = []
            grouped_metadata[metadatum.debate_identifier].append(metadatum)

    for (index, (identifier, metadata)) in enumerate(list(grouped_metadata.items())):
        for metadatum in metadata:
            metadatum.question_idx = index
    return transcripts


def read_jsons() -> list[tuple[Transcript, Path]]:
    # Replace 'data.json' with your actual file path
    directory_path = '../outputs/DataGeneration-Llama3-SingleRound-FullTrain-SFT/outputs/transcripts'
    directory = Path(directory_path)

    if not directory.exists():
        print(f"Error: Directory '{directory_path}' does not exist")
        return []

    if not directory.is_dir():
        print(f"Error: '{directory_path}' is not a directory")
        return []

    jsons_with_paths = []
    for file_path in directory.iterdir():
        if file_path.is_file():
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.loads(file.read())
                jsons_with_paths.append((Transcript.from_dict(data), file_path))

    return jsons_with_paths


def write_transcripts(transcripts_with_paths: list[tuple[Transcript, Path]]):
    for transcript, file_path in transcripts_with_paths:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(transcript.to_dict(), file, indent=2)
    print(f"Successfully wrote {len(transcripts_with_paths)} transcripts back to their original locations.")


def main():
    jsons_with_paths = read_jsons()
    transcripts = [t for t, _ in jsons_with_paths]
    renumbered_transcripts = renumber_question_indices(transcripts)
    # Re-associate renumbered transcripts with their original paths
    renumbered_jsons_with_paths = [(t, p) for t, p in
                                   zip(renumbered_transcripts, [path for _, path in jsons_with_paths])]
    print([r[1] for r in renumbered_jsons_with_paths])
    write_transcripts(renumbered_jsons_with_paths)


if __name__ == "__main__":
    main()
