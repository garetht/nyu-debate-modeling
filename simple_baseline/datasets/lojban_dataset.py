# %%
from typing import Callable

from inspect_ai.dataset import json_dataset, Sample
from transformers import AutoTokenizer

from simple_baseline import locations
from simple_baseline.data_models.lojban import LojbanTranscript

tokenizer = AutoTokenizer.from_pretrained("gradientai/Llama-3-8B-Instruct-262k")


def debater_input_creator(transcript: LojbanTranscript) -> str:
    return f"{transcript.prompt_file_content} {transcript.prompt}"


def judge_input_creator(transcript: LojbanTranscript) -> str:
    return transcript.prompt


def lojban_record_to_sample(input_creator: Callable[[LojbanTranscript], str]):
    def record_to_sample(record: dict):
        transcript = LojbanTranscript.from_dict(record)

        return Sample(
            input=input_creator(transcript),
            choices=transcript.answers,
            target=transcript.original_key,
            id=transcript.original_id,
            metadata={
                "id": transcript.original_id
            }
        )

    return record_to_sample


def filter_lojban_sample(sample: Sample):
    return sample.metadata["id"] == "jbo_1"

lojban_location = str(locations.PROJECT_ROOT / "data" / "datasets" / "lojban" / "lojban_dataset_test.jsonl")

debater_lojban_dataset = json_dataset(
    lojban_location,
    lojban_record_to_sample(debater_input_creator)
)

judge_lojban_dataset = json_dataset(
    lojban_location,
    lojban_record_to_sample(judge_input_creator)
)

