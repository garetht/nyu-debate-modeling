# %%

from inspect_ai.dataset import json_dataset, Sample
from transformers import AutoTokenizer

from simple_baseline import locations
from simple_baseline.data_models.lojban import LojbanTranscript

tokenizer = AutoTokenizer.from_pretrained("gradientai/Llama-3-8B-Instruct-262k")


def lojban_record_to_sample(record: dict):
    transcript = LojbanTranscript.from_dict(record)
    # tokens = tokenizer.encode(transcript.prompt_file_content)

    # prompt_file_content = tokenizer.decode(tokens)

    return Sample(
        input=f"{transcript.prompt_file_content} {transcript.prompt}",
        choices=transcript.answers,
        target=transcript.original_key,
        id=transcript.original_id,
        metadata={
            "id": transcript.original_id
        }
    )


def filter_lojban_sample(sample: Sample):
    return sample.metadata["id"] == "jbo_1"

lojban_dataset = json_dataset(
    str(locations.PROJECT_ROOT / "data" / "datasets" / "lojban" / "lojban_dataset_test.jsonl"),
    lojban_record_to_sample
)

