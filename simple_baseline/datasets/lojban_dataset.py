# %%

from inspect_ai.dataset import json_dataset, Sample

from simple_baseline import locations
from simple_baseline.data_models.lojban import LojbanTranscript


def lojban_record_to_sample(record: dict):
    transcript = LojbanTranscript.from_dict(record)
    return Sample(
        input=f"{transcript.prompt_file_content} {transcript.prompt}",
        choices=transcript.answers,
        target=transcript.original_key,
        id=transcript.original_id,
        metadata={
            "id": transcript.original_id
        }
    )

lojban_dataset = json_dataset(
    str(locations.PROJECT_ROOT / "data" / "datasets" / "lojban" / "lojban_dataset.jsonl"),
    lojban_record_to_sample
)

