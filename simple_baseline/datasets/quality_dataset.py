# %%

from inspect_ai.dataset import json_dataset, Sample

from simple_baseline import locations
from simple_baseline.data_models.quality import QualityTranscript


def quality_record_to_sample(record: dict):
    transcript = QualityTranscript.from_dict(record)
    return [
        Sample(
            input=f"{transcript.article} {question.question}",
            choices=question.options,
            target=chr(65 + question.gold_label - 1),
            id=question.question_unique_id,
            metadata={
                "id": f"{transcript.article_id}_{question.question_unique_id}"
            }
        )
        for question in transcript.questions
    ]

quality_dataset = json_dataset(
    str(locations.PROJECT_ROOT / "data" / "datasets" / "quality" / "QuALITY.v1.0.1.htmlstripped.dev.jsonl"),
    quality_record_to_sample
)

