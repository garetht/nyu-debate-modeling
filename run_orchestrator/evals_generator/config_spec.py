import dataclasses
import enum

from data.dataset import DatasetConfig
from models import GenerationParams
from prompts import PromptLoadingConfig


@dataclasses.dataclass
class TaskTypeParams:
    generation_params: GenerationParams
    eval_dataset_params: DatasetConfig
    data_generation_dataset_params: DatasetConfig
    prompt_config: PromptLoadingConfig


class ConfigurationType(enum.Enum):
    EVAL = "eval"
    DATA_GENERATION = "data-generation"

    def __str__(self) -> str:
        return self.value


TASK_TYPE_PARAMS = {
    "lojban": TaskTypeParams(
        generation_params=GenerationParams(
            temperature=0.5,
            max_new_tokens=2000,
        ),
        eval_dataset_params=DatasetConfig(
            dataset_type="lojban",
            split_type="test",
            shuffle_deterministically=True,
        ),
        data_generation_dataset_params=DatasetConfig(
            dataset_type="lojban",
            split_type="train",
            shuffle_deterministically=True,
        ),
        prompt_config=PromptLoadingConfig(
            file_path="/home/ubuntu/mars-arnesen-gh/garethtan/prompts/configs/lojban_prompts.yaml"
        ),
    ),
    "quality": TaskTypeParams(
        generation_params=GenerationParams(
            temperature=0.5,
        ),
        eval_dataset_params=DatasetConfig(
            dataset_type="quality",
            split_type="val",
            shuffle_deterministically=True,
        ),
        data_generation_dataset_params=DatasetConfig(
            dataset_type="quality",
            split_type="train",
            flip_sides=False,
            shuffle_deterministically=True,
        ),
        prompt_config=PromptLoadingConfig(),
    ),
}
