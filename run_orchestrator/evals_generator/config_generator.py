import argparse
import dataclasses
from itertools import product

import yaml

from data.dataset import DatasetConfig, SplitType, DatasetType
from debate import SpeechFormatStructure, AgentConfig, MultiRoundBranchingSetting
from experiments.experiment_loader import ExperimentConfig, AgentsConfig, TournamentType
from models import ModelSettings, GenerationParams, ModelType
from prompts import PromptConfig, PromptLoadingConfig


class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True

VALID_DEBATERS: dict[str, ModelSettings] = {
    "llama-3-262k-no-sft-no-rl": ModelSettings(
        model_type=ModelType.LLAMA3,
        alias="",
        require_quote_validation=True,
        model_file_path="/home/ubuntu/mars-arnesen-gh/garethtan/downloaded-models/gradientai/Llama-3-8B-Instruct-262k"
    ),
    "llama-3-262k-sft-no-rl": ModelSettings(
        model_type=ModelType.LLAMA3,
        alias="",
        model_file_path="/home/ubuntu/mars-arnesen-gh/leonidtsyplenkov/models/trained_models/llama-3-mega-merged-no-judge-speeches-31.07"
    ),
    "llama-3-262k-sft-dpo-41-judge": ModelSettings(
        model_type=ModelType.LLAMA3,
        alias="",
        model_file_path="/home/ubuntu/mars-arnesen-gh/garethtan/models/trained_models/llama-3-DPO-811-FullTrainDebateRoundTwo-full-trained"
    ),
    "o4-mini-rft-2025-09-15": ModelSettings(
        model_type=ModelType.OPENAI,
        alias="",
        model_file_path="ft:o4-mini-2025-04-16:modulo-research-ltd:michael-and-khan-data-debater-15-09:CGO8p7XH",
        require_quote_validation=True,
    ),
}

VALID_JUDGES = {
    "gpt-4-turbo-2024-04-09": ModelSettings(
        model_type=ModelType.OPENAI,
        alias="",
        model_file_path="gpt-4-turbo-2024-04-09"
    ),
    "gpt-41-sft-2025-07-31": ModelSettings(
        model_type=ModelType.OPENAI,
        alias="",
        model_file_path="ft:gpt-4.1-2025-04-14:modulo-research-ltd:michael-and-khan-data-judge-31-07:BzYGc8SU"
    ),
    "gpt-41-no-sft": ModelSettings(
        model_type=ModelType.OPENAI,
        alias="",
        model_file_path="gpt-4.1-2025-04-14"
    ),
    "llama-3-262k-no-sft-no-rl": ModelSettings(
        model_type=ModelType.LLAMA3,
        alias="",
    ),
}


@dataclasses.dataclass
class TaskTypeParams:
    generation_params: GenerationParams
    dataset_params: DatasetConfig
    prompt_config: PromptLoadingConfig

def enum_representer(dumper, data):
    return dumper.represent_str(data.name.lower())

TASK_TYPE_PARAMS = {
    "lojban": TaskTypeParams(
        generation_params=GenerationParams(
            temperature=0.5,
            max_new_tokens=2000
        ),
        dataset_params=DatasetConfig(
            dataset_type="lojban",
            split_type="test",
        ),
        prompt_config=PromptLoadingConfig(
            file_path="/home/ubuntu/mars-arnesen-gh/garethtan/prompts/configs/lojban_prompts.yaml"
        )
    ),
    "quality": TaskTypeParams(
        generation_params=GenerationParams(
            temperature=0.5
        ),
        dataset_params=DatasetConfig(
            dataset_type="quality",
            split_type="val",
        ),
        prompt_config=PromptLoadingConfig()
    )
}


def process_debaters_and_judges() -> tuple[dict[str, ModelSettings], dict[str, ModelSettings]]:
    processed_debaters = {}
    for name, debater in VALID_DEBATERS.items():
        if debater.alias != "":
            raise Exception(
                f"Alias for debater {name} was set. It should be the empty string. It is set automatically.")

        processed_debaters[name] = ModelSettings(
            model_type=debater.model_type,
            alias=f"{name}-debater",
            model_file_path=debater.model_file_path,
            require_quote_validation=debater.require_quote_validation
        )

    processed_judges = {}
    for name, judge in VALID_JUDGES.items():
        if judge.alias != "":
            raise Exception(f"Alias for judge {name} was set. It should be the empty string. It is set automatically.")

        processed_judges[name] = ModelSettings(
            model_type=judge.model_type,
            alias=f"{name}-judge",
            model_file_path=judge.model_file_path,
            require_quote_validation=judge.require_quote_validation
        )

    return processed_debaters, processed_judges


def run_config_generator(args):
    debaters, judges = process_debaters_and_judges()
    debater_judge_task_types = product(debaters.values(), judges.values(), TASK_TYPE_PARAMS.items())

    configurations = {}
    for (debater, judge, (task_type_name, task_type_params)) in debater_judge_task_types:
        name = f"{debater.alias}_{judge.alias}_{task_type_name}"
        debater = debater.model_copy(
            update={'generation_params': task_type_params.generation_params}
        )
        # debater.generation_params=task_type_params.generation_params
        print(task_type_params.generation_params)

        configurations[name] = ExperimentConfig(
            batch_size=1,
            num_speeches=2,
            flip=False,
            enable_self_debate=True,
            speech_structure=SpeechFormatStructure.DEFAULT_DEBATE,
            alternate=False,
            prompt_config=task_type_params.prompt_config,
            agents=AgentsConfig(
                debaters=[
                    AgentConfig(
                        model_settings=debater
                    )
                ],
                judge=AgentConfig(
                    model_settings=judge
                )
            ),
            dataset=task_type_params.dataset_params,
        )

        write_config = {k: v.model_dump() for k, v in configurations.items()}
        yaml.add_representer(DatasetType, enum_representer, Dumper=NoAliasDumper)
        yaml.add_representer(SplitType, enum_representer, Dumper=NoAliasDumper)
        yaml.add_representer(ModelType, enum_representer, Dumper=NoAliasDumper)
        yaml.add_representer(TournamentType, enum_representer, Dumper=NoAliasDumper)
        yaml.add_representer(SpeechFormatStructure, enum_representer, Dumper=NoAliasDumper)
        yaml.add_representer(MultiRoundBranchingSetting, enum_representer, Dumper=NoAliasDumper)

        with open(args.output, 'w') as f:
            yaml.dump(write_config, f, default_flow_style=False, sort_keys=False, Dumper=NoAliasDumper)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Validate a YAML configuration file against a JSON schema.")
    parser.add_argument(
        "--output",
        type=str,
        default="run_orchestrator/evals_generator/generated_configs/mars_experiments.yaml",
        help="Path to the YAML configuration file to validate."
    )
    args = parser.parse_args()
    run_config_generator(args)
