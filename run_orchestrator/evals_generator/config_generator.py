import argparse
import dataclasses
from collections import ChainMap
from collections.abc import Mapping
from itertools import product

import yaml

from data.dataset import DatasetConfig, SplitType, DatasetType
from debate import SpeechFormatStructure, AgentConfig, MultiRoundBranchingSetting
from experiments.experiment_loader import ExperimentConfig, AgentsConfig, TournamentType
from models import ModelSettings, GenerationParams, ModelType
from prompts import PromptLoadingConfig
from run_orchestrator.evals_generator.model_definitions import ALL_VALID_DEBATERS, ALL_VALID_JUDGES, \
    DebaterModelConfiguration, \
    DebaterTrainingRound, JudgeModelConfiguration


@dataclasses.dataclass
class ConfigArgs:
    """Configuration arguments for the YAML validator."""
    output: str = "run_orchestrator/evals_generator/generated_configs/mars_experiments.yaml"


class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


@dataclasses.dataclass
class TaskTypeParams:
    generation_params: GenerationParams
    eval_dataset_params: DatasetConfig
    data_generation_dataset_params: DatasetConfig
    prompt_config: PromptLoadingConfig


def enum_representer(dumper, data):
    return dumper.represent_str(data.name.lower())


TASK_TYPE_PARAMS = {
    "lojban": TaskTypeParams(
        generation_params=GenerationParams(
            temperature=0.5,
            max_new_tokens=2000
        ),
        eval_dataset_params=DatasetConfig(
            dataset_type="lojban",
            split_type="test",
            shuffle_deterministically=True
        ),
        data_generation_dataset_params=DatasetConfig(
            dataset_type="lojban",
            split_type="train",
            shuffle_deterministically=True
        ),
        prompt_config=PromptLoadingConfig(
            file_path="/home/ubuntu/mars-arnesen-gh/garethtan/prompts/configs/lojban_prompts.yaml"
        )
    ),
    "quality": TaskTypeParams(
        generation_params=GenerationParams(
            temperature=0.5
        ),
        eval_dataset_params=DatasetConfig(
            dataset_type="quality",
            split_type="val",
            shuffle_deterministically=True
        ),
        data_generation_dataset_params=DatasetConfig(
            dataset_type="quality",
            split_type="train",
            flip_sides=False,
            shuffle_deterministically=True
        ),
        prompt_config=PromptLoadingConfig()
    )
}


def process_debaters_and_judges() -> tuple[dict[str, DebaterModelConfiguration], dict[str, JudgeModelConfiguration]]:
    processed_debaters = {}
    for name, debater in ALL_VALID_DEBATERS.items():
        if debater.settings.alias != "":
            raise Exception(
                f"Alias for debater {name} was set. It should be the empty string. It is set automatically.")

        processed_debaters[name] = DebaterModelConfiguration(
            training_round=debater.training_round,
            is_reasoning=debater.is_reasoning,
            settings=ModelSettings(
                model_type=debater.settings.model_type,
                alias=f"{name}-debater",
                model_file_path=debater.settings.model_file_path,
                require_quote_validation=debater.settings.require_quote_validation
            )
        )

    processed_judges = {}
    for name, judge in ALL_VALID_JUDGES.items():
        if judge.settings.alias != "":
            raise Exception(f"Alias for judge {name} was set. It should be the empty string. It is set automatically.")

        processed_judges[name] = JudgeModelConfiguration(
            training_round=judge.training_round,
            settings=ModelSettings(
                model_type=judge.settings.model_type,
                alias=f"{name}-judge",
                model_file_path=judge.settings.model_file_path,
                require_quote_validation=judge.settings.require_quote_validation
            )
        )

    return processed_debaters, processed_judges


def run_config_generator(args: ConfigArgs):
    debaters, judges = process_debaters_and_judges()

    eval_configurations = generate_eval_configurations(debaters, judges)
    data_generation_configs = generate_data_generation_configurations(debaters, judges)

    write_configurations_to_disk(args, ChainMap(eval_configurations, data_generation_configs))


def generate_config_name(
        config_type: str,
        debater: DebaterModelConfiguration,
        judge: JudgeModelConfiguration,
        task_type_name: str,
) -> str:
    """Generates a configuration name."""
    return f"{config_type}--{debater.settings.alias}_{debater.training_round.display_name}--{judge.settings.alias}_{judge.training_round.display_name}--{task_type_name}"


def generate_data_generation_configurations(debaters: dict[str, DebaterModelConfiguration],
                                            judges: dict[str, JudgeModelConfiguration]) -> dict[str, ExperimentConfig]:
    # all configurations are based on Quality
    # data generation configurations are only generated for
    # SFT only models and round one DPO trained models
    debater_judge_task_types = product(
        [d for d in debaters.values() if
         d.training_round in {DebaterTrainingRound.SFT_ONLY, DebaterTrainingRound.ROUND_ONE_DPO}],
        judges.values(),
        [("quality", TASK_TYPE_PARAMS["quality"])]
    )

    configurations: dict[str, ExperimentConfig] = {}
    for (debater, judge, (task_type_name, task_type_params)) in debater_judge_task_types:
        name = generate_config_name("data-generation", debater, judge, task_type_name)
        debater = debater.settings.model_copy(
            update={'generation_params': task_type_params.generation_params}
        )

        configurations[name] = ExperimentConfig(
            batch_size=1,
            num_speeches=2,
            flip=True,
            enable_self_debate=True,
            multi_round_branching=MultiRoundBranchingSetting.HALF,
            agents=build_agent_config(debater, judge.settings),
            dataset=task_type_params.data_generation_dataset_params,
        )

    return configurations


def build_agent_config(debater: ModelSettings, judge: ModelSettings) -> AgentsConfig:
    return AgentsConfig(
        debaters=[
            AgentConfig(
                model_settings=debater
            )
        ],
        judge=AgentConfig(
            model_settings=judge
        )
    )


def generate_eval_configurations(debaters: dict[str, DebaterModelConfiguration],
                                 judges: dict[str, JudgeModelConfiguration]) -> dict[
    str, ExperimentConfig]:
    debater_judge_task_types = product(debaters.values(), judges.values(), TASK_TYPE_PARAMS.items())

    configurations: dict[str, ExperimentConfig] = {}
    for (debater, judge, (task_type_name, task_type_params)) in debater_judge_task_types:
        name = generate_config_name("eval", debater, judge, task_type_name)

        if debater.is_reasoning:
            task_type_params.generation_params.max_new_tokens = 1500

        debater = debater.settings.model_copy(
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
            agents=build_agent_config(debater, judge.settings),
            dataset=task_type_params.eval_dataset_params,
        )

    return configurations


def write_configurations_to_disk(args: ConfigArgs, configurations: Mapping[str, ExperimentConfig]):
    write_config = {k: v.model_dump() for k, v in configurations.items()}
    yaml.add_representer(DatasetType, enum_representer, Dumper=NoAliasDumper)
    yaml.add_representer(SplitType, enum_representer, Dumper=NoAliasDumper)
    yaml.add_representer(ModelType, enum_representer, Dumper=NoAliasDumper)
    yaml.add_representer(TournamentType, enum_representer, Dumper=NoAliasDumper)
    yaml.add_representer(SpeechFormatStructure, enum_representer, Dumper=NoAliasDumper)
    yaml.add_representer(MultiRoundBranchingSetting, enum_representer, Dumper=NoAliasDumper)

    with open(args.output, 'w') as f:
        yaml.dump(write_config, f, default_flow_style=False, sort_keys=True, Dumper=NoAliasDumper)


def parse_args() -> ConfigArgs:
    """Parse command line arguments and return as a ConfigArgs dataclass."""
    parser = argparse.ArgumentParser(
        description="Generates a standard_experiment style configuration for the cartesian products of debaters and judges."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=ConfigArgs.output,
        help="Where to generate the configuration file to."
    )
    args = parser.parse_args()

    return ConfigArgs(output=args.output)


if __name__ == "__main__":
    config = parse_args()
    run_config_generator(config)
