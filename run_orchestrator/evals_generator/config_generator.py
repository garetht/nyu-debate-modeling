import argparse
import dataclasses
from collections import ChainMap
from collections.abc import Mapping
from itertools import product

import yaml

from data.dataset import SplitType, DatasetType
from debate import SpeechFormatStructure, AgentConfig, MultiRoundBranchingSetting
from experiment_models import ExperimentConfig, AgentsConfig, TournamentType
from models import ModelSettings, ModelType
from run_orchestrator.evals_generator.config_spec import ConfigurationType, TASK_TYPE_PARAMS
from run_orchestrator.evals_generator.configuration_name import ConfigurationName
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

def enum_representer(dumper, data):
    return dumper.represent_str(data.name.lower())


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


def parse_config_name(raw_name: str) -> ConfigurationName:
    """Parse a serialized configuration name back into its structured representation."""
    return ConfigurationName.deserialize(raw_name)


def generate_data_generation_configurations(debaters: dict[str, DebaterModelConfiguration],
                                            judges: dict[str, JudgeModelConfiguration]) -> dict[str, ExperimentConfig]:
    # all configurations are based on Quality
    # data generation configurations are only generated for
    # SFT only models and round one DPO trained models
    eligible_debaters = [
        (name, d)
        for name, d in debaters.items()
        if d.training_round in {DebaterTrainingRound.SFT_ONLY, DebaterTrainingRound.ROUND_ONE_DPO}
    ]
    debater_judge_task_types = product(
        eligible_debaters,
        judges.items(),
        [("quality", TASK_TYPE_PARAMS["quality"])]
    )

    configurations: dict[str, ExperimentConfig] = {}
    for ((debater_name, debater), (judge_name, judge), (task_type_name, task_type_params)) in debater_judge_task_types:
        name = ConfigurationName.serialize_from_inputs(ConfigurationType.DATA_GENERATION, debater, judge, task_type_name)
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
    debater_judge_task_types = product(debaters.items(), judges.items(), TASK_TYPE_PARAMS.items())

    configurations: dict[str, ExperimentConfig] = {}
    for ((debater_name, debater), (judge_name, judge), (task_type_name, task_type_params)) in debater_judge_task_types:
        name = ConfigurationName.serialize_from_inputs(ConfigurationType.EVAL, debater, judge, task_type_name)

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


def _normalize_judge_config(
        judge: JudgeModelConfiguration | str,
) -> tuple[str, str]:
    if isinstance(judge, str):
        try:
            config = ALL_VALID_JUDGES[judge]
        except KeyError as exc:
            raise ValueError(f"Unknown judge '{judge}'.") from exc
        alias = ConfigurationName._build_judge_alias(judge)
        training_display = config.training_round.display_name
        return alias, training_display
    if not isinstance(judge, JudgeModelConfiguration):
        raise TypeError("Judge must be provided as a configuration key or JudgeModelConfiguration.")
    alias = judge.settings.alias
    if not alias:
        raise ValueError("Judge alias must be set when passing a JudgeModelConfiguration.")
    return alias, judge.training_round.display_name


def _normalize_task_type(task_type_name: str) -> str:
    if task_type_name not in TASK_TYPE_PARAMS:
        raise ValueError(f"Unknown task type '{task_type_name}'.")
    return task_type_name


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
