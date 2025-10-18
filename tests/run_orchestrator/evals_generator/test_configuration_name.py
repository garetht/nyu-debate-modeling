import pytest

from run_orchestrator.evals_generator.config_spec import ConfigurationType
from run_orchestrator.evals_generator.configuration_name import ConfigurationName
from run_orchestrator.evals_generator.model_definitions import (
    ALL_VALID_DEBATERS,
    ALL_VALID_JUDGES,
    DebaterModelConfiguration,
    JudgeModelConfiguration,
)


def _clone_debater_with_alias(name: str, alias: str) -> DebaterModelConfiguration:
    base = ALL_VALID_DEBATERS[name]
    return DebaterModelConfiguration(
        training_round=base.training_round,
        is_reasoning=base.is_reasoning,
        settings=base.settings.model_copy(update={"alias": alias}),
    )


def _clone_judge_with_alias(name: str, alias: str) -> JudgeModelConfiguration:
    base = ALL_VALID_JUDGES[name]
    return JudgeModelConfiguration(
        training_round=base.training_round,
        settings=base.settings.model_copy(update={"alias": alias}),
    )


@pytest.fixture
def sample_configs() -> tuple[tuple[str, DebaterModelConfiguration], tuple[str, JudgeModelConfiguration]]:
    debater_name = "llama-3-262k"
    judge_name = "gpt-4-turbo-2024-04-09"
    debater_alias = f"{debater_name}-debater"
    judge_alias = f"{judge_name}-judge"
    return (
        (debater_name, _clone_debater_with_alias(debater_name, debater_alias)),
        (judge_name, _clone_judge_with_alias(judge_name, judge_alias)),
    )


def test_serialize_from_inputs_formats_configuration(sample_configs):
    (debater_key, debater_cfg), (judge_key, judge_cfg) = sample_configs

    result = ConfigurationName.serialize_from_inputs(
        ConfigurationType.EVAL, debater_cfg, judge_cfg, "quality"
    )

    expected = "eval--llama-3-262k-debater_untrained--gpt-4-turbo-2024-04-09-judge_untrained--quality"
    assert result == expected


def test_serialize_deserialize_round_trip(sample_configs):
    (debater_key, debater_cfg), (judge_key, judge_cfg) = sample_configs
    serialized = ConfigurationName.serialize_from_inputs(
        ConfigurationType.DATA_GENERATION, debater_cfg, judge_cfg, "quality"
    )

    name = ConfigurationName.deserialize(serialized)

    assert name.config_type == ConfigurationType.DATA_GENERATION
    assert name.debater_key == debater_key
    assert name.judge_key == judge_key
    assert name.task_type_name == "quality"


def test_serialize_from_inputs_requires_enum(sample_configs):
    (_, debater_cfg), (_, judge_cfg) = sample_configs

    with pytest.raises(TypeError):
        ConfigurationName.serialize_from_inputs("eval", debater_cfg, judge_cfg, "quality")  # type: ignore[arg-type]


def test_serialize_from_inputs_requires_alias(sample_configs):
    (debater_key, _), (_, judge_cfg) = sample_configs
    base_debater = ALL_VALID_DEBATERS[debater_key]
    debater_without_alias = DebaterModelConfiguration(
        training_round=base_debater.training_round,
        is_reasoning=base_debater.is_reasoning,
        settings=base_debater.settings.model_copy(update={"alias": ""}),
    )

    with pytest.raises(ValueError):
        ConfigurationName.serialize_from_inputs(ConfigurationType.EVAL, debater_without_alias, judge_cfg, "quality")


def test_deserialize_rejects_unknown_debater_alias(sample_configs):
    judge_name, judge_cfg = sample_configs[1]
    judge_segment = f"{judge_cfg.settings.alias}_{judge_cfg.training_round.display_name}"
    malformed = f"eval--unknown-debater_untrained--{judge_segment}--quality"

    with pytest.raises(ValueError, match="Unknown debater alias"):
        ConfigurationName.deserialize(malformed)


def test_deserialize_rejects_mismatched_training(sample_configs):
    (debater_key, debater_cfg), (_, judge_cfg) = sample_configs
    debater_alias = f"{debater_key}-debater"
    serialized = f"eval--{debater_alias}_fake-training--{judge_cfg.settings.alias}_{judge_cfg.training_round.display_name}--quality"

    with pytest.raises(ValueError, match="Training round 'fake-training' does not match debater"):
        ConfigurationName.deserialize(serialized)
