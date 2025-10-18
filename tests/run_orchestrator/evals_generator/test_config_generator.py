import pytest

# pytest relies on ModelSettings for type compatibility, but we avoid importing torch.
from models import ModelSettings, ModelType
from run_orchestrator.evals_generator.config_spec import ConfigurationType
from run_orchestrator.evals_generator.configuration_name import ConfigurationName
from run_orchestrator.evals_generator.model_definitions import (
    DebaterModelConfiguration,
    DebaterTrainingRound,
    JudgeModelConfiguration,
    JudgeTrainingRound,
)

# Literal copy of the valid model definitions to keep the test resilient to future changes.
VALID_DEBATERS: dict[str, DebaterModelConfiguration] = {
    "llama-3-262k": DebaterModelConfiguration(
        training_round=DebaterTrainingRound.UNTRAINED,
        is_reasoning=False,
        settings=ModelSettings(
            model_type=ModelType.LLAMA3,
            alias="",
            require_quote_validation=True,
            model_file_path="/home/ubuntu/mars-arnesen-gh/garethtan/downloaded-models/gradientai/Llama-3-8B-Instruct-262k",
        ),
    ),
    "llama-3-262k-2025-07-31": DebaterModelConfiguration(
        training_round=DebaterTrainingRound.SFT_ONLY,
        is_reasoning=False,
        settings=ModelSettings(
            model_type=ModelType.LLAMA3,
            alias="",
            model_file_path="/home/ubuntu/mars-arnesen-gh/leonidtsyplenkov/models/trained_models/llama-3-mega-merged-no-judge-speeches-31.07",
        ),
    ),
    "llama-3-262k-41-judge": DebaterModelConfiguration(
        training_round=DebaterTrainingRound.ROUND_TWO_DPO,
        is_reasoning=False,
        settings=ModelSettings(
            model_type=ModelType.LLAMA3,
            alias="",
            model_file_path="/home/ubuntu/mars-arnesen-gh/garethtan/models/trained_models/llama-3-DPO-811-FullTrainDebateRoundTwo-full-trained",
        ),
    ),
    "o4-mini-rft-2025-09-15": DebaterModelConfiguration(
        training_round=DebaterTrainingRound.RFT,
        is_reasoning=True,
        settings=ModelSettings(
            model_type=ModelType.OPENAI,
            alias="",
            model_file_path="ft:o4-mini-2025-04-16:modulo-research-ltd:michael-and-khan-data-debater-15-09:CGO8p7XH",
            require_quote_validation=True,
        ),
    ),
}

VALID_JUDGES: dict[str, JudgeModelConfiguration] = {
    "gpt-4-turbo-2024-04-09": JudgeModelConfiguration(
        training_round=JudgeTrainingRound.UNTRAINED,
        settings=ModelSettings(
            model_type=ModelType.OPENAI,
            alias="",
            model_file_path="gpt-4-turbo-2024-04-09",
        ),
    ),
    "gpt-41-2025-07-31": JudgeModelConfiguration(
        training_round=JudgeTrainingRound.SFT_ONLY,
        settings=ModelSettings(
            model_type=ModelType.OPENAI,
            alias="",
            model_file_path="ft:gpt-4.1-2025-04-14:modulo-research-ltd:michael-and-khan-data-judge-31-07:BzYGc8SU",
        ),
    ),
    "gpt-41-2025-04-14": JudgeModelConfiguration(
        training_round=JudgeTrainingRound.UNTRAINED,
        settings=ModelSettings(
            model_type=ModelType.OPENAI,
            alias="",
            model_file_path="gpt-4.1-2025-04-14",
        ),
    ),
    "llama-3-262k": JudgeModelConfiguration(
        training_round=JudgeTrainingRound.UNTRAINED,
        settings=ModelSettings(
            model_type=ModelType.LLAMA3,
            alias="",
            require_quote_validation=True,
            model_file_path="/home/ubuntu/mars-arnesen-gh/garethtan/downloaded-models/gradientai/Llama-3-8B-Instruct-262k",
        ),
    ),
    "llama-3-262k-2025-10-08": JudgeModelConfiguration(
        training_round=JudgeTrainingRound.SFT_ONLY,
        settings=ModelSettings(
            model_type=ModelType.LLAMA3,
            alias="",
            model_file_path="/home/ubuntu/mars-arnesen-gh/leonidtsyplenkov/models/trained_models/llama-3-mega-merged-judge-08-10",
        ),
    ),
}


EVAL_QUALITY_CASES = [
    ("llama-3-262k", "gpt-4-turbo-2024-04-09", "eval--llama-3-262k-debater_untrained--gpt-4-turbo-2024-04-09-judge_untrained--quality"),
    ("llama-3-262k", "gpt-41-2025-07-31", "eval--llama-3-262k-debater_untrained--gpt-41-2025-07-31-judge_sft-only--quality"),
    ("llama-3-262k", "gpt-41-2025-04-14", "eval--llama-3-262k-debater_untrained--gpt-41-2025-04-14-judge_untrained--quality"),
    ("llama-3-262k", "llama-3-262k", "eval--llama-3-262k-debater_untrained--llama-3-262k-judge_untrained--quality"),
    ("llama-3-262k", "llama-3-262k-2025-10-08", "eval--llama-3-262k-debater_untrained--llama-3-262k-2025-10-08-judge_sft-only--quality"),
    ("llama-3-262k-2025-07-31", "gpt-4-turbo-2024-04-09", "eval--llama-3-262k-2025-07-31-debater_sft-only--gpt-4-turbo-2024-04-09-judge_untrained--quality"),
    ("llama-3-262k-2025-07-31", "gpt-41-2025-07-31", "eval--llama-3-262k-2025-07-31-debater_sft-only--gpt-41-2025-07-31-judge_sft-only--quality"),
    ("llama-3-262k-2025-07-31", "gpt-41-2025-04-14", "eval--llama-3-262k-2025-07-31-debater_sft-only--gpt-41-2025-04-14-judge_untrained--quality"),
    ("llama-3-262k-2025-07-31", "llama-3-262k", "eval--llama-3-262k-2025-07-31-debater_sft-only--llama-3-262k-judge_untrained--quality"),
    ("llama-3-262k-2025-07-31", "llama-3-262k-2025-10-08", "eval--llama-3-262k-2025-07-31-debater_sft-only--llama-3-262k-2025-10-08-judge_sft-only--quality"),
    ("llama-3-262k-41-judge", "gpt-4-turbo-2024-04-09", "eval--llama-3-262k-41-judge-debater_round-two-dpo--gpt-4-turbo-2024-04-09-judge_untrained--quality"),
    ("llama-3-262k-41-judge", "gpt-41-2025-07-31", "eval--llama-3-262k-41-judge-debater_round-two-dpo--gpt-41-2025-07-31-judge_sft-only--quality"),
    ("llama-3-262k-41-judge", "gpt-41-2025-04-14", "eval--llama-3-262k-41-judge-debater_round-two-dpo--gpt-41-2025-04-14-judge_untrained--quality"),
    ("llama-3-262k-41-judge", "llama-3-262k", "eval--llama-3-262k-41-judge-debater_round-two-dpo--llama-3-262k-judge_untrained--quality"),
    ("llama-3-262k-41-judge", "llama-3-262k-2025-10-08", "eval--llama-3-262k-41-judge-debater_round-two-dpo--llama-3-262k-2025-10-08-judge_sft-only--quality"),
    ("o4-mini-rft-2025-09-15", "gpt-4-turbo-2024-04-09", "eval--o4-mini-rft-2025-09-15-debater_rft--gpt-4-turbo-2024-04-09-judge_untrained--quality"),
    ("o4-mini-rft-2025-09-15", "gpt-41-2025-07-31", "eval--o4-mini-rft-2025-09-15-debater_rft--gpt-41-2025-07-31-judge_sft-only--quality"),
    ("o4-mini-rft-2025-09-15", "gpt-41-2025-04-14", "eval--o4-mini-rft-2025-09-15-debater_rft--gpt-41-2025-04-14-judge_untrained--quality"),
    ("o4-mini-rft-2025-09-15", "llama-3-262k", "eval--o4-mini-rft-2025-09-15-debater_rft--llama-3-262k-judge_untrained--quality"),
    ("o4-mini-rft-2025-09-15", "llama-3-262k-2025-10-08", "eval--o4-mini-rft-2025-09-15-debater_rft--llama-3-262k-2025-10-08-judge_sft-only--quality"),
]

DATA_GENERATION_LOJBAN_CASES = [
    ("llama-3-262k", "gpt-4-turbo-2024-04-09", "data-generation--llama-3-262k-debater_untrained--gpt-4-turbo-2024-04-09-judge_untrained--lojban"),
    ("llama-3-262k", "gpt-41-2025-07-31", "data-generation--llama-3-262k-debater_untrained--gpt-41-2025-07-31-judge_sft-only--lojban"),
    ("llama-3-262k", "gpt-41-2025-04-14", "data-generation--llama-3-262k-debater_untrained--gpt-41-2025-04-14-judge_untrained--lojban"),
    ("llama-3-262k", "llama-3-262k", "data-generation--llama-3-262k-debater_untrained--llama-3-262k-judge_untrained--lojban"),
    ("llama-3-262k", "llama-3-262k-2025-10-08", "data-generation--llama-3-262k-debater_untrained--llama-3-262k-2025-10-08-judge_sft-only--lojban"),
    ("llama-3-262k-2025-07-31", "gpt-4-turbo-2024-04-09", "data-generation--llama-3-262k-2025-07-31-debater_sft-only--gpt-4-turbo-2024-04-09-judge_untrained--lojban"),
    ("llama-3-262k-2025-07-31", "gpt-41-2025-07-31", "data-generation--llama-3-262k-2025-07-31-debater_sft-only--gpt-41-2025-07-31-judge_sft-only--lojban"),
    ("llama-3-262k-2025-07-31", "gpt-41-2025-04-14", "data-generation--llama-3-262k-2025-07-31-debater_sft-only--gpt-41-2025-04-14-judge_untrained--lojban"),
    ("llama-3-262k-2025-07-31", "llama-3-262k", "data-generation--llama-3-262k-2025-07-31-debater_sft-only--llama-3-262k-judge_untrained--lojban"),
    ("llama-3-262k-2025-07-31", "llama-3-262k-2025-10-08", "data-generation--llama-3-262k-2025-07-31-debater_sft-only--llama-3-262k-2025-10-08-judge_sft-only--lojban"),
    ("llama-3-262k-41-judge", "gpt-4-turbo-2024-04-09", "data-generation--llama-3-262k-41-judge-debater_round-two-dpo--gpt-4-turbo-2024-04-09-judge_untrained--lojban"),
    ("llama-3-262k-41-judge", "gpt-41-2025-07-31", "data-generation--llama-3-262k-41-judge-debater_round-two-dpo--gpt-41-2025-07-31-judge_sft-only--lojban"),
    ("llama-3-262k-41-judge", "gpt-41-2025-04-14", "data-generation--llama-3-262k-41-judge-debater_round-two-dpo--gpt-41-2025-04-14-judge_untrained--lojban"),
    ("llama-3-262k-41-judge", "llama-3-262k", "data-generation--llama-3-262k-41-judge-debater_round-two-dpo--llama-3-262k-judge_untrained--lojban"),
    ("llama-3-262k-41-judge", "llama-3-262k-2025-10-08", "data-generation--llama-3-262k-41-judge-debater_round-two-dpo--llama-3-262k-2025-10-08-judge_sft-only--lojban"),
    ("o4-mini-rft-2025-09-15", "gpt-4-turbo-2024-04-09", "data-generation--o4-mini-rft-2025-09-15-debater_rft--gpt-4-turbo-2024-04-09-judge_untrained--lojban"),
    ("o4-mini-rft-2025-09-15", "gpt-41-2025-07-31", "data-generation--o4-mini-rft-2025-09-15-debater_rft--gpt-41-2025-07-31-judge_sft-only--lojban"),
    ("o4-mini-rft-2025-09-15", "gpt-41-2025-04-14", "data-generation--o4-mini-rft-2025-09-15-debater_rft--gpt-41-2025-04-14-judge_untrained--lojban"),
    ("o4-mini-rft-2025-09-15", "llama-3-262k", "data-generation--o4-mini-rft-2025-09-15-debater_rft--llama-3-262k-judge_untrained--lojban"),
    ("o4-mini-rft-2025-09-15", "llama-3-262k-2025-10-08", "data-generation--o4-mini-rft-2025-09-15-debater_rft--llama-3-262k-2025-10-08-judge_sft-only--lojban"),
]


@pytest.fixture()
def processed_model_configs() -> tuple[dict[str, DebaterModelConfiguration], dict[str, JudgeModelConfiguration]]:
    debaters = {
        name: DebaterModelConfiguration(
            training_round=config.training_round,
            is_reasoning=config.is_reasoning,
            settings=config.settings.model_copy(update={"alias": f"{name}-debater"}),
        )
        for name, config in VALID_DEBATERS.items()
    }
    judges = {
        name: JudgeModelConfiguration(
            training_round=config.training_round,
            settings=config.settings.model_copy(update={"alias": f"{name}-judge"}),
        )
        for name, config in VALID_JUDGES.items()
    }
    return debaters, judges


@pytest.mark.parametrize("debater_name, judge_name, expected", EVAL_QUALITY_CASES)
def test_generate_config_name_eval_quality(processed_model_configs, debater_name, judge_name, expected):
    debaters, judges = processed_model_configs
    result = ConfigurationName.serialize_from_inputs(ConfigurationType.EVAL, debaters[debater_name], judges[judge_name], "quality")
    assert result == expected


@pytest.mark.parametrize("debater_name, judge_name, expected", DATA_GENERATION_LOJBAN_CASES)
def test_generate_config_name_data_generation_lojban(processed_model_configs, debater_name, judge_name, expected):
    debaters, judges = processed_model_configs
    result = ConfigurationName.serialize_from_inputs(ConfigurationType.DATA_GENERATION, debaters[debater_name], judges[judge_name], "lojban")
    assert result == expected


def test_generate_config_name_reflects_overridden_aliases():
    debater = DebaterModelConfiguration(
        training_round=DebaterTrainingRound.SFT_ONLY,
        is_reasoning=False,
        settings=ModelSettings(model_type=ModelType.LLAMA3, alias="custom-debater-alias"),
    )
    judge = JudgeModelConfiguration(
        training_round=JudgeTrainingRound.SFT_ONLY,
        settings=ModelSettings(model_type=ModelType.OPENAI, alias="custom-judge-alias"),
    )

    expected = "eval--custom-debater-alias_sft-only--custom-judge-alias_sft-only--quality"
    assert ConfigurationName.serialize_from_inputs(ConfigurationType.EVAL, debater, judge, "quality") == expected

