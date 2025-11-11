from pathlib import Path

from run_orchestrator.analysis.analysis_models.analysis_result import AnalysisResult
from run_orchestrator.evals_generator.configuration_name import ConfigurationName


class EvaluationConfiguration(AnalysisResult):
    config_type: str
    task_type: str
    debater_name: str
    debater_training_round: str
    debater_is_reasoning: bool
    debater_model_type: str
    debater_max_new_tokens: int
    judge_name: str
    judge_training_round: str
    judge_is_reasoning: bool
    judge_model_type: str
    judge_max_new_tokens: int


    @staticmethod
    def from_configuration_name(name: ConfigurationName) -> 'EvaluationConfiguration':
        return EvaluationConfiguration(
            config_type=str(name.config_type),
            task_type=name.task_type_name,
            debater_name=name.debater_key,
            debater_training_round=str(name.debater_config.training_round),
            debater_is_reasoning=name.debater_config.settings.is_reasoning,
            debater_max_new_tokens=name.debater_config.settings.max_new_tokens,
            debater_model_type=name.debater_config.settings.model_type,
            judge_name=name.judge_key,
            judge_training_round=str(name.judge_config.training_round),
            judge_is_reasoning=name.judge_config.settings.is_reasoning,
            judge_max_new_tokens=name.judge_config.settings.max_new_tokens,
            judge_model_type=name.judge_config.settings.model_type,
        )
