from run_orchestrator.analysis.analysis_models.analysis_result import AnalysisResult
from run_orchestrator.evals_generator.configuration_name import ConfigurationName


class EvaluationConfiguration(AnalysisResult):
    raw_name: str
    config_type: str
    task_type: str
    debater_name: str
    debater_base_model: str
    debater_training_round: str
    debater_is_reasoning: bool
    debater_model_type: str
    debater_max_new_tokens: int
    judge_base_model: str
    judge_name: str
    judge_training_round: str
    judge_model_type: str
    judge_max_new_tokens: int


    @staticmethod
    def from_configuration_name(name: ConfigurationName) -> 'EvaluationConfiguration':
        return EvaluationConfiguration(
            raw_name=name.serialize(),
            config_type=name.config_type.name,
            task_type=name.task_type_name,
            debater_name=name.debater_key,
            debater_base_model=name.debater_config.base_model.name,
            debater_training_round=name.debater_config.training_round.name,
            debater_is_reasoning=name.debater_config.is_reasoning,
            debater_max_new_tokens=name.debater_config.settings.generation_params.max_new_tokens,
            debater_model_type=name.debater_config.settings.model_type.name,
            judge_base_model=name.judge_config.base_model.name,
            judge_name=name.judge_key,
            judge_training_round=name.judge_config.training_round.name,
            judge_max_new_tokens=name.judge_config.settings.generation_params.max_new_tokens,
            judge_model_type=name.judge_config.settings.model_type.name,
        )
