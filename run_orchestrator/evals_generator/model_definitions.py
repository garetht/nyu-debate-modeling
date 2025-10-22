import dataclasses
import enum

from models import ModelSettings, ModelType


class DebaterTrainingRound(enum.Enum):
    UNTRAINED = -1
    SFT_ONLY = 0
    ROUND_ONE_DPO = 1
    ROUND_TWO_DPO = 2
    RFT = 3 # reinforcement fine-tuned

    @property
    def display_name(self):
        return self.name.lower().replace("_", "-")


class JudgeTrainingRound(enum.Enum):
    UNTRAINED = -1
    SFT_ONLY = 0

    @property
    def display_name(self):
        return self.name.lower().replace("_", "-")


@dataclasses.dataclass
class DebaterModelConfiguration:
    training_round: DebaterTrainingRound
    is_reasoning: bool
    settings: ModelSettings


@dataclasses.dataclass
class JudgeModelConfiguration:
    training_round: JudgeTrainingRound
    settings: ModelSettings


ALL_VALID_DEBATERS: dict[str, DebaterModelConfiguration] = {
    "llama-3-262k": DebaterModelConfiguration(
        training_round=DebaterTrainingRound.UNTRAINED,
        is_reasoning=False,
        settings=ModelSettings(
            model_type=ModelType.LLAMA3,
            alias="",
            require_quote_validation=True,
            model_file_path="/home/ubuntu/mars-arnesen-gh/garethtan/downloaded-models/gradientai/Llama-3-8B-Instruct-262k"
        )
    ),
    "llama-3-262k-2025-07-31": DebaterModelConfiguration(
        training_round=DebaterTrainingRound.SFT_ONLY,
        is_reasoning=False,
        settings=ModelSettings(
            model_type=ModelType.LLAMA3,
            alias="",
            model_file_path="/home/ubuntu/mars-arnesen-gh/leonidtsyplenkov/models/trained_models/llama-3-mega-merged-no-judge-speeches-31.07"
        )
    ),
    "llama-3-262k-41-sfted-judge": DebaterModelConfiguration(
        training_round=DebaterTrainingRound.ROUND_TWO_DPO,
        is_reasoning=False,
        settings=ModelSettings(
            model_type=ModelType.LLAMA3,
            alias="",
            model_file_path="/home/ubuntu/mars-arnesen-gh/garethtan/models/trained_models/llama-3-DPO-811-FullTrainDebateRoundTwo-full-trained"
        )
    ),
    "llama-3-262k-llama-sft-judge": DebaterModelConfiguration(
        training_round=DebaterTrainingRound.ROUND_ONE_DPO,
        is_reasoning=False,
        settings=ModelSettings(
            model_type=ModelType.LLAMA3,
            alias="",
            model_file_path="/home/ubuntu/mars-arnesen-gh/garethtan/models/trained_models/llama-trained-for-llama-judge-finetuned-round-one"
        )
    ),
    "llama-3-262k-4-turbo-judge": DebaterModelConfiguration(
        training_round=DebaterTrainingRound.ROUND_ONE_DPO,
        is_reasoning=False,
        settings=ModelSettings(
            model_type=ModelType.LLAMA3,
            alias="",
            model_file_path="/home/ubuntu/mars-arnesen-gh/garethtan/models/trained_models/llama-trained-for-gpt-4-turbo-round-one"
        )
    ),
    "o4-mini-rft-2025-09-15": DebaterModelConfiguration(
        training_round=DebaterTrainingRound.RFT,
        is_reasoning=True,
        settings=ModelSettings(
            model_type=ModelType.OPENAI,
            alias="",
            model_file_path="ft:o4-mini-2025-04-16:modulo-research-ltd:michael-and-khan-data-debater-15-09:CGO8p7XH",
            require_quote_validation=True,
        )
    ),
}
ALL_VALID_JUDGES = {
    "gpt-4-turbo-2024-04-09": JudgeModelConfiguration(
        training_round=JudgeTrainingRound.UNTRAINED,
        settings=ModelSettings(
            model_type=ModelType.OPENAI,
            alias="",
            model_file_path="gpt-4-turbo-2024-04-09"
        )
    ),
    "gpt-41-2025-07-31": JudgeModelConfiguration(
        training_round=JudgeTrainingRound.SFT_ONLY,
        settings=ModelSettings(
            model_type=ModelType.OPENAI,
            alias="",
            model_file_path="ft:gpt-4.1-2025-04-14:modulo-research-ltd:michael-and-khan-data-judge-31-07:BzYGc8SU"
        )
    ),
    "gpt-41-2025-04-14": JudgeModelConfiguration(
        training_round=JudgeTrainingRound.UNTRAINED,
        settings=ModelSettings(
            model_type=ModelType.OPENAI,
            alias="",
            model_file_path="gpt-4.1-2025-04-14"
        )
    ),
    "llama-3-262k": JudgeModelConfiguration(
        training_round=JudgeTrainingRound.UNTRAINED,
        settings=ModelSettings(
            model_type=ModelType.LLAMA3,
            alias="",
            model_file_path="/home/ubuntu/mars-arnesen-gh/garethtan/downloaded-models/gradientai/Llama-3-8B-Instruct-262k"
        )
    ),
    "llama-3-262k-2025-10-08": JudgeModelConfiguration(
        training_round=JudgeTrainingRound.SFT_ONLY,
        settings=ModelSettings(
            model_type=ModelType.LLAMA3,
            alias="",
            model_file_path="/home/ubuntu/mars-arnesen-gh/leonidtsyplenkov/models/trained_models/llama-3-mega-merged-judge-08-10"
        )
    )
}
