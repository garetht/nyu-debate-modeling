from enum import Enum, auto
from typing import Optional

from debate import AgentConfig, MultiRoundBranchingSetting, SpeechFormatStructure
from data import DatasetConfig
from prompts import PromptLoadingConfig
from pydantic import BaseModel, field_validator, model_validator


class AgentsConfig(BaseModel):
    debaters: list[AgentConfig]
    judge: AgentConfig


class PreviousRunConfig(BaseModel):
    file_path: str | list[str]
    replicate_topics: bool = False
    merge_results: bool = False

    @field_validator("file_path", mode="before")
    @classmethod
    def validate_file_path(cls, file_path: str | list[str]):
        if isinstance(file_path, str):
            return [file_path]
        return file_path


class TournamentType(Enum):
    ROUND_ROBIN = auto()
    SELF_PLAY_ONLY = auto()
    CUSTOM = auto()
    CAPPED_ROUND_ROBIN = auto()
    REPLICATION = auto()


class TournamentConfig(BaseModel):
    tournament_type: TournamentType = TournamentType.ROUND_ROBIN
    custom_matchups: Optional[list[tuple[str, str]]] = None
    replication_file_paths: list[str] = []

    @field_validator("tournament_type", mode="before")
    @classmethod
    def validate_tournament_type(cls, tournament_type: str | TournamentType):
        if isinstance(tournament_type, str):
            tournament_type = TournamentType[tournament_type.upper()]
        return tournament_type

    @model_validator(mode="after")
    @classmethod
    def verify_custom_settings(cls, config):
        if config.custom_matchups and config.tournament_type not in [
            TournamentType.CUSTOM,
            TournamentType.CAPPED_ROUND_ROBIN,
        ]:
            raise ValueError(
                "One cannot set custom matchups if one does not select the custom or capped round robin tournament type"
            )
        elif not config.custom_matchups and config.tournament_type == TournamentType.CUSTOM:
            raise ValueError("One cannot set the custom tournament type without setting custom matchups")
        elif config.replication_file_paths and config.tournament_type != TournamentType.REPLICATION:
            raise ValueError("One cannot use a replication_file_path without using the replication tournament type")
        elif config.tournament_type == TournamentType.REPLICATION and not config.replication_file_paths:
            raise ValueError("One cannot set the replication tournament type without setting replication file paths")
        return config


class ExperimentConfig(BaseModel):
    batch_size: int
    num_speeches: int
    flip: bool = False
    alternate: bool = False
    prompt_config: PromptLoadingConfig = PromptLoadingConfig()
    agents: AgentsConfig
    dataset: DatasetConfig
    annotations_classifier_file_path: Optional[str] = None
    enable_self_debate: bool = False
    previous_run: Optional[PreviousRunConfig] = None
    tournament: Optional[TournamentConfig] = TournamentConfig()
    speech_structure: SpeechFormatStructure = SpeechFormatStructure.DEFAULT_DEBATE
    multi_round_branching: MultiRoundBranchingSetting = MultiRoundBranchingSetting.NONE
    convert_to_double_consultancy: bool = False

    @field_validator("speech_structure", mode="before")
    @classmethod
    def validate_speech_structure(cls, speech_structure: str | SpeechFormatStructure):
        if isinstance(speech_structure, str):
            return SpeechFormatStructure[speech_structure.upper()]
        return speech_structure

    @field_validator("multi_round_branching", mode="before")
    @classmethod
    def validate_multi_round_branching(cls, multi_round_branching: str | MultiRoundBranchingSetting):
        if isinstance(multi_round_branching, str):
            return MultiRoundBranchingSetting[multi_round_branching.upper()]
        return multi_round_branching

    @model_validator(mode="after")
    def check_fields(cls, config):
        if config.flip and config.alternate:
            raise ValueError("flip and alternate cannot both be True at the same time")
        if config.convert_to_double_consultancy and config.speech_structure.num_participants == 1:
            raise ValueError("if convert_to_double_consultancy is used, then a debate format should be used")
        return config
