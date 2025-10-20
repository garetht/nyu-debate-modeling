#!/usr/bin/env python3
"""Data models and shared enumerations for the explorer application."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Final, Literal, Optional, Sequence

ConfigTypeLiteral = Literal["eval", "data-generation"]
EVAL_CONFIG_TYPE: Final[ConfigTypeLiteral] = "eval"
DATA_GENERATION_CONFIG_TYPE: Final[ConfigTypeLiteral] = "data-generation"


class GroupMode(Enum):
    """Possible presentation modes for grouped output listings."""

    ALL = ("all", "All Configurations")
    DEBATER_MODEL = ("debater_model", "Grouped by Debater Model")
    DEBATER_TRAINING = ("debater_training", "Grouped by Debater Training Round")
    JUDGE_MODEL = ("judge_model", "Grouped by Judge Model")
    JUDGE_TRAINING = ("judge_training", "Grouped by Judge Training Round")

    @property
    def key(self) -> str:
        return self.value[0]

    @property
    def label(self) -> str:
        return self.value[1]

    @classmethod
    def from_key(cls, key: str) -> Optional["GroupMode"]:
        for mode in cls:
            if mode.key == key:
                return mode
        return None


GROUP_MODE_SEQUENCE: Final[Sequence[GroupMode]] = (
    GroupMode.ALL,
    GroupMode.DEBATER_MODEL,
    GroupMode.DEBATER_TRAINING,
    GroupMode.JUDGE_MODEL,
    GroupMode.JUDGE_TRAINING,
)


@dataclass(frozen=True)
class OutputEntry:
    """Structured output metadata ready for presentation."""

    configuration: str
    config_type: ConfigTypeLiteral
    transcripts_directory: Path
    task_label: str
    debater_key: str
    debater_training: str
    judge_key: str
    judge_training: str
    transcript_count: int
    transcripts_by_day: dict[str, int]
    latest_transcript: Optional[datetime] = None


@dataclass(frozen=True)
class IdentifierStatsEntry:
    """Tree node data for debate identifier statistics."""

    transcripts_directory: Path
