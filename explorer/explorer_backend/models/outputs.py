from __future__ import annotations

from datetime import datetime
from typing import Dict, List

from pydantic import BaseModel, ConfigDict

from explorer.cli.explorer_models import ConfigTypeLiteral


class OutputSummaryResponse(BaseModel):
    """Summary information about a single output configuration."""

    configuration: str
    config_type: ConfigTypeLiteral
    task_label: str
    debater_key: str
    debater_training: str
    judge_key: str
    judge_training: str
    transcript_count: int
    transcripts_by_day: Dict[str, int]
    latest_transcript: datetime | None
    directory_size_bytes: int
    transcripts_directory: str

    model_config = ConfigDict(from_attributes=True)


class OutputGroupResponse(BaseModel):
    """Grouped outputs for a chosen grouping strategy."""

    label: str
    entries: List[OutputSummaryResponse]


class OutputsListResponse(BaseModel):
    """Response payload for listing available outputs."""

    outputs_directory: str
    group_mode: str
    entries: List[OutputSummaryResponse]
    groups: List[OutputGroupResponse]


class TranscriptFileResponse(BaseModel):
    """Metadata for a transcript file."""

    name: str
    relative_path: str
    size_bytes: int
    modified_at: datetime | None


class OutputDetailResponse(BaseModel):
    """Detailed view of a specific output configuration and its transcripts."""

    configuration: str
    config_type: ConfigTypeLiteral
    task_label: str
    debater_key: str
    debater_training: str
    judge_key: str
    judge_training: str
    transcript_count: int
    transcripts_by_day: Dict[str, int]
    latest_transcript: datetime | None
    directory_size_bytes: int
    transcripts_directory: str
    page: int
    page_size: int
    total_transcripts: int
    total_pages: int
    transcripts: List[TranscriptFileResponse]


class DebateStatsSummary(BaseModel):
    """Aggregated statistics for debate outcomes."""

    total_debates: int
    debater_a_wins: int
    debater_b_wins: int
    judge_correct: int
    first_debater_correct: int
    debater_a_win_rate: float
    debater_b_win_rate: float
    judge_accuracy: float
    first_debater_accuracy: float


class DailyDebateStatsResponse(DebateStatsSummary):
    """Debate statistics for a specific day."""

    day: str


class OutputStatsResponse(BaseModel):
    """Statistics summary for a configuration's transcripts."""

    configuration: str
    transcripts_directory: str
    json_file_count: int
    overall_stats: DebateStatsSummary
    per_day: List[DailyDebateStatsResponse]
    errors: List[str]


__all__ = [
    "DailyDebateStatsResponse",
    "DebateStatsSummary",
    "OutputDetailResponse",
    "OutputGroupResponse",
    "OutputStatsResponse",
    "OutputSummaryResponse",
    "OutputsListResponse",
    "TranscriptFileResponse",
]
