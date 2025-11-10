from dataclasses import dataclass
from pathlib import Path

from run_orchestrator.analysis.analysis_models.analysis_result import AnalysisResult


@dataclass(frozen=True)
class DebateDistributionArgs:
    folder_path: Path


@dataclass(frozen=True)
class DebateIdentifierCount:
    identifier: str
    title: str
    topic: str
    count: int


@dataclass(frozen=True)
class TitleCount:
    title: str
    count: int


@dataclass(frozen=True)
class DebateDistributionAnalysis(AnalysisResult):
    identifier_counts: tuple[DebateIdentifierCount, ...]
    title_counts: tuple[TitleCount, ...]
    transcript_count: int
