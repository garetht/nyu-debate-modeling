from dataclasses import dataclass
from pathlib import Path

from run_orchestrator.analysis.analysis_models.analysis_result import AnalysisResult


@dataclass(frozen=True)
class DebateUniquenessAnalysis(AnalysisResult):
    unique_identifiers: tuple[str, ...]
    duplicate_file_paths: tuple[Path, ...]
    total_transcripts: int
