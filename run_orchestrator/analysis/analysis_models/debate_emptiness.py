from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from run_orchestrator.analysis.analysis_models.analysis_result import AnalysisResult


@dataclass(frozen=True)
class DebateEmptinessAnalysis(AnalysisResult):
    empty_speech_counts: Counter[str]
    debater_a_empty_files: tuple[Path, ...]
    debater_b_empty_files: tuple[Path, ...]
    unique_empty_files: tuple[Path, ...]
    total_debates: int

    @property
    def total_unique_empty_debates(self) -> int:
        return len(self.unique_empty_files)
