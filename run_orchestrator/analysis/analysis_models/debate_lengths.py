from dataclasses import dataclass

from run_orchestrator.analysis.analysis_models.analysis_result import AnalysisResult


@dataclass(frozen=True)
class DebateLengthAnalysis(AnalysisResult):
    debater_a_lengths: tuple[int, ...]
    debater_b_lengths: tuple[int, ...]
    transcript_count: int
