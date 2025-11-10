from run_orchestrator.analysis.analysis_models.analysis_result import AnalysisResult


class DebateLengthAnalysis(AnalysisResult):
    debater_a_lengths: list[int]
    debater_b_lengths: list[int]
    transcript_count: int
