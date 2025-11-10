from run_orchestrator.analysis.analysis_models.analysis_result import AnalysisResult


class DebateEmptinessAnalysis(AnalysisResult):
    empty_speech_counts: dict[str, int]
    debater_a_empty_files: list[str]
    debater_b_empty_files: list[str]
    unique_empty_files: list[str]
    total_debates: int

    @property
    def total_unique_empty_debates(self) -> int:
        return len(self.unique_empty_files)
