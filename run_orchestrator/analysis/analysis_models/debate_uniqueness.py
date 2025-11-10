from pathlib import Path

from run_orchestrator.analysis.analysis_models.analysis_result import AnalysisResult


class DebateUniquenessAnalysis(AnalysisResult):
    unique_identifiers: list[str]
    duplicate_file_paths: list[Path]
    total_transcripts: int
