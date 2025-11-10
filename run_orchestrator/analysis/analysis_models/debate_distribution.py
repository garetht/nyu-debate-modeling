from pathlib import Path

from pydantic import BaseModel, ConfigDict

from run_orchestrator.analysis.analysis_models.analysis_result import AnalysisResult


class DebateDistributionArgs(BaseModel):
    model_config = ConfigDict(frozen=True)

    folder_path: Path


class DebateDistributionAnalysis(AnalysisResult):
    identifier_counts: dict[str, int]
    title_counts: dict[str, int]
    transcript_count: int
    
