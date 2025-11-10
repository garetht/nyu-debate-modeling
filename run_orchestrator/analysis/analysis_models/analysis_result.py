from pydantic import BaseModel, ConfigDict


class AnalysisResult(BaseModel):
    """Base class for analysis results enforcing immutability."""

    model_config = ConfigDict(frozen=True)
