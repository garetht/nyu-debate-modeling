import math
from typing import Dict

from pydantic import BaseModel, ConfigDict, Field, computed_field


class DebateStats(BaseModel):
    """Container for debate statistics."""

    model_config = ConfigDict(extra="forbid")

    total_debates: int = 0
    debater_a_wins: int = 0
    debater_b_wins: int = 0
    judge_correct: int = 0
    first_debater_correct: int = 0
    debater_a_probs: list[float] = Field(default_factory=list)
    debater_b_probs: list[float] = Field(default_factory=list)

    def add_debate(
        self,
        debater_a_win: bool,
        debater_b_win: bool,
        judge_correct: bool,
        first_debater_correct: bool,
        debater_a_prob: float | None = None,
        debater_b_prob: float | None = None,
    ) -> None:
        """Add results from a single debate."""
        self.total_debates += 1
        if debater_a_win:
            self.debater_a_wins += 1
        elif debater_b_win:
            self.debater_b_wins += 1

        if judge_correct:
            self.judge_correct += 1
        if first_debater_correct:
            self.first_debater_correct += 1

        # Store probabilistic decisions
        if debater_a_prob is not None:
            self.debater_a_probs.append(debater_a_prob)
        if debater_b_prob is not None:
            self.debater_b_probs.append(debater_b_prob)

    def get_percentages(self) -> Dict[str, float]:
        """Calculate percentage statistics."""
        if self.total_debates == 0:
            return {
                "debater_a_win_rate": 0.0,
                "debater_b_win_rate": 0.0,
                "judge_accuracy": 0.0,
                "first_debater_accuracy": 0.0
            }

        return {
            "debater_a_win_rate": (self.debater_a_wins / self.total_debates) * 100,
            "debater_b_win_rate": (self.debater_b_wins / self.total_debates) * 100,
            "judge_accuracy": (self.judge_correct / self.total_debates) * 100,
            "first_debater_accuracy": (self.first_debater_correct / self.total_debates) * 100
        }

    @computed_field
    def judge_accuracy(self) -> float:
        """Return the judge accuracy as the ratio of correct judgments."""
        if self.total_debates == 0:
            return 0.0
        return self.judge_correct / self.total_debates

    @computed_field
    def judge_standard_error(self) -> float:
        """Return the standard error of the judge accuracy."""
        if self.total_debates == 0:
            return 0.0

        success_rate: float = self.judge_correct / self.total_debates
        variance: float = (success_rate * (1.0 - success_rate)) / self.total_debates
        return math.sqrt(variance)

    @computed_field
    def debater_a_win_skew(self) -> float:
        """Return the skew of the first debater win rate."""
        if self.total_debates == 0:
            return 0.0

        return self.debater_a_wins / self.total_debates

    @computed_field
    def debater_a_divergence_from_uniform(self) -> int:
        """Return the divergence of the first debater win rate from uniform."""
        if self.total_debates == 0:
            return 0

        return self.debater_a_wins - (self.total_debates // 2)

    @computed_field
    def debater_b_divergence_from_uniform(self) -> int:
        """Return the skew of the second debater win rate."""
        if self.total_debates == 0:
            return 0

        return self.debater_b_wins - (self.total_debates // 2)