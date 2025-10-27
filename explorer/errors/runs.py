from __future__ import annotations

__all__ = ["RunNotFoundError"]


class RunNotFoundError(Exception):
    """Raised when a requested run does not exist in the task database."""

    def __init__(self, run_id: int) -> None:
        self.run_id: int = run_id
        super().__init__(f"Run with id {run_id} not found.")
