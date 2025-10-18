#!/usr/bin/env python3
"""Textual interface for browsing valid configuration outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

from textual.app import App, ComposeResult
from textual.widgets import DataTable, Footer, Header, Static

from run_orchestrator.evals_generator.configuration_name import ConfigurationName
from run_orchestrator.evals_generator.model_definitions import ALL_VALID_DEBATERS, ALL_VALID_JUDGES


class OutputsExplorerApp(App):
    """A Textual app that lists output directories with valid configuration names."""

    CSS = """
    Screen {
        layout: vertical;
        align: center top;
    }

    #title {
        padding: 1 2;
        width: 100%;
        border: solid green;
        content-align: center middle;
    }

    #output-table {
        padding: 1 2;
        width: 100%;
    }

    #empty-message {
        padding: 2;
        color: gray;
        text-align: center;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "refresh", "Refresh"),
    ]

    def compose(self) -> ComposeResult:
        """Compose the widgets for the application."""
        yield Header(show_clock=True)
        yield DataTable(id="output-table")
        yield Static("", id="empty-message")
        yield Footer()

    def on_mount(self) -> None:
        """Initialise the data table and populate it with entries."""
        table = self.query_one("#output-table", DataTable)
        table.cursor_type = "row"
        table.zebra_stripes = True
        table.add_columns("Debater", "Judge", "Task Type", "Name")
        self._refresh_contents()

    def action_refresh(self) -> None:
        """Reload the output list."""
        self._refresh_contents()

    def _refresh_contents(self) -> None:
        """Populate the table and empty-state message."""
        table = self.query_one("#output-table", DataTable)
        empty_message = self.query_one("#empty-message", Static)
        table.clear(columns=False)

        rows, message = self._collect_rows()

        if not rows:
            empty_message.update(message or "No outputs matched ConfigurationName.")
            table.display = False
            return

        empty_message.update("")
        table.display = True

        for row in rows:
            table.add_row(*row)

    def _collect_rows(self) -> Tuple[Sequence[tuple[str, str, str, str]], Optional[str]]:
        """Gather the table rows and an optional empty-state message."""
        outputs_dir = Path(__file__).resolve().parent.parent / "outputs"
        if not outputs_dir.exists():
            return [], f"No outputs directory found at {outputs_dir}"

        rows: list[tuple[str, str, str, str]] = []
        for item in sorted(outputs_dir.iterdir(), key=lambda path: path.name.lower()):
            if not item.is_dir():
                continue
            try:
                config = ConfigurationName.deserialize(item.name)
            except Exception:
                continue

            debater_cfg = ALL_VALID_DEBATERS.get(config.debater_key)
            judge_cfg = ALL_VALID_JUDGES.get(config.judge_key)
            if not debater_cfg or not judge_cfg:
                continue

            rows.append(
                (
                    self._format_debater(config.debater_key, debater_cfg.training_round.display_name),
                    self._format_judge(config.judge_key, judge_cfg.training_round.display_name),
                    self._format_task(config.task_type_name),
                    item.name,
                )
            )

        message = None
        if not rows:
            message = "No outputs matched ConfigurationName."

        return rows, message

    @staticmethod
    def _format_task(task_type_name: str) -> str:
        """Return a polished task type label."""
        return task_type_name.replace("_", " ").replace("-", " ").title()

    @staticmethod
    def _format_debater(key: str, training_round: str) -> str:
        """Return a formatted debater description."""
        return f"{key} ({OutputsExplorerApp._format_training(training_round)})"

    @staticmethod
    def _format_judge(key: str, training_round: str) -> str:
        """Return a formatted judge description."""
        return f"{key} ({OutputsExplorerApp._format_training(training_round)})"

    @staticmethod
    def _format_training(training_round: str) -> str:
        """Return a friendly training round label."""
        return training_round.replace("-", " ").replace("_", " ").title()


if __name__ == "__main__":
    OutputsExplorerApp().run()
