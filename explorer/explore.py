#!/usr/bin/env python3
"""Textual interface for browsing valid configuration outputs."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Sequence

from rich.text import Text
from textual.app import App, ComposeResult
from textual.widgets import DataTable, Footer, Header, Static, Tree

from run_orchestrator.evals_generator.configuration_name import ConfigurationName
from run_orchestrator.evals_generator.model_definitions import (
    ALL_VALID_DEBATERS,
    ALL_VALID_JUDGES,
)


@dataclass(frozen=True)
class OutputEntry:
    """Structured output metadata ready for presentation."""

    configuration: str
    task_label: str
    debater_key: str
    debater_training: str
    judge_key: str
    judge_training: str


class GroupMode(Enum):
    """Possible presentation modes for grouped output listings."""

    ALL = ("all", "All Configurations")
    DEBATER_MODEL = ("debater_model", "Grouped by Debater Model")
    DEBATER_TRAINING = ("debater_training", "Grouped by Debater Training Round")
    JUDGE_MODEL = ("judge_model", "Grouped by Judge Model")
    JUDGE_TRAINING = ("judge_training", "Grouped by Judge Training Round")

    @property
    def key(self) -> str:
        return self.value[0]

    @property
    def label(self) -> str:
        return self.value[1]

    @classmethod
    def from_key(cls, key: str) -> Optional["GroupMode"]:
        for mode in cls:
            if mode.key == key:
                return mode
        return None


class OutputsExplorerApp(App):
    """A Textual app that lists output directories with valid configuration names."""

    GROUP_SEQUENCE: Sequence[GroupMode] = (
        GroupMode.ALL,
        GroupMode.DEBATER_MODEL,
        GroupMode.DEBATER_TRAINING,
        GroupMode.JUDGE_MODEL,
        GroupMode.JUDGE_TRAINING,
    )

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

    #view-label {
        padding: 0 2;
        width: 100%;
        color: #8a8a8a;
        text-style: italic;
    }

    #output-table, #output-tree {
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
        ("g", "cycle_group", "Next View"),
        ("1", "set_group('all')", "All"),
        ("2", "set_group('debater_model')", "Debater"),
        ("3", "set_group('debater_training')", "Debater Training"),
        ("4", "set_group('judge_model')", "Judge"),
        ("5", "set_group('judge_training')", "Judge Training"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.group_mode: GroupMode = GroupMode.ALL

    def compose(self) -> ComposeResult:
        """Compose the widgets for the application."""
        yield Header(show_clock=True)
        yield Static("Valid outputs", id="title")
        yield Static("", id="view-label")
        yield DataTable(id="output-table")
        yield Tree("Configurations", id="output-tree")
        yield Static("", id="empty-message")
        yield Footer()

    def on_mount(self) -> None:
        """Initialise the widgets and populate them with entries."""
        table = self.query_one("#output-table", DataTable)
        table.cursor_type = "row"
        table.zebra_stripes = True
        table.add_columns("Debater", "Judge", "Task Type", "Configuration")

        tree = self.query_one("#output-tree", Tree)
        tree.root.set_label(GroupMode.ALL.label)
        tree.root.expand()

        self._refresh_contents()

    def action_refresh(self) -> None:
        """Reload the output list."""
        self._refresh_contents()

    def action_cycle_group(self) -> None:
        """Advance to the next grouping view."""
        sequence = list(self.GROUP_SEQUENCE)
        current_index = sequence.index(self.group_mode)
        next_mode = sequence[(current_index + 1) % len(sequence)]
        self._set_group_mode(next_mode)

    def action_set_group(self, mode_key: str) -> None:
        """Jump directly to a specific grouping view."""
        mode = GroupMode.from_key(mode_key)
        if mode is None:
            return
        self._set_group_mode(mode)

    def _set_group_mode(self, mode: GroupMode) -> None:
        """Apply the requested grouping mode if it changed."""
        if self.group_mode == mode:
            return
        self.group_mode = mode
        self._refresh_contents()

    def _refresh_contents(self) -> None:
        """Populate the table, tree, and empty-state indicator."""
        entries, message = self._collect_entries()
        self._update_table(entries)
        self._update_tree(entries)
        self._apply_view_state(entries, message)

    def _collect_entries(self) -> tuple[Sequence[OutputEntry], Optional[str]]:
        """Gather the output metadata and any empty-state message."""
        outputs_dir = Path(__file__).resolve().parent.parent / "outputs"
        if not outputs_dir.exists():
            return [], f"No outputs directory found at {outputs_dir}"

        entries: list[OutputEntry] = []
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

            entries.append(
                OutputEntry(
                    configuration=item.name,
                    task_label=self._format_task(config.task_type_name),
                    debater_key=config.debater_key,
                    debater_training=debater_cfg.training_round.display_name,
                    judge_key=config.judge_key,
                    judge_training=judge_cfg.training_round.display_name,
                )
            )

        if not entries:
            return [], "No outputs matched ConfigurationName."

        return entries, None

    def _update_table(self, entries: Sequence[OutputEntry]) -> None:
        """Refresh the tabular view."""
        table = self.query_one("#output-table", DataTable)
        table.clear(columns=False)
        for entry in sorted(entries, key=lambda e: e.configuration.lower()):
            table.add_row(
                self._format_debater(entry.debater_key, entry.debater_training),
                self._format_judge(entry.judge_key, entry.judge_training),
                entry.task_label,
                entry.configuration,
            )

    def _update_tree(self, entries: Sequence[OutputEntry]) -> None:
        """Refresh the grouped tree view."""
        tree = self.query_one("#output-tree", Tree)
        tree.root.set_label(self.group_mode.label)
        tree.root.remove_children()

        if not entries or self.group_mode == GroupMode.ALL:
            return

        for label, group_entries in self._group_entries(entries):
            group_node = tree.root.add(label)
            for entry in group_entries:
                group_node.add(self._format_leaf(entry))
            group_node.expand()
        tree.root.expand()

    def _group_entries(self, entries: Sequence[OutputEntry]) -> Sequence[tuple[str, Sequence[OutputEntry]]]:
        """Group entries according to the active grouping mode."""
        groups: dict[str, list[OutputEntry]] = defaultdict(list)

        for entry in entries:
            if self.group_mode == GroupMode.DEBATER_MODEL:
                key = entry.debater_key
            elif self.group_mode == GroupMode.DEBATER_TRAINING:
                key = entry.debater_training
            elif self.group_mode == GroupMode.JUDGE_MODEL:
                key = entry.judge_key
            elif self.group_mode == GroupMode.JUDGE_TRAINING:
                key = entry.judge_training
            else:
                continue
            groups[key].append(entry)

        grouped: list[tuple[str, Sequence[OutputEntry]]] = []
        for key, grouped_entries in groups.items():
            sorted_entries = sorted(grouped_entries, key=lambda e: e.configuration.lower())
            if self.group_mode == GroupMode.DEBATER_MODEL:
                label = self._format_debater(key, sorted_entries[0].debater_training)
            elif self.group_mode == GroupMode.DEBATER_TRAINING:
                label = f"Debater Training – {self._format_training(key)}"
            elif self.group_mode == GroupMode.JUDGE_MODEL:
                label = self._format_judge(key, sorted_entries[0].judge_training)
            else:
                label = f"Judge Training – {self._format_training(key)}"
            grouped.append((label, sorted_entries))

        return sorted(grouped, key=lambda item: item[0].lower())

    def _apply_view_state(
        self,
        entries: Sequence[OutputEntry],
        message: Optional[str],
    ) -> None:
        """Toggle widget visibility and status messaging."""
        table = self.query_one("#output-table", DataTable)
        tree = self.query_one("#output-tree", Tree)
        empty_message = self.query_one("#empty-message", Static)
        view_label = self.query_one("#view-label", Static)

        view_label.update(f"View: {self.group_mode.label}")

        if not entries:
            empty_message.update(message or "No outputs matched ConfigurationName.")
            table.display = False
            tree.display = False
            return

        empty_message.update("")

        if self.group_mode == GroupMode.ALL:
            table.display = True
            tree.display = False
        else:
            table.display = False
            tree.display = True

    def _format_leaf(self, entry: OutputEntry) -> Text:
        """Construct a styled leaf label for the grouped tree view."""
        leaf = Text(entry.configuration, style="bold")
        leaf.append(" — ", style="dim")
        leaf.append(f"Task: {entry.task_label}", style="yellow")
        leaf.append(" | ", style="dim")
        leaf.append(
            f"Debater: {self._format_debater(entry.debater_key, entry.debater_training)}",
            style="bright_magenta",
        )
        leaf.append(" | ", style="dim")
        leaf.append(
            f"Judge: {self._format_judge(entry.judge_key, entry.judge_training)}",
            style="bright_cyan",
        )
        return leaf

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
