#!/usr/bin/env python3
"""Textual interface for browsing valid configuration outputs."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from threading import Thread
from typing import Final, Optional, Sequence
from datetime import datetime

from rich.text import Text
from textual.app import App, ComposeResult
from textual.events import Key
from textual.widgets import Footer, Header, Static, Tree

from experiment_models.extant_debates_extractor import ExtantDebateIdentifiersExtractor
from run_orchestrator.debate_stats_analyzer import (
    DirectoryAnalysisResult,
    collect_directory_analysis,
)
from run_orchestrator.evals_generator.configuration_name import ConfigurationName
from run_orchestrator.evals_generator.config_spec import ConfigurationType
from run_orchestrator.evals_generator.model_definitions import (
    ALL_VALID_DEBATERS,
    ALL_VALID_JUDGES,
)


@dataclass(frozen=True)
class OutputEntry:
    """Structured output metadata ready for presentation."""

    configuration: str
    config_type: ConfigurationType
    transcripts_directory: Path
    task_label: str
    debater_key: str
    debater_training: str
    judge_key: str
    judge_training: str
    transcript_count: int
    transcripts_by_day: dict[str, int]
    latest_transcript: Optional[datetime]


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

    #view-label {
        padding: 0 2;
        width: 100%;
        color: #8a8a8a;
        text-style: italic;
    }

    #output-tree {
        padding: 1 2;
        width: 100%;
    }

    #empty-message {
        padding: 2;
        color: gray;
        text-align: center;
    }

    #stats-output {
        padding: 1 2;
        width: 100%;
        border: solid #444444;
    }
    """

    BINDINGS = [
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
        self.repository_root = Path(__file__).resolve().parent.parent
        self._stats_thread: Optional[Thread] = None

    def compose(self) -> ComposeResult:
        """Compose the widgets for the application."""
        yield Header(show_clock=True)
        yield Static("", id="view-label")
        yield Tree("Configurations", id="output-tree")
        yield Static("Expand a configuration to run debate statistics.", id="stats-output")
        yield Static("", id="empty-message")
        yield Footer()

    def on_mount(self) -> None:
        """Initialise the widgets and populate them with entries."""
        tree = self.query_one("#output-tree", Tree)
        tree.root.set_label(self._tree_root_label())
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

    def on_key(self, event: Key) -> None:
        """Allow exiting with Ctrl+C or Ctrl+D."""
        if event.key in ("ctrl+c", "ctrl+d"):
            event.stop()
            self.exit()
            return

    def _set_group_mode(self, mode: GroupMode) -> None:
        """Apply the requested grouping mode if it changed."""
        if self.group_mode == mode:
            return
        self.group_mode = mode
        self._refresh_contents()

    def _refresh_contents(self) -> None:
        """Populate the table, tree, and empty-state indicator."""
        entries, message = self._collect_entries()
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

            transcripts_dir = item / "outputs" / "transcripts"
            count, histogram, latest = self._collect_transcript_stats(transcripts_dir)

            entries.append(
                OutputEntry(
                    configuration=item.name,
                    config_type=config.config_type,
                    transcripts_directory=transcripts_dir,
                    task_label=self._format_task(config.task_type_name),
                    debater_key=config.debater_key,
                    debater_training=debater_cfg.training_round.display_name,
                    judge_key=config.judge_key,
                    judge_training=judge_cfg.training_round.display_name,
                    transcript_count=count,
                    transcripts_by_day=histogram,
                    latest_transcript=latest,
                )
            )

        if not entries:
            return [], "No outputs matched ConfigurationName."

        return entries, None

    def _update_tree(self, entries: Sequence[OutputEntry]) -> None:
        """Refresh the grouped tree view."""
        tree = self.query_one("#output-tree", Tree)
        tree.root.set_label(self._tree_root_label())
        tree.root.remove_children()

        if not entries:
            return

        if self.group_mode == GroupMode.ALL:
            sorted_entries = sorted(
                entries,
                key=lambda e: e.latest_transcript or datetime.min,
                reverse=True,
            )
            for entry in sorted_entries:
                self._add_leaf(tree.root, entry)
            tree.root.expand()
            return

        for label, group_entries in self._group_entries(entries):
            group_node = tree.root.add(label)
            for entry in group_entries:
                self._add_leaf(group_node, entry)
            group_node.expand()
        tree.root.expand()

    def _tree_root_label(self) -> str:
        """Return the label for the tree root based on view."""
        if self.group_mode == GroupMode.ALL:
            return "Recent Transcripts"
        return self.group_mode.label

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
        tree = self.query_one("#output-tree", Tree)
        empty_message = self.query_one("#empty-message", Static)
        view_label = self.query_one("#view-label", Static)

        view_label.update(f"View: {self.group_mode.label}")

        if not entries:
            empty_message.update(message or "No outputs matched ConfigurationName.")
            tree.display = False
            return

        empty_message.update("")
        tree.display = True

    def _add_leaf(self, parent, entry: OutputEntry) -> None:
        """Add a summary node with expandable configuration details."""
        summary = Text()
        summary.append(f"Task: {entry.task_label}", style="yellow")
        summary.append(" | ", style="dim")
        summary.append(f"Type: {entry.config_type.name}", style="white")
        summary.append(" | ", style="dim")
        summary.append(
            f"Debater: {self._format_debater(entry.debater_key, entry.debater_training)}",
            style="bright_magenta",
        )
        summary.append(" | ", style="dim")
        summary.append(
            f"Judge: {self._format_judge(entry.judge_key, entry.judge_training)}",
            style="bright_cyan",
        )
        summary.append(" | ", style="dim")
        summary.append(
            f"Transcripts: {entry.transcript_count} {self._pluralize('file', entry.transcript_count)}",
            style="bright_green",
        )
        if entry.latest_transcript:
            summary.append(" | ", style="dim")
            summary.append(
                f"Latest: {self._format_latest(entry.latest_transcript)}",
                style="bright_yellow",
            )

        leaf_node = parent.add(summary)
        leaf_node.data = entry
        leaf_node.add(Text(entry.configuration, style="bold cyan"))

        histogram = self._render_histogram(entry.transcripts_by_day)
        hist_label = leaf_node.add(Text("Transcripts per day", style="italic"))
        lines = histogram.splitlines() or ["No transcript files found."]

        for line in lines:
            hist_label.add_leaf(Text(line, style="dim", no_wrap=True))

    @staticmethod
    def _pluralize(noun: str, count: int) -> str:
        """Return a pluralized label for counts."""
        suffix = "" if count == 1 else "s"
        return f"{noun}{suffix}"

    @staticmethod
    def _collect_transcript_stats(directory: Path) -> tuple[int, dict[str, int], Optional[datetime]]:
        """Return total transcripts, a per-day histogram, and latest timestamp."""
        if not directory.exists():
            return 0, {}, None
        counts: dict[str, int] = defaultdict(int)
        latest: Optional[datetime] = None
        try:
            for child in directory.iterdir():
                if not child.is_file():
                    continue
                day = OutputsExplorerApp._parse_transcript_day(child.name)
                if day:
                    counts[day] += 1
                timestamp = OutputsExplorerApp._parse_transcript_timestamp(child.name)
                if timestamp and (latest is None or timestamp > latest):
                    latest = timestamp
        except OSError:
            return 0, {}, None

        ordered = dict(sorted(counts.items()))
        total = sum(ordered.values())
        return total, ordered, latest

    @staticmethod
    def _parse_transcript_day(filename: str) -> Optional[str]:
        """Extract YYYY-MM-DD from transcript filename."""
        if not filename:
            return None
        day_part = filename.split("_", 1)[0]
        try:
            datetime.strptime(day_part, "%Y-%m-%d")
        except ValueError:
            return None
        return day_part

    @staticmethod
    def _parse_transcript_timestamp(filename: str) -> Optional[datetime]:
        """Extract the precise timestamp from transcript filename."""
        if not filename:
            return None

        parts = filename.split("_")
        if len(parts) < 2:
            return None

        timestamp_str = "_".join(parts[:2])
        for fmt in ("%Y-%m-%d_%H:%M:%S.%f", "%Y-%m-%d_%H:%M:%S"):
            try:
                return datetime.strptime(timestamp_str, fmt)
            except ValueError:
                continue
        return None

    def on_tree_node_expanded(self, event: Tree.NodeExpanded) -> None:
        """Run the appropriate statistics when a configuration node expands."""
        entry = getattr(event.node, "data", None)
        if isinstance(entry, OutputEntry):
            self._trigger_stats_analysis(entry)

    def _trigger_stats_analysis(self, entry: OutputEntry) -> None:
        """Kick off statistics analysis for a configuration entry."""
        stats_display = self.query_one("#stats-output", Static)
        if self._stats_thread and self._stats_thread.is_alive():
            stats_display.update(Text("Statistics analysis already running...", style="yellow"))
            return

        display_path = self._format_directory_for_display(entry.transcripts_directory)
        if entry.config_type == ConfigurationType.EVAL:
            message = f"Running debate statistics analysis for {display_path}..."
        elif entry.config_type == ConfigurationType.DATA_GENERATION:
            message = f"Collecting debate identifier statistics for {display_path}..."
        else:
            stats_display.update(
                Text("Statistics are unavailable for this configuration type.", style="yellow")
            )
            return

        stats_display.update(Text(message, style="yellow"))
        self._stats_thread = Thread(target=self._run_stats_analysis, args=(entry,), daemon=True)
        self._stats_thread.start()

    def _run_stats_analysis(self, entry: OutputEntry) -> None:
        """Execute the stats analysis in a background thread."""
        try:
            if entry.config_type == ConfigurationType.EVAL:
                result = collect_directory_analysis(entry.transcripts_directory)
                self.call_from_thread(self._display_stats_result, result)
            elif entry.config_type == ConfigurationType.DATA_GENERATION:
                counts, warning = self._collect_identifier_stats(entry.transcripts_directory)
                self.call_from_thread(self._display_identifier_stats, entry, counts, warning)
            else:
                self.call_from_thread(
                    self._display_stats_error,
                    f"Unsupported configuration type '{entry.config_type.value}' for statistics.",
                )
        except (FileNotFoundError, NotADirectoryError) as exc:
            self.call_from_thread(self._display_stats_error, str(exc))
        except Exception as exc:
            self.call_from_thread(self._display_stats_error, f"Unexpected error: {exc}")

    def _display_stats_result(self, result: DirectoryAnalysisResult) -> None:
        """Render the statistics outcome inside the UI."""
        stats_display = self.query_one("#stats-output", Static)
        stats_display.update(self._format_stats_result(result))

    def _display_stats_error(self, message: str) -> None:
        """Display analysis errors."""
        stats_display = self.query_one("#stats-output", Static)
        stats_display.update(Text(f"Error running stats: {message}", style="red"))

    def _collect_identifier_stats(self, directory: Path) -> tuple[dict[str, int], Optional[str]]:
        """Return identifier counts for data-generation configurations."""
        if not directory.exists():
            return {}, f"No transcripts directory found at {directory}"
        extractor = ExtantDebateIdentifiersExtractor(str(directory))
        counts = extractor.process_files()
        return counts, None

    def _display_identifier_stats(
        self,
        entry: OutputEntry,
        counts: dict[str, int],
        warning: Optional[str],
    ) -> None:
        """Display identifier statistics in the UI."""
        stats_display = self.query_one("#stats-output", Static)
        stats_display.update(
            self._format_identifier_stats(entry.transcripts_directory, counts, warning)
        )

    def _format_directory_for_display(self, directory: Path) -> str:
        """Return a path string relative to the repository when possible."""
        try:
            relative_path = directory.relative_to(self.repository_root)
        except ValueError:
            return str(directory)
        return str(relative_path)

    def _format_stats_result(self, result: DirectoryAnalysisResult) -> Text:
        """Create a rich text summary for the stats results."""
        text = Text()
        text.append("Debate Stats\n", style="bold underline")
        text.append(f"Directory: {result.directory}\n", style="dim")
        text.append(f"JSON files located: {len(result.json_files)}\n")

        stats = result.overall_stats
        if not result.json_files:
            text.append("No JSON debate transcripts found in the directory.", style="yellow")
            return text

        if stats.total_debates == 0:
            text.append("No valid debates found to analyze.", style="yellow")
        else:
            percentages = stats.get_percentages()
            text.append(f"Total debates analyzed: {stats.total_debates}\n")
            text.append(
                f"Debater A wins: {stats.debater_a_wins} ({percentages['debater_a_win_rate']:.1f}%)\n"
            )
            text.append(
                f"Debater B wins: {stats.debater_b_wins} ({percentages['debater_b_win_rate']:.1f}%)\n"
            )
            text.append(
                f"Judge accuracy: {stats.judge_correct}/{stats.total_debates} ({percentages['judge_accuracy']:.1f}%)\n"
            )
            text.append(
                f"First debater accuracy: {stats.first_debater_correct}/{stats.total_debates} ({percentages['first_debater_accuracy']:.1f}%)\n"
            )

        if result.errors:
            text.append("\nErrors encountered:\n", style="bold red")
            max_errors = 5
            for error in result.errors[:max_errors]:
                text.append(f"- {error}\n", style="red")
            remaining = len(result.errors) - max_errors
            if remaining > 0:
                text.append(f"... {remaining} more errors\n", style="red")

        return text

    def _format_identifier_stats(
        self,
        directory: Path,
        counts: dict[str, int],
        warning: Optional[str],
    ) -> Text:
        """Create a condensed summary of identifier usage."""
        text = Text()
        text.append("Debate Identifier Stats\n", style="bold underline")
        text.append(f"Directory: {directory}\n", style="dim")

        if warning:
            text.append(warning, style="yellow")
            return text

        if not counts:
            text.append("No debate identifiers found.", style="yellow")
            return text

        total_identifiers = len(set(counts.keys()))
        text.append(f"Total unique debate identifiers: {total_identifiers}\n")

        top_five = sorted(
            counts.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:5]
        bottom_five = sorted(
            counts.items(),
            key=lambda item: item[1],
        )[:5]

        if top_five:
            text.append("\nTop 5 identifiers:\n", style="bold")
            for identifier, count in top_five:
                text.append(f"- {identifier}: {count}\n")

        if bottom_five:
            text.append("\nBottom 5 identifiers:\n", style="bold")
            for identifier, count in bottom_five:
                text.append(f"- {identifier}: {count}\n")

        return text

    def _render_histogram(self, histogram: dict[str, int]) -> str:
        """Render a compact bar chart with vertical labels."""
        if not histogram:
            return "No transcript files found."

        labels = list(histogram.keys())
        counts = [histogram[label] for label in labels]
        max_count = max(counts, default=0)

        if max_count <= 0:
            return "Transcripts counted but totals are zero."

        chart_lines: list[str] = []
        column_width: int = 12
        bar_char = "█" * column_width
        empty_char = " " * column_width
        separator_char = "─" * column_width
        joiner = " "

        # Build vertical bars with minimal width.
        for level in range(max_count, 0, -1):
            line_parts = [
                (bar_char if count >= level else empty_char)
                for count in counts
            ]
            chart_lines.append(joiner.join(line_parts).rstrip())

        # Separator line.
        chart_lines.append(joiner.join(separator_char for _ in labels).rstrip())

        # Vertical day labels.
        max_label_len = max(len(label) for label in labels)
        for index in range(max_label_len):
            line_parts = [
                (label[index] if index < len(label) else " ").center(column_width)
                for label in labels
            ]
            chart_lines.append(joiner.join(line_parts).rstrip())

        # Vertical "days ago" labels.
        today = datetime.now().date()
        ago_labels: list[str] = []
        for label in labels:
            try:
                day = datetime.strptime(label, "%Y-%m-%d").date()
                delta = (today - day).days
                if delta < 0:
                    ago_text = f"{abs(delta)}d future"
                elif delta == 0:
                    ago_text = "today"
                elif delta == 1:
                    ago_text = "1d ago"
                else:
                    ago_text = f"{delta}d ago"
            except ValueError:
                ago_text = ""
            ago_labels.append(f"({ago_text})" if ago_text else "")

        max_ago_len = max((len(text) for text in ago_labels), default=0)
        for index in range(max_ago_len):
            line_parts = [
                (text[index] if index < len(text) else " ").center(column_width)
                for text in ago_labels
            ]
            chart_lines.append(joiner.join(line_parts).rstrip())

        return "\n".join(chart_lines)

    @staticmethod
    def _format_task(task_type_name: str) -> str:
        """Return a polished task type label."""
        return task_type_name.replace("_", " ").replace("-", " ").title()

    @staticmethod
    def _format_latest(timestamp: Optional[datetime]) -> str:
        """Return a readable latest transcript timestamp."""
        if timestamp is None:
            return "—"
        return timestamp.strftime("%Y-%m-%d %H:%M:%S")

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
