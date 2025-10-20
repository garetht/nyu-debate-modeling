#!/usr/bin/env python3
"""Textual application for exploring debate outputs."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from threading import Thread
from typing import Any, Optional, Sequence

from rich.text import Text
from textual.app import App, ComposeResult
from textual.events import Key
from textual.widgets import Footer, Header, Static, Tree

from explorer.display import (
    format_debater,
    format_directory_for_display,
    format_judge,
    format_latest,
    format_task,
    format_training,
    pluralize,
    render_histogram,
    render_identifier_summary_lines,
)
from explorer.explorer_models import (
    GROUP_MODE_SEQUENCE,
    GroupMode,
    IdentifierStatsEntry,
    OutputEntry,
)
from experiment_models.extant_debates_extractor import ExtantDebateIdentifiersExtractor
from run_orchestrator.debate_stats_analyzer import (
    DebateStats,
    DirectoryAnalysisResult,
    collect_directory_analysis,
)
from run_orchestrator.evals_generator.config_spec import ConfigurationType
from run_orchestrator.evals_generator.configuration_name import ConfigurationName
from run_orchestrator.evals_generator.model_definitions import (
    ALL_VALID_DEBATERS,
    ALL_VALID_JUDGES,
)


class OutputsExplorerApp(App):
    """A Textual app that lists output directories with valid configuration names."""

    GROUP_SEQUENCE: Sequence[GroupMode] = GROUP_MODE_SEQUENCE

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
        self._identifier_cache: dict[Path, tuple[dict[str, int], Optional[str]]] = {}
        self._identifier_threads: dict[Path, Thread] = {}

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
        """Populate the tree and empty-state indicator."""
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
            config = _deserialize_configuration(item.name)
            if config is None:
                continue

            debater_cfg = ALL_VALID_DEBATERS.get(config.debater_key)
            judge_cfg = ALL_VALID_JUDGES.get(config.judge_key)
            if not debater_cfg or not judge_cfg:
                continue

            transcripts_dir = item / "outputs" / "transcripts"
            count, histogram, latest = collect_transcript_stats(transcripts_dir)

            entries.append(
                OutputEntry(
                    configuration=item.name,
                    config_type=config.config_type,
                    transcripts_directory=transcripts_dir,
                    task_label=format_task(config.task_type_name),
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
                label = format_debater(key, sorted_entries[0].debater_training)
            elif self.group_mode == GroupMode.DEBATER_TRAINING:
                label = f"Debater Training – {format_training(key)}"
            elif self.group_mode == GroupMode.JUDGE_MODEL:
                label = format_judge(key, sorted_entries[0].judge_training)
            else:
                label = f"Judge Training – {format_training(key)}"
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

    def _add_leaf(self, parent: Any, entry: OutputEntry) -> None:
        """Add a summary node with expandable configuration details."""
        summary = Text()
        summary.append(f"Task: {entry.task_label}", style="yellow")
        summary.append(" | ", style="dim")
        summary.append(f"Type: {entry.config_type.name}", style="white")
        summary.append(" | ", style="dim")
        summary.append(
            f"Debater: {format_debater(entry.debater_key, entry.debater_training)}",
            style="bright_magenta",
        )
        summary.append(" | ", style="dim")
        summary.append(
            f"Judge: {format_judge(entry.judge_key, entry.judge_training)}",
            style="bright_cyan",
        )
        summary.append(" | ", style="dim")
        summary.append(
            f"Transcripts: {entry.transcript_count} {pluralize('file', entry.transcript_count)}",
            style="bright_green",
        )
        if entry.latest_transcript:
            summary.append(" | ", style="dim")
            summary.append(
                f"Latest: {format_latest(entry.latest_transcript)}",
                style="bright_yellow",
            )

        leaf_node = parent.add(summary)
        leaf_node.data = entry
        leaf_node.add(Text(entry.configuration, style="bold cyan"))

        histogram = render_histogram(entry.transcripts_by_day)
        hist_label = leaf_node.add(Text("Transcripts per day", style="italic"))
        lines = histogram.splitlines() or ["No transcript files found."]

        for line in lines:
            hist_label.add_leaf(Text(line, style="dim", no_wrap=True))

        if entry.config_type == ConfigurationType.DATA_GENERATION:
            identifier_node = leaf_node.add(Text("Debate identifier stats", style="italic"))
            identifier_node.data = IdentifierStatsEntry(entry.transcripts_directory)
            identifier_node.allow_expand = True
            if entry.transcripts_directory in self._identifier_cache:
                counts, warning = self._identifier_cache[entry.transcripts_directory]
                self._populate_identifier_node(identifier_node, counts, warning)
            else:
                identifier_node.add(Text("Expand to load identifier statistics.", style="dim"))

    def on_tree_node_expanded(self, event: Tree.NodeExpanded) -> None:
        """Run the appropriate statistics when a configuration or stats node expands."""
        entry = getattr(event.node, "data", None)
        if isinstance(entry, OutputEntry):
            if entry.config_type == ConfigurationType.EVAL:
                self._trigger_evaluation_stats(entry)
        elif isinstance(entry, IdentifierStatsEntry):
            self._trigger_identifier_analysis(event.node, entry)

    def _trigger_evaluation_stats(self, entry: OutputEntry) -> None:
        """Kick off statistics analysis for a configuration entry."""
        stats_display = self.query_one("#stats-output", Static)
        if self._stats_thread and self._stats_thread.is_alive():
            stats_display.update(Text("Debate statistics analysis already running...", style="yellow"))
            return

        display_path = format_directory_for_display(entry.transcripts_directory, self.repository_root)
        stats_display.update(Text(f"Running debate statistics analysis for {display_path}...", style="yellow"))
        self._stats_thread = Thread(target=self._run_evaluation_stats, args=(entry,), daemon=True)
        self._stats_thread.start()

    def _run_evaluation_stats(self, entry: OutputEntry) -> None:
        """Execute the stats analysis in a background thread."""
        try:
            result = collect_directory_analysis(entry.transcripts_directory)
            self.call_from_thread(self._display_stats_result, entry, result)
        except (FileNotFoundError, NotADirectoryError) as exc:
            self.call_from_thread(self._display_stats_error, str(exc))
        except Exception as exc:  # noqa: BLE001
            self.call_from_thread(self._display_stats_error, f"Unexpected error: {exc}")

    def _display_stats_result(self, entry: OutputEntry, result: DirectoryAnalysisResult) -> None:
        """Render the statistics outcome inside the UI."""
        stats_display = self.query_one("#stats-output", Static)
        stats_display.update(self._format_stats_result(entry.config_type, result))

    def _display_stats_error(self, message: str) -> None:
        """Display analysis errors."""
        stats_display = self.query_one("#stats-output", Static)
        stats_display.update(Text(f"Error running stats: {message}", style="red"))

    def _trigger_identifier_analysis(
        self,
        node: Tree.Node,
        entry: IdentifierStatsEntry,
    ) -> None:
        """Populate the identifier statistics subtree for data-generation entries."""
        directory = entry.transcripts_directory
        cached = self._identifier_cache.get(directory)
        if cached is not None:
            counts, warning = cached
            self._populate_identifier_node(node, counts, warning)
            return

        existing_thread = self._identifier_threads.get(directory)
        if existing_thread and existing_thread.is_alive():
            return

        node.remove_children()
        node.add(Text("Loading debate identifier statistics...", style="yellow"))

        def worker() -> None:
            counts, warning = collect_identifier_stats(directory)
            self._identifier_cache[directory] = (counts, warning)
            self._identifier_threads.pop(directory, None)
            self.call_from_thread(self._populate_identifier_node, node, counts, warning)

        thread = Thread(target=worker, daemon=True)
        self._identifier_threads[directory] = thread
        thread.start()

    def _populate_identifier_node(
        self,
        node: Tree.Node,
        counts: dict[str, int],
        warning: Optional[str],
    ) -> None:
        """Render identifier statistics as children of the provided node."""
        node.remove_children()
        for line in render_identifier_summary_lines(counts, warning):
            node.add(line)

    def _format_stats_result(self, config_type: ConfigurationType, result: DirectoryAnalysisResult) -> Text:
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
            text.append("No valid debates found to analyze.\n", style="yellow")
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

        if config_type == ConfigurationType.EVAL:
            text.append("\nEmpty debate summary:\n", style="bold underline")
            total_transcripts = result.total_transcripts
            empty_transcripts = result.empty_transcripts
            if total_transcripts == 0:
                text.append("No transcripts found for this configuration.\n", style="yellow")
            else:
                empty_rate = (empty_transcripts / total_transcripts) * 100
                text.append(
                    f"Empty debates: {empty_transcripts}/{total_transcripts} ({empty_rate:.1f}%)\n"
                )
                if result.transcripts_by_day:
                    text.append("By day:\n", style="bold")
                    for day_key in sorted(result.transcripts_by_day.keys(), reverse=True):
                        day_total = result.transcripts_by_day[day_key]
                        day_empty = result.empty_transcripts_by_day.get(day_key, 0)
                        day_rate = (day_empty / day_total) * 100
                        text.append(
                            f"{day_key}: {day_empty}/{day_total} ({day_rate:.1f}%)\n",
                            style="dim",
                        )
                        examples = result.empty_transcript_examples.get(day_key, [])
                        for example in examples:
                            text.append(f"    - {example}\n", style="dim")

        if result.errors:
            text.append("\nErrors encountered:\n", style="bold red")
            max_errors = 5
            for error in result.errors[:max_errors]:
                text.append(f"- {error}\n", style="red")
            remaining = len(result.errors) - max_errors
            if remaining > 0:
                text.append(f"... {remaining} more errors\n", style="red")

        if result.stats_by_day:
            text.append("\nDaily breakdown:\n", style="bold underline")
            for day_key in sorted(result.stats_by_day.keys(), reverse=True):
                day_line = self._format_stats_line(day_key, result.stats_by_day[day_key])
                text += day_line
                text.append("\n")

        return text

    @staticmethod
    def _format_stats_line(label: str, stats: DebateStats) -> Text:
        """Return a formatted single-line summary of debate statistics."""
        percentages = stats.get_percentages()
        line = Text()
        line.append(f"{label}: ", style="bold")
        line.append(f"{stats.total_debates} {pluralize('debate', stats.total_debates)}")
        line.append(" | ", style="dim")
        line.append(
            f"A wins {stats.debater_a_wins} ({percentages['debater_a_win_rate']:.1f}%)",
            style="green",
        )
        line.append(" | ", style="dim")
        line.append(
            f"B wins {stats.debater_b_wins} ({percentages['debater_b_win_rate']:.1f}%)",
            style="cyan",
        )
        line.append(" | ", style="dim")
        line.append(
            f"Judge accuracy {percentages['judge_accuracy']:.1f}%",
            style="magenta",
        )
        line.append(" | ", style="dim")
        line.append(
            f"First debater accuracy {percentages['first_debater_accuracy']:.1f}%",
            style="yellow",
        )
        return line


def collect_transcript_stats(directory: Path) -> tuple[int, dict[str, int], Optional[datetime]]:
    """Return total transcripts, a per-day histogram, and latest timestamp."""
    if not directory.exists():
        return 0, {}, None
    counts: dict[str, int] = defaultdict(int)
    latest: Optional[datetime] = None
    try:
        for child in directory.iterdir():
            if not child.is_file():
                continue
            day = parse_transcript_day(child.name)
            if day:
                counts[day] += 1
            timestamp = parse_transcript_timestamp(child.name)
            if timestamp and (latest is None or timestamp > latest):
                latest = timestamp
    except OSError:
        return 0, {}, None

    ordered = dict(sorted(counts.items()))
    total = sum(ordered.values())
    return total, ordered, latest


def collect_identifier_stats(directory: Path) -> tuple[dict[str, int], Optional[str]]:
    """Return identifier counts for data-generation configurations."""
    if not directory.exists():
        return {}, f"No transcripts directory found at {directory}"
    extractor = ExtantDebateIdentifiersExtractor(str(directory))
    counts = extractor.process_files()
    return counts, None


def parse_transcript_day(filename: str) -> Optional[str]:
    """Extract YYYY-MM-DD from transcript filename."""
    if not filename:
        return None
    day_part = filename.split("_", 1)[0]
    try:
        datetime.strptime(day_part, "%Y-%m-%d")
    except ValueError:
        return None
    return day_part


def parse_transcript_timestamp(filename: str) -> Optional[datetime]:
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


def _deserialize_configuration(directory_name: str) -> Optional[ConfigurationName]:
    """Safely parse a directory name into a configuration instance."""
    try:
        return ConfigurationName.deserialize(directory_name)
    except Exception:  # noqa: BLE001
        return None
