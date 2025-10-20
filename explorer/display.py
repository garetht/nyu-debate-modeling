#!/usr/bin/env python3
"""Display helpers for the explorer textual interface."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Final, Optional

from rich.text import Text


MAX_BAR_WIDTH: Final[int] = 40
BAR_CHARACTER: Final[str] = "█"
SECTION_TITLE: Final[str] = "Transcripts per Day"


def render_histogram(histogram: dict[str, int]) -> str:
    """Render a professional horizontal histogram with summary details."""
    if not histogram:
        return "No transcript files found."

    items: list[tuple[str, int]] = list(histogram.items())
    counts: list[int] = [count for _, count in items]
    max_count: int = max(counts, default=0)

    if max_count <= 0:
        return "Transcripts counted but totals are zero."

    total_transcripts: int = sum(counts)
    day_count: int = len(items)
    label_width: int = max(len(label) for label, _ in items)
    count_width: int = max(len("Count"), len(str(max_count)))

    today: date = datetime.now().date()
    ago_labels: list[str] = []
    for label, _ in items:
        ago_text: str = ""
        try:
            transcript_day: date = datetime.strptime(label, "%Y-%m-%d").date()
            delta_days: int = (today - transcript_day).days
            if delta_days < 0:
                ago_text = f"{abs(delta_days)}d future"
            elif delta_days == 0:
                ago_text = "today"
            elif delta_days == 1:
                ago_text = "1d ago"
            else:
                ago_text = f"{delta_days}d ago"
        except ValueError:
            ago_text = ""
        ago_labels.append(ago_text)

    has_relative_dates: bool = any(label for label in ago_labels)
    when_width: int = (
        max(len("When"), max((len(label) for label in ago_labels), default=0))
        if has_relative_dates
        else 0
    )

    scale: float = 1.0
    if max_count > MAX_BAR_WIDTH:
        scale = float(MAX_BAR_WIDTH) / float(max_count)

    title_line: str = SECTION_TITLE
    header_line: str
    if has_relative_dates:
        header_line = (
            f"{'Date':<{label_width}}  {'When':<{when_width}}  "
            f"{'Count':>{count_width}}  Histogram"
        )
    else:
        header_line = (
            f"{'Date':<{label_width}}  {'Count':>{count_width}}  Histogram"
        )

    divider_length: int = max(len(title_line), len(header_line))
    divider: str = "─" * divider_length
    lines: list[str] = [title_line, divider, header_line, divider]

    for (label, count), ago_text in zip(items, ago_labels):
        scaled_length: int = max(
            1, int(round(float(count) * scale))
        ) if count > 0 else 0
        bar: str = BAR_CHARACTER * scaled_length
        if has_relative_dates:
            when_value: str = ago_text or "—"
            row: str = (
                f"{label:<{label_width}}  {when_value:<{when_width}}  "
                f"{count:>{count_width}}  {bar}"
            )
        else:
            row: str = (
                f"{label:<{label_width}}  {count:>{count_width}}  {bar}"
            )
        lines.append(row.rstrip())

    transcript_label: str = pluralize("transcript", total_transcripts)
    day_label: str = pluralize("day", day_count)
    peak_day: tuple[str, int] = max(
        items,
        key=lambda pair: pair[1],
    )
    peak_day_label: str = peak_day[0]
    peak_day_count: int = peak_day[1]

    lines.extend(
        [
            "",
            f"{total_transcripts} {transcript_label} across {day_count} {day_label}.",
            (
                "Peak day: "
                f"{peak_day_label} ({peak_day_count} "
                f"{pluralize('transcript', peak_day_count)})."
            ),
        ]
    )

    return "\n".join(lines)


def pluralize(noun: str, count: int) -> str:
    """Return a pluralized label for counts."""
    suffix = "" if count == 1 else "s"
    return f"{noun}{suffix}"


def format_task(task_type_name: str) -> str:
    """Return a polished task type label."""
    return task_type_name.replace("_", " ").replace("-", " ").title()


def format_latest(timestamp: Optional[datetime]) -> str:
    """Return a readable latest transcript timestamp."""
    if timestamp is None:
        return "—"
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")


def format_debater(key: str, training_round: str) -> str:
    """Return a formatted debater description."""
    return f"{key} ({format_training(training_round)})"


def format_judge(key: str, training_round: str) -> str:
    """Return a formatted judge description."""
    return f"{key} ({format_training(training_round)})"


def format_training(training_round: str) -> str:
    """Return a friendly training round label."""
    return training_round.replace("-", " ").replace("_", " ").title()


def format_directory_for_display(directory: Path, repository_root: Path) -> str:
    """Return a path string relative to the repository when possible."""
    try:
        relative_path = directory.relative_to(repository_root)
    except ValueError:
        return str(directory)
    return str(relative_path)


def render_identifier_summary_lines(
    counts: dict[str, int],
    warning: Optional[str],
) -> list[Text]:
    """Produce formatted lines describing identifier frequency details."""
    lines: list[Text] = []

    if warning:
        lines.append(Text(warning, style="yellow"))
        return lines

    if not counts:
        lines.append(Text("No debate identifiers found.", style="yellow"))
        return lines

    total_identifiers = sum(counts.values())
    unique_identifiers = len(counts)
    summary = Text()
    summary.append("Total debate identifiers counted: ", style="bold")
    summary.append(str(total_identifiers))
    summary.append(" (")
    summary.append(f"{unique_identifiers} unique", style="dim")
    summary.append(")")
    lines.append(summary)

    top_five = sorted(
        counts.items(),
        key=lambda item: (-item[1], item[0]),
    )[:5]
    bottom_five = sorted(
        counts.items(),
        key=lambda item: (item[1], item[0]),
    )[:5]

    if top_five:
        lines.append(Text("Top 5 identifiers:", style="bold"))
        for identifier, count in top_five:
            lines.append(Text(f"- {identifier}: {count}", style="green"))

    if bottom_five:
        lines.append(Text("Bottom 5 identifiers:", style="bold"))
        for identifier, count in bottom_five:
            lines.append(Text(f"- {identifier}: {count}", style="cyan"))

    return lines
