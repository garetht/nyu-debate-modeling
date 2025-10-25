from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from explorer.cli.explorer_models import ConfigTypeLiteral, EVAL_CONFIG_TYPE, GroupMode
from run_orchestrator.debate_stats_analyzer import (
    DebateStats,
    analyze_debate_file,
    collect_directory_analysis,
)
from run_orchestrator.evals_generator.configuration_name import ConfigurationName
from run_orchestrator.evals_generator.model_definitions import (
    ALL_VALID_DEBATERS,
    ALL_VALID_JUDGES,
)

from explorer.explorer_backend.models import (
    DailyDebateStatsResponse,
    DebateStatsSummary,
    OutputDetailResponse,
    OutputGroupResponse,
    OutputStatsResponse,
    OutputSummaryResponse,
    OutputsListResponse,
    TranscriptFileResponse,
)

ConfigType = ConfigTypeLiteral


class OutputsDirectoryMissingError(Exception):
    """Raised when the outputs directory does not exist."""

    def __init__(self, outputs_directory: Path) -> None:
        self.outputs_directory: Path = outputs_directory
        super().__init__(f"Outputs directory not found at {outputs_directory}")


class OutputNotFoundError(Exception):
    """Raised when the requested configuration output is absent."""

    def __init__(self, configuration: str) -> None:
        self.configuration: str = configuration
        super().__init__(f"Output configuration '{configuration}' was not found.")


class InvalidGroupModeError(Exception):
    """Raised when the provided group mode key is invalid."""

    def __init__(self, group_mode: str) -> None:
        self.group_mode: str = group_mode
        super().__init__(f"Unknown group_mode '{group_mode}'.")


class OutputStatsUnavailableError(Exception):
    """Raised when statistics are requested for a non-eval configuration."""

    def __init__(self, configuration: str, config_type: ConfigType) -> None:
        self.configuration: str = configuration
        self.config_type: ConfigType = config_type
        super().__init__(
            f"Statistics are only available for eval configurations. '{configuration}' is of type '{config_type}'."
        )


@dataclass(frozen=True)
class _OutputSummary:
    configuration: str
    config_type: ConfigType
    directory: Path
    transcripts_directory: Path
    directory_size_bytes: int
    task_label: str
    debater_key: str
    debater_training: str
    judge_key: str
    judge_training: str
    transcript_count: int
    transcripts_by_day: Dict[str, int]
    latest_transcript: datetime | None


@dataclass(frozen=True)
class _TranscriptFile:
    path: Path
    size_bytes: int
    modified_at: datetime | None


def list_outputs(group_mode_key: str | None = None) -> OutputsListResponse:
    """Return summaries of all available outputs, optionally grouped."""
    repository_root = _determine_repository_root()
    outputs_directory = repository_root / "outputs"
    if not outputs_directory.exists():
        raise OutputsDirectoryMissingError(outputs_directory)

    summaries = _collect_output_summaries(outputs_directory)
    group_mode = _resolve_group_mode(group_mode_key)

    sorted_entries = _sort_entries_for_mode(summaries, group_mode)
    summary_models = [_to_summary_model(entry, repository_root) for entry in sorted_entries]
    group_models = _build_group_models(summaries, group_mode, repository_root)

    return OutputsListResponse(
        outputs_directory=_format_path_relative(outputs_directory, repository_root),
        group_mode=group_mode.key,
        entries=summary_models,
        groups=group_models,
    )


def get_output_detail(configuration: str, page: int, page_size: int) -> OutputDetailResponse:
    """Return detailed metadata for a specific output configuration."""
    if page <= 0:
        raise ValueError("page must be greater than 0.")
    if page_size <= 0:
        raise ValueError("page_size must be greater than 0.")

    repository_root = _determine_repository_root()
    outputs_directory = repository_root / "outputs"
    if not outputs_directory.exists():
        raise OutputsDirectoryMissingError(outputs_directory)

    summary = _resolve_output_summary(configuration, outputs_directory)
    transcripts = _list_transcript_files(summary.transcripts_directory)

    total_transcripts = len(transcripts)
    total_pages = math.ceil(total_transcripts / page_size) if total_transcripts else 0
    clamped_page = _clamp_page(page, total_pages)

    start_index = (clamped_page - 1) * page_size if total_transcripts else 0
    end_index = start_index + page_size if total_transcripts else 0
    paginated_transcripts = transcripts[start_index:end_index] if total_transcripts else []

    transcript_models = [
        TranscriptFileResponse(
            name=file.path.name,
            relative_path=_format_path_relative(file.path, repository_root),
            size_bytes=file.size_bytes,
            modified_at=file.modified_at,
        )
        for file in paginated_transcripts
    ]

    transcripts_directory_str = _format_path_relative(summary.transcripts_directory, repository_root)

    return OutputDetailResponse(
        configuration=summary.configuration,
        config_type=summary.config_type,
        task_label=summary.task_label,
        debater_key=summary.debater_key,
        debater_training=summary.debater_training,
        judge_key=summary.judge_key,
        judge_training=summary.judge_training,
        transcript_count=summary.transcript_count,
        transcripts_by_day=summary.transcripts_by_day,
        latest_transcript=summary.latest_transcript,
        directory_size_bytes=summary.directory_size_bytes,
        transcripts_directory=transcripts_directory_str,
        page=clamped_page,
        page_size=page_size,
        total_transcripts=total_transcripts,
        total_pages=total_pages,
        transcripts=transcript_models,
    )


def get_output_stats(configuration: str) -> OutputStatsResponse:
    """Return aggregated debate statistics for an eval configuration."""
    repository_root = _determine_repository_root()
    outputs_directory = repository_root / "outputs"
    if not outputs_directory.exists():
        raise OutputsDirectoryMissingError(outputs_directory)

    summary = _resolve_output_summary(configuration, outputs_directory)
    if summary.config_type != EVAL_CONFIG_TYPE:
        raise OutputStatsUnavailableError(configuration, summary.config_type)

    analysis_result = collect_directory_analysis(summary.transcripts_directory)
    per_day_stats = _compute_stats_by_day(summary.transcripts_directory)

    overall_summary = _stats_to_summary(analysis_result.overall_stats)
    per_day_responses = [
        DailyDebateStatsResponse(day=day, **_stats_to_summary(stats).model_dump())
        for day, stats in sorted(per_day_stats.items(), key=lambda item: item[0], reverse=True)
        if stats.total_debates > 0
    ]

    transcripts_directory_str = _format_path_relative(summary.transcripts_directory, repository_root)

    return OutputStatsResponse(
        configuration=summary.configuration,
        transcripts_directory=transcripts_directory_str,
        json_file_count=len(analysis_result.json_files),
        overall_stats=overall_summary,
        per_day=per_day_responses,
        errors=analysis_result.errors,
    )


def _resolve_group_mode(group_mode_key: str | None) -> GroupMode:
    if group_mode_key is None:
        return GroupMode.ALL
    mode = GroupMode.from_key(group_mode_key)
    if mode is None:
        raise InvalidGroupModeError(group_mode_key)
    return mode


def _collect_output_summaries(outputs_directory: Path) -> List[_OutputSummary]:
    summaries: List[_OutputSummary] = []
    for item in sorted(outputs_directory.iterdir(), key=lambda path: path.name.lower()):
        if not item.is_dir():
            continue
        summary = _build_summary_for_directory(item)
        if summary is not None:
            summaries.append(summary)
    return summaries


def _build_summary_for_directory(directory: Path) -> _OutputSummary | None:
    config = _deserialize_configuration(directory.name)
    if config is None:
        return None
    debater_cfg = ALL_VALID_DEBATERS.get(config.debater_key)
    judge_cfg = ALL_VALID_JUDGES.get(config.judge_key)
    if debater_cfg is None or judge_cfg is None:
        return None

    transcripts_dir = directory / "outputs" / "transcripts"
    transcript_count, histogram, latest = _collect_transcript_stats(transcripts_dir)
    directory_size = _calculate_directory_size_bytes(directory)

    return _OutputSummary(
        configuration=directory.name,
        config_type=config.config_type.value,  # type: ignore[return-value]
        directory=directory,
        transcripts_directory=transcripts_dir,
        directory_size_bytes=directory_size,
        task_label=_format_task(config.task_type_name),
        debater_key=config.debater_key,
        debater_training=debater_cfg.training_round.display_name,
        judge_key=config.judge_key,
        judge_training=judge_cfg.training_round.display_name,
        transcript_count=transcript_count,
        transcripts_by_day=histogram,
        latest_transcript=latest,
    )


def _sort_entries_for_mode(entries: Sequence[_OutputSummary], group_mode: GroupMode) -> List[_OutputSummary]:
    if group_mode == GroupMode.ALL:
        return sorted(
            entries,
            key=lambda entry: entry.latest_transcript or datetime.min,
            reverse=True,
        )
    return sorted(entries, key=lambda entry: entry.configuration.lower())


def _build_group_models(
    entries: Sequence[_OutputSummary],
    group_mode: GroupMode,
    repository_root: Path,
) -> List[OutputGroupResponse]:
    if not entries:
        return []

    if group_mode == GroupMode.ALL:
        sorted_entries = _sort_entries_for_mode(entries, group_mode)
        summaries = [_to_summary_model(entry, repository_root) for entry in sorted_entries]
        return [
            OutputGroupResponse(label="Recent Transcripts", entries=summaries),
        ]

    grouped: Dict[str, List[_OutputSummary]] = {}
    for entry in entries:
        key = _group_key_for_entry(entry, group_mode)
        if key is None:
            continue
        grouped.setdefault(key, []).append(entry)

    group_responses: List[OutputGroupResponse] = []
    for key, grouped_entries in grouped.items():
        label = _group_label_for_key(key, grouped_entries, group_mode)
        sorted_group = sorted(grouped_entries, key=lambda entry: entry.configuration.lower())
        summaries = [_to_summary_model(entry, repository_root) for entry in sorted_group]
        group_responses.append(OutputGroupResponse(label=label, entries=summaries))

    return sorted(group_responses, key=lambda group: group.label.lower())


def _group_key_for_entry(entry: _OutputSummary, group_mode: GroupMode) -> str | None:
    if group_mode == GroupMode.DEBATER_MODEL:
        return entry.debater_key
    if group_mode == GroupMode.DEBATER_TRAINING:
        return entry.debater_training
    if group_mode == GroupMode.JUDGE_MODEL:
        return entry.judge_key
    if group_mode == GroupMode.JUDGE_TRAINING:
        return entry.judge_training
    return None


def _group_label_for_key(key: str, entries: Sequence[_OutputSummary], group_mode: GroupMode) -> str:
    if not entries:
        return key
    sample = entries[0]
    if group_mode == GroupMode.DEBATER_MODEL:
        return _format_model_label(sample.debater_key, sample.debater_training, prefix="Debater")
    if group_mode == GroupMode.DEBATER_TRAINING:
        return f"Debater Training – {_format_training(key)}"
    if group_mode == GroupMode.JUDGE_MODEL:
        return _format_model_label(sample.judge_key, sample.judge_training, prefix="Judge")
    if group_mode == GroupMode.JUDGE_TRAINING:
        return f"Judge Training – {_format_training(key)}"
    return key


def _format_model_label(model_key: str, training_round: str, *, prefix: str) -> str:
    formatted_training = _format_training(training_round)
    return f"{prefix}: {model_key} ({formatted_training})"


def _resolve_output_summary(configuration: str, outputs_directory: Path) -> _OutputSummary:
    for summary in _collect_output_summaries(outputs_directory):
        if summary.configuration == configuration:
            return summary
    raise OutputNotFoundError(configuration)


def _list_transcript_files(transcripts_directory: Path) -> List[_TranscriptFile]:
    if not transcripts_directory.exists():
        return []
    files: List[_TranscriptFile] = []
    for path in sorted(transcripts_directory.glob("*.json")):
        try:
            stats = path.stat()
        except OSError:
            continue
        modified_at: datetime | None
        try:
            modified_at = datetime.fromtimestamp(stats.st_mtime)
        except (OSError, OverflowError, ValueError):
            modified_at = None
        files.append(
            _TranscriptFile(
                path=path,
                size_bytes=int(stats.st_size),
                modified_at=modified_at,
            )
        )
    files.sort(key=lambda file: (file.modified_at or datetime.min, file.path.name), reverse=True)
    return files


def _compute_stats_by_day(transcripts_directory: Path) -> Dict[str, DebateStats]:
    stats_by_day: Dict[str, DebateStats] = {}
    if not transcripts_directory.exists():
        return stats_by_day
    for path in transcripts_directory.glob("*.json"):
        day = _parse_transcript_day(path.name)
        if day is None:
            continue
        file_stats, _ = analyze_debate_file(path)
        if file_stats.total_debates == 0:
            continue
        bucket = stats_by_day.setdefault(day, DebateStats())
        _merge_stats(bucket, file_stats)
    return stats_by_day


def _merge_stats(target: DebateStats, source: DebateStats) -> None:
    target.total_debates += source.total_debates
    target.debater_a_wins += source.debater_a_wins
    target.debater_b_wins += source.debater_b_wins
    target.judge_correct += source.judge_correct
    target.first_debater_correct += source.first_debater_correct
    target.debater_a_probs.extend(source.debater_a_probs)
    target.debater_b_probs.extend(source.debater_b_probs)


def _stats_to_summary(stats: DebateStats) -> DebateStatsSummary:
    percentages = stats.get_percentages()
    return DebateStatsSummary(
        total_debates=stats.total_debates,
        debater_a_wins=stats.debater_a_wins,
        debater_b_wins=stats.debater_b_wins,
        judge_correct=stats.judge_correct,
        first_debater_correct=stats.first_debater_correct,
        debater_a_win_rate=percentages["debater_a_win_rate"],
        debater_b_win_rate=percentages["debater_b_win_rate"],
        judge_accuracy=percentages["judge_accuracy"],
        first_debater_accuracy=percentages["first_debater_accuracy"],
    )


def _clamp_page(requested_page: int, total_pages: int) -> int:
    if total_pages == 0:
        return 1
    return min(requested_page, total_pages)


def _determine_repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _format_path_relative(path: Path, repository_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repository_root.resolve()))
    except ValueError:
        return str(path)


def _collect_transcript_stats(directory: Path) -> Tuple[int, Dict[str, int], datetime | None]:
    if not directory.exists():
        return 0, {}, None
    counts: Dict[str, int] = {}
    latest: datetime | None = None
    try:
        for child in directory.iterdir():
            if not child.is_file():
                continue
            day = _parse_transcript_day(child.name)
            if day is not None:
                counts[day] = counts.get(day, 0) + 1
            timestamp = _parse_transcript_timestamp(child.name)
            if timestamp is not None and (latest is None or timestamp > latest):
                latest = timestamp
    except OSError:
        return 0, {}, None

    ordered_counts = dict(sorted(counts.items()))
    total = sum(ordered_counts.values())
    return total, ordered_counts, latest


def _parse_transcript_day(filename: str) -> str | None:
    if not filename:
        return None
    day_part = filename.split("_", 1)[0]
    try:
        datetime.strptime(day_part, "%Y-%m-%d")
    except ValueError:
        return None
    return day_part


def _parse_transcript_timestamp(filename: str) -> datetime | None:
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


def _calculate_directory_size_bytes(directory: Path) -> int:
    total_size = 0
    try:
        for path in directory.rglob("*"):
            try:
                if path.is_file():
                    total_size += path.stat().st_size
            except OSError:
                continue
    except OSError:
        return 0
    return total_size


def _format_task(task_type_name: str) -> str:
    return task_type_name.replace("_", " ").replace("-", " ").title()


def _format_training(training_round: str) -> str:
    return training_round.replace("-", " ").replace("_", " ").title()


def _to_summary_model(summary: _OutputSummary, repository_root: Path) -> OutputSummaryResponse:
    transcripts_directory = _format_path_relative(summary.transcripts_directory, repository_root)
    return OutputSummaryResponse(
        configuration=summary.configuration,
        config_type=summary.config_type,
        task_label=summary.task_label,
        debater_key=summary.debater_key,
        debater_training=summary.debater_training,
        judge_key=summary.judge_key,
        judge_training=summary.judge_training,
        transcript_count=summary.transcript_count,
        transcripts_by_day=summary.transcripts_by_day,
        latest_transcript=summary.latest_transcript,
        directory_size_bytes=summary.directory_size_bytes,
        transcripts_directory=transcripts_directory,
    )


def _deserialize_configuration(directory_name: str) -> ConfigurationName | None:
    try:
        return ConfigurationName.deserialize(directory_name)
    except Exception:  # noqa: BLE001
        return None


__all__ = [
    "InvalidGroupModeError",
    "OutputNotFoundError",
    "OutputStatsUnavailableError",
    "OutputsDirectoryMissingError",
    "get_output_detail",
    "get_output_stats",
    "list_outputs",
]
