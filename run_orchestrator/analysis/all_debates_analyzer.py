from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
import pickle
from pathlib import Path
from typing import Dict, List

from run_orchestrator.analysis.debate_full_analyzer import full_debate_analysis
from run_orchestrator.analysis.analysis_models.full_debate_analysis import FullDebateAnalysis
from run_orchestrator.analysis.serializers import pydantic_to_parquet
from run_orchestrator.analysis.transcript_model import (
    Transcript,
    iter_transcripts_from_folder,
)
from run_orchestrator.evals_generator.config_spec import ConfigurationType
from run_orchestrator.evals_generator.configuration_name import ConfigurationName
from tqdm.auto import tqdm

PARQUET_FILENAME: str = "full_debate_analysis.parquet"
MAX_CONCURRENT_ANALYSES: int = 4

LOGGER = logging.getLogger(__name__)


def iter_valid_configuration_directories(outputs_root: Path) -> Iterator[tuple[ConfigurationName, Path]]:
    """Yield configuration directories whose names deserialize successfully."""

    for candidate_path in sorted(outputs_root.iterdir()):
        if not candidate_path.is_dir():
            continue

        try:
            configuration: ConfigurationName = ConfigurationName.deserialize(candidate_path.name)
        except ValueError:
            continue

        if configuration.config_type != ConfigurationType.EVAL:
            continue

        yield configuration, candidate_path


def analyze_configuration_directory(
    configuration: ConfigurationName,
    configuration_directory: Path,
) -> FullDebateAnalysis:
    """Run the full debate analysis for a single configuration directory."""

    transcripts_directory: Path = configuration_directory / "outputs" / "transcripts"
    if not transcripts_directory.is_dir():
        raise FileNotFoundError(
            f"Missing transcripts directory at '{transcripts_directory}'."
        )

    transcripts_iterator: Iterator[Transcript] = iter_transcripts_from_folder(
        transcripts_directory
    )
    analysis: FullDebateAnalysis = full_debate_analysis(
        configuration=configuration,
        transcripts=transcripts_iterator,
    )

    return analysis


def analyze_all_debates(outputs_root: Path) -> Dict[Path, FullDebateAnalysis]:
    """Analyze all debate outputs and persist the results as Parquet files."""

    analyses_by_directory: Dict[Path, FullDebateAnalysis] = {}
    valid_directories: List[tuple[ConfigurationName, Path]] = list(
        iter_valid_configuration_directories(outputs_root)
    )
    total_directories: int = len(valid_directories)
    LOGGER.info(
        "Found %d valid configuration directories under '%s'.",
        total_directories,
        outputs_root,
    )

    if total_directories == 0:
        return analyses_by_directory

    max_workers: int = min(MAX_CONCURRENT_ANALYSES, total_directories)
    aggregated_analyses: List[FullDebateAnalysis]

    if max_workers == 1:
        aggregated_analyses = _analyze_directories_sequentially(
            valid_directories=valid_directories,
            total_directories=total_directories,
            analyses_by_directory=analyses_by_directory,
        )
    elif _is_picklable(analyze_configuration_directory):
        aggregated_analyses = _analyze_directories_in_parallel(
            valid_directories=valid_directories,
            total_directories=total_directories,
            analyses_by_directory=analyses_by_directory,
            max_workers=max_workers,
        )
    else:
        LOGGER.info(
            "Falling back to sequential analysis because the analyzer callable is not picklable."
        )
        aggregated_analyses = _analyze_directories_sequentially(
            valid_directories=valid_directories,
            total_directories=total_directories,
            analyses_by_directory=analyses_by_directory,
        )

    if aggregated_analyses:
        combined_destination: Path = outputs_root / PARQUET_FILENAME
        pydantic_to_parquet(aggregated_analyses, combined_destination)
    LOGGER.info("Completed analysis for %d configuration directories.", total_directories)

    return analyses_by_directory


def _analyze_directories_in_parallel(
    *,
    valid_directories: List[tuple[ConfigurationName, Path]],
    total_directories: int,
    analyses_by_directory: Dict[Path, FullDebateAnalysis],
    max_workers: int,
) -> List[FullDebateAnalysis]:
    progress_bar = tqdm(
        valid_directories,
        desc="Analyzing configurations",
        unit="config",
    )
    progress_tracker: Iterator[tuple[ConfigurationName, Path]] = iter(progress_bar)
    aggregated_by_index: Dict[int, FullDebateAnalysis] = {}
    future_to_metadata: Dict[Future[FullDebateAnalysis], tuple[int, Path]] = {}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for index, (configuration, configuration_directory) in enumerate(valid_directories, start=1):
            progress_bar.set_description(f"Analyzing {configuration_directory.name}")
            remaining: int = total_directories - index
            LOGGER.info(
                "Processing '%s' (%d/%d; %d remaining)",
                configuration_directory.name,
                index,
                total_directories,
                remaining,
            )
            future: Future[FullDebateAnalysis] = executor.submit(
                analyze_configuration_directory,
                configuration,
                configuration_directory,
            )
            future_to_metadata[future] = (index - 1, configuration_directory)

        for future in as_completed(future_to_metadata):
            index_zero_based, configuration_directory = future_to_metadata[future]
            try:
                analysis = future.result()
            except FileNotFoundError:
                LOGGER.warning(
                    "Skipping '%s' because no transcripts directory was found.",
                    configuration_directory,
                )
                next(progress_tracker, None)
                continue
            analyses_by_directory[configuration_directory] = analysis
            aggregated_by_index[index_zero_based] = analysis
            next(progress_tracker, None)
            progress_bar.set_description(f"Completed {configuration_directory.name}")

    progress_bar.close()
    ordered_indices: List[int] = sorted(aggregated_by_index)
    return [aggregated_by_index[index] for index in ordered_indices]


def _analyze_directories_sequentially(
    *,
    valid_directories: List[tuple[ConfigurationName, Path]],
    total_directories: int,
    analyses_by_directory: Dict[Path, FullDebateAnalysis],
) -> List[FullDebateAnalysis]:
    progress_bar: Iterable[tuple[ConfigurationName, Path]] = tqdm(
        valid_directories,
        desc="Analyzing configurations",
        unit="config",
    )
    aggregated_analyses: List[FullDebateAnalysis] = []

    for index, (configuration, configuration_directory) in enumerate(progress_bar, start=1):
        progress_bar.set_description(
            f"Analyzing {configuration_directory.name}"
        )
        remaining: int = total_directories - index
        LOGGER.info(
            "Processing '%s' (%d/%d; %d remaining)",
            configuration_directory.name,
            index,
            total_directories,
            remaining,
        )

        try:
            analysis: FullDebateAnalysis = analyze_configuration_directory(
                configuration,
                configuration_directory,
            )
        except FileNotFoundError:
            LOGGER.warning(
                "Skipping '%s' because no transcripts directory was found.",
                configuration_directory,
            )
            continue

        analyses_by_directory[configuration_directory] = analysis
        aggregated_analyses.append(analysis)

    progress_bar.close()
    return aggregated_analyses


def _is_picklable(candidate: object) -> bool:
    try:
        pickle.dumps(candidate)
    except (pickle.PickleError, AttributeError, TypeError):
        return False
    return True


def main(outputs_directory: Path | None = None) -> None:
    """Entry point for running the analyzer directly."""
    resolved_outputs_directory: Path = (
        outputs_directory if outputs_directory is not None else Path("outputs")
    )
    analyze_all_debates(resolved_outputs_directory.resolve())


if __name__ == "__main__":
    main()
