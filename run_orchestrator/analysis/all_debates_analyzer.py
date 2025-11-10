from __future__ import annotations

import logging
from collections.abc import Iterator
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

LOGGER = logging.getLogger(__name__)


def iter_valid_configuration_directories(outputs_root: Path) -> Iterator[Path]:
    """Yield configuration directories whose names deserialize successfully."""

    for candidate_path in sorted(outputs_root.iterdir()):
        if not candidate_path.is_dir():
            continue

        try:
            configuration = ConfigurationName.deserialize(candidate_path.name)

            if configuration.config_type != ConfigurationType.EVAL:
                continue

        except ValueError:
            continue


        yield candidate_path


def analyze_configuration_directory(configuration_directory: Path) -> FullDebateAnalysis:
    """Run the full debate analysis for a single configuration directory."""

    transcripts_directory: Path = configuration_directory / "outputs" / "transcripts"
    if not transcripts_directory.is_dir():
        raise FileNotFoundError(
            f"Missing transcripts directory at '{transcripts_directory}'."
        )

    transcripts_iterator: Iterator[Transcript] = iter_transcripts_from_folder(
        transcripts_directory
    )
    analysis: FullDebateAnalysis = full_debate_analysis(transcripts_iterator)

    return analysis


def analyze_all_debates(outputs_root: Path) -> Dict[Path, FullDebateAnalysis]:
    """Analyze all debate outputs and persist the results as Parquet files."""

    analyses_by_directory: Dict[Path, FullDebateAnalysis] = {}

    valid_directories: List[Path] = list(
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

    progress_bar = tqdm(
        valid_directories,
        desc="Analyzing configurations",
        unit="config",
    )

    for index, configuration_directory in enumerate(progress_bar, start=1):
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
                configuration_directory
            )
        except FileNotFoundError:
            LOGGER.warning(
                "Skipping '%s' because no transcripts directory was found.",
                configuration_directory,
            )
            continue

        destination: Path = configuration_directory / PARQUET_FILENAME
        print(analysis.lengths)
        pydantic_to_parquet(analysis, destination)
        analyses_by_directory[configuration_directory] = analysis

    progress_bar.close()
    LOGGER.info("Completed analysis for %d configuration directories.", total_directories)

    return analyses_by_directory


def main(outputs_directory: Path | None = None) -> None:
    """Entry point for running the analyzer directly."""
    resolved_outputs_directory: Path = (
        outputs_directory if outputs_directory is not None else Path("outputs")
    )
    analyze_all_debates(resolved_outputs_directory.resolve())


if __name__ == "__main__":
    main()
