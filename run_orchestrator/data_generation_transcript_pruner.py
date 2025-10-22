import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

from run_orchestrator.debate_emptiness_analyzer import infer_configuration_type
from run_orchestrator.evals_generator.config_spec import ConfigurationType
from run_orchestrator.data_generation_transcript_model import (
    DataGenerationTranscript,
    read_data_generation_transcripts,
)


def summarize_distribution(transcripts: Iterable[DataGenerationTranscript]) -> Counter[int]:
    """Return a histogram of transcript counts per debate identifier."""
    counts: Counter[int] = Counter()
    grouped: Dict[str, int] = defaultdict(int)
    for transcript in transcripts:
        grouped[transcript.debate_identifier] += 1
    for quantity in grouped.values():
        counts[quantity] += 1
    return counts


def collect_prunable_transcripts(transcripts: List[DataGenerationTranscript]) -> Dict[str, List[DataGenerationTranscript]]:
    """Group transcripts by debate identifier where more than two instances exist."""
    grouped: Dict[str, List[DataGenerationTranscript]] = defaultdict(list)
    for transcript in transcripts:
        grouped[transcript.debate_identifier].append(transcript)

    prunable: Dict[str, List[DataGenerationTranscript]] = {}
    for debate_identifier, items in grouped.items():
        if len(items) > 2:
            sorted_items = sorted(items, key=lambda t: t.file_path.name)
            prunable[debate_identifier] = sorted_items[2:]
    return prunable


def filter_data_generation_transcripts(transcripts: List[DataGenerationTranscript]) -> List[DataGenerationTranscript]:
    """Return only transcripts that belong to data-generation configurations."""
    filtered: List[DataGenerationTranscript] = []
    for transcript in transcripts:
        configuration_type = infer_configuration_type(transcript.file_path)
        if configuration_type == ConfigurationType.DATA_GENERATION:
            filtered.append(transcript)
    return filtered


def delete_transcripts(transcripts_to_delete: List[DataGenerationTranscript]) -> None:
    """Remove the provided transcript files from disk."""
    for transcript in transcripts_to_delete:
        transcript.file_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Remove debate transcripts in ConfigurationType.DATA_GENERATION directories when more than "
            "two transcripts exist for the same debate identifier."
        )
    )
    parser.add_argument(
        "folder_path",
        type=str,
        help="Root folder containing transcript JSON files.",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="If set, deletes the surplus transcript files. Otherwise, only prints what would be removed.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the before/after distribution without deleting files. Implied when --delete is not provided.",
    )

    args = parser.parse_args()

    folder_path = Path(args.folder_path).resolve()
    transcripts = read_data_generation_transcripts(folder_path)

    if not transcripts:
        print(f"No transcripts found in {folder_path}. Exiting.")
        return

    data_generation_transcripts = filter_data_generation_transcripts(transcripts)
    if not data_generation_transcripts:
        print("No ConfigurationType.DATA_GENERATION transcripts found. Exiting.")
        return

    prunable = collect_prunable_transcripts(data_generation_transcripts)

    if not prunable:
        print("No debate identifiers have more than two transcripts. Nothing to do.")
        return

    before_summary = summarize_distribution(data_generation_transcripts)
    after_transcripts = list(data_generation_transcripts)
    for items in prunable.values():
        for transcript in items:
            if transcript in after_transcripts:
                after_transcripts.remove(transcript)
    after_summary = summarize_distribution(after_transcripts)

    total_to_delete = sum(len(items) for items in prunable.values())
    print(f"Identified {total_to_delete} transcripts to remove across {len(prunable)} debate identifiers.")

    print("\nDistribution of transcripts per debate identifier:")
    print("Before pruning:")
    for count, occurrences in sorted(before_summary.items()):
        print(f"  {count}: {occurrences}")
    print("After pruning:")
    for count, occurrences in sorted(after_summary.items()):
        print(f"  {count}: {occurrences}")

    for debate_identifier, items in sorted(prunable.items()):
        print(f"\nDebate identifier: {debate_identifier}")
        for transcript in items:
            print(f"  - {transcript.file_path}")

    should_delete = args.delete and not args.dry_run

    if should_delete:
        transcripts_to_delete = [transcript for items in prunable.values() for transcript in items]
        delete_transcripts(transcripts_to_delete)
        print(f"\nDeleted {len(transcripts_to_delete)} transcript files.")
    else:
        print("\nRun with --delete to remove the listed transcript files.")

    if args.dry_run or not should_delete:
        print("\nDry run complete. No files were deleted.")


if __name__ == "__main__":
    main()
