import argparse
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

from run_orchestrator.analysis.analysis_models.debate_distribution import (
    DebateDistributionArgs,
    DebateDistributionAnalysis,
)
from run_orchestrator.analysis.transcript_model import Transcript, iter_transcripts_from_folder


def parse_args() -> DebateDistributionArgs:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Analyze the distribution of debate identifiers in a folder of transcripts."
    )
    parser.add_argument(
        "folder_path",
        type=str,
        help="The path to the folder containing transcript JSON files."
    )
    parser.add_argument(
        "--keep-per-debate",
        type=int,
        default=None,
        metavar="INT",
        help="Maximum number of transcripts to keep per debate identifier. Additional transcripts are candidates for deletion.",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete transcripts exceeding the keep-per-debate limit instead of just reporting them.",
    )
    parsed_args: argparse.Namespace = parser.parse_args()

    folder_path: Path = Path(parsed_args.folder_path).resolve()

    keep_per_debate_arg: int | None = parsed_args.keep_per_debate
    if keep_per_debate_arg is not None and keep_per_debate_arg < 0:
        parser.error("--keep-per-debate must be a non-negative integer.")

    delete_flag: bool = bool(parsed_args.delete)

    return DebateDistributionArgs(
        folder_path=folder_path,
        keep_per_debate=keep_per_debate_arg,
        delete=delete_flag,
    )


def _split_identifier(identifier: str) -> tuple[str, str]:
    if "_" not in identifier:
        return identifier, ""
    title, topic = identifier.split("_", 1)
    return title, topic


def analyze_debate_distribution(transcripts: Iterable[Transcript]) -> DebateDistributionAnalysis:
    identifier_counter: Counter[str] = Counter()
    title_counter: Counter[str] = Counter()
    transcript_count: int = 0

    for transcript in transcripts:
        transcript_count += 1
        identifier: str = transcript.metadata.debate_identifier
        title, _ = _split_identifier(identifier)
        identifier_counter.update([identifier])
        title_counter.update([title])

    identifier_counts: dict[str, int] = {
        identifier: count
        for identifier, count in sorted(identifier_counter.items())
    }
    title_counts: dict[str, int] = {
        title: count for title, count in sorted(title_counter.items())
    }

    return DebateDistributionAnalysis(
        identifier_counts=identifier_counts,
        title_counts=title_counts,
        transcript_count=transcript_count,
    )


def _select_transcripts_to_delete(
    transcripts: Sequence[Transcript],
    keep_per_debate: int,
) -> list[Transcript]:
    transcripts_by_identifier: dict[str, list[Transcript]] = {}
    for transcript in transcripts:
        identifier: str = transcript.metadata.debate_identifier
        identifier_transcripts: list[Transcript] = transcripts_by_identifier.setdefault(identifier, [])
        identifier_transcripts.append(transcript)

    transcripts_to_delete: list[Transcript] = []
    for identifier_transcripts in transcripts_by_identifier.values():
        sorted_transcripts: list[Transcript] = sorted(
            identifier_transcripts,
            key=lambda transcript: str(transcript.file_path),
        )
        if len(sorted_transcripts) > keep_per_debate:
            transcripts_to_delete.extend(sorted_transcripts[keep_per_debate:])

    return transcripts_to_delete


def main() -> None:
    args = parse_args()

    transcript_iterator: Iterable[Transcript] | None = iter_transcripts_from_folder(args.folder_path)
    if transcript_iterator is None:
        return

    transcripts: list[Transcript] = list(transcript_iterator)

    analysis: DebateDistributionAnalysis = analyze_debate_distribution(transcripts)

    if analysis.transcript_count == 0:
        print(f"No transcripts found in {args.folder_path}. Exiting.")
        return

    for identifier, count in analysis.identifier_counts.items():
        print(f"{identifier}: {count}")

    count_distribution_counter: Counter[int] = Counter(analysis.identifier_counts.values())
    count_distribution: dict[int, int] = {
        occurrences: identifier_total
        for occurrences, identifier_total in sorted(count_distribution_counter.items())
    }

    print("\nCounts of identifiers by transcript frequency:")
    for occurrences, identifier_total in count_distribution.items():
        print(f"{occurrences}: {identifier_total}")

    print("\nCounts by title:")
    for title, count in analysis.title_counts.items():
        print(f"{title}: {count}")

    if args.keep_per_debate is None:
        return

    transcripts_to_delete: list[Transcript] = _select_transcripts_to_delete(
        transcripts,
        args.keep_per_debate,
    )

    if not transcripts_to_delete:
        print("\nNo transcripts exceed the keep-per-debate limit.")
        return

    identifier_deletion_counts_counter: Counter[str] = Counter(
        transcript.metadata.debate_identifier for transcript in transcripts_to_delete
    )
    identifier_deletion_counts: dict[str, int] = {
        identifier: count
        for identifier, count in sorted(identifier_deletion_counts_counter.items())
    }

    print("\nTranscripts exceeding keep-per-debate limit (identifier: surplus count):")
    for identifier, count in identifier_deletion_counts.items():
        print(f"{identifier}: {count}")

    if not args.delete:
        print("\nRe-run with --delete to remove these transcripts.")
        return

    deleted_counter: Counter[str] = Counter()
    for transcript in transcripts_to_delete:
        file_path = Path(transcript.file_path)
        identifier = transcript.metadata.debate_identifier
        try:
            file_path.unlink()
            deleted_counter.update([identifier])
        except FileNotFoundError:
            print(
                f"Warning: Transcript for identifier {identifier} could not be deleted because the file was not found."
            )
        except OSError as error:
            print(f"Error deleting transcript for identifier {identifier}: {error}")

    if deleted_counter:
        deleted_counts: dict[str, int] = {
            identifier: count for identifier, count in sorted(deleted_counter.items())
        }
        print("\nDeleted transcripts per identifier:")
        for identifier, count in deleted_counts.items():
            print(f"{identifier}: {count}")
    else:
        print("\nNo transcripts were deleted.")


if __name__ == "__main__":
    main()
