import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from run_orchestrator.analysis.analysis_models.debate_uniqueness import DebateUniquenessAnalysis
from run_orchestrator.analysis.transcript_model import Transcript, read_transcripts_from_folder


@dataclass(frozen=True)
class DebateUniquenessArgs:
    folder_path: Path
    delete_duplicates: bool
    minimum_total_count: int


def parse_args() -> DebateUniquenessArgs:
    parser = argparse.ArgumentParser(
        description="Preserve only debates that are unique by a certain key, keeping a minimum number of total files."
    )
    parser.add_argument(
        "folder_path",
        type=str,
        help="The path to the folder containing transcript JSON files."
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="If set, deletes the transcript files that are not unique."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="The minimum number of total debates to preserve."
    )
    parsed_args = parser.parse_args()

    folder_path = Path(parsed_args.folder_path).resolve()

    return DebateUniquenessArgs(
        folder_path=folder_path,
        delete_duplicates=parsed_args.delete,
        minimum_total_count=parsed_args.count
    )


def analyze_debate_uniqueness(transcripts: Iterable[Transcript]) -> DebateUniquenessAnalysis:
    unique_identifiers_seen: set[str] = set()
    unique_identifiers_in_order: list[str] = []
    duplicate_file_paths: list[Path] = []
    total_transcripts: int = 0

    for transcript in transcripts:
        total_transcripts += 1
        identifier = transcript.metadata.debate_identifier
        if identifier in unique_identifiers_seen:
            duplicate_file_paths.append(Path(transcript.file_path))
        else:
            unique_identifiers_seen.add(identifier)
            unique_identifiers_in_order.append(identifier)

    return DebateUniquenessAnalysis(
        unique_identifiers=tuple(unique_identifiers_in_order),
        duplicate_file_paths=tuple(duplicate_file_paths),
        total_transcripts=total_transcripts
    )


def main() -> None:
    args = parse_args()

    transcripts = read_transcripts_from_folder(args.folder_path)

    analysis = analyze_debate_uniqueness(transcripts)

    if analysis.total_transcripts == 0:
        print(f"No transcripts found in {args.folder_path}. Exiting.")
        return

    num_unique = len(analysis.unique_identifiers)
    num_duplicates = len(analysis.duplicate_file_paths)

    files_to_actually_delete: list[Path] = []
    num_kept_if_all_duplicates_deleted = num_unique

    if num_kept_if_all_duplicates_deleted >= args.minimum_total_count:
        files_to_actually_delete = list(analysis.duplicate_file_paths)
    else:
        num_duplicates_to_keep = args.minimum_total_count - num_kept_if_all_duplicates_deleted
        duplicate_paths = list(analysis.duplicate_file_paths)
        if num_duplicates_to_keep < len(duplicate_paths):
            files_to_actually_delete = duplicate_paths[num_duplicates_to_keep:]

    num_to_delete = len(files_to_actually_delete)
    num_to_keep = analysis.total_transcripts - num_to_delete

    print(
        f"Found {num_unique} unique debates out of {analysis.total_transcripts} total transcripts."
    )
    print(f"Found {num_duplicates} non-unique debates.")
    print(
        f"With --count {args.minimum_total_count}, planning to keep {num_to_keep} and delete {num_to_delete} transcripts."
    )

    if args.delete_duplicates:
        print(
            f"\n--delete flag is set. Deleting {num_to_delete} non-unique transcript files."
        )
        if not files_to_actually_delete:
            print("Nothing to delete.")
        else:
            for file_path in files_to_actually_delete:
                try:
                    os.remove(file_path)
                    print(f"Deleted {file_path}")
                except OSError as error:
                    print(f"Error deleting {file_path}: {error}")


if __name__ == "__main__":
    main()
