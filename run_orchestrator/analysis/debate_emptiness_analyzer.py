import argparse
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PosixPath
from typing import Iterable, Sequence

from run_orchestrator.analysis.analysis_models.debate_emptiness import DebateEmptinessAnalysis
from run_orchestrator.analysis.transcript_model import Transcript, read_transcripts_from_folder


@dataclass(frozen=True)
class DebateEmptinessCLIArgs:
    folder_path: Path
    delete: bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Graph and analyze the distribution of debate lengths in a folder of transcripts."
    )
    parser.add_argument(
        "folder_path",
        type=str,
        help="The path to the folder containing transcript JSON files.",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="If set, deletes the transcript files containing empty debates.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> DebateEmptinessCLIArgs:
    parser = build_parser()
    args = parser.parse_args(argv)
    folder_path = Path(args.folder_path).resolve()
    return DebateEmptinessCLIArgs(folder_path=folder_path, delete=args.delete)


def analyze_debate_emptiness(transcripts: Iterable[Transcript]) -> DebateEmptinessAnalysis:
    debater_a_empties: list[PosixPath] = []
    debater_b_empties: list[PosixPath] = []
    counter: Counter[str] = Counter()
    total_debates: int = 0

    for transcript in transcripts:
        total_debates += 1
        for speech in transcript.speeches:
            if len(speech.content) == 0:
                counter.update([transcript.metadata.debate_identifier])
                if speech.speaker == "Debater_A":
                    debater_a_empties.append(transcript.file_path)
                elif speech.speaker == "Debater_B":
                    debater_b_empties.append(transcript.file_path)

    unique_empty_files_paths = sorted({*debater_a_empties, *debater_b_empties}, key=str)
    return DebateEmptinessAnalysis(
        empty_speech_counts=dict(counter),
        debater_a_empty_files=[str(path) for path in debater_a_empties],
        debater_b_empty_files=[str(path) for path in debater_b_empties],
        unique_empty_files=[str(path) for path in unique_empty_files_paths],
        total_debates=total_debates,
    )


def main(argv: Sequence[str] | None = None) -> None:
    cli_args = parse_args(argv)

    transcripts = read_transcripts_from_folder(cli_args.folder_path)

    if not transcripts:
        print(f"No transcripts found in {cli_args.folder_path}. Exiting.")
        return

    analysis = analyze_debate_emptiness(transcripts)

    print(analysis.empty_speech_counts)
    print("Total number of unique empty debates")
    print(analysis.total_unique_empty_debates)
    print("Total number of debates")
    print(analysis.total_debates)

    if cli_args.delete:
        files_to_delete = analysis.unique_empty_files
        print(f"\n--delete flag is set. Deleting {len(files_to_delete)} files.")
        for file_path in files_to_delete:
            try:
                os.remove(file_path)
                print(f"Deleted {file_path}")
            except OSError as exc:
                print(f"Error deleting {file_path}: {exc}")


if __name__ == "__main__":
    main()
