import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from run_orchestrator.analysis.analysis_models.debate_lengths import DebateLengthAnalysis
from run_orchestrator.analysis.transcript_model import Transcript, iter_transcripts_from_folder


@dataclass(frozen=True)
class DebateLengthArgs:
    folder_path: Path
    output_path: Path


def parse_args() -> DebateLengthArgs:
    parser = argparse.ArgumentParser(
        description="Graph and analyze the distribution of debate lengths in a folder of transcripts."
    )
    parser.add_argument(
        "folder_path",
        type=str,
        help="The path to the folder containing transcript JSON files."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="The path to save the output graph PNG file. If not provided, it defaults to a name derived from the input folder."
    )
    parsed_args = parser.parse_args()

    folder_path = Path(parsed_args.folder_path).resolve()

    if parsed_args.output_path is None:
        safe_folder_name = "".join(
            character for character in folder_path.name if character.isalnum() or character in ('_', '-')
        ).rstrip()
        output_path = Path(f"{safe_folder_name}_debate_lengths.png").resolve()
    else:
        output_path = Path(parsed_args.output_path).resolve()

    return DebateLengthArgs(folder_path=folder_path, output_path=output_path)


def analyze_debate_lengths(transcripts: Iterable[Transcript]) -> DebateLengthAnalysis:
    debater_a_lengths: list[int] = []
    debater_b_lengths: list[int] = []
    transcript_count: int = 0

    for transcript in transcripts:
        transcript_count += 1
        for speech in transcript.speeches:
            if speech.speaker == "Debater_A":
                debater_a_lengths.append(len(speech.supplemental.response_tokens))
            elif speech.speaker == "Debater_B":
                debater_b_lengths.append(len(speech.supplemental.response_tokens))

    return DebateLengthAnalysis(
        debater_a_lengths=tuple(debater_a_lengths),
        debater_b_lengths=tuple(debater_b_lengths),
        transcript_count=transcript_count
    )


def main() -> None:
    args = parse_args()

    analysis = analyze_debate_lengths(iter_transcripts_from_folder(args.folder_path))

    if analysis.transcript_count == 0:
        print(f"No transcripts found in {args.folder_path}. Exiting.")
        return

    print(analysis.debater_a_lengths)
    print(analysis.debater_b_lengths)


if __name__ == "__main__":
    main()
