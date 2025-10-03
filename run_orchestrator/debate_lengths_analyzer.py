import argparse
from pathlib import Path

from transformers import AutoTokenizer

from run_orchestrator.transcript_model import read_transcripts_from_folder


def main():
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
    args = parser.parse_args()

    folder_path = Path(args.folder_path).resolve()

    # Determine output path
    if args.output_path is None:
        # Sanitize folder name for use as a filename
        safe_folder_name = "".join(c for c in folder_path.name if c.isalnum() or c in ('_', '-')).rstrip()
        output_path = f"{safe_folder_name}_debate_lengths.png"
    else:
        output_path = args.output_path

    transcripts = read_transcripts_from_folder(folder_path)

    if not transcripts:
        print(f"No transcripts found in {folder_path}. Exiting.")
        return

    debater_a_lengths = []
    debater_b_lengths = []

    for transcript in transcripts:
        for speech in transcript.speeches:
            if speech.speaker == "Debater_A":
                debater_a_lengths.append(len(speech.supplemental.response_tokens))
            elif speech.speaker == "Debater_B":
                debater_b_lengths.append(len(speech.supplemental.response_tokens))

    print(debater_a_lengths)
    print(debater_b_lengths)


if __name__ == '__main__':
    main()
