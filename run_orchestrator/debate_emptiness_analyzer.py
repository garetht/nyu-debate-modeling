import argparse
import os
from collections import Counter
from pathlib import Path, PosixPath

from run_orchestrator.transcript_model import read_transcripts_from_folder


def main():
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

    args = parser.parse_args()

    folder_path = Path(args.folder_path).resolve()

    transcripts = read_transcripts_from_folder(folder_path)

    if not transcripts:
        print(f"No transcripts found in {folder_path}. Exiting.")
        return

    debater_a_empties: list[PosixPath] = []
    debater_b_empties: list[PosixPath] = []
    counter = Counter()

    for transcript in transcripts:
        for speech in transcript.speeches:
            length = len(speech.content)
            if length == 0:
                counter.update([transcript.metadata.debate_identifier])
                if speech.speaker == "Debater_A":
                    debater_a_empties.append(transcript.file_path)
                elif speech.speaker == "Debater_B":
                    debater_b_empties.append(transcript.file_path)

    print(counter)
    print(len(set(debater_a_empties + debater_b_empties)))

    if args.delete:
        files_to_delete = sorted(list(set(debater_a_empties + debater_b_empties)))
        print(f"\n--delete flag is set. Deleting {len(files_to_delete)} files.")
        for file_path in files_to_delete:
            try:
                os.remove(file_path)
                print(f"Deleted {file_path}")
            except OSError as e:
                print(f"Error deleting {file_path}: {e}")


if __name__ == "__main__":
    main()
