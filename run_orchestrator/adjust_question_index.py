# %%

import json
from dataclasses import dataclass
from pathlib import Path

from run_orchestrator.transcript_model import Metadatum, Transcript


def renumber_question_indices(transcripts: list[Transcript]) -> list[Transcript]:
    grouped_metadata: dict[str, list[Metadatum]] = {}

    for transcript in transcripts:
        for metadatum in transcript.metadata:
            if metadatum.debate_identifier not in grouped_metadata:
                grouped_metadata[metadatum.debate_identifier] = []
            grouped_metadata[metadatum.debate_identifier].append(metadatum)

    for (index, (identifier, metadata)) in enumerate(list(grouped_metadata.items())):
        for metadatum in metadata:
            metadatum.question_idx = index
    return transcripts


def read_jsons() -> list[tuple[Transcript, Path]]:
    # Replace 'data.json' with your actual file path
    directory_path = 'outputs/DataGenerationLlama3MultiRoundHalfBranchedFullTrainDPO/outputs/transcripts'
    directory = Path(directory_path)

    if not directory.exists():
        print(f"Error: Directory '{directory_path}' does not exist")
        return []

    if not directory.is_dir():
        print(f"Error: '{directory_path}' is not a directory")
        return []

    jsons_with_paths = []
    for file_path in directory.iterdir():
        if file_path.is_file():
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.loads(file.read())
                jsons_with_paths.append((Transcript.from_dict(data), file_path))

    return jsons_with_paths


def write_transcripts(transcripts_with_paths: list[tuple[Transcript, Path]]):
    for transcript, file_path in transcripts_with_paths:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(transcript.to_dict(), file, indent=2)
    print(f"Successfully wrote {len(transcripts_with_paths)} transcripts back to their original locations.")


def main():
    jsons_with_paths = read_jsons()
    transcripts = [t for t, _ in jsons_with_paths]
    renumbered_transcripts = renumber_question_indices(transcripts)
    # Re-associate renumbered transcripts with their original paths
    renumbered_jsons_with_paths = [(t, p) for t, p in
                                   zip(renumbered_transcripts, [path for _, path in jsons_with_paths])]
    write_transcripts(renumbered_jsons_with_paths)


if __name__ == "__main__":
    main()
