import argparse
from collections import Counter
from pathlib import Path
from typing import Iterable

from run_orchestrator.analysis.analysis_models.debate_distribution import DebateDistributionArgs, DebateIdentifierCount, \
    TitleCount, DebateDistributionAnalysis
from run_orchestrator.analysis.transcript_model import Transcript, iter_transcripts_from_folder


def parse_args() -> DebateDistributionArgs:
    parser = argparse.ArgumentParser(
        description="Analyze the distribution of debate identifiers in a folder of transcripts."
    )
    parser.add_argument(
        "folder_path",
        type=str,
        help="The path to the folder containing transcript JSON files."
    )
    parsed_args = parser.parse_args()

    folder_path = Path(parsed_args.folder_path).resolve()

    return DebateDistributionArgs(folder_path=folder_path)


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

    identifier_counts_list: list[DebateIdentifierCount] = []
    for identifier, count in sorted(identifier_counter.items()):
        title, topic = _split_identifier(identifier)
        identifier_counts_list.append(
            DebateIdentifierCount(
                identifier=identifier,
                title=title,
                topic=topic,
                count=count,
            )
        )

    identifier_counts = tuple(identifier_counts_list)
    title_counts = tuple(
        TitleCount(title=title, count=count)
        for title, count in sorted(title_counter.items())
    )

    return DebateDistributionAnalysis(
        identifier_counts=identifier_counts,
        title_counts=title_counts,
        transcript_count=transcript_count,
    )


def main() -> None:
    args = parse_args()

    analysis = analyze_debate_distribution(iter_transcripts_from_folder(args.folder_path))

    if analysis.transcript_count == 0:
        print(f"No transcripts found in {args.folder_path}. Exiting.")
        return

    for identifier_count in analysis.identifier_counts:
        print(f"{identifier_count.identifier}: {identifier_count.count}")
        print(f"  title={identifier_count.title}, topic={identifier_count.topic}")

    print("\nCounts by title:")
    for title_count in analysis.title_counts:
        print(f"{title_count.title}: {title_count.count}")


if __name__ == "__main__":
    main()
