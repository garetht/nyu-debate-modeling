
import json
import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, TypeVar, Callable, Type, cast, Optional
import matplotlib.pyplot as plt
import numpy as np

# This script is adapted from adjust_question_index.py
# It reads transcript files from a directory, counts debate identifiers,
# and generates a matplotlib graph and textual analysis.

T = TypeVar("T")


def from_bool(x: Any) -> bool:
    assert isinstance(x, bool)
    return x


def from_int(x: Any) -> int:
    assert isinstance(x, int) and not isinstance(x, bool)
    return x


def from_str(x: Any) -> str:
    assert isinstance(x, str)
    return x


def from_list(f: Callable[[Any], T], x: Any) -> List[T]:
    assert isinstance(x, list)
    return [f(y) for y in x]


@dataclass
class Metadatum:
    debate_identifier: str

    @staticmethod
    def from_dict(obj: Any) -> 'Metadatum':
        assert isinstance(obj, dict)
        debate_identifier = from_str(obj.get("debate_identifier"))
        return Metadatum(debate_identifier)


@dataclass
class Transcript:
    metadata: Metadatum

    @staticmethod
    def from_dict(obj: Any) -> 'Transcript':
        assert isinstance(obj, dict)
        metadata_obj = obj.get("metadata")
        # Handle cases where metadata might be a list with one item
        if isinstance(metadata_obj, list):
            if len(metadata_obj) > 0:
                metadata_obj = metadata_obj[0]
            else:
                # Handle empty metadata list, maybe return a default or raise error
                # For now, let's create a dummy one to avoid crashing
                metadata_obj = {"debate_identifier": "unknown_empty-metadata"}

        metadata = Metadatum.from_dict(metadata_obj)
        return Transcript(metadata)


def read_transcripts_from_folder(folder_path: Path) -> list[Transcript]:
    """Recursively reads all JSON files in a directory and returns a list of Transcript objects."""
    if not folder_path.is_dir():
        print(f"Error: '{folder_path}' is not a directory.")
        return []

    transcripts = []
    for file_path in folder_path.rglob('*.json'):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                transcripts.append(Transcript.from_dict(data))
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {file_path}. File will be skipped.")
        except (KeyError, AssertionError, TypeError) as e:
            print(f"Warning: Data structure validation failed for {file_path}. File will be skipped. Error: {type(e).__name__}: {e}")
        except Exception as e:
            print(f"Warning: An unexpected error occurred while reading {file_path}. File will be skipped. Error: {type(e).__name__}: {e}")
    return transcripts


def print_text_summary(debate_counts: defaultdict):
    """Prints a detailed textual analysis of the debate counts."""
    total_debates = sum(sum(q.values()) for q in debate_counts.values())
    if total_debates == 0:
        print("No debates to summarize.")
        return

    print("--- Debate Identifier Analysis ---")
    print(f"Total Debates Processed: {total_debates}\n")

    counts_to_ids = defaultdict(list)
    sorted_debates = sorted(debate_counts.keys())

    for debate_name in sorted_debates:
        debate_total = sum(debate_counts[debate_name].values())
        debate_perc = (debate_total / total_debates) * 100
        print(f'Debate Type: "{debate_name}"')
        print(f"  - Total Instances: {debate_total} ({debate_perc:.2f}% of all debates)")

        print("  - Question Breakdown (subquestions are denoted by '_' separation):")
        sorted_questions = sorted(debate_counts[debate_name].keys())
        for q_id in sorted_questions:
            count = debate_counts[debate_name][q_id]
            q_perc = (count / debate_total) * 100 if debate_total > 0 else 0
            counts_to_ids[count].append((debate_name, q_id))
            print(f"    - {q_id}: {count} instances ({q_perc:.2f}%)")
        print("-" * 30 + f"\n")

    print("\n--- Equality Statistics ---")
    print("Distribution of counts across all (Debate, Question) pairs:")
    sorted_counts = sorted(counts_to_ids.keys(), reverse=True)
    for count in sorted_counts:
        pairs = counts_to_ids[count]
        if len(pairs) > 1:
            print(f"  - Count of {count} occurred {len(pairs)} times for:")
            for d_name, q_id in pairs:
                print(f"    - ('{d_name}', '{q_id}')")


def save_faceted_graph(debate_counts: defaultdict, output_path: str):
    """Generates and saves a faceted grid of bar charts for the debate distribution."""
    debate_names = sorted(debate_counts.keys())
    if not debate_names:
        print("Cannot generate graph: no data to plot.")
        return

    # Determine grid size
    n_debates = len(debate_names)
    ncols = int(np.ceil(np.sqrt(n_debates)))
    nrows = int(np.ceil(n_debates / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4.5), squeeze=False)
    axes = axes.flatten()  # Flatten to 1D array for easy iteration

    for i, debate_name in enumerate(debate_names):
        ax = axes[i]
        questions = sorted(debate_counts[debate_name].keys())
        counts = [debate_counts[debate_name][q] for q in questions]

        if not questions:
            ax.set_title(f'Debate: {debate_name}\n(No questions found)', fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        y_pos = np.arange(len(questions))
        ax.barh(y_pos, counts, align='center', height=0.6)
        ax.set_yticks(y_pos)

        # Truncate labels for display
        max_len = 35
        truncated_labels = [q[:max_len-3] + '...' if len(q) > max_len else q for q in questions]
        ax.set_yticklabels(truncated_labels, fontsize=8)

        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel('Count')
        ax.set_title(f'Debate: {debate_name} ({sum(counts)} Qs)', fontsize=10)

        # Add count labels to bars
        for j, count in enumerate(counts):
            ax.text(count, j, f' {count}', va='center', ha='left', fontsize=8)

        ax.grid(axis='x', linestyle='--', alpha=0.6)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Hide unused subplots
    for i in range(n_debates, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle('Distribution of Questions per Debate Type', fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for suptitle
    plt.savefig(output_path)
    print(f"\nFaceted graph saved to {output_path}")

def save_normalized_summary_plot(debate_counts: defaultdict, output_path: str):
    """Generates a bar chart of the avg number of debates per question for each debate type."""
    debate_names = sorted(debate_counts.keys())
    if not debate_names:
        print("Cannot generate normalized summary plot: no data to plot.")
        return

    # Calculate the average debates per question for each type
    avg_debates_per_question = {}
    for name in debate_names:
        counts = debate_counts[name]
        total_debates = sum(counts.values())
        num_questions = len(counts)
        if num_questions > 0:
            avg_debates_per_question[name] = total_debates / num_questions
        else:
            avg_debates_per_question[name] = 0

    values = [avg_debates_per_question[name] for name in debate_names]

    fig, ax = plt.subplots(figsize=(max(10, len(debate_names) * 0.6), 7))

    bars = ax.bar(debate_names, values)

    ax.set_ylabel('Average Debates per Question Type')
    ax.set_title('Normalized Distribution Summary (Avg. Debates per Question)')
    ax.set_xticks(np.arange(len(debate_names)))
    ax.set_xticklabels(debate_names, rotation=45, ha="right")
    if values:
        ax.set_ylim(0, max(values) * 1.15)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Add labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}', ha='center', va='bottom')

    fig.tight_layout()
    plt.savefig(output_path)
    print(f"Normalized summary plot saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Graph and analyze the distribution of debate identifiers in a folder of transcripts."
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
        output_path = f"{safe_folder_name}_distribution.png"
    else:
        output_path = args.output_path

    transcripts = read_transcripts_from_folder(folder_path)

    if not transcripts:
        print(f"No transcripts found in {folder_path}. Exiting.")
        return

    debate_counts = defaultdict(lambda: defaultdict(int))
    for transcript in transcripts:
        if transcript.metadata:
            identifier = transcript.metadata.debate_identifier
            parts = identifier.split('_')
            if len(parts) >= 2:
                debate_name = parts[0]
                question_id = '_'.join(parts[1:])
                debate_counts[debate_name][question_id] += 1

    if not debate_counts:
        print("No valid debate identifiers found. Exiting.")
        return

    # Generate and print the text summary
    print_text_summary(debate_counts)

    # Generate and save the faceted graph
    save_faceted_graph(debate_counts, output_path)

    # Generate and save the normalized summary plot
    norm_summary_path = Path(output_path)
    norm_summary_path = norm_summary_path.with_name(f"{norm_summary_path.stem}_normalized_summary{norm_summary_path.suffix}")
    save_normalized_summary_plot(debate_counts, str(norm_summary_path))


if __name__ == "__main__":
    main()
