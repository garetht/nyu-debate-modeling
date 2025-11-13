#!/usr/bin/env python3
"""
Debate Statistics Analyzer

This script analyzes debate transcript files that conform to the provided JSON schema
and calculates various statistics about debate outcomes and judge accuracy.
It also generates histograms of probabilistic decisions for both debaters.
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Sequence, cast

import matplotlib.pyplot as plt
import numpy as np

from run_orchestrator.analysis.analysis_models.debate_stats import DebateStats
from run_orchestrator.analysis.transcript_model import Transcript

OutputFormat = Literal["stdout", "yaml", "json"]


@dataclass(frozen=True)
class DebateStatsAnalyzerArgs:
    """CLI arguments for the debate statistics analyzer."""

    directory_path: str
    output_format: OutputFormat


def analyze_debate_statistics(transcripts: Iterable[Transcript]) -> DebateStats:
    """Aggregate debate statistics from an iterable of transcripts."""
    stats = DebateStats()

    for transcript in transcripts:
        first_debater_correct = transcript.metadata.first_debater_correct

        for speech in transcript.speeches:
            supplemental = speech.supplemental
            if supplemental is None:
                continue

            decision = supplemental.decision
            if not decision:
                continue

            debater_a_win = decision == "Debater_A"
            debater_b_win = decision == "Debater_B"
            judge_correct = (
                (first_debater_correct and debater_a_win) or
                (not first_debater_correct and debater_b_win)
            )

            probabilistic_decision = supplemental.probabilistic_decision
            debater_a_prob = probabilistic_decision.debater_a if probabilistic_decision else None
            debater_b_prob = probabilistic_decision.debater_b if probabilistic_decision else None

            stats.add_debate(
                debater_a_win=debater_a_win,
                debater_b_win=debater_b_win,
                judge_correct=judge_correct,
                first_debater_correct=first_debater_correct,
                debater_a_prob=debater_a_prob,
                debater_b_prob=debater_b_prob,
            )
            break

    return stats


def load_json_file(file_path: Path) -> Dict[str, Any] | None:
    """Load and parse a JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON in {file_path}: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error reading file {file_path}: {e}", file=sys.stderr)
        return None


@dataclass
class DirectoryAnalysisResult:
    """Aggregated statistics and metadata for a directory scan."""

    directory: Path
    json_files: List[Path] = field(default_factory=list)
    overall_stats: DebateStats = field(default_factory=DebateStats)
    errors: List[str] = field(default_factory=list)


def _transcript_has_judge_decision(transcript: Transcript) -> bool:
    """Return True if any speech contains a judge decision."""
    for speech in transcript.speeches:
        supplemental = speech.supplemental
        if supplemental is not None and supplemental.decision:
            return True
    return False


def collect_directory_analysis(directory_path: Path | str) -> DirectoryAnalysisResult:
    """Collect debate statistics for all JSON transcripts in a directory."""
    dir_path = Path(directory_path)

    if not dir_path.exists():
        raise FileNotFoundError(f"Directory '{directory_path}' does not exist.")

    if not dir_path.is_dir():
        raise NotADirectoryError(f"'{directory_path}' is not a directory.")

    json_files = sorted(dir_path.glob("*.json"))
    result = DirectoryAnalysisResult(directory=dir_path, json_files=json_files)

    if not json_files:
        return result

    transcripts: List[Transcript] = []

    for file_path in json_files:
        data = load_json_file(file_path)
        if not data:
            result.errors.append(f"Failed to load {file_path}")
            continue

        try:
            transcript = Transcript.from_dict(data, file_path)
        except (AssertionError, KeyError, TypeError, ValueError) as exc:
            result.errors.append(f"Error analyzing {file_path}: {exc}")
            continue

        transcripts.append(transcript)

        if not _transcript_has_judge_decision(transcript):
            result.errors.append(f"No judge decision found in {file_path}")

    if transcripts:
        result.overall_stats = analyze_debate_statistics(transcripts)

    return result


def plot_probabilistic_histograms(
        debater_a_probs: List[float],
        debater_b_probs: List[float],
        *,
        announce: bool = True) -> None:
    """Generate histograms for probabilistic decisions."""
    plt.style.use('default')

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Probabilistic Decision Distributions', fontsize=16, fontweight='bold')

    # Debater A probability distribution
    if debater_a_probs:
        axes[0, 0].hist(debater_a_probs, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title(f'Debater A Probability Distribution (n={len(debater_a_probs)})')
        axes[0, 0].set_xlabel('Probability')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)

        # Add statistics
        mean_a = np.mean(debater_a_probs)
        std_a = np.std(debater_a_probs)
        axes[0, 0].axvline(mean_a, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_a:.3f}')
        axes[0, 0].legend()

        # Add text box with statistics
        stats_text = f'Mean: {mean_a:.3f}\nStd: {std_a:.3f}\nMin: {min(debater_a_probs):.3f}\nMax: {max(debater_a_probs):.3f}'
        axes[0, 0].text(0.02, 0.98, stats_text, transform=axes[0, 0].transAxes,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        axes[0, 0].text(0.5, 0.5, 'No Debater A\nProbability Data',
                        transform=axes[0, 0].transAxes, ha='center', va='center', fontsize=12)
        axes[0, 0].set_title('Debater A Probability Distribution')

    # Debater B probability distribution
    if debater_b_probs:
        axes[0, 1].hist(debater_b_probs, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[0, 1].set_title(f'Debater B Probability Distribution (n={len(debater_b_probs)})')
        axes[0, 1].set_xlabel('Probability')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)

        # Add statistics
        mean_b = np.mean(debater_b_probs)
        std_b = np.std(debater_b_probs)
        axes[0, 1].axvline(mean_b, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_b:.3f}')
        axes[0, 1].legend()

        # Add text box with statistics
        stats_text = f'Mean: {mean_b:.3f}\nStd: {std_b:.3f}\nMin: {min(debater_b_probs):.3f}\nMax: {max(debater_b_probs):.3f}'
        axes[0, 1].text(0.02, 0.98, stats_text, transform=axes[0, 1].transAxes,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        axes[0, 1].text(0.5, 0.5, 'No Debater B\nProbability Data',
                        transform=axes[0, 1].transAxes, ha='center', va='center', fontsize=12)
        axes[0, 1].set_title('Debater B Probability Distribution')

    # Combined histogram
    if debater_a_probs and debater_b_probs:
        axes[1, 0].hist(debater_a_probs, bins=20, alpha=0.6, color='skyblue',
                        label=f'Debater A (n={len(debater_a_probs)})', edgecolor='black')
        axes[1, 0].hist(debater_b_probs, bins=20, alpha=0.6, color='lightcoral',
                        label=f'Debater B (n={len(debater_b_probs)})', edgecolor='black')
        axes[1, 0].set_title('Combined Probability Distributions')
        axes[1, 0].set_xlabel('Probability')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    elif debater_a_probs:
        axes[1, 0].hist(debater_a_probs, bins=20, alpha=0.7, color='skyblue',
                        label=f'Debater A (n={len(debater_a_probs)})', edgecolor='black')
        axes[1, 0].set_title('Debater A Probability Distribution')
        axes[1, 0].set_xlabel('Probability')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    elif debater_b_probs:
        axes[1, 0].hist(debater_b_probs, bins=20, alpha=0.7, color='lightcoral',
                        label=f'Debater B (n={len(debater_b_probs)})', edgecolor='black')
        axes[1, 0].set_title('Debater B Probability Distribution')
        axes[1, 0].set_xlabel('Probability')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'No Probability Data\nAvailable',
                        transform=axes[1, 0].transAxes, ha='center', va='center', fontsize=12)
        axes[1, 0].set_title('Combined Probability Distributions')

    # Box plot comparison
    if debater_a_probs or debater_b_probs:
        box_data = []
        box_labels = []

        if debater_a_probs:
            box_data.append(debater_a_probs)
            box_labels.append('Debater A')
        if debater_b_probs:
            box_data.append(debater_b_probs)
            box_labels.append('Debater B')

        bp = axes[1, 1].boxplot(box_data, labels=box_labels, patch_artist=True)
        axes[1, 1].set_title('Probability Distribution Comparison')
        axes[1, 1].set_ylabel('Probability')
        axes[1, 1].grid(True, alpha=0.3)

        # Color the boxes
        colors = ['skyblue', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors[:len(box_data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    else:
        axes[1, 1].text(0.5, 0.5, 'No Probability Data\nAvailable',
                        transform=axes[1, 1].transAxes, ha='center', va='center', fontsize=12)
        axes[1, 1].set_title('Probability Distribution Comparison')

    plt.tight_layout()
    plt.savefig('debate_probabilistic_histograms.png', dpi=300, bbox_inches='tight')

    if announce:
        print(f"\nHistograms saved as 'debate_probabilistic_histograms.png'")

        # Print summary statistics
        print("\n" + "=" * 60)
        print("PROBABILISTIC DECISION SUMMARY")
        print("=" * 60)

        if debater_a_probs:
            print(f"Debater A Probabilities:")
            print(f"  Count: {len(debater_a_probs)}")
            print(f"  Mean: {np.mean(debater_a_probs):.4f}")
            print(f"  Std: {np.std(debater_a_probs):.4f}")
            print(f"  Min: {min(debater_a_probs):.4f}")
            print(f"  Max: {max(debater_a_probs):.4f}")
            print(f"  Median: {np.median(debater_a_probs):.4f}")

        if debater_b_probs:
            print(f"\nDebater B Probabilities:")
            print(f"  Count: {len(debater_b_probs)}")
            print(f"  Mean: {np.mean(debater_b_probs):.4f}")
            print(f"  Std: {np.std(debater_b_probs):.4f}")
            print(f"  Min: {min(debater_b_probs):.4f}")
            print(f"  Max: {max(debater_b_probs):.4f}")
            print(f"  Median: {np.median(debater_b_probs):.4f}")


def summarize_probabilities(probabilities: Sequence[float]) -> Dict[str, int | float] | None:
    """Return summary statistics for a collection of probabilities."""
    if not probabilities:
        return None

    return {
        "count": len(probabilities),
        "mean": float(np.mean(probabilities)),
        "std": float(np.std(probabilities)),
        "min": float(min(probabilities)),
        "max": float(max(probabilities)),
        "median": float(np.median(probabilities)),
    }


def build_output_payload(result: DirectoryAnalysisResult) -> Dict[str, Any]:
    """Construct a structured representation of the analysis results."""
    overall_stats: DebateStats = result.overall_stats
    histogram_available: bool = bool(overall_stats.debater_a_probs or overall_stats.debater_b_probs)

    payload: Dict[str, Any] = {
        "directory": str(result.directory),
        "json_files": [str(path) for path in result.json_files],
        "statistics": {
            "total_debates": overall_stats.total_debates,
            "debater_a_wins": overall_stats.debater_a_wins,
            "debater_b_wins": overall_stats.debater_b_wins,
            "judge_correct": overall_stats.judge_correct,
            "first_debater_correct": overall_stats.first_debater_correct,
            "percentages": overall_stats.get_percentages(),
            "judge_accuracy_standard_error": overall_stats.judge_standard_error,
        },
        "probabilities": {
            "debater_a": summarize_probabilities(overall_stats.debater_a_probs),
            "debater_b": summarize_probabilities(overall_stats.debater_b_probs),
        },
        "errors": result.errors,
        "histogram_image": "debate_probabilistic_histograms.png" if histogram_available else None,
    }

    return payload


def format_output(data: Dict[str, Any], output_format: OutputFormat) -> str:
    """Serialize the analysis payload according to the requested format."""
    if output_format == "json":
        return json.dumps(data, indent=2, sort_keys=False)

    if output_format == "yaml":
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML is required for YAML output.") from exc

        return cast(str, yaml.safe_dump(data, sort_keys=False))

    raise ValueError(f"Unsupported output format: {output_format}")


def print_stdout_summary(result: DirectoryAnalysisResult) -> None:
    """Render the analysis results to standard output."""
    json_files: List[Path] = result.json_files
    print(f"Found {len(json_files)} JSON files. Analyzing...")
    print("-" * 50)

    overall_stats: DebateStats = result.overall_stats

    print("\n" + "=" * 60)
    print("OVERALL STATISTICS")
    print("=" * 60)

    if overall_stats.total_debates > 0:
        percentages = overall_stats.get_percentages()
        judge_accuracy_standard_error: float = overall_stats.judge_standard_error
        print(f"Total debates analyzed: {overall_stats.total_debates}")
        print(f"Total files processed: {len(json_files)}")
        print()
        print(f"Debater A wins: {overall_stats.debater_a_wins} ({percentages['debater_a_win_rate']:.1f}%)")
        print(f"Debater B wins: {overall_stats.debater_b_wins} ({percentages['debater_b_win_rate']:.1f}%)")
        print(
            "Judge accuracy: "
            f"{overall_stats.judge_correct}/{overall_stats.total_debates} "
            f"({percentages['judge_accuracy']:.1f}% ± {judge_accuracy_standard_error:.1f}%)"
        )
        print(
            "First debater accuracy: "
            f"{overall_stats.first_debater_correct}/{overall_stats.total_debates} "
            f"({percentages['first_debater_accuracy']:.1f}%)"
        )
    else:
        print("No valid debates found to analyze.")

    if overall_stats.debater_a_probs or overall_stats.debater_b_probs:
        plot_probabilistic_histograms(overall_stats.debater_a_probs, overall_stats.debater_b_probs)

    if result.errors:
        print("\n" + "=" * 60)
        print("ERRORS ENCOUNTERED")
        print("=" * 60)
        for error in result.errors:
            print(f"- {error}")


def parse_args(argv: Sequence[str]) -> DebateStatsAnalyzerArgs:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze debate transcripts and summarize statistics."
    )
    parser.add_argument(
        "directory_path",
        type=str,
        help="Path to the directory containing debate transcript JSON files.",
    )
    parser.add_argument(
        "--output-format",
        choices=["stdout", "yaml", "json"],
        default="stdout",
        help="Select the output format for the analysis results.",
    )

    parsed_args = parser.parse_args(argv)
    output_format = cast(OutputFormat, parsed_args.output_format)

    return DebateStatsAnalyzerArgs(
        directory_path=parsed_args.directory_path,
        output_format=output_format,
    )


def run_analysis(args: DebateStatsAnalyzerArgs) -> int:
    """Execute the analysis workflow."""
    try:
        result: DirectoryAnalysisResult = collect_directory_analysis(args.directory_path)
    except (FileNotFoundError, NotADirectoryError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not result.json_files:
        message = f"No JSON files found in '{args.directory_path}'."
        if args.output_format == "stdout":
            print(message)
        else:
            payload = build_output_payload(result)
            payload["message"] = message
            try:
                output_text = format_output(payload, args.output_format)
            except RuntimeError as exc:
                print(f"Error: {exc}", file=sys.stderr)
                return 1
            print(output_text)
        return 0

    if args.output_format == "stdout":
        print_stdout_summary(result)
        return 0

    if result.overall_stats.debater_a_probs or result.overall_stats.debater_b_probs:
        plot_probabilistic_histograms(
            result.overall_stats.debater_a_probs,
            result.overall_stats.debater_b_probs,
            announce=False,
        )

    payload = build_output_payload(result)
    try:
        output_text = format_output(payload, args.output_format)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(output_text)
    return 0


def main() -> None:
    """Entry point for the debate statistics analyzer."""
    args = parse_args(sys.argv[1:])
    exit_code = run_analysis(args)
    if exit_code != 0:
        sys.exit(exit_code)

if __name__ == "__main__":
    main()
