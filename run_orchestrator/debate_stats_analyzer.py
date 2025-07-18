#!/usr/bin/env python3
"""
Debate Statistics Analyzer

This script analyzes debate transcript files that conform to the provided JSON schema
and calculates various statistics about debate outcomes and judge accuracy.
It also generates histograms of probabilistic decisions for both debaters.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class DebateStats:
    """Container for debate statistics"""
    total_debates: int = 0
    debater_a_wins: int = 0
    debater_b_wins: int = 0
    judge_correct: int = 0
    first_debater_correct: int = 0
    debater_a_probs: List[float] = None
    debater_b_probs: List[float] = None

    def __post_init__(self):
        if self.debater_a_probs is None:
            self.debater_a_probs = []
        if self.debater_b_probs is None:
            self.debater_b_probs = []

    def add_debate(self, debater_a_win: bool, debater_b_win: bool,
                   judge_correct: bool, first_debater_correct: bool,
                   debater_a_prob: float = None, debater_b_prob: float = None):
        """Add results from a single debate"""
        self.total_debates += 1
        if debater_a_win:
            self.debater_a_wins += 1
        elif debater_b_win:
            self.debater_b_wins += 1

        if judge_correct:
            self.judge_correct += 1
        if first_debater_correct:
            self.first_debater_correct += 1

        # Store probabilistic decisions
        if debater_a_prob is not None:
            self.debater_a_probs.append(debater_a_prob)
        if debater_b_prob is not None:
            self.debater_b_probs.append(debater_b_prob)

    def get_percentages(self) -> Dict[str, float]:
        """Calculate percentage statistics"""
        if self.total_debates == 0:
            return {
                "debater_a_win_rate": 0.0,
                "debater_b_win_rate": 0.0,
                "judge_accuracy": 0.0,
                "first_debater_accuracy": 0.0
            }

        return {
            "debater_a_win_rate": (self.debater_a_wins / self.total_debates) * 100,
            "debater_b_win_rate": (self.debater_b_wins / self.total_debates) * 100,
            "judge_accuracy": (self.judge_correct / self.total_debates) * 100,
            "first_debater_accuracy": (self.first_debater_correct / self.total_debates) * 100
        }


def load_json_file(file_path: Path) -> Dict:
    """Load and parse a JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON in {file_path}: {e}")
        return None
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None


def analyze_debate_file(file_path: Path) -> Tuple[DebateStats, List[str]]:
    """Analyze a single debate transcript file"""
    stats = DebateStats()
    errors = []

    data = load_json_file(file_path)
    if not data:
        errors.append(f"Failed to load {file_path}")
        return stats, errors

    try:
        # Extract metadata
        metadata = data.get('metadata', {})
        first_debater_correct = metadata.get('first_debater_correct')

        if first_debater_correct is None:
            errors.append(f"Missing 'first_debater_correct' in metadata for {file_path}")
            return stats, errors

        # Extract speeches and find judge decisions
        speeches = data.get('speeches', [])

        for speech in speeches:
            supplemental = speech.get('supplemental')
            if supplemental and supplemental.get('decision'):
                decision = supplemental['decision']

                # Determine winners
                debater_a_win = decision == "Debater_A"
                debater_b_win = decision == "Debater_B"

                # Determine if judge is correct
                judge_correct = (
                        (first_debater_correct and decision == "Debater_A") or
                        (not first_debater_correct and decision == "Debater_B")
                )

                # Extract probabilistic decisions
                prob_decision = supplemental.get('probabilistic_decision')
                debater_a_prob = None
                debater_b_prob = None

                if prob_decision:
                    debater_a_prob = prob_decision.get('Debater_A')
                    debater_b_prob = prob_decision.get('Debater_B')

                # Add to statistics
                stats.add_debate(
                    debater_a_win=debater_a_win,
                    debater_b_win=debater_b_win,
                    judge_correct=judge_correct,
                    first_debater_correct=first_debater_correct,
                    debater_a_prob=debater_a_prob,
                    debater_b_prob=debater_b_prob
                )

                # We found a decision, so we can break
                break
        else:
            # No decision found in any speech
            errors.append(f"No judge decision found in {file_path}")

    except Exception as e:
        errors.append(f"Error analyzing {file_path}: {e}")

    return stats, errors


def analyze_directory(directory_path: str) -> None:
    """Analyze all JSON files in a directory"""
    dir_path = Path(directory_path)

    if not dir_path.exists():
        print(f"Error: Directory '{directory_path}' does not exist.")
        return

    if not dir_path.is_dir():
        print(f"Error: '{directory_path}' is not a directory.")
        return

    # Find all JSON files
    json_files = list(dir_path.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in '{directory_path}'.")
        return

    print(f"Found {len(json_files)} JSON files. Analyzing...")
    print("-" * 50)

    # Initialize overall statistics
    overall_stats = DebateStats()
    all_errors = []

    # Process each file
    for file_path in json_files:
        file_stats, errors = analyze_debate_file(file_path)

        # Add to overall statistics
        overall_stats.total_debates += file_stats.total_debates
        overall_stats.debater_a_wins += file_stats.debater_a_wins
        overall_stats.debater_b_wins += file_stats.debater_b_wins
        overall_stats.judge_correct += file_stats.judge_correct
        overall_stats.first_debater_correct += file_stats.first_debater_correct
        overall_stats.debater_a_probs.extend(file_stats.debater_a_probs)
        overall_stats.debater_b_probs.extend(file_stats.debater_b_probs)

        # Collect errors
        all_errors.extend(errors)

        # # Print per-file statistics
        # if file_stats.total_debates > 0:
        #     percentages = file_stats.get_percentages()
        #     print(f"\n{file_path.name}:")
        #     print(f"  Total debates: {file_stats.total_debates}")
        #     print(f"  Debater A wins: {file_stats.debater_a_wins} ({percentages['debater_a_win_rate']:.1f}%)")
        #     print(f"  Debater B wins: {file_stats.debater_b_wins} ({percentages['debater_b_win_rate']:.1f}%)")
        #     print(f"  Judge accuracy: {file_stats.judge_correct}/{file_stats.total_debates} ({percentages['judge_accuracy']:.1f}%)")
        #     print(f"  First debater correct: {file_stats.first_debater_correct}/{file_stats.total_debates} ({percentages['first_debater_accuracy']:.1f}%)")

    # Print overall statistics
    print("\n" + "=" * 60)
    print("OVERALL STATISTICS")
    print("=" * 60)

    if overall_stats.total_debates > 0:
        percentages = overall_stats.get_percentages()
        print(f"Total debates analyzed: {overall_stats.total_debates}")
        print(f"Total files processed: {len(json_files)}")
        print()
        print(f"Debater A wins: {overall_stats.debater_a_wins} ({percentages['debater_a_win_rate']:.1f}%)")
        print(f"Debater B wins: {overall_stats.debater_b_wins} ({percentages['debater_b_win_rate']:.1f}%)")
        print(f"Judge accuracy: {overall_stats.judge_correct}/{overall_stats.total_debates} ({percentages['judge_accuracy']:.1f}%)")
        print(f"First debater accuracy: {overall_stats.first_debater_correct}/{overall_stats.total_debates} ({percentages['first_debater_accuracy']:.1f}%)")
    else:
        print("No valid debates found to analyze.")

    # Generate histograms for probabilistic decisions
    if overall_stats.debater_a_probs or overall_stats.debater_b_probs:
        plot_probabilistic_histograms(overall_stats.debater_a_probs, overall_stats.debater_b_probs)

    # Print errors if any
    if all_errors:
        print("\n" + "=" * 60)
        print("ERRORS ENCOUNTERED")
        print("=" * 60)
        for error in all_errors:
            print(f"- {error}")


def plot_probabilistic_histograms(debater_a_probs: List[float], debater_b_probs: List[float]) -> None:
    """Generate histograms for probabilistic decisions"""
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


def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python debate_stats_analyzer.py <directory_path>")
        print("Example: python debate_stats_analyzer.py ./debate_transcripts/")
        print("\nRequires matplotlib and numpy for histogram generation:")
        print("  pip install matplotlib numpy")
        sys.exit(1)

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as e:
        print(f"Error: Required libraries not installed. {e}")
        print("Please install with: pip install matplotlib numpy")
        sys.exit(1)

    directory_path = sys.argv[1]
    analyze_directory(directory_path)


if __name__ == "__main__":
    main()
