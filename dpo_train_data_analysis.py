
import argparse
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def analyze_dpo_data(input_dir):
    """
    Analyzes DPO training data from a directory of JSON files.

    Args:
        input_dir (str): The path to the directory containing the JSON files.
    """
    preferences = []
    rejected_preferences = []
    margins = []

    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(input_dir, filename)
            with open(filepath, "r") as f:
                try:
                    data = json.load(f)
                    for speech in data.get("speeches", []):
                        supplemental = speech.get("supplemental")
                        if supplemental:
                            chosen_pref = supplemental.get("preference")
                            preferences.append(chosen_pref)
                            if supplemental.get("rejected_responses"):
                                rejected_pref = supplemental["rejected_responses"][
                                    0
                                ].get("preference")
                                rejected_preferences.append(rejected_pref)
                                if (
                                    chosen_pref is not None
                                    and rejected_pref is not None
                                ):
                                    margins.append(chosen_pref - rejected_pref)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON from file: {filepath}")

    # Filter out None values
    preferences = [p for p in preferences if p is not None]
    rejected_preferences = [p for p in rejected_preferences if p is not None]

    if not preferences and not rejected_preferences:
        print("No preference data found in the specified directory.")
        return

    # Create a figure and a set of subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # First plot: Distribution of Preference Scores
    ax1.hist([preferences, rejected_preferences], bins=20, label=["Chosen", "Rejected"], color=["blue", "red"], alpha=0.7)
    ax1.set_xlabel("Preference Score")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Distribution of Preference Scores")
    ax1.legend()
    ax1.grid(True)

    # Second plot: Distribution of Preference Margins
    if margins:
        ax2.hist(margins, bins=100, color="green", alpha=0.7)
        ax2.set_xlabel("Preference Margin (Chosen - Rejected)")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Distribution of Preference Margins")
        ax2.grid(True)
        ax2.set_xlim(0.0, 0.2)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze DPO training data and plot preference histograms."
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="The directory containing the JSON files.",
    )
    args = parser.parse_args()
    analyze_dpo_data(args.input_dir)
