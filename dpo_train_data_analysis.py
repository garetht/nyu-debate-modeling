
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

    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(input_dir, filename)
            with open(filepath, "r") as f:
                try:
                    data = json.load(f)
                    for speech in data.get("speeches", []):
                        supplemental = speech.get("supplemental")
                        if supplemental:
                            preferences.append(supplemental.get("preference"))
                            if supplemental.get("rejected_responses"):
                                rejected_preferences.append(
                                    supplemental["rejected_responses"][0].get("preference")
                                )
                except json.JSONDecodeError:
                    print(f"Error decoding JSON from file: {filepath}")

    # Filter out None values
    preferences = [p for p in preferences if p is not None]
    rejected_preferences = [p for p in rejected_preferences if p is not None]

    if not preferences and not rejected_preferences:
        print("No preference data found in the specified directory.")
        return

    # Create histograms
    plt.figure(figsize=(10, 6))
    plt.hist(
        [preferences, rejected_preferences],
        bins=20,
        label=["Chosen", "Rejected"],
        color=["blue", "red"],
        alpha=0.7,
    )
    plt.xlabel("Preference Score")
    plt.ylabel("Frequency")
    plt.title("Distribution of Preference Scores")
    plt.legend()
    plt.grid(True)
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
