import argparse
import json
import os

ADJUSTMENT_CONSTANT = 0.1

def adjust_preference_values(input_dir, undo=False):
    """
    Adjusts the preference values in DPO training data JSON files.

    Args:
        input_dir (str): The path to the directory containing the JSON files.
        undo (bool): If True, reverses the preference adjustment.
    """
    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(input_dir, filename)
            with open(filepath, "r") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON from file: {filepath}")
                    continue

            for speech in data.get("speeches", []):
                supplemental = speech.get("supplemental")
                if supplemental:
                    chosen_pref = supplemental.get("preference")
                    if supplemental.get("rejected_responses"):
                        rejected_pref = supplemental["rejected_responses"][0].get("preference")

                        if chosen_pref is not None and rejected_pref is not None:
                            if undo:
                                # Reverse the logic
                                if chosen_pref > rejected_pref:
                                    supplemental["preference"] = max(0.0, chosen_pref - ADJUSTMENT_CONSTANT)
                                    supplemental["rejected_responses"][0]["preference"] = min(1.0, rejected_pref + ADJUSTMENT_CONSTANT)
                                else:
                                    supplemental["preference"] = min(1.0, chosen_pref + ADJUSTMENT_CONSTANT)
                                    supplemental["rejected_responses"][0]["preference"] = max(0.0, rejected_pref - ADJUSTMENT_CONSTANT)
                            else:
                                # Original logic
                                if chosen_pref > rejected_pref:
                                    supplemental["preference"] = min(1.0, chosen_pref + ADJUSTMENT_CONSTANT)
                                    supplemental["rejected_responses"][0]["preference"] = max(0.0, rejected_pref - ADJUSTMENT_CONSTANT)
                                else:
                                    supplemental["preference"] = max(0.0, chosen_pref - ADJUSTMENT_CONSTANT)
                                    supplemental["rejected_responses"][0]["preference"] = min(1.0, rejected_pref + ADJUSTMENT_CONSTANT)

            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Adjust preference values in DPO training data."
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="The directory containing the JSON files.",
    )
    parser.add_argument(
        "--undo",
        action="store_true",
        help="Undo the preference adjustment.",
    )
    args = parser.parse_args()
    adjust_preference_values(args.input_dir, args.undo)
