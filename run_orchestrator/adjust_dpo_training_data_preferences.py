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
                            # Determine the adjustment direction
                            higher_pref, lower_pref = (chosen_pref, rejected_pref) if chosen_pref > rejected_pref else (rejected_pref, chosen_pref)
                            
                            if undo:
                                # Check if undo is possible
                                if higher_pref >= ADJUSTMENT_CONSTANT and lower_pref <= 1.0 - ADJUSTMENT_CONSTANT:
                                    new_higher_pref = higher_pref - ADJUSTMENT_CONSTANT
                                    new_lower_pref = lower_pref + ADJUSTMENT_CONSTANT
                                else:
                                    continue # Skip if not reversible
                            else:
                                # Check if forward adjustment is possible
                                if higher_pref <= 1.0 - ADJUSTMENT_CONSTANT and lower_pref >= ADJUSTMENT_CONSTANT:
                                    new_higher_pref = higher_pref + ADJUSTMENT_CONSTANT
                                    new_lower_pref = lower_pref - ADJUSTMENT_CONSTANT
                                else:
                                    continue # Skip if not reversible

                            # Apply the changes
                            if chosen_pref > rejected_pref:
                                supplemental["preference"] = new_higher_pref
                                supplemental["rejected_responses"][0]["preference"] = new_lower_pref
                            else:
                                supplemental["preference"] = new_lower_pref
                                supplemental["rejected_responses"][0]["preference"] = new_higher_pref

            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reversibly adjust preference values in DPO training data."
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
