import argparse
import json
import os
import shutil

DEFAULT_PREFERENCE_THRESHOLD = 0.1

def filter_preference_values(input_dir, preference_threshold):
    """
    Filters DPO training data based on preference value differences.

    Args:
        input_dir (str): The path to the directory containing the JSON files.
        preference_threshold (float): The minimum difference in preference values to keep a speech.
    """
    # Validate input directory structure
    path_parts = os.path.normpath(input_dir).split(os.sep)
    if not (len(path_parts) == 4 and path_parts[0] == 'outputs' and path_parts[2] == 'outputs' and path_parts[3] == 'transcripts'):
        print(f"Error: Input directory '{input_dir}' is not in the expected format 'outputs/<some name>/outputs/transcripts'")
        return

    # Create output directory
    output_dir = os.path.join(path_parts[0], f"{path_parts[1]}-significant", path_parts[2], path_parts[3])
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(input_dir, filename)
            output_filepath = os.path.join(output_dir, filename)
            
            with open(filepath, "r") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON from file: {filepath}")
                    continue

            if "speeches" in data:
                original_speech_count = len(data["speeches"])
                filtered_speeches = []
                for speech in data.get("speeches", []):
                    supplemental = speech.get("supplemental")
                    if supplemental:
                        preference = supplemental.get("preference")
                        rejected_responses = supplemental.get("rejected_responses")
                        if (preference is not None and 
                            rejected_responses and 
                            isinstance(rejected_responses, list) and 
                            len(rejected_responses) > 0 and 
                            "preference" in rejected_responses[0]):
                            
                            rejected_preference = rejected_responses[0].get("preference")
                            if rejected_preference is not None:
                                if abs(preference - rejected_preference) >= preference_threshold:
                                    filtered_speeches.append(speech)
                                else:
                                    continue # Skip this speech
                            else:
                                filtered_speeches.append(speech)
                        else:
                            filtered_speeches.append(speech)
                    else:
                        filtered_speeches.append(speech)
                
                data["speeches"] = filtered_speeches
                if len(filtered_speeches) < original_speech_count:
                    print(f"Filtered {original_speech_count - len(filtered_speeches)} speeches from {filename}")

            with open(output_filepath, "w") as f:
                json.dump(data, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter preference values in DPO training data."
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="The directory containing the JSON files. Must be in the format 'outputs/<some name>/outputs/transcripts'",
    )
    parser.add_argument(
        "--preference_threshold",
        type=float,
        default=DEFAULT_PREFERENCE_THRESHOLD,
        help=f"The minimum difference in preference values to keep a speech (default: {DEFAULT_PREFERENCE_THRESHOLD}).",
    )
    args = parser.parse_args()
    filter_preference_values(args.input_dir, args.preference_threshold)