import argparse
import json
import pandas as pd

def convert_khan_csv_to_jsonl(csv_path: str, jsonl_path: str, n: int = 335):
    """
    Reads the Khan et al. debate CSV, selects N rows, and dumps each row’s
    `transcript` JSON into a newline-delimited JSONL file matching the format
    expected by Arnesen et al.’s QualityDebatesLoader.
    """
    # Load the CSV; ensure transcript stays as a string
    df = pd.read_csv(csv_path, dtype={"transcript": "string"})
    # Filter to standard human_judge debates and sample N
    df = df[df['debater_config'] == 'experiments/human_judge/agent_configs/debaters/standard.yaml'].sample(n)

    with open(jsonl_path, "w", encoding="utf-8") as out_f:
        for idx, row in df.iterrows():
            # Parse the transcript JSON string into a Python dict
            transcript_obj = json.loads(row["transcript"])

            # Inject the original debate_id as debateId
            transcript_obj["debateId"] = row["debate_id"]

            # Rename story_title to storyTitle if present
            if "story_title" in transcript_obj:
                transcript_obj["storyTitle"] = transcript_obj.pop("story_title")

            # Convert answers dict to a list [correct, incorrect]
            if isinstance(transcript_obj.get("answers"), dict):
                answers = transcript_obj["answers"]
                transcript_obj["answers"] = [
                    answers.get("correct"),
                    answers.get("incorrect"),
                ]

            # Rename rounds to turns
            if "rounds" in transcript_obj:
                transcript_obj["turns"] = transcript_obj.pop("rounds")

            # Write out a single JSON object per line
            out_f.write(json.dumps(transcript_obj, ensure_ascii=False))
            out_f.write("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Khan et al. CSV debates into Arnesen JSONL format"
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to the llm_debate_human_judge_dataset.csv from Khan et al.",
    )
    parser.add_argument(
        "--jsonl",
        required=True,
        help="Output path for debates-readable.jsonl for Arnesen et al.’s loader",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=335,
        help="Number of rows to sample from the filtered CSV",
    )
    args = parser.parse_args()
    convert_khan_csv_to_jsonl(args.csv, args.jsonl, n=args.n)