import argparse
import json
import pandas as pd

def normalize_turns(raw_turns):
    """
    Convert a list of raw turns (either already normalized or in "rounds" style)
    into the schema expected by QualityDebatesLoader.
    """
    turns = []
    for t in raw_turns:
        if "role" in t:
            # Already normalized
            turns.append({
                "role": t["role"],
                "index": t.get("index", -1),
                "text": t.get("text", ""),
                "probabilities": t.get("probabilities")
            })
        else:
            # Convert "rounds"-style entries
            if t.get("correct"):
                turns.append({"role": "Debater", "index": 0, "text": t["correct"]})
            if t.get("incorrect"):
                turns.append({"role": "Debater", "index": 1, "text": t["incorrect"]})
            if t.get("judge") is not None:
                turns.append({
                    "role": "Judge",
                    "index": -1,
                    "text": t["judge"],
                    "probabilities": None
                })
    return turns


def process_csv(csv_path: str, n: int):
    """
    Read CSV, sample n rows, convert each transcript into a dict with required keys.
    """
    df = pd.read_csv(csv_path, dtype={"transcript": "string"})
    #df = df[df['debater_config'] ==
    #        'experiments/human_judge/agent_configs/debaters/standard.yaml'].sample(n)
    df = df [ df['experiment_name'] == 'Training Phase']
    print(f'n. of debates from csv: {len(df)}')

    entries = []
    for _, row in df.iterrows():
        obj = json.loads(row["transcript"])
        # Inject metadata, casting debateId to str to avoid join errors
        obj["debateId"] = str(row["debate_id"])
        obj["storyTitle"] = obj.get("story_title", obj.get("storyTitle"))
        # Convert answers
        ans = obj.get("answers", {})
        answers_list = [ans.get("correct"), ans.get("incorrect")]
        obj["answers"] = answers_list
        correct_idx = 0 if row["correct"] else 1
        obj["correctAnswer"] = answers_list[correct_idx]
        # Build normalized turns
        turns = [
            {"role": "Debater", "index": 0, "text": answers_list[0]},
            {"role": "Debater", "index": 1, "text": answers_list[1]},
            {"role": "Judge",   "index": -1,  "text": "", "probabilities": [
                1.0 if correct_idx == 0 else 0.0,
                0.0 if correct_idx == 0 else 1.0
            ]}
        ]
        obj["turns"] = turns
        # Clean up legacy keys
        obj.pop("story_title", None)
        obj.pop("rounds", None)
        entries.append(obj)
    return entries


def load_original(jsonl_path: str):
    """
    Read the original debates-readable.jsonl into a list of dicts.
    """
    entries = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # Ensure debateId is a string too
            if 'debateId' in obj:
                obj['debateId'] = str(obj['debateId'])
            entries.append(obj)
    return entries


def normalize_entries(entries: list):
    """
    Ensure each entry has a properly normalized 'turns' list.
    """
    normed = []
    for obj in entries:
        raw_turns = obj.get("turns") or obj.get("rounds", [])
        obj["turns"] = normalize_turns(raw_turns)
        obj.pop("rounds", None)
        normed.append(obj)
    return normed


def main():
    parser = argparse.ArgumentParser(
        description="Convert CSV, concatenate with original JSONL, and normalize for Arnesen loader"
    )
    parser.add_argument("--csv", required=True, help="Path to Khan et al. CSV file")
    parser.add_argument("--original", required=True, help="Path to original debates-readable.jsonl")
    parser.add_argument("--out", required=True, help="Output path for merged and normalized JSONL")
    parser.add_argument("--n", type=int, default=335, help="Number of CSV rows to sample")
    args = parser.parse_args()

    # Process CSV and original files
    csv_entries = process_csv(args.csv, args.n)
    orig_entries = load_original(args.original)

    # Combine and normalize
    combined = orig_entries + csv_entries
    normed = normalize_entries(combined)

    # Write final JSONL
    with open(args.out, 'w', encoding='utf-8') as out_f:
        for obj in normed:
            out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")

if __name__ == '__main__':
    main()