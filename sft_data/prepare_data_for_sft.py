import argparse
import json
from pathlib import Path

def reformat_chat(in_path: Path, out_path: Path):
    """
    Converts JSONL with 'prompt'/'completion' into chat-style JSONL 
    where messages end with an assistant message.
    """
    with in_path.open("r", encoding="utf-8") as src, \
         out_path.open("w", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            prompt     = entry.get("prompt", "").rstrip("\n")
            completion = entry.get("completion", "").lstrip()

            # Build messages array: user -> assistant
            messages = [
                {"role": "user",      "content": prompt},
                {"role": "assistant", "content": completion}
            ]

            dst.write(json.dumps({"messages": messages}, ensure_ascii=False))
            dst.write("\n")

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Convert prompt/completion JSONL to chat-style JSONL for SFT."
    )
    p.add_argument(
        "--input", "-i", type=Path, required=True,
        help="Path to input JSONL with prompt/completion keys"
    )
    p.add_argument(
        "--output", "-o", type=Path, required=True,
        help="Path to output JSONL with messages array ending in assistant"
    )
    args = p.parse_args()
    reformat_chat(args.input, args.output)
    print(f"Wrote chat‐style SFT examples to {args.output}")
