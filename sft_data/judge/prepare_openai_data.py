import json
from pathlib import Path

INPUT_PATH = Path("combined_for_judge.jsonl")
OUTPUT_PATH = Path("ready_for_openai_sft.jsonl")
SPLIT_MARKER = "Here is the topic they will be debating"

def split_prompt(raw_prompt: str):
    """
    Split raw_prompt at the first occurrence of SPLIT_MARKER.
    Everything before (excluding the marker) is the system content.
    The user content starts with the marker and includes the rest.
    If marker not found, fall back to a default system message and treat entire prompt as user.
    """
    idx = raw_prompt.find(SPLIT_MARKER)
    if idx == -1:
        # fallback
        system = "You are a helpful, precise AI assistant."
        user = raw_prompt
    else:
        system = raw_prompt[:idx].strip()
        user = raw_prompt[idx:].strip()
    return system, user

def transform_example(orig: dict):
    if "prompt" not in orig or "completion" not in orig:
        raise ValueError("Each line must contain 'prompt' and 'completion' fields.")
    raw_prompt = orig["prompt"]
    completion = orig["completion"]

    system_content, user_content = split_prompt(raw_prompt)

    # Ensure assistant content has a leading space to separate from preceding text
    if not completion.startswith(" "):
        completion = " " + completion

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": completion},
    ]
    return {"messages": messages}

def convert_file(input_path: Path, output_path: Path):
    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"JSON parse error on line {line_no}: {e}")
            transformed = transform_example(obj)
            fout.write(json.dumps(transformed, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    convert_file(INPUT_PATH, OUTPUT_PATH)
    print(f"Converted '{INPUT_PATH}' → '{OUTPUT_PATH}'")
