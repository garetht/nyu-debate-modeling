#!/usr/bin/env python3
import argparse
import csv
import json
import os
from typing import Dict, Iterable, List, Optional

from transformers import AutoTokenizer

QWEN3_MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"  
ASSISTANT_START = "<|im_start|>assistant\n"  # boundary right before the answer
IM_END = "<|im_end|>"

# ---------------- IO ----------------
def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

# ------------- Debate utils -------------
def prior_turns(turns: List[Dict], upto_idx_exclusive: int) -> str:
    parts = []
    for t in turns[:upto_idx_exclusive]:
        role = (t.get("role") or "").strip()
        text = (t.get("text") or "").strip()
        if not text:
            continue
        if role == "Debater":
            parts.append(f"Debater: {text}")
        elif role == "Judge":
            if text:
                parts.append(f"Judge: {text}")
        else:
            parts.append(f"{role or 'Unknown'}: {text}")
    return "\n\n".join(parts).strip()

def infer_position(turn: Dict) -> Optional[str]:
    """Even index -> Debater A, odd index -> Debater B (simple, matches your examples)."""
    idx = turn.get("index")
    if idx is None:
        return None
    return "A" if (idx % 2 == 0) else "B"

# ------------- Prompt builder -------------
def build_system() -> str:
    return (
        "You are a debater in a truth-seeking debate. "
        "Quote the story with <quote>...</quote> when citing lines. "
        "Be concise and grounded in the text."
    )

def build_user(item: Dict, turns: List[Dict], target_idx: int) -> str:
    title = (item.get("storyTitle") or "").strip()
    story = (item.get("story") or "").strip()
    question = (item.get("question") or "").strip()
    answers = item.get("answers") or []
    a_opt = answers[0] if len(answers) > 0 else ""
    b_opt = answers[1] if len(answers) > 1 else ""
    pos = infer_position(turns[target_idx]) or "A/B"
    history = prior_turns(turns, target_idx)
    history_block = f"\n\nDebate history so far:\n{history}" if history else ""
    return (
        f"Story Title: {title}\n\n"
        f"Story:\n{story}\n\n"
        f"Question: {question}\n"
        f"Option A: {a_opt}\n"
        f"Option B: {b_opt}\n\n"
        f"You are Debater {pos}. Defend your assigned option truthfully and persuasively."
        f"{history_block}\n\n"
        f"Write your next speech now."
    )

def build_messages(item: Dict, turns: List[Dict], target_idx: int) -> List[Dict]:
    system = build_system()
    user = build_user(item, turns, target_idx)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        # add_generation_prompt=True will append <|im_start|>assistant\n for us
    ]

# ------------- Length filter -------------
def within_length(tokenizer, prompt_str: str, boundary: str, max_total_tokens: int) -> bool:
    # Count tokens of prompt + boundary (the collator masks up to this boundary)
    ids_prompt = tokenizer.encode(prompt_str, add_special_tokens=False)
    ids_boundary = tokenizer.encode(boundary, add_special_tokens=False)
    total = len(ids_prompt) + len(ids_boundary)
    cap = getattr(tokenizer, "model_max_length", None)
    if cap and max_total_tokens > cap:
        max_total_tokens = cap
    return total <= max_total_tokens

# ------------- Conversion -------------
def convert_file(input_jsonl: str, output_csv: str, max_length: int, enable_thinking: bool = False) -> int:
    ensure_dir(output_csv)
    tok = AutoTokenizer.from_pretrained(QWEN3_MODEL_ID)
    # padding for later batching (trainer)
    try:
        tok.pad_token = tok.eos_token
    except Exception:
        pass
    tok.padding_side = "right"

    written = 0
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["instruction", "output"])

        for item in iter_jsonl(input_jsonl):
            turns: List[Dict] = item.get("turns") or []
            for i, t in enumerate(turns):
                if t.get("role") != "Debater":
                    continue
                output = (t.get("text") or "").strip()
                if not output:
                    continue

                messages = build_messages(item, turns, i)
                # Let Qwen3’s tokenizer build the exact ChatML string and the assistant-start marker
                prompt_str = tok.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,  # off by default for SFT
                )

                if within_length(tok, prompt_str, ASSISTANT_START, max_length):
                    w.writerow([prompt_str, output])
                    written += 1
    return written

# ------------- CLI -------------
def main():
    ap = argparse.ArgumentParser("Prepare Qwen3 SFT CSV from debate JSONL")
    ap.add_argument("--input_jsonl", required=True, help="Path to debate JSONL")
    ap.add_argument("--output_csv", required=True, help="Where to write CSV with instruction,output")
    ap.add_argument("--max_length", type=int, default=30000,
                    help="Token cap for (prompt + assistant-start boundary). "
                         "Qwen3 default context is 32,768; keep some headroom.")
    ap.add_argument("--enable_thinking", action="store_true",
                    help="If set, leave Qwen3 thinking mode ON in prompts (default OFF).")
    args = ap.parse_args()

    rows = convert_file(args.input_jsonl, args.output_csv, args.max_length, args.enable_thinking)
    print(f"Wrote {rows} rows to {args.output_csv}")

if __name__ == "__main__":
    main()

'''
python prepare_qwen_dataset.py \
  --input_jsonl /home/ubuntu/mars-arnesen-gh/leonidtsyplenkov/sft_data/debater/debater_combined_filled.jsonl \
  --output_csv  /home/ubuntu/mars-arnesen-gh/leonidtsyplenkov/sft_data/debater/training_dataset_for_debater_no_judge_speeches_qwen.csv \
  --max_length 120000
'''