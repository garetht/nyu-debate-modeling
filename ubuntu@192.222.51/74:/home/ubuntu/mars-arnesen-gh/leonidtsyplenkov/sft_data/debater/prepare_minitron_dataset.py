#!/usr/bin/env python3
import argparse
import csv
import json
import os
from typing import Dict, Iterable, List, Tuple, Optional

from transformers import AutoTokenizer

MINITRON_MODEL_ID = "mistralai/Ministral-8B-Instruct-2410"
                    #"nvidia/Mistral-NeMo-Minitron-8B-Instruct"
INST_OPEN = "[INST]"
INST_CLOSE = "[/INST]"
BOUNDARY = INST_CLOSE  # collator boundary right before assistant starts

# ---------- IO helpers ----------
def iter_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

# ---------- Prompt building ----------
def prior_turns_as_text(turns: List[Dict], upto_idx_exclusive: int) -> str:
    """Join previous turns up to (but not including) the target index."""
    parts = []
    for t in turns[:upto_idx_exclusive]:
        role = t.get("role", "").strip()
        txt = (t.get("text") or "").strip()
        if not txt:
            continue
        if role == "Debater":
            parts.append(f"Debater: {txt}")
        elif role == "Judge":
            # Keep judge content if present (often empty in your samples)
            if txt:
                parts.append(f"Judge: {txt}")
        else:
            # unknown role fallback
            parts.append(f"{role or 'Unknown'}: {txt}")
    return "\n\n".join(parts).strip()

def infer_position(turn: Dict) -> Optional[str]:
    """
    If you have a clear A/B mapping in your upstream process, plug it here.
    With the given samples there's no explicit 'A' or 'B' on the turn;
    we’ll encode by index parity as a pragmatic default:
      - index 0,2,4,... => Debater A
      - index 1,3,5,... => Debater B
    """
    idx = turn.get("index")
    if idx is None:
        return None
    return "A" if (idx % 2 == 0) else "B"

def build_system_block() -> str:
    # Keep system minimal; your pipeline can expand rules here if needed.
    return (
        "You are a debater in a truth-seeking debate. "
        "Cite exact quotes using <quote>...</quote> when quoting the story. "
        "Keep arguments concise and grounded in the text."
    )

def build_user_block(
    item: Dict,
    turns: List[Dict],
    target_idx: int,
) -> str:
    story_title = item.get("storyTitle", "").strip()
    story = (item.get("story") or "").strip()
    question = (item.get("question") or "").strip()
    answers = item.get("answers") or []
    a_opt = answers[0] if len(answers) > 0 else ""
    b_opt = answers[1] if len(answers) > 1 else ""

    # Current debater
    turn = turns[target_idx]
    pos = infer_position(turn) or "A/B"

    history = prior_turns_as_text(turns, target_idx)
    history_block = f"\n\nDebate history so far:\n{history}" if history else ""

    user = (
        f"Story Title: {story_title}\n\n"
        f"Story:\n{story}\n\n"
        f"Question: {question}\n"
        f"Option A: {a_opt}\n"
        f"Option B: {b_opt}\n\n"
        f"You are Debater {pos}. Defend your assigned option truthfully and persuasively."
        f"{history_block}\n\n"
        f"Write your next speech now."
    )
    return user

def build_instruction(system: str, user: str) -> str:
    # Mistral/Minitron answer begins right after [/INST]
    return f"{INST_OPEN}{system}\n\n{user}{INST_CLOSE}"

# ---------- Length filter ----------
def within_length(tokenizer, instruction: str, boundary: str, max_total: int) -> bool:
    ids_instr = tokenizer.encode(instruction, add_special_tokens=False)
    ids_bndry = tokenizer.encode(boundary, add_special_tokens=False)
    total = len(ids_instr) + len(ids_bndry)
    model_cap = getattr(tokenizer, "model_max_length", None)
    if model_cap and max_total > model_cap:
        max_total = model_cap
    return total <= max_total

# ---------- Conversion ----------
def convert_file(
    input_jsonl: str,
    output_csv: str,
    max_length: int,
    skip_judge_turns: bool = True,
) -> int:
    ensure_dir(output_csv)
    tok = AutoTokenizer.from_pretrained(MINITRON_MODEL_ID)
    # right padding for batching later
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
            # iterate over debater speeches only
            for i, t in enumerate(turns):
                if skip_judge_turns and (t.get("role") == "Judge"):
                    continue
                if t.get("role") != "Debater":
                    continue

                system = build_system_block()
                user = build_user_block(item, turns, i)
                instruction = build_instruction(system, user)
                output = (t.get("text") or "").strip()
                if not output:
                    continue

                if within_length(tok, instruction, BOUNDARY, max_length):
                    w.writerow([instruction, output])
                    written += 1
    return written

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser("Prepare Minitron-128K SFT CSV from debate JSONL")
    ap.add_argument("--input_jsonl", required=True, help="Path to debater JSONL")
    ap.add_argument("--output_csv", required=True, help="Path to output CSV")
    ap.add_argument("--max_length", type=int, default=120000, help="Max tokens for (instruction + boundary)")
    args = ap.parse_args()

    rows = convert_file(args.input_jsonl, args.output_csv, args.max_length)
    print(f"Wrote {rows} rows to {args.output_csv}")

if __name__ == "__main__":
    main()

'''
python prepare_minitron_dataset.py \
  --input_jsonl /home/ubuntu/mars-arnesen-gh/leonidtsyplenkov/sft_data/debater/debater_combined_filled.jsonl \
  --output_csv  /home/ubuntu/mars-arnesen-gh/leonidtsyplenkov/sft_data/debater/training_dataset_for_debater_no_judge_speeches_minitron.csv \
  --max_length 120000
'''
