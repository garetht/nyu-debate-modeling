#!/usr/bin/env python3
"""
Convert SFT-style judge JSONL to RFT-ready items for o4-mini + deterministic grader.

Input format (each line):
{
  "messages": [
    {"role": "system", "content": "...rules..."},
    {"role": "user", "content": "...debate transcript..."},
    {"role": "assistant", "content": "Debater A | 90%"}
  ]
}

Output format (each line):
{
  "messages": [
    {"role":"user","content":"<system rules>\n\n<debate transcript>\n\nNow give your answer:"}
  ],
  "gt_winner": "A",
  "gt_confidence": 0.9,
  "gt_confidence_percent": 90
}

Notes
- We fold system+user into a single **user** message so the final message is user (as recommended by OpenAI RFT docs).
- Assistant verdict is parsed into ground-truth fields for the deterministic Python grader you’ll attach to the RFT job.
"""

import argparse
import json
import re
import sys
from typing import Iterable

VERDICT_RE = re.compile(
    r"\bDebater\s*([AB])\b\s*\|\s*([0-9]{1,3}(?:\.[0-9]+)?)\s*%",
    flags=re.IGNORECASE
)

def iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield i, json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] {path}:{i}: JSON decode error: {e}", file=sys.stderr)

def fold_to_user(system_text: str, user_text: str) -> str:
    parts = []
    if system_text:
        parts.append(system_text.strip())
    if user_text:
        parts.append(user_text.strip())
    # Ensure the prompt ends with the explicit call to action:
    # (Keep whatever your original user text had; add a final nudge if missing.)
    merged = "\n\n".join(parts)
    if "Now give your answer" not in merged:
        merged = merged.rstrip() + "\n\nNow give your answer:"
    return merged

def parse_verdict(text: str):
    """
    Returns (winner_letter: 'A'|'B', confidence_float in [0.5,1.0]) or None
    """
    if not text:
        return None
    m = VERDICT_RE.search(text)
    if not m:
        return None
    winner = m.group(1).upper()
    pct = float(m.group(2))
    # clip to [50,100], then map to [0.5,1.0]
    pct = max(50.0, min(100.0, pct))
    conf = pct / 100.0
    return winner, conf, int(round(pct))

def convert_record(rec: dict, *, preserve_ids: bool = False) -> dict | None:
    msgs = rec.get("messages") or []
    if not msgs or len(msgs) < 2:
        return None

    # Extract system + user; assistant is ground truth verdict
    sys_txt = ""
    user_txt = ""
    asst_txt = ""

    for m in msgs:
        role = (m.get("role") or "").strip().lower()
        content = (m.get("content") or "")
        if role == "system":
            sys_txt += (content + "\n\n")
        elif role == "user":
            user_txt += (content + "\n\n")
        elif role == "assistant":
            asst_txt += (content + "\n")

    parsed = parse_verdict(asst_txt)
    if not parsed:
        return None
    gt_winner, gt_conf, gt_conf_pct = parsed

    merged_user = fold_to_user(sys_txt, user_txt)

    out = {
        "messages": [
            {"role": "user", "content": merged_user.strip()}
        ],
        "gt_winner": gt_winner,
        "gt_confidence": round(gt_conf, 4),
        "gt_confidence_percent": gt_conf_pct
    }

    # Carry through an id if present (optional, helps with traceability)
    if preserve_ids:
        if "id" in rec:
            out["source_id"] = rec["id"]
        elif "example_id" in rec:
            out["source_id"] = rec["example_id"]

    return out

def main():
    ap = argparse.ArgumentParser(description="Convert SFT judge JSONL to RFT items.")
    ap.add_argument("input_jsonl", help="Path to input JSONL (SFT-style).")
    ap.add_argument("output_jsonl", help="Path to write RFT-ready JSONL.")
    ap.add_argument("--preserve-ids", action="store_true", help="Copy id/example_id into output.source_id if present.")
    ap.add_argument("--skip-unparseable", action="store_true", help="Skip items that don't parse; otherwise emit warnings.")
    args = ap.parse_args()

    n_in = n_out = n_skip = 0
    with open(args.output_jsonl, "w", encoding="utf-8") as w:
        for lineno, rec in iter_jsonl(args.input_jsonl):
            n_in += 1
            out = convert_record(rec, preserve_ids=args.preserve_ids)
            if out is None:
                n_skip += 1
                msg = f"[WARN] {args.input_jsonl}:{lineno}: could not parse assistant verdict or bad format."
                if args.skip_unparseable:
                    print(msg, file=sys.stderr)
                    continue
                else:
                    print(msg, file=sys.stderr)
                    continue
            w.write(json.dumps(out, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"[INFO] Read {n_in} items; wrote {n_out}; skipped {n_skip}.", file=sys.stderr)

if __name__ == "__main__":
    main()
