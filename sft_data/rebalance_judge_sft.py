#!/usr/bin/env python3
import argparse
import json
import random
import re
from pathlib import Path

def swap_answers_and_speeches(prompt: str) -> str:
    """
    Keep Debater A as first speaker, but:
     1. Swap the two answer strings in the header.
     2. Swap the A/B speech blocks so they match the new defended positions.
    """
    # 1) Swap the header answers
    header_pattern = (
        r"(Debater A will defend the position that the answer is \")"
        r"(.+?)(\"\.)(.*?)(Debater B will defend the position that the answer is \")"
        r"(.+?)(\".)"
    )
    def _swap_header(m):
        preA, ansA, postA, mid, preB, ansB, postB = (
            m.group(1), m.group(2), m.group(3),
            m.group(4),
            m.group(5), m.group(6), m.group(7)
        )
        return f"{preA}{ansB}{postA}{mid}{preB}{ansA}{postB}"

    prompt, n_subs = re.subn(header_pattern, _swap_header, prompt, flags=re.DOTALL)
    if n_subs != 1:
        return prompt  # fallback if header didn't match

    # 2) Extract and swap the A/B speech blocks
    a_texts = re.findall(
        r"This is what Debater A said during their speech\.\n<speech>\n(.*?)\n</speech>",
        prompt, flags=re.DOTALL|re.IGNORECASE
    )
    b_texts = re.findall(
        r"This is what Debater B said during their speech\.\n<speech>\n(.*?)\n</speech>",
        prompt, flags=re.DOTALL|re.IGNORECASE
    )
    if len(a_texts) != len(b_texts):
        return prompt  # fallback

    # rebuild body
    header, rest = prompt.split("This is what Debater A said during their speech.", 1)
    new_body = []
    for a_txt, b_txt in zip(a_texts, b_texts):
        new_body.append(
            f"This is what Debater A said during their speech.\n<speech>\n{b_txt}\n</speech>"
        )
        new_body.append(
            f"This is what Debater B said during their speech.\n<speech>\n{a_txt}\n</speech>"
        )
    # footer after the last </speech>
    footer = rest.rsplit("</speech>", 1)[-1]

    return header + "".join(new_body) + footer

def rebalance(input_path: Path, output_path: Path, seed: int=None):
    if seed is not None:
        random.seed(seed)

    # load all entries
    entries = [json.loads(line) 
               for line in input_path.read_text(encoding="utf-8").splitlines() 
               if line.strip()]

    # collect indices by winner
    idx_A, idx_B = [], []
    for i, e in enumerate(entries):
        comp = e["messages"][-1]["content"].strip()
        if comp.startswith("Debater A"):
            idx_A.append(i)
        elif comp.startswith("Debater B"):
            idx_B.append(i)

    nA, nB = len(idx_A), len(idx_B)
    print(f"Before rebalance → Debater A: {nA}, Debater B: {nB}")

    # how many A->B flips needed
    flips = (nA - nB) // 2
    if flips <= 0:
        print("Already balanced or B excess; no flips performed.")
        return

    to_flip = set(random.sample(idx_A, flips))

    # apply flips and write out
    newA = newB = 0
    with output_path.open("w", encoding="utf-8") as f_out:
        for i, e in enumerate(entries):
            # default: keep as is
            winner, conf = None, None

            if i in to_flip:
                # 1) swap header answers and speech blocks
                original_prompt = e["messages"][0]["content"]
                e["messages"][0]["content"] = swap_answers_and_speeches(original_prompt)

                # 2) flip completion
                old = e["messages"][-1]["content"].strip()
                mA = re.match(r"Debater A\s*\|\s*(\d+)%", old)
                mB = re.match(r"Debater B\s*\|\s*(\d+)%", old)
                if mA:
                    conf = mA.group(1)
                    new_comp = f"Debater B | {conf}%"
                    newB += 1
                elif mB:
                    conf = mB.group(1)
                    new_comp = f"Debater A | {conf}%"
                    newA += 1
                else:
                    new_comp = old
                e["messages"][-1]["content"] = new_comp
            else:
                comp = e["messages"][-1]["content"].strip()
                if comp.startswith("Debater A"):
                    newA += 1
                elif comp.startswith("Debater B"):
                    newB += 1

            f_out.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"After  rebalance → Debater A: {newA}, Debater B: {newB}")
    print(f"Flipped {flips} examples from A to B.")
    print(f"Wrote balanced data to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rebalance Debater A/B wins by swapping answers and speeches."
    )
    parser.add_argument("-i", "--input",  type=Path, required=True,
                        help="Input JSONL (chat‐style SFT examples)")
    parser.add_argument("-o", "--output", type=Path, required=True,
                        help="Output balanced JSONL")
    parser.add_argument("-s", "--seed",   type=int, default=None,
                        help="Random seed for reproducible flips")
    args = parser.parse_args()

    rebalance(args.input, args.output, args.seed)

#python3 rebalance_judge_sft.py -i judge/judge_sft_final.jsonl -o judge/judge_sft_balanced.jsonl -s 42