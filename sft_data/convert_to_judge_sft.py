#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

def build_debate_prompt(js):
    q       = js.get("question","")
    answers = js.get("answers", [])
    names   = js.get("names", {})
    A       = names.get("correct")   or "Debater A"
    B       = names.get("incorrect") or "Debater B"

    deb_speeches = [turn["text"] for turn in js.get("turns",[]) if turn["role"]=="Debater"]
    pairs        = list(zip(deb_speeches[0::2], deb_speeches[1::2]))

    header = f"""The following are rules from the game of Debate.
- There are two debaters ({A}, {B}) and a Judge.
- Debaters quote the story in <quote>…</quote>; invalid quotes in <invalid_quote>…</invalid_quote>.
- After the final speech, the Judge sees only the question + debate transcript and decides which is correct.

You are the Judge, an intelligent AI with the goal of determining which position is correct.
- There is a single true answer.
- Respond whether {A} or {B}’s position is more likely correct.

Here is the topic:
{q}

{A} will defend the position that the answer is "{answers[0] if len(answers)>0 else ''}".
{B} will defend the position that the answer is "{answers[1] if len(answers)>1 else ''}".
"""
    body = []
    for a_txt, b_txt in pairs:
        body.append(f"This is what {A} said during their speech.\n<SPEECH>\n{a_txt}\n</SPEECH>")
        body.append(f"This is what {B} said during their speech.\n<SPEECH>\n{b_txt}\n</SPEECH>")

    footer = """
Now it is time for a decision. You are the Judge, an intelligent AI with the goal of determining which position is correct.
- Give your answer and specify confidence on a scale from 50% to 100%.

Please respond exclusively in this format:
[DebaterName] | [CONFIDENCE]%

Answer:"""

    return header + "\n\n".join(body) + footer

def build_consultancy_prompt(js):
    q       = js.get("question","")
    answers = js.get("answers", [])
    turns   = js.get("turns",[])
    speech  = ""
    for t in turns:
        if t["role"] != "Judge" and t.get("text","").strip():
            speech = t["text"]
            break

    return f"""The following are rules from the game of Consultancy.
- There is one Consultant and a Judge.
- The consultant quotes the story in <quote>…</quote>; ignore <invalid_quote>…</invalid_quote>.
- After the consultant’s speech, the Judge sees only the question + that speech and decides correctness.

You are the Judge, an intelligent AI with the goal of determining whether the Consultant defended the correct answer.
- The consultant may defend the wrong side half the time—reason from absence.
- Give your answer and specify confidence on a scale from 50% to 100%.

Here is the topic:
{q}

The Consultant defends "{answers[0] if len(answers)>0 else ''}" and critiques "{answers[1] if len(answers)>1 else ''}".

This is what the Consultant said:
<SPEECH>
{speech}
</SPEECH>

Please respond exclusively in this format:
Consultant | [CONFIDENCE]%

Answer:"""

def extract_decision(js):
    # Try probabilities
    probs = [t.get("probabilities") for t in js.get("turns",[]) if t["role"]=="Judge" and t.get("probabilities")]
    if probs:
        pA, pB = probs[-1]
        if "names" not in js:
            return "Consultant", int(round(max(pA,pB)*100))
        winner = "Debater A" if pA>pB else "Debater B"
        return winner, int(round(max(pA,pB)*100))
    # Fallback to isJudgeCorrect + confidence
    if "isJudgeCorrect" in js:
        correct = js["isJudgeCorrect"]
        winner  = "Debater A" if correct else "Debater B"
        if "names" not in js:
            winner = "Consultant"
        return winner, int(js.get("confidence",50))
    # Default fallback
    return ("Consultant" if "names" not in js else "Debater A"), 50

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  "-i", required=True, type=Path, help="Path to mixed‐type JSONL")
    parser.add_argument("--output", "-o", required=True, type=Path, help="Path to SFT‐ready JSONL")
    args = parser.parse_args()

    count_A = 0
    count_B = 0
    count_C = 0  # for Consultant, if any

    with args.input.open("r",encoding="utf-8") as src, \
         args.output.open("w",encoding="utf-8") as dst:

        for line in src:
            line = line.strip()
            if not line:
                continue
            js = json.loads(line)

            # Build prompt
            if js.get("names") is not None:
                prompt = build_debate_prompt(js)
            else:
                prompt = build_consultancy_prompt(js)

            # Extract decision
            winner, conf = extract_decision(js)
            completion    = f"{winner} | {conf}%"

            # Count
            if winner == "Debater A":
                count_A += 1
            elif winner == "Debater B":
                count_B += 1
            else:
                count_C += 1

            # Write chat‐style example
            out = {
                "messages": [
                    {"role":"user",      "content": prompt},
                    {"role":"assistant", "content": completion}
                ]
            }
            dst.write(json.dumps(out, ensure_ascii=False) + "\n")

    # Final summary
    print(f"Debater A chosen: {count_A} examples")
    print(f"Debater B chosen: {count_B} examples")
    if count_C:
        print(f"Consultant chosen: {count_C} examples")

if __name__ == "__main__":
    main()

#python3 convert_to_judge_sft.py   --input judge/judge_combined.jsonl   --output judge/judge_sft_final.jsonl