import json
from pathlib import Path

METADATA_PATH   = Path("debates-metadata.jsonl")
READABLE_PATH   = Path("debates-readable.jsonl")
OUTPUT_PATH    = Path("train_judge_human_data.jsonl")

def load_and_filter():
    """
    Yield (md, js) pairs only for those entries Michael et al. included in the paper.
    """
    with METADATA_PATH.open("r") as meta_f, READABLE_PATH.open("r") as read_f:
        for meta_line, read_line in zip(meta_f, read_f):
            md = json.loads(meta_line)
            if not md.get("includedInPaper", False):
                continue
            js = json.loads(read_line)
            yield md, js

def build_debate_prompt(js: dict) -> str:
    Q        = js["question"]
    answers  = js["answers"]  # [correct, incorrect]
    A, B     = "Debater A", "Debater B"

    # collect exactly the first four Debater speeches (two rounds)
    deb_texts = [t["text"] for t in js["turns"] if t["role"] == "Debater"]
    sA1, sB1, sA2, sB2 = (deb_texts + ["","","",""])[:4]

    return f"""You are the Judge, an intelligent AI with the goal of determining which position is correct.
- Quotes from the story appear in <quote>…</quote>; invalid quotes in <invalid_quote>…</invalid_quote>.
- Do not trust unquoted claims.

Here is the topic:
{Q}

{A} defends the position that the answer is "{answers[0]}".
{B} defends the position that the answer is "{answers[1]}".

This is what {A} said first:
<speech>
{sA1}
</speech>

This is what {B} said first:
<speech>
{sB1}
</speech>

This is what {A} said second:
<speech>
{sA2}
</speech>

This is what {B} said second:
<speech>
{sB2}
</speech>

Now decide which debater is defending the true answer and specify confidence (50–100%) in this format:
[DebaterName] | [CONFIDENCE]

Answer:"""

def build_consultancy_prompt(js: dict) -> str:
    Q       = js["question"]
    answers = js["answers"]  # [correct, incorrect]

    # first non-Judge turn is the consultant’s speech
    speech = ""
    for turn in js["turns"]:
        if turn["role"] != "Judge" and turn.get("text","").strip():
            speech = turn["text"]
            break

    return f"""You are the Judge, an intelligent AI with the goal of determining whether the Consultant is correct.
- Genuine quotations from the story appear in <quote>…</quote>; ignore <invalid_quote>…</invalid_quote>.
- The consultant may defend the wrong side half the time—reason from absence.

Here is the topic:
{Q}

The Consultant defends "{answers[0]}" and critiques "{answers[1]}".

This is what the Consultant said:
<speech>
{speech}
</speech>

Decide if the Consultant argued for the true answer. Give your answer as:
[ConsultantName] | [CONFIDENCE]

Answer:"""

def main():
    count = 0
    with OUTPUT_PATH.open("w") as out_f:
        for md, js in load_and_filter():
          if md["setting"]["isDebate"]:
              prompt = build_debate_prompt(js)
          else:
              prompt = build_consultancy_prompt(js)

          # reconstruct the completion
          winner = "Debater A" if js["isJudgeCorrect"] else "Debater B"
          probs  = [t["probabilities"] for t in js["turns"]
                    if t["role"].lower().startswith("judge") and "probabilities" in t]
          if probs:
              pA, pB = probs[-1]
              conf = int(round(max(pA, pB) * 100))
          else:
              conf = 50

          completion = f"{winner} | {conf}%"

          json.dump({"prompt": prompt+"\n", "completion": " "+completion+"\n"}, out_f)
          out_f.write("\n")
          count += 1

    print(f"Wrote {count} examples to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()