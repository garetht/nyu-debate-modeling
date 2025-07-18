#!/usr/bin/env python3
import argparse
import json
import math
import re
import pandas as pd

# --- Character‐count limits ---
DEBATER_CHAR_LIMIT     = 1500
DEBATER_QUOTE_LIMIT    = 500
CONSULTANT_CHAR_LIMIT  = 1500
CONSULTANT_QUOTE_LIMIT = 500
JUDGE_CHAR_LIMIT       = 750
JUDGE_QUOTE_LIMIT      = None

def count_quote_chars(text: str) -> int:
    total = 0
    for m in re.finditer(r'<quote>(.*?)</quote>', text, re.DOTALL):
        total += len(m.group(1))
    return total

def process_row(row: pd.Series) -> dict:
    transcript = json.loads(row['transcript'])
    story_id    = row['question_set_id']
    debate_id   = str(row['debate_id'])
    story_title = transcript.get('story_title', row['story_title'])
    story_text  = transcript.get('story', '')
    question    = row['question']
    ans         = transcript['answers']
    #answers     = [ans['incorrect'], ans['correct']]    # [distractor, correct]

    turns = []

    if row['debate_method'] == 'debate':
        # Figure out which name is A vs B
        names = transcript.get('names', {})
        name_A = 'Debater A'
        # Determine which side (correct vs incorrect) is Debater A
        # transcript['names']['correct'] is e.g. "Debater B"
        # transcript['names']['incorrect'] is the other
        correct_is_A   = (names.get('correct')   == name_A)

        if correct_is_A:
            answers = [ans['correct'], ans['incorrect']]
        else:
            answers = [ans['incorrect'], ans['correct']]

        idx = 0
        for rnd in transcript.get('rounds', []):
            text_corr = rnd.get('correct','').strip()
            text_inc  = rnd.get('incorrect','').strip()

            # Assign A's and B's text
            if correct_is_A:
                text_A, text_B = text_corr, text_inc
            else:
                # either correct_is_A or fallback
                text_A, text_B = text_inc, text_corr

            # Emit Debater A
            idx += 1
            turns.append({
                "role": "Debater",
                "index": idx,
                "text": text_A,
                "probabilities": None,
                "chars": len(text_A),
                "charLimit": DEBATER_CHAR_LIMIT,
                "quoteChars": count_quote_chars(text_A),
                "quoteCharLimit": DEBATER_QUOTE_LIMIT
            })
            # Emit Debater B
            idx += 1
            turns.append({
                "role": "Debater",
                "index": idx,
                "text": text_B,
                "probabilities": None,
                "chars": len(text_B),
                "charLimit": DEBATER_CHAR_LIMIT,
                "quoteChars": count_quote_chars(text_B),
                "quoteCharLimit": DEBATER_QUOTE_LIMIT
            })

    elif row['debate_method'] == 'consultancy':
        # Single‐speaker consultancy: one Consultant per round
        ctype = row['consultant_type']  # "correct" or "incorrect"
        idx = 0
        answers = [ans[ctype]]
        for rnd in transcript.get('rounds', []):
            text = (rnd.get(ctype) or '').strip()
            if not text:
                continue
            idx += 1
            turns.append({
                "role": "Consultant",
                "index": idx,
                "text": text,
                "probabilities": None,
                "chars": len(text),
                "charLimit": CONSULTANT_CHAR_LIMIT,
                "quoteChars": count_quote_chars(text),
                "quoteCharLimit": CONSULTANT_QUOTE_LIMIT
            })

    # Final Judge turn (common)
    p_corr = float(row['confidence']) / 100.0
    p_inc  = 1.0 - p_corr
    judge_text = row.get('explanation','').strip()
    turns.append({
        "role": "Judge",
        "index": None,
        "text": judge_text,
        "probabilities": [p_inc, p_corr],
        "chars": len(judge_text),
        "charLimit": JUDGE_CHAR_LIMIT,
        "quoteChars": count_quote_chars(judge_text),
        "quoteCharLimit": JUDGE_QUOTE_LIMIT
    })

    try:
        return {
            "storyId":        story_id,
            "storyTitle":     story_title,
            "story":          story_text,
            "question":       question,
            "answers":        answers,
            "debateId":       debate_id,
            "turns":          turns,
            "isJudgeCorrect": bool(row['correct'])
        }
    except:
        print(row['debate_method'])

def main():
    parser = argparse.ArgumentParser(
        description="Convert raw QuALITY CSV → JSONL, ordering Debater A then B"
    )
    parser.add_argument("infile",  help="Raw CSV path")
    parser.add_argument("outfile", help="Output JSONL path")
    parser.add_argument("with_consultancy", help="Include consultancy or not (yes / no) ")
    args = parser.parse_args()

    df = pd.read_csv(args.infile)
    df = df[df['experiment_name'] == 'Experiment 8']
    with_consultancy = args.with_consultancy
    with open(args.outfile, "w", encoding="utf-8") as fout:
        for _, row in df.iterrows():
            if with_consultancy == 'no' and row['debate_method'] == 'consultancy':
                continue
            out = process_row(row)
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()

#bash-3.2$ python3 converter_raw_debates.py llm_debate_human_judge_dataset.csv converted_khan_with_consultancy.jsonl yes
#bash-3.2$ python3 converter_raw_debates.py llm_debate_human_judge_dataset.csv converted_khan_only_debate.jsonl no