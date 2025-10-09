#!/usr/bin/env python3
import pandas as pd
import json
import re
from sklearn.model_selection import train_test_split

# ------------------------------ Helpers ------------------------------------ #
EOT_SPLIT = '<|eot_id|><|start_header_id|>'

def extract_chunks(instruction_text: str):
    """
    Return (system_content, user_contents_list) from your special-tagged blob.
    """
    parts = instruction_text.split(EOT_SPLIT)
    system_content = ""
    user_contents = []

    if parts:
        # first part may contain the system header+content
        pre = parts[0]
        sys_tag = '<|start_header_id|>system<|end_header_id|>\n\n'
        if sys_tag in pre:
            system_content = pre.split(sys_tag, 1)[1].strip()

    # remaining parts: pull only user messages, in order
    for part in parts[1:]:
        if part.startswith('user<|end_header_id|>'):
            user_content = (
                part.replace('user<|end_header_id|>\n\n', '')
                    .replace('<|eot_id|>', '')
                    .strip()
            )
            if user_content:
                user_contents.append(user_content)
    return system_content, user_contents

def make_messages_for_rft(instruction_text: str):
    """
    Produce a single-message array that is valid for RFT:
      messages = [ { "role": "user", "content": "<system+user concatenated>" } ]
    - No 'system' role.
    - Final entry is 'user'.
    """
    system_content, user_contents = extract_chunks(instruction_text)

    blocks = []
    if system_content:
        # Inline your system instructions as plain text at the top
        blocks.append(system_content)
    if user_contents:
        blocks.append("\n\n".join(user_contents))

    combined = "\n\n".join([b for b in blocks if b]).strip()
    if not combined:
        combined = "[EMPTY_PROMPT]"

    return [{"role": "user", "content": combined}]

def extract_first_user_message(instruction_text: str) -> str:
    """ Keep your original semantics: first user message drives topic/positions/speeches parsing. """
    parts = instruction_text.split(EOT_SPLIT)
    for i, part in enumerate(parts):
        if i == 0:
            continue
        if part.startswith('user<|end_header_id|>'):
            return (
                part.replace('user<|end_header_id|>\n\n', '')
                    .replace('<|eot_id|>', '')
                    .strip()
            )
    return ""

def clean_html_breaks(s: str) -> str:
    s = re.sub(r'<br>\s*', '\n', s, flags=re.IGNORECASE)
    s = re.sub(r'\n{3,}', '\n\n', s)
    return s.strip()

def extract_topic_positions(user_text: str):
    txt = clean_html_breaks(user_text)

    topic_match = re.search(
        r'Here\s*is\s*the\s*topic\s*.*?:\s*(.+?)(?=\n\s*Debater[_\s]?[AB]\s+will\s+defend)',
        txt, flags=re.IGNORECASE | re.DOTALL
    )
    topic = topic_match.group(1).strip() if topic_match else "[TOPIC_NOT_FOUND]"

    pos_a_match = re.search(
        r'Debater[_\s]?A\s+will\s+defend\s+the\s+position\s+that\s+the\s+answer\s+is\s*(.+?)\.\s*$',
        txt, flags=re.IGNORECASE | re.MULTILINE
    )
    pos_b_match = re.search(
        r'Debater[_\s]?B\s+will\s+defend\s+the\s+position\s+that\s+the\s+answer\s+is\s*(.+?)\.\s*$',
        txt, flags=re.IGNORECASE | re.MULTILINE
    )
    position_a = pos_a_match.group(1).strip() if pos_a_match else "[POS_A_NOT_FOUND]"
    position_b = pos_b_match.group(1).strip() if pos_b_match else "[POS_B_NOT_FOUND]"

    return topic, position_a, position_b

def extract_prior_speeches(user_text: str):
    """
    Return ordered list of prior speeches as [("A"|"B", text), ...].
    Accepts variants like:
      - "This is what Debater A said during their speech."
      - "This is what you, Debater_A, said during your previous speech."
      - "This is what your opponent, Debater_B, said during their speech."
    """
    txt = clean_html_breaks(user_text)
    heading_re = re.compile(
        r'(This\s+is\s+what\s+(?:you,\s*)?Debater[_\s]?(A|B)[^.\n]*\.\s*|'
        r'This\s+is\s+what\s+your\s+opponent,\s*Debater[_\s]?(A|B)[^.\n]*\.\s*)',
        flags=re.IGNORECASE
    )

    indices = []
    for m in heading_re.finditer(txt):
        sp = 'A' if (m.group(2) == 'A' or m.group(3) == 'A') else 'B'
        indices.append((m.start(), m.end(), sp))

    blocks = []
    if not indices:
        return blocks

    indices.append((len(txt), len(txt), None))  # sentinel

    for i in range(len(indices) - 1):
        start, end, speaker = indices[i]
        next_start, _, _ = indices[i + 1]
        content = txt[end:next_start].strip()
        content = re.split(
            r'\n\s*Now\s+it\s+is\s+(?:your\s+turn|the\s+time\s+for\s+a\s+decision)\b',
            content, flags=re.IGNORECASE
        )[0].strip()
        blocks.append((speaker, content))

    return blocks

def detect_policy_speaker(full_instruction: str, messages_user_only: list) -> str:
    """
    Detect if the policy model plays Debater A or B.
    We look in the original blob (system+user tags) first; otherwise in the (inlined) user text.
    """
    # 1) Search original full instruction blob
    if re.search(r'You\s+are\s+Debater[_\s]?A\b', full_instruction, re.IGNORECASE):
        return "A"
    if re.search(r'You\s+are\s+Debater[_\s]?B\b', full_instruction, re.IGNORECASE):
        return "B"

    # 2) Fallback: look in our single user-block text
    if messages_user_only:
        content = messages_user_only[0]["content"]
        if re.search(r'You\s+are\s+Debater[_\s]?A\b', content, re.IGNORECASE):
            return "A"
        if re.search(r'You\s+are\s+Debater[_\s]?B\b', content, re.IGNORECASE):
            return "B"

    return "A"

def build_grader_user(topic: str, pos_a: str, pos_b: str, prior_speeches_trimmed: list) -> str:
    """
    Judge's user message WITHOUT the current-round pair (we'll append the opponent's latest
    and the policy's fresh speech at grading time).
    """
    parts = []
    parts.append("Here is the topic they will be debating:")
    parts.append(topic.strip())
    parts.append(f"Debater A will defend the position that the answer is {pos_a.strip()}.")
    parts.append(f"Debater B will defend the position that the answer is {pos_b.strip()}.")

    for speaker, text in prior_speeches_trimmed:
        parts.append(f"This is what Debater {speaker} said during their speech.\n")
        parts.append(text.strip())

    return "\n\n".join(parts).strip()

# --------------------------- RFT item builder ------------------------------- #
def create_rft_items(df: pd.DataFrame):
    """
    Emit per item:
      - messages:           [ { "role": "user", "content": "<system+user inlined>" } ]  (RFT-compliant)
      - grader_user:        topic + positions + all prior speeches EXCEPT the most recent opponent speech
      - policy_speaker:     "A" | "B"
      - opponent_speaker:   "B" if A else "A"
      - opponent_last_speech: string ("" if none)
      - topic / position_A / position_B
    """
    items = []
    for idx, row in df.iterrows():
        try:
            instr = row['instruction']

            # Construct RFT-compliant messages (no system; last is user)
            messages = make_messages_for_rft(instr)

            # Parse structured fields for the grader
            user0 = extract_first_user_message(instr)
            topic, pos_a, pos_b = extract_topic_positions(user0)
            prior = extract_prior_speeches(user0)

            policy_speaker = detect_policy_speaker(instr, messages)
            opponent_speaker = "B" if policy_speaker == "A" else "A"

            # Find most recent opponent speech (if any)
            opponent_last_speech = ""
            opponent_last_idx = -1
            for i in range(len(prior) - 1, -1, -1):
                if prior[i][0] == opponent_speaker:
                    opponent_last_idx = i
                    opponent_last_speech = prior[i][1]
                    break

            # Remove that one from the history shown above the pair
            prior_trimmed = prior[:]
            if opponent_last_idx >= 0:
                prior_trimmed.pop(opponent_last_idx)

            grader_user = build_grader_user(topic, pos_a, pos_b, prior_trimmed)

            item = {
                "messages": messages,
                "grader_user": grader_user,
                "policy_speaker": policy_speaker,
                "opponent_speaker": opponent_speaker,
                "opponent_last_speech": opponent_last_speech,
                "topic": topic,
                "position_A": pos_a,
                "position_B": pos_b
            }

            # Sanity checks for RFT constraints
            assert all(m["role"] == "user" for m in item["messages"]), "RFT: only 'user' role allowed"
            assert item["messages"][-1]["role"] == "user", "RFT: last message must be user"

            items.append(item)

        except Exception as e:
            print(f"[WARN] Error processing row {idx}: {e}")
            continue

    return items

# ------------------------------ IO helpers --------------------------------- #
def save_jsonl(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

# --------------------------------- Main ------------------------------------ #
def main():
    df = pd.read_csv('training_dataset_for_debater_no_judge_speeches.csv')
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)

    # Optional downsample (match your previous script)
    df = df.iloc[:len(df)//4]

    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    print(f"Train examples: {len(train_df)}")
    print(f"Validation examples: {len(val_df)}")

    train_items = create_rft_items(train_df)
    val_items = create_rft_items(val_df)

    save_jsonl(train_items, 'train_rft_items.jsonl')
    save_jsonl(val_items, 'val_rft_items.jsonl')

    print("\nFiles created:")
    print("- train_rft_items.jsonl")
    print("- val_rft_items.jsonl")

    if train_items:
        print("\nQuick checks:")
        print("messages[0].role =", train_items[0]["messages"][0]["role"])
        print("policy_speaker   =", train_items[0]["policy_speaker"])
        print("opponent_speaker =", train_items[0]["opponent_speaker"])
        print("\nPreview grader_user:\n", train_items[0]["grader_user"][:600], " ...")

if __name__ == "__main__":
    main()
