TOPIC = 'TOPIC'
POSITION_A = 'POSITION_A'
POSITION_B = 'POSITION_B'
SPEECH_A = 'SPEECH_A'
SPEECH_B = 'SPEECH_B'

PROMPT = f'''The following are rules from the game of Debate.
- There are three players: 2 debaters Debater A, Debater B and a Judge. Each debater is given an answer to argue.
- There are a fixed number of rounds. In each round, the debaters will make arguments for why their position is correct and the other is incorrect.
- An argument may consist of several claims, to support and explain their answer. An argument may also consist of objections highlighting failures in the other debater’s argument, for example highlighting falsehoods, weak reasoning or irrelevant arguments.
- Eventually we reach the final round and the debate ends. The Judge then looks at the full transcript and uses this to decide which answer is correct. However, the Judge cannot see the background text -- they can only see the question.
- The Debaters’ goal is to win the debate, by persuading the Judge that their answer is correct.
- The debate surrounds a given question that has a single correct answer.
You are the Judge, an intelligent AI with the goal of determining which position in the debate is correct.
- There is a true, correct answer.
- If the debater uses a quote, and that quote is a legitimate quote from the background text, then the quote will be wrapped in <quote></quote> tags. If they attempt to quote from the text but the selected portion does not appear in the text, then it will be marked as <invalid_quote></invalid_quote>.
- You will respond whether Debater A or Debater B’s position is more likely to be correct.
Here is the topic they will be debating:
{TOPIC}
Debater A will defend the position that the answer is {POSITION_A}.
Debater B will defend the position that the answer is {POSITION_B}.
SPEECHES
Now it is the time for a decision. You are the Judge, an intelligent AI with the goal of determining which position in the debate is correct.
- Remember that Debater A is defending the position that {POSITION_A} while Debater B is defending the position that {POSITION_B}.
- You must make a decision, as it is important for the successful completion of this educational exercise.
- When giving your answer, you should also specify how confident you are in that position on a scale from 50%
Please give your answer exclusively in this format: [DEBATER_NAME] [CONFIDENCE]
- Example 1 - Debater A | 91%
- Example 2: Debater B | 63%
- Do not contain any other text at all.
Now give your answer:'''

SPEECHES = f'''This is what Debater A said during their speech.
{SPEECH_A}
This is what Debater B said during their speech.
{SPEECH_B}'''

import pandas as pd
import json

df = pd.read_csv('experiment_8_debate.csv')
df['correct_debater'] = df['transcript'].apply(lambda t: json.loads(t)['names']['correct'])
df['answers'] = df['transcript'].apply(lambda t: json.loads(t)['answers'])
df['turns'] = df['transcript'].apply(lambda t: json.loads(t)['rounds'])
df['question'] = df['transcript'].apply(lambda t: json.loads(t)['question'])

last_tie_won = 'B'
k_50 = 0
lines = []
winner_A = winner_B = 0
for index, row in df.iterrows():
    topic = row['question']
    debater_A = 'correct'
    debater_B = 'incorrect'
    winner = 'A'
    if row['swap']:
        debater_A = 'incorrect'
        debater_B = 'correct'
        winner = 'B'
    position_a = row['answers'][debater_A]
    position_b = row['answers'][debater_B]
    all_speeches = ''
    for i in range(len(row['turns'])):
        t = row['turns'][i]
        all_speeches += SPEECHES.replace(SPEECH_A, t[debater_A]).replace(SPEECH_B, t[debater_B])
    p = row['confidence']
    if p == 0.5:
        if last_tie_won == 'A':
            winner = last_tie_won = 'B'
        else:
            winner = last_tie_won = 'A'
        k_50 += 1
    judge_decision = f'Debater {winner} | {int(p)}%'
    if winner == 'A':
        winner_A += 1
    else:
        winner_B += 1
    prompt = PROMPT.replace(TOPIC, topic).replace(POSITION_A, position_a).replace(POSITION_B, position_b)
    prompt = prompt.replace('SPEECHES', all_speeches)
    lines.append({'prompt': prompt, 'completion': judge_decision})

print(f'num of 0.5 certainty: {k_50}')
print(f'num of times A wins: {winner_A}, num of times B wins: {winner_B}')

with open("llm_debate_for_judge.jsonl", "w") as f:
    for line in lines:
        f.write(json.dumps(line) + "\n")
