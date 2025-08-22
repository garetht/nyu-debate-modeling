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

import json

with open('debates-readable.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]

indexes_count = {}
k = 0
k_50 = 0
lines = []
winner_A = 0
winner_B = 0
last_tie_won = 'B'
first_B = 0
first_B_max = 79

for d in data:
    turns = d['turns']
    indexes = []
    for t in turns:
        if t['index'] == None:
            t['index'] = -1
        if t['index'] not in indexes:
            indexes.append(t['index'])
        
    indexes = sorted(indexes)
    if str(indexes) not in indexes_count:
        indexes_count[str(indexes)] = 1
    else:
        indexes_count[str(indexes)] += 1
    
    if len(indexes) == 3:
        topic = d['question']
        position_a = d['answers'][0]
        position_b = d['answers'][1]
        speeches_a = []
        speeches_b = []
        judge_decisions = []
        for i in range(len(turns)):
            t = turns[i]
            if i == 0 and t['index'] == -1:
                continue
            if t['index'] == 0:
                speeches_a.append(t['text'])
            elif t['index'] == 1:
                speeches_b.append(t['text'])
            elif t['index'] == -1 and t['probabilities']:
                t['probabilities'][0] = round(t['probabilities'][0], 2)
                t['probabilities'][1] = round(t['probabilities'][1], 2)
                try:
                    if t['probabilities'][0] > t['probabilities'][1]:
                        winner = 'A'
                    elif t['probabilities'][0] < t['probabilities'][1]:
                        winner = 'B'
                except:
                    print(t['probabilities'])
                    print(t)
                p = max(t['probabilities'])
                if p == 0.5:
                    k_50 += 1
                    if last_tie_won == 'A':
                            winner = last_tie_won = 'B'
                    else:
                        winner = last_tie_won = 'A'
                judge_decisions.append(f'Debater {winner} | {int(p*100)}%')
                if winner == 'A':
                    winner_A += 1
                else:
                    winner_B += 1
        if len(speeches_a) == len(speeches_b) == len(judge_decisions):
            all_speeches = ''
            for i in range(len(speeches_a)):
                speeches = SPEECHES.replace(SPEECH_A, speeches_a[i]).replace(SPEECH_B, speeches_b[i])
                all_speeches += speeches + '\n'
                prompt = PROMPT.replace(TOPIC, topic).replace(POSITION_A, position_a).replace(POSITION_B, position_b)
                prompt = prompt.replace('SPEECHES', all_speeches)
                lines.append({'prompt': prompt, 'completion': judge_decisions[i]})

import random
lines = [d for d in lines if d not in random.sample([x for x in lines if x['completion'].startswith('Debater B')], first_B_max)]

with open("human_debate_for_judge.jsonl", "w") as f:
    for line in lines:
        f.write(json.dumps(line) + "\n")
        
        
        #print(speeches_a)
        #print()
        #print(speeches_b)
        #print()
        #print(judge_decisions)
        #break
print(k)
print(indexes_count)
print(f'num of 0.5 certainty: {k_50}')
print(f'num of times A wins: {winner_A}, num of times B wins: {winner_B}')



