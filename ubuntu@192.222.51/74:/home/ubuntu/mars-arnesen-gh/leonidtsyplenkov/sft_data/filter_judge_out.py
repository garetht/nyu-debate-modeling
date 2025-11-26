import json

with open('debates-readable.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]

new_data = []

for d in data:
    turns = d['turns']
    indexes = []
    for t in turns:
        if t['index'] == None:
            t['index'] = -1
        if t['index'] not in indexes:
            indexes.append(t['index'])
    if len(indexes) == 3:
        speeches = []
        for i in range(len(turns)):
            t = turns[i]
            if t['index'] == 0 or t['index'] == 1:
                speeches.append(t)
        if turns[-1]['role'].lower() == 'judge':
            speeches.append(turns[-1])
        if turns[-2]['role'].lower() == 'judge':
            speeches.append(turns[-2])
        new_line = d
        new_line['turns'] = speeches
        new_data.append(new_line)

with open("human_debate_judge_removed.jsonl", "w") as f:
    for line in new_data:
        f.write(json.dumps(line) + "\n")

with open('converted_khan_only_debate_filled.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]

new_data = []

for d in data:
    turns = d['turns']
    indexes = []
    for t in turns:
        if t['index'] == None:
            t['index'] = -1
        if t['index'] not in indexes:
            indexes.append(t['index'])
    if len(indexes) >= 3:
        speeches = []
        for i in range(len(turns)):
            t = turns[i]
            if t['index'] != -1:
                new_index = 0 if t['index'] % 2 == 1 else 1
                t['index'] = new_index
                speeches.append(t)
        if turns[-1]['role'].lower() == 'judge':
            speeches.append(turns[-1])
        new_line = d
        new_line['turns'] = speeches
        new_data.append(new_line)


new_data = new_data[:239]
with open("llm_debate_judge_removed_239.jsonl", "w") as f:
    for line in new_data:
        f.write(json.dumps(line) + "\n")
            