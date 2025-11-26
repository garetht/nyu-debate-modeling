import unicodedata
import re

# Mapping of common “ambiguous” Unicode → ASCII
CONFUSABLES = {
    '\u2018': "'",  # left single curly quote
    '\u2019': "'",  # right single curly quote
    '\u201C': '"',  # left double curly quote
    '\u201D': '"',  # right double curly quote
    '\u2013': '-',  # en dash
    '\u2014': '-',  # em dash
    '\u2026': '...',# ellipsis
    '\u00A0': ' ',  # non-breaking space
}

def normalize_line(line: str) -> str:
    line = unicodedata.normalize('NFKC', line)
    for src, tgt in CONFUSABLES.items():
        line = line.replace(src, tgt)
    return line

def fix_jsonl(input_path: str, output_path: str):
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        for raw in fin:
            fixed = normalize_line(raw)
            fout.write(fixed)

if __name__ == "__main__":
    fix_jsonl("judge/train_combined.jsonl",
              "judge/train_combined_fixed.jsonl")
    print("Wrote cleaned file with ASCII‐only equivalents.")

