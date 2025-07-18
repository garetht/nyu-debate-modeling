#!/usr/bin/env python3
import json
from pathlib import Path


QUALITY_DIR = Path.home() / "study" / "debate" / "repo" / "nyu-debate-modeling" / "data" / "datasets" / "quality"

SCRIPT_DIR        = Path(__file__).parent
INPUT_FILE        = "converted_khan_with_consultancy.jsonl"
OUTPUT_FILE       = "converted_khan_only_with_consultancy_filled.jsonl"


def build_quality_lookup(quality_dir: Path):
    """
    Read all htmlstripped.{train,dev,test} files and build
    a dict mapping title -> article_text.
    """
    lookup = {}
    for split in ("train", "dev", "test"):
        fn = quality_dir / f"QuALITY.v1.0.1.htmlstripped.{split}"
        if not fn.exists():
            raise FileNotFoundError(f"Missing QuALITY file: {fn}")
        with fn.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                title   = obj.get("title")
                article = obj.get("article")
                if title and article:
                    # keep first-seen version
                    lookup.setdefault(title, article)
    return lookup

def fill_stories(input_path: Path, output_path: Path, lookup: dict):
    """
    Read each record from input_path, replace placeholder stories
    using lookup, and write to output_path.
    """
    with open(input_path, "r", encoding="utf-8") as inf, \
         open(output_path, "w", encoding="utf-8") as outf:
        for line in inf:
            rec = json.loads(line)
            if rec.get("story") == "Removed for public dataset release.":
                title = rec.get("storyTitle")
                full  = lookup.get(title)
                if full:
                    rec["story"] = full
                else:
                    print(f"[WARN] No QuALITY text for title: {title}")
            outf.write(json.dumps(rec, ensure_ascii=False) + "\n")

def main():
    print(f"Loading QuALITY texts from {QUALITY_DIR}…")
    quality_lookup = build_quality_lookup(QUALITY_DIR)
    print(f"Found {len(quality_lookup)} articles in QuALITY.")

    print(f"Filling stories in {INPUT_FILE} → {OUTPUT_FILE}…")
    fill_stories(INPUT_FILE, OUTPUT_FILE, quality_lookup)
    print("Done!")

if __name__ == "__main__":
    main()
