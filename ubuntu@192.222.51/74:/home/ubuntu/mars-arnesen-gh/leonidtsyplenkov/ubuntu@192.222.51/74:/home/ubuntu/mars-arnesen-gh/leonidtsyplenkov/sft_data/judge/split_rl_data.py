#!/usr/bin/env python3
import json
import random
import argparse

def split_jsonl(input_path: str, train_path: str, test_path: str, train_ratio: float = 0.8):
    # Read all lines
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # Shuffle lines
    random.shuffle(lines)

    # Compute split index
    split_idx = int(len(lines) * train_ratio)
    train_lines = lines[:split_idx]
    test_lines = lines[split_idx:]

    # Write train split
    with open(train_path, "w", encoding="utf-8") as f:
        f.writelines(train_lines)
    
    # Write test split
    with open(test_path, "w", encoding="utf-8") as f:
        f.writelines(test_lines)

    print(f"Split {len(lines)} lines → {len(train_lines)} train, {len(test_lines)} test")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split JSONL file into train/test splits")
    parser.add_argument("input", help="Path to input JSONL file")
    parser.add_argument("--train", default="train.jsonl", help="Output file for 80% split")
    parser.add_argument("--test", default="test.jsonl", help="Output file for 20% split")
    parser.add_argument("--ratio", type=float, default=0.8, help="Train split ratio (default: 0.8)")
    args = parser.parse_args()

    split_jsonl(args.input, args.train, args.test, args.ratio)


