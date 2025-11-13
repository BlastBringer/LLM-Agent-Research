#!/usr/bin/env python3
"""
Split MetaMathQA preprocessed JSONL into multiple smaller batches.

Input: metamathqa_preprocessed.jsonl
Output folder: metamathqa_batches/
Each batch: 10,000 samples (configurable)

Example usage:
    python3 split_metamathqa_batches.py --input metamathqa_preprocessed.jsonl --output-dir metamathqa_batches --batch-size 10000
"""

import json
import argparse
from pathlib import Path


def split_jsonl(input_path: str, output_dir: str, batch_size: int = 10000):
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📂 Reading input file: {input_path}")
    with open(input_path, "r", encoding="utf-8") as infile:
        lines = infile.readlines()

    total = len(lines)
    print(f"✅ Total records: {total}")

    num_batches = (total + batch_size - 1) // batch_size
    print(f"📦 Creating {num_batches} batches of up to {batch_size} records each...")

    for i in range(num_batches):
        start = i * batch_size
        end = min(start + batch_size, total)
        batch_lines = lines[start:end]

        batch_file = output_dir / f"batch_{i+1:03d}.jsonl"
        with open(batch_file, "w", encoding="utf-8") as outfile:
            outfile.writelines(batch_lines)

        print(f"✅ Saved {len(batch_lines)} records → {batch_file}")

    print(f"\n🎯 Done! All batches saved in: {output_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split MetaMathQA JSONL into 10k-sized batches")
    parser.add_argument("--input", type=str, default="metamathqa_preprocessed.jsonl",
                        help="Path to the preprocessed MetaMathQA JSONL file")
    parser.add_argument("--output-dir", type=str, default="metamathqa_batches",
                        help="Directory to store output batch files")
    parser.add_argument("--batch-size", type=int, default=10000,
                        help="Number of samples per batch (default: 10000)")

    args = parser.parse_args()
    split_jsonl(args.input, args.output_dir, args.batch_size)
