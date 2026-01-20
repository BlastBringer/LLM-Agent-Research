#!/usr/bin/env python3
"""
Download and preprocess SVAMP dataset from Hugging Face
===================================================

SVAMP (Simpler Variations of Math Problems) - A challenging benchmark for math reasoning
with simple variations of math problems designed to test compositional generalization.

Downloads from: https://huggingface.co/datasets/ChilleD/SVAMP
Extracts: Question_Concat (input), Answer (output)
Output: JSONL format with columns: input, output
"""

import json
import argparse
from pathlib import Path
from datasets import load_dataset
from typing import Dict, Any


def download_and_preprocess_svamp(
    output_dir: Path = None,
    output_filename: str = "svamp_dataset.jsonl",
    only_test: bool = True
) -> None:
    """
    Download SVAMP dataset from Hugging Face and preprocess to JSONL format.
    
    Args:
        output_dir: Directory to save preprocessed dataset (default: current dir)
        output_filename: Name of output JSONL file
        only_test: If True, only use test split (default: True)
    """
    if output_dir is None:
        output_dir = Path.cwd()
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / output_filename
    
    print("🔍 Downloading SVAMP dataset from Hugging Face...")
    print("   This may take a moment on first download...\n")
    
    try:
        # Load dataset
        ds = load_dataset("ChilleD/SVAMP")
        print(f"✓ Dataset loaded successfully!")
        print(f"  Available splits: {list(ds.keys())}\n")
        
        # Get first example to inspect columns
        test_split = ds['test']
        if test_split:
            first_example = test_split[0]
            print(f"📋 Available columns in test split:")
            for col in first_example.keys():
                print(f"   - {col}")
            print()
        
        # Process only test split
        all_problems = []
        
        split_name = 'test'
        split_dataset = ds[split_name]
        print(f"Processing '{split_name}' split ({len(split_dataset)} problems)...")
        
        # Inspect first few examples to find correct column names
        print(f"\n🔍 Inspecting first example:")
        if len(split_dataset) > 0:
            first = split_dataset[0]
            print(f"   Example keys: {list(first.keys())}")
            for key, val in first.items():
                print(f"   - {key}: {str(val)[:100]}")
        
        for idx, example in enumerate(split_dataset):
            # Try multiple possible column name combinations
            question = None
            answer = None
            
            # Try to find question
            for q_col in ['question_concat', 'Question_Concat', 'question', 'Question']:
                if q_col in example and example[q_col]:
                    question = example[q_col]
                    break
            
            # Try to find answer
            for a_col in ['Answer', 'answer', 'output', 'Output']:
                if a_col in example and example[a_col] is not None:
                    answer = str(example[a_col]).strip()
                    break
            
            # Create problem if both fields found
            if question and answer:
                problem = {
                    "input": str(question).strip(),
                    "output": answer,
                    "split": split_name
                }
                all_problems.append(problem)
            elif idx < 5:
                print(f"  ⚠ Row {idx}: Could not find fields")
                print(f"     Keys available: {list(example.keys())}")
        
        # Save to JSONL
        print(f"\n✓ Saving {len(all_problems)} problems to {output_path}...")
        
        with open(output_path, 'w') as f:
            for problem in all_problems:
                json.dump(problem, f)
                f.write('\n')
        
        print(f"✓ Successfully saved to: {output_path}\n")
        
        # Print statistics
        print("📊 Dataset Statistics:")
        print(f"   Total problems: {len(all_problems)}")
        
        # Show sample
        print("\n📝 Sample Problems:")
        if all_problems:
            for i, sample in enumerate(all_problems[:3]):
                print(f"\n   Problem {i+1}:")
                print(f"   Input:  {sample['input'][:80]}...")
                print(f"   Output: {sample['output']}")
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("   Make sure you have internet connection and datasets library installed")
        print("   Install with: pip install datasets")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Download and preprocess SVAMP dataset from Hugging Face"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save preprocessed dataset (default: current directory)"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="svamp_dataset.jsonl",
        help="Name of output JSONL file (default: svamp_dataset.jsonl)"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    download_and_preprocess_svamp(
        output_dir=output_dir,
        output_filename=args.output_file
    )


if __name__ == "__main__":
    main()
