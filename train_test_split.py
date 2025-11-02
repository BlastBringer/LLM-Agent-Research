#!/usr/bin/env python3
"""
🎯 TRAINING DATA SPLITTER
=========================

Split training data into train/validation sets to prevent catastrophic forgetting.

Features:
- Stratified split by difficulty (maintains distribution)
- Preserves metadata
- Generates statistics

Usage:
    python train_test_split.py \
        --input solver_training_data.jsonl \
        --val-split 0.2 \
        --seed 42
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

def load_jsonl(filepath: str) -> List[Dict]:
    """Load JSONL file"""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def save_jsonl(data: List[Dict], filepath: str):
    """Save to JSONL file"""
    with open(filepath, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    print(f"   📁 Saved {len(data)} examples to {filepath}")

def stratified_split(
    data: List[Dict], 
    val_split: float = 0.2, 
    difficulty_bins: int = 5
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split data with stratification by difficulty to maintain distribution.
    
    This prevents distributional drift - ensures validation set has
    same mix of easy/hard problems as training set.
    
    Args:
        data: List of training examples
        val_split: Fraction for validation (0.0-1.0)
        difficulty_bins: Number of difficulty levels to stratify by
    
    Returns:
        (train_data, val_data) tuple
    """
    # Group by difficulty bins
    binned = [[] for _ in range(difficulty_bins)]
    
    for item in data:
        # Get difficulty score (default 0.5 if not present)
        difficulty = item.get('metadata', {}).get('difficulty', 0.5)
        
        # Assign to bin
        bin_idx = min(int(difficulty * difficulty_bins), difficulty_bins - 1)
        binned[bin_idx].append(item)
    
    train, val = [], []
    
    # Split each bin proportionally
    for bin_idx, bin_data in enumerate(binned):
        if not bin_data:
            continue
        
        random.shuffle(bin_data)
        split_idx = int(len(bin_data) * (1 - val_split))
        
        train.extend(bin_data[:split_idx])
        val.extend(bin_data[split_idx:])
        
        print(f"   Bin {bin_idx} (difficulty {bin_idx/difficulty_bins:.1f}-{(bin_idx+1)/difficulty_bins:.1f}): "
              f"{split_idx} train, {len(bin_data)-split_idx} val")
    
    return train, val

def analyze_distribution(data: List[Dict], name: str):
    """Analyze and print dataset statistics"""
    if not data:
        print(f"\n⚠️  {name}: No data")
        return
    
    # Extract difficulties
    difficulties = [item.get('metadata', {}).get('difficulty', 0.5) for item in data]
    
    # Extract sources
    sources = [item.get('metadata', {}).get('source', 'unknown') for item in data]
    source_counts = defaultdict(int)
    for s in sources:
        source_counts[s] += 1
    
    # Extract review flags
    needs_review = sum(1 for item in data 
                      if item.get('metadata', {}).get('needs_review', False))
    
    print(f"\n📊 {name} Statistics:")
    print(f"   Total examples: {len(data)}")
    print(f"   Avg difficulty: {sum(difficulties)/len(difficulties):.3f}")
    print(f"   Min difficulty: {min(difficulties):.3f}")
    print(f"   Max difficulty: {max(difficulties):.3f}")
    print(f"   Sources: {dict(source_counts)}")
    print(f"   Flagged for review: {needs_review} ({needs_review/len(data)*100:.1f}%)")

def main():
    parser = argparse.ArgumentParser(
        description='Split training data into train/validation sets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic split (80/20)
  python train_test_split.py --input solver_training_data.jsonl

  # Custom split ratio
  python train_test_split.py --input solver_training_data.jsonl --val-split 0.15

  # Different random seed
  python train_test_split.py --input solver_training_data.jsonl --seed 123
        """
    )
    
    parser.add_argument('--input', type=str, default='solver_training_data.jsonl',
                       help='Input JSONL file with training data')
    parser.add_argument('--train-output', type=str, default='solver_training_train.jsonl',
                       help='Output file for training set')
    parser.add_argument('--val-output', type=str, default='solver_training_val.jsonl',
                       help='Output file for validation set')
    parser.add_argument('--val-split', type=float, default=0.2,
                       help='Fraction of data for validation (0.0-1.0)')
    parser.add_argument('--difficulty-bins', type=int, default=5,
                       help='Number of difficulty bins for stratification')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not (0.0 < args.val_split < 1.0):
        print(f"❌ Error: --val-split must be between 0.0 and 1.0")
        return 1
    
    if not Path(args.input).exists():
        print(f"❌ Error: Input file not found: {args.input}")
        return 1
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    print("=" * 70)
    print("🎯 TRAINING DATA SPLITTER")
    print("=" * 70)
    
    # Load data
    print(f"\n📂 Loading {args.input}...")
    data = load_jsonl(args.input)
    print(f"   Total examples: {len(data)}")
    
    if len(data) == 0:
        print("❌ No data to split!")
        return 1
    
    # Analyze original data
    analyze_distribution(data, "Original Data")
    
    # Perform stratified split
    print(f"\n🔀 Performing stratified split ({(1-args.val_split)*100:.0f}% train, {args.val_split*100:.0f}% val)...")
    train, val = stratified_split(data, args.val_split, args.difficulty_bins)
    
    # Save splits
    print(f"\n💾 Saving splits...")
    save_jsonl(train, args.train_output)
    save_jsonl(val, args.val_output)
    
    # Analyze splits
    analyze_distribution(train, "Training Set")
    analyze_distribution(val, "Validation Set")
    
    print("\n" + "=" * 70)
    print("✅ SPLIT COMPLETE")
    print("=" * 70)
    print(f"\nNext steps:")
    print(f"  1. Use {args.train_output} for fine-tuning")
    print(f"  2. Use {args.val_output} for evaluation")
    print(f"  3. Monitor validation metrics to prevent overfitting")
    
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())
