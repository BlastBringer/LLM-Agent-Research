#!/usr/bin/env python3
"""
Preprocess MetaMathQA dataset for training/evaluation.

Original columns: type, query, original question, response
Output columns: input, solution_steps, output

- Drops: type, original question
- Renames: query → input, response → solution_steps
- Extracts final answer from lines like "The answer is: 362000" or "The answer is: \\sqrt{5}"
"""

from datasets import load_dataset
import pandas as pd
import re
from pathlib import Path


def extract_answer(text: str):
    """Extract the final answer from a MetaMathQA response."""
    if not isinstance(text, str):
        return None

    # Common pattern in MetaMathQA responses
    match = re.search(r"The\s*answer\s*is\s*[:\-]?\s*(.*)", text, re.IGNORECASE)
    if not match:
        return None

    answer = match.group(1).strip()

    # Remove trailing punctuation or markdown/LaTeX delimiters
    answer = re.sub(r'[\.$#\s]+$', '', answer)

    # Handle LaTeX wrappers like \boxed{}, \( \), or $$...$$
    answer = re.sub(r"\\boxed\{(.*?)\}", r"\1", answer)
    answer = re.sub(r"^\$+|\$+$", "", answer)
    answer = answer.strip()

    return answer


def preprocess_metamathqa(save_path: str = "metamathqa_preprocessed.jsonl"):
    print("📥 Loading MetaMathQA dataset from Hugging Face...")
    dataset = load_dataset("meta-math/MetaMathQA", split="train")

    print(f"✅ Loaded {len(dataset)} examples.")

    # Convert to pandas DataFrame for easier processing
    df = dataset.to_pandas()

    # Drop unused columns
    df = df.drop(columns=["type", "original_question"], errors="ignore")

    # Rename columns
    df = df.rename(columns={
        "query": "input",
        "response": "solution_steps"
    })

    # Extract answers
    print("🔍 Extracting final answers...")
    df["output"] = df["solution_steps"].apply(extract_answer)

    # Drop rows where no answer was found
    before = len(df)
    df = df.dropna(subset=["output"])
    print(f"🧹 Dropped {before - len(df)} rows without detected answers.")

    # Save as JSONL
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_json(save_path, orient="records", lines=True, force_ascii=False)
    print(f"💾 Saved preprocessed dataset to: {save_path}")
    print(df.head(3))


if __name__ == "__main__":
    preprocess_metamathqa("metamathqa_preprocessed.jsonl")
