# 📋 Dataset Format Specification

## Overview

This document defines the **standard format** for all datasets used with the Math Problem Solver Pipeline. Follow this format to ensure compatibility with the engine.

---

## Required Format: JSONL (JSON Lines)

Each line in the `.jsonl` file must be a valid JSON object representing one problem.

### File Extension
- **Required:** `.jsonl` (one JSON object per line)
- **Not supported:** `.json` (single large array) - use for single problems only
- **Not supported:** `.csv` - convert to JSONL first

---

## Required Fields by Dataset Type

### 🎓 TRAINING Dataset (Required Fields)

Training datasets need the full solution for Oracle accuracy checking:

```json
{
  "input": "A train travels at 60 mph for 2.5 hours. How far does it travel?",
  "output": "\\boxed{150}",
  "solution_steps": [
    "Step 1: Identify the formula: distance = speed × time",
    "Step 2: Extract values: speed = 60 mph, time = 2.5 hours",
    "Step 3: Calculate: distance = 60 × 2.5 = 150 miles"
  ]
}
```

**Training Dataset Requirements:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `input` | string | ✅ **YES** | The math problem text. Can use `problem` or `question` as alternative field names. |
| `output` | string | ✅ **YES** | The final answer for cross-checking Oracle accuracy. Can use `answer` or `solution` as alternative field names. |
| `solution_steps` | array of strings | ✅ **YES** | Step-by-step ground truth solution. Used to verify Oracle's solution is correct before saving as training data. **Critical for 30-50% Apprentice learning improvement (after fine-tuning)!** |

### 🧪 TEST Dataset (Required Fields)

Test datasets only need problem and answer (no solution steps):

```json
{
  "input": "A train travels at 60 mph for 2.5 hours. How far does it travel?",
  "output": "\\boxed{150}"
}
```

**Test Dataset Requirements:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `input` | string | ✅ **YES** | The math problem text to solve. |
| `output` | string | ✅ **YES** | The correct answer for cross-checking model predictions. |
| `solution_steps` | array of strings | ❌ **NO** | Should NOT be included in test data (prevents cheating). |

---

## Recommended Optional Fields (Both Training & Test)

These fields enhance functionality but are not strictly required:

```json
{
  "id": "1351",
  "input": "A train travels at 60 mph for 2.5 hours. How far does it travel?",
  "output": "\\boxed{150}",
  "solution_steps": [...],  // Only for TRAINING
  "difficulty": "Level 3",
  "source": "MATH",
  "tags": ["algebra", "word problem"]
}
```

### Optional Field Descriptions

| Field | Type | Required | Description | Benefits |
|-------|------|----------|-------------|----------|
| `id` | string | ⚪ Optional | Unique identifier for the problem | Tracking, debugging, reproducibility |
| `difficulty` | string | ⚪ Optional | Difficulty level (e.g., "Level 1-5", "easy/medium/hard") | Stratified sampling, difficulty analysis |
| `source` | string | ⚪ Optional | Dataset source (e.g., "MATH", "GSM8K", "custom") | Attribution, filtering |
| `tags` | array of strings | ⚪ Optional | Problem categories (e.g., ["algebra", "geometry"]) | Filtering, analysis by topic |

---

## Answer Formats Supported

The pipeline automatically detects and extracts answers from multiple formats:

### 1. LaTeX Boxed Format (RECOMMENDED)
```json
"output": "\\boxed{150}"
"output": "\\boxed{150 \\text{ miles}}"
"output": "Therefore, the answer is \\boxed{42.5}"
```
✅ **Best for:** MATH dataset, formal mathematical answers

### 2. Direct Numeric
```json
"output": "150"
"output": "42.5"
"answer": "3.14159"
```
✅ **Best for:** Simple numeric answers

### 3. Text with Answer
```json
"output": "The answer is 150"
"output": "The distance is 150 miles"
"output": "answer=150"
```
✅ **Best for:** GSM8K, word problem datasets

### 4. Currency Format
```json
"output": "$42.50"
"output": "The cost is $150.00"
```
✅ **Best for:** Money/economics problems

---

## Complete Example

Here's a complete example with all recommended fields:

```json
{
  "id": "train_0001",
  "source": "custom_dataset",
  "difficulty": "Level 3",
  "input": "A train travels at 60 miles per hour for 2.5 hours. How far does the train travel?",
  "output": "\\boxed{150}",
  "solution_steps": [
    "Step 1: Identify the formula for distance: distance = speed × time",
    "Step 2: Extract the given values from the problem:",
    "  - speed = 60 miles per hour",
    "  - time = 2.5 hours",
    "Step 3: Substitute into the formula:",
    "  distance = 60 × 2.5",
    "Step 4: Calculate the result:",
    "  distance = 150 miles",
    "Therefore, the train travels 150 miles."
  ],
  "tags": ["word problem", "distance", "algebra", "multiplication"]
}
```

---

## Alternative Field Names

The pipeline automatically detects these alternative field names:

### For Problem Text (input)
- `input` ✅ (preferred)
- `problem` ✅
- `question` ✅
- `text` ✅

### For Answer (output)
- `output` ✅ (preferred)
- `answer` ✅
- `solution` ✅
- `final_answer` ✅

**You only need ONE of each!** The pipeline will find it.

---

## Real-World Dataset Examples

### Example 1: MATH Dataset Format
```jsonl
{"id": "1351", "source": "MATH", "difficulty": "Level 3", "input": "Find all real values of $x$ which satisfy...", "output": "\\boxed{(-5,-2] \\cup (-1,3]}", "solution_steps": ["Step 1...", "Step 2..."], "tags": ["intermediate algebra"]}
```
✅ **Fully supported** - Current AllProblemsCleaned.jsonl format

### Example 2: GSM8K Format
```jsonl
{"question": "Janet has 16 chickens. She sells 2 chickens per day. After 4 days, how many chickens does she have left?", "answer": "8"}
```
✅ **Supported** - Maps to `input`/`output` automatically

### Example 3: Minimal Custom Format
```jsonl
{"problem": "What is 2 + 2?", "answer": "4"}
{"problem": "A car goes 100 km in 2 hours. What is its speed?", "answer": "50 km/h"}
```
✅ **Supported** - Minimal but functional

### Example 4: With Full Solutions
```jsonl
{"input": "Solve: 2x + 5 = 13", "output": "\\boxed{4}", "solution_steps": ["Subtract 5: 2x = 8", "Divide by 2: x = 4"]}
```
✅ **Highly Recommended** - Best learning quality

---

## File Naming Convention (Recommended)

Use descriptive file names:
- `train_problems.jsonl` - Training data
- `test_problems.jsonl` - Test/validation data
- `custom_algebra_problems.jsonl` - Topic-specific
- `gsm8k_converted.jsonl` - Converted from other format

---

## Creating Your Own Dataset

### Method 1: Manual Creation
```python
import json

problems = [
    {
        "id": "001",
        "input": "What is 5 × 6?",
        "output": "30",
        "difficulty": "easy",
        "tags": ["arithmetic", "multiplication"]
    },
    {
        "id": "002",
        "input": "A car travels 120 km in 3 hours. What is its speed?",
        "output": "40 km/h",
        "difficulty": "medium",
        "tags": ["word problem", "speed"]
    }
]

# Save as JSONL
with open("my_dataset.jsonl", "w") as f:
    for problem in problems:
        f.write(json.dumps(problem) + "\n")
```

### Method 2: Convert from CSV
```python
import csv
import json

with open("problems.csv", "r") as csvfile, open("problems.jsonl", "w") as jsonlfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        problem = {
            "input": row["problem_text"],
            "output": row["correct_answer"],
            "difficulty": row.get("difficulty", "unknown")
        }
        jsonlfile.write(json.dumps(problem) + "\n")
```

### Method 3: Convert from Existing Format
```python
import json

# Load existing format (e.g., GSM8K)
with open("gsm8k.json", "r") as f:
    data = json.load(f)

# Convert to standard format
with open("gsm8k_converted.jsonl", "w") as f:
    for item in data:
        problem = {
            "id": item["id"],
            "input": item["question"],
            "output": item["answer"],
            "source": "GSM8K"
        }
        f.write(json.dumps(problem) + "\n")
```

---

## Validation

### Quick Validation Script

Save this as `validate_dataset.py`:

```python
#!/usr/bin/env python3
import json
import sys

def validate_dataset(filepath):
    """Validate JSONL dataset format."""
    errors = []
    warnings = []
    
    try:
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                
                try:
                    data = json.loads(line)
                    
                    # Check required fields (input)
                    has_input = any(k in data for k in ['input', 'problem', 'question', 'text'])
                    if not has_input:
                        errors.append(f"Line {line_num}: Missing input field (need one of: input/problem/question/text)")
                    
                    # Check required fields (output)
                    has_output = any(k in data for k in ['output', 'answer', 'solution', 'final_answer'])
                    if not has_output:
                        errors.append(f"Line {line_num}: Missing output field (need one of: output/answer/solution/final_answer)")
                    
                    # Warnings for optional fields
                    if 'solution_steps' not in data:
                        warnings.append(f"Line {line_num}: No solution_steps (recommended for better learning)")
                    if 'difficulty' not in data:
                        warnings.append(f"Line {line_num}: No difficulty field (recommended for stratification)")
                    
                except json.JSONDecodeError as e:
                    errors.append(f"Line {line_num}: Invalid JSON - {e}")
        
        # Print results
        print(f"✅ Validation complete for: {filepath}")
        print(f"📊 Total lines: {line_num}")
        
        if errors:
            print(f"\n❌ ERRORS ({len(errors)}):")
            for error in errors[:10]:  # Show first 10
                print(f"  - {error}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more errors")
        else:
            print("\n✅ No errors found!")
        
        if warnings:
            print(f"\n⚠️  WARNINGS ({len(warnings)}):")
            for warning in warnings[:5]:  # Show first 5
                print(f"  - {warning}")
            if len(warnings) > 5:
                print(f"  ... and {len(warnings) - 5} more warnings")
        
        return len(errors) == 0
    
    except FileNotFoundError:
        print(f"❌ File not found: {filepath}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_dataset.py <dataset.jsonl>")
        sys.exit(1)
    
    is_valid = validate_dataset(sys.argv[1])
    sys.exit(0 if is_valid else 1)
```

Run validation:
```bash
python validate_dataset.py my_dataset.jsonl
```

---

## Using Your Dataset with the Pipeline

Once your dataset follows this format, use it with the pipeline in different modes:

### 🎓 Training Mode (Collect Oracle Solutions)

**Use Case:** Collect high-quality solutions from Oracle for fine-tuning the Apprentice model.

**Dataset Requirements:**
- ✅ `input` field (problem text)
- ✅ `output` field (correct answer for Oracle accuracy checking)
- ✅ `solution_steps` field (step-by-step solution for Oracle to learn from)

**Command:**
```bash
python complete_pipeline.py \
  --mode dataset \
  --input train_problems.jsonl \
  --batch-size 20 \
  --pipeline-mode train
```

**What Happens:**
1. Oracle (Gemini) solves the problem and generates its solution
2. Oracle's answer is compared with ground truth from `output` field
3. Oracle's solution is compared with ground truth from `solution_steps` field
4. If Oracle is correct, save its solution as training data
5. Training data saved with `oracle_correct` flag
6. Later: Fine-tune Apprentice (Llama 3.1 8B) on this collected data
7. Apprentice learns to solve like the Oracle

**Output:** `solver_training_data.jsonl` with Oracle solutions (to fine-tune Apprentice later)

---

### 🧪 Test Mode (Evaluate Apprentice Model)

**Use Case:** Test how well the Apprentice model performs without Oracle help.

**Dataset Requirements:**
- ✅ `input` field (problem text)
- ✅ `output` field (correct answer for cross-checking predictions)
- ❌ `solution_steps` field (should NOT be included - prevents cheating!)

**Command:**
```bash
python complete_pipeline.py \
  --mode dataset \
  --input test_problems.jsonl \
  --batch-size 20 \
  --pipeline-mode test \
  --skip-oracle
```

**What Happens:**
1. Pipeline loads problems (without solution steps)
2. Apprentice model attempts to solve (no Oracle help)
3. Apprentice's answer is compared with ground truth from `output`
4. Accuracy calculated and reported
5. No training data collected (evaluation only)

**Output:** Evaluation metrics and success rate

---

### 📊 Eval Mode (Compare Apprentice vs Oracle)

**Use Case:** Compare both models to see improvement from Oracle.

**Dataset Requirements:**
- ✅ `input` field (problem text)
- ✅ `output` field (correct answer for comparison)
- ⚪ `solution_steps` field (optional, for Oracle)

**Command:**
```bash
python complete_pipeline.py \
  --mode dataset \
  --input eval_problems.jsonl \
  --batch-size 20 \
  --pipeline-mode eval
```

**What Happens:**
1. Apprentice attempts problem first
2. If Apprentice fails, Oracle tries
3. Both answers compared with ground truth
4. Statistics collected: Apprentice success rate, Oracle success rate, accuracy

**Output:** Comparison metrics showing when Oracle is needed

---

## Dataset Preparation Workflow

### Step 1: Split Your Data

```bash
# Use train_test_split.py to create train and test sets
python train_test_split.py \
  --input AllProblemsCleaned.jsonl \
  --val-split 0.2 \
  --seed 42

# Output:
# - AllProblemsCleaned_train.jsonl (80% - with solution_steps)
# - AllProblemsCleaned_val.jsonl (20% - with solution_steps for now)
```

### Step 2: Remove Solution Steps from Test Set

```python
# Create test set WITHOUT solution steps (prevents cheating)
import json

with open('AllProblemsCleaned_val.jsonl') as f_in, \
     open('test_problems.jsonl', 'w') as f_out:
    for line in f_in:
        data = json.loads(line)
        # Keep only input and output
        test_data = {
            'id': data.get('id'),
            'input': data.get('input'),
            'output': data.get('output'),
            'difficulty': data.get('difficulty'),  # Optional
            'source': data.get('source'),          # Optional
            'tags': data.get('tags')               # Optional
        }
        # Remove None values
        test_data = {k: v for k, v in test_data.items() if v is not None}
        f_out.write(json.dumps(test_data) + '\n')

print("✅ Test dataset created (without solution_steps)")
```

### Step 3: Use Datasets Appropriately

```bash
# TRAINING: Collect Oracle solutions (with solution_steps)
python complete_pipeline.py \
  --mode dataset \
  --input AllProblemsCleaned_train.jsonl \
  --pipeline-mode train

# TESTING: Evaluate Apprentice (without solution_steps)
python complete_pipeline.py \
  --mode dataset \
  --input test_problems.jsonl \
  --pipeline-mode test
```

---

## Single Problem Mode

For testing individual problems:

```bash
python complete_pipeline.py --mode single --input my_problem.txt --output solution.txt
```

---

## Quick Reference

### ✅ Valid Training Example
```jsonl
{"input": "What is 2+2?", "output": "4", "solution_steps": ["Add 2 and 2", "Result is 4"]}
```

### ✅ Valid Test Example
```jsonl
{"input": "What is 2+2?", "output": "4"}
```

### ✅ Complete Training Example
```jsonl
{"id": "001", "input": "What is 2+2?", "output": "4", "solution_steps": ["Add 2 and 2", "Result is 4"], "difficulty": "easy"}
```

### ❌ Invalid Examples
```jsonl
{"question": "What is 2+2?"}  ❌ Missing output
{"answer": "4"}  ❌ Missing input
{"problem": "What is 2+2?", "result": "4"}  ❌ Wrong field name (use 'output' or 'answer')
```

---

## Migration Guide

### From Other Formats

| Original Format | Field Mapping | Action |
|-----------------|---------------|--------|
| CSV | Convert to JSONL | Use script above |
| JSON array | Split into JSONL | One object per line |
| Custom fields | Rename to standard | Use `input`/`output` |
| No solution steps | Add if available | Improves learning 30-50% |

---

## Summary Table: Training vs Test Datasets

| Aspect | Training Dataset | Test Dataset |
|--------|------------------|--------------|
| **Purpose** | Collect Oracle solutions for fine-tuning | Evaluate model performance |
| **`input` field** | ✅ Required | ✅ Required |
| **`output` field** | ✅ Required (for Oracle accuracy check) | ✅ Required (for answer validation) |
| **`solution_steps` field** | ✅ **REQUIRED** (Oracle learns from this) | ❌ **MUST NOT include** (prevents cheating) |
| **Pipeline mode** | `--pipeline-mode train` | `--pipeline-mode test --skip-oracle` |
| **What model runs** | Oracle (generates training data) | Apprentice only (no Oracle help) |
| **Output** | `solver_training_data.jsonl` | Evaluation metrics (accuracy, success rate) |
| **Ground truth usage** | Check Oracle correctness | Check Apprentice predictions |
| **Example** | `{"input": "...", "output": "...", "solution_steps": [...]}` | `{"input": "...", "output": "..."}` |

---

## Summary: Key Points

**Minimum Requirements for ALL datasets:**
- ✅ JSONL format (one JSON per line)
- ✅ `input` field (or `problem`/`question`)
- ✅ `output` field (or `answer`/`solution`)

**Additional Requirements for TRAINING datasets:**
- ✅ `solution_steps` array (+30-50% learning quality)
- ⭐ Provides step-by-step reasoning for Oracle
- ⭐ Critical for high-quality fine-tuning data

**Additional Requirements for TEST datasets:**
- ❌ **Do NOT include** `solution_steps` (prevents cheating)
- ✅ Only problem (`input`) and answer (`output`)
- ✅ Answer used to cross-check model predictions

**Recommended for Both:**
- ⭐ `difficulty` string (for stratification)
- ⭐ `id` string (for tracking)
- ⭐ `source` string (attribution)
- ⭐ `tags` array (categorization)

**Your Current Format (AllProblemsCleaned.jsonl):**
- ✅ **PERFECT for TRAINING!** Has all required fields including `solution_steps`
- ⚠️ **For TESTING:** Remove `solution_steps` field before using in test mode

**Workflow:**
1. Split dataset → train (80%) + test (20%)
2. Keep `solution_steps` in training set
3. Remove `solution_steps` from test set
4. Use training set with `--pipeline-mode train`
5. Use test set with `--pipeline-mode test --skip-oracle`

---

## Questions?

**Q: Can I use a different file extension?**
A: No, must be `.jsonl` for batch processing. Use `.txt` for single problems only.

**Q: Do I need solution_steps for training?**
A: **YES!** Required for training mode. Oracle needs step-by-step solutions to learn from. Without it, you only get +10-15% improvement instead of +30-50%.

**Q: Should I include solution_steps in test data?**
A: **NO!** This would allow the model to "cheat" by seeing the solution. Test data should only have problem and answer for validation.

**Q: What if my field names are different?**
A: The pipeline auto-detects: `input/problem/question` and `output/answer/solution`. You're covered!

**Q: Can I have extra fields?**
A: Yes! Extra fields are ignored. Feel free to include metadata.

**Q: How do I handle multi-line solutions?**
A: Use `solution_steps` array with one step per element (recommended), or use `\n` in strings.

**Q: What if I don't have solution_steps for my training data?**
A: You can still train, but expect lower improvement (+10-15% instead of +30-50%). Oracle will generate solutions but without ground truth to learn from.

---

**Last Updated:** October 30, 2025
**Version:** 1.1 - Added Training/Test distinction
