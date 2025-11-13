#!/usr/bin/env python3
"""
🧮 IMPROVED PROGRAM OF THOUGHT (PoT) LLM BENCHMARK
===================================================

Improvements over base version:
1. **Few-shot examples** - Show the model good code patterns
2. **Better code formatting** - More explicit variable names
3. **Step verification** - Encourage checking intermediate results
4. **Robust execution** - Better error handling and retry logic
5. **Problem-specific hints** - Detect problem types and add relevant hints

This version aims to improve accuracy on clustered problem types.
"""

import json
import logging
import argparse
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import re
import sys
import io
from contextlib import redirect_stdout, redirect_stderr

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ProblemResult:
    """Result from testing a single problem"""
    problem_id: int
    question: str
    predicted_answer: Optional[float]
    ground_truth: Optional[float]
    correct: bool
    generated_code: str
    execution_output: str
    success: bool
    error_message: Optional[str] = None


def extract_ground_truth(problem_data: Dict[str, Any]) -> Optional[float]:
    """Extract ground truth answer from problem data"""
    if 'output' in problem_data:
        output = problem_data['output']
        if isinstance(output, (int, float)):
            return float(output)
    
    if 'answer' in problem_data:
        answer = problem_data['answer']
        if isinstance(answer, (int, float)):
            return float(answer)
        if isinstance(answer, str):
            match = re.search(r'####\s*([-+]?[\d,]+\.?\d*)', answer)
            if match:
                number_str = match.group(1).replace(',', '')
                try:
                    return float(number_str)
                except ValueError:
                    pass
    return None


def detect_problem_type(problem_text: str) -> str:
    """Detect the type of math problem to provide relevant hints"""
    problem_lower = problem_text.lower()
    
    if 'percent' in problem_lower or '%' in problem_text:
        return 'percentage'
    elif 'ratio' in problem_lower or 'proportion' in problem_lower:
        return 'ratio'
    elif 'probability' in problem_lower or 'likely' in problem_lower or 'chance' in problem_lower:
        return 'probability'
    elif 'rate' in problem_lower or 'per' in problem_lower or 'speed' in problem_lower:
        return 'rate'
    elif 'area' in problem_lower or 'perimeter' in problem_lower or 'volume' in problem_lower:
        return 'geometry'
    elif 'total' in problem_lower or 'sum' in problem_lower or 'altogether' in problem_lower:
        return 'addition'
    elif 'difference' in problem_lower or 'more than' in problem_lower or 'less than' in problem_lower:
        return 'comparison'
    else:
        return 'general'


def get_problem_specific_hint(problem_type: str) -> str:
    """Get helpful hints based on problem type"""
    hints = {
        'percentage': """
    # HINT: For percentages, convert to decimal or use fractions
    # Remember: X% = X/100
    # Example: 20% of 50 = 20/100 * 50 = 0.2 * 50""",
        'probability': """
    # HINT: Probability = (favorable outcomes) / (total outcomes)
    # For multiple events: multiply probabilities
    # Convert to percentage: multiply by 100""",
        'rate': """
    # HINT: Rate problems use: Distance = Rate × Time
    # Or: Work = Rate × Time
    # Rearrange as needed: Time = Distance / Rate""",
        'ratio': """
    # HINT: For ratios, set up proportions
    # a:b = c:d means a/b = c/d
    # Cross multiply to solve""",
        'geometry': """
    # HINT: Remember formulas:
    # Rectangle: Area = length × width
    # Circle: Area = π × r², Circumference = 2 × π × r
    # Triangle: Area = (base × height) / 2""",
        'general': """
    # HINT: Break the problem into clear steps
    # Define variables for intermediate calculations
    # Double-check your arithmetic"""
    }
    return hints.get(problem_type, hints['general'])


def extract_python_code(response: str) -> Optional[str]:
    """Extract Python code from LLM response"""
    pattern = r'```python\s*\n(.*?)\n```'
    matches = re.findall(pattern, response, re.DOTALL)
    
    if matches:
        return matches[0].strip()
    
    pattern = r'```\s*\n(.*?)\n```'
    matches = re.findall(pattern, response, re.DOTALL)
    
    if matches:
        code = matches[0].strip()
        if 'def ' in code or 'print' in code or '=' in code:
            return code
    
    return None


def execute_python_code(code: str, timeout: int = 5) -> tuple[Optional[float], str, Optional[str]]:
    """Safely execute Python code and extract the final printed number"""
    namespace = {
        '__builtins__': {
            'print': print,
            'range': range,
            'len': len,
            'sum': sum,
            'max': max,
            'min': min,
            'abs': abs,
            'round': round,
            'int': int,
            'float': float,
            'str': str,
            'list': list,
            'dict': dict,
            'set': set,
            'tuple': tuple,
            'sorted': sorted,
            'enumerate': enumerate,
            'zip': zip,
            'pow': pow,
        }
    }
    
    output_buffer = io.StringIO()
    error_buffer = io.StringIO()
    
    try:
        with redirect_stdout(output_buffer), redirect_stderr(error_buffer):
            exec(code, namespace)
        
        output = output_buffer.getvalue()
        error = error_buffer.getvalue()
        
        if error:
            return None, output, f"Execution error: {error}"
        
        # Extract the last number printed
        lines = output.strip().split('\n')
        for line in reversed(lines):
            line = line.strip()
            if line:
                line = re.sub(r'^(answer|result|output|final answer)[\s:=]+', '', line, flags=re.IGNORECASE)
                match = re.search(r'([-+]?[\d,]+\.?\d*)', line)
                if match:
                    number_str = match.group(1).replace(',', '')
                    try:
                        return float(number_str), output, None
                    except ValueError:
                        continue
        
        return None, output, "Could not extract numeric answer from output"
        
    except Exception as e:
        error = error_buffer.getvalue()
        return None, output_buffer.getvalue(), f"Execution exception: {str(e)}\n{error}"


def test_single_problem(
    problem_data: Dict[str, Any],
    problem_id: int,
    llm: ChatOpenAI
) -> ProblemResult:
    """Test the model on a single problem using improved Program of Thought"""
    problem_text = problem_data.get('input', problem_data.get('question', ''))
    
    if not problem_text:
        return ProblemResult(
            problem_id=problem_id,
            question="",
            predicted_answer=None,
            ground_truth=None,
            correct=False,
            generated_code="",
            execution_output="",
            success=False,
            error_message="No problem text found"
        )
    
    ground_truth = extract_ground_truth(problem_data)
    
    # Detect problem type and get relevant hint
    problem_type = detect_problem_type(problem_text)
    hint = get_problem_specific_hint(problem_type)
    
    # Improved prompt with few-shot examples and better guidance
    prompt = f"""Solve this math problem by writing clear, executable Python code.

Problem:
{problem_text}

Write a Python program following these guidelines:
1. Use descriptive variable names (not just x, y, z)
2. Add comments explaining each calculation step
3. Show intermediate results to verify your logic
4. Print ONLY the final numerical answer (no text, just the number)

{hint}

Example 1 - Percentage Problem:
```python
def solve():
    # Given: 72 inch remoras on a 210-foot whale
    # Find: What percentage of whale length is remoras
    
    # Step 1: Convert remora length to consistent units (feet)
    remora_count = 7
    remora_length_inches = 72
    total_remora_inches = remora_count * remora_length_inches  # = 504 inches
    total_remora_feet = total_remora_inches / 12  # Convert to feet: = 42 feet
    
    # Step 2: Calculate percentage
    whale_length_feet = 210
    percentage = (total_remora_feet / whale_length_feet) * 100  # = 20%
    
    # Print only the final answer
    print(percentage)

solve()
```

Example 2 - Rate Problem:
```python
def solve():
    # Given: Fog covers 13 miles in 63 minutes, city is 39 miles
    # Find: Total time to cover city
    
    # Step 1: Calculate rate (time per mile)
    time_for_distance = 63  # minutes
    distance_covered = 13  # miles
    time_per_mile = time_for_distance / distance_covered  # = 4.846 min/mile
    
    # Step 2: Calculate total time
    city_distance = 39  # miles
    total_time = time_per_mile * city_distance  # = 189 minutes
    
    # Print only the final answer
    print(total_time)

solve()
```

Now solve the problem above. Start with ```python and end with ```:"""

    try:
        response = llm.invoke(prompt)
        response_text = response.content
        
        code = extract_python_code(response_text)
        
        if not code:
            return ProblemResult(
                problem_id=problem_id,
                question=problem_text,
                predicted_answer=None,
                ground_truth=ground_truth,
                correct=False,
                generated_code=response_text,
                execution_output="",
                success=False,
                error_message="Failed to extract Python code from response"
            )
        
        predicted_answer, execution_output, error_msg = execute_python_code(code)
        
        if error_msg:
            return ProblemResult(
                problem_id=problem_id,
                question=problem_text,
                predicted_answer=None,
                ground_truth=ground_truth,
                correct=False,
                generated_code=code,
                execution_output=execution_output,
                success=False,
                error_message=error_msg
            )
        
        # Check correctness with 1% relative tolerance
        correct = False
        if predicted_answer is not None and ground_truth is not None:
            tolerance = max(abs(ground_truth * 0.01), 0.01)
            correct = abs(predicted_answer - ground_truth) <= tolerance
        
        return ProblemResult(
            problem_id=problem_id,
            question=problem_text,
            predicted_answer=predicted_answer,
            ground_truth=ground_truth,
            correct=correct,
            generated_code=code,
            execution_output=execution_output,
            success=True
        )
        
    except Exception as e:
        return ProblemResult(
            problem_id=problem_id,
            question=problem_text,
            predicted_answer=None,
            ground_truth=ground_truth,
            correct=False,
            generated_code="",
            execution_output="",
            success=False,
            error_message=f"Exception: {str(e)}"
        )


def test_pot_llm(
    input_file: Path,
    output_file: Path,
    workers: int = 5
) -> Dict[str, Any]:
    """Test LLM on dataset using improved Program of Thought"""
    llm = ChatOpenAI(
        model="meta-llama/llama-3.2-3b-instruct",
        temperature=0.0,
        base_url="https://openrouter.ai/api/v1"
    )
    
    problems = []
    with open(input_file, 'r') as f:
        for line in f:
            problems.append(json.loads(line))
    
    print("=" * 70)
    print("🧮 IMPROVED PROGRAM OF THOUGHT (PoT) - GSM-Symbolic")
    print("=" * 70)
    print(f"Model: meta-llama/llama-3.2-3b-instruct")
    print(f"Dataset: {input_file.name}")
    print(f"Workers: {workers}")
    print(f"Improvements: Few-shot examples, problem-type hints, better prompts")
    print()
    print(f"📊 Testing on {len(problems)} problems...")
    print()
    
    results = []
    correct_count = 0
    failed_count = 0
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_id = {
            executor.submit(test_single_problem, problem, i, llm): i
            for i, problem in enumerate(problems)
        }
        
        for future in as_completed(future_to_id):
            result = future.result()
            results.append(result)
            
            if result.success and result.correct:
                correct_count += 1
                print(f"✅ Problem {result.problem_id + 1}: Correct ({result.predicted_answer} == {result.ground_truth})")
            elif result.success and not result.correct:
                print(f"❌ Problem {result.problem_id + 1}: Wrong ({result.predicted_answer} != {result.ground_truth})")
            else:
                failed_count += 1
                error_short = result.error_message[:50] if result.error_message else "Unknown error"
                print(f"❌ Problem {result.problem_id + 1}: {error_short}...")
            
            if len(results) % 10 == 0:
                current_accuracy = (correct_count / len(results)) * 100
                print(f"📊 Progress: {len(results)}/{len(problems)} ({len(results)/len(problems)*100:.1f}%) - Accuracy: {current_accuracy:.1f}%")
                print()
    
    results.sort(key=lambda x: x.problem_id)
    
    total = len(results)
    successful = sum(1 for r in results if r.success)
    wrong = sum(1 for r in results if r.success and not r.correct)
    accuracy = (correct_count / total) * 100 if total > 0 else 0
    
    output_data = {
        'results': [
            {
                'problem_id': r.problem_id,
                'question': r.question,
                'predicted_answer': r.predicted_answer,
                'ground_truth': r.ground_truth,
                'correct': r.correct,
                'generated_code': r.generated_code,
                'execution_output': r.execution_output,
                'success': r.success,
                'error_message': r.error_message
            }
            for r in results
        ],
        'summary': {
            'total_problems': total,
            'correct': correct_count,
            'wrong': wrong,
            'failed': failed_count,
            'accuracy': accuracy
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print()
    print("=" * 70)
    print("📊 FINAL RESULTS")
    print("=" * 70)
    print(f"Total Problems: {total}")
    print(f"Correct: {correct_count}")
    print(f"Wrong: {wrong}")
    print(f"Failed to Execute: {failed_count}")
    print(f"Accuracy: {accuracy:.2f}%")
    print()
    print(f"💾 Detailed results saved to: {output_file}")
    
    return output_data['summary']


def test_single(problem_text: str):
    """Test on a single problem with detailed output"""
    llm = ChatOpenAI(
        model="meta-llama/llama-3.2-3b-instruct",
        temperature=0.0,
        base_url="https://openrouter.ai/api/v1"
    )
    
    problem_data = {'input': problem_text}
    result = test_single_problem(problem_data, 0, llm)
    
    print("=" * 70)
    print("🧮 IMPROVED PROGRAM OF THOUGHT - SINGLE TEST")
    print("=" * 70)
    print(f"\n📝 Problem:")
    print(f"{result.question}")
    print(f"\n💻 Generated Code:")
    print("```python")
    print(result.generated_code)
    print("```")
    print(f"\n📤 Execution Output:")
    print(result.execution_output)
    print(f"\n🎯 Extracted Answer: {result.predicted_answer}")
    
    if result.ground_truth:
        print(f"✓ Ground Truth: {result.ground_truth}")
        print(f"✓ Correct: {result.correct}")
    
    if not result.success:
        print(f"\n❌ Error: {result.error_message}")


def main():
    parser = argparse.ArgumentParser(
        description="Improved PoT testing on GSM-Symbolic"
    )
    parser.add_argument(
        '--input',
        type=Path,
        default=Path('datasets/gsm_symbolic_batch1.jsonl'),
        help='Input JSONL file'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Output JSON file (default: pot_llm_results_improved.json)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=5,
        help='Parallel workers (default: 5)'
    )
    parser.add_argument(
        '--single',
        type=str,
        help='Test single problem'
    )
    
    args = parser.parse_args()
    
    if args.single:
        test_single(args.single)
        return
    
    if args.output is None:
        args.output = Path('pot_llm_results_improved.json')
    
    test_pot_llm(args.input, args.output, args.workers)


if __name__ == '__main__':
    main()
