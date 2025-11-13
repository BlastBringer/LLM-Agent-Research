#!/usr/bin/env python3
"""
🧮 PROGRAM OF THOUGHT (PoT) LLM BENCHMARK - GSM-Symbolic
=========================================================

Tests bare LLM using Program of Thought approach where the model
generates Python code to solve math problems instead of natural language reasoning.

The model writes executable Python programs with:
- Step-by-step reasoning as code comments
- Actual calculations in Python
- Final answer printed at the end

This approach separates reasoning (comments) from computation (code execution).

Usage:
    # Test on full dataset
    python3 pot_llm_gsm_test.py --input gsm_symbolic_batch1.jsonl --workers 10
    
    # Test single problem
    python3 pot_llm_gsm_test.py --single "Problem text here"
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
    """
    Extract ground truth answer from problem data.
    
    Args:
        problem_data: Problem dictionary with answer field
        
    Returns:
        Ground truth as float, or None if not found
    """
    # Try 'output' field first (GSM-Symbolic format)
    if 'output' in problem_data:
        output = problem_data['output']
        if isinstance(output, (int, float)):
            return float(output)
    
    # Try 'answer' field
    if 'answer' in problem_data:
        answer = problem_data['answer']
        
        # Handle direct numeric values
        if isinstance(answer, (int, float)):
            return float(answer)
        
        # Handle string format "#### number"
        if isinstance(answer, str):
            # Try to extract number after ####
            match = re.search(r'####\s*([-+]?[\d,]+\.?\d*)', answer)
            if match:
                number_str = match.group(1).replace(',', '')
                try:
                    return float(number_str)
                except ValueError:
                    pass
            
            # Try to extract just a number
            match = re.search(r'([-+]?[\d,]+\.?\d*)', answer)
            if match:
                number_str = match.group(1).replace(',', '')
                try:
                    return float(number_str)
                except ValueError:
                    pass
    
    return None


def extract_python_code(response: str) -> Optional[str]:
    """
    Extract Python code from LLM response.
    
    Args:
        response: LLM response text
        
    Returns:
        Extracted Python code, or None if not found
    """
    # Look for code between ```python and ```
    pattern = r'```python\s*\n(.*?)\n```'
    matches = re.findall(pattern, response, re.DOTALL)
    
    if matches:
        return matches[0].strip()
    
    # Try without language specifier
    pattern = r'```\s*\n(.*?)\n```'
    matches = re.findall(pattern, response, re.DOTALL)
    
    if matches:
        # Check if it looks like Python code
        code = matches[0].strip()
        if 'def ' in code or 'print' in code or '=' in code:
            return code
    
    return None


def execute_python_code(code: str, timeout: int = 5) -> tuple[Optional[float], str, Optional[str]]:
    """
    Safely execute Python code and extract the final printed number.
    
    Args:
        code: Python code to execute
        timeout: Timeout in seconds
        
    Returns:
        Tuple of (extracted_answer, output, error_message)
    """
    # Create a restricted namespace for execution
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
        }
    }
    
    # Capture output
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
                # Try to extract a number
                # Remove common text prefixes
                line = re.sub(r'^(answer|result|output|final answer)[\s:=]+', '', line, flags=re.IGNORECASE)
                # Extract number
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
    """
    Test the model on a single problem using Program of Thought.
    
    Args:
        problem_data: Problem dictionary
        problem_id: Problem index
        llm: Language model instance
        
    Returns:
        ProblemResult with evaluation
    """
    # Extract problem text
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
    
    # Extract ground truth
    ground_truth = extract_ground_truth(problem_data)
    
    # Create Program of Thought prompt
    prompt = f"""Solve this math problem step-by-step by writing a Python program.
The program should print the final numerical answer.

Problem:
{problem_text}

Write a complete Python program that:
1. Uses comments to explain your step-by-step reasoning
2. Performs calculations using Python code
3. Prints only the final numerical answer (just the number)

Start your solution with ```python and end with ```.

Example format:
```python
def solve():
    # Step 1: Calculate the first part
    first_part = 10 * 5
    
    # Step 2: Calculate the second part
    second_part = 20 + 15
    
    # Step 3: Combine the results
    result = first_part + second_part
    
    # Print the final answer
    print(result)

solve()
```

Now solve the problem above:"""

    try:
        # Get LLM response
        response = llm.invoke(prompt)
        response_text = response.content
        
        # Extract Python code
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
        
        # Execute the code
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
        
        # Check correctness
        correct = False
        if predicted_answer is not None and ground_truth is not None:
            # Use 1% relative tolerance
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
    """
    Test LLM on dataset using Program of Thought approach.
    
    Args:
        input_file: Input JSONL file
        output_file: Output JSON file
        workers: Number of parallel workers
        
    Returns:
        Summary statistics
    """
    # Initialize model
    llm = ChatOpenAI(
        model="meta-llama/llama-3.2-3b-instruct",
        temperature=0.0,
        base_url="https://openrouter.ai/api/v1"
    )
    
    # Load problems
    problems = []
    with open(input_file, 'r') as f:
        for line in f:
            problems.append(json.loads(line))
    
    print("=" * 70)
    print("🧮 PROGRAM OF THOUGHT (PoT) LLM BENCHMARK - GSM-Symbolic")
    print("=" * 70)
    print(f"Model: meta-llama/llama-3.2-3b-instruct")
    print(f"Dataset: {input_file.name}")
    print(f"Workers: {workers}")
    print()
    print(f"📊 Testing on {len(problems)} problems...")
    print()
    
    results = []
    correct_count = 0
    failed_count = 0
    
    # Process in parallel
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
                print(f"❌ Problem {result.problem_id + 1}: {result.error_message}")
            
            # Progress update every 10 problems
            if len(results) % 10 == 0:
                current_accuracy = (correct_count / len(results)) * 100
                print(f"📊 Progress: {len(results)}/{len(problems)} ({len(results)/len(problems)*100:.1f}%) - Accuracy so far: {current_accuracy:.1f}%")
                print()
    
    # Sort results by problem_id
    results.sort(key=lambda x: x.problem_id)
    
    # Calculate statistics
    total = len(results)
    successful = sum(1 for r in results if r.success)
    wrong = sum(1 for r in results if r.success and not r.correct)
    accuracy = (correct_count / total) * 100 if total > 0 else 0
    
    # Save detailed results
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
    
    # Print summary
    print()
    print("=" * 70)
    print("📊 RESULTS")
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
    print("🧮 PROGRAM OF THOUGHT - SINGLE PROBLEM TEST")
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
        description="Test LLM using Program of Thought approach on GSM-Symbolic"
    )
    parser.add_argument(
        '--input',
        type=Path,
        default=Path('datasets/gsm_symbolic_batch1.jsonl'),
        help='Input JSONL file with problems'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Output JSON file for results (default: pot_llm_results.json)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=5,
        help='Number of parallel workers (default: 5)'
    )
    parser.add_argument(
        '--single',
        type=str,
        help='Test on a single problem (provide problem text)'
    )
    
    args = parser.parse_args()
    
    # Handle single problem mode
    if args.single:
        test_single(args.single)
        return
    
    # Set default output file if not provided
    if args.output is None:
        args.output = Path('pot_llm_results.json')
    
    # Run batch test
    test_pot_llm(args.input, args.output, args.workers)


if __name__ == '__main__':
    main()
