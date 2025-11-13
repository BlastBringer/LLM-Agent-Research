#!/usr/bin/env python3
"""
Program of Thought with Advanced Prompting Techniques for Llama 3.2 3B
Implements: Zero-shot CoT + Self-consistency + Verification
"""

import json
import argparse
import io
import re
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from collections import Counter

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class ProblemResult:
    problem_id: str
    question: str
    predicted_answer: Optional[float]
    ground_truth: Optional[float]
    correct: bool
    generated_code: str
    execution_output: str
    success: bool
    error_message: Optional[str] = None
    all_answers: Optional[List[float]] = None  # For self-consistency


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
        
        return None, output, "No numerical output found"
        
    except Exception as e:
        return None, "", f"Execution failed: {str(e)}"


def extract_ground_truth(problem_data: Dict[str, Any]) -> Optional[float]:
    """Extract ground truth answer from problem data"""
    for field in ['output', 'answer', 'ground_truth']:
        if field in problem_data:
            try:
                value = problem_data[field]
                if isinstance(value, str):
                    value = value.strip()
                    match = re.search(r'([-+]?[\d,]+\.?\d*)', value)
                    if match:
                        return float(match.group(1).replace(',', ''))
                return float(value)
            except (ValueError, TypeError):
                continue
    return None


def get_majority_answer(answers: List[Optional[float]]) -> Optional[float]:
    """Get majority answer from multiple attempts (self-consistency)"""
    # Filter out None values
    valid_answers = [a for a in answers if a is not None]
    
    if not valid_answers:
        return None
    
    # Round to 2 decimal places for comparison
    rounded = [round(a, 2) for a in valid_answers]
    
    # Get most common answer
    counter = Counter(rounded)
    most_common = counter.most_common(1)[0][0]
    
    return most_common


def test_single_problem_with_consistency(
    problem_data: Dict[str, Any], 
    llm: ChatOpenAI,
    num_samples: int = 3
) -> ProblemResult:
    """
    Test a single problem with self-consistency (multiple sampling)
    Uses advanced prompting: Zero-shot CoT + Progressive reasoning
    """
    problem_id = problem_data.get('id', 'unknown')
    problem_text = problem_data.get('input', problem_data.get('problem', problem_data.get('question', '')))
    
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
    
    # Progressive reasoning prompt optimized for Llama 3.2 3B
    # Breaks down the problem-solving process explicitly
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert math problem solver. Solve problems by writing clear Python code that shows your reasoning.

<|eot_id|><|start_header_id|>user<|end_header_id|>

Problem: {problem_text}

Let's solve this step by step:

Step 1 - UNDERSTAND: What information do we have?
Step 2 - PLAN: What do we need to calculate?
Step 3 - SOLVE: Write Python code to solve it

Write your solution as clean Python code:
- Extract the key numbers from the problem
- Do the calculations step by step
- Print ONLY the final numerical answer

```python
<|eot_id|><|start_header_id|>assistant<|end_header_id|>

```python"""

    all_answers = []
    all_codes = []
    
    # Sample multiple times for self-consistency
    for _ in range(num_samples):
        try:
            response = llm.invoke(prompt)
            response_text = response.content
            
            # Handle case where model might not include closing ```
            if '```' not in response_text:
                response_text = response_text + '\n```'
            
            code = extract_python_code(response_text)
            
            if code:
                all_codes.append(code)
                predicted_answer, _, error_msg = execute_python_code(code)
                
                if not error_msg and predicted_answer is not None:
                    all_answers.append(predicted_answer)
        except Exception:
            continue
    
    # Use self-consistency: pick majority answer
    if all_answers:
        predicted_answer = get_majority_answer(all_answers)
        best_code = all_codes[0] if all_codes else ""
        
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
            generated_code=best_code,
            execution_output=f"Sampled {len(all_answers)} times, answers: {all_answers}",
            success=True,
            all_answers=all_answers
        )
    else:
        return ProblemResult(
            problem_id=problem_id,
            question=problem_text,
            predicted_answer=None,
            ground_truth=ground_truth,
            correct=False,
            generated_code=all_codes[0] if all_codes else "",
            execution_output="",
            success=False,
            error_message="Failed to get valid answer from any sample"
        )


def test_single_problem(problem_data: Dict[str, Any], llm: ChatOpenAI) -> ProblemResult:
    """
    Test a single problem with optimized zero-shot CoT prompting
    No self-consistency (faster, single pass)
    """
    problem_id = problem_data.get('id', 'unknown')
    problem_text = problem_data.get('input', problem_data.get('problem', problem_data.get('question', '')))
    
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
    
    # Simplified but optimized prompt for Llama 3.2
    # Key insight: Give the model structure but let it fill in the reasoning
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a math problem solver. Write Python code to solve problems step by step.

<|eot_id|><|start_header_id|>user<|end_header_id|>

Problem: {problem_text}

Think step by step and write Python code to solve this. Your code should:
1. Extract the numbers from the problem
2. Calculate the answer step by step  
3. Print only the final number

```python
<|eot_id|><|start_header_id|>assistant<|end_header_id|>

```python"""

    try:
        response = llm.invoke(prompt)
        response_text = response.content
        
        # Handle case where model might not include closing ```
        if '```' not in response_text:
            response_text = response_text + '\n```'
        
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
            error_message=str(e)
        )


def run_tests(
    input_file: Path, 
    max_workers: int = 10, 
    max_problems: Optional[int] = None,
    use_self_consistency: bool = False
) -> List[ProblemResult]:
    """Run tests on multiple problems with parallel processing"""
    print(f"Loading problems from {input_file}...")
    
    problems = []
    with open(input_file, 'r') as f:
        for i, line in enumerate(f):
            if max_problems and i >= max_problems:
                break
            if line.strip():
                problems.append(json.loads(line))
    
    mode = "self-consistency" if use_self_consistency else "single-pass"
    print(f"Testing {len(problems)} problems with {max_workers} workers ({mode} mode)...")
    
    # Create LLM instance
    llm = ChatOpenAI(
        model="meta-llama/llama-3.2-3b-instruct",
        base_url="https://openrouter.ai/api/v1",
        temperature=0.0,
    )
    
    results = []
    test_func = test_single_problem_with_consistency if use_self_consistency else test_single_problem
    
    # Process problems in parallel with progress bar
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_problem = {
            executor.submit(test_func, problem, llm): problem 
            for problem in problems
        }
        
        # Collect results with progress bar
        for future in tqdm(as_completed(future_to_problem), total=len(problems), desc="Processing"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                problem = future_to_problem[future]
                print(f"\nError processing problem {problem.get('id', 'unknown')}: {e}")
                results.append(ProblemResult(
                    problem_id=problem.get('id', 'unknown'),
                    question=problem.get('input', ''),
                    predicted_answer=None,
                    ground_truth=extract_ground_truth(problem),
                    correct=False,
                    generated_code="",
                    execution_output="",
                    success=False,
                    error_message=str(e)
                ))
    
    return results


def analyze_results(results: List[ProblemResult]) -> Dict[str, Any]:
    """Analyze and print results"""
    total = len(results)
    correct = sum(1 for r in results if r.correct)
    successful = sum(1 for r in results if r.success)
    code_generated = sum(1 for r in results if r.generated_code)
    
    accuracy = (correct / total * 100) if total > 0 else 0
    success_rate = (successful / total * 100) if total > 0 else 0
    
    print("\n" + "="*60)
    print("ADVANCED POT TEST RESULTS")
    print("="*60)
    print(f"Total problems:        {total}")
    print(f"Correct answers:       {correct} ({accuracy:.1f}%)")
    print(f"Successful executions: {successful} ({success_rate:.1f}%)")
    print(f"Code generated:        {code_generated}")
    print("="*60)
    
    # Sample some errors
    errors = [r for r in results if not r.success]
    if errors:
        print(f"\nSample errors ({min(3, len(errors))} of {len(errors)}):")
        for i, error in enumerate(errors[:3], 1):
            print(f"\n{i}. Problem ID: {error.problem_id}")
            print(f"   Error: {error.error_message}")
            if error.generated_code:
                print(f"   Code: {error.generated_code[:100]}...")
    
    # Sample incorrect answers
    incorrect = [r for r in results if r.success and not r.correct]
    if incorrect:
        print(f"\nSample incorrect answers ({min(3, len(incorrect))} of {len(incorrect)}):")
        for i, result in enumerate(incorrect[:3], 1):
            print(f"\n{i}. Problem ID: {result.problem_id}")
            print(f"   Question: {result.question[:100]}...")
            print(f"   Predicted: {result.predicted_answer}")
            print(f"   Ground Truth: {result.ground_truth}")
    
    return {
        'total': total,
        'correct': correct,
        'accuracy': accuracy,
        'successful': successful,
        'success_rate': success_rate,
        'code_generated': code_generated
    }


def main():
    parser = argparse.ArgumentParser(description='Test LLM with Advanced PoT Techniques')
    parser.add_argument('--input', type=str, required=True, help='Input JSONL file')
    parser.add_argument('--output', type=str, default='pot_llm_results_advanced.json', 
                       help='Output JSON file for results')
    parser.add_argument('--workers', type=int, default=10, help='Number of parallel workers')
    parser.add_argument('--max-problems', type=int, help='Maximum number of problems to test')
    parser.add_argument('--single', type=str, help='Test a single problem (provide problem text)')
    parser.add_argument('--self-consistency', action='store_true', 
                       help='Use self-consistency (sample 3 times, take majority)')
    
    args = parser.parse_args()
    
    # Initialize LLM
    llm = ChatOpenAI(
        model="meta-llama/llama-3.2-3b-instruct",
        base_url="https://openrouter.ai/api/v1",
        temperature=0.0,
    )
    
    if args.single:
        print("Testing single problem with Advanced PoT...")
        problem_data = {'id': 'single_test', 'input': args.single, 'output': None}
        
        if args.self_consistency:
            result = test_single_problem_with_consistency(problem_data, llm)
        else:
            result = test_single_problem(problem_data, llm)
        
        print("\nGenerated Code:")
        print(result.generated_code)
        print("\nExecution Output:")
        print(result.execution_output)
        print(f"\nPredicted Answer: {result.predicted_answer}")
        print(f"Success: {result.success}")
        if result.error_message:
            print(f"Error: {result.error_message}")
        return
    
    # Run batch tests
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file {input_path} not found")
        sys.exit(1)
    
    results = run_tests(
        input_path, 
        max_workers=args.workers, 
        max_problems=args.max_problems,
        use_self_consistency=args.self_consistency
    )
    
    # Analyze and print results
    summary = analyze_results(results)
    
    # Save results to file
    output_data = {
        'summary': summary,
        'self_consistency': args.self_consistency,
        'results': [
            {
                'problem_id': r.problem_id,
                'question': r.question,
                'predicted_answer': r.predicted_answer,
                'ground_truth': r.ground_truth,
                'correct': r.correct,
                'success': r.success,
                'error_message': r.error_message,
                'generated_code': r.generated_code,
                'execution_output': r.execution_output,
                'all_answers': r.all_answers
            }
            for r in results
        ]
    }
    
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
