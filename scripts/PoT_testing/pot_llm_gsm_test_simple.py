#!/usr/bin/env python3
"""
Ultra-Simple Program of Thought - Back to Basics
No special tokens, no fancy prompting, just: "write code to solve this"
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
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from tqdm import tqdm

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

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


def extract_python_code(response: str) -> Optional[str]:
    """Extract Python code from LLM response - flexible extraction"""
    # Try standard markdown code block
    pattern = r'```(?:python)?\s*\n(.*?)\n```'
    matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
    
    if matches:
        return matches[0].strip()
    
    # If no code block markers, check if entire response looks like code
    lines = response.strip().split('\n')
    # Check if response has Python-like syntax
    has_python = any(keyword in response.lower() for keyword in ['print', '=', 'def ', 'import', '#'])
    has_natural_language = any(word in response.lower() for word in ['the answer is', 'therefore', 'we need to', 'first'])
    
    # If it looks like code and doesn't look like prose, use it
    if has_python and not has_natural_language and len(lines) > 1:
        return response.strip()
    
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
                # Remove common prefixes
                line = re.sub(r'^(answer|result|output|final answer|the answer is)[\s:=]+', '', line, flags=re.IGNORECASE)
                # Extract number
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


def test_single_problem(problem_data: Dict[str, Any], llm: ChatOpenAI) -> ProblemResult:
    """Test a single problem with ultra-simple prompting"""
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
    
    # 8-SHOT POT PROMPT - Following GSM8K standard with Python code examples
    prompt = f"""Solve math problems by writing Python code. Print only the final numerical answer.

There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?

Step-by-step reasoning:
There are 15 trees originally.
Then there were 21 trees after planting.
So they planted 
21−15=6 trees.
The answer is 6.

If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?

Reasoning:
Originally 3 cars.
2 arrived.
3+2=5.
The answer is 5.

Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?

Reasoning:
Total chocolates initially 
32+42=74.
After eating 35, 
74−35=39.
The answer is 39.

Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many did he give to Denny?

Reasoning:
Started with 20.
Now has 12.
Gave away 
20−12=8.
The answer is 8.

Shawn has five toys. For Christmas, he got two toys from each parent. How many toys does he have now?

Reasoning:
Shawn started with 5 toys.
Got 2 from mom and 2 from dad, total 

2+2=4.
5+4=9.
The answer is 9.

Nine computers in the server room. Five more installed each day from Monday to Thursday. How many now?

Reasoning:
9 computers initially.
5 added each day for 4 days, 
5×4=20.
Total now 
9+20=29.
The answer is 29.

Michael had 58 golf balls; lost 23 on Tuesday and 2 more on Wednesday. How many left?

Reasoning:
Started 58.
Lost 23, left 
58−23=35.
Lost 2 more, left 
35−2=33.
The answer is 33.

Olivia has $23. She bought 5 bagels for $3 each. How much money is left?

Reasoning:
Bagels cost 
5×3=15
Money left 
23−15=8.
The answer is 8.

Problem: {problem_text}
"""

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
            error_message=str(e)
        )


def run_tests(input_file: Path, max_workers: int = 10, max_problems: Optional[int] = None) -> List[ProblemResult]:
    """Run tests on multiple problems with parallel processing"""
    import time
    
    print(f"Loading problems from {input_file}...")
    
    problems = []
    with open(input_file, 'r') as f:
        for i, line in enumerate(f):
            if max_problems and i >= max_problems:
                break
            if line.strip():
                problems.append(json.loads(line))
    
    print(f"Testing {len(problems)} problems with {max_workers} workers...")
    
    llm = ChatOpenAI(
        model="meta-llama/llama-3.2-3b-instruct",
        base_url="https://openrouter.ai/api/v1",
        temperature=0.0,
        request_timeout=60,  # Add request timeout
    )
    
    results = []
    completed_count = 0
    start_time = time.time()
    last_progress_time = start_time
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_problem = {
            executor.submit(test_single_problem, problem, llm): problem 
            for problem in problems
        }
        
        pending_futures = set(future_to_problem.keys())
        
        with tqdm(total=len(problems), desc="Processing") as pbar:
            while pending_futures and completed_count < len(problems):
                current_time = time.time()
                
                # Check if we're stuck (no progress for 180 seconds)
                if current_time - last_progress_time > 180:
                    print(f"\n⚠️  No progress for 180s. Cancelling {len(pending_futures)} remaining tasks...")
                    for future in pending_futures:
                        future.cancel()
                    break
                
                # Try to get results with a short timeout
                done_futures = []
                for future in list(pending_futures):
                    try:
                        if future.done():
                            done_futures.append(future)
                    except Exception:
                        done_futures.append(future)
                
                # Process completed futures
                for future in done_futures:
                    pending_futures.remove(future)
                    problem = future_to_problem[future]
                    
                    try:
                        result = future.result(timeout=1)  # Short timeout since it's already done
                        results.append(result)
                        completed_count += 1
                        last_progress_time = current_time
                        pbar.update(1)
                    except TimeoutError:
                        print(f"\n⚠️  Problem {problem.get('id', 'unknown')} timed out, skipping...")
                        results.append(ProblemResult(
                            problem_id=problem.get('id', 'unknown'),
                            question=problem.get('input', ''),
                            predicted_answer=None,
                            ground_truth=extract_ground_truth(problem),
                            correct=False,
                            generated_code="",
                            execution_output="",
                            success=False,
                            error_message="Problem timed out"
                        ))
                        completed_count += 1
                        pbar.update(1)
                    except Exception as e:
                        print(f"\n⚠️  Problem {problem.get('id', 'unknown')} error: {e}")
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
                        completed_count += 1
                        pbar.update(1)
                
                # Small sleep to avoid busy waiting
                if not done_futures:
                    time.sleep(0.5)
        
        # Handle any remaining futures that weren't completed
        for future in pending_futures:
            problem = future_to_problem[future]
            print(f"\n⚠️  Problem {problem.get('id', 'unknown')} was cancelled")
            results.append(ProblemResult(
                problem_id=problem.get('id', 'unknown'),
                question=problem.get('input', ''),
                predicted_answer=None,
                ground_truth=extract_ground_truth(problem),
                correct=False,
                generated_code="",
                execution_output="",
                success=False,
                error_message="Cancelled due to timeout"
            ))
    
    print(f"\n✓ Completed {len(results)}/{len(problems)} problems")
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
    print("ULTRA-SIMPLE POT TEST RESULTS")
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
                print(f"   Generated: {error.generated_code[:150]}...")
    
    # Sample incorrect answers
    incorrect = [r for r in results if r.success and not r.correct]
    if incorrect:
        print(f"\nSample incorrect ({min(3, len(incorrect))} of {len(incorrect)}):")
        for i, result in enumerate(incorrect[:3], 1):
            print(f"\n{i}. Problem ID: {result.problem_id}")
            print(f"   Question: {result.question[:80]}...")
            print(f"   Predicted: {result.predicted_answer}, Ground Truth: {result.ground_truth}")
    
    return {
        'total': total,
        'correct': correct,
        'accuracy': accuracy,
        'successful': successful,
        'success_rate': success_rate,
        'code_generated': code_generated
    }


def main():
    parser = argparse.ArgumentParser(description='Ultra-Simple PoT Testing')
    parser.add_argument('--input', type=str, required=True, help='Input JSONL file')
    parser.add_argument('--output', type=str, default='pot_llm_results_simple.json', 
                       help='Output JSON file for results')
    parser.add_argument('--workers', type=int, default=10, help='Number of parallel workers')
    parser.add_argument('--max-problems', type=int, help='Maximum number of problems to test')
    parser.add_argument('--single', type=str, help='Test a single problem (provide problem text)')
    
    args = parser.parse_args()
    
    llm = ChatOpenAI(
        model="meta-llama/llama-3.2-3b-instruct",
        base_url="https://openrouter.ai/api/v1",
        temperature=0.0,
    )
    
    if args.single:
        print("Testing single problem with ultra-simple PoT...")
        problem_data = {'id': 'single_test', 'input': args.single, 'output': None}
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
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file {input_path} not found")
        sys.exit(1)
    
    results = run_tests(input_path, max_workers=args.workers, max_problems=args.max_problems)
    
    summary = analyze_results(results)
    
    output_data = {
        'summary': summary,
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
                'execution_output': r.execution_output
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
