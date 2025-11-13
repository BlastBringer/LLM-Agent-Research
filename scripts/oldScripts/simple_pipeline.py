#!/usr/bin/env python3
"""
Simplified Pipeline: Apprentice (PoT) + Verifier + Oracle (Bare LLM)

Architecture:
- Apprentice: Program of Thought (generates Python code) - PoT Ultra-Simple (70.2%)
- Verifier: Checks if answer matches ground truth
- Oracle: Bare LLM (natural language reasoning) when Apprentice fails

Modes:
- Train: Collect Oracle solutions when Apprentice fails (for fine-tuning)
- Test: Only Apprentice with PoT (no Oracle)
"""

import json
import argparse
import io
import re
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


@dataclass
class PipelineResult:
    problem_id: str
    problem: str
    solution_steps: Optional[str]  # From dataset
    ground_truth: Optional[float]
    
    # Apprentice (PoT)
    apprentice_code: str
    apprentice_output: str
    apprentice_answer: Optional[float]
    apprentice_correct: bool
    apprentice_error: Optional[str]
    
    # Oracle (Bare LLM) - only in train mode
    oracle_reasoning: Optional[str] = None
    oracle_answer: Optional[float] = None
    oracle_correct: Optional[bool] = None
    oracle_used: bool = False
    
    # Final result
    final_answer: Optional[float] = None
    final_correct: bool = False
    source: str = ""  # "apprentice" or "oracle"


def extract_python_code(response: str) -> Optional[str]:
    """Extract Python code from LLM response - flexible extraction"""
    # Try standard markdown code block
    pattern = r'```(?:python)?\s*\n(.*?)\n```'
    matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
    
    if matches:
        return matches[0].strip()
    
    # If no code block markers, check if entire response looks like code
    lines = response.strip().split('\n')
    has_python = any(keyword in response.lower() for keyword in ['print', '=', 'def ', '#'])
    has_natural_language = any(word in response.lower() for word in ['the answer is', 'therefore', 'we need to', 'first,'])
    
    # If it looks like code and doesn't look like prose, use it
    if has_python and not has_natural_language and len(lines) > 1:
        return response.strip()
    
    return None


def execute_python_code(code: str) -> tuple[Optional[float], str, Optional[str]]:
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
                line = re.sub(r'^(answer|result|output|final answer|the answer is)[\s:=]+', '', line, flags=re.IGNORECASE)
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


def extract_answer_from_text(text: str) -> Optional[float]:
    """Extract numerical answer from natural language text"""
    # Look for common answer patterns
    patterns = [
        r'(?:answer|result|final answer|solution)[\s:=]+([0-9,.]+)',
        r'(?:is|equals|=)[\s]+([0-9,.]+)',
        r'([0-9,.]+)[\s]*(?:\.|$)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text.strip(), re.IGNORECASE)
        if match:
            try:
                return float(match.group(1).replace(',', ''))
            except ValueError:
                continue
    
    return None


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


def answers_match(predicted: Optional[float], ground_truth: Optional[float], tolerance_pct: float = 1.0) -> bool:
    """Check if predicted answer matches ground truth within tolerance"""
    if predicted is None or ground_truth is None:
        return False
    
    tolerance = max(abs(ground_truth * (tolerance_pct / 100)), 0.01)
    return abs(predicted - ground_truth) <= tolerance


class Apprentice:
    """Apprentice solver using Program of Thought (PoT)"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def solve(self, problem: str) -> tuple[str, str, Optional[float], Optional[str]]:
        """
        Solve using PoT (generate Python code and execute)
        
        Returns: (code, output, answer, error)
        """
        # Ultra-simple prompt (70.2% accuracy - our best PoT)
        prompt = f"""Write Python code to solve this math problem. Print only the final numerical answer.

Problem: {problem}

Python code:"""
        
        try:
            response = self.llm.invoke(prompt)
            response_text = response.content
            
            code = extract_python_code(response_text)
            
            if not code:
                return response_text, "", None, "Failed to extract Python code"
            
            answer, output, error = execute_python_code(code)
            
            return code, output, answer, error
            
        except Exception as e:
            return "", "", None, str(e)


class Oracle:
    """Oracle solver using Bare LLM (natural language reasoning)"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def solve(self, problem: str, with_code: bool = True) -> tuple[str, Optional[float], Optional[str], Optional[str]]:
        """
        Solve using natural language reasoning
        
        Args:
            problem: The math problem
            with_code: If True, also generate Python code equivalent
        
        Returns: (reasoning, answer, code, error)
        """
        if with_code:
            prompt = f"""Solve this math problem step by step, then provide the Python code equivalent.

Problem: {problem}

Provide:
1. Your step-by-step reasoning
2. The final numerical answer
3. Python code that solves this problem

Format:
REASONING:
[your step-by-step solution]

ANSWER: [numerical answer]

CODE:
```python
[python code]
```"""
        else:
            prompt = f"""Solve this math problem step by step.

Problem: {problem}

Provide your step-by-step reasoning and the final numerical answer."""
        
        try:
            response = self.llm.invoke(prompt)
            reasoning = response.content
            
            # Extract answer
            answer = extract_answer_from_text(reasoning)
            
            # Extract code if requested
            code = None
            if with_code:
                code = extract_python_code(reasoning)
            
            return reasoning, answer, code, None
            
        except Exception as e:
            return "", None, None, str(e)


def process_single_problem(
    problem_data: Dict[str, Any],
    apprentice: Apprentice,
    oracle: Optional[Oracle],
    mode: str
) -> PipelineResult:
    """Process a single problem through the pipeline"""
    
    problem_id = problem_data.get('id', 'unknown')
    problem = problem_data.get('input', problem_data.get('problem', problem_data.get('question', '')))
    solution_steps = problem_data.get('solution_steps', problem_data.get('solution', None))
    ground_truth = extract_ground_truth(problem_data)
    
    if not problem:
        return PipelineResult(
            problem_id=str(problem_id),
            problem="",
            solution_steps=solution_steps,
            ground_truth=ground_truth,
            apprentice_code="",
            apprentice_output="",
            apprentice_answer=None,
            apprentice_correct=False,
            apprentice_error="No problem text found",
            source="none"
        )
    
    # Step 1: Apprentice tries with PoT
    app_code, app_output, app_answer, app_error = apprentice.solve(problem)
    app_correct = answers_match(app_answer, ground_truth)
    
    # Step 2: Verifier checks result
    result = PipelineResult(
        problem_id=str(problem_id),
        problem=problem,
        solution_steps=solution_steps,
        ground_truth=ground_truth,
        apprentice_code=app_code,
        apprentice_output=app_output,
        apprentice_answer=app_answer,
        apprentice_correct=app_correct,
        apprentice_error=app_error,
    )
    
    # Step 3: Oracle intervention (only in train mode when Apprentice fails)
    if mode == "train" and oracle and not app_correct:
        oracle_reasoning, oracle_answer, oracle_code, oracle_error = oracle.solve(problem, with_code=True)
        oracle_correct = answers_match(oracle_answer, ground_truth)
        
        result.oracle_reasoning = oracle_reasoning
        result.oracle_answer = oracle_answer
        result.oracle_correct = oracle_correct
        result.oracle_used = True
        
        # Use Oracle's code if available and better
        if oracle_code and oracle_correct:
            result.apprentice_code = oracle_code  # Store Oracle's code for fine-tuning
    
    # Step 4: Determine final answer and source
    if app_correct:
        result.final_answer = app_answer
        result.final_correct = True
        result.source = "apprentice"
    elif mode == "train" and result.oracle_used and result.oracle_correct:
        result.final_answer = oracle_answer
        result.final_correct = True
        result.source = "oracle"
    else:
        result.final_answer = app_answer if app_answer is not None else result.oracle_answer
        result.final_correct = False
        result.source = "apprentice" if mode == "test" else ("oracle" if result.oracle_used else "apprentice")
    
    return result


def run_pipeline(
    input_file: Path,
    output_file: Path,
    mode: str,
    max_workers: int = 10,
    max_problems: Optional[int] = None
):
    """Run the pipeline on dataset"""
    
    print(f"Loading problems from {input_file}...")
    
    problems = []
    with open(input_file, 'r') as f:
        for i, line in enumerate(f):
            if max_problems and i >= max_problems:
                break
            if line.strip():
                problems.append(json.loads(line))
    
    print(f"Processing {len(problems)} problems in {mode.upper()} mode with {max_workers} workers...")
    
    # Initialize models
    llm = ChatOpenAI(
        model="meta-llama/llama-3.2-3b-instruct",
        base_url="https://openrouter.ai/api/v1",
        temperature=0.0,
    )
    
    apprentice = Apprentice(llm)
    oracle = Oracle(llm) if mode == "train" else None
    
    results = []
    
    # Process problems in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_problem = {
            executor.submit(process_single_problem, problem, apprentice, oracle, mode): problem
            for problem in problems
        }
        
        for future in tqdm(as_completed(future_to_problem), total=len(problems), desc="Processing"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                problem = future_to_problem[future]
                print(f"\nError processing problem {problem.get('id', 'unknown')}: {e}")
    
    # Analyze results
    total = len(results)
    apprentice_correct = sum(1 for r in results if r.apprentice_correct)
    oracle_used = sum(1 for r in results if r.oracle_used)
    final_correct = sum(1 for r in results if r.final_correct)
    
    print("\n" + "="*70)
    print(f"PIPELINE RESULTS - {mode.upper()} MODE")
    print("="*70)
    print(f"Total problems:           {total}")
    print(f"Apprentice correct:       {apprentice_correct} ({apprentice_correct/total*100:.1f}%)")
    if mode == "train":
        print(f"Oracle invoked:           {oracle_used} ({oracle_used/total*100:.1f}%)")
        oracle_correct = sum(1 for r in results if r.oracle_used and r.oracle_correct)
        print(f"Oracle correct:           {oracle_correct}/{oracle_used} ({oracle_correct/oracle_used*100:.1f}% of invocations)" if oracle_used > 0 else "Oracle correct:           0/0 (0.0%)")
    print(f"Final accuracy:           {final_correct} ({final_correct/total*100:.1f}%)")
    print("="*70)
    
    # Save results to JSONL for fine-tuning
    print(f"\nSaving results to {output_file}...")
    with open(output_file, 'w') as f:
        for result in results:
            # Convert to dict for JSON serialization
            result_dict = asdict(result)
            f.write(json.dumps(result_dict) + '\n')
    
    print(f"Results saved! {len(results)} problems processed.")
    
    # Sample some results
    if mode == "train":
        oracle_saves = [r for r in results if r.oracle_used and r.oracle_correct and not r.apprentice_correct]
        if oracle_saves:
            print(f"\nOracle saved {len(oracle_saves)} problems that Apprentice failed!")
            print("\nSample Oracle interventions:")
            for i, result in enumerate(oracle_saves[:3], 1):
                print(f"\n{i}. Problem ID: {result.problem_id}")
                print(f"   Problem: {result.problem[:100]}...")
                print(f"   Apprentice answer: {result.apprentice_answer}")
                print(f"   Oracle answer: {result.oracle_answer}")
                print(f"   Ground truth: {result.ground_truth}")


def run_single_problem(problem_text: str, mode: str):
    """Run pipeline on a single problem"""
    
    print(f"Running in {mode.upper()} mode...")
    
    llm = ChatOpenAI(
        model="meta-llama/llama-3.2-3b-instruct",
        base_url="https://openrouter.ai/api/v1",
        temperature=0.0,
    )
    
    apprentice = Apprentice(llm)
    oracle = Oracle(llm) if mode == "train" else None
    
    problem_data = {
        'id': 'single_test',
        'input': problem_text,
        'output': None
    }
    
    result = process_single_problem(problem_data, apprentice, oracle, mode)
    
    print("\n" + "="*70)
    print("APPRENTICE (Program of Thought)")
    print("="*70)
    print("\nGenerated Code:")
    print(result.apprentice_code)
    print("\nExecution Output:")
    print(result.apprentice_output)
    print(f"\nAnswer: {result.apprentice_answer}")
    if result.apprentice_error:
        print(f"Error: {result.apprentice_error}")
    
    if mode == "train" and result.oracle_used:
        print("\n" + "="*70)
        print("ORACLE (Bare LLM)")
        print("="*70)
        print("\nReasoning:")
        print(result.oracle_reasoning)
        print(f"\nAnswer: {result.oracle_answer}")
    
    print("\n" + "="*70)
    print("FINAL RESULT")
    print("="*70)
    print(f"Final Answer: {result.final_answer}")
    print(f"Source: {result.source.upper()}")


def main():
    parser = argparse.ArgumentParser(
        description='Simplified Pipeline: Apprentice (PoT) + Verifier + Oracle',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train mode (collect Oracle solutions for fine-tuning)
  python3 simple_pipeline.py --input data.jsonl --output train_output.jsonl --mode train
  
  # Test mode (only Apprentice with PoT)
  python3 simple_pipeline.py --input data.jsonl --output test_output.jsonl --mode test
  
  # Single problem
  python3 simple_pipeline.py --single "A store has 240 items. 35%% are on sale. How many?" --mode train
        """
    )
    
    parser.add_argument('--input', type=str, help='Input JSONL file')
    parser.add_argument('--output', type=str, help='Output JSONL file')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'test'],
                       help='train: Use Oracle when Apprentice fails | test: Only Apprentice')
    parser.add_argument('--workers', type=int, default=10, help='Number of parallel workers')
    parser.add_argument('--max-problems', type=int, help='Maximum number of problems to process')
    parser.add_argument('--single', type=str, help='Process a single problem (provide problem text)')
    
    args = parser.parse_args()
    
    if args.single:
        run_single_problem(args.single, args.mode)
    else:
        if not args.input or not args.output:
            parser.error("--input and --output are required for dataset mode")
        
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: Input file {input_path} not found")
            sys.exit(1)
        
        output_path = Path(args.output)
        
        run_pipeline(
            input_path,
            output_path,
            args.mode,
            max_workers=args.workers,
            max_problems=args.max_problems
        )


if __name__ == "__main__":
    main()
