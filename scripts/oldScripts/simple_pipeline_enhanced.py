#!/usr/bin/env python3
"""
Enhanced Simplified Pipeline: Apprentice (PoT) + Oracle (Enhanced PoT)

Architecture:
- Apprentice: Program of Thought (generates Python code) - Llama 3.2 3B via API
- Oracle: Enhanced PoT with better model (generates well-commented Python code)

Fine-tuning Format:
- {"instruction": "problem", "response": "python_code", "correct": true/false}
- Use Apprentice's code if correct, Oracle's if not

Modes:
- Train: Collect correct solutions (Apprentice if correct, else Oracle)
- Test: Only Apprentice (evaluation)
"""

import json
import argparse
import io
import re
import math
import signal
import threading
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeoutError
from tqdm import tqdm

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


# Timeout exception
class TimeoutError(Exception):
    pass


def timeout_handler(func, args=(), kwargs=None, timeout_duration=60, default=None):
    """Execute function with timeout using threading"""
    if kwargs is None:
        kwargs = {}
    
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None
            self.exception = None
            
        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except Exception as e:
                self.exception = e
    
    thread = InterruptableThread()
    thread.daemon = True
    thread.start()
    thread.join(timeout_duration)
    
    if thread.is_alive():
        # Timeout occurred
        return default, TimeoutError(f"Function call timed out after {timeout_duration} seconds")
    
    if thread.exception:
        return default, thread.exception
    
    return thread.result, None


# Global flag for graceful shutdown
shutdown_requested = False

def handle_shutdown_signal(signum, frame):
    """Handle SIGINT (Ctrl+C) and SIGTERM gracefully"""
    global shutdown_requested
    shutdown_requested = True
    print("\n\nShutdown signal received. Finishing current problems and saving progress...")

# Register signal handlers
signal.signal(signal.SIGINT, handle_shutdown_signal)
signal.signal(signal.SIGTERM, handle_shutdown_signal)


@dataclass
class PipelineResult:
    problem_id: str
    problem: str
    ground_truth: Optional[float]
    
    # For fine-tuning dataset (simplified format)
    instruction: str  # The problem
    response: str  # The correct Python code (from Apprentice if correct, else Oracle)
    
    # Metadata for analysis
    source: str = ""  # "apprentice" or "oracle" or "oracle_with_steps"
    final_answer: Optional[float] = None
    final_correct: bool = False


def extract_python_code(response: str) -> Optional[str]:
    """Extract Python code from LLM response - flexible extraction"""
    # Try standard markdown code block
    pattern = r'```(?:python)?\s*\n(.*?)\n```'
    matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
    
    if matches:
        return matches[-1].strip()  # Return last code block
    
    # If no code block markers, check if entire response looks like code
    lines = response.strip().split('\n')
    has_python = any(keyword in response.lower() for keyword in ['print', '=', 'def ', '#'])
    has_natural_language = any(word in response.lower() for word in ['the answer is', 'therefore', 'we need to', 'first,'])
    
    # If it looks like code and doesn't look like prose, use it
    if has_python and not has_natural_language and len(lines) > 1:
        return response.strip()
    
    return None


def execute_python_code(code: str) -> Tuple[Optional[float], str, Optional[str]]:
    """
    Execute Python code and extract numerical answer from output
    
    Returns: (answer, output, error)
    """
    try:
        # Restricted namespace for safety
        safe_namespace = {
            'print': print,
            'abs': abs,
            'round': round,
            'min': min,
            'max': max,
            'sum': sum,
            'len': len,
            'range': range,
            'enumerate': enumerate,
            'zip': zip,
            'sorted': sorted,
            'reversed': reversed,
            'map': map,
            'filter': filter,
            'int': int,
            'float': float,
            'str': str,
            'list': list,
            'dict': dict,
            'set': set,
            'tuple': tuple,
            'math': math,
            '__builtins__': {}
        }
        
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(code, safe_namespace)
        
        output = stdout_capture.getvalue()
        errors = stderr_capture.getvalue()
        
        if errors:
            return None, output, errors
        
        # Extract numerical answer from output
        answer = extract_answer_from_output(output)
        
        return answer, output, None
    
    except Exception as e:
        return None, "", str(e)


def extract_answer_from_output(output: str) -> Optional[float]:
    """Extract numerical answer from code output"""
    if not output:
        return None
    
    # Get last line (usually contains the final answer)
    lines = [line.strip() for line in output.strip().split('\n') if line.strip()]
    if not lines:
        return None
    
    last_line = lines[-1]
    
    # Remove common formatting (%, $, commas, etc.)
    cleaned = re.sub(r'[^\d\.\-\+e]', ' ', last_line)
    
    # Find all numbers
    numbers = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', cleaned)
    
    if numbers:
        try:
            return float(numbers[-1])  # Return last number found
        except ValueError:
            pass
    
    return None


def extract_answer_from_text(text: str) -> Optional[float]:
    """Extract numerical answer from natural language text"""
    if not text:
        return None
    
    # Look for common answer patterns
    patterns = [
        r'(?:answer|result|solution)(?:\s+is)?[:\s]+\$?([-+]?[\d,]+\.?\d*)',
        r'\$?([-+]?[\d,]+\.?\d*)\s*$',
        r'####\s+([-+]?[\d,]+\.?\d*)',
        r'= ?\$?([-+]?[\d,]+\.?\d*)\s*$'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
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
    
    def solve(self, problem: str, timeout: int = 60) -> Tuple[str, str, Optional[float], Optional[str]]:
        """
        Solve using PoT (generate Python code and execute)
        
        Args:
            problem: The math problem
            timeout: Timeout in seconds for LLM call (default 60s)
        
        Returns: (code, output, answer, error)
        """
        # Ultra-simple prompt (70.2% accuracy - our best PoT)
        prompt = f"""Write Python code to solve this math problem. Print only the final numerical answer.

Problem: {problem}

Python code:"""
        
        try:
            # Call LLM with timeout protection
            result, error = timeout_handler(
                self.llm.invoke,
                args=(prompt,),
                timeout_duration=timeout,
                default=None
            )
            
            if error:
                return "", "", None, f"LLM timeout or error: {str(error)}"
            
            response_text = result.content
            
            code = extract_python_code(response_text)
            
            if not code:
                return response_text, "", None, "Failed to extract Python code"
            
            answer, output, error = execute_python_code(code)
            
            return code, output, answer, error
            
        except Exception as e:
            return "", "", None, str(e)


class Oracle:
    """Oracle solver - generates well-commented PoT code (no ReAct overhead)"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def solve(self, problem: str, solution_steps: Optional[str] = None, timeout: int = 60) -> Tuple[str, Optional[float], Optional[str]]:
        """
        Generate well-commented Python code to solve the problem
        
        Args:
            problem: The math problem
            solution_steps: Optional ground truth solution steps (used when Oracle fails first attempt)
            timeout: Timeout in seconds for LLM calls (default 60s)
        
        Returns: (code, answer, error)
        """
        try:
            # First attempt: Direct PoT with detailed comments
            if solution_steps is None:
                prompt = f"""Write Python code to solve this math problem. Use clear variable names and add comments explaining each step.

Problem: {problem}

Python code:"""
            else:
                # Second attempt: Use solution steps as guidance
                prompt = f"""Write Python code to solve this math problem. Follow the solution steps provided.

Problem: {problem}

Solution Steps:
{solution_steps}

Write Python code that implements the above solution. Use clear variable names and add comments for each step. Print only the final numerical answer.

Python code:"""
            
            # Call LLM with timeout protection
            result, error = timeout_handler(
                self.llm.invoke,
                args=(prompt,),
                timeout_duration=timeout,
                default=None
            )
            
            if error:
                return "", None, f"LLM timeout or error: {str(error)}"
            
            code = extract_python_code(result.content)
            
            if not code:
                code = result.content  # Try using raw response
            
            # Execute the code
            answer, output, exec_error = execute_python_code(code)
            
            if exec_error:
                # Try to fix the code with timeout protection
                debug_prompt = f"""The following code has an error. Fix it.

Problem: {problem}

Code:
```python
{code}
```

Error: {exec_error}

Provide corrected Python code:"""
                
                # Try debug with timeout
                debug_result, debug_error = timeout_handler(
                    self.llm.invoke,
                    args=(debug_prompt,),
                    timeout_duration=timeout,
                    default=None
                )
                
                if debug_error or not debug_result:
                    return code, None, f"Code execution failed: {exec_error}. Debug timeout."
                
                fixed_code = extract_python_code(debug_result.content)
                
                if fixed_code:
                    answer, output, exec_error = execute_python_code(fixed_code)
                    code = fixed_code
                    
                    if exec_error:
                        return code, None, f"Code execution failed after fix: {exec_error}"
                else:
                    return code, None, f"Code execution failed: {exec_error}"
            
            return code, answer, None
            
        except Exception as e:
            return "", None, str(e)


def process_single_problem(
    problem_data: Dict[str, Any],
    apprentice: Apprentice,
    oracle: Optional[Oracle],
    mode: str = "test",
    use_oracle: bool = False
) -> PipelineResult:
    """Process a single problem through the pipeline
    
    Args:
        problem_data: Problem dictionary
        apprentice: Apprentice solver
        oracle: Oracle solver (optional)
        mode: 'train' or 'test'
        use_oracle: If True, use Oracle even in test mode
    """
    
    problem_id = str(problem_data.get('id', '0'))
    problem = problem_data.get('input', problem_data.get('question', ''))
    solution_steps = problem_data.get('solution_steps', problem_data.get('solution', ''))
    ground_truth = extract_ground_truth(problem_data)
    
    # Step 1: Apprentice solves with PoT (with 60s timeout)
    apprentice_code, apprentice_output, apprentice_answer, apprentice_error = apprentice.solve(problem, timeout=60)
    apprentice_correct = answers_match(apprentice_answer, ground_truth)
    
    # Determine the correct response for fine-tuning
    if mode == "test" and not use_oracle:
        # Test mode without Oracle: Just use apprentice
        return PipelineResult(
            problem_id=problem_id,
            problem=problem,
            ground_truth=ground_truth,
            instruction=problem,
            response=apprentice_code,
            source="apprentice",
            final_answer=apprentice_answer,
            final_correct=apprentice_correct
        )
    
    # Train mode or test mode with Oracle: Find the correct solution
    if apprentice_correct:
        # Apprentice got it right, use its code
        return PipelineResult(
            problem_id=problem_id,
            problem=problem,
            ground_truth=ground_truth,
            instruction=problem,
            response=apprentice_code,
            source="apprentice",
            final_answer=apprentice_answer,
            final_correct=True
        )
    
    # Apprentice failed, try Oracle
    if oracle is not None and ground_truth is not None:
        try:
            # First attempt: Oracle without solution steps
            oracle_code, oracle_answer, oracle_error = oracle.solve(problem, timeout=90)
            oracle_correct = answers_match(oracle_answer, ground_truth)
            
            if oracle_correct:
                # Oracle got it right
                return PipelineResult(
                    problem_id=problem_id,
                    problem=problem,
                    ground_truth=ground_truth,
                    instruction=problem,
                    response=oracle_code,
                    source="oracle",
                    final_answer=oracle_answer,
                    final_correct=True
                )
            
            # Oracle failed too, give it the solution steps (only in train mode)
            if solution_steps and mode == "train":
                oracle_code_guided, oracle_answer_guided, oracle_error_guided = oracle.solve(problem, solution_steps, timeout=90)
                oracle_correct_guided = answers_match(oracle_answer_guided, ground_truth)
                
                if oracle_correct_guided:
                    # Oracle got it with guidance
                    return PipelineResult(
                        problem_id=problem_id,
                        problem=problem,
                        ground_truth=ground_truth,
                        instruction=problem,
                        response=oracle_code_guided,
                        source="oracle_with_steps",
                        final_answer=oracle_answer_guided,
                        final_correct=True
                    )
                
                # Even with guidance, Oracle failed - use the guided attempt anyway
                return PipelineResult(
                    problem_id=problem_id,
                    problem=problem,
                    ground_truth=ground_truth,
                    instruction=problem,
                    response=oracle_code_guided if oracle_code_guided else oracle_code,
                    source="oracle_with_steps",
                    final_answer=oracle_answer_guided if oracle_answer_guided else oracle_answer,
                    final_correct=False
                )
            
            # No solution steps available or test mode, use Oracle's attempt
            return PipelineResult(
                problem_id=problem_id,
                problem=problem,
                ground_truth=ground_truth,
                instruction=problem,
                response=oracle_code if oracle_code else apprentice_code,
                source="oracle" if oracle_code else "apprentice",
                final_answer=oracle_answer if oracle_answer else apprentice_answer,
                final_correct=False
            )
        
        except Exception as e:
            # Oracle failed with exception, use apprentice's attempt
            print(f"\nOracle exception for problem {problem_id}: {str(e)}")
            return PipelineResult(
                problem_id=problem_id,
                problem=problem,
                ground_truth=ground_truth,
                instruction=problem,
                response=apprentice_code,
                source="apprentice",
                final_answer=apprentice_answer,
                final_correct=False
            )
    
    # No Oracle available, use apprentice's attempt
    return PipelineResult(
        problem_id=problem_id,
        problem=problem,
        ground_truth=ground_truth,
        instruction=problem,
        response=apprentice_code,
        source="apprentice",
        final_answer=apprentice_answer,
        final_correct=False
    )


def run_pipeline(
    input_file: Path,
    output_file: Path,
    mode: str = "test",
    max_problems: Optional[int] = None,
    workers: int = 5,
    oracle_model: str = "google/gemini-flash-1.5-8b",
    use_oracle: bool = False
):
    """Run the pipeline on a dataset
    
    Args:
        input_file: Input JSONL file
        output_file: Output JSONL file
        mode: 'train' or 'test'
        max_problems: Maximum problems to process
        workers: Number of parallel workers
        oracle_model: Model for Oracle
        use_oracle: If True, use Oracle even in test mode (default False)
    """
    
    # Initialize Apprentice LLM (Llama 3.2 3B via OpenRouter API)
    apprentice_llm = ChatOpenAI(
        model="meta-llama/llama-3.2-3b-instruct",
        base_url="https://openrouter.ai/api/v1",
        temperature=0.0,
        max_tokens=2048
    )
    
    # Initialize Oracle LLM (more powerful model from OpenRouter)
    oracle_llm = ChatOpenAI(
        model=oracle_model,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.0,
        max_tokens=4096
    ) if (mode == "train" or use_oracle) else None
    
    # Initialize components
    apprentice = Apprentice(apprentice_llm)
    oracle = Oracle(oracle_llm) if (mode == "train" or use_oracle) else None
    
    # Load dataset
    with open(input_file) as f:
        problems = [json.loads(line) for line in f]
    
    if max_problems:
        problems = problems[:max_problems]
    
    print(f"Processing {len(problems)} problems in {mode} mode...")
    print(f"Apprentice Model: meta-llama/llama-3.2-3b-instruct (OpenRouter API)")
    if mode == "train" or use_oracle:
        print(f"Oracle Model: {oracle_model}")
    if use_oracle and mode == "test":
        print(f"Oracle enabled in test mode")
    print(f"Using {workers} workers")
    print(f"Output: {output_file}")
    
    results = []
    
    # Open output file in append mode - write results immediately as they complete
    with open(output_file, 'a') as out_f:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_single_problem, problem, apprentice, oracle, mode, use_oracle): problem
                for problem in problems
            }
            
            with tqdm(total=len(problems), desc="Processing") as pbar:
                for future in as_completed(futures):
                    # Check if shutdown was requested
                    if shutdown_requested:
                        print("\n\nShutdown in progress. Saving completed results and exiting...")
                        executor.shutdown(wait=False, cancel_futures=True)
                        break
                    
                    try:
                        # Get result with timeout (120s max per problem)
                        result = future.result(timeout=120)
                        results.append(result)
                        
                        # Write this result immediately (append mode)
                        fine_tune_data = {
                            "instruction": result.instruction,
                            "response": result.response,
                            "correct": result.final_correct
                        }
                        out_f.write(json.dumps(fine_tune_data) + '\n')
                        out_f.flush()  # Force write to disk immediately
                        
                    except FutureTimeoutError:
                        print(f"\n⚠️  Problem timed out after 120s, skipping...")
                    except Exception as e:
                        print(f"\nError processing problem: {e}")
                        import traceback
                        traceback.print_exc()
                    
                    pbar.update(1)
    
    # Sort results by problem_id (for statistics only, already written to file)
    results.sort(key=lambda x: int(x.problem_id))
    
    # Print statistics
    total = len(results)
    correct = sum(1 for r in results if r.final_correct)
    apprentice_source = sum(1 for r in results if r.source == "apprentice")
    oracle_source = sum(1 for r in results if r.source == "oracle")
    oracle_with_steps = sum(1 for r in results if r.source == "oracle_with_steps")
    
    print(f"\n{'='*60}")
    print(f"Results ({mode} mode):")
    print(f"{'='*60}")
    print(f"Total problems: {total}")
    print(f"Correct solutions: {correct}/{total} ({correct/total*100:.1f}%)")
    
    if mode == "train":
        print(f"\nSource breakdown:")
        print(f"  Apprentice: {apprentice_source}/{total} ({apprentice_source/total*100:.1f}%)")
        print(f"  Oracle: {oracle_source}/{total} ({oracle_source/total*100:.1f}%)")
        print(f"  Oracle with steps: {oracle_with_steps}/{total} ({oracle_with_steps/total*100:.1f}%)")
    
    print(f"{'='*60}")


def run_single_problem(problem_text: str, mode: str = "test", oracle_model: str = "google/gemini-flash-1.5-8b", use_oracle: bool = False):
    """Run pipeline on a single problem (for testing)"""
    
    # Initialize Apprentice LLM (Llama 3.2 3B)
    apprentice_llm = ChatOpenAI(
        model="meta-llama/llama-3.2-3b-instruct",
        base_url="https://openrouter.ai/api/v1",
        temperature=0.0,
        max_tokens=2048
    )
    
    # Initialize Oracle LLM (more powerful model)
    oracle_llm = ChatOpenAI(
        model=oracle_model,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.0,
        max_tokens=4096
    ) if (mode == "train" or use_oracle) else None
    
    # Initialize components
    apprentice = Apprentice(apprentice_llm)
    oracle = Oracle(oracle_llm) if (mode == "train" or use_oracle) else None
    
    print(f"Apprentice Model: meta-llama/llama-3.2-3b-instruct")
    if mode == "train" or use_oracle:
        print(f"Oracle Model: {oracle_model}")
    
    # Create problem data
    problem_data = {
        'id': '0',
        'input': problem_text,
        'solution_steps': None,
        'output': None
    }
    
    result = process_single_problem(problem_data, apprentice, oracle, mode, use_oracle)
    
    print(f"\n{'='*60}")
    print(f"PROBLEM:")
    print(result.problem)
    print(f"\n{'='*60}")
    print(f"FINE-TUNING FORMAT:")
    print(f"Instruction: {result.instruction}")
    print(f"Response (Python code):")
    print(result.response)
    print(f"\n{'='*60}")
    print(f"FINAL RESULT:")
    print(f"Answer: {result.final_answer}")
    print(f"Source: {result.source}")
    print(f"Ground Truth: {result.ground_truth}")
    print(f"Correct: {result.final_correct}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Enhanced Simplified Pipeline: Apprentice + Oracle with ReAct")
    parser.add_argument('--input', type=str, help='Input JSONL file')
    parser.add_argument('--output', type=str, help='Output JSONL file')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='test',
                        help='Mode: train (collect Oracle solutions) or test (Apprentice only by default)')
    parser.add_argument('--max-problems', type=int, help='Maximum number of problems to process')
    parser.add_argument('--workers', type=int, default=5, help='Number of parallel workers')
    parser.add_argument('--single', type=str, help='Run on a single problem (for testing)')
    parser.add_argument('--oracle-model', type=str, default='google/gemini-flash-1.5-8b',
                        help='Model to use for Oracle (default: google/gemini-flash-1.5-8b). Options: google/gemini-flash-1.5-8b, google/gemini-2.0-flash-exp:free, google/gemma-2-27b-it, openai/gpt-4o-mini, meta-llama/llama-3.1-70b-instruct, etc.')
    parser.add_argument('--use-oracle', action='store_true',
                        help='Use Oracle even in test mode (default: False). Useful for comparing Apprentice vs Oracle accuracy.')
    
    args = parser.parse_args()
    
    if args.single:
        run_single_problem(args.single, args.mode, args.oracle_model, args.use_oracle)
    elif args.input and args.output:
        run_pipeline(
            Path(args.input),
            Path(args.output),
            mode=args.mode,
            max_problems=args.max_problems,
            workers=args.workers,
            oracle_model=args.oracle_model,
            use_oracle=args.use_oracle
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
