#!/usr/bin/env python3
"""
Program of Thought with PROPER Tool-Calling (Llama 3.2 Native)
Uses the model's built-in tool-calling to execute Python code iteratively.
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

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
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
    iterations: int = 0


@tool
def execute_python(code: str) -> str:
    """
    Execute Python code and return the result.
    Use this tool to perform calculations, solve math problems, or execute any Python code.
    The code should print the final answer.
    
    Args:
        code: Python code to execute
        
    Returns:
        The output of the code execution
    """
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
            return f"Error: {error}"
        
        return output if output else "Code executed but produced no output"
        
    except Exception as e:
        return f"Execution failed: {str(e)}"


def extract_final_answer(text: str) -> Optional[float]:
    """Extract numerical answer from text"""
    # Look for patterns like "answer is 42" or "final answer: 42" or just "42"
    patterns = [
        r'(?:answer|result|final|output)[\s:=]+([0-9,.]+)',
        r'^([0-9,.]+)$',
        r'([0-9,.]+)\s*$',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text.strip(), re.IGNORECASE | re.MULTILINE)
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


def test_single_problem_with_tools(problem_data: Dict[str, Any], llm: ChatOpenAI) -> ProblemResult:
    """Test a single problem using proper tool-calling"""
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
    
    # Create agent with Python execution tool
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a math problem solver. You have access to a Python code executor.

To solve math problems:
1. Use the execute_python tool to write and run Python code
2. The code should calculate the answer and print it
3. After getting the result, state the final answer clearly

Be concise and solve the problem directly."""),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    tools = [execute_python]
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=False,
        max_iterations=5,
        handle_parsing_errors=True
    )
    
    try:
        result = agent_executor.invoke({"input": problem_text})
        output_text = result.get('output', '')
        
        # Extract numerical answer from agent output
        predicted_answer = extract_final_answer(output_text)
        
        # Also check intermediate steps for Python output
        if predicted_answer is None and 'intermediate_steps' in result:
            for step in result['intermediate_steps']:
                if len(step) > 1:
                    tool_output = str(step[1])
                    predicted_answer = extract_final_answer(tool_output)
                    if predicted_answer is not None:
                        break
        
        # Check correctness
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
            generated_code=str(result.get('intermediate_steps', [])),
            execution_output=output_text,
            success=predicted_answer is not None,
            iterations=len(result.get('intermediate_steps', []))
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


def run_tests(input_file: Path, max_workers: int = 5, max_problems: Optional[int] = None) -> List[ProblemResult]:
    """Run tests on multiple problems with parallel processing"""
    print(f"Loading problems from {input_file}...")
    
    problems = []
    with open(input_file, 'r') as f:
        for i, line in enumerate(f):
            if max_problems and i >= max_problems:
                break
            if line.strip():
                problems.append(json.loads(line))
    
    print(f"Testing {len(problems)} problems with {max_workers} workers (tool-calling mode)...")
    
    llm = ChatOpenAI(
        model="meta-llama/llama-3.2-3b-instruct",
        base_url="https://openrouter.ai/api/v1",
        temperature=0.0,
    )
    
    results = []
    
    # Lower max_workers for tool-calling (more complex per request)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_problem = {
            executor.submit(test_single_problem_with_tools, problem, llm): problem 
            for problem in problems
        }
        
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
    avg_iterations = sum(r.iterations for r in results) / total if total > 0 else 0
    
    accuracy = (correct / total * 100) if total > 0 else 0
    success_rate = (successful / total * 100) if total > 0 else 0
    
    print("\n" + "="*60)
    print("TOOL-CALLING POT TEST RESULTS")
    print("="*60)
    print(f"Total problems:        {total}")
    print(f"Correct answers:       {correct} ({accuracy:.1f}%)")
    print(f"Successful executions: {successful} ({success_rate:.1f}%)")
    print(f"Avg iterations:        {avg_iterations:.1f}")
    print("="*60)
    
    # Sample some errors
    errors = [r for r in results if not r.success]
    if errors:
        print(f"\nSample errors ({min(3, len(errors))} of {len(errors)}):")
        for i, error in enumerate(errors[:3], 1):
            print(f"\n{i}. Problem ID: {error.problem_id}")
            print(f"   Error: {error.error_message}")
    
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
        'avg_iterations': avg_iterations
    }


def main():
    parser = argparse.ArgumentParser(description='PoT with Tool-Calling')
    parser.add_argument('--input', type=str, required=True, help='Input JSONL file')
    parser.add_argument('--output', type=str, default='pot_llm_results_toolcalling.json', 
                       help='Output JSON file for results')
    parser.add_argument('--workers', type=int, default=5, help='Number of parallel workers')
    parser.add_argument('--max-problems', type=int, help='Maximum number of problems to test')
    parser.add_argument('--single', type=str, help='Test a single problem')
    
    args = parser.parse_args()
    
    llm = ChatOpenAI(
        model="meta-llama/llama-3.2-3b-instruct",
        base_url="https://openrouter.ai/api/v1",
        temperature=0.0,
    )
    
    if args.single:
        print("Testing single problem with tool-calling...")
        problem_data = {'id': 'single_test', 'input': args.single, 'output': None}
        result = test_single_problem_with_tools(problem_data, llm)
        
        print(f"\nPredicted Answer: {result.predicted_answer}")
        print(f"Success: {result.success}")
        print(f"Iterations: {result.iterations}")
        print(f"\nOutput: {result.execution_output}")
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
                'iterations': r.iterations,
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
