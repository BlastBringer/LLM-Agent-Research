#!/usr/bin/env python3
"""
Bare LLM Benchmark for GSM-Symbolic
Tests Llama 3.2-3B directly without any pipeline processing
"""

import json
import os
import re
import math
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from dotenv import load_dotenv

# Try to import LangChain
try:
    from langchain_openai import ChatOpenAI
    LANGCHAIN_AVAILABLE = True
except ImportError:
    print("❌ LangChain not available. Install with: pip install langchain langchain-openai")
    exit(1)

load_dotenv()


def calculator_tool(expression: str) -> str:
    """
    A safe calculator that evaluates mathematical expressions.
    Supports basic arithmetic: +, -, *, /, **, (), and common math functions.
    
    Args:
        expression: Mathematical expression to evaluate (e.g., "63 / 13" or "4.846 * 39")
    
    Returns:
        The result of the calculation as a string
    """
    try:
        # Remove any whitespace
        expression = expression.strip()
        
        # Create a safe namespace with only math functions
        safe_dict = {
            'abs': abs,
            'round': round,
            'min': min,
            'max': max,
            'sum': sum,
            'pow': pow,
            'sqrt': math.sqrt,
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'pi': math.pi,
            'e': math.e,
        }
        
        # Evaluate the expression safely
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        
        # Format the result nicely
        if isinstance(result, float):
            # Round to reasonable precision
            if abs(result - round(result)) < 0.0001:
                return str(int(round(result)))
            else:
                return f"{result:.4f}".rstrip('0').rstrip('.')
        else:
            return str(result)
            
    except Exception as e:
        return f"Error: Could not evaluate '{expression}'. {str(e)}"


def process_calculator_expressions(text: str) -> str:
    """
    Find and evaluate CALC[expression] in the text.
    Replaces each CALC[...] with the computed result.
    
    Args:
        text: Text containing CALC[...] expressions
        
    Returns:
        Text with CALC[...] replaced by calculated values
    """
    def replace_calc(match):
        expression = match.group(1)
        result = calculator_tool(expression)
        return f"{result}"
    
    # Find all CALC[...] patterns and replace with calculated values
    processed = re.sub(r'CALC\[(.*?)\]', replace_calc, text)
    return processed


def extract_numeric_answer(text: str) -> Optional[float]:
    """
    Extract numeric answer from LLM response.
    Looks for patterns like:
    - "The answer is 42"
    - "Final answer: 42"
    - "= 42"
    - "#### 42" (GSM format)
    """
    # Try various patterns
    patterns = [
        r'####\s*([+-]?\d+\.?\d*)',  # GSM format
        r'[Ff]inal [Aa]nswer[:\s]+([+-]?\d+\.?\d*)',
        r'[Tt]he answer is[:\s]+([+-]?\d+\.?\d*)',
        r'[Aa]nswer[:\s]+([+-]?\d+\.?\d*)',
        r'=\s*([+-]?\d+\.?\d*)\s*$',  # Ends with = number
        r'([+-]?\d+\.?\d*)\s*$',  # Last number in response
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.MULTILINE)
        if match:
            try:
                return float(match.group(1))
            except (ValueError, IndexError):
                continue
    
    return None


def compare_answers(predicted: Optional[float], ground_truth: float, tolerance: float = 0.01) -> bool:
    """
    Compare predicted answer with ground truth.
    Uses relative tolerance for percentage-based comparison.
    """
    if predicted is None:
        return False
    
    # Exact match
    if predicted == ground_truth:
        return True
    
    # Handle zero ground truth
    if ground_truth == 0:
        return abs(predicted) < 1e-6
    
    # Relative tolerance check
    relative_diff = abs(predicted - ground_truth) / abs(ground_truth)
    return relative_diff <= tolerance


def test_single_problem(llm, problem_data, problem_num, prompt_template, use_calculator=False):
    """Test a single problem (for parallel execution)."""
    problem_text = problem_data.get("input", "")
    ground_truth = problem_data.get("output", 0)
    
    # Convert ground truth to float if it's not already
    if isinstance(ground_truth, str):
        try:
            ground_truth = float(ground_truth)
        except ValueError:
            return {
                "problem_num": problem_num,
                "input": problem_text,
                "ground_truth": ground_truth,
                "predicted": None,
                "correct": False,
                "response": None,
                "error": f"Invalid ground truth '{ground_truth}'"
            }
    
    # Get LLM response
    try:
        prompt = prompt_template.format(problem=problem_text)
        response = llm.invoke(prompt)
        response_text = response.content
        
        # Process calculator expressions if enabled
        if use_calculator:
            response_text = process_calculator_expressions(response_text)
        
        # Extract answer
        predicted = extract_numeric_answer(response_text)
        
        if predicted is None:
            is_correct = False
        else:
            is_correct = compare_answers(predicted, ground_truth)
        
        return {
            "problem_num": problem_num,
            "input": problem_text,
            "ground_truth": ground_truth,
            "predicted": predicted,
            "correct": is_correct,
            "response": response_text,
            "error": None
        }
        
    except Exception as e:
        return {
            "problem_num": problem_num,
            "input": problem_text,
            "ground_truth": ground_truth,
            "predicted": None,
            "correct": False,
            "response": None,
            "error": str(e)
        }


def test_bare_llm(input_file: str, limit: Optional[int] = None, workers: int = 4, use_calculator: bool = False):
    """
    Test bare LLM (no pipeline) on GSM-Symbolic dataset.
    
    Args:
        input_file: Path to input JSONL file
        limit: Optional limit on number of problems
        workers: Number of parallel workers
        use_calculator: Whether to process CALC[...] expressions in responses
    """
    # Initialize LLM
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    model_name = os.getenv("APPRENTICE_MODEL", "meta-llama/llama-3.2-3b-instruct")
    
    print("=" * 70)
    print("🧪 BARE LLM BENCHMARK - GSM-Symbolic")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Dataset: {input_file}")
    print(f"Workers: {workers}")
    print(f"Calculator: {use_calculator}")
    print()
    
    llm = ChatOpenAI(
        model=model_name,
        temperature=0.1,
        openai_api_key=api_key,
        openai_api_base=base_url,
        max_tokens=1000
    )
    
    # Load dataset
    problems = []
    with open(input_file, 'r') as f:
        for line in f:
            if line.strip():
                problems.append(json.loads(line))
    
    if limit:
        problems = problems[:limit]
    
    print(f"📊 Testing on {len(problems)} problems...")
    print()
    
    # Prompt template with optional calculator instructions
    calculator_instructions = ""
    if use_calculator:
        calculator_instructions = """
YOU HAVE ACCESS TO A CALCULATOR!
Whenever you need to perform any calculation, write it like this:
CALC[expression]

Examples:
- CALC[63 / 13] for division
- CALC[4.8462 * 39] for multiplication
- CALC[100 + 50 - 20] for mixed operations

Use CALC[...] for ALL calculations to ensure accuracy!

"""
    
    prompt_template = f"""{calculator_instructions}Solve this math problem step-by-step and provide the final numerical answer.

Problem:
{{problem}}

Solve it step-by-step{"using CALC[...] for all calculations" if use_calculator else ""} and end your response with:
#### [your numeric answer]

For example:
#### 42

Now solve:"""
    
    # Track results with thread-safe counters
    correct = 0
    total = 0
    failed_to_parse = 0
    results = []
    lock = Lock()
    
    # Process problems in parallel
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Submit all tasks
        future_to_problem = {
            executor.submit(test_single_problem, llm, problem_data, i + 1, prompt_template, use_calculator): i 
            for i, problem_data in enumerate(problems)
        }
        
        # Process completed tasks
        for future in as_completed(future_to_problem):
            result = future.result()
            
            with lock:
                total += 1
                results.append(result)
                
                if result["error"]:
                    print(f"❌ Problem {result['problem_num']}: Error - {result['error']}")
                    failed_to_parse += 1
                elif result["predicted"] is None:
                    print(f"❌ Problem {result['problem_num']}: Failed to extract numeric answer")
                    failed_to_parse += 1
                elif result["correct"]:
                    correct += 1
                    print(f"✅ Problem {result['problem_num']}: Correct ({result['predicted']} == {result['ground_truth']})")
                else:
                    print(f"❌ Problem {result['problem_num']}: Wrong ({result['predicted']} != {result['ground_truth']})")
                
                # Show progress
                if total % 10 == 0:
                    print(f"📊 Progress: {total}/{len(problems)} ({total/len(problems)*100:.1f}%) - Accuracy so far: {correct/total*100:.1f}%")
    
    # Sort results by problem number
    results.sort(key=lambda x: x["problem_num"])
    
    # Print summary
    print()
    print("=" * 70)
    print("📊 RESULTS")
    print("=" * 70)
    print(f"Total Problems: {total}")
    print(f"Correct: {correct}")
    print(f"Wrong: {total - correct - failed_to_parse}")
    print(f"Failed to Parse: {failed_to_parse}")
    print(f"Accuracy: {correct / total * 100:.2f}%")
    print()
    
    # Save detailed results
    output_suffix = "_calculator" if use_calculator else ""
    output_file = f"bare_llm_results{output_suffix}.json"
    with open(output_file, 'w') as f:
        json.dump({
            "model": model_name,
            "dataset": input_file,
            "use_calculator": use_calculator,
            "total": total,
            "correct": correct,
            "wrong": total - correct - failed_to_parse,
            "failed_to_parse": failed_to_parse,
            "accuracy": correct / total if total > 0 else 0,
            "results": results
        }, f, indent=2)
    
    print(f"💾 Detailed results saved to: {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test bare LLM on GSM-Symbolic dataset")
    parser.add_argument("--input", default="gsm_symbolic_batch1.jsonl", 
                       help="Input JSONL file")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of problems to test")
    parser.add_argument("--workers", type=int, default=4,
                       help="Number of parallel workers (default: 4)")
    parser.add_argument("--calculator", action="store_true",
                       help="Enable calculator support (processes CALC[...] expressions)")
    
    args = parser.parse_args()
    
    test_bare_llm(args.input, args.limit, args.workers, args.calculator)
