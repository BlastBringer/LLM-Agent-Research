#!/usr/bin/env python3
"""
Single Problem Solver with Full Reasoning Output
=================================================
This program solves a single math problem from a text file using:
- Student: CoT + PoT with self-consistency
- Oracle: 8-shot CoT + calculator with self-consistency

Outputs detailed self-consistency logs and reasoning text for analysis.
No ground truth comparison - purely for problem solving and reasoning inspection.
"""

import os
import json
import re
import argparse
from typing import Optional, List, Tuple, Dict, Any
from pathlib import Path
from collections import Counter

from langchain_openai import ChatOpenAI

# Set OpenRouter API key
os.environ["OPENAI_API_KEY"] = "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"

# 8-shot CoT examples for Oracle (same as dual solver)
COT_EXAMPLES = """Q: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
A: Let's think step by step. Janet starts with 16 eggs per day. She eats 3 for breakfast, so she has 16 - 3 = 13 eggs left. She bakes muffins with 4 eggs, so she has 13 - 4 = 9 eggs left. She sells these 9 eggs for $2 each, so she makes 9 × 2 = $18. The answer is 18.

Q: A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?
A: Let's think step by step. The robe takes 2 bolts of blue fiber. It takes half that much white fiber, which is 2 ÷ 2 = 1 bolt of white fiber. In total, it takes 2 + 1 = 3 bolts. The answer is 3.

Q: Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?
A: Let's think step by step. Josh bought the house for $80,000 and put in $50,000 in repairs, so his total cost is 80,000 + 50,000 = $130,000. The repairs increased the value by 150%, so the increase is 80,000 × 1.5 = $120,000. The new value is 80,000 + 120,000 = $200,000. His profit is 200,000 - 130,000 = $70,000. The answer is 70000.

Q: James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week?
A: Let's think step by step. James runs 3 sprints each time, and he does this 3 times a week, so he runs 3 × 3 = 9 sprints per week. Each sprint is 60 meters, so the total distance is 9 × 60 = 540 meters. The answer is 540.

Q: Every day, Wally eats 4 meals. Each meal consists of some fruits and vegetables. If Wally eats 3 fruits in each meal, how many fruits does he eat in 10 days?
A: Let's think step by step. Wally eats 4 meals per day. In each meal, he eats 3 fruits, so per day he eats 4 × 3 = 12 fruits. Over 10 days, he eats 12 × 10 = 120 fruits. The answer is 120.

Q: A merchant wants to make a choice of purchase between 2 purchase plans: jewelry worth $5,000 or electronic gadgets worth $8,000. His financial advisor speculates that the jewelry market will go up 2.5% while the electronic gadgets market will rise 1.2% within the same month. If the merchant is looking to maximize profit at the end of this month by making a choice, how much profit would this be?
A: Let's think step by step. For jewelry: initial value is $5,000, increase is 2.5%, so profit is 5,000 × 0.025 = $125. For electronics: initial value is $8,000, increase is 1.2%, so profit is 8,000 × 0.012 = $96. The jewelry gives more profit. The maximum profit is $125. The answer is 125.

Q: Two trains leave San Rafael at the same time. They begin traveling westward, both traveling for 80 miles. The next day, they travel northwards, covering 150 miles. What's the distance covered by each train in the two days?
A: Let's think step by step. On the first day, each train travels 80 miles westward. On the second day, each train travels 150 miles northward. The total distance covered by each train is 80 + 150 = 230 miles. The answer is 230.

Q: Jill gets paid $20 per hour to teach and $30 to be a cheerleading coach. If she works 50 weeks a year, 35 hours a week as a teacher and 15 hours a week as a coach, what's her annual salary?
A: Let's think step by step. As a teacher, Jill earns $20 per hour and works 35 hours per week, so she earns 20 × 35 = $700 per week. As a coach, she earns $30 per hour and works 15 hours per week, so she earns 30 × 15 = $450 per week. Her total weekly earnings are 700 + 450 = $1,150. Over 50 weeks, her annual salary is 1,150 × 50 = $57,500. The answer is 57500."""

# PoT prompt template
POT_PROMPT_TEMPLATE = """Write Python code to solve this math problem. Return only the final answer as a number.

Q: {question}

Write Python code to solve this:"""


def extract_answer_from_cot(response: str) -> Optional[float]:
    """Extract numerical answer from CoT response"""
    # Try case-insensitive patterns first
    patterns = [
        r"(?:the\s+)?answer\s+is\s*:?\s*\$?\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)",
        r"(?:the\s+)?(?:answer|final answer)\s+(?:is|=)\s*\$?\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)",
        r"=\s*\$?\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:\.|$)",
        r"(?:therefore|thus|so)\s*(?:the\s+)?(?:answer|total)\s+(?:is|=)\s*\$?\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)",
        r"makes\s+(?:\$)?(-?\d+(?:,\d{3})*(?:\.\d+)?)",
        r"(?:total|sum|result)\s+(?:is|=)\s*\$?\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)",
    ]
    
    response_lower = response.lower()
    
    for pattern in patterns:
        match = re.search(pattern, response_lower)
        if match:
            try:
                value = match.group(1).replace(',', '')
                return float(value)
            except (ValueError, AttributeError):
                continue
    
    # Last resort: look for any number at end of response
    # (sometimes models just give a number as final answer)
    numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', response)
    if numbers:
        try:
            # Return the last number found (most likely to be final answer)
            return float(numbers[-1].replace(',', ''))
        except ValueError:
            pass
    
    return None


def extract_code_from_pot(response: str) -> Optional[str]:
    """Extract Python code from model response"""
    code_pattern = r"```python\n(.*?)\n```"
    match = re.search(code_pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    lines = response.strip().split('\n')
    code_lines = []
    for line in lines:
        if any(keyword in line for keyword in ['=', 'print', 'def', 'return', 'import']):
            code_lines.append(line)
    
    return '\n'.join(code_lines) if code_lines else None


def execute_code(code: str, timeout: int = 5) -> Optional[float]:
    """Safely execute Python code and extract numerical result"""
    try:
        local_vars = {}
        exec(code, {"__builtins__": __builtins__}, local_vars)
        
        for var_name in ['answer', 'result', 'total', 'output']:
            if var_name in local_vars:
                value = local_vars[var_name]
                if isinstance(value, (int, float)):
                    return float(value)
        
        for value in local_vars.values():
            if isinstance(value, (int, float)):
                return float(value)
        
        return None
    except Exception:
        return None


def majority_vote(answers: List[Optional[float]]) -> Optional[float]:
    """Get majority vote from list of answers, filtering out None values"""
    valid_answers = [a for a in answers if a is not None]
    if not valid_answers:
        return None
    
    counter = Counter(valid_answers)
    most_common = counter.most_common(1)[0][0]
    return most_common


def solve_via_cot(question: str, student_llm: ChatOpenAI, num_samples: int = 5) -> Tuple[Optional[float], Dict[str, Any], str]:
    """Student solves via Chain-of-Thought reasoning with self-consistency
    Returns: (answer, logging_data, reasoning_text)
    """
    answers = []
    reasonings = []
    log_data = {
        'samples': [],
        'final_answer': None,
        'all_samples': []
    }
    
    for sample_idx in range(num_samples):
        try:
            prompt = f"Q: {question}\nA: Let's think step by step."
            response = student_llm.invoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            answer = extract_answer_from_cot(response_text)
            answers.append(answer)
            reasonings.append(response_text)
            log_data['samples'].append({
                'sample': sample_idx + 1,
                'answer': answer,
                # Remove reasoning from logs - it's too verbose
                'status': 'success' if answer is not None else 'extraction_failed'
            })
        except Exception as e:
            answers.append(None)
            reasonings.append("")
            log_data['samples'].append({
                'sample': sample_idx + 1,
                'answer': None,
                'status': 'error',
                'error': str(e)
            })
    
    final_answer = majority_vote(answers)
    log_data['all_samples'] = answers
    log_data['final_answer'] = final_answer
    
    # Get reasoning from first successful sample that matches final answer
    final_reasoning = ""
    if final_answer is not None:
        for i, ans in enumerate(answers):
            if ans == final_answer:
                final_reasoning = reasonings[i]
                break
    
    return final_answer, log_data, final_reasoning


def solve_via_pot(question: str, student_llm: ChatOpenAI, num_samples: int = 5) -> Tuple[Optional[float], Dict[str, Any], str]:
    """Student solves via Program-of-Thought (code generation) with self-consistency
    Returns: (answer, logging_data, reasoning_text)
    Note: reasoning_text is empty for PoT as it's code-based
    """
    answers = []
    codes = []
    log_data = {
        'samples': [],
        'final_answer': None,
        'all_samples': []
    }
    
    for sample_idx in range(num_samples):
        try:
            prompt = POT_PROMPT_TEMPLATE.format(question=question)
            response = student_llm.invoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            code = extract_code_from_pot(response_text)
            if code:
                answer = execute_code(code)
                answers.append(answer)
                codes.append(code)
                log_data['samples'].append({
                    'sample': sample_idx + 1,
                    'answer': answer,
                    # Store code reference for debugging but don't show full code
                    'status': 'success' if answer is not None else 'execution_failed'
                })
            else:
                answers.append(None)
                codes.append("")
                log_data['samples'].append({
                    'sample': sample_idx + 1,
                    'answer': None,
                    'status': 'no_code_extracted'
                })
        except Exception as e:
            answers.append(None)
            codes.append("")
            log_data['samples'].append({
                'sample': sample_idx + 1,
                'answer': None,
                'status': 'error',
                'error': str(e)
            })
    
    final_answer = majority_vote(answers)
    log_data['all_samples'] = answers
    log_data['final_answer'] = final_answer
    
    # Get code from first successful sample that matches final answer
    final_code = ""
    if final_answer is not None:
        for i, ans in enumerate(answers):
            if ans == final_answer:
                final_code = codes[i]
                break
    
    # Return code as reasoning for PoT
    final_reasoning = f"```python\n{final_code}\n```" if final_code else ""
    
    return final_answer, log_data, final_reasoning

def oracle_solve_with_calculator(question: str, oracle_llm: ChatOpenAI, cot_answer: Optional[float], pot_answer: Optional[float], num_samples: int = 5) -> Tuple[Optional[float], Dict[str, Any], str]:
    """Oracle solves using 8-shot CoT + Self-Consistency + Calculator access
    Returns: (answer, logging_data, reasoning_text)
    """
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Oracle sample took too long (>120s)")
    
    answers = []
    reasonings = []
    log_data = {
        'samples': [],
        'final_answer': None,
        'all_samples': [],
        'student_answers': {'cot': cot_answer, 'pot': pot_answer}
    }
    
    for sample_idx in range(num_samples):
        try:
            prompt = f"""{COT_EXAMPLES}

You can use Python code as a calculator to verify your calculations if needed.
To use the calculator, write Python code in a code block like this:
```python
result = 5 * 3
```

Now solve this problem:

Q: {question}
A: Let's think step by step."""
            
            # Set timeout for LLM call
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(120)  # 120 second timeout
            
            try:
                response = oracle_llm.invoke(prompt)
            finally:
                signal.alarm(0)  # Cancel alarm
            response_text = response.content if hasattr(response, 'content') else str(response)
            reasonings.append(response_text)
            
            # First, check if there's any code to execute
            code = extract_code_from_pot(response_text)
            if code:
                code_answer = execute_code(code)
                if code_answer is not None:
                    answers.append(code_answer)
                    log_data['samples'].append({
                        'sample': sample_idx + 1,
                        'answer': code_answer,
                        'status': 'used_calculator'
                    })
                    continue
            
            # Otherwise, extract answer from reasoning
            answer = extract_answer_from_cot(response_text)
            answers.append(answer)
            log_data['samples'].append({
                'sample': sample_idx + 1,
                'answer': answer,
                'status': 'used_reasoning'
            })
            
        except TimeoutError as e:
            answers.append(None)
            reasonings.append("")
            log_data['samples'].append({
                'sample': sample_idx + 1,
                'answer': None,
                'status': 'timeout',
                'error': str(e)
            })
            print(f"⚠ Sample {sample_idx + 1} timed out: {e}")
        except Exception as e:
            answers.append(None)
            reasonings.append("")
            log_data['samples'].append({
                'sample': sample_idx + 1,
                'answer': None,
                'status': 'error',
                'error': str(e)
            })
            print(f"⚠ Sample {sample_idx + 1} error: {e}")
    
    final_answer = majority_vote(answers)
    log_data['all_samples'] = answers
    log_data['final_answer'] = final_answer
    
    # Get reasoning from first successful sample that matches final answer
    final_reasoning = ""
    if final_answer is not None:
        for i, ans in enumerate(answers):
            if ans == final_answer:
                final_reasoning = reasonings[i]
                break
    
    return final_answer, log_data, final_reasoning

def solve_single_problem(
    question: str,
    student_model: str = "meta-llama/llama-3.2-3b-instruct",
    oracle_model: str = "google/gemma-3-27b-it",
    num_samples: int = 5,
    oracle_samples: int = 5
) -> Dict[str, Any]:
    """Solve a single problem and return detailed results with full SC logs"""
    
    print(f"\n{'='*60}")
    print("Single Problem Solver")
    print(f"{'='*60}")
    print(f"Student Model: {student_model}")
    print(f"Oracle Model: {oracle_model}")
    print(f"Student SC Samples: {num_samples}")
    print(f"Oracle SC Samples: {oracle_samples}")
    print(f"{'='*60}\n")
    
    # Initialize LLMs
    student_llm = ChatOpenAI(
        model=student_model,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.7,
        max_tokens=512,
        request_timeout=90,
    )
    
    oracle_llm = ChatOpenAI(
        model=oracle_model,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.7,
        max_tokens=1024,
        request_timeout=120,
    )
    
    print("Solving via Chain-of-Thought...")
    cot_answer, cot_log, cot_reasoning = solve_via_cot(question, student_llm, num_samples)
    print(f"✓ CoT Answer: {cot_answer}")
    
    print("\nSolving via Program-of-Thought...")
    pot_answer, pot_log, pot_reasoning = solve_via_pot(question, student_llm, num_samples)
    print(f"✓ PoT Answer: {pot_answer}")
    
    # Check agreement
    final_answer = None
    method = None
    oracle_called = False
    oracle_log = None
    oracle_reasoning = ""
    
    if cot_answer is not None and pot_answer is not None:
        if abs(cot_answer - pot_answer) < 1e-6:
            final_answer = cot_answer
            method = "agreement"
            print(f"\n✓ Agreement! Final Answer: {final_answer}")
        else:
            print(f"\n⚠ Disagreement (CoT: {cot_answer}, PoT: {pot_answer})")
            print("Calling Oracle for tiebreaker...")
            final_answer, oracle_log, oracle_reasoning = oracle_solve_with_calculator(
                question, oracle_llm, cot_answer, pot_answer, oracle_samples
            )
            method = "oracle_tiebreaker"
            oracle_called = True
            print(f"✓ Oracle Answer: {final_answer}")
    elif cot_answer is not None:
        final_answer = cot_answer
        method = "cot_only"
        print(f"\n✓ CoT Only. Final Answer: {final_answer}")
    elif pot_answer is not None:
        final_answer = pot_answer
        method = "pot_only"
        print(f"\n✓ PoT Only. Final Answer: {final_answer}")
    else:
        print("\n⚠ Both methods failed. Calling Oracle...")
        final_answer, oracle_log, oracle_reasoning = oracle_solve_with_calculator(
            question, oracle_llm, None, None, oracle_samples
        )
        method = "oracle_fallback"
        oracle_called = True
        print(f"✓ Oracle Answer: {final_answer}")
    
    # Choose reasoning based on method
    if method == "agreement":
        reasoning = cot_reasoning
    elif method in ["oracle_tiebreaker", "oracle_fallback"]:
        reasoning = oracle_reasoning
    elif method == "cot_only":
        reasoning = cot_reasoning
    elif method == "pot_only":
        reasoning = pot_reasoning
    else:
        reasoning = ""
    
    # Build result with full SC logs
    result = {
        'question': question,
        'cot_answer': cot_answer,
        'pot_answer': pot_answer,
        'final_answer': final_answer,
        'method': method,
        'oracle_called': oracle_called,
        'reasoning': reasoning,
        'self_consistency_log': {
            'cot': cot_log,
            'pot': pot_log,
            'oracle': oracle_log
        }
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Single Problem Solver with Full Reasoning Output')
    parser.add_argument('--input', type=str, required=True, help='Input text file with single problem')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file')
    parser.add_argument('--student-model', type=str, default='meta-llama/llama-3.2-3b-instruct',
                       help='Student model (default: Llama 3.2 3B)')
    parser.add_argument('--oracle-model', type=str, default='google/gemma-3-27b-it',
                       help='Oracle model (default: Gemma 2 27B)')
    parser.add_argument('--num-samples', type=int, default=5, help='Student SC samples (default: 5)')
    parser.add_argument('--oracle-samples', type=int, default=5, help='Oracle SC samples (default: 5)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file {input_path} not found")
        return
    
    # Read problem from text file
    with open(input_path, 'r') as f:
        question = f.read().strip()
    
    if not question:
        print("Error: Input file is empty")
        return
    
    print(f"Problem: {question}\n")
    
    # Solve the problem
    result = solve_single_problem(
        question=question,
        student_model=args.student_model,
        oracle_model=args.oracle_model,
        num_samples=args.num_samples,
        oracle_samples=args.oracle_samples
    )
    
    # Save result
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✓ Results saved to: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
