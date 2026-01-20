#!/usr/bin/env python3
"""
Dual Solver + Enhanced Oracle Pipeline with Self-Consistency
Student (Llama 3.2 3B) solves via both CoT and PoT with self-consistency (5 samples each)
If they agree → final answer
If they disagree → Oracle (Gemma 3 27B) with 8-shot CoT + Self-Consistency + Calculator
"""

import json
import re
import argparse
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from collections import Counter

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


# ============================================================
# 8-Shot CoT Examples (for reasoning-based solving)
# ============================================================
COT_EXAMPLES = """
Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: Let's think step by step. There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: Let's think step by step. There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Let's think step by step. Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Let's think step by step. Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.

Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: Let's think step by step. Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.

Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
A: Let's think step by step. There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 = 29. The answer is 29.

Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
A: Let's think step by step. Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.

Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: Let's think step by step. Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 = 8. The answer is 8.
"""


# ============================================================
# Simple PoT Prompt (no few-shot, direct instruction)
# ============================================================
POT_PROMPT_TEMPLATE = """Solve this math problem by writing Python code.

Problem: {question}

Write Python code to solve this problem. Your code should:
1. Define variables for the given values
2. Perform the necessary calculations
3. Store the final answer in a variable called 'answer'

Only provide the Python code, nothing else."""


def extract_answer_from_cot(response: str) -> Optional[float]:
    """Extract numerical answer from CoT reasoning - robust extraction"""
    if not response or not isinstance(response, str):
        return None
    
    response_lower = response.lower()
    
    # 1) Try explicit "answer is" patterns (most reliable)
    patterns = [
        r'(?:the\s+)?answer\s+(?:is|=)\s*:?\s*\$?\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)',
        r'the\s+answer\s+is\s+(-?\d+(?:,\d{3})*(?:\.\d+)?)',
        r'answer\s*:?\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)',
        r'####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)',
        r'(?:therefore|thus|so),?\s+(?:the\s+)?(?:answer|result)\s+(?:is|=)\s*:?\s*\$?\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)',
        r'(?:result|total|value)\s+(?:is|=)\s*:?\s*\$?\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response_lower)
        if match:
            try:
                value_str = match.group(1).replace(',', '')
                return float(value_str)
            except (ValueError, AttributeError):
                continue
    
    # 2) Last resort: extract all numbers and return the last one
    numbers = re.findall(r'-?\d+(?:\.\d+)?', response)
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            pass
    
    return None


def extract_code_from_pot(response: str) -> Optional[str]:
    """Extract Python code from PoT response - improved extraction"""
    # First try to extract from markdown code blocks
    code_block_match = re.search(r'```(?:python)?\s*\n(.*?)\n```', response, re.DOTALL)
    if code_block_match:
        code = code_block_match.group(1).strip()
        if code and '=' in code:
            # Ensure 'answer' variable exists
            if 'answer' not in code.lower():
                var_matches = re.findall(r'(\w+)\s*=', code)
                if var_matches:
                    code += f"\nanswer = {var_matches[-1]}"
            return code
    
    # Remove markdown artifacts
    response = re.sub(r'```python\s*', '', response)
    response = re.sub(r'```\s*', '', response)
    
    # Try to extract lines with Python code
    lines = response.strip().split('\n')
    code_lines = []
    in_code_section = False
    
    for line in lines:
        stripped = line.strip()
        
        # Skip empty lines at start
        if not code_lines and not stripped:
            continue
            
        # Check if this looks like code
        has_assignment = '=' in stripped and not stripped.startswith('=')
        has_operators = any(op in stripped for op in ['+', '-', '*', '/', '(', ')'])
        is_comment = stripped.startswith('#')
        
        if has_assignment or (in_code_section and (has_operators or is_comment)):
            code_lines.append(line)
            in_code_section = True
        elif in_code_section and not stripped:
            # Empty line within code section
            continue
        elif in_code_section and stripped and len(stripped.split()) > 8:
            # Long text line, probably explanation - stop
            break
    
    if not code_lines:
        return None
    
    code = '\n'.join(code_lines).strip()
    
    # Ensure 'answer' variable exists
    if code and 'answer' not in code.lower():
        var_matches = re.findall(r'(\w+)\s*=', code)
        if var_matches:
            last_var = var_matches[-1]
            # Don't add if it's a loop variable or common temp var
            if last_var not in ['i', 'j', 'k', 'temp', 'tmp']:
                code += f"\nanswer = {last_var}"
    
    return code if code else None


def execute_code(code: str) -> Optional[float]:
    """Execute Python code and return answer"""
    try:
        namespace = {}
        exec(code, namespace)
        
        for var in ['answer', 'Answer', 'result', 'Result']:
            if var in namespace:
                return float(namespace[var])
        
        return None
    except Exception:
        return None


def extract_ground_truth(problem: Dict[str, Any]) -> Optional[float]:
    """
    Extract ground truth answer from problem. Handles LaTeX, fractions, and numeric answers.
    NOTE: This is ONLY used for checking correctness after solving.
    The 'output' field is never shown to the model during solving.
    """
    for field in ['answer', 'output', 'target']:  # Try 'answer' first (MATH dataset)
        if field not in problem:
            continue
            
        value = problem[field]
        text = str(value).strip()
        
        # 1) Already numeric
        if isinstance(value, (int, float)):
            return float(value)
        
        # 2) Handle LaTeX fractions: \frac{a}{b}
        frac_match = re.search(r'\\frac\{([^}]+)\}\{([^}]+)\}', text)
        if frac_match:
            try:
                num_str = frac_match.group(1).strip()
                den_str = frac_match.group(2).strip()
                num = float(num_str)
                den = float(den_str)
                if den != 0:
                    return num / den
            except (ValueError, ZeroDivisionError):
                pass
        
        # 3) Handle LaTeX expressions: \left(...\right), \boxed{...}, etc.
        cleaned = text
        cleaned = re.sub(r'\\boxed\{([^}]*)\}', r'\1', cleaned)  # \boxed{x} -> x
        cleaned = re.sub(r'\\left\(', '(', cleaned)
        cleaned = re.sub(r'\\right\)', ')', cleaned)
        cleaned = re.sub(r'\\text\{([^}]*)\}', r'\1', cleaned)  # \text{Evelyn} -> Evelyn
        cleaned = re.sub(r'\$', '', cleaned)  # Remove $ delimiters
        cleaned = re.sub(r'\\', '', cleaned)  # Remove remaining backslashes
        
        # 4) Try extraction from cleaned text using CoT patterns
        answer = extract_answer_from_cot(cleaned)
        if answer is not None:
            return answer
        
        # 5) Try last numeric token (handle commas, decimals, negatives)
        numbers = re.findall(r'-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?', cleaned)
        if numbers:
            try:
                # Return the last number found
                last_num = numbers[-1].replace(',', '')
                return float(last_num)
            except ValueError:
                pass
    
    return None


def majority_vote(answers: List[Optional[float]]) -> Optional[float]:
    """Get majority vote from list of answers, filtering out None values"""
    valid_answers = [a for a in answers if a is not None]
    if not valid_answers:
        return None
    
    # Count occurrences
    counter = Counter(valid_answers)
    
    # Get most common answer
    most_common = counter.most_common(1)[0][0]
    return most_common


def solve_via_cot(question: str, student_llm: ChatOpenAI, num_samples: int = 5) -> Tuple[Optional[float], Dict[str, Any], str]:
    """Student solves via Chain-of-Thought reasoning with self-consistency (no examples)
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
                'status': 'success'
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
                log_data['samples'].append({
                    'sample': sample_idx + 1,
                    'answer': answer,
                    'status': 'code_executed'
                })
            else:
                answers.append(None)
                log_data['samples'].append({
                    'sample': sample_idx + 1,
                    'answer': None,
                    'status': 'could_not_extract_code'
                })
        except Exception as e:
            answers.append(None)
            log_data['samples'].append({
                'sample': sample_idx + 1,
                'answer': None,
                'status': 'error',
                'error': str(e)
            })
    
    final_answer = majority_vote(answers)
    log_data['all_samples'] = answers
    log_data['final_answer'] = final_answer
    
    # PoT is code-based, no reasoning text to return
    final_reasoning = ""
    
    return final_answer, log_data, final_reasoning


def oracle_solve_with_calculator(question: str, oracle_llm: ChatOpenAI, cot_answer: Optional[float], pot_answer: Optional[float], num_samples: int = 5) -> Tuple[Optional[float], Dict[str, Any], str]:
    """Oracle solves using 8-shot CoT + Self-Consistency + Calculator access
    Returns: (answer, logging_data, reasoning_text)
    """
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
            # Enhanced prompt with 8-shot examples and calculator instruction
            prompt = f"""{COT_EXAMPLES}

You can use Python code as a calculator to verify your calculations if needed.
To use the calculator, write Python code in a code block like this:
```python
result = 5 * 3
```

Now solve this problem:

Q: {question}
A: Let's think step by step."""
            
            response = oracle_llm.invoke(prompt)
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


def evaluate_single_problem(
    problem: Dict,
    student_llm: ChatOpenAI,
    oracle_llm: ChatOpenAI,
    num_samples: int = 5,
    oracle_samples: int = 5,
    enable_sc_logging: bool = False
) -> Dict[str, Any]:
    """Evaluate a single problem using dual solver + enhanced oracle with self-consistency"""
    # Try multiple field names for problem text
    question = problem.get('problem', problem.get('input', problem.get('question', '')))
    ground_truth = extract_ground_truth(problem)
    problem_id = problem.get('id', problem.get('unique_id', 'unknown'))
    
    # Step 1: Student solves via CoT with self-consistency
    cot_answer, cot_log, cot_reasoning = solve_via_cot(question, student_llm, num_samples)
    
    # Step 2: Student solves via PoT with self-consistency
    pot_answer, pot_log, pot_reasoning = solve_via_pot(question, student_llm, num_samples)
    
    # Step 3: Check agreement
    final_answer = None
    method = None
    oracle_called = False
    oracle_log = None
    oracle_reasoning = ""
    
    if cot_answer is not None and pot_answer is not None:
        if abs(cot_answer - pot_answer) < 1e-6:
            # Agreement - use this answer
            final_answer = cot_answer
            method = "agreement"
        else:
            # Disagreement - call oracle with enhanced capabilities
            final_answer, oracle_log, oracle_reasoning = oracle_solve_with_calculator(question, oracle_llm, cot_answer, pot_answer, oracle_samples)
            method = "oracle_tiebreaker"
            oracle_called = True
    elif cot_answer is not None:
        # Only CoT succeeded
        final_answer = cot_answer
        method = "cot_only"
    elif pot_answer is not None:
        # Only PoT succeeded
        final_answer = pot_answer
        method = "pot_only"
    else:
        # Both failed - call oracle
        final_answer, oracle_log, oracle_reasoning = oracle_solve_with_calculator(question, oracle_llm, None, None, oracle_samples)
        method = "oracle_fallback"
        oracle_called = True
    
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
    
    # Check correctness
    correct = False
    if final_answer is not None and ground_truth is not None:
        correct = abs(final_answer - ground_truth) < 1e-6
    
    # Build result - conditionally include SC logs
    result = {
        'problem_id': problem_id,
        'question': question,
        'ground_truth': ground_truth,
        'cot_answer': cot_answer,
        'pot_answer': pot_answer,
        'final_answer': final_answer,
        'method': method,
        'oracle_called': oracle_called,
        'correct': correct,
        'reasoning': reasoning
    }
    
    # Add self-consistency logs only if enabled
    if enable_sc_logging:
        result['self_consistency_log'] = {
            'cot': cot_log,
            'pot': pot_log,
            'oracle': oracle_log
        }
    
    return result


def run_pipeline(
    input_file: Path,
    output_file: Path,
    max_problems: Optional[int] = None,
    student_model: str = "meta-llama/llama-3.2-3b-instruct",
    oracle_model: str = "google/gemma-3-27b-it",
    workers: int = 10,
    num_samples: int = 5,
    oracle_samples: int = 5,
    enable_sc_logging: bool = False
):
    """Run the dual solver + enhanced oracle pipeline with self-consistency"""
    print(f"\n{'='*60}")
    print("Dual Solver + Enhanced Oracle Pipeline")
    print("(CoT + PoT with Self-Consistency)")
    print("(Oracle: 8-shot CoT + Self-Consistency + Calculator)")
    print(f"{'='*60}")
    print(f"Student Model: {student_model}")
    print(f"Oracle Model: {oracle_model}")
    print(f"Student Self-Consistency Samples: {num_samples}")
    print(f"Oracle Self-Consistency Samples: {oracle_samples}")
    print(f"Workers: {workers}")
    print(f"SC Logging: {'Enabled' if enable_sc_logging else 'Disabled'}")
    print(f"{'='*60}\n")
    
    # Load problems
    print(f"Loading problems from {input_file}...")
    problems = []
    with open(input_file) as f:
        for line in f:
            if line.strip():
                problems.append(json.loads(line))
    
    if max_problems:
        problems = problems[:max_problems]
    
    print(f"✓ Loaded {len(problems)} problems\n")
    
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
    
    # Process problems with incremental saving
    results = []
    correct = 0
    oracle_calls = 0
    agreements = 0
    timeouts = 0
    completed = 0
    last_progress = time.time()
    
    # Initialize output file with empty structure
    initial_data = {
        'config': {
            'student_model': student_model,
            'oracle_model': oracle_model,
            'num_samples': num_samples,
            'oracle_samples': oracle_samples,
            'strategy': 'Dual Solver (CoT + PoT) with Enhanced Oracle (8-shot CoT + SC + Calculator)'
        },
        'summary': {},
        'results': []
    }
    with open(output_file, 'w') as f:
        json.dump(initial_data, f, indent=2)
    
    print("Starting evaluation...\n")
    print(f"💾 Results will be saved incrementally to: {output_file}\n")
    
    def save_incremental_results():
        """Save current results to file"""
        total = len(results)
        accuracy = (100 * correct / total) if total > 0 else 0
        agreement_rate = (100 * agreements / total) if total > 0 else 0
        oracle_rate = (100 * oracle_calls / total) if total > 0 else 0
        timeout_rate = (100 * timeouts / total) if total > 0 else 0
        
        # Count methods
        method_counts = {}
        for r in results:
            method = r.get('method', 'unknown')
            method_counts[method] = method_counts.get(method, 0) + 1
        
        output_data = {
            'config': {
                'student_model': student_model,
                'oracle_model': oracle_model,
                'num_samples': num_samples,
                'oracle_samples': oracle_samples,
                'strategy': 'Dual Solver (CoT + PoT) with Enhanced Oracle (8-shot CoT + SC + Calculator)'
            },
            'summary': {
                'total': total,
                'correct': correct,
                'accuracy': accuracy,
                'agreements': agreements,
                'agreement_rate': agreement_rate,
                'oracle_calls': oracle_calls,
                'oracle_rate': oracle_rate,
                'timeouts': timeouts,
                'timeout_rate': timeout_rate,
                'method_breakdown': method_counts
            },
            'results': results
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_problem = {
            executor.submit(evaluate_single_problem, p, student_llm, oracle_llm, num_samples, oracle_samples, enable_sc_logging): p
            for p in problems
        }
        
        pending = set(future_to_problem.keys())
        
        with tqdm(total=len(problems), desc="Evaluating") as pbar:
            while pending and completed < len(problems):
                current = time.time()
                
                # Timeout check: 300 seconds without progress
                if current - last_progress > 300:
                    print(f"\n⚠️  No progress for 300s. Saving current results and cancelling {len(pending)} remaining...")
                    save_incremental_results()
                    for f in pending:
                        f.cancel()
                    break
                
                done = [f for f in list(pending) if f.done()]
                
                for future in done:
                    pending.remove(future)
                    
                    try:
                        result = future.result(timeout=180)
                        results.append(result)
                        
                        if result['correct']:
                            correct += 1
                        if result['oracle_called']:
                            oracle_calls += 1
                        if result['method'] == 'agreement':
                            agreements += 1
                        
                        completed += 1
                        last_progress = current
                        pbar.update(1)
                        
                        # Save incrementally every 5 problems
                        if completed % 5 == 0:
                            save_incremental_results()
                        
                        if completed % 50 == 0:
                            acc = 100 * correct / completed
                            agreement_rate = 100 * agreements / completed
                            oracle_rate = 100 * oracle_calls / completed
                            print(f"\n[Progress] {completed}/{len(problems)}")
                            print(f"  Accuracy: {acc:.2f}%")
                            print(f"  Agreement Rate: {agreement_rate:.1f}%")
                            print(f"  Oracle Called: {oracle_rate:.1f}%")
                    
                    except Exception as e:
                        problem = future_to_problem[future]
                        error_str = str(e)
                        is_timeout = 'timeout' in error_str.lower()
                        
                        results.append({
                            'problem_id': problem.get('id', 'unknown'),
                            'question': problem.get('input', ''),
                            'ground_truth': extract_ground_truth(problem),
                            'cot_answer': None,
                            'pot_answer': None,
                            'final_answer': None,
                            'method': 'timeout' if is_timeout else 'error',
                            'oracle_called': False,
                            'correct': False,
                            'error': error_str
                        })
                        
                        if is_timeout:
                            timeouts += 1
                        
                        completed += 1
                        pbar.update(1)
                        
                        # Save even on errors
                        if completed % 5 == 0:
                            save_incremental_results()
                
                if not done:
                    time.sleep(0.5)
    
    # Final save
    save_incremental_results()
    
    print(f"\n✓ Completed {len(results)}/{len(problems)} problems")
    
    # Calculate final statistics
    total = len(results)
    accuracy = (100 * correct / total) if total > 0 else 0
    agreement_rate = (100 * agreements / total) if total > 0 else 0
    oracle_rate = (100 * oracle_calls / total) if total > 0 else 0
    timeout_rate = (100 * timeouts / total) if total > 0 else 0
    
    # Method breakdown
    method_counts = {}
    for r in results:
        method = r.get('method', 'unknown')
        method_counts[method] = method_counts.get(method, 0) + 1
    
    # Print summary
    print(f"\n{'='*60}")
    print("PIPELINE RESULTS")
    print(f"{'='*60}")
    print(f"Total problems:     {total}")
    print(f"Correct:            {correct}")
    print(f"Accuracy:           {accuracy:.2f}%")
    print(f"\nStudent Agreement:  {agreements} ({agreement_rate:.1f}%)")
    print(f"Oracle Called:      {oracle_calls} ({oracle_rate:.1f}%)")
    print(f"Timeouts:           {timeouts} ({timeout_rate:.1f}%)")
    print(f"\nMethod Breakdown:")
    for method, count in sorted(method_counts.items()):
        pct = 100 * count / total
        print(f"  {method}: {count} ({pct:.1f}%)")
    print(f"{'='*60}")
    print(f"\n✓ Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Dual Solver + Enhanced Oracle Pipeline')
    parser.add_argument('--input', type=str, required=True, help='Input JSONL file')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file')
    parser.add_argument('--student-model', type=str, default='meta-llama/llama-3.2-3b-instruct',
                       help='Student model (default: Llama 3.2 3B)')
    parser.add_argument('--oracle-model', type=str, default='google/gemma-3-27b-it',
                       help='Oracle model (default: Gemma 3 27B)')
    parser.add_argument('--max-problems', type=int, help='Max problems to evaluate')
    parser.add_argument('--workers', type=int, default=10, help='Parallel workers (default: 10)')
    parser.add_argument('--num-samples', type=int, default=5, help='Student self-consistency samples (default: 5)')
    parser.add_argument('--oracle-samples', type=int, default=5, help='Oracle self-consistency samples (default: 5)')
    parser.add_argument('--enable-sc-logging', action='store_true', 
                       help='Enable detailed self-consistency logging in output (default: disabled)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file {input_path} not found")
        return
    
    run_pipeline(
        input_file=input_path,
        output_file=Path(args.output),
        max_problems=args.max_problems,
        student_model=args.student_model,
        oracle_model=args.oracle_model,
        workers=args.workers,
        num_samples=args.num_samples,
        oracle_samples=args.oracle_samples,
        enable_sc_logging=args.enable_sc_logging
    )


if __name__ == "__main__":
    main()
