#!/usr/bin/env python3
"""
Dual Solver + Oracle Pipeline with Reasoning Generation
Same as dual_solver_oracle_pipeline.py, but adds a reasoning generation step:
- When CoT and PoT agree → Use CoT reasoning directly
- When Oracle is called → Generate reasoning for Oracle's answer
This provides a 'reasoning' field for DeepEval evaluation
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

# DeepEval imports
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

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


# ============================================================
# Reasoning Generation Prompt
# ============================================================
REASONING_PROMPT_TEMPLATE = """You are given a math problem and its correct answer. Explain the solution step by step.

Problem: {question}

Answer: {answer}

Provide a clear, step-by-step explanation of how to solve this problem and arrive at the answer. Do not think if its correct or wrong, just reason the solution
step by step showing all calculations. And dont make comments if problem statement is wrong.

Your response:"""


def extract_answer_from_cot(response: str) -> Optional[float]:
    """Extract numerical answer from CoT reasoning"""
    # Look for "The answer is X" pattern
    match = re.search(r'(?:the answer is|answer:|answer =)\s*(-?\d+(?:\.\d+)?)', response, re.IGNORECASE)
    if match:
        return float(match.group(1))
    
    # Look for #### format
    match = re.search(r'####\s*(-?\d+(?:\.\d+)?)', response)
    if match:
        return float(match.group(1))
    
    # Try to find the last number in the response
    numbers = re.findall(r'-?\d+(?:\.\d+)?', response)
    if numbers:
        return float(numbers[-1])
    
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
    Extract ground truth answer from problem.
    NOTE: This is ONLY used for checking correctness after solving.
    The 'output' field is never shown to the model during solving.
    """
    for field in ['output', 'target', 'answer']:
        if field in problem:
            value = problem[field]
            if isinstance(value, (int, float)):
                return float(value)
            match = re.search(r'-?\d+(?:\.\d+)?', str(value))
            if match:
                return float(match.group(0))
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


def solve_via_cot(question: str, student_llm: ChatOpenAI, num_samples: int = 5) -> Tuple[Optional[float], str]:
    """Student solves via Chain-of-Thought reasoning with self-consistency. Returns (answer, reasoning)"""
    answers = []
    all_reasoning = []
    
    for _ in range(num_samples):
        try:
            prompt = f"{COT_EXAMPLES}\nQ: {question}\nA: Let's think step by step."
            response = student_llm.invoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            all_reasoning.append(response_text)
            answer = extract_answer_from_cot(response_text)
            answers.append(answer)
        except Exception:
            answers.append(None)
    
    # Return majority vote and first reasoning as representative
    return majority_vote(answers), all_reasoning[0] if all_reasoning else ""


def solve_via_pot(question: str, student_llm: ChatOpenAI, num_samples: int = 5) -> Optional[float]:
    """Student solves via Program-of-Thought (code generation) with self-consistency"""
    answers = []
    
    for _ in range(num_samples):
        try:
            prompt = POT_PROMPT_TEMPLATE.format(question=question)
            response = student_llm.invoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            code = extract_code_from_pot(response_text)
            if code:
                answer = execute_code(code)
                answers.append(answer)
            else:
                answers.append(None)
        except Exception:
            answers.append(None)
    
    return majority_vote(answers)


def oracle_solve_react(question: str, oracle_llm: ChatOpenAI, cot_answer: Optional[float], pot_answer: Optional[float]) -> Optional[float]:
    """Oracle solves using ReAct with code execution when student answers disagree"""
    try:
        # Simple ReAct prompt - not overloading
        prompt = f"""You are a math problem solver. Solve this problem step by step.

Problem: {question}

You can:
1. Think through the problem logically
2. Write Python code to verify your calculations

Think carefully and provide the final numerical answer.

Your response should end with: "Final Answer: <number>"
"""
        
        response = oracle_llm.invoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Try to extract from "Final Answer:" pattern
        match = re.search(r'Final Answer:\s*(-?\d+(?:\.\d+)?)', response_text, re.IGNORECASE)
        if match:
            return float(match.group(1))
        
        # Try other patterns
        return extract_answer_from_cot(response_text)
        
    except Exception:
        return None


def generate_reasoning(question: str, answer: float, student_llm: ChatOpenAI) -> str:
    """Generate step-by-step reasoning for the final answer using the student model"""
    try:
        prompt = REASONING_PROMPT_TEMPLATE.format(question=question, answer=answer)
        response = student_llm.invoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        return response_text.strip()
    except Exception as e:
        return f"Reasoning generation failed: {str(e)}"


def evaluate_single_problem(
    problem: Dict,
    student_llm: ChatOpenAI,
    oracle_llm: ChatOpenAI,
    num_samples: int = 5
) -> Dict[str, Any]:
    """Evaluate a single problem using dual solver + oracle with self-consistency and reasoning generation"""
    question = problem.get('input', problem.get('question', ''))
    ground_truth = extract_ground_truth(problem)
    problem_id = problem.get('id', 'unknown')
    
    # Step 1: Student solves via CoT with self-consistency (returns answer and reasoning)
    cot_answer, cot_reasoning = solve_via_cot(question, student_llm, num_samples)
    
    # Step 2: Student solves via PoT with self-consistency
    pot_answer = solve_via_pot(question, student_llm, num_samples)
    
    # Step 3: Check agreement
    final_answer = None
    final_reasoning = ""
    method = None
    oracle_called = False
    reasoning_source = None
    
    if cot_answer is not None and pot_answer is not None:
        if abs(cot_answer - pot_answer) < 1e-6:
            # Agreement - use CoT answer and reasoning
            final_answer = cot_answer
            final_reasoning = cot_reasoning
            method = "agreement"
            reasoning_source = "cot_direct"
        else:
            # Disagreement - call oracle
            final_answer = oracle_solve_react(question, oracle_llm, cot_answer, pot_answer)
            method = "oracle_tiebreaker"
            oracle_called = True
            
            # Generate reasoning for oracle's answer
            if final_answer is not None:
                final_reasoning = generate_reasoning(question, final_answer, student_llm)
                reasoning_source = "generated_for_oracle"
            else:
                final_reasoning = "Oracle failed to provide answer"
                reasoning_source = "none"
    elif cot_answer is not None:
        # Only CoT succeeded
        final_answer = cot_answer
        final_reasoning = cot_reasoning
        method = "cot_only"
        reasoning_source = "cot_direct"
    elif pot_answer is not None:
        # Only PoT succeeded - generate reasoning
        final_answer = pot_answer
        final_reasoning = generate_reasoning(question, pot_answer, student_llm)
        method = "pot_only"
        reasoning_source = "generated_for_pot"
    else:
        # Both failed - call oracle
        final_answer = oracle_solve_react(question, oracle_llm, None, None)
        method = "oracle_fallback"
        oracle_called = True
        
        # Generate reasoning for oracle's answer
        if final_answer is not None:
            final_reasoning = generate_reasoning(question, final_answer, student_llm)
            reasoning_source = "generated_for_oracle"
        else:
            final_reasoning = "Oracle failed to provide answer"
            reasoning_source = "none"
    
    # Check correctness
    correct = False
    if final_answer is not None and ground_truth is not None:
        correct = abs(final_answer - ground_truth) < 1e-6
    
    return {
        'problem_id': problem_id,
        'question': question,
        'ground_truth': ground_truth,
        'cot_answer': cot_answer,
        'pot_answer': pot_answer,
        'final_answer': final_answer,
        'final_reasoning': final_reasoning,
        'reasoning_source': reasoning_source,
        'method': method,
        'oracle_called': oracle_called,
        'correct': correct
    }


def run_pipeline(
    input_file: Path,
    output_file: Path,
    max_problems: Optional[int] = None,
    student_model: str = "meta-llama/llama-3.2-3b-instruct",
    oracle_model: str = "google/gemma-2-27b-it",
    workers: int = 10,
    num_samples: int = 5
):
    """Run the dual solver + oracle pipeline with self-consistency and reasoning generation"""
    print(f"\n{'='*60}")
    print("Dual Solver + Oracle Pipeline (with Reasoning Generation)")
    print(f"{'='*60}")
    print(f"Student Model: {student_model}")
    print(f"Oracle Model: {oracle_model}")
    print(f"Self-Consistency Samples: {num_samples}")
    print(f"Workers: {workers}")
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
    completed = 0
    last_progress = time.time()
    
    # Initialize output file with empty structure
    initial_data = {
        'config': {
            'student_model': student_model,
            'oracle_model': oracle_model,
            'num_samples': num_samples,
            'strategy': 'Dual Solver (CoT + PoT) with Self-Consistency, Oracle Tiebreaker, and Reasoning Generation'
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
        
        # Count methods and reasoning sources
        method_counts = {}
        reasoning_sources = {}
        for r in results:
            method = r.get('method', 'unknown')
            method_counts[method] = method_counts.get(method, 0) + 1
            
            rsrc = r.get('reasoning_source', 'unknown')
            reasoning_sources[rsrc] = reasoning_sources.get(rsrc, 0) + 1
        
        output_data = {
            'config': {
                'student_model': student_model,
                'oracle_model': oracle_model,
                'num_samples': num_samples,
                'strategy': 'Dual Solver (CoT + PoT) with Self-Consistency, Oracle Tiebreaker, and Reasoning Generation'
            },
            'summary': {
                'total': total,
                'correct': correct,
                'accuracy': accuracy,
                'agreements': agreements,
                'agreement_rate': agreement_rate,
                'oracle_calls': oracle_calls,
                'oracle_rate': oracle_rate,
                'method_breakdown': method_counts,
                'reasoning_source_breakdown': reasoning_sources
            },
            'results': results
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_problem = {
            executor.submit(evaluate_single_problem, p, student_llm, oracle_llm, num_samples): p
            for p in problems
        }
        
        pending = set(future_to_problem.keys())
        
        with tqdm(total=len(problems), desc="Evaluating") as pbar:
            while pending and completed < len(problems):
                current = time.time()
                
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
                        result = future.result(timeout=1)
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
                        
                        # Save incrementally every 10 problems
                        if completed % 10 == 0:
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
                        results.append({
                            'problem_id': problem.get('id', 'unknown'),
                            'question': problem.get('input', ''),
                            'ground_truth': extract_ground_truth(problem),
                            'cot_answer': None,
                            'pot_answer': None,
                            'final_answer': None,
                            'final_reasoning': "",
                            'reasoning_source': 'error',
                            'method': 'error',
                            'oracle_called': False,
                            'correct': False,
                            'error': str(e)
                        })
                        completed += 1
                        pbar.update(1)
                        
                        # Save even on errors
                        if completed % 10 == 0:
                            save_incremental_results()
                
                if not done:
                    time.sleep(0.5)
    
    # Final save
    save_incremental_results()
    
    print(f"\n✓ Completed {len(results)}/{len(problems)} problems")
    
    # DeepEval Benchmarking - Evaluate reasoning quality
    print(f"\n{'='*60}")
    print("Running DeepEval Reasoning Evaluation...")
    print(f"{'='*60}\n")
    
    test_cases = []
    for result in results:
        if result['final_reasoning'] and result['final_answer'] is not None:
            test_case = LLMTestCase(
                input=result['question'],
                actual_output=result['final_reasoning'],  # Evaluate the reasoning, not just the answer
                expected_output=f"The answer is {result['ground_truth']}"
            )
            test_cases.append(test_case)
    
    print(f"Created {len(test_cases)} test cases for DeepEval\n")
    
    deepeval_metrics = {}
    
    if test_cases:
        try:
            from concurrent.futures import TimeoutError as FuturesTimeoutError
            
            print("Evaluating Answer Relevancy (with timeout protection)...")
            
            # Create metric with fast model and no async
            answer_relevancy_metric = AnswerRelevancyMetric(
                threshold=0.7,
                model="gpt-3.5-turbo",  # Faster model for evaluation
                async_mode=False,        # Disable async to ensure synchronous execution
                include_reason=False     # Skip reason to speed up
            )
            
            answer_scores = []
            successful = 0
            failed = 0
            
            # Process with thread pool for timeout handling
            with ThreadPoolExecutor(max_workers=1) as eval_executor:
                for i, test_case in enumerate(tqdm(test_cases, desc="DeepEval")):
                    try:
                        # Submit with 30 second timeout per test case
                        def evaluate_test_case(tc, metric):
                            metric.measure(tc)
                            return metric.score
                        
                        future = eval_executor.submit(evaluate_test_case, test_case, answer_relevancy_metric)
                        score = future.result(timeout=30)
                        answer_scores.append(float(score))
                        successful += 1
                        
                    except FuturesTimeoutError:
                        if i % 5 == 0:
                            print(f"\n⏱️ Timeout on test case {i}, using neutral score (0.5)")
                        answer_scores.append(0.5)
                        failed += 1
                    except Exception as e:
                        error_msg = str(e)[:80]
                        if i % 5 == 0:
                            print(f"\n⚠️ Error on test case {i}: {error_msg}")
                        answer_scores.append(0.5)
                        failed += 1
            
            avg_answer_relevancy = sum(answer_scores) / len(answer_scores) if answer_scores else 0
            
            print(f"\n✓ Answer Relevancy Evaluation Complete")
            print(f"  Successful: {successful}/{len(test_cases)}")
            print(f"  Failed/Timeout: {failed}/{len(test_cases)}")
            print(f"  Average Score: {avg_answer_relevancy:.4f}")
            
            deepeval_metrics = {
                'answer_relevancy': {
                    'average_score': float(avg_answer_relevancy),
                    'successful_evaluations': successful,
                    'failed_evaluations': failed,
                    'total_evaluations': len(test_cases),
                    'individual_scores': [float(s) for s in answer_scores]
                }
            }
            
        except Exception as e:
            print(f"⚠️ DeepEval evaluation encountered an error: {str(e)[:200]}")
            print("Continuing without DeepEval metrics...")
    else:
        print("⚠️ No test cases to evaluate")
    
    # Calculate final statistics
    total = len(results)
    accuracy = (100 * correct / total) if total > 0 else 0
    agreement_rate = (100 * agreements / total) if total > 0 else 0
    oracle_rate = (100 * oracle_calls / total) if total > 0 else 0
    
    # Method breakdown
    method_counts = {}
    reasoning_sources = {}
    for r in results:
        method = r.get('method', 'unknown')
        method_counts[method] = method_counts.get(method, 0) + 1
        
        rsrc = r.get('reasoning_source', 'unknown')
        reasoning_sources[rsrc] = reasoning_sources.get(rsrc, 0) + 1
    
    # Save final results with DeepEval metrics
    output_data = {
        'config': {
            'student_model': student_model,
            'oracle_model': oracle_model,
            'num_samples': num_samples,
            'strategy': 'Dual Solver (CoT + PoT) with Self-Consistency, Oracle Tiebreaker, and Reasoning Generation'
        },
        'summary': {
            'total': total,
            'correct': correct,
            'accuracy': accuracy,
            'agreements': agreements,
            'agreement_rate': agreement_rate,
            'oracle_calls': oracle_calls,
            'oracle_rate': oracle_rate,
            'method_breakdown': method_counts,
            'reasoning_source_breakdown': reasoning_sources
        },
        'deepeval_metrics': deepeval_metrics,
        'results': results
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✓ Final results with DeepEval metrics saved to {output_file}\n")
    
    # Print summary
    print(f"\n{'='*60}")
    print("PIPELINE RESULTS")
    print(f"{'='*60}")
    print(f"Total problems:     {total}")
    print(f"Correct:            {correct}")
    print(f"Accuracy:           {accuracy:.2f}%")
    print(f"\nStudent Agreement:  {agreements} ({agreement_rate:.1f}%)")
    print(f"Oracle Called:      {oracle_calls} ({oracle_rate:.1f}%)")
    
    if deepeval_metrics and 'answer_relevancy' in deepeval_metrics:
        ar = deepeval_metrics['answer_relevancy']
        print(f"\nDeepEval Reasoning Quality:")
        print(f"  Answer Relevancy: {ar['average_score']:.4f}")
        print(f"  Evaluations:      {ar['successful_evaluations']}/{ar['total_evaluations']}")
    
    print(f"\nMethod Breakdown:")
    for method, count in sorted(method_counts.items()):
        pct = 100 * count / total
        print(f"  {method}: {count} ({pct:.1f}%)")
    print(f"\nReasoning Source Breakdown:")
    for rsrc, count in sorted(reasoning_sources.items()):
        pct = 100 * count / total
        print(f"  {rsrc}: {count} ({pct:.1f}%)")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Dual Solver + Oracle Pipeline with Reasoning Generation')
    parser.add_argument('--input', type=str, required=True, help='Input JSONL file')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file')
    parser.add_argument('--student-model', type=str, default='meta-llama/llama-3.2-3b-instruct',
                       help='Student model (default: Llama 3.2 3B)')
    parser.add_argument('--oracle-model', type=str, default='google/gemma-2-27b-it',
                       help='Oracle model (default: Gemma 2 27B)')
    parser.add_argument('--max-problems', type=int, help='Max problems to evaluate')
    parser.add_argument('--workers', type=int, default=10, help='Parallel workers (default: 10)')
    parser.add_argument('--num-samples', type=int, default=5, help='Self-consistency samples (default: 5)')
    
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
        num_samples=args.num_samples
    )


if __name__ == "__main__":
    main()
