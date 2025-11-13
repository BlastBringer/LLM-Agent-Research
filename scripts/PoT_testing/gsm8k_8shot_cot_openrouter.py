#!/usr/bin/env python3
"""
GSM8K 8-Shot Chain-of-Thought Evaluation - Llama 3.2 3B via OpenRouter
Replicating the standard GSM8K evaluation with self-consistency voting
"""

import json
import re
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from tqdm import tqdm
import time

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


# ============================================================
# 8-Shot Chain-of-Thought Examples (Standard GSM8K)
# ============================================================
FEWSHOT_EXAMPLES = [
    (
        "Roger has 5 tennis balls. He buys 2 more cans of tennis balls. "
        "Each can has 3 balls. How many tennis balls does he have now?",
        "Roger starts with 5 balls. Each can has 3 balls, so 2 cans have 2×3=6 balls. "
        "Total = 5 + 6 = 11.\n#### 11"
    ),
    (
        "There are 15 trees in the grove. Each tree has 12 apples. "
        "How many apples are there in total?",
        "Each tree has 12 apples and there are 15 trees, so total = 15×12=180.\n#### 180"
    ),
    (
        "Leah had 32 chocolates and ate 8. Then she bought 10 more. "
        "How many chocolates does she have now?",
        "Start with 32, ate 8 → 24 left. Buys 10 more → 24 + 10 = 34.\n#### 34"
    ),
    (
        "A train travels 60 miles per hour for 3 hours. "
        "How far does it travel?",
        "Distance = speed × time = 60×3 = 180 miles.\n#### 180"
    ),
    (
        "A box contains 12 pencils. How many pencils are there in 5 such boxes?",
        "Each box has 12, so 5×12=60 pencils.\n#### 60"
    ),
    (
        "A car goes 40 miles in 1 hour. How far will it go in 4 hours?",
        "Distance = 40×4 = 160 miles.\n#### 160"
    ),
    (
        "If 5 books cost $20, how much do 8 books cost?",
        "Each book costs 20/5=4. For 8 books: 8×4=32.\n#### 32"
    ),
    (
        "A rectangle has length 8 cm and width 3 cm. Find its perimeter.",
        "Perimeter = 2×(8+3) = 22 cm.\n#### 22"
    ),
]


def build_8shot_cot_prompt(problem_text: str) -> str:
    """Build the 8-shot Chain-of-Thought prompt"""
    prompt = ""
    for question, answer in FEWSHOT_EXAMPLES:
        prompt += f"Q: {question}\nA: Let's think step by step.\n{answer}\n\n"
    prompt += f"Q: {problem_text}\nA: Let's think step by step.\n"
    return prompt


def extract_numeric_answer(text: str) -> Optional[float]:
    """Extract numeric answer from model response"""
    # Look for #### format first (standard GSM8K format)
    match = re.search(r'####\s*([-+]?\d*\.?\d+)', text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    
    # Fallback: extract last number in the text
    matches = re.findall(r'[-+]?\d*\.?\d+', text)
    if matches:
        try:
            return float(matches[-1])
        except ValueError:
            pass
    
    return None


def extract_ground_truth(problem_data: Dict[str, Any]) -> Optional[float]:
    """Extract ground truth answer from problem data"""
    for field in ['output', 'answer', 'ground_truth']:
        if field in problem_data:
            try:
                value = str(problem_data[field])
                match = re.search(r'[-+]?\d*\.?\d+', value)
                if match:
                    return float(match.group(0))
            except (ValueError, TypeError):
                continue
    return None


def generate_with_self_consistency(
    llm: ChatOpenAI,
    prompt: str,
    num_samples: int = 5,
    temperature: float = 0.7
) -> List[str]:
    """Generate multiple responses for self-consistency voting"""
    responses = []
    
    for _ in range(num_samples):
        try:
            response = llm.invoke(prompt)
            responses.append(response.content)
        except Exception as e:
            print(f"\n⚠️  Generation error: {e}")
            continue
    
    return responses


def majority_vote(predictions: List[float]) -> Optional[float]:
    """Return the most common prediction (majority vote)"""
    if not predictions:
        return None
    
    # Count occurrences
    from collections import Counter
    counts = Counter(predictions)
    
    # Return most common
    most_common = counts.most_common(1)[0][0]
    return most_common


def evaluate_single_problem(
    problem_data: Dict[str, Any],
    llm: ChatOpenAI,
    use_self_consistency: bool = True,
    num_samples: int = 5
) -> Dict[str, Any]:
    """Evaluate a single problem with 8-shot CoT"""
    problem_id = problem_data.get('id', 'unknown')
    problem_text = problem_data.get('input', problem_data.get('question', ''))
    ground_truth = extract_ground_truth(problem_data)
    
    if not problem_text:
        return {
            'problem_id': problem_id,
            'question': problem_text,
            'ground_truth': ground_truth,
            'predicted': None,
            'correct': False,
            'error': 'No problem text found'
        }
    
    # Build 8-shot CoT prompt
    prompt = build_8shot_cot_prompt(problem_text)
    
    try:
        if use_self_consistency:
            # Generate multiple samples and vote
            responses = generate_with_self_consistency(
                llm, prompt, num_samples=num_samples
            )
            
            # Extract predictions from all responses
            predictions = [
                extract_numeric_answer(resp) 
                for resp in responses 
                if extract_numeric_answer(resp) is not None
            ]
            
            # Majority vote
            predicted = majority_vote(predictions)
            raw_responses = responses
        else:
            # Single sample (greedy decoding)
            response = llm.invoke(prompt)
            predicted = extract_numeric_answer(response.content)
            raw_responses = [response.content]
        
        # Check correctness with 1% tolerance
        correct = False
        if predicted is not None and ground_truth is not None:
            tolerance = max(abs(ground_truth * 0.01), 0.01)
            correct = abs(predicted - ground_truth) <= tolerance
        
        return {
            'problem_id': problem_id,
            'question': problem_text,
            'ground_truth': ground_truth,
            'predicted': predicted,
            'correct': correct,
            'raw_responses': raw_responses if use_self_consistency else None,
            'error': None
        }
    
    except Exception as e:
        return {
            'problem_id': problem_id,
            'question': problem_text,
            'ground_truth': ground_truth,
            'predicted': None,
            'correct': False,
            'error': str(e)
        }


def run_evaluation(
    input_file: Path,
    output_file: Path,
    use_self_consistency: bool = True,
    num_samples: int = 5,
    max_problems: Optional[int] = None,
    temperature: float = 0.7,
    model: str = "meta-llama/llama-3.2-3b-instruct",
    workers: int = 10
):
    """Run GSM8K evaluation with 8-shot CoT using multithreading"""
    print(f"\n{'='*60}")
    print("GSM8K 8-Shot Chain-of-Thought Evaluation")
    print(f"{'='*60}")
    print(f"Model: {model}")
    print(f"Self-Consistency: {use_self_consistency}")
    if use_self_consistency:
        print(f"Samples per problem: {num_samples}")
    print(f"Temperature: {temperature}")
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
    
    # Initialize LLM
    llm = ChatOpenAI(
        model=model,
        base_url="https://openrouter.ai/api/v1",
        temperature=temperature,
        max_tokens=512,
        request_timeout=90,
    )
    
    # Evaluate problems with multithreading
    results = []
    correct = 0
    completed_count = 0
    last_progress_time = time.time()
    
    print("Starting evaluation with multithreading...\n")
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_problem = {
            executor.submit(
                evaluate_single_problem, 
                problem, 
                llm,
                use_self_consistency,
                num_samples
            ): problem
            for problem in problems
        }
        
        pending_futures = set(future_to_problem.keys())
        
        with tqdm(total=len(problems), desc="Evaluating") as pbar:
            while pending_futures and completed_count < len(problems):
                current_time = time.time()
                
                # Check if stuck (no progress for 180 seconds)
                if current_time - last_progress_time > 180:
                    print(f"\n⚠️  No progress for 180s. Cancelling {len(pending_futures)} remaining tasks...")
                    for future in pending_futures:
                        future.cancel()
                    break
                
                # Get completed futures
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
                        result = future.result(timeout=1)
                        results.append(result)
                        
                        if result['correct']:
                            correct += 1
                        
                        completed_count += 1
                        last_progress_time = current_time
                        pbar.update(1)
                        
                        # Progress update every 50 problems
                        if completed_count % 50 == 0:
                            accuracy = 100 * correct / completed_count
                            print(f"\n[Progress] {completed_count}/{len(problems)} | Accuracy: {accuracy:.2f}%")
                        
                    except TimeoutError:
                        print(f"\n⚠️  Problem {problem.get('id', 'unknown')} timed out")
                        results.append({
                            'problem_id': problem.get('id', 'unknown'),
                            'question': problem.get('input', ''),
                            'ground_truth': extract_ground_truth(problem),
                            'predicted': None,
                            'correct': False,
                            'error': 'Timed out'
                        })
                        completed_count += 1
                        pbar.update(1)
                        
                    except Exception as e:
                        print(f"\n⚠️  Problem {problem.get('id', 'unknown')} error: {e}")
                        results.append({
                            'problem_id': problem.get('id', 'unknown'),
                            'question': problem.get('input', ''),
                            'ground_truth': extract_ground_truth(problem),
                            'predicted': None,
                            'correct': False,
                            'error': str(e)
                        })
                        completed_count += 1
                        pbar.update(1)
                
                # Small sleep to avoid busy waiting
                if not done_futures:
                    time.sleep(0.5)
        
        # Handle any remaining futures
        for future in pending_futures:
            problem = future_to_problem[future]
            print(f"\n⚠️  Problem {problem.get('id', 'unknown')} was cancelled")
            results.append({
                'problem_id': problem.get('id', 'unknown'),
                'question': problem.get('input', ''),
                'ground_truth': extract_ground_truth(problem),
                'predicted': None,
                'correct': False,
                'error': 'Cancelled due to timeout'
            })
    
    print(f"\n✓ Completed {len(results)}/{len(problems)} problems")
    
    # Calculate final accuracy
    total = len(results)
    accuracy = (100 * correct / total) if total > 0 else 0
    
    # Save results
    output_data = {
        'config': {
            'model': model,
            'self_consistency': use_self_consistency,
            'num_samples': num_samples if use_self_consistency else 1,
            'temperature': temperature,
        },
        'summary': {
            'total_problems': total,
            'correct': correct,
            'accuracy': accuracy,
        },
        'results': results
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Total problems:  {total}")
    print(f"Correct:         {correct}")
    print(f"Accuracy:        {accuracy:.2f}%")
    print(f"{'='*60}")
    print(f"\n✓ Results saved to: {output_file}")
    
    # Show some examples
    errors = [r for r in results if not r['correct'] and r['error'] is None]
    if errors:
        print(f"\nSample incorrect answers (showing 3/{len(errors)}):")
        for i, result in enumerate(errors[:3], 1):
            print(f"\n{i}. Problem ID: {result['problem_id']}")
            print(f"   Question: {result['question'][:80]}...")
            print(f"   Predicted: {result['predicted']}, Ground Truth: {result['ground_truth']}")


def main():
    parser = argparse.ArgumentParser(
        description='GSM8K 8-Shot CoT Evaluation via OpenRouter'
    )
    parser.add_argument('--input', type=str, required=True,
                       help='Input JSONL file (GSM8K test set)')
    parser.add_argument('--output', type=str, 
                       default='gsm8k_8shot_cot_results.json',
                       help='Output JSON file for results')
    parser.add_argument('--model', type=str,
                       default='meta-llama/llama-3.2-3b-instruct',
                       help='Model to use (default: llama-3.2-3b-instruct)')
    parser.add_argument('--self-consistency', action='store_true',
                       help='Use self-consistency voting (multiple samples)')
    parser.add_argument('--num-samples', type=int, default=5,
                       help='Number of samples for self-consistency (default: 5)')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Temperature for sampling (default: 0.7)')
    parser.add_argument('--max-problems', type=int,
                       help='Maximum number of problems to evaluate')
    parser.add_argument('--workers', type=int, default=10,
                       help='Number of parallel workers (default: 10)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file {input_path} not found")
        return
    
    run_evaluation(
        input_file=input_path,
        output_file=Path(args.output),
        use_self_consistency=args.self_consistency,
        num_samples=args.num_samples,
        max_problems=args.max_problems,
        temperature=args.temperature,
        model=args.model,
        workers=args.workers
    )


if __name__ == "__main__":
    main()
