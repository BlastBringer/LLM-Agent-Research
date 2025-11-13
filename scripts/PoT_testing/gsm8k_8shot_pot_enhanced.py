#!/usr/bin/env python3
"""
GSM8K 8-Shot Program-of-Thought (PoT) Evaluation - Enhanced Version
Combining proven 8-shot CoT methodology with Program-of-Thought execution
"""

import json
import re
import argparse
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from tqdm import tqdm
from collections import Counter

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


# ============================================================
# 8-Shot Program-of-Thought Examples (Python Code)
# These examples match the standard GSM8K 8-shot CoT problems
# ============================================================
FEWSHOT_EXAMPLES = [
    {
        "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "code": """trees_before = 15
trees_after = 21
trees_planted = trees_after - trees_before
answer = trees_planted"""
    },
    {
        "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "code": """cars_initial = 3
cars_arrived = 2
total_cars = cars_initial + cars_arrived
answer = total_cars"""
    },
    {
        "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "code": """leah_chocolates = 32
sister_chocolates = 42
total_chocolates = leah_chocolates + sister_chocolates
chocolates_eaten = 35
chocolates_left = total_chocolates - chocolates_eaten
answer = chocolates_left"""
    },
    {
        "question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "code": """jason_had = 20
jason_has = 12
lollipops_given = jason_had - jason_has
answer = lollipops_given"""
    },
    {
        "question": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        "code": """toys_initial = 5
toys_from_mom = 2
toys_from_dad = 2
total_toys = toys_initial + toys_from_mom + toys_from_dad
answer = total_toys"""
    },
    {
        "question": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
        "code": """computers_initial = 9
computers_per_day = 5
days = 4
computers_added = computers_per_day * days
total_computers = computers_initial + computers_added
answer = total_computers"""
    },
    {
        "question": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
        "code": """golf_balls_initial = 58
lost_tuesday = 23
lost_wednesday = 2
golf_balls_left = golf_balls_initial - lost_tuesday - lost_wednesday
answer = golf_balls_left"""
    },
    {
        "question": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "code": """money_initial = 23
bagels = 5
price_per_bagel = 3
money_spent = bagels * price_per_bagel
money_left = money_initial - money_spent
answer = money_left"""
    }
]


def build_8shot_pot_prompt(question: str) -> str:
    """Build prompt with 8-shot PoT examples"""
    prompt_parts = [
        "Solve math word problems by writing Python code. "
        "Store the final numerical answer in a variable called 'answer'. "
        "Use clear variable names and simple arithmetic operations.\n"
    ]
    
    # Add 8-shot examples
    for i, example in enumerate(FEWSHOT_EXAMPLES, 1):
        prompt_parts.append(f"\nQ: {example['question']}\n")
        prompt_parts.append(f"```python\n{example['code']}\n```\n")
    
    # Add the actual question
    prompt_parts.append(f"\nQ: {question}\n")
    prompt_parts.append("```python\n")
    
    return "".join(prompt_parts)


def extract_python_code(response: str) -> Optional[str]:
    """Extract Python code from LLM response"""
    # Try to find code in markdown code blocks first
    code_block_pattern = r'```(?:python)?\s*(.*?)```'
    matches = re.findall(code_block_pattern, response, re.DOTALL)
    
    if matches:
        # Get the last code block (most likely to be the answer)
        code = matches[-1].strip()
        # Ensure it has 'answer =' in it
        if 'answer' in code:
            return code
    
    # If no code block or no 'answer' variable, try to extract code-like content
    lines = response.strip().split('\n')
    code_lines = []
    in_code = False
    
    for line in lines:
        stripped = line.strip()
        
        # Detect start of code (variable assignment or calculation)
        if '=' in stripped and not stripped.startswith('#'):
            in_code = True
            code_lines.append(line)
        # Continue collecting if we're in code block
        elif in_code:
            # Stop if we hit explanatory text
            if stripped and not stripped.startswith('#') and '=' not in stripped and not any(op in stripped for op in ['+', '-', '*', '/', '(', ')']):
                # Check if it looks like a sentence
                if len(stripped.split()) > 5:
                    break
            code_lines.append(line)
        # Also collect standalone comments
        elif stripped.startswith('#'):
            if in_code:
                code_lines.append(line)
    
    if code_lines:
        code = '\n'.join(code_lines).strip()
        # Make sure it has answer variable
        if 'answer' in code:
            return code
        # If not, try to add answer = <last_variable>
        last_var_match = re.findall(r'(\w+)\s*=', code)
        if last_var_match:
            code += f"\nanswer = {last_var_match[-1]}"
            return code
    
    return None


def execute_code(code: str, timeout: int = 5) -> Optional[float]:
    """Execute Python code and extract the answer variable"""
    try:
        # Create a clean namespace
        namespace = {}
        
        # Execute the code
        exec(code, namespace)
        
        # Get the answer variable
        if 'answer' in namespace:
            result = namespace['answer']
            # Convert to float if possible
            try:
                return float(result)
            except (ValueError, TypeError):
                return None
        
        return None
    except Exception as e:
        return None


def extract_ground_truth(problem: Dict[str, Any]) -> Optional[float]:
    """Extract ground truth answer from problem"""
    # Try different field names
    target = None
    for field in ['output', 'target', 'answer']:
        if field in problem:
            target = problem[field]
            break
    
    if target is None:
        return None
    
    # Extract number from target
    if isinstance(target, (int, float)):
        return float(target)
    
    # Try to extract from #### format
    match = re.search(r'####\s*(-?\d+(?:\.\d+)?)', str(target))
    if match:
        return float(match.group(1))
    
    # Try to find any number
    match = re.search(r'-?\d+(?:\.\d+)?', str(target))
    if match:
        return float(match.group(0))
    
    return None


def generate_single_solution(problem: Dict, llm: ChatOpenAI) -> Dict[str, Any]:
    """Generate a single solution for a problem"""
    question = problem.get('input', problem.get('question', ''))
    ground_truth = extract_ground_truth(problem)
    
    try:
        # Build prompt and get response
        prompt = build_8shot_pot_prompt(question)
        response = llm.invoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Extract and execute code
        code = extract_python_code(response_text)
        if not code:
            return {
                'predicted': None,
                'code': None,
                'raw_response': response_text,
                'error': 'Failed to extract Python code'
            }
        
        # Execute code
        predicted = execute_code(code)
        
        return {
            'predicted': predicted,
            'code': code,
            'raw_response': response_text,
            'error': None if predicted is not None else 'Code execution failed'
        }
        
    except Exception as e:
        return {
            'predicted': None,
            'code': None,
            'raw_response': None,
            'error': str(e)
        }


def generate_with_self_consistency(
    problem: Dict, 
    llm: ChatOpenAI, 
    num_samples: int = 5
) -> Dict[str, Any]:
    """Generate multiple solutions and use majority voting"""
    solutions = []
    
    for _ in range(num_samples):
        solution = generate_single_solution(problem, llm)
        solutions.append(solution)
    
    # Collect all predicted values
    predictions = [s['predicted'] for s in solutions if s['predicted'] is not None]
    
    if not predictions:
        # No successful predictions
        return {
            'predicted': None,
            'all_predictions': [],
            'solutions': solutions,
            'error': 'All samples failed'
        }
    
    # Majority vote
    vote_counts = Counter(predictions)
    majority_prediction = vote_counts.most_common(1)[0][0]
    
    return {
        'predicted': majority_prediction,
        'all_predictions': predictions,
        'vote_distribution': dict(vote_counts),
        'solutions': solutions,
        'error': None
    }


def evaluate_single_problem(
    problem: Dict,
    llm: ChatOpenAI,
    use_self_consistency: bool = True,
    num_samples: int = 5
) -> Dict[str, Any]:
    """Evaluate a single problem"""
    question = problem.get('input', problem.get('question', ''))
    ground_truth = extract_ground_truth(problem)
    problem_id = problem.get('id', 'unknown')
    
    if use_self_consistency:
        result = generate_with_self_consistency(problem, llm, num_samples)
    else:
        result = generate_single_solution(problem, llm)
    
    predicted = result.get('predicted')
    correct = False
    
    if predicted is not None and ground_truth is not None:
        # Allow small floating point differences
        correct = abs(predicted - ground_truth) < 1e-6
    
    return {
        'problem_id': problem_id,
        'question': question,
        'ground_truth': ground_truth,
        'predicted': predicted,
        'correct': correct,
        'error': result.get('error'),
        'details': result
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
    """Run GSM8K evaluation with 8-shot PoT using multithreading"""
    print(f"\n{'='*60}")
    print("GSM8K 8-Shot Program-of-Thought (PoT) Evaluation")
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
    
    # Calculate execution success rate
    successful_executions = sum(1 for r in results if r['predicted'] is not None)
    execution_rate = (100 * successful_executions / total) if total > 0 else 0
    
    # Save results
    output_data = {
        'config': {
            'model': model,
            'self_consistency': use_self_consistency,
            'num_samples': num_samples if use_self_consistency else 1,
            'temperature': temperature,
            'method': '8-shot Program-of-Thought (PoT)'
        },
        'summary': {
            'total_problems': total,
            'correct': correct,
            'accuracy': accuracy,
            'successful_executions': successful_executions,
            'execution_rate': execution_rate
        },
        'results': results
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Total problems:         {total}")
    print(f"Correct:                {correct}")
    print(f"Accuracy:               {accuracy:.2f}%")
    print(f"Successful executions:  {successful_executions} ({execution_rate:.1f}%)")
    print(f"{'='*60}")
    print(f"\n✓ Results saved to: {output_file}")
    
    # Show some examples
    errors = [r for r in results if not r['correct'] and r.get('error')]
    if errors:
        print(f"\nSample errors (showing 3/{len(errors)}):")
        for i, result in enumerate(errors[:3], 1):
            print(f"\n{i}. Problem ID: {result['problem_id']}")
            print(f"   Error: {result['error']}")
    
    incorrect = [r for r in results if not r['correct'] and not r.get('error')]
    if incorrect:
        print(f"\nSample incorrect answers (showing 3/{len(incorrect)}):")
        for i, result in enumerate(incorrect[:3], 1):
            print(f"\n{i}. Problem ID: {result['problem_id']}")
            print(f"   Question: {result['question'][:80]}...")
            print(f"   Predicted: {result['predicted']}, Ground Truth: {result['ground_truth']}")


def main():
    parser = argparse.ArgumentParser(description='GSM8K 8-Shot PoT Evaluation')
    parser.add_argument('--input', type=str, required=True,
                       help='Input JSONL file with GSM8K problems')
    parser.add_argument('--output', type=str, required=True,
                       help='Output JSON file for results')
    parser.add_argument('--model', type=str, default='meta-llama/llama-3.2-3b-instruct',
                       help='Model to use via OpenRouter')
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
