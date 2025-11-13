#!/usr/bin/env python3
"""
GSM8K 8-Shot PoT - Final Optimized Version
Using the EXACT structure that achieved 77.2% with CoT, but with Python code instead
"""

import json
import re
import argparse
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from collections import Counter

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


# ============================================================
# 8-Shot Examples - EXACT problems from CoT, but with Python code
# ============================================================
FEWSHOT_EXAMPLES = """
Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?

# Python code to solve
trees_start = 15
trees_end = 21
trees_planted = trees_end - trees_start
answer = trees_planted

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?

# Python code to solve
cars_start = 3
cars_arrive = 2
total_cars = cars_start + cars_arrive
answer = total_cars

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?

# Python code to solve
leah = 32
sister = 42
total = leah + sister
ate = 35
left = total - ate
answer = left

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?

# Python code to solve
jason_had = 20
jason_now = 12
gave_to_denny = jason_had - jason_now
answer = gave_to_denny

Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?

# Python code to solve
toys_start = 5
from_mom = 2
from_dad = 2
total_toys = toys_start + from_mom + from_dad
answer = total_toys

Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?

# Python code to solve
computers_start = 9
per_day = 5
days = 4
added = per_day * days
total_computers = computers_start + added
answer = total_computers

Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?

# Python code to solve
start = 58
lost_tuesday = 23
lost_wednesday = 2
remaining = start - lost_tuesday - lost_wednesday
answer = remaining

Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?

# Python code to solve
money_start = 23
bagels = 5
price = 3
spent = bagels * price
money_left = money_start - spent
answer = money_left
"""


def build_prompt(question: str) -> str:
    """Build the complete prompt"""
    return f"{FEWSHOT_EXAMPLES}\nQ: {question}\n\n# Python code to solve\n"


def extract_code(response: str) -> Optional[str]:
    """Extract Python code from response"""
    # Remove markdown code blocks if present
    response = re.sub(r'```python\s*', '', response)
    response = re.sub(r'```\s*', '', response)
    
    # Find lines with variable assignments
    lines = response.strip().split('\n')
    code_lines = []
    
    for line in lines:
        stripped = line.strip()
        # Include lines that look like code
        if '=' in stripped or stripped.startswith('#'):
            code_lines.append(line)
        # Stop at explanatory text
        elif stripped and len(stripped.split()) > 6 and not any(c in stripped for c in ['=', '+', '-', '*', '/', '(', ')']):
            break
    
    code = '\n'.join(code_lines).strip()
    
    # Ensure we have 'answer =' 
    if 'answer' not in code.lower():
        # Find the last variable and set it as answer
        var_matches = re.findall(r'(\w+)\s*=', code)
        if var_matches:
            last_var = var_matches[-1]
            code += f"\nanswer = {last_var}"
    
    return code if code else None


def execute_code(code: str) -> Optional[float]:
    """Execute code and extract answer"""
    try:
        namespace = {}
        exec(code, namespace)
        
        # Try both 'answer' and 'Answer'
        for var in ['answer', 'Answer', 'result', 'Result']:
            if var in namespace:
                return float(namespace[var])
        
        return None
    except Exception:
        return None


def extract_ground_truth(problem: Dict[str, Any]) -> Optional[float]:
    """Extract ground truth"""
    for field in ['output', 'target', 'answer']:
        if field in problem:
            value = problem[field]
            if isinstance(value, (int, float)):
                return float(value)
            # Extract number from string
            match = re.search(r'-?\d+(?:\.\d+)?', str(value))
            if match:
                return float(match.group(0))
    return None


def solve_single(problem: Dict, llm: ChatOpenAI) -> Dict[str, Any]:
    """Generate one solution"""
    question = problem.get('input', problem.get('question', ''))
    
    try:
        prompt = build_prompt(question)
        response = llm.invoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        code = extract_code(response_text)
        if not code:
            return {'predicted': None, 'error': 'No code extracted'}
        
        predicted = execute_code(code)
        return {
            'predicted': predicted,
            'code': code,
            'error': None if predicted is not None else 'Execution failed'
        }
    except Exception as e:
        return {'predicted': None, 'error': str(e)}


def solve_with_self_consistency(
    problem: Dict,
    llm: ChatOpenAI,
    num_samples: int = 5
) -> Dict[str, Any]:
    """Generate multiple solutions and vote"""
    solutions = []
    
    for _ in range(num_samples):
        sol = solve_single(problem, llm)
        solutions.append(sol)
    
    # Collect valid predictions
    predictions = [s['predicted'] for s in solutions if s['predicted'] is not None]
    
    if not predictions:
        return {
            'predicted': None,
            'all_predictions': [],
            'solutions': solutions,
            'error': 'All samples failed'
        }
    
    # Majority vote
    counts = Counter(predictions)
    majority = counts.most_common(1)[0][0]
    
    return {
        'predicted': majority,
        'all_predictions': predictions,
        'vote_counts': dict(counts),
        'solutions': solutions,
        'error': None
    }


def evaluate_problem(
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
        result = solve_with_self_consistency(problem, llm, num_samples)
    else:
        result = solve_single(problem, llm)
    
    predicted = result.get('predicted')
    correct = False
    
    if predicted is not None and ground_truth is not None:
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
    workers: int = 15
):
    """Run evaluation"""
    print(f"\n{'='*60}")
    print("GSM8K 8-Shot PoT - Final Optimized Version")
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
    
    # Evaluate with multithreading
    results = []
    correct = 0
    completed = 0
    last_progress = time.time()
    
    print("Starting evaluation...\n")
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_problem = {
            executor.submit(evaluate_problem, p, llm, use_self_consistency, num_samples): p
            for p in problems
        }
        
        pending = set(future_to_problem.keys())
        
        with tqdm(total=len(problems), desc="Evaluating") as pbar:
            while pending and completed < len(problems):
                current = time.time()
                
                # Timeout check
                if current - last_progress > 180:
                    print(f"\n⚠️  No progress for 180s. Cancelling {len(pending)} remaining...")
                    for f in pending:
                        f.cancel()
                    break
                
                # Check done futures
                done = [f for f in list(pending) if f.done()]
                
                for future in done:
                    pending.remove(future)
                    problem = future_to_problem[future]
                    
                    try:
                        result = future.result(timeout=1)
                        results.append(result)
                        
                        if result['correct']:
                            correct += 1
                        
                        completed += 1
                        last_progress = current
                        pbar.update(1)
                        
                        if completed % 50 == 0:
                            acc = 100 * correct / completed
                            print(f"\n[Progress] {completed}/{len(problems)} | Accuracy: {acc:.2f}%")
                    
                    except Exception as e:
                        results.append({
                            'problem_id': problem.get('id', 'unknown'),
                            'question': problem.get('input', ''),
                            'ground_truth': extract_ground_truth(problem),
                            'predicted': None,
                            'correct': False,
                            'error': str(e)
                        })
                        completed += 1
                        pbar.update(1)
                
                if not done:
                    time.sleep(0.5)
    
    print(f"\n✓ Completed {len(results)}/{len(problems)} problems")
    
    # Calculate stats
    total = len(results)
    accuracy = (100 * correct / total) if total > 0 else 0
    successful = sum(1 for r in results if r['predicted'] is not None)
    exec_rate = (100 * successful / total) if total > 0 else 0
    
    # Save results
    output_data = {
        'config': {
            'model': model,
            'method': '8-shot PoT (Final Optimized)',
            'self_consistency': use_self_consistency,
            'num_samples': num_samples if use_self_consistency else 1,
            'temperature': temperature,
        },
        'summary': {
            'total': total,
            'correct': correct,
            'accuracy': accuracy,
            'successful_executions': successful,
            'execution_rate': exec_rate
        },
        'results': results
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Total problems:         {total}")
    print(f"Correct:                {correct}")
    print(f"Accuracy:               {accuracy:.2f}%")
    print(f"Successful executions:  {successful} ({exec_rate:.1f}%)")
    print(f"{'='*60}")
    print(f"\n✓ Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='GSM8K 8-Shot PoT - Final Version')
    parser.add_argument('--input', type=str, required=True, help='Input JSONL file')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file')
    parser.add_argument('--model', type=str, default='meta-llama/llama-3.2-3b-instruct', 
                       help='Model via OpenRouter')
    parser.add_argument('--self-consistency', action='store_true',
                       help='Use self-consistency voting')
    parser.add_argument('--num-samples', type=int, default=5,
                       help='Samples for self-consistency (default: 5)')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Temperature (default: 0.7)')
    parser.add_argument('--max-problems', type=int,
                       help='Max problems to evaluate')
    parser.add_argument('--workers', type=int, default=15,
                       help='Parallel workers (default: 15)')
    
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
