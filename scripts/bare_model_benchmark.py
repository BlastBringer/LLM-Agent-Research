#!/usr/bin/env python3
"""
🔬 BARE MODEL BENCHMARK
======================

Simple benchmark for raw LLM performance without any preprocessing.
Tests Llama 3.2 3B Instruct on clean datasets with input/output format.

Usage:
    python3 bare_model_benchmark.py --input svamp_dataset.jsonl --output results.json
    python3 bare_model_benchmark.py --input dataset.jsonl --output results.json --samples 100 --workers 10
"""

import json
import argparse
import logging
import os
import sys
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import time

from dotenv import load_dotenv

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    print("❌ LangChain not available. Install: pip install langchain langchain-openai python-dotenv")
    sys.exit(1)

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BareModelBenchmark:
    """Benchmark raw LLM performance."""
    
    def __init__(self, model_name: str = "meta-llama/llama-3.2-3b-instruct", temperature: float = 0.1):
        """Initialize the LLM."""
        self.model_name = model_name
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://openrouter.io/api/v1")
        self.temperature = temperature
        
        if not self.api_key:
            raise ValueError("❌ OPENAI_API_KEY not found in .env file")
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            openai_api_key=self.api_key,
            openai_api_base=self.base_url,
            max_tokens=500,
            timeout=60
        )
        
        logger.info(f"✅ Initialized: {self.model_name}")
        logger.info(f"🌡️  Temperature: {self.temperature}")
    
    def extract_answer(self, response: str) -> Optional[float]:
        """Extract numerical answer from model response."""
        # Try "Final Answer: X" pattern
        match = re.search(r'Final Answer:\s*([^\n]+)', response, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1).strip())
            except ValueError:
                pass
        
        # Try "answer is X" pattern
        match = re.search(r'(?:answer|result) is\s*([^\n]+)', response, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1).strip())
            except ValueError:
                pass
        
        # Try boxed answer
        match = re.search(r'\\boxed\{([^}]+)\}', response)
        if match:
            try:
                return float(match.group(1).strip())
            except ValueError:
                pass
        
        # Fallback: try to find any number in response
        numbers = re.findall(r'-?\d+\.?\d*', response)
        if numbers:
            try:
                return float(numbers[-1])
            except ValueError:
                pass
        
        return None
    
    def solve_problem(self, problem: Dict[str, Any], problem_id: int) -> Dict[str, Any]:
        """Solve a single problem."""
        start_time = time.time()
        
        try:
            question = problem.get('input', problem.get('question', ''))
            ground_truth_raw = problem.get('output', problem.get('answer', ''))
            
            if not question or not ground_truth_raw:
                return {
                    'problem_id': problem_id,
                    'question': question,
                    'ground_truth': None,
                    'prediction': None,
                    'correct': False,
                    'time': 0,
                    'error': 'Missing question or answer'
                }
            
            # Convert ground truth to float
            try:
                ground_truth = float(ground_truth_raw)
            except (ValueError, TypeError):
                return {
                    'problem_id': problem_id,
                    'question': question[:100],
                    'ground_truth': None,
                    'prediction': None,
                    'correct': False,
                    'time': 0,
                    'error': f'Cannot convert ground truth to float: {ground_truth_raw}'
                }
            
            # Create prompt
            prompt = f"""Solve this math problem and provide the final numerical answer.

Problem:
{question}

Provide the final answer as a single number. Format: Final Answer: [number]

Solution:"""
            
            # Call model
            response = self.llm.invoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Extract answer
            prediction = self.extract_answer(response_text)
            
            # Check correctness
            correct = False
            if prediction is not None:
                correct = abs(prediction - ground_truth) < 1e-6
            
            elapsed = time.time() - start_time
            
            return {
                'problem_id': problem_id,
                'question': question[:100],
                'ground_truth': ground_truth,
                'prediction': prediction,
                'correct': correct,
                'time': elapsed,
                'error': None
            }
        
        except Exception as e:
            elapsed = time.time() - start_time
            return {
                'problem_id': problem_id,
                'question': problem.get('input', '')[:100],
                'ground_truth': None,
                'prediction': None,
                'correct': False,
                'time': elapsed,
                'error': str(e)
            }
    
    def benchmark(self, problems: List[Dict[str, Any]], num_workers: int = 5) -> List[Dict[str, Any]]:
        """Run benchmark with multithreading."""
        results = []
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            futures = [
                executor.submit(self.solve_problem, problem, idx)
                for idx, problem in enumerate(problems)
            ]
            
            # Collect results as they complete
            for idx, future in enumerate(futures, 1):
                result = future.result()
                results.append(result)
                
                # Progress indicator
                if idx % 10 == 0:
                    logger.info(f"Progress: {idx}/{len(problems)}")
        
        return results
    
    def calculate_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate benchmark statistics."""
        total = len(results)
        correct = sum(1 for r in results if r['correct'])
        errors = sum(1 for r in results if r['error'] is not None)
        total_time = sum(r['time'] for r in results)
        avg_time = total_time / total if total > 0 else 0
        
        accuracy = (correct / total * 100) if total > 0 else 0
        
        return {
            'total': total,
            'correct': correct,
            'errors': errors,
            'accuracy': round(accuracy, 2),
            'avg_time_per_problem': round(avg_time, 2),
            'total_time': round(total_time, 2)
        }


def load_dataset(input_file: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load dataset from JSONL or JSON."""
    path = Path(input_file)
    
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {input_file}")
    
    problems = []
    
    if path.suffix == '.jsonl':
        with open(path, 'r') as f:
            for line in f:
                if line.strip():
                    problems.append(json.loads(line))
    elif path.suffix == '.json':
        with open(path, 'r') as f:
            data = json.load(f)
            problems = data if isinstance(data, list) else data.get('results', [])
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    
    if max_samples:
        problems = problems[:max_samples]
    
    logger.info(f"✅ Loaded {len(problems)} problems from {input_file}")
    return problems


def main():
    parser = argparse.ArgumentParser(description="Bare model benchmark")
    parser.add_argument('--input', required=True, help='Input dataset file (.jsonl or .json)')
    parser.add_argument('--output', default='bare_model_results.json', help='Output results file')
    parser.add_argument('--samples', type=int, default=None, help='Number of samples to process (default: all)')
    parser.add_argument('--workers', type=int, default=5, help='Number of worker threads (default: 5)')
    parser.add_argument('--model', type=str, default='meta-llama/llama-3.2-3b-instruct', help='Model name')
    parser.add_argument('--temperature', type=float, default=0.1, help='Temperature (default: 0.1)')
    
    args = parser.parse_args()
    
    # Load dataset
    try:
        problems = load_dataset(args.input, max_samples=args.samples)
    except Exception as e:
        logger.error(f"❌ Failed to load dataset: {e}")
        sys.exit(1)
    
    # Initialize benchmark
    try:
        benchmark = BareModelBenchmark(
            model_name=args.model,
            temperature=args.temperature
        )
    except Exception as e:
        logger.error(f"❌ Failed to initialize model: {e}")
        sys.exit(1)
    
    # Run benchmark
    logger.info(f"🚀 Starting benchmark on {len(problems)} problems...")
    logger.info(f"📊 Using {args.workers} worker threads")
    
    results = benchmark.benchmark(problems, num_workers=args.workers)
    stats = benchmark.calculate_stats(results)
    
    # Save results
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'model': args.model,
            'temperature': args.temperature,
            'workers': args.workers,
            'input_file': args.input
        },
        'stats': stats,
        'results': results
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"✅ Results saved to: {args.output}")
    
    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    print(f"Total problems:    {stats['total']}")
    print(f"Correct:           {stats['correct']}")
    print(f"Accuracy:          {stats['accuracy']}%")
    print(f"Errors:            {stats['errors']}")
    print(f"Avg time/problem:  {stats['avg_time_per_problem']}s")
    print(f"Total time:        {stats['total_time']}s")
    print("="*60)


if __name__ == "__main__":
    main()
