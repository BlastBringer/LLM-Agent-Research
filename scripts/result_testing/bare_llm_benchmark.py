#!/usr/bin/env python3
"""
🔬 BARE LLM BENCHMARK
=====================

Test raw Llama 3.1 8B performance WITHOUT any preprocessing pipeline.
No information retrieval, no parsing, no unit conversion - just pure LLM.

This gives us the baseline accuracy to compare against the full pipeline.

Usage:
    python3 bare_llm_benchmark.py --input dataset.jsonl --output results.json
    python3 bare_llm_benchmark.py --input dataset.jsonl --batch-size 20
"""

import sys
import os
import json
import logging
import argparse
import time
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv

# Import LangChain
try:
    from langchain_openai import ChatOpenAI
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("❌ LangChain not available. Install: pip install langchain langchain-openai")
    sys.exit(1)

# Import ground truth utilities
from ground_truth_utils import extract_ground_truth_from_problem, extract_numeric_answer

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result for a single problem."""
    problem_id: int
    problem_text: str
    ground_truth: Optional[float]
    ground_truth_raw: str
    llm_answer: Optional[float]
    llm_raw_response: str
    is_correct: bool
    processing_time: float
    error: Optional[str] = None


class BareLLMBenchmark:
    """
    Benchmark raw LLM performance without any preprocessing.
    """
    
    def __init__(self, model_name: str = None, temperature: float = 0.1):
        """Initialize the bare LLM."""
        self.model_name = model_name or os.getenv("APPRENTICE_MODEL", "meta-llama/llama-3.1-8b-instruct")
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
        self.temperature = temperature
        
        if not self.api_key:
            logger.error("❌ No API key found. Set OPENAI_API_KEY in .env")
            sys.exit(1)
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            openai_api_key=self.api_key,
            openai_api_base=self.base_url,
            max_tokens=2000
        )
        
        logger.info(f"✅ Initialized: {self.model_name}")
        logger.info(f"🌡️  Temperature: {self.temperature}")
    
    def create_prompt(self, problem_text: str) -> str:
        """
        Create a simple prompt for the LLM.
        Just ask it to solve the problem and give a final answer.
        """
        prompt = f"""Solve this math problem step by step.

Problem:
{problem_text}

Provide your solution with clear reasoning steps, and end with your final answer in the format:
Final Answer: [your numerical answer]

Solution:"""
        return prompt
    
    def extract_answer_from_response(self, response: str) -> Optional[float]:
        """
        Extract numerical answer from LLM response.
        
        Looks for patterns like:
        - "Final Answer: 150"
        - "The answer is 150"
        - "\\boxed{150}"
        """
        # Try "Final Answer: X" pattern
        match = re.search(r'Final Answer:\s*([^\n]+)', response, re.IGNORECASE)
        if match:
            answer_str = match.group(1).strip()
            numeric = extract_numeric_answer(answer_str)
            if numeric is not None:
                return numeric
        
        # Try "answer is X" pattern
        match = re.search(r'(?:answer|result) is\s*([^\n]+)', response, re.IGNORECASE)
        if match:
            answer_str = match.group(1).strip()
            numeric = extract_numeric_answer(answer_str)
            if numeric is not None:
                return numeric
        
        # Try \boxed{X} pattern
        match = re.search(r'\\boxed\{([^}]+)\}', response)
        if match:
            answer_str = match.group(1).strip()
            numeric = extract_numeric_answer(answer_str)
            if numeric is not None:
                return numeric
        
        # Try to find any number in the last few lines
        lines = response.strip().split('\n')
        for line in reversed(lines[-5:]):  # Check last 5 lines
            numeric = extract_numeric_answer(line)
            if numeric is not None:
                return numeric
        
        return None
    
    def solve_problem(
        self,
        problem_id: int,
        problem_text: str,
        ground_truth: Optional[float],
        ground_truth_raw: str
    ) -> BenchmarkResult:
        """
        Solve a single problem with bare LLM.
        
        Args:
            problem_id: Problem number
            problem_text: The problem text
            ground_truth: Ground truth numeric answer
            ground_truth_raw: Ground truth raw string
        
        Returns:
            BenchmarkResult with answer and correctness
        """
        start_time = time.time()
        
        try:
            # Create prompt
            prompt = self.create_prompt(problem_text)
            
            # Call LLM
            response = self.llm.invoke(prompt)
            llm_raw_response = response.content
            
            # Extract answer
            llm_answer = self.extract_answer_from_response(llm_raw_response)
            
            # Check correctness
            is_correct = False
            if llm_answer is not None and ground_truth is not None:
                # Compare with 0.1% tolerance
                relative_diff = abs(llm_answer - ground_truth) / abs(ground_truth) if ground_truth != 0 else abs(llm_answer)
                is_correct = relative_diff < 0.001
            
            processing_time = time.time() - start_time
            
            return BenchmarkResult(
                problem_id=problem_id,
                problem_text=problem_text,
                ground_truth=ground_truth,
                ground_truth_raw=ground_truth_raw,
                llm_answer=llm_answer,
                llm_raw_response=llm_raw_response,
                is_correct=is_correct,
                processing_time=processing_time,
                error=None
            )
        
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"❌ Problem {problem_id} failed: {e}")
            
            return BenchmarkResult(
                problem_id=problem_id,
                problem_text=problem_text,
                ground_truth=ground_truth,
                ground_truth_raw=ground_truth_raw,
                llm_answer=None,
                llm_raw_response="",
                is_correct=False,
                processing_time=processing_time,
                error=str(e)
            )
    
    def benchmark_dataset(
        self,
        problems: List[Dict[str, Any]],
        batch_size: int = 10,
        use_parallel: bool = True
    ) -> List[BenchmarkResult]:
        """
        Benchmark the LLM on a dataset of problems.
        
        Args:
            problems: List of problem dicts with 'input' and 'output' fields
            batch_size: Number of problems to process in parallel
            use_parallel: Whether to use parallel processing
        
        Returns:
            List of BenchmarkResult objects
        """
        logger.info("=" * 70)
        logger.info("🔬 BARE LLM BENCHMARK")
        logger.info("=" * 70)
        logger.info(f"📊 Total problems: {len(problems)}")
        logger.info(f"🤖 Model: {self.model_name}")
        logger.info(f"⚡ Batch size: {batch_size}")
        logger.info(f"🔧 Parallel: {use_parallel}")
        logger.info("")
        
        results = []
        
        if use_parallel:
            # Parallel processing
            max_workers = min(batch_size, multiprocessing.cpu_count())
            logger.info(f"🚀 Using {max_workers} workers\n")
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                
                for i, problem_data in enumerate(problems, 1):
                    # Extract ground truth
                    ground_truth, ground_truth_raw, _ = extract_ground_truth_from_problem(problem_data)
                    problem_text = problem_data.get('input', '')
                    
                    if not problem_text:
                        logger.warning(f"⚠️  Problem {i} has no 'input' field, skipping")
                        continue
                    
                    # Submit to executor
                    future = executor.submit(
                        self.solve_problem,
                        i,
                        problem_text,
                        ground_truth,
                        ground_truth_raw
                    )
                    futures.append(future)
                
                # Collect results
                for j, future in enumerate(futures, 1):
                    try:
                        result = future.result(timeout=300)  # 5 min timeout
                        results.append(result)
                        
                        # Progress logging
                        if j % 10 == 0:
                            logger.info(f"✅ Processed: {j}/{len(futures)}")
                    
                    except Exception as e:
                        logger.error(f"❌ Problem failed: {e}")
        
        else:
            # Sequential processing
            for i, problem_data in enumerate(problems, 1):
                logger.info(f"Processing {i}/{len(problems)}...")
                
                # Extract ground truth
                ground_truth, ground_truth_raw, _ = extract_ground_truth_from_problem(problem_data)
                problem_text = problem_data.get('input', '')
                
                if not problem_text:
                    logger.warning(f"⚠️  Problem {i} has no 'input' field, skipping")
                    continue
                
                result = self.solve_problem(i, problem_text, ground_truth, ground_truth_raw)
                results.append(result)
        
        return results
    
    def calculate_statistics(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Calculate accuracy and other statistics."""
        total = len(results)
        correct = sum(1 for r in results if r.is_correct)
        incorrect = sum(1 for r in results if not r.is_correct and r.error is None)
        errors = sum(1 for r in results if r.error is not None)
        no_answer = sum(1 for r in results if r.llm_answer is None and r.error is None)
        
        accuracy = (correct / total * 100) if total > 0 else 0.0
        avg_time = sum(r.processing_time for r in results) / total if total > 0 else 0.0
        
        return {
            'total_problems': total,
            'correct': correct,
            'incorrect': incorrect,
            'errors': errors,
            'no_answer': no_answer,
            'accuracy': accuracy,
            'avg_processing_time': avg_time
        }
    
    def print_statistics(self, stats: Dict[str, Any]):
        """Print benchmark statistics."""
        logger.info("")
        logger.info("=" * 70)
        logger.info("📊 BENCHMARK RESULTS")
        logger.info("=" * 70)
        logger.info(f"🤖 Model: {self.model_name}")
        logger.info(f"📝 Total Problems: {stats['total_problems']}")
        logger.info(f"✅ Correct: {stats['correct']}")
        logger.info(f"❌ Incorrect: {stats['incorrect']}")
        logger.info(f"🤷 No Answer: {stats['no_answer']}")
        logger.info(f"⚠️  Errors: {stats['errors']}")
        logger.info("")
        logger.info(f"🎯 ACCURACY: {stats['accuracy']:.2f}%")
        logger.info(f"⏱️  Avg Time: {stats['avg_processing_time']:.2f}s")
        logger.info("=" * 70)
    
    def save_results(self, results: List[BenchmarkResult], stats: Dict[str, Any], output_file: str):
        """Save results to JSON file."""
        output_data = {
            'metadata': {
                'model': self.model_name,
                'temperature': self.temperature,
                'timestamp': datetime.now().isoformat(),
                'total_problems': stats['total_problems']
            },
            'statistics': stats,
            'results': [
                {
                    'problem_id': r.problem_id,
                    'problem_text': r.problem_text[:200] + '...' if len(r.problem_text) > 200 else r.problem_text,
                    'ground_truth': r.ground_truth,
                    'ground_truth_raw': r.ground_truth_raw,
                    'llm_answer': r.llm_answer,
                    'is_correct': r.is_correct,
                    'processing_time': r.processing_time,
                    'error': r.error
                }
                for r in results
            ]
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"💾 Results saved to: {output_file}")


def load_dataset(input_file: str) -> List[Dict[str, Any]]:
    """Load dataset from JSONL file."""
    path = Path(input_file)
    
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    if path.suffix == '.jsonl':
        # JSONL format (one JSON object per line)
        problems = []
        with open(path, 'r') as f:
            for line in f:
                if line.strip():
                    problems.append(json.loads(line))
        return problems
    
    elif path.suffix == '.json':
        # JSON array format
        with open(path, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            else:
                return [data]
    
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}. Use .json or .jsonl")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark bare LLM performance without preprocessing pipeline"
    )
    
    parser.add_argument('--input', required=True,
                       help='Input dataset file (.json or .jsonl)')
    parser.add_argument('--output', default='bare_llm_results.json',
                       help='Output results file (default: bare_llm_results.json)')
    parser.add_argument('--batch-size', type=int, default=10,
                       help='Batch size for parallel processing (default: 10)')
    parser.add_argument('--no-parallel', action='store_true',
                       help='Disable parallel processing')
    parser.add_argument('--model', type=str, default=None,
                       help='Override model name (default: from .env APPRENTICE_MODEL)')
    parser.add_argument('--temperature', type=float, default=0.1,
                       help='Temperature for LLM (default: 0.1)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of problems to process (for testing)')
    
    args = parser.parse_args()
    
    # Load dataset
    try:
        logger.info(f"📂 Loading dataset from: {args.input}")
        problems = load_dataset(args.input)
        logger.info(f"✅ Loaded {len(problems)} problems")
        
        # Apply limit if specified
        if args.limit:
            problems = problems[:args.limit]
            logger.info(f"🔬 Limited to first {args.limit} problems for testing")
        
    except Exception as e:
        logger.error(f"❌ Failed to load dataset: {e}")
        return 1
    
    # Initialize benchmark
    benchmark = BareLLMBenchmark(
        model_name=args.model,
        temperature=args.temperature
    )
    
    # Run benchmark
    try:
        start_time = time.time()
        
        results = benchmark.benchmark_dataset(
            problems,
            batch_size=args.batch_size,
            use_parallel=not args.no_parallel
        )
        
        total_time = time.time() - start_time
        
        # Calculate statistics
        stats = benchmark.calculate_statistics(results)
        stats['total_time'] = total_time
        
        # Print results
        benchmark.print_statistics(stats)
        
        # Save results
        benchmark.save_results(results, stats, args.output)
        
        logger.info("")
        logger.info("✅ BENCHMARK COMPLETE")
        logger.info(f"⏱️  Total time: {total_time:.2f}s")
        
        return 0
    
    except Exception as e:
        logger.error(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
