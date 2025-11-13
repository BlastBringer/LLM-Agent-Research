#!/usr/bin/env python3
"""
Evaluate reasoning quality from existing results JSON using DeepEval
Takes a results JSON file and evaluates the final_reasoning field
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase


def evaluate_reasoning_deepeval(results_file: Path, output_file: Path = None):
    """
    Evaluate reasoning from a results JSON file using DeepEval
    
    Args:
        results_file: Path to JSON file with results (must have 'question' and 'final_reasoning' fields)
        output_file: Optional path to save results with DeepEval metrics
    """
    
    print(f"\n{'='*60}")
    print("DeepEval Reasoning Evaluation")
    print(f"{'='*60}\n")
    
    # Load results
    print(f"Loading results from {results_file}...")
    with open(results_file) as f:
        data = json.load(f)
    
    results = data.get('results', [])
    print(f"✓ Loaded {len(results)} results\n")
    
    # Create test cases
    test_cases = []
    test_case_indices = []
    
    for i, result in enumerate(results):
        if result.get('final_reasoning') and result.get('question'):
            test_case = LLMTestCase(
                input=result['question'],
                actual_output=result['final_reasoning']
            )
            test_cases.append(test_case)
            test_case_indices.append(i)
    
    print(f"Created {len(test_cases)} test cases for evaluation\n")
    
    if not test_cases:
        print("⚠️ No test cases found with reasoning!")
        return
    
    try:
        print("Evaluating Answer Relevancy (with timeout protection)...\n")
        
        # Create metric
        answer_relevancy_metric = AnswerRelevancyMetric(
            threshold=0.7,
            model="gpt-3.5-turbo",
            async_mode=False,
            include_reason=False
        )
        
        answer_scores = []
        successful = 0
        failed = 0
        
        # Process with thread pool for timeout handling
        with ThreadPoolExecutor(max_workers=1) as executor:
            for i, test_case in enumerate(tqdm(test_cases, desc="DeepEval")):
                try:
                    def evaluate_test_case(tc, metric):
                        metric.measure(tc)
                        return metric.score
                    
                    future = executor.submit(evaluate_test_case, test_case, answer_relevancy_metric)
                    score = future.result(timeout=60)
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
        
        print(f"\n{'='*60}")
        print("DEEPEVAL EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"Total test cases:        {len(test_cases)}")
        print(f"Successful:              {successful}/{len(test_cases)}")
        print(f"Failed/Timeout:          {failed}/{len(test_cases)}")
        print(f"\nAverage Relevancy Score: {avg_answer_relevancy:.4f}")
        print(f"{'='*60}\n")
        
        # Add metrics to results if output file specified
        if output_file:
            data['deepeval_metrics'] = {
                'answer_relevancy': {
                    'average_score': float(avg_answer_relevancy),
                    'successful_evaluations': successful,
                    'failed_evaluations': failed,
                    'total_evaluations': len(test_cases),
                    'individual_scores': [float(s) for s in answer_scores]
                }
            }
            
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"✓ Results with DeepEval metrics saved to: {output_file}\n")
        
        # Also show score distribution
        if answer_scores:
            high = sum(1 for s in answer_scores if s > 0.8)
            medium = sum(1 for s in answer_scores if 0.5 <= s <= 0.8)
            low = sum(1 for s in answer_scores if s < 0.5)
            
            print(f"Score Distribution:")
            print(f"  High (>0.8):    {high} ({100*high/len(answer_scores):.1f}%)")
            print(f"  Medium (0.5-0.8): {medium} ({100*medium/len(answer_scores):.1f}%)")
            print(f"  Low (<0.5):     {low} ({100*low/len(answer_scores):.1f}%)")
    
    except Exception as e:
        print(f"❌ Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description='Evaluate reasoning quality using DeepEval')
    parser.add_argument('--input', type=str, required=True, help='Input JSON results file')
    parser.add_argument('--output', type=str, help='Output JSON file with DeepEval metrics (optional)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file {input_path} not found")
        return
    
    output_path = Path(args.output) if args.output else None
    
    evaluate_reasoning_deepeval(input_path, output_path)


if __name__ == "__main__":
    main()
