#!/usr/bin/env python3
"""
🚀 COMPLETE MATH PROBLEM SOLVER PIPELINE
=========================================

This is the complete end-to-end pipeline integrating:
1. Templatization (remove names)
2. Parsing (extract equations)
3. Variable Extraction (get values)
4. Unit Standardization (convert to SI)
5. Solving (Apprentice → Verifier → Oracle)

TWO MODES:
----------
1. SINGLE MODE: Solve one problem from input file, write detailed output
2. DATASET MODE: Batch process many problems, save all to training data

Usage:
    # Single problem
    python3 main_pipeline.py --mode single --input problem.txt --output solution.txt
    
    # Dataset batch processing
    python3 main_pipeline.py --mode dataset --input problems.jsonl --batch-size 10
"""

import sys
import os
import json
import logging
import argparse
from typing import Dict, Any, List
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import time

# Add modules to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Reasoning'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Solver'))

# Import all components
from Reasoning.templatizer import WordProblemTemplatizer
from Reasoning.parser import MathematicalProblemParser
from Reasoning.variable_extractor import VariableExtractor
from Reasoning.unit_standardizer import UnitStandardizer
from Solver.solver_agent import SolverAgent
from ground_truth_utils import extract_ground_truth_from_problem

# For parallel processing
try:
    from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
    import multiprocessing
    PARALLEL_AVAILABLE = True
except ImportError:
    PARALLEL_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CompletePipeline:
    """
    Complete end-to-end pipeline from raw problem text to solution.
    """
    
    def __init__(self, verbose: bool = True, disable_unit_standardization: bool = False):
        """Initialize all pipeline components."""
        self.verbose = verbose
        self.disable_unit_standardization = disable_unit_standardization
        
        if verbose:
            logger.info("=" * 70)
            logger.info("🚀 INITIALIZING COMPLETE PIPELINE")
            logger.info("=" * 70)
        
        # Stage 1: Preprocessing
        self.templatizer = WordProblemTemplatizer()
        self.parser = MathematicalProblemParser()
        self.variable_extractor = VariableExtractor()
        self.unit_standardizer = UnitStandardizer()
        
        # Stage 2: Solving
        self.solver = SolverAgent()
        
        if verbose:
            logger.info("✅ All components initialized")
            logger.info("")
    
    def solve_single_problem(
        self,
        problem_text: str,
        show_verbose: bool = True,
        skip_oracle: bool = False,
        problem_metadata: dict = None
    ) -> Dict[str, Any]:
        """
        Solve a single problem with full verbosity and detailed output.
        
        VALIDATION STRATEGY (per architecture diagram):
        - SINGLE MODE: Uses SymPy verifier to check apprentice's answer
        - This ensures the verifier logic is tested and used as designed
        
        Args:
            problem_text: The word problem as text
            show_verbose: Whether to show detailed progress
            skip_oracle: If True, use apprentice-only mode (for testing)
            problem_metadata: Full problem data dict with ground truth (optional)
        
        Returns:
            Complete results dictionary
        """
        start_time = time.time()
        
        if show_verbose:
            logger.info("=" * 70)
            logger.info("🔵 SINGLE PROBLEM MODE (Validation: SymPy Verifier)")
            logger.info("=" * 70)
            logger.info(f"📝 Problem: {problem_text[:100]}...")
            logger.info("")
        
        # Stage 1: Templatization
        if show_verbose:
            logger.info("=" * 70)
            logger.info("📝 STAGE 1: TEMPLATIZATION")
            logger.info("=" * 70)
        
        templatization_result = self.templatizer.templatize_problem(problem_text)
        
        if show_verbose:
            logger.info(f"✅ Templated: {templatization_result.templated_problem[:80]}...")
            logger.info(f"📋 Legend entries: {len(templatization_result.legend)}")
        
        # Stage 2: Parsing
        if show_verbose:
            logger.info("")
            logger.info("=" * 70)
            logger.info("📐 STAGE 2: PARSING")
            logger.info("=" * 70)
        
        parse_result = self.parser.parse_problem(templatization_result)
        
        if show_verbose:
            logger.info(f"✅ Problem Type: {parse_result.problem_type}")
            logger.info(f"📊 Equations: {len(parse_result.equations)}")
            logger.info(f"🎯 Target: {parse_result.target_variable}")
        
        # Stage 3: Variable Extraction
        if show_verbose:
            logger.info("")
            logger.info("=" * 70)
            logger.info("🔢 STAGE 3: VARIABLE EXTRACTION")
            logger.info("=" * 70)
        
        extraction_result = self.variable_extractor.extract_variables(
            templatization_result.templated_problem,
            parse_result.all_variables,
            [eq.equation_string for eq in parse_result.equations]
        )
        
        if show_verbose:
            logger.info(f"✅ Variables extracted: {len(extraction_result.variables)}")
        
        # Stage 4: Unit Standardization (optional)
        if not self.disable_unit_standardization:
            if show_verbose:
                logger.info("")
                logger.info("=" * 70)
                logger.info("⚖️ STAGE 4: UNIT STANDARDIZATION")
                logger.info("=" * 70)
            
            standardization_result = self.unit_standardizer.standardize_variables(
                extraction_result.variables
            )
            
            if show_verbose:
                logger.info(f"✅ Unit system: {standardization_result.unit_system}")
                logger.info(f"📊 Conversions: {len(standardization_result.conversions_applied)}")
        else:
            standardization_result = None
        
        # Stage 5: Solving
        if show_verbose:
            logger.info("")
            logger.info("=" * 70)
            logger.info("🧠 STAGE 5: SOLVING")
            logger.info("=" * 70)
        
        # Prepare problem data for solver
        problem_data = {
            'original_problem': problem_text,
            'templatization': {
                'templated_problem': templatization_result.templated_problem,
                'legend': templatization_result.legend
            },
            'parsing': {
                'equations': [asdict(eq) for eq in parse_result.equations],
                'target_variable': parse_result.target_variable,
                'all_variables': parse_result.all_variables
            }
        }
        # Include unit standardization if enabled
        if standardization_result is not None:
            problem_data['unit_standardization'] = {
                'standardized_variables': {
                    k: asdict(v) for k, v in standardization_result.standardized_variables.items()
                }
            }
        else:
            # Provide variable extraction to solver when standardization is disabled
            problem_data['variable_extraction'] = {
                'variables': {k: asdict(v) for k, v in extraction_result.variables.items()}
            }
        
        # Add ground truth metadata if available
        if problem_metadata:
            problem_data['metadata'] = problem_metadata
        
        # Solve!
        if skip_oracle:
            # TEST MODE: Apprentice-only evaluation
            if show_verbose:
                logger.info("")
                logger.info("=" * 70)
                logger.info("🧪 APPRENTICE-ONLY MODE (Oracle Disabled)")
                logger.info("=" * 70)
            solve_result = self.solver.solve_apprentice_only(problem_data, verbose=show_verbose)
        else:
            # TRAIN MODE: Full pipeline with Oracle fallback
            # IMPORTANT: For single mode, always use verifier (not ground truth)
            solve_result = self.solver.solve(problem_data, verbose=show_verbose, use_ground_truth=False)
        
        # Stage 6: Convert answer back to original units (only if standardization enabled)
        if not self.disable_unit_standardization and standardization_result is not None:
            if show_verbose:
                logger.info("")
                logger.info("=" * 70)
                logger.info("🔄 STAGE 6: UNIT CONVERSION (Back to Original)")
                logger.info("=" * 70)
            final_answer_converted, final_unit = self.unit_standardizer.convert_answer_to_original_units(
                answer_value=solve_result.final_answer,
                target_variable=parse_result.target_variable,
                standardization_result=standardization_result
            )
            if show_verbose:
                logger.info(f"✅ Answer in SI units: {solve_result.final_answer}")
                logger.info(f"✅ Answer in original units: {final_answer_converted} {final_unit}")
        else:
            final_answer_converted = solve_result.final_answer
            final_unit = 'unknown'
        
        # Calculate total time
        total_time = time.time() - start_time
        
        if show_verbose:
            logger.info("")
            logger.info("=" * 70)
            logger.info("✅ PIPELINE COMPLETE")
            logger.info("=" * 70)
            logger.info(f"🎯 Final Answer: {final_answer_converted} {final_unit}")
            logger.info(f"✅ Verification: {'CORRECT' if solve_result.is_correct else 'NEEDS REVIEW'}")
            logger.info(f"🧠 Solver Used: {solve_result.solver_used}")
            logger.info(f"⏱️  Total Time: {total_time:.2f}s")
            logger.info("")
        
        # Return complete results
        return {
            'problem': problem_text,
            'stages': {
                'templatization': asdict(templatization_result),
                'parsing': asdict(parse_result),
                'variable_extraction': asdict(extraction_result),
                'unit_standardization': asdict(standardization_result) if standardization_result is not None else {
                    'standardized_variables': {}
                }
            },
            'solution': {
                'final_answer_si': solve_result.final_answer,  # Answer in SI units
                'final_answer': final_answer_converted,  # Answer in original units
                'final_unit': final_unit,  # Unit of the final answer
                'is_correct': solve_result.is_correct,
                'solver_used': solve_result.solver_used,
                'confidence': solve_result.confidence,
                'reasoning_steps': solve_result.apprentice_solution.reasoning_steps if solve_result.apprentice_solution else [],
                'oracle_steps': solve_result.oracle_solution.reasoning_steps if solve_result.oracle_solution else []
            },
            'metadata': {
                'processing_time': total_time,
                'timestamp': datetime.now().isoformat()
            }
        }
    
    def solve_dataset(
        self,
        problems: List[str],
        batch_size: int = 10,
        use_parallel: bool = True,
        skip_oracle: bool = False
    ) -> Dict[str, Any]:
        """
        Batch process multiple problems from a dataset.
        Uses parallel processing for speed.
        
        VALIDATION STRATEGY (per architecture diagram):
        - DATASET MODE: Uses ground truth from 'output' column for validation
        - Skips SymPy verifier since we have correct answers
        - This is more reliable and faster for training data collection
        
        Args:
            problems: List of problem texts (or dicts with 'text' and 'metadata')
            batch_size: Number of problems to process in parallel
            use_parallel: Whether to use parallel processing
            skip_oracle: If True, only use apprentice (for testing/evaluation)
        
        Returns:
            Summary statistics
        """
        logger.info("=" * 70)
        mode_desc = "Apprentice-Only Testing" if skip_oracle else "Training with Oracle Fallback"
        logger.info(f"🔵 DATASET MODE ({mode_desc} | Validation: Ground Truth)")
        logger.info("=" * 70)
        logger.info(f"📊 Total problems: {len(problems)}")
        logger.info(f"⚡ Batch size: {batch_size}")
        logger.info(f"🔧 Parallel: {use_parallel and PARALLEL_AVAILABLE}")
        logger.info("")
        
        # Suppress verbose logging from sub-modules during batch processing
        # Save original log levels
        original_levels = {}
        modules_to_quiet = [
            'Reasoning.templatizer',
            'Reasoning.parser', 
            'Reasoning.variable_extractor',
            'Reasoning.unit_standardizer',
            'Solver.apprentice',
            'Solver.oracle',
            'Solver.verifier',
            'Solver.solver_agent',
            'httpx'
        ]
        
        for module_name in modules_to_quiet:
            module_logger = logging.getLogger(module_name)
            original_levels[module_name] = module_logger.level
            module_logger.setLevel(logging.WARNING)  # Only show warnings and errors
        
        start_time = time.time()
        results = []
        
        if use_parallel and PARALLEL_AVAILABLE:
            # Parallel processing
            max_workers = min(batch_size, multiprocessing.cpu_count())
            logger.info(f"🚀 Using {max_workers} workers")
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for i, problem in enumerate(problems):
                    # Handle both dict format (with metadata) and string format
                    if isinstance(problem, dict):
                        problem_text = problem['text']
                        problem_metadata = problem.get('metadata')
                    else:
                        problem_text = problem
                        problem_metadata = None
                    
                    future = executor.submit(self._solve_single_quiet, problem_text, i+1, problem_metadata, skip_oracle)
                    futures.append(future)
                
                # Collect results with progress updates
                for i, future in enumerate(futures, 1):
                    try:
                        result = future.result(timeout=300)  # 5 min timeout per problem
                        results.append(result)
                        
                        # Show progress every problem
                        logger.info(f"✅ Progress: {i}/{len(problems)} ({(i/len(problems)*100):.1f}%)")
                    except Exception as e:
                        logger.error(f"❌ Problem failed: {e}")
                        results.append({'error': str(e)})
        else:
            # Sequential processing
            for i, problem in enumerate(problems, 1):
                try:
                    # Handle both dict format (with metadata) and string format
                    if isinstance(problem, dict):
                        problem_text = problem['text']
                        problem_metadata = problem.get('metadata')
                    else:
                        problem_text = problem
                        problem_metadata = None
                    
                    result = self._solve_single_quiet(problem_text, i, problem_metadata, skip_oracle)
                    results.append(result)
                    logger.info(f"✅ Progress: {i}/{len(problems)} ({(i/len(problems)*100):.1f}%)")
                except Exception as e:
                    logger.error(f"❌ Problem {i} failed: {e}")
                    results.append({'error': str(e)})
        
        total_time = time.time() - start_time
        
        # Restore original logging levels
        for module_name, level in original_levels.items():
            logging.getLogger(module_name).setLevel(level)
        
        # Calculate statistics
        successful = sum(1 for r in results if 'error' not in r)
        failed = len(results) - successful
        
        avg_time = total_time / len(problems) if problems else 0
        
        logger.info("")
        logger.info("=" * 70)
        logger.info("📊 DATASET PROCESSING COMPLETE")
        logger.info("=" * 70)
        logger.info(f"✅ Successful: {successful}/{len(problems)}")
        logger.info(f"❌ Failed: {failed}")
        logger.info(f"⏱️  Total Time: {total_time:.2f}s")
        logger.info(f"⚡ Avg Time/Problem: {avg_time:.2f}s")
        logger.info(f"💾 Training data saved to: {self.solver.training_data_file}")
        logger.info("")
        
        # Print solver statistics
        self.solver.print_statistics()
        
        return {
            'total_problems': len(problems),
            'successful': successful,
            'failed': failed,
            'total_time': total_time,
            'avg_time_per_problem': avg_time,
            'training_data_file': self.solver.training_data_file,
            'results': results
        }
    
    def _solve_single_quiet(self, problem_text: str, problem_num: int, problem_metadata: dict = None, skip_oracle: bool = False) -> Dict[str, Any]:
        """
        Solve a single problem without verbose output (for batch/dataset mode).
        
        This uses GROUND TRUTH validation (not SymPy verifier) since it's for dataset mode.
        
        Args:
            problem_text: The problem text
            problem_num: Problem number for tracking
            problem_metadata: Full problem data with ground truth
            skip_oracle: If True, only use apprentice (for testing)
        """
        try:
            # Quick processing without verbose logs
            templatization_result = self.templatizer.templatize_problem(problem_text)
            parse_result = self.parser.parse_problem(templatization_result)
            extraction_result = self.variable_extractor.extract_variables(
                templatization_result.templated_problem,
                parse_result.all_variables,
                [eq.equation_string for eq in parse_result.equations]
            )
            if not self.disable_unit_standardization:
                standardization_result = self.unit_standardizer.standardize_variables(
                    extraction_result.variables
                )
            else:
                standardization_result = None
            
            # Prepare for solver
            problem_data = {
                'original_problem': problem_text,
                'parsing': {
                    'equations': [asdict(eq) for eq in parse_result.equations],
                    'target_variable': parse_result.target_variable,
                    'all_variables': parse_result.all_variables
                },
            }
            if standardization_result is not None:
                problem_data['unit_standardization'] = {
                    'standardized_variables': {
                        k: asdict(v) for k, v in standardization_result.standardized_variables.items()
                    }
                }
            else:
                problem_data['variable_extraction'] = {
                    'variables': {k: asdict(v) for k, v in extraction_result.variables.items()}
                }
            
            
            # Add ground truth metadata if available
            if problem_metadata:
                problem_data['metadata'] = problem_metadata
            
            # Solve (non-verbose)
            # IMPORTANT: For dataset mode, always use ground truth validation (not verifier)
            if skip_oracle:
                # TEST MODE: Apprentice-only evaluation with ground truth
                solve_result = self.solver.solve_apprentice_only(problem_data, verbose=False, use_ground_truth=True)
            else:
                # TRAIN MODE: Full pipeline with Oracle fallback
                solve_result = self.solver.solve(problem_data, verbose=False, use_ground_truth=True)
            
            # Convert answer back to original units
            if standardization_result is not None:
                final_answer_converted, final_unit = self.unit_standardizer.convert_answer_to_original_units(
                    answer_value=solve_result.final_answer,
                    target_variable=parse_result.target_variable,
                    standardization_result=standardization_result
                )
            else:
                final_answer_converted, final_unit = solve_result.final_answer, 'unknown'
            
            # Extract ground truth (from top-level or metadata)
            gt_numeric, gt_raw, gt_unit = extract_ground_truth_from_problem(problem_data)
            
            # Build enriched result payload for JSONL
            return {
                'problem_num': problem_num,
                'input': problem_text,
                'ground_truth_raw': gt_raw,
                'ground_truth_numeric': gt_numeric,
                'ground_truth_unit': gt_unit,
                'answer_si': solve_result.final_answer,  # Answer in SI units
                'answer': final_answer_converted,  # Answer in original units
                'unit': final_unit,
                'correct': solve_result.is_correct,
                'verification_method': getattr(solve_result.verification, 'verification_method', 'unknown'),
                'difference': getattr(solve_result.verification, 'difference', None),
                'solver': solve_result.solver_used,
                'apprentice_confidence': solve_result.confidence,
                'apprentice_reasoning_steps': (solve_result.apprentice_solution.reasoning_steps if solve_result.apprentice_solution else []),
                'apprentice_raw_response': (solve_result.apprentice_solution.raw_response if solve_result.apprentice_solution else None)
            }
        except Exception as e:
            logger.error(f"Problem {problem_num} error: {e}")
            return {'problem_num': problem_num, 'error': str(e)}


def load_input(input_path: str) -> List[str]:
    """Load problem(s) from input file."""
    path = Path(input_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    if path.suffix == '.txt':
        # Single problem from text file
        with open(path, 'r') as f:
            content = f.read().strip()
            return [content]
    
    elif path.suffix == '.json':
        # Single or multiple problems from JSON
        with open(path, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            else:
                return [data.get('problem', str(data))]
    
    elif path.suffix == '.jsonl':
        # Multiple problems from JSONL
        problems = []
        with open(path, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    # Extract problem text (try different field names)
                    problem_text = data.get('input') or data.get('problem') or data.get('question') or data.get('text')
                    if problem_text:
                        # Store both text and full metadata for ground truth
                        problems.append({
                            'text': problem_text,
                            'metadata': data  # Keep full data for ground truth extraction
                        })
        return problems
    
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def save_output(results: Dict[str, Any], output_path: str):
    """Save results to output file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        # Write formatted output
        f.write("=" * 70 + "\n")
        f.write("MATH PROBLEM SOLUTION\n")
        f.write("=" * 70 + "\n\n")
        
        # Problem
        f.write("PROBLEM:\n")
        f.write(results['problem'] + "\n\n")
        
        # Solution
        f.write("=" * 70 + "\n")
        f.write("SOLUTION:\n")
        f.write("=" * 70 + "\n\n")
        
        solution = results['solution']
        
        # Format answer with unit
        answer_str = f"{solution['final_answer']}"
        if 'final_unit' in solution and solution['final_unit'] != 'unknown':
            answer_str += f" {solution['final_unit']}"
        
        f.write(f"Final Answer: {answer_str}\n")
        f.write(f"Verification: {'✅ CORRECT' if solution['is_correct'] else '⚠️ NEEDS REVIEW'}\n")
        f.write(f"Solver Used: {solution['solver_used']}\n")
        f.write(f"Confidence: {solution['confidence']:.2%}\n\n")
        
        # Reasoning steps
        if solution['reasoning_steps']:
            f.write("REASONING STEPS:\n")
            f.write("-" * 70 + "\n")
            for i, step in enumerate(solution['reasoning_steps'], 1):
                f.write(f"{i}. {step}\n")
            f.write("\n")
        
        # Oracle steps if used
        if solution['oracle_steps']:
            f.write("ORACLE (TEACHER) SOLUTION:\n")
            f.write("-" * 70 + "\n")
            for i, step in enumerate(solution['oracle_steps'], 1):
                f.write(f"{i}. {step}\n")
            f.write("\n")
        
        # Metadata
        f.write("=" * 70 + "\n")
        f.write("METADATA:\n")
        f.write("=" * 70 + "\n")
        f.write(f"Processing Time: {results['metadata']['processing_time']:.2f}s\n")
        f.write(f"Timestamp: {results['metadata']['timestamp']}\n")
    
    logger.info(f"💾 Output saved to: {output_path}")


def main():
    """Main entry point with command-line interface."""
    parser = argparse.ArgumentParser(
        description='Complete Math Problem Solver Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single problem mode
  python3 main_pipeline.py --mode single --input problem.txt --output solution.txt
  
  # Dataset mode (batch processing)
  python3 main_pipeline.py --mode dataset --input problems.jsonl --batch-size 20
        """
    )
    
    parser.add_argument('--mode', choices=['single', 'dataset'], required=True,
                       help='Processing mode')
    parser.add_argument('--input', required=True,
                       help='Input file (txt/json for single, jsonl for dataset)')
    parser.add_argument('--output', default='solution.txt',
                       help='Output file for single mode (default: solution.txt)')
    parser.add_argument('--batch-size', type=int, default=10,
                       help='Batch size for parallel processing in dataset mode')
    parser.add_argument('--no-parallel', action='store_true',
                       help='Disable parallel processing')
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce verbosity')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of problems to process (for testing)')
    
    # NEW: Pipeline mode for train/test/eval
    parser.add_argument('--pipeline-mode', type=str, default='train',
                       choices=['train', 'test', 'eval'],
                       help='Pipeline mode: train (collect oracle data), test (apprentice only), eval (compare both)')
    parser.add_argument('--skip-oracle', action='store_true',
                       help='Skip oracle calls (for testing apprentice only)')
    parser.add_argument('--no-unit-standardization', action='store_true',
                       help='Disable unit standardization stage (use raw variable extraction)')
    
    args = parser.parse_args()
    
    # Set pipeline mode behavior
    if args.pipeline_mode == 'test':
        logger.info("🧪 TEST MODE: Apprentice-only evaluation (no Oracle calls)")
        args.skip_oracle = True
    elif args.pipeline_mode == 'eval':
        logger.info("📊 EVAL MODE: Comparing Apprentice vs Oracle")
    else:
        logger.info("🎓 TRAIN MODE: Collecting Oracle solutions for fine-tuning")
    
    
    # Load input
    try:
        problems = load_input(args.input)
        
        # Apply limit if specified
        if args.limit and args.limit > 0:
            problems = problems[:args.limit]
            logger.info(f"✅ Loaded {len(problems)} problem(s) from {args.input} (limited from dataset)")
        else:
            logger.info(f"✅ Loaded {len(problems)} problem(s) from {args.input}")
    except Exception as e:
        logger.error(f"❌ Failed to load input: {e}")
        return 1
    
    # Initialize pipeline
    pipeline = CompletePipeline(verbose=not args.quiet, disable_unit_standardization=args.no_unit_standardization)
    
    try:
        if args.mode == 'single':
            # Single problem mode
            if len(problems) != 1:
                logger.warning(f"Multiple problems found, using only the first one")
            
            # Handle both dict format and string format
            if isinstance(problems[0], dict):
                problem_text = problems[0]['text']
                problem_metadata = problems[0].get('metadata')
            else:
                problem_text = problems[0]
                problem_metadata = None
            
            result = pipeline.solve_single_problem(problem_text, show_verbose=not args.quiet, 
                                                   skip_oracle=args.skip_oracle,
                                                   problem_metadata=problem_metadata)
            save_output(result, args.output)
            
            logger.info("=" * 70)
            logger.info("✅ SINGLE PROBLEM MODE COMPLETE")
            logger.info("=" * 70)
            
        else:
            # Dataset mode
            stats = pipeline.solve_dataset(
                problems,
                batch_size=args.batch_size,
                use_parallel=not args.no_parallel,
                skip_oracle=args.skip_oracle
            )
            
            # Save results to JSONL if specified
            if args.output and args.output.endswith('.jsonl'):
                logger.info(f"💾 Saving results to: {args.output}")
                import json
                with open(args.output, 'w') as f:
                    for result in stats.get('results', []):
                        f.write(json.dumps(result) + '\n')
                logger.info(f"✅ Results saved: {len(stats.get('results', []))} problems to {args.output}")
            
            logger.info("=" * 70)
            logger.info("✅ DATASET MODE COMPLETE")
            logger.info("=" * 70)
            logger.info(f"📊 Results:")
            logger.info(f"   Processed: {stats['successful']}/{stats['total_problems']}")
            logger.info(f"   Training examples: {pipeline.solver.get_training_data_count()}")
        
        return 0
    
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
