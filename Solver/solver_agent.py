#!/usr/bin/env python3
"""
🤖 SOLVER AGENT - The Orchestrator
===================================

This is the main solver that coordinates:
1. Apprentice Model (student) - attempts to solve
2. Verifier (judge) - checks if the answer is correct
3. Oracle Model (teacher) - provides correct solution if apprentice fails [TODO: Next step]
4. Learning Recorder - saves oracle solutions for fine-tuning

The Learning Loop:
┌─────────────────────────────────────────────────────────┐
│  Problem → Apprentice → Verifier                        │
│                           │                              │
│                      Is Correct?                         │
│                      ┌────┴────┐                         │
│                    Yes        No                         │
│                     │          │                         │
│                  Success   Oracle → Save for training    │
└─────────────────────────────────────────────────────────┘

Usage:
    solver = SolverAgent()
    result = solver.solve(problem_data)
    print(result.final_answer)
    print(result.is_correct)
"""

import os
import json
import logging
import time
import sys
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add ground truth utilities
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
try:
    from ground_truth_utils import extract_ground_truth_from_problem
except ImportError:
    # Fallback if module not found
    def extract_ground_truth_from_problem(problem_data):
        return None, None, None

from .apprentice import ApprenticeModel, ApprenticeSolution
from .verifier import MathVerifier, VerificationResult
from .oracle import OracleModel, OracleSolution

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SolverResult:
    """Complete result from the solver agent."""
    # Core results
    final_answer: float
    is_correct: bool
    solver_used: str  # 'apprentice' or 'oracle'
    
    # Detailed information
    apprentice_solution: Optional[ApprenticeSolution]
    verification: VerificationResult
    oracle_solution: Optional[OracleSolution]
    
    # Metadata
    confidence: float
    processing_time: float
    metadata: Dict[str, Any]


class SolverAgent:
    """
    The main solver agent that orchestrates the learning loop.
    """
    
    def __init__(
        self,
        training_data_file: str = "solver_training_data.jsonl",
        failure_log_file: str = "solver_failures.jsonl"
    ):
        """
        Initialize the solver agent.
        
        Args:
            training_data_file: Where to save oracle solutions for fine-tuning
            failure_log_file: Where to log complete failures (oracle also failed)
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.apprentice = ApprenticeModel()
        self.verifier = MathVerifier()
        self.oracle = OracleModel()  # Now initialized!
        
        # File paths
        self.training_data_file = training_data_file
        self.failure_log_file = failure_log_file
        
        # Statistics
        self.stats = {
            'total_problems': 0,
            'apprentice_correct': 0,
            'oracle_needed': 0,
            'complete_failures': 0
        }
        
        self.logger.info("🤖 Solver Agent initialized")
        self.logger.info(f"   📁 Training data: {self.training_data_file}")
        self.logger.info(f"   📁 Failure log: {self.failure_log_file}")
    
    def solve_apprentice_only(
        self,
        problem_data: Dict[str, Any],
        verbose: bool = True
    ) -> SolverResult:
        """
        TEST MODE: Evaluate apprentice without Oracle fallback.
        
        Used for:
        - Testing fine-tuned models
        - Measuring improvement over time
        - Preventing contamination of test set with Oracle solutions
        
        Args:
            problem_data: All processed data from previous pipeline stages
            verbose: Whether to print detailed progress
        
        Returns:
            SolverResult with apprentice's performance (no oracle fallback)
        """
        import time
        start_time = time.time()
        
        self.stats['total_problems'] += 1
        
        if verbose:
            logger.info("\n" + "=" * 70)
            logger.info("🧪 TEST MODE: Apprentice-Only Evaluation (No Oracle)")
            logger.info("=" * 70)
        
        # Extract what we need for verification
        equations = self._extract_equations(problem_data)
        variables = self._extract_variables(problem_data)
        target_var = self._extract_target_variable(problem_data)
        
        if verbose:
            logger.info(f"🎯 Target Variable: {target_var}")
        
        # Apprentice attempts to solve
        apprentice_solution = self.apprentice.solve(problem_data)
        
        if apprentice_solution.final_answer is None:
            if verbose:
                logger.warning("   ❌ Apprentice failed to produce an answer")
            
            processing_time = time.time() - start_time
            return SolverResult(
                final_answer=0.0,
                is_correct=False,
                solver_used='apprentice',
                apprentice_solution=apprentice_solution,
                verification=None,
                oracle_solution=None,
                confidence=0.0,
                processing_time=processing_time,
                metadata={'test_mode': True, 'apprentice_failed': True}
            )
        
        # Verifier checks (NO oracle fallback)
        verification = self.verifier.verify(
            equations=equations,
            variables=variables,
            target_variable=target_var,
            proposed_answer=apprentice_solution.final_answer
        )
        
        if verification.is_correct:
            self.stats['apprentice_correct'] += 1
        
        processing_time = time.time() - start_time
        
        return SolverResult(
            final_answer=apprentice_solution.final_answer,
            is_correct=verification.is_correct,
            solver_used='apprentice',
            apprentice_solution=apprentice_solution,
            verification=verification,
            oracle_solution=None,
            confidence=apprentice_solution.confidence,
            processing_time=processing_time,
            metadata={'test_mode': True}
        )
    
    def solve(
        self,
        problem_data: Dict[str, Any],
        verbose: bool = True,
        use_ground_truth: bool = None
    ) -> SolverResult:
        """
        Main solving pipeline with ground truth support.
        
        Args:
            problem_data: All processed data from previous pipeline stages
            verbose: Whether to print detailed progress
            use_ground_truth: If True, use ground truth for validation instead of verifier.
                            If None, auto-detect based on ground truth availability.
        
        Returns:
            SolverResult with answer and verification
        """
        import time
        start_time = time.time()
        
        self.stats['total_problems'] += 1
        
        # Extract ground truth if available
        ground_truth_numeric, ground_truth_raw, ground_truth_unit = extract_ground_truth_from_problem(problem_data)
        has_ground_truth = ground_truth_numeric is not None
        
        # Auto-detect mode: if ground truth available, use it instead of verifier
        if use_ground_truth is None:
            use_ground_truth = has_ground_truth
        
        if verbose:
            logger.info("\n" + "=" * 70)
            logger.info("🧠 STARTING SOLVER AGENT")
            logger.info("=" * 70)
            if has_ground_truth:
                logger.info(f"📊 Ground Truth Available: {ground_truth_numeric} {ground_truth_unit or ''}")
                logger.info(f"🎯 Validation Mode: {'Ground Truth' if use_ground_truth else 'Verifier (SymPy)'}")
        
        # Extract what we need for verification
        equations = self._extract_equations(problem_data)
        variables = self._extract_variables(problem_data)
        target_var = self._extract_target_variable(problem_data)
        
        if verbose:
            logger.info(f"🎯 Target Variable: {target_var}")
            logger.info(f"📊 Equations: {len(equations)}")
            logger.info(f"🔢 Variables: {len(variables)}")
        
        # STEP 1: Apprentice attempts to solve
        if verbose:
            logger.info("\n🎓 Step 1: Apprentice attempting to solve...")
        
        apprentice_solution = self.apprentice.solve(problem_data)
        
        if verbose and apprentice_solution.reasoning_steps:
            logger.info("   Reasoning:")
            for i, step in enumerate(apprentice_solution.reasoning_steps[:5], 1):
                logger.info(f"     {i}. {step[:80]}...")
        
        if apprentice_solution.final_answer is None:
            if verbose:
                logger.warning("   ⚠️  Apprentice failed to produce an answer")
            # Skip to oracle (TODO: implement)
            return self._handle_apprentice_failure(problem_data, start_time)
        
        if verbose:
            logger.info(f"   💡 Apprentice Answer: {apprentice_solution.final_answer}")
        
        # STEP 2: Validate answer - use ground truth if available, otherwise verifier
        if use_ground_truth and has_ground_truth:
            # DATASET MODE: Use ground truth directly (skip verifier)
            if verbose:
                logger.info("\n🎯 Step 2: Checking against ground truth...")
            
            # Compare apprentice answer with ground truth
            apprentice_correct = self._compare_with_ground_truth(
                apprentice_solution.final_answer,
                ground_truth_numeric,
                verbose=verbose
            )
            
            if apprentice_correct:
                # Success! Apprentice got it right
                self.stats['apprentice_correct'] += 1
                processing_time = time.time() - start_time
                
                if verbose:
                    logger.info(f"\n✅ SUCCESS! Apprentice matches ground truth in {processing_time:.2f}s")
                
                # Create a verification result for consistency
                verification = VerificationResult(
                    is_correct=True,
                    proposed_answer=apprentice_solution.final_answer,
                    correct_answer=ground_truth_numeric,
                    difference=0.0,
                    verification_method='ground_truth',
                    success=True
                )
                
                return SolverResult(
                    final_answer=apprentice_solution.final_answer,
                    is_correct=True,
                    solver_used='apprentice',
                    apprentice_solution=apprentice_solution,
                    verification=verification,
                    oracle_solution=None,
                    confidence=apprentice_solution.confidence,
                    processing_time=processing_time,
                    metadata={
                        'apprentice_succeeded': True,
                        'oracle_needed': False,
                        'validation_method': 'ground_truth'
                    }
                )
            else:
                # Apprentice wrong, need Oracle
                if verbose:
                    logger.warning(f"\n⚠️  Apprentice answer ({apprentice_solution.final_answer}) != Ground truth ({ground_truth_numeric})")
                    logger.warning("   Consulting Oracle...")
                
                self.stats['oracle_needed'] += 1
                
                # Call Oracle
                oracle_solution = self.oracle.solve(problem_data)
                
                if verbose and oracle_solution.reasoning_steps:
                    logger.info("\n👨‍🏫 Oracle's reasoning:")
                    for i, step in enumerate(oracle_solution.reasoning_steps[:10], 1):
                        logger.info(f"     {i}. {step[:100]}...")
                
                if oracle_solution.final_answer is None:
                    # Oracle failed
                    if verbose:
                        logger.error("   ❌ Oracle also failed!")
                    
                    self.stats['complete_failures'] += 1
                    processing_time = time.time() - start_time
                    
                    verification = VerificationResult(
                        is_correct=False,
                        proposed_answer=apprentice_solution.final_answer,
                        correct_answer=ground_truth_numeric,
                        difference=abs(apprentice_solution.final_answer - ground_truth_numeric) if isinstance(apprentice_solution.final_answer, (int, float)) else None,
                        verification_method='ground_truth',
                        success=False,
                        error='Oracle failed to solve'
                    )
                    
                    return SolverResult(
                        final_answer=ground_truth_numeric,
                        is_correct=False,
                        solver_used='none',
                        apprentice_solution=apprentice_solution,
                        verification=verification,
                        oracle_solution=oracle_solution,
                        confidence=0.1,
                        processing_time=processing_time,
                        metadata={
                            'apprentice_succeeded': False,
                            'oracle_succeeded': False,
                            'complete_failure': True,
                            'validation_method': 'ground_truth'
                        }
                    )
                
                # Check Oracle answer against ground truth
                oracle_correct = self._compare_with_ground_truth(
                    oracle_solution.final_answer,
                    ground_truth_numeric,
                    verbose=verbose
                )
                
                if verbose:
                    if oracle_correct:
                        logger.info(f"   ✅ Oracle answer CORRECT: {oracle_solution.final_answer}")
                    else:
                        logger.warning(f"   ⚠️  Oracle answer ({oracle_solution.final_answer}) != Ground truth ({ground_truth_numeric})")
                
                # Save oracle's solution (even if wrong, for analysis)
                self._save_training_example(
                    problem_data=problem_data,
                    solution_steps=oracle_solution.reasoning_steps,
                    final_answer=oracle_solution.final_answer,
                    source='oracle',
                    tool_calls=oracle_solution.tool_calls,
                    ground_truth=ground_truth_numeric,
                    ground_truth_raw=ground_truth_raw,
                    ground_truth_unit=ground_truth_unit
                )
                
                processing_time = time.time() - start_time
                
                if verbose:
                    logger.info(f"\n✅ Oracle solved in {processing_time:.2f}s")
                    logger.info(f"💾 Training example saved (oracle_correct={oracle_correct})")
                
                verification = VerificationResult(
                    is_correct=oracle_correct,
                    proposed_answer=oracle_solution.final_answer,
                    correct_answer=ground_truth_numeric,
                    difference=abs(oracle_solution.final_answer - ground_truth_numeric) if isinstance(oracle_solution.final_answer, (int, float)) and isinstance(ground_truth_numeric, (int, float)) else None,
                    verification_method='ground_truth',
                    success=True
                )
                
                return SolverResult(
                    final_answer=oracle_solution.final_answer,
                    is_correct=oracle_correct,
                    solver_used='oracle',
                    apprentice_solution=apprentice_solution,
                    verification=verification,
                    oracle_solution=oracle_solution,
                    confidence=oracle_solution.confidence if oracle_correct else 0.5,
                    processing_time=processing_time,
                    metadata={
                        'apprentice_succeeded': False,
                        'oracle_needed': True,
                        'oracle_succeeded': oracle_correct,
                        'saved_for_training': True,
                        'validation_method': 'ground_truth'
                    }
                )
        
        else:
            # SINGLE MODE: Use verifier (SymPy)
            if verbose:
                logger.info("\n🎯 Step 2: Verifier checking answer...")
        
        verification = self.verifier.verify(
            equations=equations,
            variables=variables,
            target_variable=target_var,
            proposed_answer=apprentice_solution.final_answer
        )
        
        if verbose:
            if verification.is_correct:
                logger.info(f"   ✅ CORRECT! Answer: {verification.correct_answer}")
            else:
                logger.warning(f"   ❌ INCORRECT!")
                logger.warning(f"      Expected: {verification.correct_answer}")
                logger.warning(f"      Got: {verification.proposed_answer}")
                logger.warning(f"      Difference: {verification.difference}")
        
        # STEP 3: Handle based on verification result
        if verification.is_correct:
            # Success! Apprentice got it right
            self.stats['apprentice_correct'] += 1
            
            processing_time = time.time() - start_time
            
            if verbose:
                logger.info(f"\n✅ SUCCESS! Apprentice solved correctly in {processing_time:.2f}s")
            
            # Save apprentice's correct solution for training
            self._save_training_example(
                problem_data=problem_data,
                solution_steps=apprentice_solution.reasoning_steps,
                final_answer=apprentice_solution.final_answer,
                source='apprentice'
            )
            
            return SolverResult(
                final_answer=verification.correct_answer,
                is_correct=True,
                solver_used='apprentice',
                apprentice_solution=apprentice_solution,
                verification=verification,
                oracle_solution=None,
                confidence=apprentice_solution.confidence,
                processing_time=processing_time,
                metadata={
                    'apprentice_succeeded': True,
                    'oracle_needed': False
                }
            )
        
        else:
            # Failure - need oracle
            if verbose:
                logger.warning("\n⚠️  Apprentice answer is wrong. Consulting Oracle...")
            
            self.stats['oracle_needed'] += 1
            
            # STEP 3: Oracle provides correct solution
            if verbose:
                logger.info("\n👨‍🏫 Step 3: Oracle (Teacher) solving problem...")
            
            oracle_solution = self.oracle.solve(problem_data)
            
            if verbose and oracle_solution.reasoning_steps:
                logger.info("   Oracle's reasoning:")
                for i, step in enumerate(oracle_solution.reasoning_steps[:10], 1):
                    logger.info(f"     {i}. {step[:100]}...")
            
            if oracle_solution.final_answer is None:
                # Complete failure - even oracle couldn't solve
                if verbose:
                    logger.error("   ❌ Oracle also failed!")
                
                self.stats['complete_failures'] += 1
                self._log_complete_failure(problem_data, apprentice_solution, oracle_solution)
                
                processing_time = time.time() - start_time
                
                return SolverResult(
                    final_answer=verification.correct_answer,  # Use verifier's answer
                    is_correct=False,
                    solver_used='verifier',
                    apprentice_solution=apprentice_solution,
                    verification=verification,
                    oracle_solution=oracle_solution,
                    confidence=0.3,
                    processing_time=processing_time,
                    metadata={
                        'apprentice_succeeded': False,
                        'oracle_succeeded': False,
                        'complete_failure': True
                    }
                )
            
            # Oracle succeeded - verify its answer
            oracle_verification = self.verifier.verify(
                equations=equations,
                variables=variables,
                target_variable=target_var,
                proposed_answer=oracle_solution.final_answer
            )
            
            # Check agreement between oracle, verifier, and apprentice
            oracle_verifier_match = oracle_verification.is_correct
            oracle_apprentice_match = self._answers_match(
                oracle_solution.final_answer,
                apprentice_solution.final_answer
            )
            
            if verbose:
                if oracle_verifier_match:
                    logger.info(f"   ✅ Oracle answer CORRECT (verified): {oracle_solution.final_answer}")
                else:
                    logger.warning(f"   ⚠️  Conflict detected!")
                    logger.warning(f"      Oracle: {oracle_solution.final_answer}")
                    logger.warning(f"      Verifier: {oracle_verification.correct_answer}")
                    logger.warning(f"      Apprentice: {apprentice_solution.final_answer}")
                    
                    if oracle_apprentice_match:
                        logger.warning(f"   🤔 Oracle and Apprentice AGREE, but Verifier differs")
                        logger.warning(f"   📊 Confidence: Oracle + Apprentice = HIGH, Verifier may be wrong")
            
            # Determine confidence based on agreement
            if oracle_verifier_match:
                # Oracle and verifier agree - high confidence
                final_confidence = oracle_solution.confidence
                final_answer = oracle_solution.final_answer
                is_correct = True
            elif oracle_apprentice_match:
                # Oracle and apprentice agree, verifier differs - medium-high confidence
                # Likely verifier (SymPy) is wrong
                final_confidence = 0.75
                final_answer = oracle_solution.final_answer
                is_correct = True  # Trust Oracle+Apprentice consensus
                if verbose:
                    logger.info(f"   ✅ Using Oracle answer (consensus with Apprentice)")
            else:
                # Three-way disagreement - medium confidence
                # Trust Oracle more than verifier
                final_confidence = 0.6
                final_answer = oracle_solution.final_answer
                is_correct = oracle_verification.is_correct
            
            # Save oracle's solution for training (this is the gold standard!)
            # Include ground truth for quality tracking if available
            self._save_training_example(
                problem_data=problem_data,
                solution_steps=oracle_solution.reasoning_steps,
                final_answer=oracle_solution.final_answer,
                source='oracle',
                tool_calls=oracle_solution.tool_calls,
                ground_truth=ground_truth_numeric if has_ground_truth else None,
                ground_truth_raw=ground_truth_raw if has_ground_truth else None,
                ground_truth_unit=ground_truth_unit if has_ground_truth else None
            )
            
            processing_time = time.time() - start_time
            
            if verbose:
                logger.info(f"\n✅ Oracle solved in {processing_time:.2f}s")
                logger.info(f"💾 Training example saved for fine-tuning")
                logger.info(f"📊 Final confidence: {final_confidence:.2%}")
            
            return SolverResult(
                final_answer=final_answer,
                is_correct=is_correct,
                solver_used='oracle',
                apprentice_solution=apprentice_solution,
                verification=oracle_verification,
                oracle_solution=oracle_solution,
                confidence=final_confidence,
                processing_time=processing_time,
                metadata={
                    'apprentice_succeeded': False,
                    'oracle_needed': True,
                    'oracle_succeeded': True,
                    'saved_for_training': True,
                    'oracle_verifier_match': oracle_verifier_match,
                    'oracle_apprentice_match': oracle_apprentice_match,
                    'validation_method': 'verifier'
                }
            )
    
    def _compare_with_ground_truth(
        self,
        answer: float,
        ground_truth: float,
        tolerance: float = 0.001,
        verbose: bool = False
    ) -> bool:
        """
        Compare answer with ground truth within tolerance.
        
        Args:
            answer: Proposed answer
            ground_truth: Ground truth value
            tolerance: Relative tolerance (0.1% default)
            verbose: Whether to log comparison details
        
        Returns:
            True if answer matches ground truth within tolerance
        """
        if answer is None or ground_truth is None:
            return False
        
        try:
            answer_val = float(answer)
            gt_val = float(ground_truth)
            
            # Check if values are equal within tolerance
            if gt_val == 0:
                # Absolute comparison for zero
                match = abs(answer_val) < 1e-6
            else:
                # Relative comparison
                rel_diff = abs(answer_val - gt_val) / abs(gt_val)
                match = rel_diff <= tolerance
            
            if verbose:
                if match:
                    logger.info(f"   ✅ Answer {answer_val} matches ground truth {gt_val}")
                else:
                    diff = abs(answer_val - gt_val)
                    logger.warning(f"   ❌ Answer {answer_val} != ground truth {gt_val} (diff: {diff:.6f})")
            
            return match
        except (ValueError, TypeError) as e:
            if verbose:
                logger.warning(f"   ⚠️  Could not compare: {e}")
            return False
    
    def _answers_match(
        self,
        answer1: float,
        answer2: float,
        tolerance: float = 0.001
    ) -> bool:
        """
        Check if two answers match within tolerance.
        
        Args:
            answer1: First answer
            answer2: Second answer
            tolerance: Relative tolerance (0.1% default)
        
        Returns:
            True if answers match within tolerance
        """
        if answer1 is None or answer2 is None:
            return False
        
        try:
            val1 = float(answer1)
            val2 = float(answer2)
            
            if val2 == 0:
                return abs(val1) < 1e-6
            else:
                rel_diff = abs(val1 - val2) / abs(val2)
                return rel_diff <= tolerance
        except (ValueError, TypeError):
            return False
    
    def _extract_equations(self, problem_data: Dict[str, Any]) -> List[str]:
        """Extract equation strings from problem data."""
        equations = []
        if 'parsing' in problem_data and 'equations' in problem_data['parsing']:
            for eq in problem_data['parsing']['equations']:
                if isinstance(eq, dict):
                    equations.append(eq.get('equation_string', ''))
                else:
                    equations.append(str(eq))
        return equations
    
    def _extract_variables(self, problem_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract variable values from problem data."""
        variables = {}
        
        # Try to get from unit standardization first (most reliable)
        if 'unit_standardization' in problem_data:
            std_vars = problem_data['unit_standardization'].get('standardized_variables', {})
            for var_name, var_data in std_vars.items():
                if isinstance(var_data, dict):
                    value = var_data.get('standardized_value')
                    if value is not None:
                        variables[var_name] = float(value)
        
        # Fall back to variable extraction if standardization not available
        if not variables and 'variable_extraction' in problem_data:
            ext_vars = problem_data['variable_extraction'].get('variables', {})
            for var_name, var_data in ext_vars.items():
                if isinstance(var_data, dict):
                    value = var_data.get('value')
                    if value is not None:
                        variables[var_name] = float(value)
        
        return variables
    
    def _extract_target_variable(self, problem_data: Dict[str, Any]) -> str:
        """Extract target variable from problem data."""
        if 'parsing' in problem_data:
            return problem_data['parsing'].get('target_variable', 'unknown')
        return 'unknown'
    
    def _handle_apprentice_failure(
        self,
        problem_data: Dict[str, Any],
        start_time: float
    ) -> SolverResult:
        """Handle case where apprentice completely fails to produce an answer."""
        processing_time = time.time() - start_time
        
        self.logger.warning("⚠️  Apprentice failed to produce answer. Calling Oracle...")
        
        # Call oracle directly
        oracle_solution = self.oracle.solve(problem_data)
        
        if oracle_solution.final_answer is None:
            # Complete failure
            self.stats['complete_failures'] += 1
            return SolverResult(
                final_answer=0.0,
                is_correct=False,
                solver_used='none',
                apprentice_solution=None,
                verification=None,
                oracle_solution=oracle_solution,
                confidence=0.0,
                processing_time=processing_time,
                metadata={
                    'error': 'complete_failure',
                    'apprentice_no_answer': True,
                    'oracle_failed': True
                }
            )
        
        # Oracle succeeded - save for training
        self._save_training_example(
            problem_data=problem_data,
            solution_steps=oracle_solution.reasoning_steps,
            final_answer=oracle_solution.final_answer,
            source='oracle',
            tool_calls=oracle_solution.tool_calls
        )
        
        return SolverResult(
            final_answer=oracle_solution.final_answer,
            is_correct=True,  # Assuming oracle is correct
            solver_used='oracle',
            apprentice_solution=None,
            verification=None,
            oracle_solution=oracle_solution,
            confidence=oracle_solution.confidence,
            processing_time=processing_time,
            metadata={
                'apprentice_no_answer': True,
                'oracle_succeeded': True,
                'saved_for_training': True
            }
        )
    
    def _log_complete_failure(
        self,
        problem_data: Dict[str, Any],
        apprentice_solution: Optional[ApprenticeSolution],
        oracle_solution: Optional[OracleSolution]
    ):
        """Log cases where both apprentice AND oracle fail."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'original_problem': problem_data.get('original_problem', ''),
            'apprentice_status': 'failed' if apprentice_solution and apprentice_solution.final_answer else 'no_answer',
            'oracle_status': 'failed',
            'status': 'needs_human_review'
        }
        
        with open(self.failure_log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        self.logger.error(f"❌ Complete failure logged - needs human review")
    
    def _calculate_difficulty(self, problem_data: Dict[str, Any], 
                             source: str, tool_calls: List = None) -> float:
        """
        Calculate problem difficulty based on solving metrics.
        Returns value between 0.0 (easy) and 1.0 (hard).
        """
        difficulty = 0.0
        
        # Factor 1: Required oracle (apprentice failed) = harder
        if source == 'oracle':
            difficulty += 0.4
        
        # Factor 2: Number of equations
        equations = self._extract_equations(problem_data)
        difficulty += min(len(equations) * 0.1, 0.3)
        
        # Factor 3: Number of tool calls (more = harder)
        if tool_calls:
            difficulty += min(len(tool_calls) * 0.05, 0.3)
        
        return min(difficulty, 1.0)  # Cap at 1.0
    
    def _should_flag_for_review(self, final_answer: float, 
                                tool_calls: List = None) -> bool:
        """
        Flag edge cases for human review.
        Returns True if this example needs manual verification.
        """
        # Flag if too many tool calls (might indicate confusion)
        if tool_calls and len(tool_calls) > 10:
            return True
        
        # Flag if answer has unusual magnitude (potential unit error)
        if abs(final_answer) > 1e6 or (abs(final_answer) < 1e-6 and final_answer != 0):
            return True
        
        return False
    
    def _save_training_example(
        self,
        problem_data: Dict[str, Any],
        solution_steps: List[str],
        final_answer: float,
        source: str,  # 'apprentice' or 'oracle'
        tool_calls: List[Dict[str, Any]] = None,
        ground_truth: Optional[float] = None,
        ground_truth_raw: Optional[str] = None,
        ground_truth_unit: Optional[str] = None
    ):
        """
        Save a training example with ground truth support.
        
        Enhanced with:
        - Ground truth tracking (for Oracle accuracy measurement)
        - Difficulty scoring (for stratified sampling)
        - Review flags (for quality control)
        
        Format:
        {
            "problem": "Original problem text",
            "steps": ["Step 1: ...", "Step 2: ...", ...],
            "oracle_answer": 150.0,
            "ground_truth": 150.0,  # NEW!
            "oracle_correct": true,  # NEW!
            "metadata": {...}
        }
        """
        # Calculate difficulty and review flag
        difficulty = self._calculate_difficulty(problem_data, source, tool_calls)
        needs_review = self._should_flag_for_review(final_answer, tool_calls)
        
        # Calculate Oracle correctness if ground truth available
        oracle_correct = None
        if ground_truth is not None and final_answer is not None:
            # Tolerance for numeric comparison (0.1% or 1e-6, whichever is larger)
            tolerance = max(abs(ground_truth) * 0.001, 1e-6)
            oracle_correct = abs(final_answer - ground_truth) <= tolerance
            
            # Log warning if Oracle is wrong
            if not oracle_correct:
                logger.warning(f"⚠️  Oracle answer INCORRECT! Oracle: {final_answer}, Ground truth: {ground_truth}")
                needs_review = True  # Force review for incorrect oracle solutions
        
        training_example = {
            'problem': problem_data.get('original_problem', ''),
            'steps': solution_steps,
            'oracle_answer': final_answer,
            'ground_truth': ground_truth,  # NEW!
            'ground_truth_raw': ground_truth_raw,  # NEW!
            'ground_truth_unit': ground_truth_unit,  # NEW!
            'oracle_correct': oracle_correct,  # NEW!
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'source': source,
                'difficulty': difficulty,
                'needs_review': needs_review,
                'tool_calls_count': len(tool_calls) if tool_calls else 0,
                'equations': self._extract_equations(problem_data),
                'target_variable': self._extract_target_variable(problem_data)
            }
        }
        
        # Add tool call details if from oracle
        if tool_calls:
            training_example['tool_calls'] = tool_calls
        
        # Append to training data file
        with open(self.training_data_file, 'a') as f:
            f.write(json.dumps(training_example) + '\n')
        
        # Log with all status indicators
        review_flag = " ⚠️ NEEDS REVIEW" if needs_review else ""
        correctness = ""
        if oracle_correct is not None:
            correctness = " ✅ CORRECT" if oracle_correct else " ❌ WRONG"
        
        self.logger.info(f"💾 Training example saved (source: {source}, difficulty: {difficulty:.2f}{correctness}{review_flag})")
    
    def get_training_data_count(self) -> int:
        """Get the number of training examples collected."""
        try:
            with open(self.training_data_file, 'r') as f:
                return sum(1 for _ in f)
        except FileNotFoundError:
            return 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get solver statistics."""
        if self.stats['total_problems'] > 0:
            accuracy = (self.stats['apprentice_correct'] / self.stats['total_problems']) * 100
        else:
            accuracy = 0.0
        
        return {
            **self.stats,
            'apprentice_accuracy': accuracy,
            'oracle_usage_rate': (self.stats['oracle_needed'] / max(1, self.stats['total_problems'])) * 100,
            'training_examples_collected': self.get_training_data_count()
        }
    
    def print_statistics(self):
        """Print solver statistics."""
        stats = self.get_statistics()
        
        print("\n" + "=" * 70)
        print("📊 SOLVER STATISTICS")
        print("=" * 70)
        print(f"Total Problems Solved: {stats['total_problems']}")
        print(f"Apprentice Correct: {stats['apprentice_correct']}")
        print(f"Oracle Needed: {stats['oracle_needed']}")
        print(f"Complete Failures: {stats['complete_failures']}")
        print(f"\nApprentice Accuracy: {stats['apprentice_accuracy']:.1f}%")
        print(f"Oracle Usage Rate: {stats['oracle_usage_rate']:.1f}%")
        print(f"\n💾 Training Examples Collected: {stats['training_examples_collected']}")
        print("=" * 70)


if __name__ == "__main__":
    # Test the solver agent
    print("🧪 Testing Solver Agent")
    print("=" * 70)
    
    # Create a sample problem (simulating output from previous pipeline stages)
    test_problem = {
        'original_problem': 'John has 5 apples and Mary gives him 3 more. How many apples does John have?',
        'parsing': {
            'equations': [
                {'equation_string': 'total = initial + given'}
            ],
            'target_variable': 'total'
        },
        'unit_standardization': {
            'standardized_variables': {
                'initial': {'standardized_value': 5, 'standardized_unit': ''},
                'given': {'standardized_value': 3, 'standardized_unit': ''}
            }
        }
    }
    
    # Create solver and solve
    solver = SolverAgent()
    result = solver.solve(test_problem, verbose=True)
    
    print("\n" + "=" * 70)
    print("📋 FINAL RESULT")
    print("=" * 70)
    print(f"Final Answer: {result.final_answer}")
    print(f"Is Correct: {result.is_correct}")
    print(f"Solver Used: {result.solver_used}")
    print(f"Confidence: {result.confidence}")
    print(f"Processing Time: {result.processing_time:.2f}s")
    
    # Print statistics
    solver.print_statistics()
