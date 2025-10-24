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
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
from datetime import datetime

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
    
    def solve(
        self,
        problem_data: Dict[str, Any],
        verbose: bool = True
    ) -> SolverResult:
        """
        Main solving pipeline.
        
        Args:
            problem_data: All processed data from previous pipeline stages
            verbose: Whether to print detailed progress
        
        Returns:
            SolverResult with answer and verification
        """
        import time
        start_time = time.time()
        
        self.stats['total_problems'] += 1
        
        if verbose:
            logger.info("\n" + "=" * 70)
            logger.info("🧠 STARTING SOLVER AGENT")
            logger.info("=" * 70)
        
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
        
        # STEP 2: Verifier checks the answer
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
            
            if verbose:
                if oracle_verification.is_correct:
                    logger.info(f"   ✅ Oracle answer CORRECT: {oracle_solution.final_answer}")
                else:
                    logger.warning(f"   ⚠️  Oracle answer differs from verifier!")
                    logger.warning(f"      Oracle: {oracle_solution.final_answer}")
                    logger.warning(f"      Verifier: {oracle_verification.correct_answer}")
            
            # Save oracle's solution for training (this is the gold standard!)
            self._save_training_example(
                problem_data=problem_data,
                solution_steps=oracle_solution.reasoning_steps,
                final_answer=oracle_solution.final_answer,
                source='oracle',
                tool_calls=oracle_solution.tool_calls
            )
            
            processing_time = time.time() - start_time
            
            if verbose:
                logger.info(f"\n✅ Oracle solved in {processing_time:.2f}s")
                logger.info(f"💾 Training example saved for fine-tuning")
            
            return SolverResult(
                final_answer=oracle_solution.final_answer,
                is_correct=oracle_verification.is_correct,
                solver_used='oracle',
                apprentice_solution=apprentice_solution,
                verification=oracle_verification,
                oracle_solution=oracle_solution,
                confidence=oracle_solution.confidence,
                processing_time=processing_time,
                metadata={
                    'apprentice_succeeded': False,
                    'oracle_needed': True,
                    'oracle_succeeded': True,
                    'saved_for_training': True
                }
            )
    
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
    
    def _save_training_example(
        self,
        problem_data: Dict[str, Any],
        solution_steps: List[str],
        final_answer: float,
        source: str,  # 'apprentice' or 'oracle'
        tool_calls: List[Dict[str, Any]] = None
    ):
        """
        Save a training example in the format needed for fine-tuning.
        
        Format:
        {
            "problem": "Original problem text",
            "steps": ["Step 1: ...", "Step 2: ...", ...],
            "answer": 42.0,
            "metadata": {...}
        }
        """
        training_example = {
            'problem': problem_data.get('original_problem', ''),
            'steps': solution_steps,
            'answer': final_answer,
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'source': source,  # Whether this came from apprentice or oracle
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
        
        self.logger.info(f"💾 Training example saved (source: {source})")
    
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
