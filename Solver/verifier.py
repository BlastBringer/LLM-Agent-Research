#!/usr/bin/env python3
"""
🎯 MATH VERIFIER - The Deterministic Judge
===========================================

This module provides a 100% reliable ground truth for mathematical answers.
It doesn't care about reasoning, only the final numerical result.

Key Features:
- Uses SymPy for symbolic equation solving
- Falls back to NumPy for numerical evaluation
- Sandboxed eval() for safety
- Returns detailed solution steps for debugging

Usage:
    verifier = MathVerifier()
    result = verifier.verify(
        equations=['total = apples + oranges'],
        variables={'apples': 5, 'oranges': 3},
        target_variable='total',
        proposed_answer=8
    )
    print(result.is_correct)  # True
    print(result.correct_answer)  # 8.0
"""

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import sympy as sp
import numpy as np
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Result of mathematical verification."""
    is_correct: bool
    correct_answer: float
    proposed_answer: float
    difference: float
    solution_steps: List[str]
    verification_method: str  # 'sympy', 'numpy', or 'eval'
    confidence: float
    metadata: Dict[str, Any]


class MathVerifier:
    """
    Deterministic mathematical verifier using symbolic and numerical computation.
    This is the ground truth for our learning system.
    """
    
    def __init__(self, tolerance: float = 1e-6):
        """
        Initialize the verifier.
        
        Args:
            tolerance: Floating point comparison tolerance (default: 1e-6)
        """
        self.tolerance = tolerance
        self.logger = logging.getLogger(__name__)
        self.logger.info("🎯 Math Verifier initialized")
    
    def verify(
        self,
        equations: List[str],
        variables: Dict[str, float],
        target_variable: str,
        proposed_answer: float
    ) -> VerificationResult:
        """
        Verify a proposed answer against the correct solution.
        
        Args:
            equations: List of equation strings (e.g., ['total = apples + oranges'])
            variables: Dictionary of known variable values
            target_variable: The variable we're solving for
            proposed_answer: The answer proposed by the apprentice
            
        Returns:
            VerificationResult with correctness and details
        """
        self.logger.info(f"🔍 Verifying answer for target: {target_variable}")
        
        # Step 1: Compute the correct answer
        correct_answer, solution_steps, method = self._solve_equations(
            equations, variables, target_variable
        )
        
        if correct_answer is None:
            self.logger.error("❌ Could not compute correct answer")
            return VerificationResult(
                is_correct=False,
                correct_answer=0.0,
                proposed_answer=proposed_answer,
                difference=float('inf'),
                solution_steps=["Error: Could not solve equations"],
                verification_method='failed',
                confidence=0.0,
                metadata={'error': 'solver_failed'}
            )
        
        # Step 2: Compare answers (ensure both are float to handle int vs float comparison)
        try:
            correct_answer_float = float(correct_answer)
            proposed_answer_float = float(proposed_answer)
        except (ValueError, TypeError) as e:
            self.logger.error(f"❌ Cannot convert answers to float: {e}")
            return VerificationResult(
                is_correct=False,
                correct_answer=correct_answer if correct_answer else 0.0,
                proposed_answer=proposed_answer,
                difference=float('inf'),
                solution_steps=["Error: Invalid numeric values"],
                verification_method='failed',
                confidence=0.0,
                metadata={'error': 'type_conversion_failed'}
            )
        
        difference = abs(correct_answer_float - proposed_answer_float)
        is_correct = difference < self.tolerance
        
        # Step 3: Build result
        result = VerificationResult(
            is_correct=is_correct,
            correct_answer=correct_answer_float,
            proposed_answer=proposed_answer_float,
            difference=difference,
            solution_steps=solution_steps,
            verification_method=method,
            confidence=1.0 if is_correct else 0.0,
            metadata={
                'tolerance': self.tolerance,
                'equations_count': len(equations),
                'variables_count': len(variables)
            }
        )
        
        if is_correct:
            self.logger.info(f"✅ CORRECT! Answer: {correct_answer_float}")
        else:
            self.logger.warning(
                f"❌ INCORRECT! Expected: {correct_answer_float}, Got: {proposed_answer_float}, "
                f"Difference: {difference}"
            )
        
        return result
    
    def _solve_equations(
        self,
        equations: List[str],
        known_values: Dict[str, float],
        target_var: str
    ) -> Tuple[Optional[float], List[str], str]:
        """
        Solve the equations to find the target variable's value.
        
        Returns:
            (answer, solution_steps, method_used)
        """
        # Try SymPy first (most reliable)
        result = self._solve_with_sympy(equations, known_values, target_var)
        if result is not None:
            answer, steps = result
            return answer, steps, 'sympy'
        
        # Fall back to direct substitution
        result = self._solve_with_substitution(equations, known_values, target_var)
        if result is not None:
            answer, steps = result
            return answer, steps, 'substitution'
        
        # Last resort: safe eval
        result = self._solve_with_eval(equations, known_values, target_var)
        if result is not None:
            answer, steps = result
            return answer, steps, 'eval'
        
        return None, ["Failed to solve"], 'failed'
    
    def _solve_with_sympy(
        self,
        equations: List[str],
        known_values: Dict[str, float],
        target_var: str
    ) -> Optional[Tuple[float, List[str]]]:
        """Solve using SymPy's symbolic solver."""
        try:
            steps = ["Using SymPy symbolic solver"]
            
            # Parse equations into SymPy format
            sympy_equations = []
            all_symbols = set()
            
            for eq_str in equations:
                # Split on '=' to get left and right sides
                if '=' not in eq_str:
                    continue
                
                left, right = eq_str.split('=', 1)
                left = left.strip()
                right = right.strip()
                
                # Parse both sides
                transformations = standard_transformations + (implicit_multiplication_application,)
                left_expr = parse_expr(left, transformations=transformations)
                right_expr = parse_expr(right, transformations=transformations)
                
                # Create equation
                equation = sp.Eq(left_expr, right_expr)
                sympy_equations.append(equation)
                
                # Collect symbols
                all_symbols.update(equation.free_symbols)
                
                steps.append(f"Parsed: {equation}")
            
            # Substitute known values
            substitutions = {sp.Symbol(name): value for name, value in known_values.items()}
            
            for i, eq in enumerate(sympy_equations):
                sympy_equations[i] = eq.subs(substitutions)
                steps.append(f"After substitution: {sympy_equations[i]}")
            
            # Solve for target variable
            target_symbol = sp.Symbol(target_var)
            
            # If only one equation, solve directly
            if len(sympy_equations) == 1:
                solutions = sp.solve(sympy_equations[0], target_symbol)
                if solutions:
                    answer = float(solutions[0] if isinstance(solutions, list) else solutions)
                    steps.append(f"Solution: {target_var} = {answer}")
                    return answer, steps
            else:
                # Multiple equations - solve system
                solutions = sp.solve(sympy_equations, target_symbol)
                if solutions and target_symbol in solutions:
                    answer = float(solutions[target_symbol])
                    steps.append(f"Solution: {target_var} = {answer}")
                    return answer, steps
            
            return None
            
        except Exception as e:
            self.logger.debug(f"SymPy solver failed: {e}")
            return None
    
    def _solve_with_substitution(
        self,
        equations: List[str],
        known_values: Dict[str, float],
        target_var: str
    ) -> Optional[Tuple[float, List[str]]]:
        """Solve by direct substitution (for simple cases)."""
        try:
            steps = ["Using direct substitution"]
            
            # Look for the equation that defines the target variable
            target_equation = None
            for eq in equations:
                if '=' in eq:
                    left, right = eq.split('=', 1)
                    if left.strip() == target_var:
                        target_equation = right.strip()
                        break
            
            if not target_equation:
                return None
            
            steps.append(f"Found equation: {target_var} = {target_equation}")
            
            # Build namespace with known values and safe math functions
            namespace = known_values.copy()
            namespace.update({
                'abs': abs, 'round': round, 'min': min, 'max': max,
                'pow': pow, 'sqrt': np.sqrt, 'sin': np.sin, 'cos': np.cos,
                'tan': np.tan, 'exp': np.exp, 'log': np.log,
                'pi': np.pi, 'e': np.e
            })
            
            # Evaluate the expression
            answer = eval(target_equation, {"__builtins__": {}}, namespace)
            answer = float(answer)
            
            steps.append(f"Evaluated: {target_var} = {answer}")
            return answer, steps
            
        except Exception as e:
            self.logger.debug(f"Substitution solver failed: {e}")
            return None
    
    def _solve_with_eval(
        self,
        equations: List[str],
        known_values: Dict[str, float],
        target_var: str
    ) -> Optional[Tuple[float, List[str]]]:
        """Safe eval as last resort."""
        try:
            steps = ["Using safe eval"]
            
            # Create a safe namespace
            namespace = known_values.copy()
            namespace.update({
                'abs': abs, 'round': round, 'min': min, 'max': max,
                'pow': pow, 'sqrt': np.sqrt,
                'pi': np.pi, 'e': np.e
            })
            
            # Execute equations in order
            for eq in equations:
                if '=' in eq:
                    left, right = eq.split('=', 1)
                    var_name = left.strip()
                    expression = right.strip()
                    
                    # Evaluate
                    value = eval(expression, {"__builtins__": {}}, namespace)
                    namespace[var_name] = float(value)
                    steps.append(f"Computed: {var_name} = {value}")
            
            # Get the target variable's value
            if target_var in namespace:
                return namespace[target_var], steps
            
            return None
            
        except Exception as e:
            self.logger.debug(f"Eval solver failed: {e}")
            return None
    
    def batch_verify(
        self,
        test_cases: List[Dict[str, Any]]
    ) -> List[VerificationResult]:
        """
        Verify multiple test cases at once.
        
        Args:
            test_cases: List of dicts with keys: 'equations', 'variables', 
                       'target_variable', 'proposed_answer'
        
        Returns:
            List of VerificationResults
        """
        results = []
        for i, case in enumerate(test_cases):
            self.logger.info(f"Verifying test case {i+1}/{len(test_cases)}")
            result = self.verify(**case)
            results.append(result)
        
        accuracy = sum(1 for r in results if r.is_correct) / len(results) * 100
        self.logger.info(f"📊 Batch accuracy: {accuracy:.1f}%")
        
        return results


# Standalone function for quick verification
def verify_solution(
    equations: List[str],
    variables: Dict[str, float],
    target_variable: str,
    proposed_answer: float,
    tolerance: float = 1e-6
) -> VerificationResult:
    """Quick verification without creating a verifier instance."""
    verifier = MathVerifier(tolerance=tolerance)
    return verifier.verify(equations, variables, target_variable, proposed_answer)


if __name__ == "__main__":
    # Test the verifier
    print("🧪 Testing Math Verifier")
    print("=" * 50)
    
    # Test case 1: Simple addition
    result = verify_solution(
        equations=['total = apples + oranges'],
        variables={'apples': 5, 'oranges': 3},
        target_variable='total',
        proposed_answer=8
    )
    print(f"Test 1 - Simple addition: {'✅ PASS' if result.is_correct else '❌ FAIL'}")
    print(f"  Expected: {result.correct_answer}, Got: {result.proposed_answer}")
    
    # Test case 2: More complex
    result = verify_solution(
        equations=['distance = speed * time'],
        variables={'speed': 60, 'time': 2},
        target_variable='distance',
        proposed_answer=120
    )
    print(f"\nTest 2 - Multiplication: {'✅ PASS' if result.is_correct else '❌ FAIL'}")
    print(f"  Expected: {result.correct_answer}, Got: {result.proposed_answer}")
    
    # Test case 3: Wrong answer
    result = verify_solution(
        equations=['total = a + b + c'],
        variables={'a': 10, 'b': 20, 'c': 30},
        target_variable='total',
        proposed_answer=50  # Wrong!
    )
    print(f"\nTest 3 - Intentional error: {'✅ PASS' if not result.is_correct else '❌ FAIL'}")
    print(f"  Expected: {result.correct_answer}, Got: {result.proposed_answer}")
    print(f"  Difference: {result.difference}")
