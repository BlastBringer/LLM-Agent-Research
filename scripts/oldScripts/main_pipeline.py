#!/usr/bin/env python3
"""
🚀 MAIN PIPELINE - COMPLETE WORD PROBLEM SOLVER
================================================

This is the main orchestrator that runs the COMPLETE pipeline as per architecture:

INFORMATION RETRIEVAL:
1. Templatization: Convert word problem → generic template + legend
2. Parsing: Extract equations, variables, and target from template  
3. Variable Extraction: Extract values with units
4. Unit Standardization: Convert to SI units

SOLVER ARCHITECTURE:
5. Student Model (Apprentice): Attempts to solve problem
6. Verifier: Checks answer using SymPy equation solving
7. Teacher Model (Oracle): Provides correct solution if student fails
8. Memory Store: Saves oracle solutions for fine-tuning

Each component is independent and can be tested separately.
"""

import sys
import os
from typing import Dict, Any
from dataclasses import asdict
import json
import logging
import time

# Add modules to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Reasoning'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Solver'))

# Import Information Retrieval components
from Reasoning.templatizer import WordProblemTemplatizer, templatize_word_problem, TemplatizationResult
from Reasoning.parser import MathematicalProblemParser, parse_math_problem, ParseResult
from Reasoning.variable_extractor import VariableExtractor, extract_variables_from_problem
from Reasoning.unit_standardizer import UnitStandardizer, standardize_units

# Import Solver Architecture components
from Solver.solver_agent import SolverAgent

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class WordProblemSolver:
    """
    Main orchestrator for the COMPLETE word problem solving pipeline.
    Implements the full architecture from the diagram:
    - Information Retrieval (Templatization → Parsing → Variable Extraction → Unit Standardization)
    - Solver Architecture (Student Model → Verifier → Teacher Model)
    """
    
    def __init__(self):
        """Initialize ALL pipeline components."""
        logger.info("🚀 Initializing Complete Word Problem Solver Pipeline")
        logger.info("-" * 60)
        
        # Information Retrieval Components
        self.templatizer = WordProblemTemplatizer()
        self.parser = MathematicalProblemParser()
        self.variable_extractor = VariableExtractor()
        self.unit_standardizer = UnitStandardizer()
        
        # Solver Architecture Components
        self.solver = SolverAgent()
        
        logger.info("✅ All pipeline components initialized")
        logger.info("=" * 60)
    
    def solve(self, problem: str, verbose: bool = True) -> Dict[str, Any]:
        """
        Run the COMPLETE pipeline on a word problem (as per architecture diagram).
        
        Args:
            problem: The word problem to solve
            verbose: Whether to print detailed steps
            
        Returns:
            Dictionary containing results from each stage INCLUDING final solution
        """
        start_time = time.time()
        
        if verbose:
            logger.info("\n" + "=" * 70)
            logger.info("📝 ORIGINAL PROBLEM")
            logger.info("=" * 70)
            print(f"\n{problem}\n")
        
        # ========== INFORMATION RETRIEVAL ==========
        
        # Stage 1: Templatization
        if verbose:
            logger.info("=" * 70)
            logger.info("🔄 STAGE 1: TEMPLATIZATION")
            logger.info("=" * 70)
        
        templatization_result = templatize_word_problem(problem)
        
        if verbose:
            print(f"\n📋 Templatized: {templatization_result.templated_problem}")
            if templatization_result.legend:
                print(f"\n🗺️  Legend:")
                for placeholder, original in sorted(templatization_result.legend.items()):
                    print(f"   {placeholder} → {original}")
        
        # Stage 2: Parsing
        if verbose:
            logger.info("\n" + "=" * 70)
            logger.info("📐 STAGE 2: MATHEMATICAL PARSING")
            logger.info("=" * 70)
        
        parse_result = self.parser.parse_problem(templatization_result)
        
        if verbose:
            print(f"\n🎯 Problem Type: {parse_result.problem_type}")
            print(f"📊 Equations Needed: {parse_result.num_equations_needed}")
            print(f"🎪 Target Variable: {parse_result.target_variable}")
            
            if parse_result.equations:
                print(f"\n📋 Equations:")
                for i, eq in enumerate(parse_result.equations, 1):
                    print(f"   [{i}] {eq.equation_string}")
                    if eq.description:
                        print(f"       → {eq.description}")
        
        # Stage 3: Variable Extraction
        if verbose:
            logger.info("\n" + "=" * 70)
            logger.info("🔢 STAGE 3: VARIABLE EXTRACTION")
            logger.info("=" * 70)
        
        extraction_result = self.variable_extractor.extract_variables(
            templatization_result.templated_problem,
            parse_result.all_variables,
            [eq.equation_string for eq in parse_result.equations]
        )
        
        if verbose:
            print(f"\n📊 Extraction Method: {extraction_result.extraction_method.upper()}")
            if extraction_result.variables:
                print(f"\n🔢 Extracted Variables:")
                for var_name, var_val in extraction_result.variables.items():
                    unit_str = f" {var_val.unit}" if var_val.unit else ""
                    print(f"   {var_name} = {var_val.value}{unit_str}")
            else:
                print("\n⚠️  No variables extracted (may be dimensionless)")
        
        # Stage 4: Unit Standardization
        if verbose:
            logger.info("\n" + "=" * 70)
            logger.info("⚖️ STAGE 4: UNIT STANDARDIZATION")
            logger.info("=" * 70)
        
        standardization_result = self.unit_standardizer.standardize_variables(
            extraction_result.variables
        )
        
        if verbose:
            print(f"\n📏 Unit System: {standardization_result.unit_system}")
            
            if standardization_result.conversions_applied:
                print(f"\n✅ Conversions Applied:")
                for conv in standardization_result.conversions_applied:
                    print(f"   • {conv}")
            else:
                print(f"\n✅ No conversions needed (all units already standardized)")
            
            if standardization_result.standardized_variables:
                print(f"\n📊 Standardized Variables:")
                for var_name, std_qty in standardization_result.standardized_variables.items():
                    unit_str = f" {std_qty.standardized_unit}" if std_qty.standardized_unit else ""
                    print(f"   {var_name} = {std_qty.standardized_value}{unit_str}")
        
        # ========== SOLVER ARCHITECTURE ==========
        
        # Stage 5: Solving (Student Model → Verifier → Teacher Model)
        if verbose:
            logger.info("\n" + "=" * 70)
            logger.info("🤖 STAGE 5: SOLVER ARCHITECTURE")
            logger.info("=" * 70)
        
        # Prepare problem data for solver
        problem_data = {
            'original_problem': problem,
            'templatization': {
                'templated_problem': templatization_result.templated_problem,
                'legend': templatization_result.legend
            },
            'parsing': {
                'equations': [asdict(eq) for eq in parse_result.equations],
                'target_variable': parse_result.target_variable,
                'all_variables': parse_result.all_variables
            },
            'variable_extraction': {
                'method': extraction_result.extraction_method,
                'variables': {k: asdict(v) for k, v in extraction_result.variables.items()}
            },
            'unit_standardization': {
                'unit_system': standardization_result.unit_system,
                'standardized_variables': {k: asdict(v) for k, v in standardization_result.standardized_variables.items()}
            }
        }
        
        # Solve with Student Model → Verifier → Teacher Model loop
        solve_result = self.solver.solve(problem_data, verbose=verbose)
        
        # Stage 6: Convert answer back to original units
        if verbose:
            logger.info("\n" + "=" * 70)
            logger.info("� STAGE 6: UNIT CONVERSION (Back to Original)")
            logger.info("=" * 70)
        
        # Convert the final answer from SI units back to original units
        final_answer_converted, final_unit = self.unit_standardizer.convert_answer_to_original_units(
            answer_value=solve_result.final_answer,
            target_variable=parse_result.target_variable,
            standardization_result=standardization_result
        )
        
        total_time = time.time() - start_time
        
        if verbose:
            print(f"\n✅ Answer in SI units: {solve_result.final_answer}")
            print(f"✅ Answer in original units: {final_answer_converted} {final_unit}")
            
            logger.info("\n" + "=" * 70)
            logger.info("✅ PIPELINE COMPLETE")
            logger.info("=" * 70)
            print(f"\n🎯 Final Answer: {final_answer_converted} {final_unit}")
            print(f"✅ Verification: {'CORRECT ✓' if solve_result.is_correct else 'NEEDS REVIEW ✗'}")
            print(f"🧠 Solver Used: {solve_result.solver_used.upper()}")
            print(f"⏱️  Total Time: {total_time:.2f}s")
        
        # Return COMPLETE results (not partial like before)
        return {
            'original_problem': problem,
            'stages': {
                'templatization': {
                    'templated_problem': templatization_result.templated_problem,
                    'legend': templatization_result.legend,
                    'confidence': templatization_result.confidence_score
                },
                'parsing': {
                    'problem_type': parse_result.problem_type,
                    'num_equations': parse_result.num_equations_needed,
                    'target_variable': parse_result.target_variable,
                    'equations': [asdict(eq) for eq in parse_result.equations],
                    'confidence': parse_result.confidence_score
                },
                'variable_extraction': {
                    'method': extraction_result.extraction_method,
                    'variables': {k: asdict(v) for k, v in extraction_result.variables.items()},
                    'confidence': extraction_result.confidence_score
                },
                'unit_standardization': {
                    'unit_system': standardization_result.unit_system,
                    'conversions': standardization_result.conversions_applied,
                    'standardized_variables': {k: asdict(v) for k, v in standardization_result.standardized_variables.items()},
                    'unit_consistency': standardization_result.unit_consistency,
                    'confidence': standardization_result.confidence_score
                }
            },
            'solution': {
                'final_answer_si': solve_result.final_answer,
                'final_answer': final_answer_converted,
                'final_unit': final_unit,
                'is_correct': solve_result.is_correct,
                'solver_used': solve_result.solver_used,
                'confidence': solve_result.confidence,
                'processing_time': total_time
            }
        }

def main():
    """Main entry point for the pipeline."""
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "🧮 WORD PROBLEM SOLVER PIPELINE" + " " * 22 + "║")
    print("╚" + "═" * 68 + "╝\n")
    
    # Sample problems to demonstrate
    sample_problems = [
        "John has 5 apples and Mary has 3 oranges. How many fruits do they have together?",
        "A train travels 120 miles in 2 hours. What is its average speed?",
        "Lisa earns $85,000 per year. If she gets a 12% raise, what will be her new salary?",
    ]
    
    # Initialize solver
    solver = WordProblemSolver()
    
    # Check command-line arguments
    if len(sys.argv) > 1:
        # User provided a problem via command line
        problem = " ".join(sys.argv[1:])
        print(f"\n📝 Solving your problem...\n")
        result = solver.solve(problem, verbose=True)
    else:
        # Interactive mode
        print("\nChoose a mode:")
        print("1. 📝 Enter your own word problem")
        print("2. 🎯 Run demo with sample problems")
        print("3. 🔄 Interactive mode (solve multiple problems)")
        print()
        
        try:
            choice = input("Enter your choice (1, 2, or 3): ").strip()
            
            if choice == "1":
                # Single custom problem
                print("\n" + "─" * 70)
                problem = input("📝 Enter your word problem: ").strip()
                if problem:
                    result = solver.solve(problem, verbose=True)
                else:
                    print("⚠️ No problem provided. Exiting.")
                    return
                    
            elif choice == "2":
                # Demo mode with samples
                print("\n🎯 Running demo with sample problems...\n")
                
                for i, problem in enumerate(sample_problems, 1):
                    print(f"\n{'─' * 70}")
                    print(f"SAMPLE PROBLEM {i}/{len(sample_problems)}")
                    print(f"{'─' * 70}")
                    
                    result = solver.solve(problem, verbose=True)
                    
                    if i < len(sample_problems):
                        input("\n⏎ Press Enter to continue to next problem...")
                        
            elif choice == "3":
                # Interactive mode - solve multiple problems
                print("\n🔄 Interactive Mode")
                print("=" * 70)
                print("Enter your word problems one at a time.")
                print("Type 'quit', 'exit', or 'q' to stop.\n")
                
                problem_count = 0
                while True:
                    print("\n" + "─" * 70)
                    problem = input("📝 Enter word problem (or 'quit' to exit): ").strip()
                    
                    if problem.lower() in ['quit', 'exit', 'q']:
                        print(f"\n✅ Solved {problem_count} problem(s) in this session.")
                        break
                    
                    if not problem:
                        print("⚠️ Please enter a problem.")
                        continue
                    
                    problem_count += 1
                    result = solver.solve(problem, verbose=True)
                    
                    # Ask if user wants to continue
                    print("\n" + "─" * 70)
                    cont = input("➡️  Solve another problem? (y/n): ").strip().lower()
                    if cont not in ['y', 'yes', '']:
                        print(f"\n✅ Solved {problem_count} problem(s) in this session.")
                        break
            else:
                print("❌ Invalid choice. Running demo by default.")
                # Run demo as fallback
                for i, problem in enumerate(sample_problems, 1):
                    print(f"\n{'─' * 70}")
                    print(f"SAMPLE PROBLEM {i}/{len(sample_problems)}")
                    print(f"{'─' * 70}")
                    result = solver.solve(problem, verbose=True)
                    if i < len(sample_problems):
                        input("\n⏎ Press Enter to continue...")
                        
        except EOFError:
            print("\n\n⚠️ Input stream closed. Exiting.")
            return
    
    print("\n" + "=" * 70)
    print("✅ Pipeline execution complete!")
    print("=" * 70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Pipeline interrupted by user")
    except Exception as e:
        logger.error(f"❌ Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
