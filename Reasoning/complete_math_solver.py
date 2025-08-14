#!/usr/bin/env python3
"""
🧮 COMPLETE MATHEMATICAL REASONING SYSTEM
=========================================

The ultimate math solver that integrates:
- Enhanced Problem Parser
- Enhanced Problem Classifier  
- Enhanced ReAct Math Agent
- Comprehensive Solution Generation

This system can solve ANY type of mathematical problem with proper reasoning.
"""

import sys
import os
import json
from typing import Dict, Any, List

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our enhanced components
from enhanced_problem_parser import EnhancedProblemParser
from enhanced_problem_classifier import EnhancedProblemClassifier
from enhanced_react_math_agent import EnhancedReActMathAgent

class CompleteMathSolver:
    """
    The ultimate mathematical problem solver that can handle any type of math problem.
    """
    
    def __init__(self):
        print("🚀 INITIALIZING COMPLETE MATH SOLVER")
        print("=" * 60)
        
        # Initialize all components
        print("📝 Loading Enhanced Parser...")
        self.parser = EnhancedProblemParser()
        
        print("🔍 Loading Enhanced Classifier...")
        self.classifier = EnhancedProblemClassifier()
        
        print("🧠 Loading Enhanced ReAct Agent...")
        self.react_agent = EnhancedReActMathAgent()
        
        print("✅ Complete Math Solver ready!")
        print("   Capabilities: Algebra, Calculus, Geometry, Statistics, and more!")
        print("=" * 60)
    
    def solve(self, problem: str, detailed_output: bool = True) -> Dict[str, Any]:
        """
        Solve any mathematical problem with complete reasoning.
        
        Args:
            problem: The mathematical problem to solve
            detailed_output: Whether to include detailed analysis
            
        Returns:
            Complete solution with reasoning steps
        """
        print(f"\n🎯 SOLVING: {problem}")
        print("=" * 80)
        
        # Step 1: Parse the problem
        print("\n📝 STEP 1: PARSING PROBLEM")
        print("-" * 40)
        parsed_result = self.parser.parse(problem)
        print(f"✅ Parsed successfully using strategy: {parsed_result.get('strategy_used', 'unknown')}")
        
        # Step 2: Classify the problem
        print("\n🔍 STEP 2: CLASSIFYING PROBLEM")
        print("-" * 40)
        classification = self.classifier.classify_detailed(problem)
        print(f"✅ Category: {classification['primary_category']}")
        print(f"   Subcategory: {classification['subcategory']}")
        print(f"   Difficulty: {classification['difficulty_level']}")
        print(f"   Confidence: {classification['confidence']:.2f}")
        
        # Step 3: Solve using ReAct reasoning
        print("\n🧠 STEP 3: REACT REASONING & SOLVING")
        print("-" * 40)
        solution_result = self.react_agent.solve_problem(problem)
        
        # Step 4: Generate comprehensive result
        print("\n📊 STEP 4: GENERATING FINAL RESULT")
        print("-" * 40)
        
        complete_result = {
            "original_problem": problem,
            "parsing_result": parsed_result,
            "classification": classification,
            "solution": solution_result,
            "summary": self._generate_summary(problem, parsed_result, classification, solution_result),
            "educational_explanation": self._generate_explanation(problem, classification, solution_result) if detailed_output else None
        }
        
        print("✅ Complete solution generated!")
        return complete_result
    
    def _generate_summary(self, problem: str, parsed: Dict, classified: Dict, solved: Dict) -> Dict[str, str]:
        """Generate a concise summary of the solution process."""
        return {
            "problem_type": classified.get('primary_category', 'unknown'),
            "solution_approach": f"Used {solved.get('iterations_used', 0)} reasoning steps",
            "final_answer": solved.get('solution', 'No solution found'),
            "difficulty_assessment": classified.get('difficulty_level', 'unknown'),
            "tools_recommended": ', '.join(classified.get('tools_needed', []))
        }
    
    def _generate_explanation(self, problem: str, classified: Dict, solved: Dict) -> str:
        """Generate an educational explanation of the solution process."""
        
        explanation_parts = []
        
        # Problem type explanation
        problem_type = classified.get('primary_category', 'unknown')
        explanation_parts.append(f"This is a {problem_type} problem.")
        
        # Difficulty explanation
        difficulty = classified.get('difficulty_level', 'unknown')
        if difficulty == 'basic':
            explanation_parts.append("This is a fundamental problem that uses basic mathematical concepts.")
        elif difficulty == 'intermediate':
            explanation_parts.append("This problem requires intermediate mathematical techniques.")
        elif difficulty == 'advanced':
            explanation_parts.append("This is an advanced problem requiring sophisticated mathematical methods.")
        
        # Solution approach
        if solved.get('reasoning_steps'):
            explanation_parts.append(f"The solution required {len(solved['reasoning_steps'])} logical steps.")
        
        # Key concepts
        concepts = classified.get('mathematical_concepts', [])
        if concepts:
            explanation_parts.append(f"Key concepts involved: {', '.join(concepts)}.")
        
        return ' '.join(explanation_parts)
    
    def interactive_mode(self):
        """
        Run the solver in interactive mode for testing different problems.
        """
        print("\n🧮 INTERACTIVE MATH SOLVER MODE")
        print("=" * 60)
        print("Enter any mathematical problem and get a complete solution!")
        print("Examples:")
        print("  - Word problems: 'If a train travels 60 mph for 2 hours...'")
        print("  - Equations: 'Solve for x: 2x + 5 = 15'")
        print("  - Calculus: 'Find the derivative of x² + 3x + 2'")
        print("  - Geometry: 'Find the area of a circle with radius 5'")
        print("  - And much more!")
        print("\nType 'quit' to exit.")
        print("=" * 60)
        
        while True:
            try:
                problem = input("\n🔢 Enter your math problem: ").strip()
                
                if problem.lower() in ['quit', 'exit', 'q']:
                    print("👋 Thanks for using the Complete Math Solver!")
                    break
                
                if not problem:
                    continue
                
                # Solve the problem
                result = self.solve(problem, detailed_output=True)
                
                # Display results
                self._display_solution(result)
                
            except KeyboardInterrupt:
                print("\n👋 Thanks for using the Complete Math Solver!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                print("Please try again with a different problem.")
    
    def _display_solution(self, result: Dict[str, Any]):
        """Display the solution in a user-friendly format."""
        
        print("\n" + "=" * 60)
        print("📋 COMPLETE SOLUTION")
        print("=" * 60)
        
        # Basic info
        summary = result['summary']
        print(f"🎯 Problem Type: {summary['problem_type']}")
        print(f"📊 Difficulty: {summary['difficulty_assessment']}")
        print(f"🔧 Tools Used: {summary['tools_recommended']}")
        
        # Solution
        print(f"\n✅ FINAL ANSWER:")
        print(f"   {summary['final_answer']}")
        
        # Educational explanation
        if result.get('educational_explanation'):
            print(f"\n💡 EXPLANATION:")
            print(f"   {result['educational_explanation']}")
        
        # Detailed reasoning (if available)
        if result['solution'].get('reasoning_steps'):
            print(f"\n🧠 REASONING STEPS:")
            for i, step in enumerate(result['solution']['reasoning_steps'], 1):
                print(f"   {i}. {step.get('thought', 'N/A')}")
                if step.get('tool_result'):
                    print(f"      → Result: {step['tool_result']}")
        
        print("=" * 60)
    
    def batch_test(self, problems: List[str]):
        """
        Test the solver with a batch of problems.
        """
        print("\n🧪 BATCH TESTING MODE")
        print("=" * 60)
        
        results = []
        for i, problem in enumerate(problems, 1):
            print(f"\n🔢 PROBLEM {i}/{len(problems)}: {problem}")
            print("-" * 50)
            
            try:
                result = self.solve(problem, detailed_output=False)
                results.append({
                    "problem": problem,
                    "success": result['solution']['status'] == 'solved',
                    "final_answer": result['summary']['final_answer'],
                    "problem_type": result['summary']['problem_type']
                })
                print(f"✅ Solved: {result['summary']['final_answer']}")
                
            except Exception as e:
                results.append({
                    "problem": problem,
                    "success": False,
                    "error": str(e),
                    "problem_type": "unknown"
                })
                print(f"❌ Failed: {str(e)}")
        
        # Summary
        successful = sum(1 for r in results if r['success'])
        print(f"\n📊 BATCH TEST SUMMARY:")
        print(f"   Total problems: {len(problems)}")
        print(f"   Successful: {successful}")
        print(f"   Success rate: {successful/len(problems)*100:.1f}%")
        
        return results

# Example usage and demonstration
if __name__ == "__main__":
    # Initialize the complete solver
    solver = CompleteMathSolver()
    
    # Demo problems covering different categories
    demo_problems = [
        "Solve for x: 2x + 5 = 15",
        "Find the derivative of f(x) = x² + 3x + 2",
        "A rectangle has length 8 cm and width 5 cm. What is its area?",
        "If a train travels 60 mph for 2 hours, how far does it go?",
        "Solve the system of equations: x + y = 10, x - y = 2",
        "Find the integral of 2x dx from 0 to 5",
        "What is the circumference of a circle with radius 4?",
        "Calculate 25% of 200"
    ]
    
    print("\n🎯 DEMONSTRATION MODE")
    print("Choose an option:")
    print("1. Interactive mode (enter your own problems)")
    print("2. Batch test with demo problems")
    print("3. Solve a single specific problem")
    
    try:
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            solver.interactive_mode()
        elif choice == "2":
            solver.batch_test(demo_problems)
        elif choice == "3":
            problem = input("Enter your problem: ").strip()
            if problem:
                result = solver.solve(problem)
                solver._display_solution(result)
        else:
            print("Running interactive mode by default...")
            solver.interactive_mode()
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("Please make sure all dependencies are installed.")
