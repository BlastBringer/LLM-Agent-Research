#!/usr/bin/env python3
"""
🎯 COMPLETE PROBLEM SOLVER - UNIFIED REASONING ENGINE
===================================================

Complete integration of all reasoning components with contextual memory 
tracking and response generation. This is the main entry point for the 
enhanced reasoning engine.
"""

import os
import sys
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Add paths for imports
current_dir = os.path.dirname(__file__)
sys.path.append(current_dir)

try:
    from enhanced_problem_parser import EnhancedProblemParser
    from enhanced_problem_classifier import EnhancedProblemClassifier
    from enhanced_subtask_identifier import EnhancedSubtaskIdentifier
    from enhanced_agent_delegator import EnhancedAgentDelegator
    from contextual_memory_tracker import ContextualMemoryTracker
    from response_generator import ResponseGenerator
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"❌ Component import error: {e}")
    COMPONENTS_AVAILABLE = False

@dataclass
class SolutionResult:
    """Complete solution result from the reasoning engine."""
    success: bool
    final_answer: str
    reasoning_steps: List[Dict[str, Any]]
    confidence: float
    execution_time: float
    method_used: str
    parsed_data: Dict[str, Any]
    classification: Dict[str, Any]
    subtasks: List[Dict[str, Any]]
    delegation_strategy: Dict[str, Any]
    error_message: Optional[str] = None

class CompleteProblemSolver:
    """
    Main unified reasoning engine that coordinates all components with 
    contextual memory tracking and comprehensive response generation.
    """
    
    def __init__(self):
        """Initialize the complete problem solver."""
        print("🎯 Initializing Complete Problem Solver...")
        
        if COMPONENTS_AVAILABLE:
            # Initialize all reasoning components
            self.parser = EnhancedProblemParser()
            self.classifier = EnhancedProblemClassifier()
            self.subtask_identifier = EnhancedSubtaskIdentifier()
            self.agent_delegator = EnhancedAgentDelegator()
            self.memory_tracker = ContextualMemoryTracker()
            self.response_generator = ResponseGenerator()
            
            print("✅ All reasoning components initialized successfully!")
        else:
            print("❌ Components not available - using fallback mode")
            self._initialize_fallback_components()
        
        self.solution_history = []
        print("🎯 Complete Problem Solver ready!")
    
    def _initialize_fallback_components(self):
        """Initialize fallback components when imports fail."""
        self.memory_tracker = MockMemoryTracker()
        self.response_generator = MockResponseGenerator()
    
    def solve_problem(self, problem: str, include_steps: bool = True, 
                     include_metadata: bool = False) -> SolutionResult:
        """
        Complete problem solving with full reasoning pipeline.
        
        Args:
            problem: Mathematical problem to solve
            include_steps: Whether to include detailed reasoning steps
            include_metadata: Whether to include component metadata
            
        Returns:
            Complete solution result with reasoning steps
        """
        start_time = time.time()
        
        print(f"\n🎯 COMPLETE PROBLEM SOLVING")
        print("=" * 50)
        print(f"Problem: {problem}")
        print("=" * 50)
        
        try:
            # Step 1: Parse the problem
            self.memory_tracker.add_step(
                thought="Need to parse and understand the mathematical problem",
                action_taken="Enhanced Problem Parser - parsing problem",
                observation="Parsing problem to extract mathematical components"
            )
            
            print("📋 Step 1: Parsing problem...")
            parsed_result = self.parser.parse(problem)
            
            self.memory_tracker.add_step(
                thought="Problem parsed successfully",
                action_taken=f"Parsed problem with {parsed_result.get('success', False)} success",
                observation=f"Extracted: {parsed_result.get('summary', 'No summary')}"
            )
            
            if not parsed_result.get('success', False):
                return self._create_error_result(
                    "Problem parsing failed", 
                    time.time() - start_time,
                    parsed_result.get('error', 'Unknown parsing error')
                )
            
            print(f"✅ Parse success: {parsed_result.get('success')}")
            
            # Step 2: Classify the problem
            print("🔍 Step 2: Classifying problem...")
            
            self.memory_tracker.add_step(
                thought="Need to classify the type of mathematical problem",
                action_taken="Enhanced Problem Classifier - analyzing problem type",
                observation="Determining mathematical category and subcategory"
            )
            
            classification_result = self.classifier.classify_detailed(problem)
            
            self.memory_tracker.add_step(
                thought="Problem classification completed",
                action_taken=f"Classified as {classification_result.get('primary_category', 'unknown')}",
                observation=f"Confidence: {classification_result.get('confidence', 0):.2f}"
            )
            
            print(f"✅ Classification: {classification_result.get('primary_category', 'unknown')}")
            
            # Step 3: Identify subtasks
            print("🎯 Step 3: Identifying subtasks...")
            
            self.memory_tracker.add_step(
                thought="Need to break down problem into manageable subtasks",
                action_taken="Enhanced Subtask Identifier - analyzing problem complexity",
                observation="Identifying required subtasks for solution"
            )
            
            subtasks_result = self.subtask_identifier.identify_subtasks(
                problem, 
                classification_result.get('primary_category', 'general')
            )
            
            # subtasks_result is a List[Subtask], not a dict
            subtasks_count = len(subtasks_result)
            
            self.memory_tracker.add_step(
                thought="Subtask identification completed",
                action_taken=f"Identified {subtasks_count} subtasks",
                observation=f"Subtasks: {[st.type for st in subtasks_result]}"
            )
            
            print(f"✅ Found {subtasks_count} subtasks")
            
            # Step 4: Make delegation decision
            print("🚦 Step 4: Making delegation decision...")
            
            self.memory_tracker.add_step(
                thought="Need to decide on execution strategy for solving the problem",
                action_taken="Enhanced Agent Delegator - analyzing delegation options",
                observation="Determining optimal execution strategy"
            )
            
            delegation_result = self.agent_delegator.make_delegation_decision(
                problem,
                {'subtasks': subtasks_result},  # Wrap list in dict for compatibility
                classification_result.get('primary_category', 'general')
            )
            
            strategy = delegation_result.get('strategy', 'unknown')
            
            self.memory_tracker.add_step(
                thought="Delegation strategy determined",
                action_taken=f"Selected strategy: {strategy}",
                observation=f"Reasoning: {delegation_result.get('reasoning', 'No reasoning provided')}"
            )
            
            print(f"✅ Strategy: {strategy}")
            
            # Step 5: Execute the strategy
            print("⚡ Step 5: Executing strategy...")
            
            self.memory_tracker.add_step(
                thought=f"Executing {strategy} strategy to solve the problem",
                action_taken=f"Strategy execution via {strategy}",
                observation="Beginning strategy execution"
            )
            
            execution_result = self.agent_delegator.execute_delegation_strategy(
                delegation_result.get('strategy', 'internal_react'),
                problem,
                {
                    'parsed_data': parsed_result,
                    'classification': classification_result,
                    'subtasks': {'subtasks': subtasks_result}  # Wrap list in dict for compatibility
                }
            )
            
            self.memory_tracker.add_step(
                thought="Strategy execution completed",
                action_taken=f"Execution result: {execution_result.get('success', False)}",
                observation=f"Solution: {execution_result.get('result', 'No result')[:100]}..."
            )
            
            print(f"✅ Execution completed!")
            
            # Step 6: Generate final response
            execution_time = time.time() - start_time
            
            self.memory_tracker.add_step(
                thought="Generating comprehensive final response",
                action_taken="Response Generator - formatting final answer",
                observation=f"Total execution time: {execution_time:.2f}s"
            )
            
            # Get full history for response generation
            history_log = self.memory_tracker.get_full_history()
            
            # Generate formatted response
            if include_steps:
                formatted_response = self.response_generator.generate_response(
                    execution_result.get('result', 'No solution found'),
                    history_log,
                    include_metadata
                )
            else:
                formatted_response = self.response_generator.generate_summary_response(
                    execution_result.get('result', 'No solution found'),
                    history_log
                )
            
            # Create complete solution result
            solution_result = SolutionResult(
                success=execution_result.get('success', False),
                final_answer=formatted_response,
                reasoning_steps=history_log,
                confidence=execution_result.get('confidence', 0.7),
                execution_time=execution_time,
                method_used=strategy,
                parsed_data=parsed_result,
                classification=classification_result,
                subtasks=subtasks_result,
                delegation_strategy=delegation_result
            )
            
            # Store in history
            self.solution_history.append(solution_result)
            
            print(f"📊 FINAL RESULT:")
            print(f"   Success: {'✅ YES' if solution_result.success else '❌ NO'}")
            print(f"   Method: {solution_result.method_used}")
            print(f"   Confidence: {solution_result.confidence:.2f}")
            print(f"   Time: {solution_result.execution_time:.2f}s")
            
            return solution_result
            
        except Exception as e:
            error_time = time.time() - start_time
            error_msg = f"Complete problem solving error: {str(e)}"
            print(f"❌ {error_msg}")
            
            self.memory_tracker.add_step(
                thought="Critical error occurred during problem solving",
                action_taken=f"Error handling: {str(e)}",
                observation="Problem solving failed"
            )
            
            return self._create_error_result(error_msg, error_time, str(e))
    
    def _create_error_result(self, message: str, execution_time: float, 
                           error_details: str) -> SolutionResult:
        """Create an error result."""
        history_log = self.memory_tracker.get_full_history()
        
        error_response = self.response_generator.generate_response(
            f"Error: {message}",
            history_log,
            False
        )
        
        return SolutionResult(
            success=False,
            final_answer=error_response,
            reasoning_steps=history_log,
            confidence=0.0,
            execution_time=execution_time,
            method_used="error_handling",
            parsed_data={},
            classification={},
            subtasks=[],
            delegation_strategy={},
            error_message=error_details
        )
    
    def solve_multiple_problems(self, problems: List[str], 
                              include_steps: bool = True) -> List[SolutionResult]:
        """
        Solve multiple problems in sequence.
        
        Args:
            problems: List of mathematical problems
            include_steps: Whether to include detailed steps
            
        Returns:
            List of solution results
        """
        results = []
        
        print(f"\n🎯 SOLVING {len(problems)} PROBLEMS")
        print("=" * 40)
        
        for i, problem in enumerate(problems, 1):
            print(f"\n🔢 Problem {i}/{len(problems)}:")
            
            # Reset memory tracker for each problem
            self.memory_tracker = ContextualMemoryTracker()
            
            result = self.solve_problem(problem, include_steps)
            results.append(result)
            
            print(f"Problem {i} completed: {'✅' if result.success else '❌'}")
        
        return results
    
    def get_solution_history(self) -> List[SolutionResult]:
        """Get history of all solved problems."""
        return self.solution_history.copy()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for solved problems."""
        if not self.solution_history:
            return {"total_problems": 0, "success_rate": 0.0}
        
        total = len(self.solution_history)
        successful = sum(1 for result in self.solution_history if result.success)
        
        avg_time = sum(result.execution_time for result in self.solution_history) / total
        avg_confidence = sum(result.confidence for result in self.solution_history) / total
        
        methods_used = {}
        for result in self.solution_history:
            method = result.method_used
            methods_used[method] = methods_used.get(method, 0) + 1
        
        return {
            "total_problems": total,
            "successful_problems": successful,
            "success_rate": successful / total,
            "average_execution_time": avg_time,
            "average_confidence": avg_confidence,
            "methods_used": methods_used
        }

class MockMemoryTracker:
    """Mock memory tracker for fallback mode."""
    def __init__(self):
        self.history = []
    
    def add_step(self, thought: str, action_taken: str, observation: str, metadata=None):
        self.history.append({
            "thought": thought,
            "action": action_taken,
            "observation": observation
        })
    
    def get_full_history(self):
        return self.history

class MockResponseGenerator:
    """Mock response generator for fallback mode."""
    def generate_response(self, final_answer, history_log, include_metadata=False):
        return f"Mock response: {final_answer}"
    
    def generate_summary_response(self, final_answer, history_log):
        return f"Mock summary: {final_answer}"

# Global instance
complete_solver = CompleteProblemSolver()

def solve_math_problem(problem: str, include_steps: bool = True, 
                      include_metadata: bool = False) -> SolutionResult:
    """
    Convenience function to solve a mathematical problem.
    
    Args:
        problem: Mathematical problem to solve
        include_steps: Whether to include detailed reasoning steps
        include_metadata: Whether to include component metadata
        
    Returns:
        Complete solution result
    """
    return complete_solver.solve_problem(problem, include_steps, include_metadata)

def get_solver() -> CompleteProblemSolver:
    """Get the global complete solver instance."""
    return complete_solver

if __name__ == "__main__":
    print("🎯 Complete Problem Solver Testing")
    print("=" * 40)
    
    # Test problems
    test_problems = [
        "Simplify the expression: (3x + 2x) - (x - 4)",
        "Solve for x: 2x + 5 = 13",
        "Find the derivative of f(x) = x^2 + 3x + 2"
    ]
    
    print(f"\n🧮 Testing with {len(test_problems)} problems...")
    
    for problem in test_problems:
        print(f"\n" + "="*60)
        result = solve_math_problem(problem, include_steps=True)
        
        if result.success:
            print(f"✅ SUCCESS: {problem}")
        else:
            print(f"❌ FAILED: {problem}")
            print(f"Error: {result.error_message}")
    
    # Show performance metrics
    metrics = complete_solver.get_performance_metrics()
    print(f"\n📊 Performance Metrics:")
    print(f"   Success Rate: {metrics['success_rate']:.1%}")
    print(f"   Average Time: {metrics['average_execution_time']:.2f}s")
    print(f"   Average Confidence: {metrics['average_confidence']:.2f}")
    
    print(f"\n✅ Complete Problem Solver testing completed!")
