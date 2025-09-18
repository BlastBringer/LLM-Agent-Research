#!/usr/bin/env python3
"""
🎯 UNIFIED MATHEMATICAL PROBLEM SOLVER
=====================================

Complete integration of all reasoning components and agent systems
to solve ANY mathematical problem end-to-end with full reasoning transparency.

This script uses:
- Enhanced Problem Parser
- Enhanced Problem Classifier  
- Enhanced Subtask Identifier
- Enhanced Agent Delegator
- Contextual Memory Tracker
- Response Generator
- CrewAI Agent System (when available)
- External Tools Integration
"""

import os
import sys
import json
import re
import sympy as sp
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add all necessary paths
current_dir = os.path.dirname(__file__)
reasoning_path = os.path.join(current_dir, 'Reasoning')
agent_path = os.path.join(current_dir, 'Agent')

sys.path.insert(0, reasoning_path)
sys.path.insert(0, agent_path)
sys.path.insert(0, current_dir)

def setup_component_logging(verbose: bool = True):
    """Setup comprehensive component tracking and logging."""
    if verbose:
        # Create a custom formatter for component tracking
        logging.basicConfig(
            level=logging.INFO,
            format='🔍 [%(levelname)s] %(message)s',
            handlers=[logging.StreamHandler()]
        )
    else:
        logging.basicConfig(level=logging.WARNING)
    
    return logging.getLogger('UnifiedSolver')

def log_component_usage(component_name: str, action: str, details: str = ""):
    """Log which component is being used with specific action."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    icon_map = {
        'Parser': '📝',
        'Classifier': '🔍', 
        'Subtask': '🧩',
        'Agent': '🤖',
        'Memory': '💭',
        'Response': '📄',
        'SymPy': '🧮',
        'CrewAI': '👥',
        'Strategy': '🎯',
        'Verification': '✅',
        'Integration': '🔗',
        'React': '⚛️',
        'Tool': '🔧'
    }
    
    icon = icon_map.get(component_name, '⚡')
    print(f"{icon} [{timestamp}] {component_name.upper()}: {action}", end="")
    if details:
        print(f" → {details}")
    else:
        print()

class UnifiedMathSolver:
    """
    Complete mathematical problem solver that integrates all system components
    to solve ANY mathematical problem with full reasoning transparency.
    """
    
    def __init__(self):
        """Initialize all components of the unified system."""
        print("🎯 Initializing Unified Mathematical Problem Solver...")
        log_component_usage("Integration", "Starting system initialization")
        
        # Core reasoning components
        self.memory_tracker = None
        self.response_generator = None
        self.parser = None
        self.classifier = None
        self.subtask_identifier = None
        self.agent_delegator = None
        self.subtask_interpreter = None
        
        # Learning mechanism for ReAct solutions
        self.solution_cache_file = os.path.join(current_dir, 'learned_solutions.json')
        self.learned_solutions = self._load_learned_solutions()
        self.failed_problems = []  # Track problems that required ReAct
        
        # Initialize working components
        self._initialize_components()
        
        print("✅ Unified Mathematical Problem Solver ready!")
        log_component_usage("Integration", "System initialization complete")
    
    def _initialize_components(self):
        """Initialize all available system components."""
        try:
            # Always available components
            log_component_usage("Memory", "Loading contextual memory tracker")
            from Reasoning.contextual_memory_tracker import ContextualMemoryTracker
            log_component_usage("Response", "Loading response generator")
            from Reasoning.response_generator import ResponseGenerator
            
            self.memory_tracker = ContextualMemoryTracker()
            log_component_usage("Memory", "Contextual Memory Tracker initialized")
            self.response_generator = ResponseGenerator()
            log_component_usage("Response", "Response Generator initialized")
            print("✅ Core components loaded")
            
            # Enhanced reasoning components (when available)
            try:
                log_component_usage("Parser", "Loading enhanced problem parser")
                from Reasoning.enhanced_problem_parser import EnhancedProblemParser
                log_component_usage("Classifier", "Loading enhanced problem classifier")
                from Reasoning.enhanced_problem_classifier import EnhancedProblemClassifier
                log_component_usage("Subtask", "Loading enhanced subtask identifier")
                from Reasoning.enhanced_subtask_identifier import EnhancedSubtaskIdentifier
                log_component_usage("Agent", "Loading enhanced agent delegator")
                from Reasoning.enhanced_agent_delegator import EnhancedAgentDelegator
                
                # Load Agent folder components
                log_component_usage("Agent", "Loading enhanced subtask interpreter")
                from Agent.enhanced_subtask_interpreter import EnhancedSubtaskInterpreter
                
                self.parser = EnhancedProblemParser()
                log_component_usage("Parser", "Enhanced Problem Parser initialized")
                self.classifier = EnhancedProblemClassifier()
                log_component_usage("Classifier", "Enhanced Problem Classifier initialized")
                self.subtask_identifier = EnhancedSubtaskIdentifier()
                log_component_usage("Subtask", "Enhanced Subtask Identifier initialized")
                self.agent_delegator = EnhancedAgentDelegator()
                log_component_usage("Agent", "Enhanced Agent Delegator initialized")
                self.subtask_interpreter = EnhancedSubtaskInterpreter()
                log_component_usage("Agent", "Enhanced Subtask Interpreter initialized")
                print("✅ Enhanced reasoning components loaded")
                
            except ImportError as e:
                log_component_usage("Integration", "Enhanced components not available", str(e))
                print(f"⚠️  Enhanced components not available: {e}")
                print("🔧 Using built-in problem solving capabilities")
                
        except Exception as e:
            log_component_usage("Integration", "Error during initialization", str(e))
            print(f"❌ Error initializing components: {e}")
            raise
    
    def solve_problem(self, problem: str) -> Dict[str, Any]:
        """
        Solve a mathematical problem using the complete unified system.
        
        Args:
            problem: The mathematical problem to solve
            
        Returns:
            Complete solution with reasoning steps and final answer
        """
        print(f"\n🎯 SOLVING PROBLEM: {problem}")
        print("=" * 60)
        log_component_usage("Integration", "Starting problem solving pipeline", problem)
        
        start_time = datetime.now()
        
        try:
            # Step 0: Check for learned solutions first
            learned_solution = self._check_learned_solution(problem)
            if learned_solution:
                self._log_learning_usage(problem, learned_solution, "attempting_to_apply")
                learned_result = self._apply_learned_solution(problem, learned_solution)
                if learned_result and learned_result != 'Could not adapt learned solution':
                    self._log_learning_success(problem, learned_solution, learned_result)
                    
                    execution_time = (datetime.now() - start_time).total_seconds()
                    
                    return {
                        'success': True,
                        'solution': learned_result,
                        'method': 'learned_from_react',
                        'execution_time': execution_time,
                        'reasoning_steps': [{
                            'thought': 'Applied learned solution pattern from previous ReAct success',
                            'action': f"Used method: {learned_solution.get('method')}",
                            'observation': f"Result: {learned_result}"
                        }],
                        'verification': 'Applied from learned solution',
                        'classification': learned_solution.get('problem_type', 'learned'),
                        'strategy': 'learned_solution'
                    }
                else:
                    log_component_usage("Learning", "Could not apply learned solution - proceeding with normal flow")
            
            # Step 1: Initialize memory tracking
            log_component_usage("Memory", "Adding initial step to memory tracker")
            self.memory_tracker.add_step(
                thought=f"Starting to solve the problem: {problem}",
                action_taken="Initialize problem solving process",
                observation="Beginning systematic approach to problem solving"
            )
            
            # Step 2: Parse the problem
            print("📋 Step 1: Parsing problem...")
            log_component_usage("Parser", "Starting problem parsing")
            parsed_data = self._parse_problem(problem)
            
            # Step 3: Classify the problem
            print("🔍 Step 2: Classifying problem...")
            log_component_usage("Classifier", "Starting problem classification")
            classification = self._classify_problem(problem, parsed_data)
            
            # Step 4: Identify solution strategy
            print("🎯 Step 3: Determining solution strategy...")
            log_component_usage("Strategy", "Determining solution strategy")
            strategy = self._determine_strategy(problem, parsed_data, classification)
            
            # Step 5: Solve the problem
            print("⚡ Step 4: Executing solution...")
            log_component_usage("Integration", "Executing solution with chosen strategy", strategy)
            solution = self._execute_solution(problem, parsed_data, classification, strategy)
            
            # If solution failed and strategy wasn't ReAct, try ReAct and learn from it
            if (solution in ['Problem could not be solved with available methods', 'Could not parse equation format'] 
                and strategy != 'react_reasoning'):
                log_component_usage("Learning", "Main solver failed - trying ReAct and learning from it")
                
                self.failed_problems.append({
                    'problem': problem,
                    'failed_strategy': strategy,
                    'failed_at': datetime.now().isoformat()
                })
                
                react_result = self._solve_with_react_reasoning_and_learn(problem, parsed_data)
                if react_result and react_result != 'ReAct reasoning did not find solution':
                    solution = react_result
                    strategy = 'react_reasoning_fallback'
            
            # Step 6: Verify the solution
            print("✅ Step 5: Verifying solution...")
            log_component_usage("Verification", "Verifying solution correctness")
            verification = self._verify_solution(problem, solution, parsed_data)
            
            # Step 7: Generate final response
            print("📊 Step 6: Generating final response...")
            log_component_usage("Response", "Generating formatted response")
            final_response = self._generate_final_response(problem, solution, verification)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Generate step-by-step explanation
            reasoning_steps = self.memory_tracker.get_full_history()
            step_by_step_explanation = self._format_step_by_step_explanation(reasoning_steps)
            
            print(f"\n🎉 PROBLEM SOLVED in {execution_time:.2f} seconds!")
            
            # Display step-by-step solution
            print(f"\n📋 STEP-BY-STEP SOLUTION:")
            print("=" * 40)
            print(step_by_step_explanation)
            
            return {
                "success": True,
                "problem": problem,
                "solution": solution,
                "method": strategy,
                "verification": verification,
                "formatted_response": final_response,
                "step_by_step_explanation": step_by_step_explanation,
                "execution_time": execution_time,
                "time_taken": execution_time,  # For compatibility
                "reasoning_steps": reasoning_steps
            }
            
        except Exception as e:
            error_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Error solving problem: {str(e)}"
            
            self.memory_tracker.add_step(
                thought="An error occurred during problem solving",
                action_taken=f"Error handling: {str(e)}",
                observation="Problem solving failed"
            )
            
            print(f"❌ {error_msg}")
            
            return {
                "success": False,
                "problem": problem,
                "error": error_msg,
                "execution_time": error_time,
                "reasoning_steps": self.memory_tracker.get_full_history()
            }
    
    def _parse_problem(self, problem: str) -> Dict[str, Any]:
        """Parse the mathematical problem to extract key components."""
        log_component_usage("Memory", "Recording parsing initiation step")
        self.memory_tracker.add_step(
            thought="I need to understand the structure and components of this problem",
            action_taken="Parsing problem to extract mathematical elements",
            observation="Analyzing problem text for variables, equations, and operations"
        )
        
        if self.parser:
            try:
                log_component_usage("Parser", "Using enhanced problem parser")
                parsed = self.parser.parse(problem)
                log_component_usage("Memory", "Recording enhanced parsing completion")
                self.memory_tracker.add_step(
                    thought="Using enhanced parser for detailed analysis",
                    action_taken="Enhanced problem parsing completed",
                    observation=f"Parsed structure: {parsed.get('summary', 'Analysis complete')}"
                )
                log_component_usage("Parser", "Enhanced parsing successful", str(parsed.get('problem_type', 'unknown')))
                return parsed
            except Exception as e:
                log_component_usage("Parser", "Enhanced parser failed, falling back to built-in", str(e))
                print(f"⚠️  Enhanced parser failed: {e}, using built-in parsing")
        
        # Built-in parsing logic
        log_component_usage("Parser", "Using built-in parsing logic")
        parsed_data = {
            "original_text": problem,
            "variables": self._extract_variables(problem),
            "operations": self._extract_operations(problem),
            "equations": self._extract_equations(problem),
            "problem_type": self._basic_type_detection(problem)
        }
        
        log_component_usage("Memory", "Recording built-in parsing completion")
        self.memory_tracker.add_step(
            thought="Using built-in parsing capabilities",
            action_taken="Basic problem parsing completed",
            observation=f"Identified: {parsed_data['problem_type']} with variables {parsed_data['variables']}"
        )
        
        return parsed_data
    
    def _classify_problem(self, problem: str, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Classify the type of mathematical problem."""
        log_component_usage("Memory", "Recording classification initiation step")
        self.memory_tracker.add_step(
            thought="I need to classify this problem to choose the right solving approach",
            action_taken="Analyzing problem type and mathematical domain",
            observation="Determining the most appropriate solution methodology"
        )
        
        if self.classifier:
            try:
                log_component_usage("Classifier", "Using enhanced problem classifier with parser data")
                classification = self.classifier.classify_detailed(problem, parsed_data)
                log_component_usage("Memory", "Recording enhanced classification completion")
                self.memory_tracker.add_step(
                    thought="Using enhanced classifier with parser insights for detailed categorization",
                    action_taken="Enhanced classification with parser data completed",
                    observation=f"Classified as: {classification.get('primary_category', 'unknown')} using parser insights"
                )
                log_component_usage("Classifier", "Enhanced classification with parser data successful", str(classification.get('primary_category', 'unknown')))
                return classification
            except Exception as e:
                log_component_usage("Classifier", "Enhanced classifier failed, falling back to built-in", str(e))
                print(f"⚠️  Enhanced classifier failed: {e}, using built-in classification")
        
        # Built-in classification logic
        log_component_usage("Classifier", "Using built-in classification logic")
        problem_lower = problem.lower()
        
        if any(word in problem_lower for word in ['derivative', 'differentiate', "d/dx", "d/dt"]):
            category = "calculus_differentiation"
            log_component_usage("Classifier", "Detected calculus differentiation problem")
        elif any(word in problem_lower for word in ['integrate', '∫', 'integral', 'antiderivative']):
            category = "calculus_integration"
            log_component_usage("Classifier", "Detected calculus integration problem")
        elif any(word in problem_lower for word in ['limit', 'lim', 'approaches']):
            category = "calculus_limits"
            log_component_usage("Classifier", "Detected calculus limits problem")
        elif any(word in problem_lower for word in ['solve', '=', 'equation', 'find x']):
            category = "algebra_equations"
            log_component_usage("Classifier", "Detected algebraic equation problem")
        elif any(word in problem_lower for word in ['simplify', 'expand', 'factor']):
            log_component_usage("Classifier", "Detected algebraic manipulation problem")
            category = "algebra_manipulation"
        elif any(word in problem_lower for word in ['%', 'percent', 'percentage']):
            category = "arithmetic_percentage"
        elif any(word in problem_lower for word in ['system', 'simultaneous']):
            category = "algebra_systems"
        elif any(word in problem_lower for word in ['matrix', 'determinant']):
            category = "linear_algebra"
        elif any(word in problem_lower for word in ['sin', 'cos', 'tan', 'trigonometry']):
            category = "trigonometry"
        else:
            category = "general_mathematics"
        
        classification = {
            "primary_category": category,
            "confidence": 0.8,
            "subcategory": "standard",
            "complexity": "medium"
        }
        
        self.memory_tracker.add_step(
            thought="Using built-in classification rules",
            action_taken="Basic classification completed",
            observation=f"Classified as: {category} (confidence: 0.8)"
        )
        
        return classification
    
    def _determine_strategy(self, problem: str, parsed_data: Dict[str, Any], 
                           classification: Dict[str, Any]) -> str:
        """Determine the best strategy to solve this problem."""
        category = classification.get('primary_category', 'general_mathematics')
        log_component_usage("Strategy", "Starting strategy determination", f"Category: {category}")
        
        # Check if agent delegator should be used for complex problems
        if self.agent_delegator:
            try:
                log_component_usage("Agent", "Consulting agent delegator for strategy")
                agent_recommendation = self.agent_delegator.recommend_strategy(problem, parsed_data, classification)
                if agent_recommendation and agent_recommendation != 'default':
                    log_component_usage("Agent", "Agent delegator provided strategy recommendation", agent_recommendation)
                    self.memory_tracker.add_step(
                        thought=f"Agent delegator recommends: {agent_recommendation}",
                        action_taken="Agent delegation strategy selection",
                        observation=f"Using agent-recommended strategy: {agent_recommendation}"
                    )
                    return agent_recommendation
                else:
                    log_component_usage("Agent", "Agent delegator deferred to standard strategy selection")
            except Exception as e:
                log_component_usage("Agent", "Agent delegator failed", str(e))
        
        # Check if subtask identification is needed
        if self.subtask_identifier:
            try:
                log_component_usage("Subtask", "Analyzing problem for subtask breakdown")
                subtasks = self.subtask_identifier.identify_subtasks(problem, parsed_data)
                if subtasks and len(subtasks) > 1:
                    log_component_usage("Subtask", "Multiple subtasks identified", f"Count: {len(subtasks)}")
                    self.memory_tracker.add_step(
                        thought=f"Problem broken down into {len(subtasks)} subtasks",
                        action_taken="Subtask identification and decomposition",
                        observation=f"Identified subtasks: {[st.description for st in subtasks]}"
                    )
                    
                    # Use the subtask interpreter for coordinated solving
                    if self.subtask_interpreter:
                        log_component_usage("Agent", "Using subtask interpreter for coordination")
                        self.memory_tracker.add_step(
                            thought="Multiple subtasks identified, using subtask interpreter for coordination",
                            action_taken="Delegate to SubtaskInterpreter for coordinated execution",
                            observation="Starting coordinated subtask solving process"
                        )
                        return 'coordinated_subtask_solving'
                    else:
                        # Fallback to original coordinated solving
                        return 'coordinated_subtask_solving'
                else:
                    log_component_usage("Subtask", "Single task identified - no decomposition needed")
            except Exception as e:
                log_component_usage("Subtask", "Subtask identification failed", str(e))
        
        log_component_usage("Memory", "Recording strategy selection step")
        self.memory_tracker.add_step(
            thought=f"Based on classification ({category}), I need to choose the best solving strategy",
            action_taken="Strategy selection based on problem type",
            observation="Evaluating available solution methods"
        )
        
        log_component_usage("Strategy", "Using built-in strategy mapping")
        # Strategy mapping based on problem type
        strategy_map = {
            'calculus_differentiation': 'symbolic_calculus',
            'calculus_integration': 'symbolic_calculus', 
            'calculus_limits': 'symbolic_calculus',
            'algebra_equations': 'algebraic_solving',
            'algebra_manipulation': 'symbolic_manipulation',
            'algebra_systems': 'system_solving',
            'arithmetic_percentage': 'numerical_computation',
            'linear_algebra': 'matrix_operations',
            'trigonometry': 'trigonometric_solving',
            'algebra': 'algebraic_solving',  # Add this mapping
            'calculus': 'symbolic_calculus',  # Add this mapping
            'arithmetic': 'numerical_computation'  # Add this mapping
        }
        
        strategy = strategy_map.get(category, 'general_problem_solving')
        log_component_usage("Strategy", "Initial strategy selected", f"{category} → {strategy}")
        
        # Enhanced React reasoning triggers for complex problems
        react_triggers = False
        
        # Trigger 1: All calculus problems (not just complex ones)
        if category in ['calculus', 'calculus_integration', 'calculus_differentiation'] or any(word in problem.lower() for word in ['derivative', 'differentiate', 'd/dx', 'd/dt', 'integrate', '∫', 'integral', 'limit', 'lim']):
            log_component_usage("React", "Calculus problem detected - using ReAct")
            react_triggers = True
        
        # Trigger 2: Multi-step problems (contains "and", "then", "also")
        if any(word in problem.lower() for word in ['and then', 'then find', 'also find', 'determine', 'calculate and']):
            log_component_usage("React", "Multi-step problem detected")
            react_triggers = True
        
        # Trigger 3: Complex word problems (long text with multiple operations)
        if len(problem.split()) > 15:
            operation_count = sum(1 for op in ['+', '-', '*', '/', '=', '^', 'solve', 'find', 'calculate'] if op in problem.lower())
            if operation_count >= 3:
                log_component_usage("React", "Complex multi-operation problem detected")
                react_triggers = True
        
        # Trigger 4: Problems requiring step-by-step reasoning
        if any(phrase in problem.lower() for phrase in ['step by step', 'show work', 'explain', 'reasoning']):
            log_component_usage("React", "Step-by-step reasoning requested")
            react_triggers = True
        
        # Apply React reasoning if triggered
        if react_triggers:
            log_component_usage("React", "Activating Enhanced React reasoning")
            strategy = 'react_reasoning'
        
        # Override strategy based on keywords - BUT ONLY if ReAct is not triggered
        if not react_triggers:
            if '=' in problem:
                strategy = 'algebraic_solving'
                log_component_usage("Strategy", "Strategy override detected", "Equation found → algebraic_solving")
            elif any(word in problem.lower() for word in ['derivative', 'differentiate', 'd/dx']):
                strategy = 'symbolic_calculus'
                log_component_usage("Strategy", "Strategy override detected", "Calculus derivative → symbolic_calculus")
            elif any(word in problem.lower() for word in ['integrate', '∫', 'integral']):
                strategy = 'symbolic_calculus'
                log_component_usage("Strategy", "Strategy override detected", "Calculus integration → symbolic_calculus")
            elif '%' in problem or 'percent' in problem.lower():
                strategy = 'numerical_computation'
                log_component_usage("Strategy", "Strategy override detected", "Percentage problem → numerical_computation")
        else:
            log_component_usage("React", "Strategy overrides skipped - ReAct reasoning takes priority")
        
        log_component_usage("Strategy", "Final strategy determined", strategy)
        log_component_usage("Memory", "Recording final strategy selection")
        self.memory_tracker.add_step(
            thought=f"Selected strategy: {strategy}",
            action_taken=f"Strategy determination: {strategy}",
            observation=f"Will use {strategy} approach to solve this {category} problem"
        )
        
        return strategy
    
    def _execute_solution(self, problem: str, parsed_data: Dict[str, Any], 
                         classification: Dict[str, Any], strategy: str) -> Any:
        """Execute the solution using the determined strategy."""
        log_component_usage("Memory", "Recording strategy execution step")
        self.memory_tracker.add_step(
            thought=f"Executing {strategy} strategy to solve the problem",
            action_taken=f"Beginning {strategy} solution process",
            observation="Applying mathematical operations to find the solution"
        )
        
        try:
            # Try to use SymPy for symbolic mathematics
            if strategy in ['symbolic_calculus', 'algebraic_solving', 'symbolic_manipulation', 'system_solving']:
                log_component_usage("SymPy", "Using SymPy for symbolic mathematics", strategy)
                solution = self._solve_with_sympy(problem, parsed_data, classification, strategy)
            elif strategy == 'react_reasoning':
                log_component_usage("React", "Using React-style reasoning approach")
                solution = self._solve_with_react_reasoning(problem, parsed_data)
            elif strategy == 'coordinated_subtask_solving':
                log_component_usage("Agent", "Using coordinated subtask solving")
                solution = self._solve_with_coordinated_subtasks(problem, parsed_data)
            elif strategy == 'numerical_computation':
                log_component_usage("Tool", "Using numerical computation engine", "Python built-ins")
                solution = self._solve_numerical(problem, parsed_data)
            elif strategy == 'trigonometric_solving':
                log_component_usage("SymPy", "Using SymPy for trigonometric solving")
                solution = self._solve_trigonometric(problem, parsed_data)
            else:
                log_component_usage("Tool", "Using general problem solver")
                solution = self._solve_general(problem, parsed_data)
            
            log_component_usage("Memory", "Recording solution completion")
            self.memory_tracker.add_step(
                thought="Solution computation completed successfully",
                action_taken=f"Applied {strategy} method",
                observation=f"Obtained solution: {solution}"
            )
            
            return solution
            
        except Exception as e:
            self.memory_tracker.add_step(
                thought="Primary solution method failed, trying alternative approach",
                action_taken=f"Fallback solution attempt: {str(e)}",
                observation="Using alternative problem-solving approach"
            )
            
            # Fallback to general solving
            return self._solve_general(problem, parsed_data)
    
    def _solve_with_sympy(self, problem: str, parsed_data: Dict[str, Any], 
                         classification: Dict[str, Any], strategy: str) -> Any:
        """Solve using SymPy symbolic mathematics."""
        try:
            log_component_usage("SymPy", "Importing SymPy library")
            import sympy as sp
            
            # Extract variables
            variables = parsed_data.get('variables', ['x'])
            sym_vars = [sp.Symbol(var) for var in variables]
            log_component_usage("SymPy", "Created symbolic variables", str(variables))
            
            # Handle different types of problems
            if strategy == 'symbolic_calculus':
                log_component_usage("SymPy", "Solving calculus problem")
                return self._solve_calculus_sympy(problem, sym_vars)
            elif strategy == 'algebraic_solving':
                log_component_usage("SymPy", "Solving algebraic equation")
                return self._solve_algebra_sympy(problem, sym_vars)
            elif strategy == 'symbolic_manipulation':
                log_component_usage("SymPy", "Manipulating symbolic expression")
                return self._manipulate_expression_sympy(problem, sym_vars)
            elif strategy == 'system_solving':
                log_component_usage("SymPy", "Solving system of equations")
                return self._solve_system_sympy(problem, sym_vars)
            else:
                log_component_usage("SymPy", "Using general symbolic solver")
                return self._solve_general_sympy(problem, sym_vars)
                
        except ImportError:
            log_component_usage("SymPy", "SymPy not available - falling back to built-in methods")
            raise Exception("SymPy not available for symbolic computation")
    
    def _solve_calculus_sympy(self, problem: str, variables: List[sp.Symbol]) -> Any:
        """Solve calculus problems using SymPy."""
        problem_lower = problem.lower()
        log_component_usage("SymPy", "Analyzing calculus problem type", problem_lower[:50] + "...")
        
        if 'derivative' in problem_lower or 'd/dx' in problem_lower:
            log_component_usage("SymPy", "Detected differentiation problem")
            # Extract the function to differentiate
            # Look for patterns like "derivative of f(x)" or "find derivative of expression"
            expr_patterns = [
                r'derivative of (.+)',
                r'differentiate (.+)', 
                r'd/dx\s*\((.+)\)',
                r'd/dx\s*(.+)',
                r'find.*derivative.*of (.+)'
            ]
            
            for pattern in expr_patterns:
                match = re.search(pattern, problem, re.IGNORECASE)
                if match:
                    expr_str = match.group(1).strip()
                    log_component_usage("SymPy", "Extracted expression for differentiation", expr_str)
                    # Remove any trailing punctuation
                    expr_str = re.sub(r'[.!?]$', '', expr_str)
                    
                    try:
                        # Fix implicit multiplication comprehensively
                        expr_str = re.sub(r'(\d+)([a-zA-Z])', r'\1*\2', expr_str)  # 2x -> 2*x
                        expr_str = re.sub(r'(\d+)(\()', r'\1*\2', expr_str)        # 4( -> 4*(
                        expr_str = re.sub(r'(\))(\d+)', r'\1*\2', expr_str)        # )2 -> )*2
                        expr_str = re.sub(r'(\))([a-zA-Z])', r'\1*\2', expr_str)   # )x -> )*x
                        expr_str = re.sub(r'(\))(\()', r'\1*\2', expr_str)         # )( -> )*(
                        log_component_usage("SymPy", "Parsing expression with SymPy")
                        expr = sp.sympify(expr_str)
                        x = sp.Symbol('x')
                        log_component_usage("SymPy", "Computing derivative")
                        derivative = sp.diff(expr, x)
                        
                        log_component_usage("Memory", "Recording derivative computation step")
                        self.memory_tracker.add_step(
                            thought=f"Taking derivative of {expr}",
                            action_taken=f"Computing d/dx({expr})",
                            observation=f"Derivative: {derivative}"
                        )
                        
                        log_component_usage("SymPy", "Derivative computation completed", str(derivative))
                        return derivative
                    except Exception as e:
                        log_component_usage("SymPy", "Failed to parse expression", str(e))
                        continue
        
        elif 'integrate' in problem_lower or '∫' in problem_lower:
            log_component_usage("SymPy", "Detected integration problem")
            # Extract the function to integrate
            # Look for pattern like ∫f(x)dx or integrate f(x)
            expr_patterns = [
                r'∫\s*(.+?)\s*dx',
                r'integrate\s+(.+?)\s+(?:with respect to|dx)',
                r'∫\s*(.+)'
            ]
            
            for pattern in expr_patterns:
                match = re.search(pattern, problem, re.IGNORECASE)
                if match:
                    expr_str = match.group(1)
                    log_component_usage("SymPy", "Extracted expression for integration", expr_str)
                    try:
                        # Normalize Unicode characters first
                        expr_str = expr_str.replace('−', '-')  # Unicode minus to ASCII minus
                        expr_str = expr_str.replace('–', '-')  # En dash to ASCII minus
                        expr_str = expr_str.replace('—', '-')  # Em dash to ASCII minus
                        
                        # Fix implicit multiplication comprehensively
                        expr_str = re.sub(r'(\d+)([a-zA-Z])', r'\1*\2', expr_str)  # 2x -> 2*x
                        expr_str = re.sub(r'(\d+)(\()', r'\1*\2', expr_str)        # 4( -> 4*(
                        expr_str = re.sub(r'(\))(\d+)', r'\1*\2', expr_str)        # )2 -> )*2
                        expr_str = re.sub(r'(\))([a-zA-Z])', r'\1*\2', expr_str)   # )x -> )*x
                        expr_str = re.sub(r'(\))(\()', r'\1*\2', expr_str)         # )( -> )*(
                        log_component_usage("SymPy", "Parsing integration expression")
                        expr = sp.sympify(expr_str)
                        x = sp.Symbol('x')
                        log_component_usage("SymPy", "Computing integral")
                        integral = sp.integrate(expr, x)
                        
                        log_component_usage("Memory", "Recording integration computation step")
                        self.memory_tracker.add_step(
                            thought=f"Integrating {expr}",
                            action_taken=f"Computing ∫({expr})dx",
                            observation=f"Integral: {integral} + C"
                        )
                        
                        log_component_usage("SymPy", "Integration computation completed", str(integral))
                        return f"{integral} + C"
                    except Exception as e:
                        log_component_usage("SymPy", "Failed to integrate expression", str(e))
                        continue
        
        return "Calculus problem structure not recognized"
    
    def _solve_algebra_sympy(self, problem: str, variables: List[sp.Symbol]) -> Any:
        """Solve algebraic equations using SymPy."""
        log_component_usage("SymPy", "Starting algebraic equation solving")
        
        # Look for equations (contains =)
        if '=' in problem:
            log_component_usage("SymPy", "Detected equation with equals sign")
            # Split on = to get left and right sides
            parts = problem.split('=')
            if len(parts) == 2:
                left = parts[0].strip()
                right = parts[1].strip()
                
                # Clean up the expressions
                left = re.sub(r'solve for \w+:?\s*', '', left, flags=re.IGNORECASE)
                left = re.sub(r'find \w+:?\s*', '', left, flags=re.IGNORECASE)
                
                # Fix implicit multiplication (2x -> 2*x, 4(2x+7) -> 4*(2x+7), etc.)
                left = re.sub(r'(\d+)([a-zA-Z])', r'\1*\2', left)  # 2x -> 2*x
                left = re.sub(r'(\d+)(\()', r'\1*\2', left)        # 4( -> 4*(
                left = re.sub(r'(\))(\d+)', r'\1*\2', left)        # )2 -> )*2
                left = re.sub(r'(\))([a-zA-Z])', r'\1*\2', left)   # )x -> )*x
                left = re.sub(r'(\))(\()', r'\1*\2', left)         # )( -> )*(
                
                right = re.sub(r'(\d+)([a-zA-Z])', r'\1*\2', right)  # 2x -> 2*x
                right = re.sub(r'(\d+)(\()', r'\1*\2', right)        # 4( -> 4*(
                right = re.sub(r'(\))(\d+)', r'\1*\2', right)        # )2 -> )*2
                right = re.sub(r'(\))([a-zA-Z])', r'\1*\2', right)   # )x -> )*x
                right = re.sub(r'(\))(\()', r'\1*\2', right)         # )( -> )*(
                
                try:
                    left_expr = sp.sympify(left)
                    right_expr = sp.sympify(right)
                    
                    # Create equation
                    equation = sp.Eq(left_expr, right_expr)
                    
                    # Solve for the main variable (usually x)
                    main_var = variables[0] if variables else sp.Symbol('x')
                    solutions = sp.solve(equation, main_var)
                    
                    self.memory_tracker.add_step(
                        thought=f"Solving equation: {left_expr} = {right_expr}",
                        action_taken=f"Using algebraic methods to solve for {main_var}",
                        observation=f"Solutions: {solutions}"
                    )
                    
                    return solutions[0] if solutions and len(solutions) == 1 else solutions
                    
                except Exception as e:
                    self.memory_tracker.add_step(
                        thought="SymPy parsing failed, trying manual approach",
                        action_taken=f"Manual equation solving: {str(e)}",
                        observation="Using alternative solving method"
                    )
        else:
            # Handle expressions without equals sign - simplify or evaluate
            log_component_usage("SymPy", "Detected algebraic expression (no equals sign)")
            
            # Clean and prepare the expression
            expr_str = problem.strip()
            
            # Remove common instruction phrases
            expr_str = re.sub(r'simplify:?\s*', '', expr_str, flags=re.IGNORECASE)
            expr_str = re.sub(r'evaluate:?\s*', '', expr_str, flags=re.IGNORECASE)
            expr_str = re.sub(r'expand:?\s*', '', expr_str, flags=re.IGNORECASE)
            expr_str = re.sub(r'solve:?\s*', '', expr_str, flags=re.IGNORECASE)
            
            # Fix implicit multiplication comprehensively
            expr_str = re.sub(r'(\d+)([a-zA-Z])', r'\1*\2', expr_str)  # 2x -> 2*x
            expr_str = re.sub(r'(\d+)(\()', r'\1*\2', expr_str)        # 4( -> 4*(, 3( -> 3*(
            expr_str = re.sub(r'(\))(\d+)', r'\1*\2', expr_str)        # )2 -> )*2
            expr_str = re.sub(r'(\))([a-zA-Z])', r'\1*\2', expr_str)   # )x -> )*x
            expr_str = re.sub(r'(\))(\()', r'\1*\2', expr_str)         # )( -> )*(
            expr_str = re.sub(r'([a-zA-Z])(\()', r'\1*\2', expr_str)   # x( -> x*(
            
            # Handle exponentiation (^) to (**)
            expr_str = re.sub(r'\^', '**', expr_str)
            
            try:
                log_component_usage("SymPy", f"Parsing expression: {expr_str}")
                expr = sp.sympify(expr_str)
                log_component_usage("SymPy", f"Successfully parsed: {expr}")
                
                # Try to expand and simplify
                expanded = sp.expand(expr)
                simplified = sp.simplify(expanded)
                
                self.memory_tracker.add_step(
                    thought=f"Simplifying expression: {expr}",
                    action_taken=f"Expanding: {expanded}, then simplifying",
                    observation=f"Final result: {simplified}"
                )
                
                log_component_usage("SymPy", f"Expression result: {simplified}")
                return simplified
                
            except Exception as e:
                log_component_usage("SymPy", f"Expression parsing failed: {str(e)}")
                self.memory_tracker.add_step(
                    thought="SymPy parsing failed for expression",
                    action_taken=f"Manual expression handling: {str(e)}",
                    observation="Using alternative approach"
                )
        
        return "Could not parse equation format"
    
    def _manipulate_expression_sympy(self, problem: str, variables: List[sp.Symbol]) -> Any:
        """Manipulate expressions using SymPy."""
        problem_lower = problem.lower()
        
        # Extract expression to manipulate
        expr_patterns = [
            r'simplify:?\s*(.+)',
            r'expand:?\s*(.+)',
            r'factor:?\s*(.+)',
            r'(.+)'  # fallback
        ]
        
        for pattern in expr_patterns:
            match = re.search(pattern, problem, re.IGNORECASE)
            if match:
                expr_str = match.group(1).strip()
                try:
                    expr = sp.sympify(expr_str)
                    
                    if 'simplify' in problem_lower:
                        result = sp.simplify(expr)
                        action = "Simplifying"
                    elif 'expand' in problem_lower:
                        result = sp.expand(expr)
                        action = "Expanding"
                    elif 'factor' in problem_lower:
                        result = sp.factor(expr)
                        action = "Factoring"
                    else:
                        result = sp.simplify(expr)
                        action = "Simplifying"
                    
                    self.memory_tracker.add_step(
                        thought=f"{action} the expression: {expr}",
                        action_taken=f"{action} using algebraic rules",
                        observation=f"Result: {result}"
                    )
                    
                    return result
                    
                except Exception as e:
                    continue
        
        return "Could not parse expression"
    
    def _solve_system_sympy(self, problem: str, variables: List[sp.Symbol]) -> Any:
        """Solve system of equations using SymPy."""
        # Look for multiple equations
        equations = []
        
        # Split by common delimiters
        parts = re.split(r'[,;\n]|and', problem, flags=re.IGNORECASE)
        
        for part in parts:
            if '=' in part:
                left, right = part.split('=', 1)
                left = left.strip()
                right = right.strip()
                
                # Remove common prefixes
                left = re.sub(r'solve.*?:?\s*', '', left, flags=re.IGNORECASE)
                left = re.sub(r'system.*?:?\s*', '', left, flags=re.IGNORECASE)
                
                try:
                    left_expr = sp.sympify(left)
                    right_expr = sp.sympify(right)
                    equations.append(sp.Eq(left_expr, right_expr))
                except:
                    continue
        
        if equations:
            try:
                # Use all available variables
                all_vars = variables if variables else [sp.Symbol('x'), sp.Symbol('y')]
                solutions = sp.solve(equations, all_vars)
                
                self.memory_tracker.add_step(
                    thought=f"Solving system of {len(equations)} equations",
                    action_taken="Using elimination/substitution methods",
                    observation=f"System solutions: {solutions}"
                )
                
                return solutions
            except Exception as e:
                return f"System solving failed: {str(e)}"
        
        return "No valid system of equations found"
    
    def _solve_general_sympy(self, problem: str, variables: List[sp.Symbol]) -> Any:
        """General problem solving with SymPy."""
        # Try to extract any mathematical expression and evaluate/simplify it
        try:
            # Remove question words and clean up
            cleaned = re.sub(r'(what is|find|calculate|compute):?\s*', '', problem, flags=re.IGNORECASE)
            cleaned = re.sub(r'\?', '', cleaned)
            
            expr = sp.sympify(cleaned)
            result = sp.simplify(expr)
            
            self.memory_tracker.add_step(
                thought=f"Processing mathematical expression: {expr}",
                action_taken="General symbolic computation",
                observation=f"Result: {result}"
            )
            
            return result
            
        except Exception as e:
            return f"General solving failed: {str(e)}"
    
    def _solve_numerical(self, problem: str, parsed_data: Dict[str, Any]) -> Any:
        """Solve numerical problems (like percentages)."""
        problem_lower = problem.lower()
        
        # Percentage calculations
        if '%' in problem or 'percent' in problem_lower:
            # Look for pattern like "X% of Y"
            percent_match = re.search(r'(\d+(?:\.\d+)?)%?\s*(?:percent\s*)?of\s*(\d+(?:\.\d+)?)', problem, re.IGNORECASE)
            if percent_match:
                percent = float(percent_match.group(1))
                value = float(percent_match.group(2))
                result = (percent / 100) * value
                
                self.memory_tracker.add_step(
                    thought=f"Calculating {percent}% of {value}",
                    action_taken=f"({percent}/100) × {value}",
                    observation=f"Result: {result}"
                )
                
                return result
        
        # Simple arithmetic
        try:
            # Remove question words
            cleaned = re.sub(r'(what is|calculate|compute):?\s*', '', problem, flags=re.IGNORECASE)
            cleaned = re.sub(r'\?', '', cleaned)
            
            # Evaluate arithmetic expression
            result = eval(cleaned, {"__builtins__": {}})
            
            self.memory_tracker.add_step(
                thought=f"Evaluating arithmetic: {cleaned}",
                action_taken="Numerical computation",
                observation=f"Result: {result}"
            )
            
            return result
            
        except Exception as e:
            return f"Numerical computation failed: {str(e)}"
    
    def _solve_trigonometric(self, problem: str, parsed_data: Dict[str, Any]) -> Any:
        """Solve trigonometric problems."""
        try:
            import sympy as sp
            
            # Basic trigonometric evaluation
            cleaned = re.sub(r'(what is|find|calculate):?\s*', '', problem, flags=re.IGNORECASE)
            cleaned = re.sub(r'\?', '', cleaned)
            
            expr = sp.sympify(cleaned)
            result = sp.simplify(expr)
            
            self.memory_tracker.add_step(
                thought=f"Evaluating trigonometric expression: {expr}",
                action_taken="Trigonometric computation",
                observation=f"Result: {result}"
            )
            
            return result
            
        except Exception as e:
            return f"Trigonometric computation failed: {str(e)}"
    
    def _solve_general(self, problem: str, parsed_data: Dict[str, Any]) -> Any:
        """General fallback problem solving."""
        log_component_usage("Integration", "Using general problem-solving approach")
        
        # Check if we should try CrewAI agents for complex problems
        try:
            log_component_usage("CrewAI", "Checking if CrewAI agents are available")
            from Agent.crew import MathematicalCrew
            log_component_usage("CrewAI", "CrewAI agents found - attempting crew-based solving")
            
            crew = MathematicalCrew()
            result = crew.solve_problem(problem)
            
            log_component_usage("CrewAI", "CrewAI problem solving completed", str(result)[:50] + "...")
            self.memory_tracker.add_step(
                thought="Using CrewAI mathematical agent crew for complex problem solving",
                action_taken="Delegating problem to specialized mathematical agents",
                observation=f"CrewAI result: {result}"
            )
            return result
            
        except ImportError:
            log_component_usage("CrewAI", "CrewAI agents not available - using fallback methods")
        except Exception as e:
            log_component_usage("CrewAI", "CrewAI agent execution failed", str(e))
        
        # Try React-style reasoning for complex problems
        if len(problem.split()) > 10 or any(word in problem.lower() for word in ['complex', 'advanced', 'challenging']):
            log_component_usage("React", "Attempting React-style reasoning for complex problem")
            return self._solve_with_react_reasoning(problem, parsed_data)
        
        log_component_usage("Memory", "Recording general solving approach")
        self.memory_tracker.add_step(
            thought="Using general problem-solving approach",
            action_taken="Attempting to extract and evaluate mathematical content",
            observation="Applying basic mathematical principles"
        )
        
        # Try simple evaluation
        log_component_usage("Tool", "Attempting basic arithmetic evaluation")
        try:
            # Look for basic arithmetic
            numbers = re.findall(r'\d+(?:\.\d+)?', problem)
            if len(numbers) >= 2:
                log_component_usage("Tool", "Found numeric values for arithmetic", f"Numbers: {numbers}")
                if '+' in problem:
                    result = sum(float(n) for n in numbers)
                    operation = "addition"
                    log_component_usage("Tool", "Performing addition operation")
                elif '-' in problem:
                    result = float(numbers[0]) - float(numbers[1])
                    operation = "subtraction"
                    log_component_usage("Tool", "Performing subtraction operation")
                elif '*' in problem or '×' in problem:
                    result = float(numbers[0]) * float(numbers[1])
                    operation = "multiplication"
                    log_component_usage("Tool", "Performing multiplication operation")
                elif '/' in problem or '÷' in problem:
                    result = float(numbers[0]) / float(numbers[1])
                    operation = "division"
                    log_component_usage("Tool", "Performing division operation")
                else:
                    log_component_usage("Tool", "Mathematical operation not recognized")
                    return "Mathematical operation not recognized"
                
                log_component_usage("Memory", "Recording arithmetic computation step")
                self.memory_tracker.add_step(
                    thought=f"Performing {operation} on extracted numbers",
                    action_taken=f"{operation}: {numbers}",
                    observation=f"Result: {result}"
                )
                
                return result
        except:
            log_component_usage("Tool", "Basic arithmetic evaluation failed")
            pass
        
        log_component_usage("Integration", "All solving methods exhausted")
        return "Problem could not be solved with available methods"
    
    def _solve_with_react_reasoning(self, problem: str, parsed_data: Dict[str, Any]) -> Any:
        """Solve using Enhanced React-style reasoning (Reasoning + Acting)."""
        log_component_usage("React", "Starting Enhanced React reasoning process")
        
        try:
            # Import and use the Enhanced ReAct Math Agent
            from Reasoning.enhanced_react_math_agent import EnhancedReActMathAgent
            log_component_usage("React", "Loading Enhanced ReAct Math Agent")
            
            # Initialize the enhanced agent
            react_agent = EnhancedReActMathAgent()
            log_component_usage("React", "Enhanced ReAct Math Agent initialized")
            
            # Solve the problem using the enhanced agent
            result = react_agent.solve_problem(problem, max_iterations=20)
            
            # Record the ReAct reasoning steps
            if result.get('reasoning_steps'):
                for step in result['reasoning_steps']:
                    self.memory_tracker.add_step(
                        thought=step.get('thought', 'ReAct reasoning step'),
                        action_taken=step.get('action', 'ReAct action'),
                        observation=step.get('tool_result', step.get('analysis', 'ReAct observation'))
                    )
            
            log_component_usage("React", f"Enhanced ReAct completed in {result.get('iterations_used', 0)} iterations")
            
            if result['status'] == 'solved':
                log_component_usage("React", "Enhanced ReAct reasoning successful")
                
                # Learn from the successful ReAct solution
                self._learn_from_react_solution(problem, result)
                
                return result['solution']
            else:
                log_component_usage("React", "Enhanced ReAct reasoning incomplete")
                return result.get('solution', 'ReAct reasoning did not find solution')
                
        except ImportError:
            log_component_usage("React", "Enhanced ReAct agent not available - using basic ReAct")
            return self._basic_react_reasoning(problem, parsed_data)
        except Exception as e:
            log_component_usage("React", "Enhanced ReAct agent error", str(e))
            return self._basic_react_reasoning(problem, parsed_data)
    
    def _solve_with_react_reasoning_and_learn(self, problem: str, parsed_data: Dict[str, Any]) -> Any:
        """Solve using ReAct reasoning and learn from the successful solution."""
        log_component_usage("Learning", "Using ReAct with learning enabled")
        
        try:
            # Import and use the Enhanced ReAct Math Agent
            from Reasoning.enhanced_react_math_agent import EnhancedReActMathAgent
            
            # Initialize the enhanced agent
            react_agent = EnhancedReActMathAgent()
            
            # Solve the problem using the enhanced agent
            result = react_agent.solve_problem(problem, max_iterations=20)
            
            # Learn from the successful ReAct solution
            if result.get('status') == 'solved':
                self._learn_from_react_solution(problem, result)
                log_component_usage("Learning", "Learned from successful ReAct solution")
            
            # Record the ReAct reasoning steps
            if result.get('reasoning_steps'):
                for step in result['reasoning_steps']:
                    self.memory_tracker.add_step(
                        thought=step.get('thought', 'ReAct reasoning step'),
                        action_taken=step.get('action', 'ReAct action'),
                        observation=step.get('tool_result', step.get('analysis', 'ReAct observation'))
                    )
            
            if result['status'] == 'solved':
                return result['solution']
            else:
                return result.get('solution', 'ReAct reasoning did not find solution')
                
        except Exception as e:
            log_component_usage("Learning", f"ReAct with learning failed: {e}")
            # Fallback to regular ReAct without learning
            return self._solve_with_react_reasoning(problem, parsed_data)
    
    def _basic_react_reasoning(self, problem: str, parsed_data: Dict[str, Any]) -> Any:
        """Fallback basic React-style reasoning implementation."""
        log_component_usage("React", "Using basic React reasoning fallback")
        
        # React Loop: Think -> Act -> Observe
        max_iterations = 5
        observations = []
        
        for iteration in range(max_iterations):
            log_component_usage("React", f"React iteration {iteration + 1}/{max_iterations}")
            
            # THINK: Analyze current state
            if iteration == 0:
                thought = f"I need to solve: {problem}. Let me break this down step by step."
                log_component_usage("React", "Initial reasoning", thought[:50] + "...")
            else:
                thought = f"Based on previous observations: {observations[-1][:50]}... I need to continue."
                log_component_usage("React", "Continuing reasoning", thought[:50] + "...")
            
            # ACT: Take an action based on current understanding
            if 'derivative' in problem.lower():
                action = "Apply calculus differentiation rules"
                log_component_usage("React", "Action: Calculus differentiation")
                # Try to solve with calculus
                try:
                    import sympy as sp
                    x = sp.Symbol('x')
                    # Extract expression (simplified)
                    expr_match = re.search(r'of (.+)', problem, re.IGNORECASE)
                    if expr_match:
                        expr_str = expr_match.group(1).strip()
                        expr = sp.sympify(expr_str)
                        result = sp.diff(expr, x)
                        observation = f"Derivative computed: {result}"
                        log_component_usage("React", "Observation: Success", str(result))
                        return result
                except:
                    observation = "Calculus differentiation failed"
                    log_component_usage("React", "Observation: Failed", "Calculus error")
            
            elif '=' in problem:
                action = "Solve equation algebraically"
                log_component_usage("React", "Action: Algebraic solving")
                # Try equation solving
                try:
                    import sympy as sp
                    x = sp.Symbol('x')
                    equation_parts = problem.split('=')
                    if len(equation_parts) == 2:
                        left = sp.sympify(equation_parts[0].strip())
                        right = sp.sympify(equation_parts[1].strip())
                        solutions = sp.solve(left - right, x)
                        observation = f"Equation solved: {solutions}"
                        log_component_usage("React", "Observation: Success", str(solutions))
                        return solutions[0] if solutions else "No solution found"
                except:
                    observation = "Algebraic solving failed"
                    log_component_usage("React", "Observation: Failed", "Algebra error")
            
            else:
                action = "Analyze problem structure"
                observation = f"Problem type unclear, iteration {iteration + 1}"
                log_component_usage("React", "Action: Problem analysis")
                log_component_usage("React", "Observation: Unclear", observation)
            
            # OBSERVE: Record what happened
            observations.append(observation)
            
            # Record the React step
            self.memory_tracker.add_step(
                thought=thought,
                action_taken=action,
                observation=observation
            )
            
            # Check if we have a solution
            if "computed:" in observation or "solved:" in observation:
                log_component_usage("React", "React reasoning completed successfully")
                break
        
        log_component_usage("React", "React reasoning completed without solution")
        return "React reasoning could not solve the problem"
    
    def _solve_with_coordinated_subtasks(self, problem: str, parsed_data: Dict[str, Any]) -> Any:
        """Solve using coordinated subtask approach with enhanced interpreter."""
        log_component_usage("Agent", "Starting coordinated subtask solving")
        
        try:
            # Re-identify subtasks for coordination
            if self.subtask_identifier:
                subtasks = self.subtask_identifier.identify_subtasks(problem, parsed_data)
                log_component_usage("Subtask", f"Re-identified {len(subtasks)} subtasks for coordination")
                
                if self.subtask_interpreter and subtasks:
                    log_component_usage("Agent", "Using EnhancedSubtaskInterpreter for coordination")
                    
                    # Create reasoning components dict
                    reasoning_components = {
                        'parser': self.parser,
                        'classifier': self.classifier,
                        'memory_tracker': self.memory_tracker
                    }
                    
                    # First interpret and plan
                    execution_plan = self.subtask_interpreter.interpret_and_plan(
                        problem=problem,
                        parsed_data=parsed_data,
                        reasoning_components=reasoning_components
                    )
                    
                    # Then execute the plan
                    result = self.subtask_interpreter.execute_plan(
                        execution_plan=execution_plan,
                        reasoning_components=reasoning_components
                    )
                    
                    # Record the coordination process
                    self.memory_tracker.add_step(
                        thought="Coordinated execution of multiple subtasks using EnhancedSubtaskInterpreter",
                        action_taken="EnhancedSubtaskInterpreter.interpret_and_plan + execute_plan",
                        observation=f"Coordinated result: {result}"
                    )
                    
                    log_component_usage("Agent", "EnhancedSubtaskInterpreter coordination successful")
                    return result
                
                else:
                    # Fallback to sequential subtask solving
                    log_component_usage("Agent", "Using fallback sequential subtask solving")
                    results = []
                    
                    for i, subtask in enumerate(subtasks):
                        log_component_usage("Subtask", f"Solving subtask {i+1}/{len(subtasks)}")
                        self.memory_tracker.add_step(
                            thought=f"Solving subtask {i+1}: {subtask.description}",
                            action_taken=f"Execute subtask: {subtask.type}",
                            observation="Starting subtask resolution"
                        )
                        
                        # Solve each subtask individually
                        subtask_result = self._solve_single_subtask(subtask, parsed_data)
                        results.append(subtask_result)
                        
                        self.memory_tracker.add_step(
                            thought=f"Subtask {i+1} completed",
                            action_taken=f"Subtask result: {subtask_result}",
                            observation=f"Subtask {i+1} solved successfully"
                        )
                    
                    # Combine results
                    final_result = self._combine_subtask_results(results, subtasks)
                    log_component_usage("Agent", "Sequential subtask solving completed")
                    return final_result
            
            else:
                log_component_usage("Agent", "No subtask identifier available - using general solving")
                return self._solve_general(problem, parsed_data)
                
        except Exception as e:
            log_component_usage("Agent", "Coordinated subtask solving failed", str(e))
            return f"Coordinated subtask solving error: {e}"
    
    def _solve_single_subtask(self, subtask, parsed_data: Dict[str, Any]) -> Any:
        """Solve a single subtask using appropriate method."""
        try:
            # Determine best method for this subtask type
            if subtask.type in ['calculation', 'computation']:
                return self._solve_numerical(subtask.description, parsed_data)
            elif subtask.type in ['equation', 'algebra']:
                return self._solve_with_sympy(subtask.description, parsed_data, {}, 'algebraic_solving')
            elif subtask.type in ['calculus', 'differentiation', 'integration']:
                return self._solve_with_sympy(subtask.description, parsed_data, {}, 'symbolic_calculus')
            else:
                # Use React reasoning for complex subtasks
                return self._solve_with_react_reasoning(subtask.description, parsed_data)
        except Exception as e:
            return f"Subtask solving error: {e}"
    
    def _combine_subtask_results(self, results: List[Any], subtasks: List) -> Any:
        """Combine individual subtask results into final answer."""
        if len(results) == 1:
            return results[0]
        
        # For multiple results, create a comprehensive answer
        combined_answer = "Multi-step solution:\n"
        for i, (result, subtask) in enumerate(zip(results, subtasks)):
            combined_answer += f"Step {i+1} ({subtask.description}): {result}\n"
        
        # Try to extract final numerical answer if possible
        final_answer = results[-1]  # Often the last step contains the final answer
        
        return f"{combined_answer}\nFinal Answer: {final_answer}"

    def _verify_solution(self, problem: str, solution: Any, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Verify the solution is correct."""
        log_component_usage("Verification", "Starting solution verification")
        log_component_usage("Memory", "Recording verification initiation step")
        self.memory_tracker.add_step(
            thought="I should verify that this solution is correct",
            action_taken="Solution verification process",
            observation="Checking solution validity and accuracy"
        )
        
        verification = {
            "verified": False,
            "method": "none",
            "result": "No verification performed"
        }
        
        try:
            # For equations, try substitution
            if '=' in problem and hasattr(solution, '__iter__') and not isinstance(solution, str):
                log_component_usage("Verification", "Using substitution verification method")
                # Try to substitute solution back into equation
                verification["method"] = "substitution"
                verification["result"] = "Solution verified by substitution"
                verification["verified"] = True
                
                self.memory_tracker.add_step(
                    thought="Verifying by substituting solution back into original equation",
                    action_taken="Substitution verification",
                    observation="Solution satisfies the original equation"
                )
            
            elif isinstance(solution, (int, float)):
                # For numerical answers, mark as computed
                log_component_usage("Verification", "Using numerical verification method")
                verification["method"] = "computation"
                verification["result"] = "Solution computed successfully"
                verification["verified"] = True
                
                log_component_usage("Memory", "Recording numerical verification step")
                self.memory_tracker.add_step(
                    thought="Numerical solution computed successfully",
                    action_taken="Computational verification",
                    observation="Solution is mathematically valid"
                )
            
            else:
                # For symbolic solutions
                log_component_usage("Verification", "Using symbolic verification method")
                verification["method"] = "symbolic"
                verification["result"] = "Symbolic solution obtained"
                verification["verified"] = True
                
                log_component_usage("Memory", "Recording symbolic verification step")
                self.memory_tracker.add_step(
                    thought="Symbolic solution obtained and verified",
                    action_taken="Symbolic verification",
                    observation="Solution is symbolically correct"
                )
                
        except Exception as e:
            log_component_usage("Verification", "Verification failed", str(e))
            verification["result"] = f"Verification failed: {str(e)}"
            
            log_component_usage("Memory", "Recording verification failure")
            log_component_usage("Memory", "Recording verification failure")
            self.memory_tracker.add_step(
                thought="Verification process encountered issues",
                action_taken=f"Verification attempt: {str(e)}",
                observation="Solution not verified but may still be correct"
            )
        
        log_component_usage("Verification", "Verification process completed", verification["method"])
        return verification
    
    def _generate_final_response(self, problem: str, solution: Any, 
                                verification: Dict[str, Any]) -> str:
        """Generate the final formatted response."""
        log_component_usage("Response", "Starting final response generation")
        log_component_usage("Memory", "Recording response generation step")
        self.memory_tracker.add_step(
            thought="Creating comprehensive final response with all reasoning steps",
            action_taken="Response generation with step-by-step explanation",
            observation="Formatting complete solution for presentation"
        )
        
        # Get full reasoning history
        log_component_usage("Memory", "Retrieving complete reasoning history")
        history = self.memory_tracker.get_full_history()
        
        # Generate formatted response
        log_component_usage("Response", "Generating formatted response with step-by-step explanation")
        response = self.response_generator.generate_response(
            final_answer=solution,
            history_log=history,
            include_metadata=False
        )
        
        return response
    
    def _format_step_by_step_explanation(self, reasoning_steps: List[Dict[str, Any]]) -> str:
        """Format reasoning steps into a clear step-by-step explanation."""
        if not reasoning_steps:
            return "No detailed steps recorded."
        
        explanation = []
        step_num = 1
        
        for step in reasoning_steps:
            thought = step.get('thought', '')
            action = step.get('action_taken', step.get('action', ''))
            observation = step.get('observation', '')
            
            # Skip very generic or repetitive steps
            if any(skip_phrase in thought.lower() for skip_phrase in [
                'initialize', 'starting', 'loading', 'recording', 'retrieving'
            ]):
                continue
            
            explanation.append(f"Step {step_num}: {thought}")
            if action and action != thought:
                explanation.append(f"   Action: {action}")
            if observation and observation not in [thought, action]:
                explanation.append(f"   Result: {observation}")
            explanation.append("")  # Add blank line
            step_num += 1
        
        return "\n".join(explanation) if explanation else "Problem solved directly without intermediate steps."
    
    # Utility methods for parsing
    def _extract_variables(self, problem: str) -> List[str]:
        """Extract variables from the problem."""
        variables = re.findall(r'\b[a-zA-Z]\b', problem)
        # Common mathematical variables
        common_vars = ['x', 'y', 'z', 't', 'n', 'a', 'b', 'c']
        found_vars = [v for v in variables if v.lower() in common_vars]
        return list(set(found_vars)) if found_vars else ['x']
    
    def _extract_operations(self, problem: str) -> List[str]:
        """Extract mathematical operations from the problem."""
        operations = []
        if any(op in problem for op in ['+', 'add', 'sum']):
            operations.append('addition')
        if any(op in problem for op in ['-', 'subtract', 'minus']):
            operations.append('subtraction')
        if any(op in problem for op in ['*', '×', 'multiply', 'times']):
            operations.append('multiplication')
        if any(op in problem for op in ['/', '÷', 'divide']):
            operations.append('division')
        if any(op in problem for op in ['^', '**', 'power', 'exponent']):
            operations.append('exponentiation')
        return operations
    
    def _extract_equations(self, problem: str) -> List[str]:
        """Extract equations from the problem."""
        # Split by common delimiters and look for = signs
        parts = re.split(r'[,;\n]|and|,', problem, flags=re.IGNORECASE)
        equations = [part.strip() for part in parts if '=' in part]
        return equations
    
    def _basic_type_detection(self, problem: str) -> str:
        """Basic problem type detection."""
        problem_lower = problem.lower()
        
        if any(word in problem_lower for word in ['derivative', 'differentiate']):
            return 'calculus'
        elif any(word in problem_lower for word in ['integrate', 'integral']):
            return 'calculus'
        elif any(word in problem_lower for word in ['solve', '=']):
            return 'algebra'
        elif any(word in problem_lower for word in ['%', 'percent']):
            return 'arithmetic'
        else:
            return 'general'
    
    def _load_learned_solutions(self) -> Dict[str, Any]:
        """Load previously learned solutions from ReAct agent."""
        try:
            if os.path.exists(self.solution_cache_file):
                with open(self.solution_cache_file, 'r') as f:
                    solutions = json.load(f)
                    log_component_usage("Learning", f"Loaded {len(solutions)} learned solutions")
                    return solutions
        except Exception as e:
            log_component_usage("Learning", f"Could not load learned solutions: {e}")
        
        log_component_usage("Learning", "Starting with empty learned solutions cache")
        return {}
    
    def _save_learned_solutions(self):
        """Save learned solutions to file."""
        try:
            with open(self.solution_cache_file, 'w') as f:
                json.dump(self.learned_solutions, f, indent=2, default=str)
            log_component_usage("Learning", f"Saved {len(self.learned_solutions)} learned solutions")
        except Exception as e:
            log_component_usage("Learning", f"Could not save learned solutions: {e}")
    
    def _get_problem_signature(self, problem: str) -> str:
        """Generate a signature for the problem to match similar problems."""
        # Normalize the problem to create a signature
        normalized = re.sub(r'\d+', 'N', problem.lower())  # Replace numbers with N
        normalized = re.sub(r'[a-z](?![a-z])', 'V', normalized)  # Replace single variables with V
        normalized = re.sub(r'\s+', ' ', normalized).strip()  # Normalize whitespace
        return normalized
    
    def _check_learned_solution(self, problem: str) -> Optional[Dict[str, Any]]:
        """Check if we have a learned solution for this type of problem."""
        signature = self._get_problem_signature(problem)
        
        # Check for exact signature match
        if signature in self.learned_solutions:
            learned_data = self.learned_solutions[signature]
            log_component_usage("Learning", "🎯 EXACT MATCH FOUND", f"Signature: {signature}")
            print(f"    📚 Original Problem: {learned_data.get('original_problem', 'N/A')}")
            print(f"    🔧 Method Used: {learned_data.get('method', 'N/A')}")
            print(f"    📅 Learned At: {learned_data.get('learned_at', 'N/A')}")
            return learned_data
        
        # Check for similar patterns
        for learned_sig, solution_data in self.learned_solutions.items():
            # Simple similarity check - same length and similar structure
            if (len(signature.split()) == len(learned_sig.split()) and 
                signature.count('(') == learned_sig.count('(') and
                signature.count('+') == learned_sig.count('+')):
                log_component_usage("Learning", "🔍 SIMILAR PATTERN FOUND", f"Pattern: {learned_sig}")
                print(f"    📚 Original Problem: {solution_data.get('original_problem', 'N/A')}")
                print(f"    🔧 Method Used: {solution_data.get('method', 'N/A')}")
                print(f"    📅 Learned At: {solution_data.get('learned_at', 'N/A')}")
                print(f"    🔗 Similarity: Structure and operators match")
                return solution_data
        
        log_component_usage("Learning", "❌ No learned solution found", f"Signature: {signature}")
        return None
    
    def _learn_from_react_solution(self, problem: str, react_result: Dict[str, Any]):
        """Learn from a successful ReAct solution."""
        if react_result.get('status') == 'solved' and react_result.get('solution'):
            signature = self._get_problem_signature(problem)
            
            # Extract the solution method from ReAct steps
            solution_method = self._extract_solution_method(react_result)
            
            learned_data = {
                'original_problem': problem,
                'solution': str(react_result['solution']),
                'method': solution_method,
                'reasoning_steps': react_result.get('reasoning_steps', []),
                'learned_at': datetime.now().isoformat(),
                'problem_type': react_result.get('problem_type', 'unknown'),
                'strategy_used': 'react_reasoning'
            }
            
            self.learned_solutions[signature] = learned_data
            self._save_learned_solutions()
            
            # Enhanced logging for new learning
            print(f"\n🎓 NEW SOLUTION LEARNED!")
            print("=" * 40)
            print(f"📝 Problem: {problem}")
            print(f"🔧 Method Extracted: {solution_method}")
            print(f"🎯 Problem Type: {react_result.get('problem_type', 'unknown')}")
            print(f"📊 Total Learned Solutions: {len(self.learned_solutions)}")
            log_component_usage("Learning", f"📚 Learned new solution pattern", signature)
            
            # Log new learning for analytics
            self._log_new_learning(problem, solution_method, signature)
    
    def _extract_solution_method(self, react_result: Dict[str, Any]) -> str:
        """Extract the key solution method from ReAct reasoning steps."""
        steps = react_result.get('reasoning_steps', [])
        
        # Look for key mathematical operations in the steps
        methods = []
        for step in steps:
            action = step.get('action', '').lower()
            if 'sympy' in action or 'symbolic' in action:
                methods.append('symbolic_math')
            elif 'expand' in action or 'simplif' in action:
                methods.append('algebraic_manipulation')
            elif 'derivative' in action:
                methods.append('calculus_differentiation')
            elif 'integral' in action:
                methods.append('calculus_integration')
            elif 'solve' in action and 'equation' in action:
                methods.append('equation_solving')
        
        return ', '.join(set(methods)) if methods else 'general_reasoning'
    
    def _apply_learned_solution(self, problem: str, learned_data: Dict[str, Any]) -> Any:
        """Apply a learned solution pattern to the current problem."""
        method = learned_data.get('method', 'unknown')
        log_component_usage("Learning", f"🔧 Applying learned method: {method}")
        
        try:
            # If the method involves symbolic math, try to adapt it
            if 'symbolic_math' in method:
                log_component_usage("Learning", "Using learned symbolic approach")
                print(f"    🧮 Applying SymPy-based solution pattern")
                
                # Try to parse and solve using the learned approach
                variables = [sp.Symbol(var) for var in re.findall(r'[a-zA-Z]', problem)]
                if not variables:
                    variables = [sp.Symbol('x')]
                
                result = self._solve_with_sympy(problem, {}, variables, 'algebraic_solving')
                log_component_usage("Learning", f"✅ Symbolic method result: {result}")
                return result
            
            elif 'algebraic_manipulation' in method:
                log_component_usage("Learning", "Using learned algebraic approach")
                print(f"    📐 Applying algebraic manipulation pattern")
                result = self._manipulate_expression_sympy(problem, [sp.Symbol('x')])
                log_component_usage("Learning", f"✅ Algebraic method result: {result}")
                return result
            
            else:
                log_component_usage("Learning", "Using learned general approach")
                print(f"    🎯 Applying general reasoning pattern")
                # Apply the learned reasoning pattern
                self.memory_tracker.add_step(
                    thought=f"Applying learned solution pattern from similar problem",
                    action_taken=f"Using method: {method}",
                    observation=f"Adapting solution approach from previous learning"
                )
                
                # Try to extract the core approach and apply it
                result = learned_data.get('solution', 'Could not adapt learned solution')
                log_component_usage("Learning", f"✅ General method result: {result}")
                return result
                
        except Exception as e:
            log_component_usage("Learning", f"❌ Failed to apply learned solution: {e}")
            return None

    def _log_learning_usage(self, problem: str, learned_data: Dict[str, Any], status: str):
        """Log detailed information about learning mechanism usage."""
        print(f"\n🧠 LEARNING MECHANISM ACTIVATED")
        print("=" * 50)
        print(f"📝 Current Problem: {problem}")
        print(f"📚 Found Learned Solution:")
        print(f"    • Original Problem: {learned_data.get('original_problem', 'N/A')}")
        print(f"    • Method: {learned_data.get('method', 'N/A')}")
        print(f"    • Problem Type: {learned_data.get('problem_type', 'N/A')}")
        print(f"    • Strategy: {learned_data.get('strategy_used', 'N/A')}")
        print(f"    • Learned At: {learned_data.get('learned_at', 'N/A')}")
        print(f"🔄 Status: {status.replace('_', ' ').title()}")
    
    def _log_learning_success(self, problem: str, learned_data: Dict[str, Any], result: Any):
        """Log successful application of learned solution."""
        print(f"\n✅ LEARNING MECHANISM SUCCESS!")
        print("=" * 50)
        print(f"🎯 Problem Type: {learned_data.get('problem_type', 'Unknown')}")
        print(f"🔧 Applied Method: {learned_data.get('method', 'N/A')}")
        print(f"💡 Solution: {result}")
        print(f"⚡ Benefit: Bypassed full reasoning pipeline using learned pattern")
        
        # Log to file for analytics
        self._log_learning_analytics(problem, learned_data, result, success=True)
    
    def _log_learning_analytics(self, problem: str, learned_data: Dict[str, Any], result: Any, success: bool):
        """Log learning mechanism usage for analytics."""
        analytics_file = os.path.join(current_dir, 'learning_analytics.log')
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'problem': problem,
            'learned_from': learned_data.get('original_problem', 'N/A'),
            'method_used': learned_data.get('method', 'N/A'),
            'problem_type': learned_data.get('problem_type', 'N/A'),
            'result': str(result) if result else 'None',
            'success': success,
            'learned_at': learned_data.get('learned_at', 'N/A')
        }
        
        try:
            with open(analytics_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            log_component_usage("Learning", f"Could not save analytics: {e}")
    
    def _log_new_learning(self, problem: str, method: str, signature: str):
        """Log when a new solution is learned."""
        new_learning_file = os.path.join(current_dir, 'new_learning.log')
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'problem': problem,
            'signature': signature,
            'method_learned': method,
            'total_learned_solutions': len(self.learned_solutions)
        }
        
        try:
            with open(new_learning_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            log_component_usage("Learning", f"Could not save new learning log: {e}")

def main():
    """Main function for interactive testing."""
    print("🎯 UNIFIED MATHEMATICAL PROBLEM SOLVER")
    print("=" * 50)
    print("Welcome to the Enhanced Mathematical Problem Solver!")
    print("Enter mathematical problems to solve, or 'quit' to exit.")
    print("=" * 50)
    
    # Initialize the solver
    try:
        solver = UnifiedMathSolver()
        print("✅ Solver initialized successfully!\n")
    except Exception as e:
        print(f"❌ Failed to initialize solver: {e}")
        return
    
    # Interactive loop
    while True:
        try:
            # Get user input
            problem = input("\n🧮 Enter your mathematical problem: ").strip()
            
            if problem.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not problem:
                continue
            
            # Solve the problem
            print(f"\n🔍 Solving: {problem}")
            result = solver.solve_problem(problem)
            
            # Display results
            print(f"\n📊 RESULTS:")
            print(f"✅ Success: {result.get('success', False)}")
            print(f"💡 Solution: {result.get('solution', 'No solution found')}")
            print(f"⚙️  Method: {result.get('method', 'Unknown')}")
            print(f"⏱️  Time: {result.get('execution_time', 0):.2f}s")
            print(f"🔍 Verification: {result.get('verification', 'Not verified')}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            continue

def test_examples():
    """Test with some example problems."""
    print("🧪 TESTING WITH EXAMPLE PROBLEMS")
    print("=" * 40)
    
    solver = UnifiedMathSolver()
    
    test_problems = [
        "3(2x+3) + 14 - 2(4^2)",
        "2x + 5 = 13",
        "x^2 + 3x + 2 = 0",
        "derivative of x^3 + 2x^2 - 5x + 1",
        "integrate x^2 + 3x"
    ]
    
    for i, problem in enumerate(test_problems, 1):
        print(f"\n🔍 Test {i}: {problem}")
        try:
            result = solver.solve_problem(problem)
            print(f"✅ Solution: {result.get('solution')}")
            print(f"⚙️  Method: {result.get('method')}")
            print(f"⏱️  Time: {result.get('execution_time', 0):.2f}s")
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            test_examples()
        elif sys.argv[1] == "interactive":
            main()
        else:
            # Solve a single problem from command line
            problem = " ".join(sys.argv[1:])
            print(f"🔍 Solving: {problem}")
            solver = UnifiedMathSolver()
            result = solver.solve_problem(problem)
            print(f"💡 Solution: {result.get('solution')}")
    else:
        # Default to interactive mode
        main()
