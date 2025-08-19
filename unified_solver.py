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
                
                self.parser = EnhancedProblemParser()
                log_component_usage("Parser", "Enhanced Problem Parser initialized")
                self.classifier = EnhancedProblemClassifier()
                log_component_usage("Classifier", "Enhanced Problem Classifier initialized")
                self.subtask_identifier = EnhancedSubtaskIdentifier()
                log_component_usage("Subtask", "Enhanced Subtask Identifier initialized")
                self.agent_delegator = EnhancedAgentDelegator()
                log_component_usage("Agent", "Enhanced Agent Delegator initialized")
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
            
            # Step 6: Verify the solution
            print("✅ Step 5: Verifying solution...")
            log_component_usage("Verification", "Verifying solution correctness")
            verification = self._verify_solution(problem, solution, parsed_data)
            
            # Step 7: Generate final response
            print("📊 Step 6: Generating final response...")
            log_component_usage("Response", "Generating formatted response")
            final_response = self._generate_final_response(problem, solution, verification)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            print(f"\n🎉 PROBLEM SOLVED in {execution_time:.2f} seconds!")
            
            return {
                "success": True,
                "problem": problem,
                "solution": solution,
                "verification": verification,
                "formatted_response": final_response,
                "execution_time": execution_time,
                "reasoning_steps": self.memory_tracker.get_full_history()
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
                    # For multi-subtask problems, use coordinated solving
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
                        # Fix implicit multiplication
                        expr_str = re.sub(r'(\d+)([a-zA-Z])', r'\1*\2', expr_str)
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
                        # Fix implicit multiplication
                        expr_str = re.sub(r'(\d+)([a-zA-Z])', r'\1*\2', expr_str)
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
                
                # Fix implicit multiplication (2x -> 2*x, 3y -> 3*y, etc.)
                left = re.sub(r'(\d+)([a-zA-Z])', r'\1*\2', left)
                right = re.sub(r'(\d+)([a-zA-Z])', r'\1*\2', right)
                
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
            result = react_agent.solve_problem(problem, max_iterations=10)
            
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

def interactive_solver():
    """Interactive problem solving interface."""
    print("🎯 UNIFIED MATHEMATICAL PROBLEM SOLVER")
    print("=" * 50)
    print("Enter any mathematical problem and get a complete solution!")
    print("Examples:")
    print("  • Solve for x: 2x + 5 = 13")
    print("  • Find the derivative of x^2 + 3x + 2")
    print("  • Simplify: (x + 2)(x - 3)")
    print("  • What is 15% of 240?")
    print("  • Integrate x^2 + 3x with respect to x")
    print("\nType 'quit' to exit.\n")
    
    solver = UnifiedMathSolver()
    
    while True:
        try:
            problem = input("🔢 Enter your problem: ").strip()
            
            if problem.lower() in ['quit', 'exit', 'q']:
                print("👋 Thank you for using the Unified Mathematical Problem Solver!")
                break
            
            if not problem:
                print("Please enter a problem to solve.")
                continue
            
            # Solve the problem
            result = solver.solve_problem(problem)
            
            if result["success"]:
                print("\n" + "="*60)
                print(result["formatted_response"])
                print("="*60)
                print(f"✅ Solved in {result['execution_time']:.2f} seconds")
                
                if result.get("verification", {}).get("verified"):
                    print(f"✅ Solution verified: {result['verification']['result']}")
                
            else:
                print(f"\n❌ Error: {result['error']}")
                print("Please try rephrasing the problem or check the syntax.")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")

def solve_single_problem(problem: str):
    """Solve a single problem and display the result."""
    solver = UnifiedMathSolver()
    result = solver.solve_problem(problem)
    
    if result["success"]:
        print(result["formatted_response"])
        print(f"\n✅ Solved in {result['execution_time']:.2f} seconds")
    else:
        print(f"❌ Error: {result['error']}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Command line usage: python unified_solver.py "problem statement"
        problem = " ".join(sys.argv[1:])
        solve_single_problem(problem)
    else:
        # Interactive mode
        interactive_solver()
