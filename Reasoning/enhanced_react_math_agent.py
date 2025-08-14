#!/usr/bin/env python3
"""
🧠 ENHANCED REACT-BASED MATHEMATICAL REASONING ENGINE
===================================================

Advanced ReAct (Reasoning + Acting) agent specifically designed for 
mathematical problem solving with step-by-step reasoning.
"""

import openai
import os
import re
import sympy
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dotenv import load_dotenv
import json

# Load .env from the parent directory
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

class MathematicalTools:
    """
    Comprehensive mathematical tools for solving various problems.
    """
    
    @staticmethod
    def basic_calculator(expression: str) -> str:
        """Evaluate basic mathematical expressions safely."""
        try:
            # Replace common mathematical notation
            expression = expression.replace('^', '**')
            expression = expression.replace('π', str(np.pi))
            expression = expression.replace('e', str(np.e))
            
            # Use eval with restricted globals for safety
            allowed_names = {
                k: v for k, v in vars(np).items() 
                if not k.startswith('_')
            }
            allowed_names.update({
                'abs': abs, 'round': round, 'max': max, 'min': min,
                'sum': sum, 'pow': pow
            })
            
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            return str(result)
        except Exception as e:
            return f"Calculator Error: {str(e)}"
    
    @staticmethod
    def symbolic_solver(equation: str, variable: str = 'x') -> str:
        """Solve symbolic equations using SymPy."""
        try:
            # Parse the equation
            if '=' in equation:
                left, right = equation.split('=', 1)
                eq = sympy.Eq(sympy.sympify(left.strip()), sympy.sympify(right.strip()))
            else:
                eq = sympy.sympify(equation)
            
            # Define the variable
            var = sympy.Symbol(variable)
            
            # Solve the equation
            solutions = sympy.solve(eq, var)
            
            if solutions:
                return f"Solutions for {variable}: {solutions}"
            else:
                return f"No solutions found for {variable}"
                
        except Exception as e:
            return f"Symbolic Solver Error: {str(e)}"
    
    @staticmethod
    def derivative_calculator(function: str, variable: str = 'x') -> str:
        """Calculate derivatives using SymPy."""
        try:
            var = sympy.Symbol(variable)
            func = sympy.sympify(function)
            derivative = sympy.diff(func, var)
            return f"d/d{variable}({function}) = {derivative}"
        except Exception as e:
            return f"Derivative Error: {str(e)}"
    
    @staticmethod
    def integral_calculator(function: str, variable: str = 'x', limits: Optional[Tuple] = None) -> str:
        """Calculate integrals using SymPy."""
        try:
            var = sympy.Symbol(variable)
            func = sympy.sympify(function)
            
            if limits:
                a, b = limits
                integral = sympy.integrate(func, (var, a, b))
                return f"∫[{a} to {b}] {function} d{variable} = {integral}"
            else:
                integral = sympy.integrate(func, var)
                return f"∫ {function} d{variable} = {integral} + C"
        except Exception as e:
            return f"Integral Error: {str(e)}"
    
    @staticmethod
    def geometry_calculator(shape: str, **parameters) -> str:
        """Calculate geometric properties."""
        try:
            shape = shape.lower()
            
            if shape == 'circle':
                if 'radius' in parameters:
                    r = parameters['radius']
                    area = np.pi * r**2
                    circumference = 2 * np.pi * r
                    return f"Circle (r={r}): Area = {area:.4f}, Circumference = {circumference:.4f}"
                    
            elif shape == 'rectangle':
                if 'length' in parameters and 'width' in parameters:
                    l, w = parameters['length'], parameters['width']
                    area = l * w
                    perimeter = 2 * (l + w)
                    return f"Rectangle ({l}×{w}): Area = {area}, Perimeter = {perimeter}"
                    
            elif shape == 'triangle':
                if 'base' in parameters and 'height' in parameters:
                    b, h = parameters['base'], parameters['height']
                    area = 0.5 * b * h
                    return f"Triangle (base={b}, height={h}): Area = {area}"
                    
            return f"Unsupported shape or missing parameters for {shape}"
            
        except Exception as e:
            return f"Geometry Error: {str(e)}"
    
    @staticmethod
    def matrix_calculator(operation: str, matrix1: List[List], matrix2: Optional[List[List]] = None) -> str:
        """Perform matrix operations."""
        try:
            A = np.array(matrix1)
            
            if operation == 'determinant':
                det = np.linalg.det(A)
                return f"Determinant: {det}"
                
            elif operation == 'inverse':
                inv = np.linalg.inv(A)
                return f"Inverse:\n{inv}"
                
            elif operation == 'multiply' and matrix2:
                B = np.array(matrix2)
                result = np.dot(A, B)
                return f"Matrix multiplication result:\n{result}"
                
            return f"Unsupported matrix operation: {operation}"
            
        except Exception as e:
            return f"Matrix Error: {str(e)}"
    
    @staticmethod
    def statistics_calculator(data: List[float], operation: str) -> str:
        """Calculate statistical measures."""
        try:
            data = np.array(data)
            
            if operation == 'mean':
                return f"Mean: {np.mean(data)}"
            elif operation == 'median':
                return f"Median: {np.median(data)}"
            elif operation == 'std':
                return f"Standard Deviation: {np.std(data)}"
            elif operation == 'variance':
                return f"Variance: {np.var(data)}"
            elif operation == 'all':
                return f"Mean: {np.mean(data)}, Median: {np.median(data)}, Std: {np.std(data)}"
                
        except Exception as e:
            return f"Statistics Error: {str(e)}"

class EnhancedReActMathAgent:
    """
    Enhanced ReAct agent for mathematical reasoning with step-by-step problem solving.
    """
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.api_base = "https://openrouter.ai/api/v1"
        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.api_base)
        self.tools = MathematicalTools()
        
        # Available tools registry
        self.available_tools = {
            "calculator": self.tools.basic_calculator,
            "symbolic_solver": self.tools.symbolic_solver,
            "derivative": self.tools.derivative_calculator,
            "integral": self.tools.integral_calculator,
            "geometry": self.tools.geometry_calculator,
            "matrix": self.tools.matrix_calculator,
            "statistics": self.tools.statistics_calculator
        }
        
        print("🧠 Enhanced ReAct Math Agent initialized with comprehensive tools.")
    
    def solve_problem(self, problem: str, max_iterations: int = 20) -> Dict[str, Any]:
        """
        Solve a mathematical problem using ReAct reasoning.
        
        Args:
            problem: The mathematical problem to solve
            max_iterations: Maximum number of reasoning iterations
            
        Returns:
            Dictionary containing the solution and reasoning steps
        """
        print(f"🎯 Solving problem: {problem}")
        print("=" * 60)
        
        reasoning_history = []
        current_context = problem
        
        for iteration in range(max_iterations):
            print(f"\n🔄 ITERATION {iteration + 1}")
            print("-" * 30)
            
            # Think: Analyze current situation and decide next action
            thought = self._think(current_context, reasoning_history)
            print(f"💭 THOUGHT: {thought['reasoning']}")
            
            # Act: Execute the decided action
            if thought['action'] == 'SOLVE':
                # Final solution step
                solution = self._generate_final_solution(current_context, reasoning_history)
                print(f"✅ FINAL SOLUTION: {solution}")
                
                return {
                    "problem": problem,
                    "solution": solution,
                    "reasoning_steps": reasoning_history,
                    "iterations_used": iteration + 1,
                    "status": "solved"
                }
                
            elif thought['action'] == 'USE_TOOL':
                # Use a mathematical tool
                tool_result = self._use_tool(thought['tool'], thought['parameters'])
                print(f"🔧 TOOL RESULT: {tool_result}")
                
                reasoning_history.append({
                    "iteration": iteration + 1,
                    "thought": thought['reasoning'],
                    "action": thought['action'],
                    "tool_used": thought['tool'],
                    "tool_result": tool_result
                })
                
                current_context += f"\nTool result: {tool_result}"
                
            elif thought['action'] == 'ANALYZE':
                # Analyze the problem further
                analysis = self._analyze_problem(current_context)
                print(f"🔍 ANALYSIS: {analysis}")
                
                reasoning_history.append({
                    "iteration": iteration + 1,
                    "thought": thought['reasoning'],
                    "action": thought['action'],
                    "analysis": analysis
                })
                
                current_context += f"\nAnalysis: {analysis}"
                
            # Observe: Update context based on new information
            current_context = self._observe(current_context, reasoning_history)
        
        # If max iterations reached without solution
        return {
            "problem": problem,
            "solution": "Unable to solve within iteration limit",
            "reasoning_steps": reasoning_history,
            "iterations_used": max_iterations,
            "status": "incomplete"
        }
    
    def _think(self, context: str, history: List[Dict]) -> Dict[str, Any]:
        """ReAct thinking step - analyze and decide next action."""
        
        history_text = "\n".join([
            f"Step {h['iteration']}: {h['thought']}" for h in history[-3:]  # Last 3 steps
        ])
        
        prompt = f"""
        You are solving a mathematical problem step by step using ReAct reasoning.
        
        CURRENT CONTEXT:
        {context}
        
        PREVIOUS REASONING STEPS:
        {history_text}
        
        AVAILABLE ACTIONS:
        1. ANALYZE - Further analyze the problem to understand it better
        2. USE_TOOL - Use a mathematical tool to compute something
        3. SOLVE - Provide the final solution (only when ready)
        
        AVAILABLE TOOLS:
        - calculator: For basic arithmetic (e.g., "3 * 4 + 2")
        - symbolic_solver: For equations (e.g., "2*x + 3 = 7", variable="x")
        - derivative: For derivatives (e.g., "x**2 + 3*x", variable="x")
        - integral: For integrals (e.g., "2*x", variable="x")
        - geometry: For geometric calculations
        - matrix: For matrix operations
        - statistics: For statistical calculations
        
        Think step by step and decide your next action. Return ONLY a JSON object:
        {{
            "reasoning": "your reasoning about what to do next",
            "action": "ANALYZE/USE_TOOL/SOLVE",
            "tool": "tool_name (if action is USE_TOOL)",
            "parameters": "parameters for the tool (if action is USE_TOOL)"
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model="google/gemini-2.0-flash-001",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=300
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            else:
                return {
                    "reasoning": "Unable to parse response, analyzing problem",
                    "action": "ANALYZE"
                }
                
        except Exception as e:
            print(f"❌ Thinking error: {e}")
            return {
                "reasoning": f"Error in thinking process: {e}",
                "action": "ANALYZE"
            }
    
    def _use_tool(self, tool_name: str, parameters: str) -> str:
        """Execute a mathematical tool with given parameters."""
        print(f"🔧 Using tool '{tool_name}' with parameters: '{parameters}'")  # Debug line
        
        try:
            if tool_name not in self.available_tools:
                return f"Tool '{tool_name}' not available"
            
            tool_func = self.available_tools[tool_name]
            
            # Parse parameters based on tool type
            if tool_name == "calculator":
                return tool_func(parameters)
                
            elif tool_name == "symbolic_solver":
                # Parse equation and variable, clean up parameters
                clean_params = parameters.replace("equation=", "").replace("variable=", "")
                parts = clean_params.split(',')
                equation = parts[0].strip()
                variable = parts[1].strip() if len(parts) > 1 else 'x'
                return tool_func(equation, variable)
                
            elif tool_name == "derivative":
                # Clean up parameters that might have "expression=" prefix
                clean_params = parameters.replace("expression=", "").replace("function=", "").replace("variable=", "")
                parts = clean_params.split(',')
                function = parts[0].strip()
                variable = parts[1].strip() if len(parts) > 1 else 'x'
                print(f"🔧 Calling derivative with function='{function}', variable='{variable}'")  # Debug
                return tool_func(function, variable)
                
            elif tool_name == "integral":
                parts = parameters.split(',')
                function = parts[0].strip()
                variable = parts[1].strip() if len(parts) > 1 else 'x'
                return tool_func(function, variable)
                
            else:
                return f"Tool '{tool_name}' parameter parsing not implemented"
                
        except Exception as e:
            return f"Tool execution error: {str(e)}"
    
    def _analyze_problem(self, context: str) -> str:
        """Analyze the problem to extract key information."""
        
        prompt = f"""
        Analyze this mathematical problem and extract key information:
        
        {context}
        
        Provide a brief analysis covering:
        1. What type of problem this is
        2. What information is given
        3. What needs to be found
        4. What approach should be used
        
        Keep the analysis concise and focused.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="google/gemini-2.0-flash-001",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Analysis error: {str(e)}"
    
    def _generate_final_solution(self, context: str, history: List[Dict]) -> str:
        """Generate the final solution based on all reasoning steps."""
        
        steps_text = "\n".join([
            f"Step {h['iteration']}: {h.get('tool_result', h.get('analysis', h['thought']))}"
            for h in history
        ])
        
        prompt = f"""
        Based on the following reasoning process, provide the final solution:
        
        ORIGINAL PROBLEM:
        {context.split('Tool result:')[0].split('Analysis:')[0]}
        
        REASONING STEPS:
        {steps_text}
        
        Provide a clear, concise final answer with the solution.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="google/gemini-2.0-flash-001",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Solution generation error: {str(e)}"
    
    def _observe(self, context: str, history: List[Dict]) -> str:
        """Update context based on new observations."""
        # For now, just return the current context
        # In a more advanced version, this could synthesize information
        return context

# Example usage and testing
if __name__ == "__main__":
    agent = EnhancedReActMathAgent()
    
    # Test problems
    test_problems = [
        "Solve for x: 2x + 5 = 15",
        "Find the derivative of f(x) = x² + 3x + 2",
        "A rectangle has length 8 cm and width 5 cm. What is its area?",
        "Calculate the integral of 2x from 0 to 5"
    ]
    
    print("🧠 TESTING ENHANCED REACT MATH AGENT")
    print("=" * 60)
    
    for i, problem in enumerate(test_problems, 1):
        print(f"\n🔢 TEST PROBLEM {i}: {problem}")
        print("=" * 60)
        
        result = agent.solve_problem(problem)
        
        print(f"\n📊 FINAL RESULT:")
        print(f"   Status: {result['status']}")
        print(f"   Iterations: {result['iterations_used']}")
        print(f"   Solution: {result['solution']}")
        
        input(f"\nPress Enter to continue to next problem...")
