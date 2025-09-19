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
                
                # Fix implicit multiplication comprehensively
                left = left.strip()
                right = right.strip()
                
                # Apply comprehensive implicit multiplication fixes
                left = re.sub(r'(\d+)([a-zA-Z])', r'\1*\2', left)    # 2x -> 2*x
                left = re.sub(r'(\d+)(\()', r'\1*\2', left)          # 4( -> 4*(
                left = re.sub(r'(\))(\d+)', r'\1*\2', left)          # )2 -> )*2
                left = re.sub(r'(\))([a-zA-Z])', r'\1*\2', left)     # )x -> )*x
                left = re.sub(r'(\))(\()', r'\1*\2', left)           # )( -> )*(
                
                right = re.sub(r'(\d+)([a-zA-Z])', r'\1*\2', right)  # 2x -> 2*x
                right = re.sub(r'(\d+)(\()', r'\1*\2', right)        # 4( -> 4*(
                right = re.sub(r'(\))(\d+)', r'\1*\2', right)        # )2 -> )*2
                right = re.sub(r'(\))([a-zA-Z])', r'\1*\2', right)   # )x -> )*x
                right = re.sub(r'(\))(\()', r'\1*\2', right)         # )( -> )*(
                
                eq = sympy.Eq(sympy.sympify(left), sympy.sympify(right))
            else:
                # Fix implicit multiplication for single expressions too
                equation = re.sub(r'(\d+)([a-zA-Z])', r'\1*\2', equation)
                equation = re.sub(r'(\d+)(\()', r'\1*\2', equation)
                equation = re.sub(r'(\))(\d+)', r'\1*\2', equation)
                equation = re.sub(r'(\))([a-zA-Z])', r'\1*\2', equation)
                equation = re.sub(r'(\))(\()', r'\1*\2', equation)
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
        self.model_name = os.getenv("MODEL_NAME", "meta-llama/llama-3.1-8b-instruct:free")
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
    
    def create_enhanced_prompt(self, problem: str, iteration: int = 0, previous_observations: List[str] = None) -> str:
        """
        Create enhanced prompt with DeepSeek-Math style formatting.
        
        This implements the enhanced prompting strategy from DeepSeek-Math research
        which significantly improves mathematical reasoning performance.
        
        Args:
            problem: The mathematical problem to solve
            iteration: Current iteration number
            previous_observations: Previous tool results and observations
            
        Returns:
            Enhanced prompt string with structured format
        """
        if iteration == 0:
            # Initial prompt with clear structure
            prompt = f"""You are an expert mathematician solving a complex problem step-by-step.

Problem: {problem}

Please solve this step-by-step using the following format:
THOUGHT: [Your detailed reasoning about what to do next]
ACTION: [The specific tool or method you will use]
OBSERVATION: [The result from your action]

Available tools and their usage:
- calculator(expression): For numerical computations
- symbolic_solver(equation): For solving equations symbolically  
- derivative_calculator(function, variable): For derivatives
- integral_calculator(function, variable): For integrals
- equation_solver(equation): For algebraic equations
- matrix_calculator(operation, matrix): For matrix operations
- trigonometry_calculator(expression): For trig functions
- factorization_tool(expression): For factoring
- simplify_expression(expression): For simplification
- graph_analyzer(function): For function analysis

Important instructions:
1. Show your reasoning clearly in THOUGHT sections
2. Use appropriate tools for each step
3. Build upon previous observations
4. When you reach the final answer, put it in \\boxed{{answer}} format
5. Be systematic and thorough

Begin your solution:
THOUGHT: Let me analyze this problem and determine the best approach."""

        else:
            # Continuation prompt with context
            context = ""
            if previous_observations:
                context = "\n\nPrevious observations:\n" + "\n".join(previous_observations[-3:])  # Last 3 observations
            
            prompt = f"""Continue solving the problem: {problem}

{context}

Continue with your step-by-step solution:
THOUGHT: Based on the previous observations, I need to..."""

        return prompt

    def create_final_answer_prompt(self, problem: str, solution_steps: List[str]) -> str:
        """
        Create a prompt to extract and format the final answer.
        
        Args:
            problem: Original problem
            solution_steps: List of solution steps taken
            
        Returns:
            Prompt to generate final formatted answer
        """
        steps_summary = "\n".join(solution_steps[-5:])  # Last 5 steps
        
        return f"""Based on the solution steps below, provide the final answer to this problem:

Problem: {problem}

Solution steps:
{steps_summary}

Please provide the final answer in this format:
The final answer is \\boxed{{[your answer here]}}.

Make sure the answer is:
1. Mathematically correct
2. In its simplest form
3. Properly formatted (e.g., fractions simplified, decimals rounded appropriately)
4. Clearly stated within the \\boxed{{}} format

Final answer:"""

    def solve_problem(self, problem: str, max_iterations: int = 20) -> Dict[str, Any]:
        """
        Solve a mathematical problem using enhanced ReAct reasoning with DeepSeek-Math prompting.
        
        Args:
            problem: The mathematical problem to solve
            max_iterations: Maximum number of reasoning iterations
            
        Returns:
            Dictionary containing the solution and reasoning steps
        """
        print(f"🎯 Solving problem with enhanced prompting: {problem}")
        print("=" * 60)
        
        reasoning_history = []
        observations_history = []
        current_context = problem
        
        for iteration in range(max_iterations):
            print(f"\n🔄 ITERATION {iteration + 1}")
            print("-" * 30)
            
            # Create enhanced prompt for this iteration
            enhanced_prompt = self.create_enhanced_prompt(
                problem=problem, 
                iteration=iteration, 
                previous_observations=observations_history
            )
            
            # Get enhanced reasoning using the structured prompt
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are an expert mathematician. Follow the THOUGHT-ACTION-OBSERVATION format precisely."},
                        {"role": "user", "content": enhanced_prompt}
                    ],
                    temperature=0.1,  # Low temperature for mathematical precision
                    max_tokens=1000
                )
                
                llm_response = response.choices[0].message.content
                print(f"🧠 LLM Response:\n{llm_response}")
                
                # Parse the structured response
                thought_match = re.search(r'THOUGHT:\s*(.+?)(?=ACTION:|$)', llm_response, re.DOTALL)
                action_match = re.search(r'ACTION:\s*(.+?)(?=OBSERVATION:|THOUGHT:|$)', llm_response, re.DOTALL)
                
                if thought_match:
                    thought_text = thought_match.group(1).strip()
                    print(f"💭 THOUGHT: {thought_text}")
                    
                    if action_match:
                        action_text = action_match.group(1).strip()
                        print(f"⚡ ACTION: {action_text}")
                        
                        # Execute the action using our enhanced tool system
                        tool_result = self._execute_enhanced_action(action_text)
                        print(f"👀 OBSERVATION: {tool_result}")
                        
                        # Store the reasoning step
                        step_data = {
                            'iteration': iteration + 1,
                            'thought': thought_text,
                            'action': action_text,
                            'tool_result': tool_result,
                            'analysis': f"Step {iteration + 1} completed successfully"
                        }
                        reasoning_history.append(step_data)
                        observations_history.append(tool_result)
                        
                        # Check if we have a final answer in boxed format
                        # BUT allow for multiple iterations for complex problems
                        has_boxed_answer = '\\boxed{' in llm_response or '\\boxed{' in tool_result
                        
                        if has_boxed_answer:
                            print("� Found boxed answer - checking if solution is complete...")
                            
                            # For complex problems, continue for at least 2-3 iterations
                            min_iterations = 2 if any(word in problem.lower() for word in [
                                'step by step', 'show work', 'explain', 'detailed', 'verify', 'check'
                            ]) else 1
                            
                            if iteration + 1 >= min_iterations:
                                print("🎉 Solution complete after sufficient iterations!")
                                
                                # Extract the boxed answer
                                boxed_pattern = r'\\boxed\{([^}]+)\}'
                                boxed_match = re.search(boxed_pattern, llm_response + " " + tool_result)
                                final_answer = boxed_match.group(1) if boxed_match else tool_result
                                
                                return {
                                    'status': 'solved',
                                    'solution': final_answer,
                                    'reasoning_steps': reasoning_history,
                                    'iterations_used': iteration + 1,
                                    'method': 'enhanced_react_reasoning'
                                }
                            else:
                                print(f"🔄 Continuing for more detailed reasoning... (iteration {iteration + 1}/{min_iterations})")
                        
                        # Update context for next iteration
                        current_context = f"Problem: {problem}\nCurrent progress: {tool_result}"
                        
                    else:
                        print("⚠️  No ACTION found in response")
                        break
                else:
                    print("⚠️  No THOUGHT found in response")
                    break
                    
            except Exception as e:
                print(f"❌ Error in iteration {iteration + 1}: {e}")
                break
        
        # If we reach here, try to extract final answer from the last steps
        print("\n🔍 Extracting final answer from reasoning history...")
        
        if reasoning_history:
            # Look for boxed answers in any step
            for step in reversed(reasoning_history):
                for field in ['thought', 'action', 'tool_result']:
                    text = str(step.get(field, ''))
                    boxed_match = re.search(r'\\boxed\{([^}]+)\}', text)
                    if boxed_match:
                        final_answer = boxed_match.group(1)
                        return {
                            'status': 'solved',
                            'solution': final_answer,
                            'reasoning_steps': reasoning_history,
                            'iterations_used': len(reasoning_history),
                            'method': 'enhanced_react_reasoning'
                        }
            
            # If no boxed answer, use final answer extraction prompt
            solution_steps = [f"Step {s['iteration']}: {s['tool_result']}" for s in reasoning_history]
            final_prompt = self.create_final_answer_prompt(problem, solution_steps)
            
            try:
                final_response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "Extract and format the final answer clearly."},
                        {"role": "user", "content": final_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=200
                )
                
                final_text = final_response.choices[0].message.content
                boxed_match = re.search(r'\\boxed\{([^}]+)\}', final_text)
                final_answer = boxed_match.group(1) if boxed_match else reasoning_history[-1]['tool_result']
                
                return {
                    'status': 'solved',
                    'solution': final_answer,
                    'reasoning_steps': reasoning_history,
                    'iterations_used': len(reasoning_history),
                    'method': 'enhanced_react_reasoning'
                }
                
            except Exception as e:
                print(f"⚠️  Final answer extraction failed: {e}")
        
        return {
            'status': 'incomplete',
            'solution': reasoning_history[-1]['tool_result'] if reasoning_history else 'No solution found',
            'reasoning_steps': reasoning_history,
            'iterations_used': len(reasoning_history),
            'method': 'enhanced_react_reasoning'
        }

    def _execute_enhanced_action(self, action: str) -> str:
        """
        Execute an action with enhanced tool parsing.
        
        Args:
            action: The action string from the LLM
            
        Returns:
            Result from executing the action
        """
        # Parse different action formats
        action = action.strip()
        tool_name = "unknown"  # Default value
        
        # Handle tool calls in various formats
        if '(' in action and ')' in action:
            # Extract tool name and parameters
            tool_match = re.match(r'(\w+)\((.*)\)', action)
            if tool_match:
                tool_name = tool_match.group(1).lower()
                params = tool_match.group(2).strip('\'"')
                
                # Map to our tools
                if tool_name in ['calculator', 'calculate']:
                    return self.calculator(params)
                elif tool_name in ['symbolic_solver', 'solve']:
                    return self.symbolic_solver(params)
                elif tool_name in ['derivative_calculator', 'derivative']:
                    parts = params.split(',')
                    func = parts[0].strip() if parts else params
                    var = parts[1].strip() if len(parts) > 1 else 'x'
                    return self.derivative_calculator(func, var)
                elif tool_name in ['integral_calculator', 'integrate']:
                    parts = params.split(',')
                    func = parts[0].strip() if parts else params
                    var = parts[1].strip() if len(parts) > 1 else 'x'
                    return self.integral_calculator(func, var)
                # Add more tool mappings as needed
        
        # Enhanced text-based action parsing
        action_lower = action.lower()
        if 'derivative' in action_lower or 'differentiate' in action_lower:
            # Try to extract function from the action text
            func_match = re.search(r'(?:of|derivative of|differentiate)\s+([^,\n]+)', action, re.IGNORECASE)
            if func_match:
                func = func_match.group(1).strip()
                return self.derivative_calculator(func)
        elif 'integral' in action_lower or 'integrate' in action_lower:
            # Try to extract function from the action text
            func_match = re.search(r'(?:integrate|integral of)\s+([^,\n]+)', action, re.IGNORECASE)
            if func_match:
                func = func_match.group(1).strip()
                return self.integral_calculator(func)
        elif 'solve' in action_lower and '=' in action:
            # Extract equation to solve
            eq_match = re.search(r'solve\s+([^,\n]+)', action, re.IGNORECASE)
            if eq_match:
                equation = eq_match.group(1).strip()
                return self.equation_solver(equation)
        elif 'simplify' in action_lower:
            # Extract expression to simplify
            expr_match = re.search(r'simplify\s+([^,\n]+)', action, re.IGNORECASE)
            if expr_match:
                expression = expr_match.group(1).strip()
                return self.expression_simplifier(expression)
        
        # Fallback
        return f"Could not parse action: '{action}'. Available tools: equation_solver, expression_simplifier, derivative_calculator, integral_calculator"

    def _act(self, thought: Dict[str, Any]) -> str:
        """Execute an action based on the thought."""
        try:
            action = thought.get('action', '')
            params = thought.get('parameters', '')
            
            if 'solve' in action.lower() and '=' in params:
                return self.equation_solver(params)
            elif 'simplify' in action.lower():
                return self.expression_simplifier(params)
            elif 'derivative' in action.lower() or 'differentiate' in action.lower():
                return self.derivative_calculator(params)
            elif 'integral' in action.lower() or 'integrate' in action.lower():
                return self.integral_calculator(params)
            else:
                return f"Could not execute action: {action} with parameters: {params}"
        except Exception as e:
            return f"Error executing action: {str(e)}"

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
