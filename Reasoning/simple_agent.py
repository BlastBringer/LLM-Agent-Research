import json
import re
from typing import Dict, Any, List
import sympy
class SimpleAgent:
    """
    A simple agent that can execute different tools based on subtasks.
    This replaces the complex LangChain ReAct agent with a simpler implementation.
    """
    def __init__(self):
        self.tools = {
            "Calculator": self._calculator_tool,
            "SymbolicSolver": self._symbolic_solver_tool,
            "GeneralQuery": self._general_query_tool,
            "WebSearch": self._web_search_tool
        }
        print("🤖 Simple Agent initialized with tools:", list(self.tools.keys()))
    
    def _calculator_tool(self, tool_input: Dict[str, Any]) -> str:
        """
        Basic calculator functionality with safety checks.
        
        Args:
            tool_input: Dictionary containing 'expression' key.
            
        Returns:
            String result of the calculation.
        """
        expression = tool_input.get("expression", "")
        print(f"🧮 Calculator: Evaluating '{expression}'")
        
        try:
            # Safe evaluation with restricted builtins
            allowed_names = {
                "abs": abs, "round": round, "min": min, "max": max,
                "sum": sum, "pow": pow, "sqrt": lambda x: x**0.5,
                "pi": 3.14159265359, "e": 2.71828182846,
                "__builtins__": {}
            }
            
            # Basic safety check - no dangerous functions
            dangerous_patterns = ['import', 'exec', 'eval', '__', 'open', 'file']
            if any(pattern in expression.lower() for pattern in dangerous_patterns):
                return "Calculator Error: Expression contains potentially dangerous operations"
            
            result = eval(expression, allowed_names, {})
            return f"Calculation result: {result}"
            
        except ZeroDivisionError:
            return "Calculator Error: Division by zero"
        except SyntaxError as e:
            return f"Calculator Error: Invalid syntax - {str(e)}"
        except Exception as e:
            return f"Symbolic Solver Error: {str(e)}"
    
    def _solve_single_equation(self, equation: str, variable: str) -> str:
        """
        Solve a single linear equation for one variable.
        
        Args:
            equation: String equation like "2*x + 3 = 7"
            variable: Variable to solve for like "x"
            
        Returns:
            Solution string
        """
        try:
            # Simple pattern matching for basic linear equations
            # This handles cases like: ax + b = c, ax = c, x + b = c, etc.
            
            if "=" not in equation:
                return f"Error: No equals sign found in equation '{equation}'"
            
            left, right = equation.split("=")
            left, right = left.strip(), right.strip()
            
            # Try to evaluate the right side
            try:
                right_value = eval(right, {"__builtins__": {}}, {})
            except:
                return f"Error: Cannot evaluate right side '{right}'"
            
            # Simple cases for left side
            if left == variable:
                # x = value
                return f"Solution: {variable} = {right_value}"
            
            # Pattern: coefficient * variable
            pattern1 = rf"(\d+\.?\d*)\s*\*\s*{variable}"
            match1 = re.search(pattern1, left)
            if match1 and left.replace(match1.group(0), "").strip() == "":
                coeff = float(match1.group(1))
                result = right_value / coeff
                return f"Solution: {variable} = {result}"
            
            # Pattern: variable + constant = value
            pattern2 = rf"{variable}\s*\+\s*(\d+\.?\d*)"
            match2 = re.search(pattern2, left)
            if match2:
                constant = float(match2.group(1))
                result = right_value - constant
                return f"Solution: {variable} = {result}"
            
            # Pattern: variable - constant = value
            pattern3 = rf"{variable}\s*-\s*(\d+\.?\d*)"
            match3 = re.search(pattern3, left)
            if match3:
                constant = float(match3.group(1))
                result = right_value + constant
                return f"Solution: {variable} = {result}"
            
            return f"Cannot solve equation '{equation}' - pattern not recognized"
            
        except Exception as e:
            return f"Error solving single equation: {str(e)}"
    


    def _solve_two_equations(self, equations: List[str], variables: List[str]) -> str:
        try:
         # Convert variable names to SymPy symbols
            syms = sympy.symbols(' '.join(variables))
        # Parse equations into SymPy Eq objects
            eqs = []
            for eq in equations:
                left, right = eq.split("=")
                eqs.append(sympy.Eq(sympy.sympify(left.strip()), sympy.sympify(right.strip())))
        # Solve the system
            sol = sympy.solve(eqs, syms, dict=True)
            return f"Two-equation solution: {sol}"
        except Exception as e:
            return f"Error solving two equations: {str(e)}"
    
    def _general_query_tool(self, tool_input: Dict[str, Any]) -> str:
        """
        Handle general queries that don't fit other tools.
        
        Args:
            tool_input: Dictionary containing the query data.
            
        Returns:
            String response to the query.
        """
        print("🤔 GeneralQuery: Processing general reasoning task")
        
        try:
            # Extract different types of input
            if "text" in tool_input:
                query_text = tool_input["text"]
                return f"General reasoning result: Processed query '{query_text}'. This would involve logical reasoning steps."
            
            elif "problem_data" in tool_input:
                problem_data = tool_input["problem_data"]
                return f"General reasoning result: Analyzed problem data with {len(problem_data)} components."
            
            else:
                # Handle the full parsed data structure
                components = []
                for key, value in tool_input.items():
                    if key != "error":
                        components.append(f"{key}: {str(value)[:50]}...")
                
                return f"General reasoning result: Processed {len(components)} data components: {', '.join(components)}"
                
        except Exception as e:
            return f"General Query Error: {str(e)}"
    
    def _web_search_tool(self, tool_input: Dict[str, Any]) -> str:
        """
        Mock web search tool (replace with real search API in production).
        
        Args:
            tool_input: Dictionary containing 'query' key.
            
        Returns:
            Mock search results.
        """
        query = tool_input.get("query", "")
        print(f"🔍 WebSearch: Searching for '{query}'")
        
        # Mock search results
        mock_results = [
            f"Search result 1 for '{query}': Mathematical concepts and formulas",
            f"Search result 2 for '{query}': Step-by-step solution examples",
            f"Search result 3 for '{query}': Related mathematical theorems"
        ]
        
        return f"Web search results for '{query}': {json.dumps(mock_results, indent=2)}"
    
    def run(self, subtask: Dict[str, Any]) -> str:
        """
        Execute the given subtask using the appropriate tool.
        
        Args:
            subtask: Dictionary containing tool_name, tool_input, and human_readable_goal.
            
        Returns:
            String result from executing the subtask.
        """
        if not subtask or not isinstance(subtask, dict):
            return "Agent Error: Invalid subtask provided"
        
        tool_name = subtask.get("tool_name", "GeneralQuery")
        tool_input = subtask.get("tool_input", {})
        goal = subtask.get("human_readable_goal", "No goal specified")
        
        print(f"\n🤖 Agent executing subtask:")
        print(f"   Tool: {tool_name}")
        print(f"   Goal: {goal}")
        
        if tool_name not in self.tools:
            available_tools = list(self.tools.keys())
            return f"Agent Error: Unknown tool '{tool_name}'. Available tools: {available_tools}"
        
        try:
            tool_function = self.tools[tool_name]
            result = tool_function(tool_input)
            print(f"   ✅ Tool execution completed")
            return result
            
        except Exception as e:
            error_msg = f"Agent Error: Failed to execute {tool_name} - {str(e)}"
            print(f"   ❌ {error_msg}")
            return error_msg
    
    def get_available_tools(self) -> List[str]:
        """
        Get list of available tools.
        
        Returns:
            List of tool names.
        """
        return list(self.tools.keys())
    
    def add_tool(self, tool_name: str, tool_function):
        """
        Add a new tool to the agent.
        
        Args:
            tool_name: Name of the new tool.
            tool_function: Function that implements the tool.
        """
        self.tools[tool_name] = tool_function
        print(f"🔧 Added new tool: {tool_name}")

    def _symbolic_solver_tool(self, tool_input: Dict[str, Any]) -> str:
        """
        Simplified symbolic solver for basic equation systems.
        
        Args:
            tool_input: Dictionary containing 'equations' and 'variables_to_solve'.
            
        Returns:
            String result of the equation solving.
        """
        equations = tool_input.get("equations", [])
        variables = tool_input.get("variables_to_solve", [])
        
        print(f"🔧 SymbolicSolver: Solving {len(equations)} equations for {variables}")
        
        try:
            # This is a simplified solver - in a real implementation, you'd use SymPy
            # For now, we'll try to handle some basic linear equation cases
            
            if len(equations) == 1 and len(variables) == 1:
                # Single equation, single variable
                return self._solve_single_equation(equations[0], variables[0])
            
            elif len(equations) == 2 and len(variables) == 2:
                # Two equations, two variables - try substitution method
                return self._solve_two_equations(equations, variables)
            
            else:
                # For more complex cases, return a simulated solution
                solution = {}
                for i, var in enumerate(variables):
                    solution[var] = f"solution_value_{i + 1}"
                
                result = {
                    "method": "simulated_solution",
                    "equations_input": equations,
                    "variables": variables,
                    "solution": solution,
                    "note": "This is a simulated solution. In a real implementation, use SymPy."
                }
                
                return f"Symbolic solution: {json.dumps(result, indent=2)}"
                
        except Exception as e:
            return f"Symbolic Solver Error: {str(e)}"

# --- Example Usage ---
if __name__ == "__main__":
    agent = SimpleAgent()
    
    # Test Calculator
    calc_subtask = {
        "tool_name": "Calculator",
        "tool_input": {"expression": "25 * 4 + 18 / 3"},
        "human_readable_goal": "Calculate arithmetic expression"
    }
    print("--- Testing Calculator ---")
    result1 = agent.run(calc_subtask)
    print(f"Result: {result1}\n")
    
    # Test SymbolicSolver
    solver_subtask = {
        "tool_name": "SymbolicSolver",
        "tool_input": {
            "equations": ["2*x + 3 = 7"],
            "variables_to_solve": ["x"]
        },
        "human_readable_goal": "Solve linear equation"
    }
    print("--- Testing SymbolicSolver ---")
    result2 = agent.run(solver_subtask)
    print(f"Result: {result2}\n")
    
    # Test GeneralQuery
    general_subtask = {
        "tool_name": "GeneralQuery",
        "tool_input": {"text": "What is the relationship between algebra and geometry?"},
        "human_readable_goal": "Answer general mathematical question"
    }
    print("--- Testing GeneralQuery ---")
    result3 = agent.run(general_subtask)
    print(f"Result: {result3}\n")
    
    # Test error handling
    error_subtask = {
        "tool_name": "NonExistentTool",
        "tool_input": {},
        "human_readable_goal": "Test error handling"
    }
    print("--- Testing Error Handling ---")
    result4 = agent.run(error_subtask)
    print(f"Result: {result4}\n")
    
    