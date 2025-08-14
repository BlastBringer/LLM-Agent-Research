#!/usr/bin/env python3
"""
🤖 EXTERNAL AGENT SYSTEM - BASE FRAMEWORK
=========================================

This represents the external Agent system that receives subtasks
from the Reasoning Engine and executes them using specialized tools.
"""

import asyncio
import json
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """
    Base class for all specialized mathematical agents.
    """
    
    def __init__(self, agent_name: str, capabilities: List[str]):
        self.agent_name = agent_name
        self.capabilities = capabilities
        self.tool_registry = {}
        self.status = "available"
        
    @abstractmethod
    def process_subtask(self, subtask: Dict[str, Any]) -> Dict[str, Any]:
        """Process a subtask and return results."""
        pass
    
    def register_tool(self, tool_name: str, tool_function):
        """Register a tool with this agent."""
        self.tool_registry[tool_name] = tool_function
        
    def get_available_tools(self) -> List[str]:
        """Get list of available tools."""
        return list(self.tool_registry.keys())

class SubtaskInterpreter:
    """
    Interprets subtasks received from the Reasoning Engine.
    """
    
    def __init__(self):
        print("🔍 Subtask Interpreter initialized.")
    
    def interpret_subtask(self, subtask: Dict[str, Any]) -> Dict[str, Any]:
        """
        Interpret and validate a subtask from the Reasoning Engine.
        
        Args:
            subtask: Raw subtask from the delegation system
            
        Returns:
            Interpreted and validated subtask
        """
        interpreted = {
            "task_id": subtask.get('subtask_id', 'unknown'),
            "operation": subtask.get('operation_type', 'unknown'),
            "description": subtask.get('description', ''),
            "input_data": self._parse_input_data(subtask.get('input_data', '')),
            "output_requirements": subtask.get('output_format', 'general'),
            "priority": subtask.get('priority', 3),
            "validation_criteria": subtask.get('validation_criteria', {}),
            "agent_requirements": subtask.get('agent_requirements', {})
        }
        
        return interpreted
    
    def _parse_input_data(self, input_data: Any) -> Dict[str, Any]:
        """Parse and structure input data."""
        if isinstance(input_data, str):
            # Try to extract mathematical expressions, equations, etc.
            parsed = {
                "raw_input": input_data,
                "equations": self._extract_equations(input_data),
                "expressions": self._extract_expressions(input_data),
                "numbers": self._extract_numbers(input_data)
            }
        else:
            parsed = {"raw_input": input_data}
            
        return parsed
    
    def _extract_equations(self, text: str) -> List[str]:
        """Extract equations from text."""
        import re
        # Simple equation detection (can be enhanced)
        equations = re.findall(r'[^=]+=[^=]+', text)
        return [eq.strip() for eq in equations]
    
    def _extract_expressions(self, text: str) -> List[str]:
        """Extract mathematical expressions."""
        import re
        # Simple expression detection
        expressions = re.findall(r'[a-zA-Z0-9+\-*/^().\s]+', text)
        return [expr.strip() for expr in expressions if any(op in expr for op in ['+', '-', '*', '/', '^'])]
    
    def _extract_numbers(self, text: str) -> List[float]:
        """Extract numbers from text."""
        import re
        numbers = re.findall(r'-?\d+\.?\d*', text)
        return [float(num) for num in numbers]

class ToolSelector:
    """
    Selects appropriate tools based on the subtask requirements.
    """
    
    def __init__(self):
        # Tool capability mapping
        self.tool_capabilities = {
            "symbolic_solver": ["equation_solving", "system_solving"],
            "derivative_calculator": ["derivative_calculation", "differentiation"],
            "integral_calculator": ["integral_calculation", "integration"],
            "area_calculator": ["area_calculation", "geometry"],
            "volume_calculator": ["volume_calculation", "3d_geometry"],
            "matrix_calculator": ["matrix_operations", "linear_algebra"],
            "stats_calculator": ["statistics", "data_analysis"],
            "expression_simplifier": ["algebraic_manipulation", "simplification"],
            "equation_validator": ["validation", "verification"]
        }
        
        print("🔧 Tool Selector initialized.")
    
    def select_tools(self, subtask: Dict[str, Any], available_tools: List[str]) -> List[str]:
        """
        Select the most appropriate tools for a subtask.
        
        Args:
            subtask: Interpreted subtask
            available_tools: List of tools available to the agent
            
        Returns:
            List of recommended tools
        """
        operation = subtask.get('operation', 'unknown')
        requirements = subtask.get('agent_requirements', {}).get('required_capabilities', [])
        
        selected_tools = []
        
        # Score each available tool
        for tool in available_tools:
            score = self._score_tool(tool, operation, requirements)
            if score > 0:
                selected_tools.append((tool, score))
        
        # Sort by score and return tool names
        selected_tools.sort(key=lambda x: x[1], reverse=True)
        return [tool[0] for tool in selected_tools[:3]]  # Top 3 tools
    
    def _score_tool(self, tool: str, operation: str, requirements: List[str]) -> int:
        """Score a tool's suitability for the operation."""
        score = 0
        
        tool_caps = self.tool_capabilities.get(tool, [])
        
        # Direct operation match
        if operation in tool_caps:
            score += 10
        
        # Requirement match
        for req in requirements:
            if req in tool_caps:
                score += 5
        
        # Partial matches
        if operation in tool:
            score += 3
        
        return score

class ToolWrapper:
    """
    Wraps external tools and provides a standardized interface.
    """
    
    def __init__(self, tool_name: str, tool_function):
        self.tool_name = tool_name
        self.tool_function = tool_function
        self.execution_count = 0
        self.error_count = 0
    
    def execute(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        """
        Execute the wrapped tool with error handling and logging.
        
        Args:
            input_data: Input for the tool
            **kwargs: Additional parameters
            
        Returns:
            Standardized tool result
        """
        self.execution_count += 1
        
        try:
            result = self.tool_function(input_data, **kwargs)
            
            return {
                "tool_name": self.tool_name,
                "status": "success",
                "result": result,
                "execution_time": "simulated_time",
                "error_message": None
            }
        
        except Exception as e:
            self.error_count += 1
            return {
                "tool_name": self.tool_name,
                "status": "error",
                "result": None,
                "execution_time": "simulated_time",
                "error_message": str(e)
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics for this tool."""
        return {
            "executions": self.execution_count,
            "errors": self.error_count,
            "success_rate": (self.execution_count - self.error_count) / max(self.execution_count, 1)
        }

class ExecutionEngine:
    """
    Manages the execution of tools and coordination of agent operations.
    """
    
    def __init__(self):
        self.active_executions = {}
        self.execution_history = []
        print("⚡ Execution Engine initialized.")
    
    def execute_task(self, subtask: Dict[str, Any], selected_tools: List[str], tool_registry: Dict) -> Dict[str, Any]:
        """
        Execute a subtask using the selected tools.
        
        Args:
            subtask: The subtask to execute
            selected_tools: List of tools to use
            tool_registry: Available tools
            
        Returns:
            Execution result
        """
        task_id = subtask.get('task_id', 'unknown')
        
        print(f"⚡ Executing task {task_id} with tools: {selected_tools}")
        
        execution_results = []
        
        for tool_name in selected_tools:
            if tool_name in tool_registry:
                tool_wrapper = tool_registry[tool_name]
                
                # Execute tool with subtask input
                tool_result = tool_wrapper.execute(
                    subtask.get('input_data', {}),
                    operation=subtask.get('operation', 'unknown')
                )
                
                execution_results.append(tool_result)
            else:
                execution_results.append({
                    "tool_name": tool_name,
                    "status": "not_available",
                    "result": None,
                    "error_message": f"Tool {tool_name} not found in registry"
                })
        
        # Aggregate results
        final_result = self._aggregate_tool_results(execution_results, subtask)
        
        # Store in history
        self.execution_history.append({
            "task_id": task_id,
            "tools_used": selected_tools,
            "result": final_result,
            "timestamp": "simulated_timestamp"
        })
        
        return final_result
    
    def _aggregate_tool_results(self, tool_results: List[Dict], subtask: Dict) -> Dict[str, Any]:
        """Aggregate results from multiple tools."""
        
        successful_results = [r for r in tool_results if r['status'] == 'success']
        
        if not successful_results:
            return {
                "status": "failed",
                "result": "No tools executed successfully",
                "confidence": 0.0,
                "tools_used": [r['tool_name'] for r in tool_results],
                "errors": [r.get('error_message') for r in tool_results if r.get('error_message')]
            }
        
        # For now, use the first successful result
        # In practice, this would be more sophisticated
        primary_result = successful_results[0]
        
        return {
            "status": "success",
            "result": primary_result['result'],
            "confidence": 0.95,  # Simulated confidence
            "tools_used": [r['tool_name'] for r in successful_results],
            "reasoning_steps": [f"Applied {r['tool_name']}" for r in successful_results],
            "validation_status": "passed"  # Simulated validation
        }

class OutputCleaner:
    """
    Cleans and formats output before sending back to the Reasoning Engine.
    """
    
    def __init__(self):
        print("🧹 Output Cleaner initialized.")
    
    def clean_output(self, raw_result: Dict[str, Any], subtask: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean and format the output according to requirements.
        
        Args:
            raw_result: Raw result from execution engine
            subtask: Original subtask with requirements
            
        Returns:
            Cleaned and formatted result
        """
        output_format = subtask.get('output_requirements', 'general')
        
        cleaned = {
            "subtask_id": subtask.get('task_id', 'unknown'),
            "status": raw_result.get('status', 'unknown'),
            "result": self._format_result(raw_result.get('result'), output_format),
            "confidence_score": raw_result.get('confidence', 0.0),
            "reasoning_steps": raw_result.get('reasoning_steps', []),
            "validation_status": raw_result.get('validation_status', 'unknown'),
            "execution_metadata": {
                "tools_used": raw_result.get('tools_used', []),
                "execution_time": "simulated_time",
                "errors": raw_result.get('errors', [])
            }
        }
        
        return cleaned
    
    def _format_result(self, result: Any, output_format: str) -> str:
        """Format result according to the specified format."""
        
        if output_format == 'variable_solution':
            return f"Solution: {result}"
        elif output_format == 'derivative_expression':
            return f"Derivative: {result}"
        elif output_format == 'numerical_value':
            return str(result)
        else:
            return str(result)

# Example Agent Implementation
class AlgebraSpecialistAgent(BaseAgent):
    """
    Specialized agent for algebraic operations.
    """
    
    def __init__(self):
        super().__init__("algebra_specialist", ["equation_solving", "system_solving", "algebraic_manipulation"])
        
        # Initialize components
        self.interpreter = SubtaskInterpreter()
        self.tool_selector = ToolSelector()
        self.execution_engine = ExecutionEngine()
        self.output_cleaner = OutputCleaner()
        
        # Register tools (simulated)
        self._register_algebra_tools()
    
    def _register_algebra_tools(self):
        """Register algebra-specific tools."""
        
        def symbolic_solver(input_data, **kwargs):
            # Simulated symbolic solver
            if isinstance(input_data, dict) and input_data.get('equations'):
                return "x = 5"  # Simulated solution
            return "algebraic_solution"
        
        def equation_validator(input_data, **kwargs):
            # Simulated validator
            return "validation_passed"
        
        self.register_tool("symbolic_solver", ToolWrapper("symbolic_solver", symbolic_solver))
        self.register_tool("equation_validator", ToolWrapper("equation_validator", equation_validator))
    
    def process_subtask(self, subtask: Dict[str, Any]) -> Dict[str, Any]:
        """Process an algebraic subtask."""
        
        # Step 1: Interpret the subtask
        interpreted_task = self.interpreter.interpret_subtask(subtask)
        
        # Step 2: Select appropriate tools
        available_tools = self.get_available_tools()
        selected_tools = self.tool_selector.select_tools(interpreted_task, available_tools)
        
        # Step 3: Execute the task
        execution_result = self.execution_engine.execute_task(
            interpreted_task, selected_tools, self.tool_registry
        )
        
        # Step 4: Clean and format output
        final_result = self.output_cleaner.clean_output(execution_result, interpreted_task)
        
        return final_result

# Example usage
if __name__ == "__main__":
    # Create an algebra specialist agent
    agent = AlgebraSpecialistAgent()
    
    # Test with a sample subtask
    test_subtask = {
        "subtask_id": "ST_001",
        "description": "Solve the equation 2x + 5 = 15",
        "operation_type": "algebra",
        "input_data": "2x + 5 = 15",
        "output_format": "variable_solution",
        "priority": 1
    }
    
    result = agent.process_subtask(test_subtask)
    
    print("\n🤖 AGENT EXECUTION RESULT:")
    print("=" * 40)
    print(json.dumps(result, indent=2))
