#!/usr/bin/env python3
"""
🛠️ ENHANCED MATHEMATICAL TOOLS FOR CREWAI AGENTS
==============================================

Mathematical tools that can be used by CrewAI agents to solve specific subtasks.
These tools bridge the CrewAI agent system with actual mathematical computations.
"""

import sympy
import numpy as np
import json
from typing import Dict, List, Any, Optional, Union
from crewai_tools import tool
import re

@tool("symbolic_math_calculator")
def symbolic_math_calculator(expression: str, operation: str = "simplify", variable: str = "x") -> str:
    """
    Perform symbolic mathematics operations using SymPy.
    
    Args:
        expression: Mathematical expression as string
        operation: Type of operation (simplify, solve, expand, factor, derivative, integral)
        variable: Variable to use for operations (default: x)
    
    Returns:
        String result of the mathematical operation
    """
    try:
        # Parse the expression
        expr = sympy.sympify(expression)
        var = sympy.Symbol(variable)
        
        if operation == "simplify":
            result = sympy.simplify(expr)
        elif operation == "solve":
            # Assume equation format like "x^2 - 4 = 0"
            if "=" in expression:
                left, right = expression.split("=")
                eq = sympy.Eq(sympy.sympify(left.strip()), sympy.sympify(right.strip()))
                result = sympy.solve(eq, var)
            else:
                result = sympy.solve(expr, var)
        elif operation == "expand":
            result = sympy.expand(expr)
        elif operation == "factor":
            result = sympy.factor(expr)
        elif operation == "derivative":
            result = sympy.diff(expr, var)
        elif operation == "integral":
            result = sympy.integrate(expr, var)
        else:
            result = f"Unknown operation: {operation}"
        
        return f"Result of {operation} on {expression}: {result}"
        
    except Exception as e:
        return f"SymPy calculation error: {str(e)}"

@tool("numerical_calculator")
def numerical_calculator(expression: str, precision: int = 6) -> str:
    """
    Perform numerical calculations with specified precision.
    
    Args:
        expression: Mathematical expression to evaluate
        precision: Number of decimal places (default: 6)
    
    Returns:
        Numerical result as string
    """
    try:
        # Replace common mathematical notation
        expression = expression.replace('^', '**')
        expression = expression.replace('π', str(np.pi))
        expression = expression.replace('e', str(np.e))
        
        # Safe evaluation
        allowed_names = {
            k: v for k, v in vars(np).items() 
            if not k.startswith('_')
        }
        allowed_names.update({
            'abs': abs, 'round': round, 'max': max, 'min': min,
            'sum': sum, 'pow': pow
        })
        
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        
        if isinstance(result, (int, float)):
            return f"Numerical result: {round(result, precision)}"
        else:
            return f"Numerical result: {result}"
        
    except Exception as e:
        return f"Numerical calculation error: {str(e)}"

@tool("geometry_calculator")
def geometry_calculator(shape: str, parameters: str) -> str:
    """
    Calculate geometric properties for various shapes.
    
    Args:
        shape: Type of shape (circle, rectangle, triangle, sphere, etc.)
        parameters: JSON string with shape parameters
    
    Returns:
        Calculated geometric properties
    """
    try:
        # Parse parameters
        params = json.loads(parameters) if isinstance(parameters, str) else parameters
        shape = shape.lower()
        
        if shape == "circle":
            radius = params.get('radius', 0)
            area = np.pi * radius**2
            circumference = 2 * np.pi * radius
            return f"Circle (r={radius}): Area = {area:.4f}, Circumference = {circumference:.4f}"
            
        elif shape == "rectangle":
            length = params.get('length', 0)
            width = params.get('width', 0)
            area = length * width
            perimeter = 2 * (length + width)
            return f"Rectangle ({length}×{width}): Area = {area}, Perimeter = {perimeter}"
            
        elif shape == "triangle":
            base = params.get('base', 0)
            height = params.get('height', 0)
            area = 0.5 * base * height
            return f"Triangle (base={base}, height={height}): Area = {area}"
            
        elif shape == "sphere":
            radius = params.get('radius', 0)
            volume = (4/3) * np.pi * radius**3
            surface_area = 4 * np.pi * radius**2
            return f"Sphere (r={radius}): Volume = {volume:.4f}, Surface Area = {surface_area:.4f}"
            
        else:
            return f"Shape '{shape}' not supported. Available: circle, rectangle, triangle, sphere"
            
    except Exception as e:
        return f"Geometry calculation error: {str(e)}"

@tool("equation_solver")
def equation_solver(equation: str, variables: str = "x") -> str:
    """
    Solve equations for specified variables.
    
    Args:
        equation: Equation to solve (e.g., "2*x + 3 = 7")
        variables: Variables to solve for (comma-separated)
    
    Returns:
        Solutions for the equation
    """
    try:
        # Parse variables
        var_list = [v.strip() for v in variables.split(',')]
        symbols = [sympy.Symbol(v) for v in var_list]
        
        # Parse equation
        if '=' in equation:
            left, right = equation.split('=', 1)
            eq = sympy.Eq(sympy.sympify(left.strip()), sympy.sympify(right.strip()))
        else:
            eq = sympy.sympify(equation)
        
        # Solve equation
        if len(symbols) == 1:
            solutions = sympy.solve(eq, symbols[0])
        else:
            solutions = sympy.solve(eq, symbols)
        
        return f"Solutions for {variables}: {solutions}"
        
    except Exception as e:
        return f"Equation solving error: {str(e)}"

@tool("calculus_operations")
def calculus_operations(function: str, operation: str, variable: str = "x", 
                       limits: Optional[str] = None) -> str:
    """
    Perform calculus operations (derivatives, integrals, limits).
    
    Args:
        function: Function to operate on
        operation: Type of operation (derivative, integral, limit, critical_points)
        variable: Variable to use (default: x)
        limits: For definite integrals, format: "a,b"
    
    Returns:
        Result of calculus operation
    """
    try:
        func = sympy.sympify(function)
        var = sympy.Symbol(variable)
        
        if operation == "derivative":
            result = sympy.diff(func, var)
            return f"d/d{variable}({function}) = {result}"
            
        elif operation == "integral":
            if limits:
                a, b = [float(x.strip()) for x in limits.split(',')]
                result = sympy.integrate(func, (var, a, b))
                return f"∫[{a} to {b}] {function} d{variable} = {result}"
            else:
                result = sympy.integrate(func, var)
                return f"∫ {function} d{variable} = {result} + C"
                
        elif operation == "critical_points":
            derivative = sympy.diff(func, var)
            critical_points = sympy.solve(derivative, var)
            return f"Critical points of {function}: {critical_points}"
            
        elif operation == "limit":
            # For limits, assume approaching 0 unless specified
            limit_point = 0
            result = sympy.limit(func, var, limit_point)
            return f"lim({variable}→{limit_point}) {function} = {result}"
            
        else:
            return f"Unknown calculus operation: {operation}"
            
    except Exception as e:
        return f"Calculus operation error: {str(e)}"

@tool("matrix_operations")
def matrix_operations(operation: str, matrix_data: str, matrix2_data: Optional[str] = None) -> str:
    """
    Perform matrix operations.
    
    Args:
        operation: Type of operation (determinant, inverse, multiply, eigenvalues)
        matrix_data: First matrix as JSON string
        matrix2_data: Second matrix as JSON string (for operations requiring two matrices)
    
    Returns:
        Result of matrix operation
    """
    try:
        # Parse first matrix
        matrix1 = json.loads(matrix_data)
        A = np.array(matrix1)
        
        if operation == "determinant":
            det = np.linalg.det(A)
            return f"Determinant: {det}"
            
        elif operation == "inverse":
            inv = np.linalg.inv(A)
            return f"Matrix inverse:\n{inv}"
            
        elif operation == "eigenvalues":
            eigenvals = np.linalg.eigvals(A)
            return f"Eigenvalues: {eigenvals}"
            
        elif operation == "multiply" and matrix2_data:
            matrix2 = json.loads(matrix2_data)
            B = np.array(matrix2)
            result = np.dot(A, B)
            return f"Matrix multiplication result:\n{result}"
            
        else:
            return f"Operation '{operation}' not supported or missing matrix data"
            
    except Exception as e:
        return f"Matrix operation error: {str(e)}"

@tool("statistics_calculator")
def statistics_calculator(data: str, operations: str = "all") -> str:
    """
    Calculate statistical measures for datasets.
    
    Args:
        data: Comma-separated numerical data
        operations: Statistics to calculate (mean, median, std, var, all)
    
    Returns:
        Calculated statistical measures
    """
    try:
        # Parse data
        numbers = [float(x.strip()) for x in data.split(',')]
        dataset = np.array(numbers)
        
        results = []
        
        if operations == "all" or "mean" in operations:
            results.append(f"Mean: {np.mean(dataset):.4f}")
            
        if operations == "all" or "median" in operations:
            results.append(f"Median: {np.median(dataset):.4f}")
            
        if operations == "all" or "std" in operations:
            results.append(f"Standard Deviation: {np.std(dataset):.4f}")
            
        if operations == "all" or "var" in operations:
            results.append(f"Variance: {np.var(dataset):.4f}")
            
        if operations == "all" or "min" in operations:
            results.append(f"Minimum: {np.min(dataset):.4f}")
            
        if operations == "all" or "max" in operations:
            results.append(f"Maximum: {np.max(dataset):.4f}")
        
        return "Statistical analysis: " + ", ".join(results)
        
    except Exception as e:
        return f"Statistics calculation error: {str(e)}"

@tool("expression_parser")
def expression_parser(text: str) -> str:
    """
    Parse mathematical expressions from natural language text.
    
    Args:
        text: Natural language containing mathematical expressions
    
    Returns:
        Extracted and parsed mathematical expressions
    """
    try:
        # Find mathematical patterns
        patterns = {
            'equations': re.findall(r'[a-zA-Z0-9\+\-\*\/\^\(\)\s]+=\s*[a-zA-Z0-9\+\-\*\/\^\(\)\s]+', text),
            'functions': re.findall(r'f\([a-zA-Z]\)\s*=\s*[a-zA-Z0-9\+\-\*\/\^\(\)\s]+', text),
            'derivatives': re.findall(r'd[a-zA-Z]?\/d[a-zA-Z]', text),
            'integrals': re.findall(r'∫|integral', text),
            'numbers': re.findall(r'-?\d+\.?\d*', text)
        }
        
        result = "Parsed mathematical elements:\n"
        for pattern_type, matches in patterns.items():
            if matches:
                result += f"- {pattern_type.capitalize()}: {matches}\n"
        
        return result if any(patterns.values()) else "No mathematical expressions found"
        
    except Exception as e:
        return f"Expression parsing error: {str(e)}"

# Tool registry for easy access
MATHEMATICAL_TOOLS = {
    'symbolic_math_calculator': symbolic_math_calculator,
    'numerical_calculator': numerical_calculator,
    'geometry_calculator': geometry_calculator,
    'equation_solver': equation_solver,
    'calculus_operations': calculus_operations,
    'matrix_operations': matrix_operations,
    'statistics_calculator': statistics_calculator,
    'expression_parser': expression_parser
}

def get_available_tools() -> List[str]:
    """Get list of available mathematical tools."""
    return list(MATHEMATICAL_TOOLS.keys())

def get_tool_description(tool_name: str) -> str:
    """Get description of a specific tool."""
    tool_func = MATHEMATICAL_TOOLS.get(tool_name)
    if tool_func:
        return tool_func.__doc__ or f"Tool: {tool_name}"
    return f"Tool '{tool_name}' not found"

if __name__ == "__main__":
    print("🛠️ Enhanced Mathematical Tools for CrewAI Agents")
    print("=" * 50)
    
    print(f"Available tools: {len(MATHEMATICAL_TOOLS)}")
    for tool_name in get_available_tools():
        print(f"  - {tool_name}")
    
    print(f"\n🧪 Testing tools...")
    
    # Test symbolic calculator
    print(f"\nTesting symbolic_math_calculator:")
    result = symbolic_math_calculator("x**2 + 3*x + 2", "derivative", "x")
    print(f"  Result: {result}")
    
    # Test numerical calculator
    print(f"\nTesting numerical_calculator:")
    result = numerical_calculator("2 + 3 * 4")
    print(f"  Result: {result}")
    
    print(f"\n✅ Tools testing completed!")
