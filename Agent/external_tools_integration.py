#!/usr/bin/env python3
"""
🔧 EXTERNAL TOOLS INTEGRATION
===========================

Integration with external tools as shown in the diagram:
- SymPy for symbolic mathematics
- Python Sandbox for safe code execution
- Search Engine for mathematical knowledge lookup
"""

import subprocess
import tempfile
import os
import requests
import json
import sys
import sympy
from typing import Dict, List, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SymPyTool:
    """Enhanced SymPy integration tool."""
    
    def __init__(self):
        self.session_vars = {}
        self.history = []
    
    def execute_sympy(self, expression: str, operation: str = "auto") -> Dict[str, Any]:
        """
        Execute SymPy operations with enhanced capabilities.
        
        Args:
            expression: Mathematical expression
            operation: Type of operation (auto, solve, simplify, expand, etc.)
            
        Returns:
            Result dictionary with success status and result
        """
        try:
            # Auto-detect operation if not specified
            if operation == "auto":
                operation = self._detect_operation(expression)
            
            # Parse expression
            expr = sympy.sympify(expression)
            
            # Execute based on operation
            if operation == "solve":
                if "=" in expression:
                    # Equation solving
                    left, right = expression.split("=")
                    eq = sympy.Eq(sympy.sympify(left.strip()), sympy.sympify(right.strip()))
                    result = sympy.solve(eq)
                else:
                    # Expression solving (find roots)
                    result = sympy.solve(expr)
                    
            elif operation == "simplify":
                result = sympy.simplify(expr)
                
            elif operation == "expand":
                result = sympy.expand(expr)
                
            elif operation == "factor":
                result = sympy.factor(expr)
                
            elif operation == "derivative":
                # Auto-detect variable or use x
                variables = expr.free_symbols
                var = list(variables)[0] if variables else sympy.Symbol('x')
                result = sympy.diff(expr, var)
                
            elif operation == "integral":
                variables = expr.free_symbols
                var = list(variables)[0] if variables else sympy.Symbol('x')
                result = sympy.integrate(expr, var)
                
            elif operation == "limit":
                variables = expr.free_symbols
                var = list(variables)[0] if variables else sympy.Symbol('x')
                result = sympy.limit(expr, var, 0)  # Default approach 0
                
            else:
                result = expr  # Return as-is for unknown operations
            
            # Store in history
            self.history.append({
                'expression': expression,
                'operation': operation,
                'result': str(result)
            })
            
            return {
                'success': True,
                'result': str(result),
                'operation': operation,
                'latex': sympy.latex(result) if hasattr(result, '__iter__') is False else str(result),
                'numeric_value': float(result) if result.is_number else None
            }
            
        except Exception as e:
            logger.error(f"SymPy execution error: {e}")
            return {
                'success': False,
                'result': f"SymPy error: {str(e)}",
                'operation': operation,
                'error': str(e)
            }
    
    def _detect_operation(self, expression: str) -> str:
        """Auto-detect the intended mathematical operation."""
        expr_lower = expression.lower()
        
        if "=" in expression:
            return "solve"
        elif "derivative" in expr_lower or "diff" in expr_lower:
            return "derivative"
        elif "integral" in expr_lower or "integrate" in expr_lower:
            return "integral"
        elif "simplify" in expr_lower:
            return "simplify"
        elif "expand" in expr_lower:
            return "expand"
        elif "factor" in expr_lower:
            return "factor"
        else:
            return "simplify"  # Default operation

class PythonSandbox:
    """Safe Python code execution sandbox."""
    
    def __init__(self):
        self.allowed_imports = [
            'math', 'numpy', 'sympy', 'matplotlib.pyplot',
            'scipy', 'pandas', 'statistics'
        ]
        self.restricted_functions = [
            'exec', 'eval', 'open', 'file', 'input', 'raw_input',
            '__import__', 'reload', 'compile'
        ]
    
    def execute_python_code(self, code: str, timeout: int = 30) -> Dict[str, Any]:
        """
        Execute Python code in a safe sandbox environment.
        
        Args:
            code: Python code to execute
            timeout: Maximum execution time in seconds
            
        Returns:
            Execution result dictionary
        """
        try:
            # Security check
            if not self._is_code_safe(code):
                return {
                    'success': False,
                    'result': 'Code contains restricted functions or imports',
                    'error': 'Security violation'
                }
            
            # Create temporary file for code execution
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                # Add safe imports and setup
                safe_code = self._prepare_safe_code(code)
                f.write(safe_code)
                temp_file = f.name
            
            try:
                # Execute code with timeout
                result = subprocess.run(
                    [sys.executable, temp_file],
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
                
                if result.returncode == 0:
                    return {
                        'success': True,
                        'result': result.stdout.strip(),
                        'stderr': result.stderr.strip() if result.stderr else None
                    }
                else:
                    return {
                        'success': False,
                        'result': result.stderr.strip(),
                        'error': f"Execution failed with code {result.returncode}"
                    }
                    
            finally:
                # Clean up temporary file
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                    
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'result': 'Code execution timed out',
                'error': 'Timeout'
            }
        except Exception as e:
            logger.error(f"Sandbox execution error: {e}")
            return {
                'success': False,
                'result': f"Sandbox error: {str(e)}",
                'error': str(e)
            }
    
    def _is_code_safe(self, code: str) -> bool:
        """Check if code is safe for execution."""
        code_lower = code.lower()
        
        # Check for restricted functions
        for func in self.restricted_functions:
            if func in code_lower:
                return False
        
        # Check for file operations
        if any(op in code_lower for op in ['open(', 'file(', 'with open']):
            return False
        
        # Check for network operations
        if any(net in code_lower for net in ['urllib', 'requests', 'socket', 'http']):
            return False
        
        return True
    
    def _prepare_safe_code(self, code: str) -> str:
        """Prepare code with safe imports and error handling."""
        safe_imports = """
import math
import numpy as np
import sympy
import matplotlib.pyplot as plt
from scipy import *
import statistics
import sys

# Redirect stdout to capture output
import io
from contextlib import redirect_stdout, redirect_stderr

output_buffer = io.StringIO()
error_buffer = io.StringIO()

try:
    with redirect_stdout(output_buffer), redirect_stderr(error_buffer):
"""
        
        # Indent the user code
        indented_code = '\n'.join(['        ' + line for line in code.split('\n')])
        
        safe_code = safe_imports + indented_code + """
    
    print(output_buffer.getvalue())
    if error_buffer.getvalue():
        print("STDERR:", error_buffer.getvalue(), file=sys.stderr)
        
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
"""
        return safe_code

class SearchEngine:
    """Mathematical knowledge search engine."""
    
    def __init__(self):
        self.search_apis = {
            'wolfram': 'http://api.wolframalpha.com/v2/query',
            'wikipedia': 'https://en.wikipedia.org/api/rest_v1/page/summary/',
            'mathworld': 'https://mathworld.wolfram.com/search/'
        }
        self.cache = {}
    
    def search_mathematical_knowledge(self, query: str, source: str = "wikipedia") -> Dict[str, Any]:
        """
        Search for mathematical knowledge from various sources.
        
        Args:
            query: Search query
            source: Search source (wikipedia, mathworld, wolfram)
            
        Returns:
            Search result dictionary
        """
        try:
            # Check cache first
            cache_key = f"{source}:{query}"
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            if source == "wikipedia":
                result = self._search_wikipedia(query)
            elif source == "mathworld":
                result = self._search_mathworld(query)
            elif source == "wolfram":
                result = self._search_wolfram(query)
            else:
                result = {
                    'success': False,
                    'result': f"Unknown search source: {source}",
                    'error': 'Invalid source'
                }
            
            # Cache successful results
            if result.get('success'):
                self.cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return {
                'success': False,
                'result': f"Search error: {str(e)}",
                'error': str(e)
            }
    
    def _search_wikipedia(self, query: str) -> Dict[str, Any]:
        """Search Wikipedia for mathematical concepts."""
        try:
            # Format query for Wikipedia API
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '_')}"
            
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'success': True,
                    'result': data.get('extract', 'No extract available'),
                    'title': data.get('title', query),
                    'source': 'wikipedia',
                    'url': data.get('content_urls', {}).get('desktop', {}).get('page', '')
                }
            else:
                return {
                    'success': False,
                    'result': f"Wikipedia search failed: {response.status_code}",
                    'error': f"HTTP {response.status_code}"
                }
                
        except requests.RequestException as e:
            return {
                'success': False,
                'result': f"Wikipedia search error: {str(e)}",
                'error': str(e)
            }
    
    def _search_mathworld(self, query: str) -> Dict[str, Any]:
        """Search MathWorld for mathematical concepts."""
        # Simplified implementation - would need proper API access
        return {
            'success': True,
            'result': f"MathWorld search for '{query}' - would need API integration",
            'source': 'mathworld',
            'note': 'Mock implementation - requires API setup'
        }
    
    def _search_wolfram(self, query: str) -> Dict[str, Any]:
        """Search Wolfram Alpha for mathematical computation."""
        # Simplified implementation - would need API key
        return {
            'success': True,
            'result': f"Wolfram Alpha search for '{query}' - would need API key",
            'source': 'wolfram',
            'note': 'Mock implementation - requires API key'
        }

class ExternalToolsManager:
    """Manager for all external tools integration."""
    
    def __init__(self):
        self.sympy_tool = SymPyTool()
        self.python_sandbox = PythonSandbox()
        self.search_engine = SearchEngine()
        self.tool_stats = {
            'sympy_calls': 0,
            'sandbox_calls': 0,
            'search_calls': 0
        }
    
    def execute_with_external_tool(self, 
                                  tool_name: str, 
                                  operation: str, 
                                  data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute operation using specified external tool.
        
        Args:
            tool_name: Name of external tool (sympy, sandbox, search)
            operation: Operation to perform
            data: Data for the operation
            
        Returns:
            Execution result
        """
        try:
            if tool_name == "sympy":
                self.tool_stats['sympy_calls'] += 1
                expression = data.get('expression', '')
                op_type = data.get('operation', operation)
                return self.sympy_tool.execute_sympy(expression, op_type)
                
            elif tool_name == "sandbox":
                self.tool_stats['sandbox_calls'] += 1
                code = data.get('code', '')
                timeout = data.get('timeout', 30)
                return self.python_sandbox.execute_python_code(code, timeout)
                
            elif tool_name == "search":
                self.tool_stats['search_calls'] += 1
                query = data.get('query', '')
                source = data.get('source', 'wikipedia')
                return self.search_engine.search_mathematical_knowledge(query, source)
                
            else:
                return {
                    'success': False,
                    'result': f"Unknown external tool: {tool_name}",
                    'error': 'Invalid tool'
                }
                
        except Exception as e:
            logger.error(f"External tool execution error: {e}")
            return {
                'success': False,
                'result': f"External tool error: {str(e)}",
                'error': str(e)
            }
    
    def get_tool_statistics(self) -> Dict[str, Any]:
        """Get usage statistics for external tools."""
        total_calls = sum(self.tool_stats.values())
        return {
            'total_calls': total_calls,
            'sympy_calls': self.tool_stats['sympy_calls'],
            'sandbox_calls': self.tool_stats['sandbox_calls'],
            'search_calls': self.tool_stats['search_calls'],
            'sympy_percentage': (self.tool_stats['sympy_calls'] / total_calls * 100) if total_calls > 0 else 0,
            'sandbox_percentage': (self.tool_stats['sandbox_calls'] / total_calls * 100) if total_calls > 0 else 0,
            'search_percentage': (self.tool_stats['search_calls'] / total_calls * 100) if total_calls > 0 else 0
        }

# Global external tools manager
external_tools = ExternalToolsManager()

def execute_external_tool(tool_name: str, operation: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to execute external tools.
    
    Args:
        tool_name: Name of external tool
        operation: Operation to perform
        data: Data for operation
        
    Returns:
        Execution result
    """
    return external_tools.execute_with_external_tool(tool_name, operation, data)

def get_external_tools_manager() -> ExternalToolsManager:
    """Get the global external tools manager."""
    return external_tools

if __name__ == "__main__":
    print("🔧 External Tools Integration Testing")
    print("=" * 40)
    
    # Test SymPy tool
    print("\n🧮 Testing SymPy Tool:")
    sympy_result = execute_external_tool("sympy", "solve", {
        'expression': 'x^2 + 3*x + 2 = 0'
    })
    print(f"SymPy Result: {sympy_result}")
    
    # Test Python Sandbox
    print("\n🐍 Testing Python Sandbox:")
    sandbox_result = execute_external_tool("sandbox", "execute", {
        'code': 'import math\nresult = math.sqrt(16)\nprint(f"Square root of 16 is {result}")'
    })
    print(f"Sandbox Result: {sandbox_result}")
    
    # Test Search Engine
    print("\n🔍 Testing Search Engine:")
    search_result = execute_external_tool("search", "search", {
        'query': 'quadratic equation',
        'source': 'wikipedia'
    })
    print(f"Search Result: {search_result}")
    
    # Get statistics
    print("\n📊 Tool Usage Statistics:")
    stats = external_tools.get_tool_statistics()
    print(f"Statistics: {stats}")
    
    print("\n✅ External tools integration testing completed!")
