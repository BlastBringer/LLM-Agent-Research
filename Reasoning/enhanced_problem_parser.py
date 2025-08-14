import openai
import json
import os
import re
from typing import Dict, Any, List, Optional, Union
from dotenv import load_dotenv
import random

# Load .env from the parent directory
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

class EnhancedProblemParser:
    """
    Enhanced mathematical problem parser with advanced prompt engineering,
    multiple parsing strategies, and comprehensive error handling.
    
    Features:
    - Multi-format problem support (MATH, AMPS, word problems, etc.)
    - Adaptive parsing strategies based on problem type
    - Comprehensive variable and equation extraction
    - LaTeX and plain text handling
    - Validation and error recovery
    - Template-based parsing for consistency
    """
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.api_base = "https://openrouter.ai/api/v1"
        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.api_base)
        
        # Problem type patterns for initial classification
        self.problem_patterns = {
            "system_of_equations": [
                r"system of.*equations?", r"solve.*equations?.*simultaneously",
                r"find.*x.*and.*y", r"two.*equations?", r"three.*equations?"
            ],
            "quadratic": [
                r"x\^?2", r"quadratic", r"parabola", r"complete.*square",
                r"solve.*equation.*x\^?2"
            ],
            "calculus": [
                r"derivative", r"integral", r"limit", r"d/dx", r"∫", r"∑",
                r"maximize", r"minimize", r"rate.*change", r"tangent.*line"
            ],
            "geometry": [
                r"triangle", r"circle", r"rectangle", r"angle", r"area",
                r"perimeter", r"volume", r"surface.*area", r"coordinate"
            ],
            "algebra": [
                r"factor", r"expand", r"simplify", r"polynomial", r"rational"
            ],
            "arithmetic": [
                r"calculate", r"compute", r"evaluate", r"^[0-9+\-*/().\s]+$"
            ],
            "word_problem": [
                r"company", r"sells", r"costs", r"total", r"revenue", r"profit",
                r"age", r"speed", r"distance", r"time", r"rate"
            ]
        }

        print("🚀 Problem Parser initialized.")
    
    def classify_problem_type(self, problem: str) -> str:
        """Quick classification to choose parsing strategy."""
        problem_lower = problem.lower()
        
        # Check for LaTeX indicators
        if any(latex in problem for latex in ["\\[", "\\(", "$", "\\frac", "\\sum", "\\int"]):
            if any(re.search(pattern, problem_lower) for pattern in self.problem_patterns["calculus"]):
                return "calculus_latex"
            elif any(re.search(pattern, problem_lower) for pattern in self.problem_patterns["system_of_equations"]):
                return "system_latex"
            else:
                return "math_latex"
        
        # Check for specific types
        for problem_type, patterns in self.problem_patterns.items():
            if any(re.search(pattern, problem_lower) for pattern in patterns):
                return problem_type
        
        return "general_math"
    
    def create_enhanced_parser_prompt(self, problem: str, problem_type: str) -> str:
        
        base_instructions = """You are an expert mathematical problem parser. Your job is to convert math problems into perfectly structured JSON format.

CRITICAL RULES:
1. Extract ALL mathematical information accurately
2. Use Python-compatible variable names and expressions  
3. Return ONLY valid JSON, no explanations
4. Handle LaTeX properly (convert to readable format when needed)
5. Identify ALL variables, equations, constraints, and objectives"""

        # Choose examples based on problem type
        if problem_type in ["system_of_equations", "system_latex"]:
            examples = self._get_system_examples()
        elif problem_type in ["calculus", "calculus_latex"]:
            examples = self._get_calculus_examples()
        elif problem_type == "word_problem":
            examples = self._get_word_problem_examples()
        elif problem_type in ["quadratic", "algebra"]:
            examples = self._get_algebra_examples()
        elif problem_type == "arithmetic":
            examples = self._get_arithmetic_examples()
        else:
            examples = self._get_general_examples()
        
        return f"""{base_instructions}

{examples}

Now parse this problem:

Problem: "{problem}"

JSON Output:"""

    def _get_system_examples(self) -> str:
        return """
EXAMPLES for Systems of Equations:

Problem: "A company sells notebooks and pens. Each notebook costs $50 and each pen costs $20. They sold 120 items total and made $3,800 revenue. How many notebooks were sold?"

JSON Output:
{
  "problem_type": "system_of_linear_equations",
  "variables": {
    "n": {"description": "number of notebooks", "domain": "non_negative_integer"},
    "p": {"description": "number of pens", "domain": "non_negative_integer"}
  },
  "equations": [
    "n + p = 120",
    "50 * n + 20 * p = 3800"
  ],
  "constraints": ["n >= 0", "p >= 0"],
  "objective": "find n",
  "context": "word_problem",
  "solution_method": "substitution_or_elimination"
}

Problem: "Solve the system: x + y = 10, x - y = 2"

JSON Output:
{
  "problem_type": "system_of_linear_equations", 
  "variables": {
    "x": {"description": "first variable", "domain": "real"},
    "y": {"description": "second variable", "domain": "real"}
  },
  "equations": ["x + y = 10", "x - y = 2"],
  "constraints": [],
  "objective": "solve for x and y",
  "context": "pure_math",
  "solution_method": "elimination"
}"""

    def _get_calculus_examples(self) -> str:
        return """
EXAMPLES for Calculus:

Problem: "Find the derivative of f(x) = x^3 + 2x^2 - 5x + 1"

JSON Output:
{
  "problem_type": "derivative",
  "function": "x**3 + 2*x**2 - 5*x + 1",
  "variable": "x",
  "operation": "differentiation",
  "objective": "find f'(x)",
  "context": "pure_math",
  "method": "power_rule"
}

Problem: "Evaluate ∫(2x + 3) dx from 0 to 5"

JSON Output:
{
  "problem_type": "definite_integral",
  "integrand": "2*x + 3",
  "variable": "x", 
  "limits": {"lower": 0, "upper": 5},
  "operation": "integration",
  "objective": "evaluate definite integral",
  "context": "pure_math",
  "method": "fundamental_theorem"
}"""

    def _get_word_problem_examples(self) -> str:
        return """
EXAMPLES for Word Problems:

Problem: "Sarah is 3 times as old as her brother. The sum of their ages is 24. How old is Sarah?"

JSON Output:
{
  "problem_type": "linear_equation_word_problem",
  "variables": {
    "s": {"description": "Sarah's age", "domain": "positive_integer"},
    "b": {"description": "brother's age", "domain": "positive_integer"}
  },
  "relationships": ["s = 3 * b"],
  "equations": ["s + b = 24"],
  "constraints": ["s > 0", "b > 0"],
  "objective": "find s",
  "context": "age_problem",
  "substitution_target": "s"
}

Problem: "A train travels 240 miles in 4 hours. What is its average speed?"

JSON Output:
{
  "problem_type": "rate_calculation",
  "given_values": {
    "distance": {"value": 240, "unit": "miles"},
    "time": {"value": 4, "unit": "hours"}
  },
  "formula": "speed = distance / time",
  "objective": "find average speed",
  "context": "motion_problem",
  "calculation": "240 / 4"
}"""

    def _get_algebra_examples(self) -> str:
        return """
EXAMPLES for Algebra:

Problem: "Factor x^2 + 5x + 6"

JSON Output:
{
  "problem_type": "factoring",
  "expression": "x**2 + 5*x + 6",
  "variable": "x",
  "operation": "factorization",
  "objective": "factor completely",
  "context": "polynomial_algebra",
  "method": "quadratic_factoring"
}

Problem: "Solve x^2 - 7x + 12 = 0"

JSON Output:
{
  "problem_type": "quadratic_equation",
  "equation": "x**2 - 7*x + 12 = 0",
  "variable": "x",
  "coefficients": {"a": 1, "b": -7, "c": 12},
  "objective": "solve for x",
  "context": "pure_math",
  "solution_methods": ["factoring", "quadratic_formula"]
}"""

    def _get_arithmetic_examples(self) -> str:
        return """
EXAMPLES for Arithmetic:

Problem: "Calculate 25 * 4 + 18 / 3"

JSON Output:
{
  "problem_type": "arithmetic_expression",
  "expression": "25 * 4 + 18 / 3",
  "operation": "evaluation",
  "objective": "compute numerical result",
  "context": "pure_calculation",
  "order_of_operations": true
}

Problem: "What is 15% of 240?"

JSON Output:
{
  "problem_type": "percentage_calculation", 
  "base_value": 240,
  "percentage": 15,
  "calculation": "0.15 * 240",
  "objective": "find percentage value",
  "context": "percentage_problem"
}"""

    def _get_general_examples(self) -> str:
        return """
EXAMPLES for General Math:

Problem: "Find all real values of x which satisfy (1/(x+1)) + (6/(x+5)) >= 1"

JSON Output:
{
  "problem_type": "inequality",
  "inequality": "(1/(x+1)) + (6/(x+5)) >= 1",
  "variable": "x",
  "domain_restrictions": ["x != -1", "x != -5"],
  "objective": "find solution set",
  "context": "rational_inequality",
  "method": "sign_analysis"
}

Problem: "Complete the square for 2x^2 + 8x + 3"

JSON Output:
{
  "problem_type": "complete_the_square",
  "expression": "2*x**2 + 8*x + 3",
  "variable": "x",
  "coefficients": {"a": 2, "b": 8, "c": 3},
  "objective": "express in vertex form",
  "context": "algebraic_manipulation",
  "method": "completing_square"
}"""

    def parse(self, problem: str) -> Dict[str, Any]:
        """
        Enhanced parsing with multiple strategies and error recovery.
        """
        print(f"🚀 Enhanced parsing: {problem[:60]}...")
        
        # Step 1: Classify problem type for strategy selection
        problem_type = self.classify_problem_type(problem)
        print(f"Detected type: {problem_type}")
        
        # Step 2: Try primary parsing strategy
        try:
            result = self._attempt_parse(problem, problem_type)
            if self._validate_parse_result(result):
                print("Primary parsing successful")
                return self._enrich_parse_result(result, problem)
        except Exception as e:
            print(f"Primary parse failed: {str(e)}")
        
        # Step 3: Try fallback strategies
        for fallback_type in ["general_math", "word_problem", "arithmetic"]:
            if fallback_type != problem_type:
                try:
                    print(f"Trying fallback: {fallback_type}")
                    result = self._attempt_parse(problem, fallback_type)
                    if self._validate_parse_result(result):
                        print(f"Fallback parsing successful with {fallback_type}")
                        return self._enrich_parse_result(result, problem)
                except Exception as e:
                    continue
        
        # Step 4: Final emergency parsing
        print(" Using emergency parsing")
        return self._emergency_parse(problem)
    
    def _attempt_parse(self, problem: str, problem_type: str) -> Dict[str, Any]:
        """Attempt parsing with a specific strategy."""
        prompt = self.create_enhanced_parser_prompt(problem, problem_type)
        
        response = self.client.chat.completions.create(
            model="google/gemini-2.0-flash-001",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.05,  # Very low for consistency
            max_tokens=2048,   # More tokens for complex problems
            top_p=0.9
        )
        
        raw_output = response.choices[0].message.content.strip()
        return self._clean_and_parse_json(raw_output)
    
    def _clean_and_parse_json(self, raw_output: str) -> Dict[str, Any]:
        """Enhanced JSON cleaning and parsing."""
        # Remove markdown formatting
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_output.strip())
        
        # Try to find JSON block
        json_patterns = [
            r'\{.*\}',  # Standard JSON
            r'\[.*\]',  # Array JSON (less common but possible)
        ]
        
        for pattern in json_patterns:
            match = re.search(pattern, cleaned, re.DOTALL)
            if match:
                json_str = match.group(0)
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    continue
        
        # If direct parsing fails, try to fix common issues
        return self._attempt_json_repair(cleaned)
    
    def _attempt_json_repair(self, json_str: str) -> Dict[str, Any]:
        """Attempt to repair malformed JSON."""
        repairs = [
            # Fix single quotes to double quotes
            (r"'([^']*)':", r'"\1":'),
            # Fix trailing commas
            (r',(\s*[}\]])', r'\1'),
            # Fix missing quotes on keys
            (r'(\w+)(\s*:)', r'"\1"\2'),
        ]
        
        repaired = json_str
        for pattern, replacement in repairs:
            repaired = re.sub(pattern, replacement, repaired)
        
        try:
            return json.loads(repaired)
        except json.JSONDecodeError as e:
            raise Exception(f"JSON repair failed: {e}")
    
    def _validate_parse_result(self, result: Dict[str, Any]) -> bool:
        """Validate that the parse result is reasonable."""
        if not isinstance(result, dict):
            return False
        
        # Must have problem_type
        if "problem_type" not in result:
            return False
        
        # Check for reasonable structure based on problem type
        problem_type = result.get("problem_type", "")
        
        if "equation" in problem_type and "equations" not in result and "equation" not in result:
            return False
        
        if "expression" in problem_type and "expression" not in result:
            return False
        
        return True
    
    def _enrich_parse_result(self, result: Dict[str, Any], original_problem: str) -> Dict[str, Any]:
        """Add metadata and computed fields to the parse result."""
        result["metadata"] = {
            "original_problem": original_problem,
            "problem_length": len(original_problem),
            "has_latex": any(latex in original_problem for latex in ["\\", "$", "^", "_"]),
            "detected_numbers": re.findall(r'-?\d+\.?\d*', original_problem),
            "parsing_strategy": "enhanced_parser_v1"
        }
        
        # Add complexity estimation
        complexity_score = self._estimate_complexity(result, original_problem)
        result["metadata"]["complexity_score"] = complexity_score
        
        return result
    
    def _estimate_complexity(self, result: Dict[str, Any], problem: str) -> int:
        """Estimate problem complexity on a scale of 1-10."""
        score = 1
        
        # Base complexity from problem type
        complexity_map = {
            "arithmetic": 1,
            "linear": 2,
            "quadratic": 3,
            "system": 4,
            "calculus": 6,
            "inequality": 5,
            "geometry": 4
        }
        
        problem_type = result.get("problem_type", "").lower()
        for key, value in complexity_map.items():
            if key in problem_type:
                score = max(score, value)
        
        # Complexity modifiers
        if "variables" in result:
            score += min(len(result["variables"]), 3)
        
        if "equations" in result:
            score += min(len(result["equations"]), 3)
        
        if any(latex in problem for latex in ["\\frac", "\\int", "\\sum", "\\prod"]):
            score += 2
        
        return min(score, 10)
    
    def _emergency_parse(self, problem: str) -> Dict[str, Any]:
        """Emergency parsing when all strategies fail."""
        return {
            "problem_type": "parsing_failed",
            "original_problem": problem,
            "error": "Could not parse with any strategy",
            "extracted_numbers": re.findall(r'-?\d+\.?\d*', problem),
            "extracted_variables": list(set(re.findall(r'\b[a-zA-Z]\b', problem))),
            "emergency_parse": True,
            "suggested_action": "manual_review_required"
        }
    
    def batch_parse(self, problems: List[str], show_progress: bool = True) -> List[Dict[str, Any]]:
        """Parse multiple problems efficiently."""
        results = []
        total = len(problems)
        
        for i, problem in enumerate(problems):
            if show_progress and i % 10 == 0:
                print(f"📊 Progress: {i}/{total} ({i/total*100:.1f}%)")
            
            try:
                result = self.parse(problem)
                results.append(result)
            except Exception as e:
                results.append({
                    "problem_type": "batch_parse_error",
                    "original_problem": problem,
                    "error": str(e),
                    "batch_index": i
                })
        
        return results
    
    def analyze_parsing_quality(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the quality of parsing results."""
        total = len(results)
        successful = sum(1 for r in results if r.get("problem_type") != "parsing_failed")
        emergency = sum(1 for r in results if r.get("emergency_parse", False))
        
        problem_types = {}
        for result in results:
            ptype = result.get("problem_type", "unknown")
            problem_types[ptype] = problem_types.get(ptype, 0) + 1
        
        return {
            "total_problems": total,
            "successful_parses": successful,
            "success_rate": successful / total if total > 0 else 0,
            "emergency_parses": emergency,
            "problem_type_distribution": problem_types,
            "average_complexity": sum(r.get("metadata", {}).get("complexity_score", 0) for r in results) / total if total > 0 else 0
        }


# Example usage and testing
if __name__ == "__main__":
    parser = EnhancedProblemParser()
    
    # Test with diverse problem types
    test_problems = [
        # System of equations
        "A company sells notebooks and pens. Each notebook costs ₹50 and each pen costs ₹20. On a certain day, the company sold a total of 120 items and made ₹3,800 in revenue. How many notebooks were sold?",
        
        # Calculus
        "Find the derivative of f(x) = 3x^4 - 2x^3 + x^2 - 5x + 7",
        
        # LaTeX problem
        "Find all real values of $x$ which satisfy \\[\\frac{1}{x + 1} + \\frac{6}{x + 5} \\ge 1.\\]",
        
        # Quadratic
        "Solve the equation 2x^2 + 7x - 15 = 0",
        
        # Arithmetic
        "Calculate 25 * 4 + 18 / 3 - 12",
        
        # Geometry  
        "Find the area of a triangle with base 12 cm and height 8 cm",
        
        # Complex algebra
        "Complete the square for the expression 3x^2 + 12x + 7"
    ]
    
    print("🧪 TESTING ENHANCED PARSER")
    print("=" * 60)
    
    results = []
    for i, problem in enumerate(test_problems, 1):
        print(f"\n🧪 TEST {i}: {problem[:50]}...")
        print("-" * 50)
        
        try:
            result = parser.parse(problem)
            results.append(result)
            
            print("✅ Parse successful!")
            print(f"   Type: {result.get('problem_type')}")
            print(f"   Complexity: {result.get('metadata', {}).get('complexity_score', 'N/A')}")
            
            # Show key extracted information
            if 'variables' in result:
                print(f"   Variables: {list(result['variables'].keys())}")
            if 'equations' in result:
                print(f"   Equations: {len(result['equations'])} found")
            if 'expression' in result:
                print(f"   Expression: {result['expression']}")
                
        except Exception as e:
            print(f"❌ Parse failed: {e}")
            results.append({"error": str(e), "problem": problem})
    
    # Analyze results
    print(f"\n📊 PARSING ANALYSIS")
    print("=" * 40)
    analysis = parser.analyze_parsing_quality(results)
    print(f"Success Rate: {analysis['success_rate']:.1%}")
    print(f"Average Complexity: {analysis['average_complexity']:.1f}/10")
    print(f"Problem Types: {analysis['problem_type_distribution']}")
