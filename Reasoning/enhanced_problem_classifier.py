#!/usr/bin/env python3
"""
🧮 ENHANCED MATHEMATICAL PROBLEM CLASSIFIER
==========================================

Advanced classifier that can handle all types of mathematical problems
with high precision and detailed categorization.
"""

import openai
import os
import re
from typing import Dict, List, Any
from dotenv import load_dotenv

# Load .env from the parent directory
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

class EnhancedProblemClassifier:
    """
    Advanced mathematical problem classifier with detailed categorization.
    """
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.api_base = "https://openrouter.ai/api/v1"
        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.api_base)
        
        # Comprehensive problem categories
        self.categories = {
            # Algebra
            "linear_equations": ["linear equation", "solve for x", "first degree"],
            "quadratic_equations": ["quadratic", "x²", "x^2", "parabola", "discriminant"],
            "polynomial_equations": ["polynomial", "cubic", "x³", "x^3", "degree"],
            "system_of_equations": ["system", "simultaneous", "multiple equations"],
            "inequalities": ["inequality", "<", ">", "≤", "≥", "less than", "greater than"],
            "exponential_logarithmic": ["exponential", "logarithm", "log", "ln", "e^", "exp"],
            
            # Calculus
            "derivatives": ["derivative", "differentiate", "d/dx", "rate of change", "slope"],
            "integrals": ["integral", "integrate", "∫", "antiderivative", "area under curve"],
            "limits": ["limit", "approaching", "tends to", "lim"],
            "series_sequences": ["series", "sequence", "sum", "convergence", "divergence"],
            
            # Geometry
            "plane_geometry": ["triangle", "circle", "rectangle", "polygon", "area", "perimeter"],
            "coordinate_geometry": ["coordinate", "distance", "midpoint", "slope", "line equation"],
            "solid_geometry": ["volume", "surface area", "sphere", "cylinder", "cone", "prism"],
            "trigonometry": ["sin", "cos", "tan", "trigonometric", "angle", "triangle"],
            
            # Statistics & Probability
            "statistics": ["mean", "median", "mode", "standard deviation", "variance", "average"],
            "probability": ["probability", "chance", "odds", "random", "outcome", "event"],
            "combinatorics": ["combination", "permutation", "factorial", "arrangement"],
            
            # Number Theory
            "number_theory": ["prime", "factor", "divisible", "gcd", "lcm", "modular"],
            "sequences": ["arithmetic", "geometric", "fibonacci", "sequence"],
            
            # Applied Mathematics
            "word_problems": ["word problem", "real world", "application", "scenario"],
            "optimization": ["maximize", "minimize", "optimal", "best", "maximum", "minimum"],
            "physics_problems": ["velocity", "acceleration", "force", "motion", "physics"],
            "finance_problems": ["interest", "compound", "investment", "loan", "percentage"],
            
            # Matrix & Linear Algebra
            "matrix_operations": ["matrix", "determinant", "inverse", "eigenvalue", "vector"],
            "linear_algebra": ["linear transformation", "vector space", "basis", "dimension"],
            
            # Discrete Mathematics
            "graph_theory": ["graph", "vertex", "edge", "tree", "network"],
            "logic": ["logic", "boolean", "truth table", "proof", "theorem"],
            
            # Basic Operations
            "arithmetic": ["addition", "subtraction", "multiplication", "division", "basic"],
            "fraction_operations": ["fraction", "decimal", "percentage", "ratio", "proportion"]
        }
        
        print("🔍 Enhanced Problem Classifier initialized with 25+ categories.")
    
    def classify_detailed(self, problem: str) -> Dict[str, Any]:
        """
        Performs detailed classification with confidence scores and subcategories.
        
        Args:
            problem: The math problem text to classify.
            
        Returns:
            Dictionary with detailed classification information.
        """
        print(f"🔍 Analyzing problem: {problem[:60]}...")
        
        # First, get AI-based classification
        ai_classification = self._get_ai_classification(problem)
        
        # Then, use pattern matching for validation
        pattern_classification = self._pattern_based_classification(problem)
        
        # Determine difficulty level
        difficulty = self._assess_difficulty(problem)
        
        # Extract mathematical concepts
        concepts = self._extract_concepts(problem)
        
        # Determine required tools/methods
        tools_needed = self._determine_tools(problem, ai_classification)
        
        result = {
            "primary_category": ai_classification.get("category", "other"),
            "subcategory": ai_classification.get("subcategory", "general"),
            "confidence": ai_classification.get("confidence", 0.5),
            "difficulty_level": difficulty,
            "mathematical_concepts": concepts,
            "tools_needed": tools_needed,
            "pattern_matches": pattern_classification,
            "problem_characteristics": self._analyze_characteristics(problem),
            "estimated_solution_steps": self._estimate_solution_complexity(problem)
        }
        
        print(f"✅ Classification complete: {result['primary_category']} ({result['confidence']:.2f} confidence)")
        return result
    
    def _get_ai_classification(self, problem: str) -> Dict[str, Any]:
        """Get AI-based classification with detailed analysis."""
        
        prompt = f"""
        Analyze this mathematical problem and provide detailed classification:

        Problem: "{problem}"

        Classify into these main categories:
        1. algebra (linear_equations, quadratic_equations, polynomial_equations, system_of_equations, inequalities, exponential_logarithmic)
        2. calculus (derivatives, integrals, limits, series_sequences)
        3. geometry (plane_geometry, coordinate_geometry, solid_geometry, trigonometry)
        4. statistics_probability (statistics, probability, combinatorics)
        5. applied_math (word_problems, optimization, physics_problems, finance_problems)
        6. discrete_math (number_theory, sequences, graph_theory, logic)
        7. linear_algebra (matrix_operations, linear_algebra)
        8. arithmetic (basic_arithmetic, fraction_operations)

        Return ONLY a JSON object with:
        {{
            "category": "main_category",
            "subcategory": "specific_subcategory",
            "confidence": 0.95,
            "reasoning": "brief explanation"
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model="google/gemini-2.0-flash-001",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Extract JSON from response
            import json
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            else:
                return {"category": "other", "subcategory": "general", "confidence": 0.3}
                
        except Exception as e:
            print(f"❌ AI classification error: {e}")
            return {"category": "other", "subcategory": "general", "confidence": 0.1}
    
    def _pattern_based_classification(self, problem: str) -> List[str]:
        """Use pattern matching to identify problem types."""
        problem_lower = problem.lower()
        matches = []
        
        for category, keywords in self.categories.items():
            for keyword in keywords:
                if keyword.lower() in problem_lower:
                    matches.append(category)
                    break
        
        return matches
    
    def _assess_difficulty(self, problem: str) -> str:
        """Assess the difficulty level of the problem."""
        problem_lower = problem.lower()
        
        # Advanced concepts indicate higher difficulty
        advanced_indicators = [
            "integral", "derivative", "limit", "series", "matrix", "eigenvalue",
            "differential", "complex", "multivariable", "partial", "optimization"
        ]
        
        intermediate_indicators = [
            "quadratic", "system", "trigonometric", "logarithm", "inequality",
            "polynomial", "function", "graph", "coordinate"
        ]
        
        if any(indicator in problem_lower for indicator in advanced_indicators):
            return "advanced"
        elif any(indicator in problem_lower for indicator in intermediate_indicators):
            return "intermediate"
        else:
            return "basic"
    
    def _extract_concepts(self, problem: str) -> List[str]:
        """Extract key mathematical concepts from the problem."""
        concepts = []
        problem_lower = problem.lower()
        
        concept_patterns = {
            "variables": r'[a-z]\s*[=\+\-\*\/]',
            "equations": r'=',
            "functions": r'f\(.*\)',
            "derivatives": r"d[xy]|derivative|d/d",
            "integrals": r"integral|∫",
            "geometry": r"triangle|circle|area|volume|angle",
            "probability": r"probability|chance|random",
            "statistics": r"mean|median|average|standard"
        }
        
        for concept, pattern in concept_patterns.items():
            if re.search(pattern, problem_lower):
                concepts.append(concept)
        
        return concepts
    
    def _determine_tools(self, problem: str, classification: Dict) -> List[str]:
        """Determine what tools/methods are needed to solve the problem."""
        tools = []
        
        category = classification.get("category", "")
        problem_lower = problem.lower()
        
        # Basic tools everyone needs
        tools.append("symbolic_calculator")
        
        # Category-specific tools
        if "calculus" in category:
            tools.extend(["derivative_calculator", "integral_calculator"])
        
        if "geometry" in category:
            tools.extend(["geometry_formulas", "coordinate_calculator"])
        
        if "matrix" in category or "linear_algebra" in category:
            tools.extend(["matrix_calculator", "linear_solver"])
        
        if "graph" in problem_lower or "plot" in problem_lower:
            tools.append("graphing_tool")
        
        if any(word in problem_lower for word in ["solve", "equation", "system"]):
            tools.append("equation_solver")
        
        if "optimize" in problem_lower or "maximum" in problem_lower or "minimum" in problem_lower:
            tools.append("optimization_solver")
        
        return list(set(tools))
    
    def _analyze_characteristics(self, problem: str) -> Dict[str, bool]:
        """Analyze characteristics of the problem."""
        problem_lower = problem.lower()
        
        return {
            "has_word_problem": any(word in problem_lower for word in ["if", "when", "find", "calculate", "determine"]),
            "has_multiple_steps": len(problem.split('.')) > 2 or len(problem.split(',')) > 3,
            "requires_proof": any(word in problem_lower for word in ["prove", "show", "demonstrate"]),
            "has_real_world_context": any(word in problem_lower for word in ["company", "person", "car", "house", "money"]),
            "has_variables": bool(re.search(r'[a-z]\s*[=\+\-\*\/]', problem_lower)),
            "has_numerical_computation": bool(re.search(r'\d+', problem)),
            "requires_graphing": any(word in problem_lower for word in ["graph", "plot", "draw", "sketch"])
        }
    
    def _estimate_solution_complexity(self, problem: str) -> int:
        """Estimate the number of steps needed to solve the problem."""
        base_steps = 1
        
        # Add steps based on problem characteristics
        if "system" in problem.lower():
            base_steps += 3
        if "quadratic" in problem.lower() or "x²" in problem or "x^2" in problem:
            base_steps += 2
        if any(word in problem.lower() for word in ["derivative", "integral"]):
            base_steps += 3
        if "word problem" in problem.lower() or len(problem.split()) > 20:
            base_steps += 2
        
        return min(base_steps, 10)  # Cap at 10 steps

    def classify(self, problem: str) -> str:
        """
        Simple classification method for backward compatibility.
        Returns just the primary category.
        """
        detailed = self.classify_detailed(problem)
        return detailed["primary_category"]

# Example usage and testing
if __name__ == "__main__":
    classifier = EnhancedProblemClassifier()
    
    # Test problems across different categories
    test_problems = [
        "Solve for x: 2x + 5 = 15",
        "Find the derivative of f(x) = x² + 3x + 2",
        "A rectangle has length 8 cm and width 5 cm. What is its area?",
        "If a train travels 60 mph for 2 hours, how far does it go?",
        "Solve the system: x + y = 10, x - y = 2",
        "Find the integral of 2x dx from 0 to 5",
        "What is the probability of rolling a 6 on a fair dice?",
        "Simplify the matrix multiplication of [[1,2],[3,4]] and [[5,6],[7,8]]"
    ]
    
    print("🎯 TESTING ENHANCED CLASSIFIER")
    print("=" * 60)
    
    for i, problem in enumerate(test_problems, 1):
        print(f"\n🔢 PROBLEM {i}: {problem}")
        print("-" * 40)
        
        result = classifier.classify_detailed(problem)
        print(f"Category: {result['primary_category']}")
        print(f"Subcategory: {result['subcategory']}")
        print(f"Difficulty: {result['difficulty_level']}")
        print(f"Tools needed: {', '.join(result['tools_needed'])}")
        print(f"Estimated steps: {result['estimated_solution_steps']}")
