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
    
    def classify_detailed(self, problem: str, parsed_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Performs detailed classification with confidence scores and subcategories.
        
        Args:
            problem: The math problem text to classify.
            parsed_data: Optional parsed data from the problem parser for enhanced classification.
            
        Returns:
            Dictionary with detailed classification information.
        """
        print(f"🔍 Analyzing problem: {problem[:60]}...")
        
        if parsed_data:
            print(f"🔗 Using parsed data for enhanced classification: {parsed_data.get('problem_type', 'Unknown type')}")
        
        # First, get AI-based classification (enhanced with parser data)
        ai_classification = self._get_ai_classification(problem, parsed_data)
        
        # Then, use pattern matching for validation (enhanced with parser data)
        pattern_classification = self._pattern_based_classification(problem, parsed_data)
        
        # Determine difficulty level (enhanced with parser insights)
        difficulty = self._assess_difficulty(problem, parsed_data)
        
        # Extract mathematical concepts (enhanced with parser data)
        concepts = self._extract_concepts(problem, parsed_data)
        
        # Determine required tools/methods
        tools_needed = self._determine_tools(problem, ai_classification, parsed_data)
        
        # Enhanced classification result with parser integration
        result = {
            "primary_category": ai_classification.get("category", "other"),
            "subcategory": ai_classification.get("subcategory", "general"),
            "confidence": ai_classification.get("confidence", 0.5),
            "difficulty_level": difficulty,
            "mathematical_concepts": concepts,
            "tools_needed": tools_needed,
            "pattern_matches": pattern_classification,
            "problem_characteristics": self._analyze_characteristics(problem, parsed_data),
            "estimated_solution_steps": self._estimate_solution_complexity(problem, parsed_data),
            "parser_insights": parsed_data if parsed_data else None  # Include parser data
        }
        
        print(f"✅ Classification complete: {result['primary_category']} ({result['confidence']:.2f} confidence)")
        return result
    
    def _get_ai_classification(self, problem: str, parsed_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get AI-based classification with detailed analysis, enhanced with parser data."""
        
        # Build enhanced prompt with parser data
        parsed_info = ""
        if parsed_data:
            parsed_info = f"""
        
        Parser Analysis Results:
        - Problem Type: {parsed_data.get('problem_type', 'Unknown')}
        - Variables: {parsed_data.get('variables', [])}
        - Operations: {parsed_data.get('operations', [])}
        - Equations: {parsed_data.get('equations', [])}
        - Mathematical Elements: {parsed_data.get('mathematical_elements', {})}
        - Summary: {parsed_data.get('summary', 'No summary available')}
        """
        
        prompt = f"""
        Analyze this mathematical problem and provide detailed classification:

        Problem: "{problem}"
        {parsed_info}

        Classify into these main categories:
        1. algebra (linear_equations, quadratic_equations, polynomial_equations, system_of_equations, inequalities, exponential_logarithmic)
        2. calculus (derivatives, integrals, limits, series_sequences)
        3. geometry (plane_geometry, coordinate_geometry, solid_geometry, trigonometry)
        4. statistics_probability (statistics, probability, combinatorics)
        5. applied_math (word_problems, optimization, physics_problems, finance_problems)
        6. discrete_math (number_theory, sequences, graph_theory, logic)
        7. linear_algebra (matrix_operations, linear_algebra)
        8. arithmetic (basic_arithmetic, fraction_operations)

        Use the parser analysis to enhance your classification accuracy.

        Return ONLY a JSON object with:
        {{
            "category": "main_category",
            "subcategory": "specific_subcategory",
            "confidence": 0.95,
            "reasoning": "brief explanation including parser insights"
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
    
    def _pattern_based_classification(self, problem: str, parsed_data: Dict[str, Any] = None) -> List[str]:
        """Use pattern matching to identify problem types, enhanced with parser data."""
        problem_lower = problem.lower()
        matches = []
        
        # Use parser data to enhance pattern matching
        if parsed_data:
            # Check parser-identified problem type first
            parser_type = parsed_data.get('problem_type', '').lower()
            if parser_type:
                # Map parser types to our categories
                parser_mappings = {
                    'algebra': ['linear_equations', 'quadratic_equations', 'polynomial_equations'],
                    'calculus': ['derivatives', 'integrals', 'limits'],
                    'geometry': ['plane_geometry', 'coordinate_geometry', 'trigonometry'],
                    'arithmetic': ['arithmetic', 'fraction_operations'],
                    'trigonometry': ['trigonometry'],
                    'statistics': ['statistics'],
                    'probability': ['probability']
                }
                
                for category_group, subcategories in parser_mappings.items():
                    if category_group in parser_type:
                        matches.extend(subcategories)
            
            # Use parser-identified mathematical elements
            math_elements = parsed_data.get('mathematical_elements', {})
            if math_elements.get('derivatives'):
                matches.append('derivatives')
            if math_elements.get('integrals'):
                matches.append('integrals')
            if math_elements.get('equations'):
                matches.append('linear_equations')
        
        # Traditional pattern matching
        for category, keywords in self.categories.items():
            for keyword in keywords:
                if keyword.lower() in problem_lower:
                    matches.append(category)
                    break
        
        return list(set(matches))  # Remove duplicates
    
    def _assess_difficulty(self, problem: str, parsed_data: Dict[str, Any] = None) -> str:
        """Assess the difficulty level of the problem, enhanced with parser insights."""
        problem_lower = problem.lower()
        
        # Use parser data for enhanced difficulty assessment
        difficulty_score = 0
        
        if parsed_data:
            # Check complexity indicators from parser
            variables = parsed_data.get('variables', [])
            operations = parsed_data.get('operations', [])
            equations = parsed_data.get('equations', [])
            
            # More variables/equations = higher difficulty
            difficulty_score += len(variables) * 0.5
            difficulty_score += len(equations) * 1.0
            
            # Complex operations = higher difficulty
            complex_ops = ['derivative', 'integral', 'limit', 'matrix', 'trigonometric']
            for op in operations:
                if any(complex_op in str(op).lower() for complex_op in complex_ops):
                    difficulty_score += 2
        
        # Advanced concepts indicate higher difficulty
        advanced_indicators = [
            "integral", "derivative", "limit", "series", "matrix", "eigenvalue",
            "differential", "complex", "multivariable", "partial", "optimization"
        ]
        
        intermediate_indicators = [
            "quadratic", "system", "trigonometric", "logarithm", "inequality",
            "polynomial", "function", "graph", "coordinate"
        ]
        
        # Add to difficulty score based on indicators
        for indicator in advanced_indicators:
            if indicator in problem_lower:
                difficulty_score += 3
        
        for indicator in intermediate_indicators:
            if indicator in problem_lower:
                difficulty_score += 1
        
        # Determine final difficulty level
        if difficulty_score >= 4:
            return "advanced"
        elif difficulty_score >= 2:
            return "intermediate"
        else:
            return "basic"
    
    def _extract_concepts(self, problem: str, parsed_data: Dict[str, Any] = None) -> List[str]:
        """Extract key mathematical concepts from the problem, enhanced with parser data."""
        concepts = []
        problem_lower = problem.lower()
        
        # Enhanced concept extraction using parser data
        if parsed_data:
            # Use parser's mathematical elements
            math_elements = parsed_data.get('mathematical_elements', {})
            for element_type, elements in math_elements.items():
                if elements:  # If elements exist
                    concepts.append(element_type)
            
            # Use parser's identified variables and operations
            variables = parsed_data.get('variables', [])
            operations = parsed_data.get('operations', [])
            
            if variables:
                concepts.append("variables")
            if operations:
                concepts.extend([str(op).lower() for op in operations if str(op)])
        
        # Traditional pattern-based concept extraction
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
        
        return list(set(concepts))  # Remove duplicates
    
    def _determine_tools(self, problem: str, classification: Dict, parsed_data: Dict[str, Any] = None) -> List[str]:
        """Determine what tools/methods are needed to solve the problem, enhanced with parser insights."""
        tools = []
        
        category = classification.get("category", "")
        problem_lower = problem.lower()
        
        # Enhanced tool determination using parser data
        if parsed_data:
            operations = parsed_data.get('operations', [])
            math_elements = parsed_data.get('mathematical_elements', {})
            
            # Suggest tools based on parser-identified operations
            for op in operations:
                op_str = str(op).lower()
                if 'derivative' in op_str:
                    tools.append("calculus_engine")
                elif 'integral' in op_str:
                    tools.append("integration_solver")
                elif 'equation' in op_str:
                    tools.append("equation_solver")
            
            # Suggest tools based on mathematical elements
            if math_elements.get('derivatives'):
                tools.append("calculus_engine")
            if math_elements.get('integrals'):
                tools.append("integration_solver")
            if math_elements.get('equations'):
                tools.append("equation_solver")
        
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
        
        return list(set(tools))  # Remove duplicates
    
    def _analyze_characteristics(self, problem: str, parsed_data: Dict[str, Any] = None) -> Dict[str, bool]:
        """Analyze characteristics of the problem, enhanced with parser data."""
        problem_lower = problem.lower()
        
        characteristics = {
            "has_word_problem": any(word in problem_lower for word in ["if", "when", "find", "calculate", "determine"]),
            "has_multiple_steps": len(problem.split('.')) > 2 or len(problem.split(',')) > 3,
            "requires_proof": any(word in problem_lower for word in ["prove", "show", "demonstrate"]),
            "has_real_world_context": any(word in problem_lower for word in ["company", "person", "car", "house", "money"]),
            "has_variables": bool(re.search(r'[a-z]\s*[=\+\-\*\/]', problem_lower)),
            "has_numerical_computation": bool(re.search(r'\d+', problem)),
            "requires_graphing": any(word in problem_lower for word in ["graph", "plot", "draw", "sketch"])
        }
        
        # Enhanced characteristics using parser data
        if parsed_data:
            variables = parsed_data.get('variables', [])
            equations = parsed_data.get('equations', [])
            operations = parsed_data.get('operations', [])
            
            # Override/enhance characteristics with parser insights
            characteristics["has_variables"] = len(variables) > 0
            characteristics["has_multiple_equations"] = len(equations) > 1
            characteristics["has_complex_operations"] = any(
                'derivative' in str(op).lower() or 'integral' in str(op).lower() 
                for op in operations
            )
        
        return characteristics
    
    def _estimate_solution_complexity(self, problem: str, parsed_data: Dict[str, Any] = None) -> int:
        """Estimate the number of steps needed to solve the problem, enhanced with parser insights."""
        base_steps = 1
        
        # Enhanced complexity estimation using parser data
        if parsed_data:
            variables = parsed_data.get('variables', [])
            equations = parsed_data.get('equations', [])
            operations = parsed_data.get('operations', [])
            
            # Add steps based on parser insights
            base_steps += len(equations)  # Each equation adds complexity
            base_steps += len(variables) * 0.5  # More variables = more complexity
            
            # Complex operations increase step count
            complex_ops = ['derivative', 'integral', 'limit', 'matrix']
            for op in operations:
                if any(complex_op in str(op).lower() for complex_op in complex_ops):
                    base_steps += 2
        
        # Traditional complexity estimation
        if "system" in problem.lower():
            base_steps += 3
        if "quadratic" in problem.lower() or "x²" in problem or "x^2" in problem:
            base_steps += 2
        if any(word in problem.lower() for word in ["derivative", "integral"]):
            base_steps += 3
        if "word problem" in problem.lower() or len(problem.split()) > 20:
            base_steps += 2
        
        return min(int(base_steps), 10)  # Cap at 10 steps

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
