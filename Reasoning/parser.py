#!/usr/bin/env python3
"""
📐 MATHEMATICAL PROBLEM PARSER ENGINE
====================================

This module parses templatized word problems to extract:
1. Number of equations needed
2. Mathematical equations from the problem
3. Variables and their relationships
4. The target variable to solve for
5. Constraints and conditions

Key Features:
- LLM-powered equation extraction from templatized problems
- Chain-of-thought reasoning for equation formation
- Multi-equation problem handling
- Variable dependency analysis
- Target variable identification

Input: TemplatizationResult from templatizer
Output: ParseResult with equations, variables, and solving strategy

Author: LLM Agent Research Team
Date: October 2025
"""

import os
import sys
import re
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

# Add paths for imports
current_dir = os.path.dirname(__file__)
sys.path.insert(0, current_dir)

# Import the templatizer
try:
    from templatizer import TemplatizationResult, WordProblemTemplatizer
    TEMPLATIZER_AVAILABLE = True
except ImportError:
    print("⚠️ Templatizer not available in same directory")
    TEMPLATIZER_AVAILABLE = False
    TemplatizationResult = None

# LangChain imports
try:
    from langchain_core.prompts import PromptTemplate
    from langchain_openai import ChatOpenAI
    LANGCHAIN_AVAILABLE = True
    print("✅ LangChain with OpenAI available for parsing")
except ImportError:
    print("⚠️ LangChain not available, parser will have limited functionality")
    LANGCHAIN_AVAILABLE = False

from dotenv import load_dotenv
load_dotenv()

@dataclass
class Equation:
    """Represents a single mathematical equation."""
    equation_string: str
    variables: List[str]
    description: str
    equation_type: str  # e.g., "linear", "quadratic", "constraint"

@dataclass
class ParseResult:
    """Result of parsing a mathematical problem."""
    original_problem: str
    templatized_problem: str
    legend: Dict[str, str]
    
    # Core parsing results
    num_equations_needed: int
    equations: List[Equation]
    all_variables: List[str]
    target_variable: str
    target_variable_description: str
    
    # Analysis details
    problem_type: str
    difficulty: str
    reasoning_steps: List[str]
    constraints: List[str]
    
    # Metadata
    confidence_score: float
    processing_time: float
    metadata: Dict[str, Any]

class MathematicalProblemParser:
    """
    Advanced parser that extracts mathematical equations from templatized word problems.
    Uses LLM-powered chain-of-thought reasoning to identify equations and variables.
    """
    
    def __init__(self, api_key: str = None, model: str = None):
        """Initialize the parser with LLM support."""
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model if model is not None else os.getenv("MODEL_NAME", "google/gemini-2.0-flash-001")
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
        self.llm = None
        
        # Initialize LLM
        self._initialize_llm()
        
        self.logger.info("🔧 Mathematical Problem Parser initialized")
    
    def _initialize_llm(self):
        """Initialize the LangChain LLM."""
        if not LANGCHAIN_AVAILABLE or not self.api_key:
            self.logger.warning("⚠️ LLM not available - parser will have limited functionality")
            self.logger.warning("   Set OPENAI_API_KEY in .env for full parsing capabilities")
            return
        
        try:
            self.llm = ChatOpenAI(
                model=self.model,
                temperature=0.1,  # Low temperature for precise equation extraction
                openai_api_key=self.api_key,
                openai_api_base=self.base_url
            )
            self.logger.info(f"✅ Parser LLM initialized: {self.model}")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize LLM: {e}")
            self.llm = None
    
    def _analyze_problem_structure_with_llm(self, templatized_problem: str) -> Dict[str, Any]:
        """
        Use LLM with chain-of-thought to analyze the problem structure.
        This is the core intelligence of the parser.
        """
        if not self.llm:
            return self._analyze_problem_structure_fallback(templatized_problem)
        
        prompt = PromptTemplate(
            input_variables=["problem"],
            template="""
You are an expert mathematical problem analyzer. Analyze this templatized word problem and extract its mathematical structure.

TEMPLATIZED PROBLEM:
{problem}

Think step-by-step and provide a detailed analysis:

1. **UNDERSTAND THE PROBLEM**: What is being asked? What information is given?

2. **IDENTIFY VARIABLES**: List all the variables/unknowns in this problem. Use the placeholder names from the template (e.g., [Person1], [Item1]) or create appropriate variable names.

3. **COUNT EQUATIONS NEEDED**: How many independent equations are needed to solve this problem completely? Consider:
   - Simple problems: 1 equation (e.g., "x + 5 = 10")
   - Problems with multiple unknowns: Multiple equations (e.g., system of equations)
   - Sequential problems: May need intermediate equations

4. **FORMULATE EQUATIONS**: Write out each equation in mathematical notation. Be precise.

5. **IDENTIFY TARGET VARIABLE**: Which variable must be solved to answer the question?

6. **CLASSIFY PROBLEM TYPE**: What type of math problem is this? (linear equation, system of equations, rate problem, percentage, etc.)

Provide your analysis in this JSON format:
{{
    "understanding": "Brief description of what the problem asks",
    "variables": [
        {{"name": "variable_name", "description": "what this represents", "unit": "unit if applicable"}}
    ],
    "num_equations_needed": <number>,
    "equations": [
        {{
            "equation": "mathematical equation string",
            "description": "what this equation represents",
            "type": "equation type",
            "variables": ["list", "of", "variables"]
        }}
    ],
    "target_variable": "variable_to_solve",
    "target_description": "what this variable represents in the context",
    "problem_type": "type of problem",
    "difficulty": "easy/medium/hard",
    "constraints": ["any constraints or conditions"],
    "reasoning": ["step 1", "step 2", "..."],
    "confidence": 0.95
}}

Be precise and mathematical. Extract equations that can actually be solved.
"""
        )
        
        try:
            chain = prompt | self.llm
            response = chain.invoke({"problem": templatized_problem})
            
            # Extract response content
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())
                return analysis
            else:
                self.logger.warning("Could not parse LLM response as JSON")
                return self._analyze_problem_structure_fallback(templatized_problem)
                
        except Exception as e:
            self.logger.error(f"LLM analysis failed: {e}")
            return self._analyze_problem_structure_fallback(templatized_problem)
    
    def _analyze_problem_structure_fallback(self, templatized_problem: str) -> Dict[str, Any]:
        """Fallback analysis using rule-based approach when LLM is not available."""
        self.logger.info("Using rule-based fallback for parsing")
        
        # Extract numbers
        numbers = re.findall(r'\d+(?:\.\d+)?', templatized_problem)
        
        # Extract placeholders (variables)
        placeholders = re.findall(r'\[(\w+\d+)\]', templatized_problem)
        
        # Detect problem type based on keywords
        problem_lower = templatized_problem.lower()
        
        if 'how many' in problem_lower or 'total' in problem_lower:
            problem_type = "counting/summation"
            num_equations = 1
        elif 'system' in problem_lower or len(placeholders) > 2:
            problem_type = "system of equations"
            num_equations = len(set(placeholders))
        elif 'rate' in problem_lower or 'speed' in problem_lower:
            problem_type = "rate problem"
            num_equations = 1
        elif 'percent' in problem_lower or '%' in templatized_problem:
            problem_type = "percentage problem"
            num_equations = 1
        else:
            problem_type = "general arithmetic"
            num_equations = 1
        
        # Create basic equation structure
        variables = [{"name": ph, "description": f"Value for {ph}", "unit": "unknown"} 
                    for ph in set(placeholders)]
        
        if not variables:
            variables = [{"name": "x", "description": "unknown value", "unit": "unknown"}]
        
        return {
            "understanding": "Basic mathematical problem",
            "variables": variables,
            "num_equations_needed": num_equations,
            "equations": [{
                "equation": "equation_to_be_determined",
                "description": "Main equation",
                "type": problem_type,
                "variables": [v["name"] for v in variables]
            }],
            "target_variable": variables[0]["name"] if variables else "x",
            "target_description": "The value to find",
            "problem_type": problem_type,
            "difficulty": "medium",
            "constraints": [],
            "reasoning": ["Fallback rule-based analysis"],
            "confidence": 0.5
        }
    
    def parse_problem(self, templatization_result: "TemplatizationResult") -> ParseResult:
        """
        Main parsing method that takes templatization output and extracts equations.
        
        Args:
            templatization_result: Output from the templatizer
            
        Returns:
            ParseResult: Complete parsing with equations and target variable
        """
        start_time = datetime.now()
        
        self.logger.info("📐 Starting problem parsing...")
        
        # Extract problem details
        original = templatization_result.original_problem
        templatized = templatization_result.templated_problem
        legend = templatization_result.legend
        
        self.logger.info(f"📝 Templatized: {templatized}")
        
        # Analyze problem structure using LLM
        self.logger.info("🤖 Analyzing problem structure with LLM...")
        analysis = self._analyze_problem_structure_with_llm(templatized)
        
        # Build equations from analysis
        equations = []
        for eq_data in analysis.get('equations', []):
            equation = Equation(
                equation_string=eq_data.get('equation', ''),
                variables=eq_data.get('variables', []),
                description=eq_data.get('description', ''),
                equation_type=eq_data.get('type', 'unknown')
            )
            equations.append(equation)
        
        # Extract all variables
        all_variables = [v['name'] for v in analysis.get('variables', [])]
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Build result
        result = ParseResult(
            original_problem=original,
            templatized_problem=templatized,
            legend=legend,
            num_equations_needed=analysis.get('num_equations_needed', 1),
            equations=equations,
            all_variables=all_variables,
            target_variable=analysis.get('target_variable', 'x'),
            target_variable_description=analysis.get('target_description', 'unknown'),
            problem_type=analysis.get('problem_type', 'unknown'),
            difficulty=analysis.get('difficulty', 'medium'),
            reasoning_steps=analysis.get('reasoning', []),
            constraints=analysis.get('constraints', []),
            confidence_score=analysis.get('confidence', 0.5),
            processing_time=processing_time,
            metadata={
                'analysis_method': 'llm' if self.llm else 'fallback',
                'num_variables': len(all_variables),
                'understanding': analysis.get('understanding', '')
            }
        )
        
        self.logger.info(f"✅ Parsing complete: {len(equations)} equation(s) extracted")
        self.logger.info(f"🎯 Target variable: {result.target_variable}")
        
        return result
    
    def parse_from_text(self, problem_text: str) -> ParseResult:
        """
        Convenience method to parse directly from text.
        Automatically runs templatization first.
        """
        if not TEMPLATIZER_AVAILABLE:
            raise ImportError("Templatizer not available. Cannot parse from text.")
        
        # Run templatization first
        templatizer = WordProblemTemplatizer(api_key=self.api_key, model=self.model)
        templatization_result = templatizer.templatize_problem(problem_text)
        
        # Parse the templatized result
        return self.parse_problem(templatization_result)

# Convenience functions
def create_parser(api_key: str = None, model: str = "gemini-2.0-flash-exp") -> MathematicalProblemParser:
    """Create a new parser instance."""
    return MathematicalProblemParser(api_key=api_key, model=model)

def parse_math_problem(problem_text: str, api_key: str = None) -> ParseResult:
    """Quick function to parse a problem from text."""
    parser = create_parser(api_key=api_key)
    return parser.parse_from_text(problem_text)

# Example usage and testing
if __name__ == "__main__":
    print("📐 Mathematical Problem Parser")
    print("=" * 50)
    
    # Test problems
    test_problems = [
        "John has 5 apples and Mary has 3 oranges. How many fruits do they have together?",
        
        "Sarah bought 3 books for $15 each. She paid with a $50 bill. How much change did she receive?",
        
        "A train travels 120 miles in 2 hours. What is its average speed?",
        
        "The sum of two numbers is 15 and their difference is 3. What are the two numbers?",
        
        "Lisa earns $85,000 per year. If she gets a 12% raise, what will be her new salary?"
    ]
    
    try:
        parser = create_parser()
        
        for i, problem in enumerate(test_problems, 1):
            print(f"\n{'='*60}")
            print(f"🧪 Test Problem {i}:")
            print(f"   {problem}")
            print("-" * 60)
            
            result = parser.parse_from_text(problem)
            
            print(f"\n📊 PARSING RESULTS:")
            print(f"   Problem Type: {result.problem_type}")
            print(f"   Difficulty: {result.difficulty}")
            print(f"   Equations Needed: {result.num_equations_needed}")
            print(f"   Variables: {result.all_variables}")
            print(f"   🎯 Target Variable: {result.target_variable}")
            print(f"      Description: {result.target_variable_description}")
            
            print(f"\n📐 EQUATIONS:")
            for j, eq in enumerate(result.equations, 1):
                print(f"   {j}. {eq.equation_string}")
                print(f"      Type: {eq.equation_type}")
                print(f"      Variables: {eq.variables}")
                print(f"      Description: {eq.description}")
            
            if result.constraints:
                print(f"\n⚠️  CONSTRAINTS:")
                for constraint in result.constraints:
                    print(f"   - {constraint}")
            
            print(f"\n💭 REASONING:")
            for step in result.reasoning_steps:
                print(f"   • {step}")
            
            print(f"\n📈 Confidence: {result.confidence_score:.2f}")
            print(f"⏱️  Processing Time: {result.processing_time:.3f}s")
            
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 Make sure:")
        print("   - LangChain is installed: pip install langchain-google-genai")
        print("   - Templatizer is in the same directory")
        print("   - GOOGLE_API_KEY is set in .env file")
