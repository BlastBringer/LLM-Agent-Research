#!/usr/bin/env python3
"""
🎓 APPRENTICE MODEL - The Learning Student
==========================================

This is the "student" model that learns over time through fine-tuning.
Currently uses Llama 3 8B via OpenRouter API.

Later, we'll:
1. Download Llama 3 8B locally
2. Fine-tune it with QLoRA using collected training data
3. Gradually improve its performance

Key Features:
- Structured JSON output for consistency
- Step-by-step reasoning format
- Temperature tuning for reliability
- Detailed logging for debugging

Usage:
    apprentice = ApprenticeModel()
    solution = apprentice.solve(problem_data)
    print(solution.final_answer)
    print(solution.reasoning_steps)
"""

import os
import json
import logging
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Try to import LangChain
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import PromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("⚠️  LangChain not available. Install with: pip install langchain langchain-openai")

# Try to import Ollama support
try:
    from langchain_ollama import ChatOllama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    ChatOllama = None

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ApprenticeSolution:
    """Result from the apprentice model."""
    final_answer: Optional[float]
    reasoning_steps: List[str]
    raw_response: str
    extraction_method: str  # 'json' or 'pattern'
    confidence: float
    metadata: Dict[str, Any]


class ApprenticeModel:
    """
    The Apprentice (Student) Model.
    Uses Llama 3 8B to solve math problems with step-by-step reasoning.
    """
    
    def __init__(
        self,
        model_name: str = None,
        api_key: str = None,
        base_url: str = None,
        temperature: float = 0.1
    ):
        
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.use_ollama = os.getenv("USE_OLLAMA", "0").lower() in ("1", "true", "yes")
        self.model_name = model_name or os.getenv("APPRENTICE_MODEL", "meta-llama/llama-3-8b-instruct")
        self.local_model = os.getenv("APPRENTICE_LOCAL_MODEL", "llama3.1:8b")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.temperature = temperature
        
        # Initialize LLM
        self.llm = None
        self.provider = None
        self._initialize_llm()
        
        self.logger.info(f"🎓 Apprentice Model initialized: {self.provider} - {self.model_name if not self.use_ollama else self.local_model}")
    
    def _initialize_llm(self):
        """Initialize the LangChain LLM (Ollama or OpenRouter)."""
        if not LANGCHAIN_AVAILABLE:
            self.logger.error("❌ LangChain not available")
            return
        
        # Try Ollama first if enabled
        if self.use_ollama:
            if not OLLAMA_AVAILABLE:
                self.logger.warning("⚠️ Ollama support not available. Install: pip install langchain-ollama")
                self.logger.info("ℹ️  Falling back to OpenRouter...")
                self.use_ollama = False
            else:
                try:
                    self.llm = ChatOllama(
                        model=self.local_model,
                        temperature=self.temperature,
                        base_url=self.ollama_base_url
                    )
                    self.provider = "Ollama (Local)"
                    self.logger.info(f"✅ Apprentice using Ollama locally: {self.local_model}")
                    return
                except Exception as e:
                    self.logger.error(f"❌ Failed to initialize Ollama: {e}")
                    self.logger.info("ℹ️  Falling back to OpenRouter...")
                    self.use_ollama = False
        
        # Use OpenRouter if Ollama not enabled or failed
        if not self.api_key:
            self.logger.error("❌ No API key found. Set OPENAI_API_KEY in .env")
            return
        
        try:
            self.llm = ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                openai_api_key=self.api_key,
                openai_api_base=self.base_url,
                max_tokens=2000
            )
            self.provider = "OpenRouter (API)"
            self.logger.info(f"✅ Apprentice using OpenRouter: {self.model_name}")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize LLM: {e}")
            self.llm = None
    
    def solve(self, problem_data: Dict[str, Any]) -> ApprenticeSolution:
        """
        Solve a math problem using the apprentice model.
        
        Args:
            problem_data: Dictionary containing all processed data:
                - original_problem: Original text
                - templatization: Templated version
                - parsing: Equations and target variable
                - variable_extraction: Variable values
                - unit_standardization: Standardized units
        
        Returns:
            ApprenticeSolution with answer and reasoning
        """
        if not self.llm:
            return self._create_error_solution("LLM not initialized")
        
        self.logger.info("🧠 Apprentice attempting to solve...")
        
        # Build the prompt
        prompt = self._build_prompt(problem_data)
        
        try:
            # Get response from LLM
            response = self.llm.invoke(prompt)
            raw_response = response.content
            
            self.logger.debug(f"Raw response: {raw_response}")
            
            # Parse the response
            solution = self._parse_response(raw_response)
            
            self.logger.info(f"✅ Apprentice proposed answer: {solution.final_answer}")
            return solution
            
        except Exception as e:
            self.logger.error(f"❌ Apprentice failed: {e}")
            return self._create_error_solution(str(e))
    
    def _build_prompt(self, problem_data: Dict[str, Any]) -> str:
        """
        Build a detailed prompt for the apprentice.
        This prompt teaches the model to reason step-by-step.
        """
        # Extract relevant data
        original_problem = problem_data.get('original_problem', '')
        
        # Get equations
        equations = []
        if 'parsing' in problem_data and 'equations' in problem_data['parsing']:
            equations = [eq.get('equation_string', '') for eq in problem_data['parsing']['equations']]
        
        # Get variables
        variables = {}
        have_standardized = False
        if 'unit_standardization' in problem_data and problem_data['unit_standardization'].get('standardized_variables'):
            std_vars = problem_data['unit_standardization'].get('standardized_variables', {})
            for var_name, var_data in std_vars.items():
                value = var_data.get('standardized_value', 0)
                unit = var_data.get('standardized_unit', '')
                variables[var_name] = f"{value} {unit}".strip()
            have_standardized = True
        elif 'variable_extraction' in problem_data and problem_data['variable_extraction'].get('variables'):
            ext_vars = problem_data['variable_extraction'].get('variables', {})
            for var_name, var_data in ext_vars.items():
                # var_data is dict with keys: value, unit, raw_text, confidence
                value = var_data.get('value', 0)
                unit = var_data.get('unit', '') or ''
                variables[var_name] = f"{value} {unit}".strip()
        
        # Get target variable
        target_var = ''
        if 'parsing' in problem_data:
            target_var = problem_data['parsing'].get('target_variable', '')
        
        # Build the prompt
        prompt = f"""You are a math problem solver. Solve the following problem step-by-step.

PROBLEM:
{original_problem}

GIVEN INFORMATION:
"""
        
        if variables:
            for var, val in variables.items():
                prompt += f"- {var} = {val}\n"
        else:
            prompt += "- No variables provided (you may need to extract them from the problem)\n"
        
        if equations:
            prompt += f"\nEQUATIONS:\n"
            for eq in equations:
                prompt += f"- {eq}\n"
        
        if target_var:
            prompt += f"\nFIND: {target_var}\n"

        if have_standardized:
            prompt += """
⚠️ IMPORTANT: Use ONLY the standardized values above. Do NOT convert units yourself.
Your answer must be in the SAME UNITS as the given standardized values (SI units).

INSTRUCTIONS FOR SOLVING:
1. Start by stating what you need to find (the target variable)
2. Write down the formula or equation you'll use
3. Substitute the ACTUAL NUMBERS from GIVEN INFORMATION into the equation
4. Show the arithmetic calculation step-by-step
5. State the final numerical answer

CRITICAL: Your reasoning_steps MUST show ACTUAL CALCULATIONS with NUMBERS, not generic descriptions!

GOOD EXAMPLE (shows actual math):
"reasoning_steps": [
    "We need to find the total time in seconds",
    "Using formula: time = distance / speed",
    "Substitute values: time = 120 / 30",
    "Calculate: 120 ÷ 30 = 4",
    "Final answer: 4 seconds"
]

BAD EXAMPLE (too generic - DO NOT DO THIS):
"reasoning_steps": [
    "Understand what we're looking for",
    "Identify the formula or equation", 
    "Substitute the values",
    "Perform the calculation"
]

OUTPUT FORMAT (YOU MUST USE THIS EXACT JSON FORMAT):
```json
{
    "reasoning_steps": [
        "State what we're solving for: [target variable in context]",
        "Formula: [actual equation with variable names]",
        "Substitute: [equation with actual numbers from GIVEN INFORMATION]",
        "Calculate: [show arithmetic with numbers, e.g., 63 ÷ 13 = 4.846]",
        "Apply: [if multi-step, show next calculation, e.g., 4.846 × 39 = 189]",
        "Result: [final answer with unit]"
    ],
    "final_answer": <numeric_value_in_SI_units>
}
```

CRITICAL REMINDERS:
- SHOW ACTUAL NUMBERS in every reasoning step (not placeholders like [value] or [calculation])
- Write out arithmetic: "63 ÷ 13 = 4.846" not "divide the values"
- Use ONLY standardized values from GIVEN INFORMATION above
- If multiple steps, show each calculation explicitly
- Output ONLY the JSON in the format shown above
"""
        else:
            prompt += """
⚠️ IMPORTANT: Use ONLY the given values above. Keep units consistent if present; do not invent conversions.

INSTRUCTIONS FOR SOLVING:
1. Start by stating what you need to find (the target variable)
2. Write down the formula or equation you'll use
3. Substitute the ACTUAL NUMBERS from GIVEN INFORMATION into the equation
4. Show the arithmetic calculation step-by-step with actual numbers
5. State the final numerical answer

CRITICAL: Your reasoning_steps MUST show ACTUAL CALCULATIONS with NUMBERS, not generic descriptions!

GOOD EXAMPLE (shows actual math):
"reasoning_steps": [
    "We need to find the total time in minutes",
    "For rate problems, first find rate: rate = time_per_segment / distance_per_segment",
    "Substitute: rate = 63 minutes / 13 miles",
    "Calculate rate: 63 ÷ 13 = 4.846 minutes per mile",
    "Apply rate to total distance: time = rate × total_distance = 4.846 × 39 miles",
    "Calculate: 4.846 × 39 = 189.0 minutes",
    "Final answer: 189.0 minutes"
]

BAD EXAMPLE (too generic - DO NOT DO THIS):
"reasoning_steps": [
    "Understand what we're looking for",
    "Identify the formula",
    "Substitute the values", 
    "Perform the calculation"
]

OUTPUT FORMAT (YOU MUST USE THIS EXACT JSON FORMAT):
```json
{
    "reasoning_steps": [
        "State what we're solving for: [target variable in context]",
        "Formula needed: [actual equation with variable names]",
        "Substitute: [equation with actual numbers from GIVEN INFORMATION]",
        "Calculate: [show arithmetic with numbers, e.g., 63 ÷ 13 = 4.846]",
        "Apply (if multi-step): [next calculation, e.g., 4.846 × 39 = 189.0]",
        "Result: [final answer with unit if applicable]"
    ],
    "final_answer": <numeric_value>
}
```

CRITICAL REMINDERS:
- SHOW ACTUAL NUMBERS in every reasoning step (not placeholders)
- Write out arithmetic: "63 ÷ 13 = 4.846" not "divide the values"
- For rate problems: ALWAYS calculate rate first, then apply it (show both calculations)
- Use ONLY values from GIVEN INFORMATION above
- Output ONLY the JSON in the format shown above
"""
        
        return prompt
    
    def _parse_response(self, raw_response: str) -> ApprenticeSolution:
        """
        Parse the apprentice's response to extract answer and reasoning.
        Tries JSON parsing first, then falls back to pattern matching.
        """
        # Method 1: Try to parse as JSON
        solution = self._parse_json_response(raw_response)
        if solution:
            return solution
        
        # Method 2: Pattern matching fallback
        solution = self._parse_with_patterns(raw_response)
        if solution:
            return solution
        
        # Method 3: Failed to parse
        return ApprenticeSolution(
            final_answer=None,
            reasoning_steps=["Failed to parse response"],
            raw_response=raw_response,
            extraction_method='failed',
            confidence=0.0,
            metadata={'error': 'parsing_failed'}
        )
    
    def _parse_json_response(self, response: str) -> Optional[ApprenticeSolution]:
        """Try to parse response as JSON."""
        try:
            # Remove markdown code blocks if present
            cleaned = response.strip()
            if '```' in cleaned:
                # Extract content between ```json and ```
                match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', cleaned, re.DOTALL)
                if match:
                    cleaned = match.group(1)
            
            # Parse JSON
            data = json.loads(cleaned)
            
            # Extract fields
            reasoning_steps = data.get('reasoning_steps', [])
            final_answer = data.get('final_answer')
            
            if final_answer is not None:
                final_answer = float(final_answer)
            
            # Quality check: Warn if reasoning is too generic
            confidence = 0.9
            if reasoning_steps:
                generic_indicators = [
                    'understand what',
                    'identify the formula',
                    'substitute the values',
                    'perform the calculation',
                    '[target]',
                    '[formula',
                    '[show',
                    '[value'
                ]
                generic_count = sum(
                    1 for step in reasoning_steps 
                    for indicator in generic_indicators 
                    if indicator.lower() in step.lower()
                )
                # Check if any step contains actual numbers
                has_numbers = any(
                    re.search(r'\d+\.?\d*\s*[÷×+\-=/]\s*\d+\.?\d*', step) 
                    for step in reasoning_steps
                )
                
                if generic_count >= 3 or not has_numbers:
                    self.logger.warning("⚠️  Reasoning steps appear too generic (no actual calculations shown)")
                    confidence = 0.6
            
            return ApprenticeSolution(
                final_answer=final_answer,
                reasoning_steps=reasoning_steps,
                raw_response=response,
                extraction_method='json',
                confidence=confidence,
                metadata={
                    'parsed_successfully': True,
                    'has_detailed_reasoning': confidence >= 0.9
                }
            )
            
        except Exception as e:
            self.logger.debug(f"JSON parsing failed: {e}")
            return None
    
    def _parse_with_patterns(self, response: str) -> Optional[ApprenticeSolution]:
        """Fallback: Extract answer using regex patterns."""
        try:
            # Look for common patterns
            patterns = [
                r'final[_\s]answer[:\s]+(\d+\.?\d*)',
                r'answer[:\s]+(\d+\.?\d*)',
                r'result[:\s]+(\d+\.?\d*)',
                r'=\s*(\d+\.?\d*)\s*$',
            ]
            
            final_answer = None
            for pattern in patterns:
                match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
                if match:
                    final_answer = float(match.group(1))
                    break
            
            # Extract reasoning steps (split by newlines, filter non-empty)
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            reasoning_steps = [line for line in lines if len(line) > 10][:10]  # Max 10 steps
            
            return ApprenticeSolution(
                final_answer=final_answer,
                reasoning_steps=reasoning_steps,
                raw_response=response,
                extraction_method='pattern',
                confidence=0.6,
                metadata={'pattern_matched': final_answer is not None}
            )
            
        except Exception as e:
            self.logger.debug(f"Pattern extraction failed: {e}")
            return None
    
    def _create_error_solution(self, error_msg: str) -> ApprenticeSolution:
        """Create an error solution."""
        return ApprenticeSolution(
            final_answer=None,
            reasoning_steps=[f"Error: {error_msg}"],
            raw_response="",
            extraction_method='error',
            confidence=0.0,
            metadata={'error': error_msg}
        )


if __name__ == "__main__":
    # Test the apprentice
    print("🧪 Testing Apprentice Model")
    print("=" * 70)
    
    # Create a sample problem
    test_problem = {
        'original_problem': 'John has 5 apples and Mary gives him 3 more. How many apples does John have now?',
        'parsing': {
            'equations': [
                {'equation_string': 'total_apples = initial_apples + given_apples'}
            ],
            'target_variable': 'total_apples'
        },
        'unit_standardization': {
            'standardized_variables': {
                'initial_apples': {'standardized_value': 5, 'standardized_unit': ''},
                'given_apples': {'standardized_value': 3, 'standardized_unit': ''}
            }
        }
    }
    
    apprentice = ApprenticeModel()
    if apprentice.llm:
        solution = apprentice.solve(test_problem)
        
        print("\n📝 Reasoning Steps:")
        for i, step in enumerate(solution.reasoning_steps, 1):
            print(f"  {i}. {step}")
        
        print(f"\n🎯 Final Answer: {solution.final_answer}")
        print(f"📊 Confidence: {solution.confidence}")
        print(f"🔧 Method: {solution.extraction_method}")
    else:
        print("❌ Apprentice LLM not initialized. Check your .env file.")
