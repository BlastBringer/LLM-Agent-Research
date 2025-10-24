#!/usr/bin/env python3
"""
🔢 VARIABLE EXTRACTOR ENGINE
============================

This module extracts numerical values and assigns them to variables from:
1. The templatized problem text
2. The parsed equations

It bridges the gap between symbolic equations and actual values:
- Input: "total_fruits = apples + oranges" + "[Person1] has 5 apples and [Person2] has 3 oranges"
- Output: {apples: 5, oranges: 3}

Key Features:
- LLM-powered value extraction from natural language
- Pattern-based fallback for reliability
- Handles various number formats (integers, decimals, fractions)
- Extracts units alongside values
- Maps variable names to their numerical values

Author: LLM Agent Research Team
Date: October 2025
"""

import os
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# LangChain imports
try:
    from langchain_core.prompts import PromptTemplate
    from langchain_openai import ChatOpenAI
    LANGCHAIN_AVAILABLE = True
    print("✅ LangChain with OpenAI available for variable extraction")
except ImportError:
    print("⚠️ LangChain not available, using pattern-based fallback only")
    LANGCHAIN_AVAILABLE = False

@dataclass
class VariableValue:
    """Represents a variable with its value and unit."""
    name: str
    value: float
    unit: Optional[str]
    raw_text: str  # Original text where it was found
    confidence: float

@dataclass
class ExtractionResult:
    """Result of variable extraction."""
    variables: Dict[str, VariableValue]
    all_quantities: List[Tuple[float, str]]  # All (value, unit) pairs found
    extraction_method: str  # 'llm' or 'pattern'
    confidence_score: float
    processing_steps: List[str]
    metadata: Dict[str, Any]

class VariableExtractor:
    """
    Extracts numerical values from text and maps them to variables.
    Uses LLM for intelligent extraction with pattern-based fallback.
    """
    
    def __init__(self, api_key: str = None, model: str = None):
        """Initialize the variable extractor."""
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model if model is not None else os.getenv("MODEL_NAME", "google/gemini-2.0-flash-001")
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
        self.llm = None
        
        # Initialize LLM
        self._initialize_llm()
        
        self.logger.info("🔧 Variable Extractor initialized")
    
    def _initialize_llm(self):
        """Initialize the LangChain LLM."""
        if not LANGCHAIN_AVAILABLE or not self.api_key:
            self.logger.warning("⚠️ LLM not available - using pattern-based extraction only")
            return
        
        try:
            self.llm = ChatOpenAI(
                model=self.model,
                temperature=0.0,  # Zero temperature for precise extraction
                openai_api_key=self.api_key,
                openai_api_base=self.base_url
            )
            self.logger.info(f"✅ Variable Extractor LLM initialized: {self.model}")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize LLM: {e}")
            self.llm = None
    
    def extract_with_llm(self, templatized_problem: str, variables: List[str]) -> Dict[str, VariableValue]:
        """
        Use LLM to extract variable values from text.
        The LLM understands context better than regex.
        """
        if not self.llm:
            return {}
        
        prompt_template = PromptTemplate(
            input_variables=["problem", "variables"],
            template="""You are a mathematical variable extractor. Given a word problem and a list of variables, extract the numerical value and unit for each variable.

Problem:
{problem}

Variables to find: {variables}

For each variable, identify:
1. Its numerical value
2. Its unit (if any)
3. The exact text where you found it

Return ONLY a JSON object in this format:
{{
  "variable_name": {{
    "value": <number>,
    "unit": "<unit or null>",
    "text": "<original text>"
  }}
}}

Example:
Problem: "A train travels 120 miles in 2 hours."
Variables: ["distance", "time"]
Response:
{{
  "distance": {{"value": 120, "unit": "miles", "text": "120 miles"}},
  "time": {{"value": 2, "unit": "hours", "text": "2 hours"}}
}}

Now extract for the given problem. Return ONLY the JSON, no explanation."""
        )
        
        try:
            chain = prompt_template | self.llm
            response = chain.invoke({
                "problem": templatized_problem,
                "variables": ", ".join(variables)
            })
            
            # Parse LLM response
            import json
            content = response.content.strip()
            
            # Remove markdown code blocks if present
            if content.startswith("```"):
                content = re.sub(r'^```(?:json)?\s*|\s*```$', '', content, flags=re.MULTILINE)
            
            extracted_data = json.loads(content)
            
            # Convert to VariableValue objects
            result = {}
            for var_name, data in extracted_data.items():
                result[var_name] = VariableValue(
                    name=var_name,
                    value=float(data["value"]),
                    unit=data.get("unit"),
                    raw_text=data.get("text", ""),
                    confidence=0.9  # High confidence for LLM extraction
                )
            
            self.logger.info(f"✅ LLM extracted {len(result)} variables")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ LLM extraction failed: {e}")
            return {}
    
    def extract_with_patterns(self, text: str) -> List[Tuple[float, Optional[str], str]]:
        """
        Pattern-based extraction as fallback.
        Returns list of (value, unit, raw_text) tuples.
        """
        results = []
        
        # Pattern: number followed by optional unit
        # Matches: "5 apples", "120 miles", "2 hours", "$15", "3.5 kg"
        patterns = [
            # Number with unit
            r'(\d+\.?\d*)\s*([a-zA-Z$€£¥]+(?:/[a-zA-Z]+)?)',
            # Just number (for dimensionless quantities)
            r'(\d+\.?\d*)',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                try:
                    value = float(match.group(1))
                    unit = match.group(2) if len(match.groups()) > 1 else None
                    raw_text = match.group(0)
                    results.append((value, unit, raw_text))
                except (ValueError, IndexError):
                    continue
        
        self.logger.info(f"📊 Pattern extraction found {len(results)} quantities")
        return results
    
    def map_quantities_to_variables(
        self,
        quantities: List[Tuple[float, Optional[str], str]],
        variables: List[str],
        equations: List[str]
    ) -> Dict[str, VariableValue]:
        """
        Map extracted quantities to variable names using heuristics.
        This is used when LLM extraction fails.
        """
        result = {}
        used_quantities = set()
        
        # Simple heuristic: match based on order and context
        # This is a fallback, so it's basic
        for i, var in enumerate(variables):
            if i < len(quantities) and i not in used_quantities:
                value, unit, text = quantities[i]
                result[var] = VariableValue(
                    name=var,
                    value=value,
                    unit=unit,
                    raw_text=text,
                    confidence=0.6  # Lower confidence for pattern matching
                )
                used_quantities.add(i)
        
        return result
    
    def extract_variables(
        self,
        templatized_problem: str,
        variables: List[str],
        equations: List[str]
    ) -> ExtractionResult:
        """
        Main extraction method.
        
        Args:
            templatized_problem: The templatized problem text
            variables: List of variable names from parser
            equations: List of equation strings from parser
            
        Returns:
            ExtractionResult with variable values mapped
        """
        self.logger.info("🔢 Starting variable extraction...")
        processing_steps = []
        
        # Try LLM extraction first
        extracted_vars = {}
        extraction_method = "pattern"
        
        if self.llm and variables:
            self.logger.info("🤖 Attempting LLM extraction...")
            processing_steps.append("Attempting LLM-based extraction")
            extracted_vars = self.extract_with_llm(templatized_problem, variables)
            
            if extracted_vars:
                extraction_method = "llm"
                processing_steps.append(f"LLM extracted {len(extracted_vars)} variables")
        
        # Fallback to pattern-based extraction
        if not extracted_vars:
            self.logger.info("📊 Using pattern-based extraction...")
            processing_steps.append("Falling back to pattern-based extraction")
            quantities = self.extract_with_patterns(templatized_problem)
            processing_steps.append(f"Found {len(quantities)} quantities")
            
            extracted_vars = self.map_quantities_to_variables(
                quantities, variables, equations
            )
            processing_steps.append(f"Mapped to {len(extracted_vars)} variables")
        
        # Calculate confidence
        if extracted_vars:
            avg_confidence = sum(v.confidence for v in extracted_vars.values()) / len(extracted_vars)
        else:
            avg_confidence = 0.0
        
        # Get all quantities for reference
        all_quantities = self.extract_with_patterns(templatized_problem)
        
        result = ExtractionResult(
            variables=extracted_vars,
            all_quantities=all_quantities,
            extraction_method=extraction_method,
            confidence_score=avg_confidence,
            processing_steps=processing_steps,
            metadata={
                "num_variables": len(variables),
                "num_extracted": len(extracted_vars),
                "num_quantities_found": len(all_quantities)
            }
        )
        
        self.logger.info(f"✅ Extraction complete: {len(extracted_vars)} variables")
        return result


def extract_variables_from_problem(
    templatized_problem: str,
    variables: List[str],
    equations: List[str]
) -> ExtractionResult:
    """
    Convenience function to extract variables without creating an instance.
    
    Args:
        templatized_problem: The templatized problem text
        variables: List of variable names
        equations: List of equations
        
    Returns:
        ExtractionResult with extracted values
    """
    extractor = VariableExtractor()
    return extractor.extract_variables(templatized_problem, variables, equations)


if __name__ == "__main__":
    """Demo and testing."""
    print("\n" + "="*70)
    print("🔢 VARIABLE EXTRACTOR DEMO")
    print("="*70)
    
    # Example 1: Simple problem
    problem1 = "[Person1] has 5 apples and [Person2] has 3 oranges. How many fruits total?"
    vars1 = ["apples", "oranges", "total_fruits"]
    eqs1 = ["total_fruits = apples + oranges"]
    
    print("\n📝 Example 1: Simple Addition")
    print(f"Problem: {problem1}")
    print(f"Variables: {vars1}")
    
    result1 = extract_variables_from_problem(problem1, vars1, eqs1)
    print(f"\n✅ Extracted {len(result1.variables)} variables:")
    for var_name, var_val in result1.variables.items():
        print(f"   {var_name} = {var_val.value} {var_val.unit or ''}")
    
    # Example 2: Rate problem
    problem2 = "A train travels 120 miles in 2 hours. What is its average speed?"
    vars2 = ["distance", "time", "speed"]
    eqs2 = ["speed = distance / time"]
    
    print("\n📝 Example 2: Rate Problem")
    print(f"Problem: {problem2}")
    print(f"Variables: {vars2}")
    
    result2 = extract_variables_from_problem(problem2, vars2, eqs2)
    print(f"\n✅ Extracted {len(result2.variables)} variables:")
    for var_name, var_val in result2.variables.items():
        print(f"   {var_name} = {var_val.value} {var_val.unit or ''}")
    
    print("\n" + "="*70)
    print("✅ Demo complete!")
    print("="*70)
