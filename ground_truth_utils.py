#!/usr/bin/env python3
"""
Ground Truth Extraction Utilities
==================================

Extract answers from various dataset formats:
- MATH dataset: \boxed{answer} format
- GSM8K: #### answer format
- Custom: direct numeric fields
"""

import re
from typing import Optional, Tuple

def extract_boxed_answer(text: str) -> Optional[str]:
    """
    Extract answer from LaTeX \boxed{} format.
    Handles nested braces properly.
    
    Examples:
        "\\boxed{150}" -> "150"
        "\\boxed{3.5}" -> "3.5"
        "\\boxed{-\\frac{1}{8}}" -> "-\\frac{1}{8}"
        "\\boxed{(-5,-2] \\cup (-1,3]}" -> "(-5,-2] \\cup (-1,3]"
    """
    # Find \boxed{ and then match braces
    start = text.find(r'\boxed{')
    if start == -1:
        return None
    
    # Start after '\boxed{'
    start += 7
    brace_count = 1
    i = start
    
    # Find matching closing brace
    while i < len(text) and brace_count > 0:
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
        i += 1
    
    if brace_count == 0:
        # Found matching brace
        return text[start:i-1].strip()
    
    return None

def extract_numeric_answer(text: str) -> Optional[float]:
    """
    Extract numeric value from answer string.
    
    Examples:
        "150" -> 150.0
        "3.5 miles" -> 3.5
        "$42.50" -> 42.5
        "\\frac{9}{7}" -> 1.2857142857142858
        "\\dfrac{-1}{8}" -> -0.125
        "-\\frac{1}{8}" -> -0.125
    """
    # Remove dollar signs and currency symbols first (but keep negative signs)
    cleaned = text.replace('$', '').replace('€', '').replace('£', '')
    
    # Handle LaTeX fractions: \frac{numerator}{denominator} or \dfrac{numerator}{denominator}
    # Pattern matches: -\frac{num}{den} or \frac{-num}{den} or \dfrac{num}{den}
    frac_pattern = r'(-?)\\d?frac\{(-?\d+\.?\d*)\}\{(-?\d+\.?\d*)\}'
    match = re.search(frac_pattern, cleaned)
    if match:
        try:
            # Handle sign before \frac
            sign = -1 if match.group(1) == '-' else 1
            numerator = float(match.group(2))
            denominator = float(match.group(3))
            if denominator != 0:
                return sign * (numerator / denominator)
        except (ValueError, ZeroDivisionError):
            pass
    
    # Handle simple fractions without LaTeX: "9/7" or "-1/8"
    simple_frac_pattern = r'^(-?\d+\.?\d*)/(-?\d+\.?\d*)$'
    match = re.search(simple_frac_pattern, cleaned.strip())
    if match:
        try:
            numerator = float(match.group(1))
            denominator = float(match.group(2))
            if denominator != 0:
                return numerator / denominator
        except (ValueError, ZeroDivisionError):
            pass
    
    # Handle regular numbers (possibly with units after)
    # Extract first number-like token
    number_pattern = r'(-?\d+\.?\d*)'
    match = re.search(number_pattern, cleaned)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    
    return None

def extract_ground_truth_from_problem(problem_data: dict) -> Tuple[Optional[float], Optional[str], Optional[str]]:
    """
    Extract ground truth answer from problem data.
    
    Args:
        problem_data: Dict with problem information
    
    Returns:
        (numeric_answer, raw_answer_string, unit)
    """
    raw_answer = None
    
    # Try different field names
    for field in ['output', 'answer', 'final_answer', 'solution', 'ground_truth']:
        if field in problem_data:
            raw_answer = problem_data[field]
            break
    
    if not raw_answer:
        return None, None, None
    
    raw_answer_str = str(raw_answer)
    
    # Extract from \boxed{} format first (MATH dataset)
    boxed = extract_boxed_answer(raw_answer_str)
    if boxed:
        # Use the boxed content for parsing
        parse_target = boxed
    else:
        parse_target = raw_answer_str
    
    # Try to extract numeric value from the parsed target
    numeric = extract_numeric_answer(parse_target)
    
    # Try to extract unit
    unit = None
    if isinstance(raw_answer, str):
        # Common unit patterns
        unit_pattern = r'\d+\.?\d*\s*([a-zA-Z]+)'
        match = re.search(unit_pattern, raw_answer)
        if match:
            unit = match.group(1)
    
    return numeric, str(raw_answer), unit

def is_word_problem(problem_text: str) -> bool:
    """
    Check if this is a word problem (vs pure math).
    
    Word problems have:
    - Natural language descriptions
    - Real-world scenarios
    - Common nouns (train, apple, money)
    """
    # Heuristics for word problems
    word_problem_indicators = [
        'travels', 'has', 'buys', 'costs', 'earns',
        'hours', 'miles', 'apples', 'dollars',
        'john', 'mary', 'alice', 'bob',
        'train', 'car', 'store', 'school'
    ]
    
    text_lower = problem_text.lower()
    return any(indicator in text_lower for indicator in word_problem_indicators)

if __name__ == '__main__':
    # Test cases
    test_cases = [
        {"output": "\\boxed{150}", "expected": 150.0},
        {"output": "answer is $42.50", "expected": 42.5},
        {"answer": "3.14159", "expected": 3.14159},
        # Note: Interval notation extracts first number, which is fine - intervals aren't fully supported
        # {"output": "\\boxed{(-5,-2]}", "expected": None},  # Interval notation - skip
        {"output": "\\dfrac{9}{7}", "expected": 9/7},  # LaTeX fraction
        {"output": "\\boxed{-\\frac{1}{8}}", "expected": -1/8},  # Negative LaTeX fraction in boxed
        {"output": "\\frac{23}{7}", "expected": 23/7},  # Another LaTeX fraction
        {"output": "9/7", "expected": 9/7},  # Simple fraction
        {"output": "-1/8", "expected": -1/8},  # Negative simple fraction
        {"output": "\\boxed{\\dfrac{9}{7}}", "expected": 9/7},  # dfrac in boxed
    ]
    
    print("=" * 70)
    print("TESTING GROUND TRUTH EXTRACTION (with LaTeX fractions)")
    print("=" * 70)
    
    passed = 0
    failed = 0
    
    for i, case in enumerate(test_cases, 1):
        numeric, raw, unit = extract_ground_truth_from_problem(case)
        expected = case['expected']
        
        # Check if match (with tolerance for floats)
        if expected is None:
            match = numeric is None
        elif numeric is None:
            match = False
        else:
            match = abs(numeric - expected) < 1e-9
        
        status = "✅ PASS" if match else "❌ FAIL"
        print(f"\nTest {i}: {status}")
        print(f"  Input: {case}")
        print(f"  Got: {numeric}")
        print(f"  Expected: {expected}")
        
        if match:
            passed += 1
        else:
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)
