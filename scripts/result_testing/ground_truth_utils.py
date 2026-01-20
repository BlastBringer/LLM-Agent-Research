#!/usr/bin/env python3
"""
Ground Truth Extraction Utilities
==================================

Extract answers from various dataset formats:
- MATH dataset: \boxed{} format with LaTeX
- GSM8K: #### answer format
- Custom: direct numeric fields
- Complex LaTeX: sqrt, fractions, infinity, complex numbers

Supports:
- Numbers: 150, 3.14
- Fractions: \frac{9}{7}, \dfrac{-1}{8}
- Sqrt: \sqrt{2}, \frac{\sqrt{7}}{14}
- Complex: i, -11+27i
- Infinity: \infty, -\infty
- Intervals: [-3,2], (0,\infty)
- Lists: 4,6,14,15
- Expressions: Returns None (non-numeric)
"""

import re
import math
from typing import Optional, Tuple, Union, List, Dict

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
    Extract numeric value from answer string, including LaTeX expressions.
    
    Examples:
        "150" -> 150.0
        "3.5 miles" -> 3.5
        "$42.50" -> 42.5
        "\\frac{9}{7}" -> 1.2857142857142858
        "\\dfrac{-1}{8}" -> -0.125
        "\\sqrt{2}" -> 1.4142135623730951
        "\\frac{\\sqrt{7}}{14}" -> 0.18898...
        "i" -> None (complex - not supported for comparison)
        "-11+27i" -> None (complex)
        "\\infty" -> float('inf')
        "-\\infty" -> float('-inf')
        "[-3,2]" -> None (interval - not supported)
        "7(x+3)(x-3)" -> None (expression - not supported)
    
    Returns:
        float if numeric, None if non-numeric (complex/interval/expression)
    """
    if not text:
        return None
    
    original_text = text
    
    # Remove dollar signs and currency symbols (but keep negative signs)
    cleaned = text.replace('$', '').replace('€', '').replace('£', '').strip()
    
    # Check for non-numeric patterns first (return None early)
    # 1. Complex numbers: "i", "-11+27i", "3-4i"
    if re.search(r'\bi\b', cleaned) or re.search(r'[+-]\s*\d*i', cleaned):
        return None
    
    # 2. Intervals: "[-3,2]", "[0,\\infty)", "(-\\infty,0)"
    if re.match(r'[\[\(].*,.*[\]\)]', cleaned):
        return None
    
    # 3. Algebraic expressions with variables: "x", "2x^9", "7(x+3)"
    # Check for variable names (letters except 'e' for scientific notation)
    if re.search(r'\b[a-df-hj-z]\b', cleaned.lower()) or re.search(r'\d+[a-z]', cleaned.lower()):
        return None
    
    # 4. Multiple numbers (lists/tuples): "4,6,14,15" or "12, 10, 6"
    comma_parts = [p.strip() for p in cleaned.split(',')]
    if len(comma_parts) > 1:
        # Check if all parts are numbers
        try:
            numbers = [float(p) for p in comma_parts if p]
            if len(numbers) > 1:
                return None  # It's a list, not a single number
        except ValueError:
            return None
    
    # Now handle numeric LaTeX patterns
    
    # 5. Infinity: \\infty or -\\infty
    if r'\infty' in cleaned:
        if cleaned.startswith('-') or cleaned.startswith('(-'):
            return float('-inf')
        else:
            return float('inf')
    
    # 6. Square root with fraction: \\frac{\\sqrt{n}}{d} or \\frac{n}{\\sqrt{d}}
    sqrt_frac_pattern = r'(-?)\\d?frac\{\\sqrt\{([^}]+)\}\}\{([^}]+)\}'
    match = re.search(sqrt_frac_pattern, cleaned)
    if match:
        try:
            sign = -1 if match.group(1) == '-' else 1
            numerator = math.sqrt(float(match.group(2)))
            denominator = float(match.group(3))
            if denominator != 0:
                return sign * (numerator / denominator)
        except (ValueError, ZeroDivisionError):
            pass
    
    # Pattern: \\frac{n}{\\sqrt{d}}
    sqrt_denom_pattern = r'(-?)\\d?frac\{([^}]+)\}\{\\sqrt\{([^}]+)\}\}'
    match = re.search(sqrt_denom_pattern, cleaned)
    if match:
        try:
            sign = -1 if match.group(1) == '-' else 1
            numerator = float(match.group(2))
            denominator = math.sqrt(float(match.group(3)))
            if denominator != 0:
                return sign * (numerator / denominator)
        except (ValueError, ZeroDivisionError):
            pass
    
    # 7. Square root: \\sqrt{n}
    sqrt_pattern = r'\\sqrt\{([^}]+)\}'
    match = re.search(sqrt_pattern, cleaned)
    if match:
        try:
            value = float(match.group(1))
            return math.sqrt(value)
        except ValueError:
            pass
    
    # 8. LaTeX fractions: \\frac{numerator}{denominator} or \\dfrac{numerator}{denominator}
    # Pattern matches: -\\frac{num}{den} or \\frac{-num}{den} or \\dfrac{num}{den}
    frac_pattern = r'(-?)\\d?frac\{(-?\d+\.?\d*)\}\{(-?\d+\.?\d*)\}'
    match = re.search(frac_pattern, cleaned)
    if match:
        try:
            # Handle sign before \\frac
            sign = -1 if match.group(1) == '-' else 1
            numerator = float(match.group(2))
            denominator = float(match.group(3))
            if denominator != 0:
                return sign * (numerator / denominator)
        except (ValueError, ZeroDivisionError):
            pass
    
    # 9. Simple fractions without LaTeX: "9/7" or "-1/8"
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
    
    # 10. Regular numbers (possibly with units after)
    # Extract first number-like token
    number_pattern = r'(-?\d+\.?\d*)'
    match = re.search(number_pattern, cleaned)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    
    return None

def parse_interval_expression(text: str) -> Optional[List[Dict[str, Union[float, bool]]]]:
    """
    Parse LaTeX/ASCII interval expressions possibly joined by \cup.
    Returns a list of intervals, each as:
      {"start": float('-inf')|float|float('inf'), "end": same,
       "include_start": bool, "include_end": bool}

    Examples:
      "(-\\infty, 3]" -> [{start:-inf, end:3.0, include_start:False, include_end:True}]
      "[-3,2) \\cup (4,5]" -> two intervals
    """
    if not text:
        return None

    s = text.strip()
    # Normalize LaTeX spacing
    s = s.replace('\\,', ' ').replace('\\;', ' ').replace('\\ ', ' ')
    # Split by union
    parts = [p.strip() for p in re.split(r"\\cup|\s+\\cup\s+", s) if p.strip()]

    def to_num(tok: str) -> Optional[float]:
        tok = tok.strip()
        # Handle \infty
        if '\\infty' in tok or '∞' in tok:
            return float('inf') if not tok.strip().startswith('-') else float('-inf')
        # Use numeric extractor for general cases (supports frac/sqrt)
        val = extract_numeric_answer(tok)
        return val

    intervals: List[Dict[str, Union[float, bool]]] = []
    for part in parts:
        m = re.match(r"^\s*([\[\(])\s*(.*?)\s*,\s*(.*?)\s*([\]\)])\s*$", part)
        if not m:
            return None  # Unrecognized interval part
        lbr, a_str, b_str, rbr = m.groups()
        a = to_num(a_str)
        b = to_num(b_str)
        if a is None or b is None:
            return None
        include_start = (lbr == '[')
        include_end = (rbr == ']')
        # Normalize order if needed
        start = a
        end = b
        intervals.append({
            'start': float(start),
            'end': float(end),
            'include_start': include_start,
            'include_end': include_end
        })
    return intervals if intervals else None

def extract_interval_from_problem(problem_data: dict) -> Optional[List[Dict[str, Union[float, bool]]]]:
    """
    Extract an interval/set-style ground truth from problem data, if present.
    Looks for an 'output' string (top-level or in metadata) that parses as intervals.
    """
    raw_answer = None
    # Look top-level, then metadata
    if isinstance(problem_data, dict):
        raw_answer = problem_data.get('output') or problem_data.get('answer') or None
        if raw_answer is None:
            meta = problem_data.get('metadata') or {}
            if isinstance(meta, dict):
                raw_answer = meta.get('output') or meta.get('answer') or None
    if not raw_answer:
        return None
    return parse_interval_expression(str(raw_answer))

def extract_ground_truth_from_problem(problem_data: dict) -> Tuple[Optional[float], Optional[str], Optional[str]]:
    """
    Extract ground truth answer from problem data.
    
    This function supports two layouts:
    - Top-level keys (e.g., {'output': '...'} )
    - Nested metadata (e.g., {'metadata': {'input': '...', 'output': '...'}})
    
    Args:
        problem_data: Dict with problem information
    
    Returns:
        (numeric_answer, raw_answer_string, unit)
    """
    raw_answer = None
    
    # Helper to search fields in a mapping
    def find_answer_in(mapping: dict) -> Optional[str]:
        for field in ['output', 'answer', 'final_answer', 'solution', 'ground_truth']:
            if field in mapping and mapping[field] is not None:
                return mapping[field]
        return None
    
    # 1) Look at top-level first
    raw_answer = find_answer_in(problem_data)
    
    # 2) Fallback: look inside metadata blob
    if raw_answer is None:
        meta = problem_data.get('metadata') or {}
        if isinstance(meta, dict):
            raw_answer = find_answer_in(meta)
    
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
    # Test cases covering all LaTeX patterns
    test_cases = [
        # Basic numbers
        {"output": "\\boxed{150}", "expected": 150.0, "desc": "Boxed integer"},
        {"output": "answer is $42.50", "expected": 42.5, "desc": "Dollar amount"},
        {"output": "3.14159", "expected": 3.14159, "desc": "Decimal"},
        {"output": "-25", "expected": -25.0, "desc": "Negative integer"},
        
        # Fractions
        {"output": "\\dfrac{9}{7}", "expected": 9/7, "desc": "dfrac"},
        {"output": "\\boxed{-\\frac{1}{8}}", "expected": -1/8, "desc": "Negative frac in boxed"},
        {"output": "\\frac{23}{7}", "expected": 23/7, "desc": "Basic frac"},
        {"output": "9/7", "expected": 9/7, "desc": "Simple fraction"},
        {"output": "-1/8", "expected": -1/8, "desc": "Negative simple fraction"},
        {"output": "\\boxed{\\dfrac{9}{7}}", "expected": 9/7, "desc": "dfrac in boxed"},
        {"output": "\\boxed{\\frac{7}{2}}", "expected": 3.5, "desc": "frac in boxed"},
        
        # Square roots
        {"output": "\\sqrt{4}", "expected": 2.0, "desc": "Perfect square root"},
        {"output": "\\sqrt{2}", "expected": math.sqrt(2), "desc": "Square root of 2"},
        {"output": "\\boxed{\\frac{\\sqrt{7}}{14}}", "expected": math.sqrt(7)/14, "desc": "Sqrt in numerator"},
        
        # Infinity
        {"output": "\\infty", "expected": float('inf'), "desc": "Positive infinity"},
        {"output": "-\\infty", "expected": float('-inf'), "desc": "Negative infinity"},
        {"output": "[0,\\infty)", "expected": None, "desc": "Interval with infinity (non-numeric)"},
        
        # Non-numeric (should return None)
        {"output": "i", "expected": None, "desc": "Imaginary unit"},
        {"output": "-11+27i", "expected": None, "desc": "Complex number"},
        {"output": "[-3,2]", "expected": None, "desc": "Interval notation"},
        {"output": "(-\\infty,0)", "expected": None, "desc": "Interval to zero"},
        {"output": "x \\in [-2,7]", "expected": None, "desc": "Set notation"},
        {"output": "7(x+3)(x-3)", "expected": None, "desc": "Algebraic expression"},
        {"output": "2x^9 - 8x^7", "expected": None, "desc": "Polynomial"},
        {"output": "4,6,14,15", "expected": None, "desc": "List of numbers"},
        {"output": "12, 10, 6", "expected": None, "desc": "List with spaces"},
    ]
    
    print("=" * 80)
    print("TESTING GROUND TRUTH EXTRACTION (All LaTeX Patterns)")
    print("=" * 80)
    
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
        elif math.isinf(expected) and math.isinf(numeric):
            match = (expected > 0) == (numeric > 0)  # Same sign of infinity
        else:
            match = abs(numeric - expected) < 1e-9
        
        status = "✅ PASS" if match else "❌ FAIL"
        print(f"\nTest {i}: {status}")
        print(f"  Description: {case['desc']}")
        print(f"  Input: {case['output']}")
        print(f"  Got: {numeric}")
        print(f"  Expected: {expected}")
        
        if match:
            passed += 1
        else:
            failed += 1
    
    print("\n" + "=" * 80)
    print(f"Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("✅ ALL TESTS PASSED!")
    print("=" * 80)
