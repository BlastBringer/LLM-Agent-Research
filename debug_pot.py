#!/usr/bin/env python3
"""Debug PoT extraction issues"""

import json
import re
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

POT_EXAMPLES = """
Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?

trees_start = 15
trees_end = 21
trees_planted = trees_end - trees_start
answer = trees_planted

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?

cars_start = 3
cars_arrive = 2
total_cars = cars_start + cars_arrive
answer = total_cars

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?

leah = 32
sister = 42
total = leah + sister
ate = 35
left = total - ate
answer = left
"""

def extract_code_from_pot(response: str):
    """Extract Python code from PoT response"""
    print(f"\n{'='*60}")
    print("RAW RESPONSE:")
    print(response)
    print(f"{'='*60}\n")
    
    # Remove markdown code blocks
    response = re.sub(r'```python\s*', '', response)
    response = re.sub(r'```\s*', '', response)
    
    # Find lines with assignments
    lines = response.strip().split('\n')
    code_lines = []
    
    for line in lines:
        stripped = line.strip()
        # Skip empty lines and comments at the start
        if not code_lines and (not stripped or stripped.startswith('#')):
            continue
        # Add assignment lines and comments
        if '=' in stripped or stripped.startswith('#'):
            code_lines.append(line)
        # Stop at long text (likely explanation)
        elif stripped and len(stripped.split()) > 6 and not any(c in stripped for c in ['=', '+', '-', '*', '/', '(', ')']):
            break
    
    code = '\n'.join(code_lines).strip()
    
    print("EXTRACTED CODE:")
    print(code)
    print()
    
    # Ensure 'answer' variable exists
    if code and 'answer' not in code.lower():
        var_matches = re.findall(r'(\w+)\s*=', code)
        if var_matches:
            code += f"\nanswer = {var_matches[-1]}"
            print("ADDED answer variable:")
            print(code)
            print()
    
    return code if code else None


def execute_code(code: str):
    """Execute Python code and return answer"""
    try:
        namespace = {}
        exec(code, namespace)
        
        for var in ['answer', 'Answer', 'result', 'Result']:
            if var in namespace:
                return float(namespace[var])
        
        return None
    except Exception as e:
        print(f"EXECUTION ERROR: {e}")
        return None


def test_pot_with_model():
    """Test actual PoT generation with the model"""
    student_llm = ChatOpenAI(
        model="meta-llama/llama-3.2-3b-instruct",
        base_url="https://openrouter.ai/api/v1",
        temperature=0.7,
        max_tokens=512,
        request_timeout=90,
    )
    
    test_questions = [
        "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
        "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
    ]
    
    for question in test_questions:
        print(f"\n{'#'*60}")
        print(f"QUESTION: {question}")
        print(f"{'#'*60}")
        
        prompt = f"{POT_EXAMPLES}\nQ: {question}\n\n# Python code to solve\n"
        
        print("\nCALLING MODEL...")
        response = student_llm.invoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        code = extract_code_from_pot(response_text)
        
        if code:
            print(f"FINAL CODE TO EXECUTE:")
            print(code)
            print()
            answer = execute_code(code)
            print(f"ANSWER: {answer}")
        else:
            print("NO CODE EXTRACTED")


if __name__ == "__main__":
    test_pot_with_model()
