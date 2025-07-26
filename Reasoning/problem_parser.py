import openai
import json
import os
import re
from dotenv import load_dotenv

load_dotenv()

class ProblemParser:
    """
    Parses math problems into structured JSON format.
    """
    def __init__(self):
        self.api_key = os.getenv("API_KEY")
        self.api_base = "https://openrouter.ai/api/v1"
        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.api_base)
        print("📝 Problem Parser initialized.")

    def create_parser_prompt(self, problem: str) -> str:
        """Creates a structured prompt for parsing."""
        return f"""
You are a precise math-to-JSON parser. Convert the given math word problem into a structured JSON object.
Do not solve the problem. Only extract the problem's structure.

Follow these rules:
1. Use Python-style syntax for equations (e.g., total = boys + girls).
2. Your response must be a single, valid JSON object and nothing else.

Here are examples:

Problem: "A farm has 150 animals, consisting of chickens and pigs. If there are 400 legs in total, how many chickens are there?"

JSON Output:
{{
  "problem_type": "system_of_linear_equations",
  "variables": {{
    "c": "number of chickens",
    "p": "number of pigs"
  }},
  "equations": [
    "c + p = 150",
    "2 * c + 4 * p = 400"
  ],
  "target_variable": "c"
}}

Problem: "Calculate 25 * 4 + 18 / 3"

JSON Output:
{{
  "problem_type": "simple_arithmetic",
  "expression_to_evaluate": "25 * 4 + 18 / 3"
}}

Now, parse the following problem:

Problem: "{problem}"

JSON Output:
"""

    def parse(self, problem: str) -> dict:
        """
        Parses a math problem into structured format.
        
        Args:
            problem: The math problem text to parse.
            
        Returns:
            A dictionary containing the parsed problem structure.
        """
        print(f"📝 Parsing problem: {problem[:50]}...")
        
        prompt = self.create_parser_prompt(problem)
        
        try:
            response = self.client.chat.completions.create(
                model="google/gemini-2.0-flash-001",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1024
            )
            
            raw_output = response.choices[0].message.content.strip()
            
            # Clean markdown formatting if present
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_output.strip())
            
            # Try to parse JSON
            parsed_data = json.loads(cleaned)
            print("✅ Successfully parsed problem structure.")
            return parsed_data
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing error: {e}")
            print(f"Raw output: {raw_output}")
            return {"error": "Failed to parse JSON", "raw_output": raw_output}
        except Exception as e:
            print(f"❌ API error: {e}")
            return {"error": f"API error: {str(e)}"}

    def clean_and_parse_json(self, raw_output: str) -> dict:
        """
        Extracts and parses JSON from raw output.
        
        Args:
            raw_output: Raw text that should contain JSON.
            
        Returns:
            Parsed JSON dictionary or None if parsing fails.
        """
        # Look for JSON block in the output
        match = re.search(r'\{.*\}', raw_output, re.DOTALL)
        if not match:
            print("❌ No JSON block found in output.")
            return None
            
        json_str = match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"❌ JSON Decode Error: {e}")
            return None

# Example usage
if __name__ == "__main__":
    parser = ProblemParser()
    
    # Test problems
    test_problems = [
        "A company sells notebooks and pens. Each notebook costs ₹50 and each pen costs ₹20. On a certain day, the company sold a total of 120 items and made ₹3,800 in revenue. How many notebooks were sold?",
        "Calculate 25 * 4 + 18 / 3",
        "If x + y = 10 and x - y = 2, find x and y"
    ]
    
    for problem in test_problems:
        print(f"\n--- Parsing Problem ---")
        print(f"Problem: {problem}")
        result = parser.parse(problem)
        print("Parsed Result:")
        print(json.dumps(result, indent=2))