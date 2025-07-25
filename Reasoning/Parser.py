import openai
import json
import os
import re
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("API_KEY")  
api_base = "https://openrouter.ai/api/v1"
model_name = "nousresearch/nous-hermes-2-mixtral-8x7b-dpo"

client = openai.OpenAI(api_key=api_key, base_url=api_base)


def create_parser_prompt(problem: str) -> str:
    """
    Creates a few-shot prompt to guide the MoE model more reliably.
    """
    return f"""
You are a precise math-to-JSON parser. Your task is to convert the given math word problem into a structured JSON object.
Do not solve the problem. Only extract the problem's structure.

Follow these rules:
1. Use Python-style syntax for equations (e.g., total = boys + girls).
2. Your response must be a single, valid JSON object and nothing else. Do not add explanations or markdown formatting around the JSON.

---
Here is an example:

Problem: "A farm has 150 animals, consisting of chickens and pigs. If there are 400 legs in total, how many chickens are there?"

JSON Output:
{{
  "problem_type": "system_of_linear_equations",
  "variables": [
    "c: number of chickens",
    "p: number of pigs",
    "total_animals: total number of animals",
    "total_legs: total number of legs"
  ],
  "equations": [
    "total_animals = c + p",
    "total_animals = 150",
    "total_legs = 2 * c + 4 * p",
    "total_legs = 400"
  ],
  "target_variable": "c"
}}
---
Example 2:

Problem: "A company sells notebooks and pens. Each notebook costs ₹50 and each pen costs ₹20. On a certain day, the company sold a total of 120 items and made ₹3,800 in revenue. How many notebooks were sold?"

JSON Output:
{{
  "problem_type": "system_of_linear_equations",
  "variables": [
    "n: number of notebooks",
    "p: number of pens",
    "total_items: total items sold",
    "total_revenue: total revenue in rupees"
  ],
  "equations": [
    "total_items = n + p",
    "total_items = 120",
    "total_revenue = 50 * n + 20 * p",
    "total_revenue = 3800"
  ],
  "target_variable": "n"
}}

---
Now, parse the following problem:

Problem: "{problem}"

JSON Output:
"""


def call_parser(problem: str):
    prompt = create_parser_prompt(problem)
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=1024
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ API error: {e}")
        return None


def clean_and_parse_json(raw_output: str):
    match = re.search(r'\{.*\}', raw_output, re.DOTALL)
    if not match:
        print("❌ No JSON block found in output.")
        return None
    json_str = match.group(0)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print("❌ JSON Decode Error:", e)
        return None


if __name__ == "__main__":
    user_input = input("Enter a math word problem: ").strip()
    if not user_input:
        print("No input provided.")
    else:
        raw_output = call_parser(user_input)
        if raw_output:
            parsed = clean_and_parse_json(raw_output)
            if parsed:
                print("\n✅ Parsed JSON Output:")
                print(json.dumps(parsed, indent=2))
                try:
                    with open("parsed_output.jsonl", "a") as f:
                        f.write(json.dumps(parsed) + "\n")
                    print("📝 Output appended to parsed_output.jsonl")
                except IOError as e:
                    print("❌ File write error:", e)
            else:
                print("⚠️ Raw output (not parsed):\n", raw_output)
