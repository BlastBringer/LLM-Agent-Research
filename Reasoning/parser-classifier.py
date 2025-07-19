import openai
import json
import re

# --- API Setup ---
api_key = ""
api_base = "https://openrouter.ai/api/v1"

client = openai.OpenAI(api_key=api_key, base_url=api_base)

# --- Unified Parsing Function ---
def call_parser(prompt):
    structured_prompt = f"""
You are a parser for math and reasoning problems. Your job is to convert word problems into a structured JSON format that captures their core structure.

Use the following fields:

- "type": a short label for the domain (e.g., "percentage word problem", "number arrangement", "logical deduction").
- "inputs": a list of known quantities, objects, sequences, or constraints. Each should be a dictionary with a "label" and "value" key.
- "variables": a list of all symbols, numbers, entities, or values that appear in reasoning steps.
- "representations": symbolic expressions, stepwise transformations, or intermediate equations used to model or analyze the problem.
- "reasoning_strategy": a concise description of the method used to solve the problem, in natural language.
- "target": the final expression, answer, or output being solved for (can be a symbol, number, or sequence).
- "options" (optional): include this field only if multiple-choice options (A, B, C, etc.) are present in the prompt. This should be a list of cleaned answer options.

⚠️ Do NOT solve the problem.
⚠️ Do NOT add markdown, explanations, or extra formatting.
⚠️ Return ONLY a clean, valid JSON object.

Problem:
\"\"\"{prompt}\"\"\"
"""

    response = client.chat.completions.create(
        model="google/gemini-2.0-flash-001",
        messages=[
            {"role": "system", "content": "You are a parser that converts math and logic problems into structured symbolic reasoning."},
            {"role": "user", "content": structured_prompt}
        ],
        temperature=0.2,
        max_tokens=800
    )

    return response.choices[0].message.content.strip()

# --- Main Driver ---
if __name__ == "__main__":
    user_input = input("Enter a math or reasoning problem: ").strip()
    if not user_input:
        print("No input given.")
    else:
        try:
            # Call the LLM parser
            raw_output = call_parser(user_input)

            # Remove markdown code fences if present
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_output.strip())

            # Try parsing into JSON
            parsed_json = json.loads(cleaned)

            print("\n🧠 Parsed JSON Output:")
            print(json.dumps(parsed_json, indent=2))

            # Optionally append to dataset
            with open("parsed_output.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(parsed_json) + "\n")

        except Exception as e:
            print("❌ Failed to parse LLM output as JSON.")
            print("Raw output:\n", raw_output)
            print("Error:", e)
