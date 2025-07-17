import openai
import json
import re  # <-- import for regex cleanup

# Replace this with your actual API key
api_key = ""
api_base = "https://openrouter.ai/api/v1"

client = openai.OpenAI(api_key=api_key, base_url=api_base)

def call_parser(prompt):
    structured_prompt = f"""
You are a math parser. Your task is to convert the given math word problem into a structured JSON object. 
Do not solve the problem. Just extract:
- type of problem
- list of variables used
- list of symbolic equations (as strings)
- the target variable to solve for (as string)

Use only Python-style syntax for equations (e.g., 0.25 * total_students = scholarship_girls + scholarship_boys). 
Ensure your response is valid JSON. Here is the problem:

"{prompt}"

Respond ONLY with the JSON object. Do not add explanations or markdown.
"""
    response = client.chat.completions.create(
        model="google/gemini-2.0-flash-001",
        messages=[
            {"role": "system", "content": "You are a math parser that extracts symbolic structure from math problems."},
            {"role": "user", "content": structured_prompt}
        ],
        temperature=0.2,
        max_tokens=512
    )
    return response.choices[0].message.content.strip()

# --- MAIN ---
if _name_ == "_main_":
    user_input = input("Enter a math word problem: ").strip()
    if not user_input:
        print("No input given.")
    else:
        try:
            raw_output = call_parser(user_input)

            # 🧹 Strip markdown-style code fences like json ... 
            cleaned = re.sub(r"^(?:json)?\s*|\s*$", "", raw_output.strip())

            # Try parsing JSON
            parsed_json = json.loads(cleaned)
            print("\nParsed JSON Output:")
            print(json.dumps(parsed_json, indent=2))

            # Optionally write to a .jsonl file
            with open("parsed_output.jsonl", "a") as f:
                f.write(json.dumps(parsed_json) + "\n")

        except Exception as e:
            print("❌ Failed to parse LLM output as JSON.")
            print("Raw output:\n", raw_output)
            print("Error:", e)