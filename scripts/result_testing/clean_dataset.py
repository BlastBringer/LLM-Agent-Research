import json
import re

def extract_final_answer(text):
    """Extract numeric value after '####'."""
    match = re.search(r'####\s*([-+]?\d*\.?\d+)', text)
    return float(match.group(1)) if match else None

# --- Preprocess GSM Symbolic Dataset ---
input_file = "gsm_symbolic_main.jsonl"
output_file = "gsm_symbolic_main_cleaned.jsonl"

with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
    for line in fin:
        data = json.loads(line)
        original_output = data.get("output", "")
        answer = extract_final_answer(original_output)

        # Keep original steps as a single string
        solution_text = original_output.strip()

        # Update output & solution_steps
        data["output"] = answer if answer is not None else None
        data["solution_steps"] = solution_text

        json.dump(data, fout)
        fout.write("\n")

print(f"✅ Preprocessing complete. Cleaned file saved as {output_file}")
