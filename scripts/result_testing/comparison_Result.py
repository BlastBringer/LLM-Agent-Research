import json
import csv

# File paths (change these to your actual paths if needed)
main_file = "gsm_symbolic_batch1.jsonl"
test_output_file = "test_output.jsonl"
output_csv = "comparisons.csv"

# Read both JSONL files
with open(main_file, "r") as f1, open(test_output_file, "r") as f2:
    main_data = [json.loads(line) for line in f1]
    test_data = [json.loads(line) for line in f2]

# Ensure they have the same length
if len(main_data) != len(test_data):
    print(f"Warning: Length mismatch — main={len(main_data)} test={len(test_data)}")
    
# Combine output and answer
rows = []
for i in range(min(len(main_data), len(test_data))):
    # Get output from main file (can be string or number)
    ground_truth = main_data[i].get("output", "")
    if not isinstance(ground_truth, str):
        ground_truth = str(ground_truth)
    ground_truth = ground_truth.strip()
    
    # Get answer from test output (can be string or number)
    predicted_answer = test_data[i].get("answer", "")
    if not isinstance(predicted_answer, str):
        predicted_answer = str(predicted_answer)
    predicted_answer = predicted_answer.strip()
    
    rows.append({"ground_truth": ground_truth, "predicted_answer": predicted_answer})

# Write to CSV
with open(output_csv, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["ground_truth", "predicted_answer"])
    writer.writeheader()
    writer.writerows(rows)

print(f"✅ CSV saved to {output_csv} with {len(rows)} rows.")
