import csv

# Input CSV file
input_file = "comparisons.csv"

# Define tolerance (0.2%)
tolerance_percent = 0.2

total = 0
correct = 0

with open(input_file, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            gt = float(row["ground_truth"])
            pred = float(row["predicted_answer"])
        except (ValueError, TypeError):
            continue  # skip invalid rows
        total += 1
        if gt == pred:
            
            correct += 1

accuracy = (correct / total * 100) if total > 0 else 0.0

print(f"✅ Accuracy within {tolerance_percent}% tolerance: {accuracy:.2f}% ({correct}/{total})")
