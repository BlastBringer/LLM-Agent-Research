from datasets import load_dataset
import json
import os

def preprocess_gsm8k(output_dir="gsm8k_preprocessed"):
    os.makedirs(output_dir, exist_ok=True)

    print("📥 Loading GSM8K dataset from Hugging Face...")
    dataset = load_dataset("openai/gsm8k", "main")

    # --------------------
    # TRAIN SPLIT
    # --------------------
    print("🧮 Processing train split...")
    train_data = []
    for item in dataset["train"]:
        question = item["question"].strip()
        answer = item["answer"].strip()

        if "####" in answer:
            reasoning, final_answer = answer.split("####", 1)
            reasoning = reasoning.strip()
            final_answer = final_answer.strip()
        else:
            reasoning = answer
            final_answer = ""

        train_data.append({
            "input": question,
            "output": final_answer,
            "solution_steps": reasoning
        })

    # --------------------
    # TEST SPLIT
    # --------------------
    print("🧪 Processing test split...")
    test_data = []
    for item in dataset["test"]:
        question = item["question"].strip()
        answer = item["answer"].strip()

        if "####" in answer:
            _, final_answer = answer.split("####", 1)
            final_answer = final_answer.strip()
        else:
            final_answer = ""

        test_data.append({
            "input": question,
            "output": final_answer
        })

    # --------------------
    # SAVE BOTH SPLITS
    # --------------------
    train_path = os.path.join(output_dir, "gsm8k_train_preprocessed.jsonl")
    test_path = os.path.join(output_dir, "gsm8k_test_preprocessed.jsonl")

    with open(train_path, "w", encoding="utf-8") as f_train:
        for entry in train_data:
            f_train.write(json.dumps(entry, ensure_ascii=False) + "\n")

    with open(test_path, "w", encoding="utf-8") as f_test:
        for entry in test_data:
            f_test.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"✅ Done! Saved {len(train_data)} train samples to {train_path}")
    print(f"✅ Done! Saved {len(test_data)} test samples to {test_path}")

if __name__ == "__main__":
    preprocess_gsm8k()
