# save as scripts/hf_to_jsonl_gsm_symbolic.py (or run in a notebook)
from datasets import load_dataset
import json

def to_solution_steps(answer_text: str):
    # Split per-line; keep everything except the final "#### ..."
    lines = [ln.strip() for ln in (answer_text or "").splitlines() if ln.strip()]
    if not lines:
        return []
    # Drop the last line if it starts with ####
    if lines[-1].lstrip().startswith("####"):
        return lines[:-1]
    return lines

def export_split(ds, out_path: str):
    with open(out_path, "w") as f:
        for ex in ds:
            rec = {
                # Your pipeline expects these in dataset mode
                "input": ex["question"],
                "output": ex["answer"],  # keep full text; parser extracts final number from "#### 20"
                # Optional: handy for train-mode fallback when both models miss
                "solution_steps": to_solution_steps(ex["answer"]),
                # Keep provenance if helpful
                "id": ex.get("id"),
                "original_id": ex.get("original_id"),
                "canary": ex.get("canary"),
            }
            f.write(json.dumps(rec) + "\n")

if __name__ == "__main__":
    # Load GSM-Symbolic main
    ds = load_dataset("apple/GSM-Symbolic", name="main")
    # Choose the split you want to test; many use 'test' for reporting
    split_name = "test" if "test" in ds else "validation" if "validation" in ds else "train"
    export_split(ds[split_name], "gsm_symbolic_main.jsonl")
    print(f"✅ Wrote gsm_symbolic_main.jsonl from split: {split_name}")