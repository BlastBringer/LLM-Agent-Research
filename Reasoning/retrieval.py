import json
import os
from tqdm import tqdm
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# ====== 1. Load the dataset ======
def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

# Path to GSM dataset (update this)
DATA_PATH = "train.jsonl"

gsm_data = load_jsonl(DATA_PATH)
print(f"Loaded {len(gsm_data)} examples from {DATA_PATH}")

# ====== 2. Prepare text for embedding ======
# Combine question + solution or just question
texts = [f"Question: {d['question']}\nAnswer: {d.get('answer', '')}" for d in gsm_data]

# ====== 3. Generate embeddings ======
print("Generating embeddings...")

# Use a small, fast embedding model (can replace with OpenAI or other)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

# ====== 4. Build FAISS index ======
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)  # cosine similarity (normalized vectors)
index.add(embeddings)
print(f"FAISS index built with {index.ntotal} vectors.")

# ====== 5. Save the index + metadata ======
faiss.write_index(index, "gsm_faiss.index")

with open("gsm_metadata.pkl", "wb") as f:
    pickle.dump(gsm_data, f)

print("✅ Saved FAISS index and metadata.")

# ====== 6. Example: Query the vector DB ======
def query_gsm_db(query, top_k=3):
    q_emb = model.encode([query], normalize_embeddings=True)
    scores, indices = index.search(q_emb, top_k)
    results = []
    for idx, score in zip(indices[0], scores[0]):
        item = gsm_data[idx]
        results.append({
            "question": item["question"],
            "answer": item["answer"],
            "score": float(score)
        })
    return results

# Example usage
if __name__ == "__main__":
    q = "A farmer has 3 cows and buys 5 more. How many does he have?"
    results = query_gsm_db(q)
    for r in results:
        print(f"\nScore: {r['score']:.4f}\nQ: {r['question']}\nA: {r['answer']}")
