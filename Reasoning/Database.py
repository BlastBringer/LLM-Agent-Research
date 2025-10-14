# Install required packages if you haven't already:
# pip install datasets langchain sentence-transformers faiss-cpu

from datasets import load_dataset
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# -----------------------------
# Step 1: Load the GSM8K dataset
# -----------------------------
dataset = load_dataset("openai/gsm8k", "main")
train_data = dataset['train']

# Extract word problems
word_problems = [item['question'] for item in train_data]

# -----------------------------
# Step 2: Preprocess the word problems
# -----------------------------
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
documents = []

for problem in word_problems:
    chunks = text_splitter.split_text(problem)
    for chunk in chunks:
        documents.append(Document(page_content=chunk))

# -----------------------------
# Step 3: Generate embeddings using Hugging Face
# -----------------------------
# You can choose any sentence-transformers model, e.g., 'sentence-transformers/all-MiniLM-L6-v2'
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# -----------------------------
# Step 4: Create vector store (FAISS) and add documents
# -----------------------------
vector_store = FAISS.from_documents(documents, embedding_model)

# Optional: Save the FAISS vector store locally
vector_store.save_local("word_problems_faiss_hf")

print("✅ Word problems have been embedded using Hugging Face and stored in FAISS!")
