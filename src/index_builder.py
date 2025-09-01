# src/index_builder.py

import os
import faiss
import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

# =========================
# Config
# =========================
DATA_PATH = "Data/kenya_constitution.csv"   # Update if your file is different
INDEX_PATH = "Data/faiss_index"
PICKLE_PATH = "Data/chunks.pkl"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# =========================
# Step 1: Load Dataset
# =========================
def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    if "Text" not in df.columns:
        raise ValueError("CSV must contain a 'Text' column with constitution text")
    return df

# =========================
# Step 2: Split into Chunks
# =========================
def split_into_chunks(texts, chunk_size=200, overlap=50):
    chunks = []
    for text in texts:
        words = text.split()
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
    return chunks

# =========================
# Step 3: Embed Chunks
# =========================
def build_index(chunks, model_name=MODEL_NAME):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, show_progress_bar=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return index, embeddings

# =========================
# Main Script
# =========================
if __name__ == "__main__":
    print("Loading data...")
    df = load_data()
    
    print(" Splitting into chunks...")
    chunks = split_into_chunks(df["Text"].astype(str).tolist())
    print(f"Created {len(chunks)} chunks")

    print(" Building FAISS index...")
    index, embeddings = build_index(chunks)

    # Save index
    print(f"Saving index to {INDEX_PATH}...")
    faiss.write_index(index, INDEX_PATH)

    # Save chunks (for later retrieval mapping)
    with open(PICKLE_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print("Indexing complete!")