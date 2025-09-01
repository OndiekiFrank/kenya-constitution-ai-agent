import os
import faiss
import numpy as np
import pandas as pd
import pickle

# Force sentence-transformers to use Torch only (avoid TensorFlow issues)
os.environ["USE_TF"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from sentence_transformers import SentenceTransformer

# Paths
DATA_PATH = "Data"
INDEX_FILE = os.path.join(DATA_PATH, "faiss_index.bin")
META_FILE = os.path.join(DATA_PATH, "id_to_metadata.pkl")
CSV_FILE = os.path.join(DATA_PATH, "kenya_constitution_prepared.csv")

# --- Load FAISS index ---
if not os.path.exists(INDEX_FILE):
    raise FileNotFoundError(f"FAISS index not found at {INDEX_FILE}")
index = faiss.read_index(INDEX_FILE)

# --- Load metadata mapping ---
id_to_metadata = {}
if os.path.exists(META_FILE):
    with open(META_FILE, "rb") as f:
        id_to_metadata = pickle.load(f)

# --- Load CSV as fallback ---
corpus = None
if os.path.exists(CSV_FILE):
    corpus = pd.read_csv(CSV_FILE)

# --- Load embedding model (CPU only) ---
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")


def retrieve(query: str, k: int = 5):
    """Retrieve top-k passages relevant to the query."""
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    D, I = index.search(q_emb.reshape(1, -1), k)

    results = []
    for idx, score in zip(I[0], D[0]):
        meta = id_to_metadata.get(idx, {})
        if not meta and corpus is not None and idx < len(corpus):
            meta = corpus.iloc[idx].to_dict()
        results.append({"id": int(idx), "score": float(score), "metadata": meta})
    return results


def synthesize_answer(query: str, hits: list):
    """Concatenate retrieved text into a simple answer."""
    texts = []
    for hit in hits:
        text = hit["metadata"].get("Text_English") or hit["metadata"].get("Text_Kiswahili") or str(hit["metadata"])
        texts.append(text)
    context = " ".join(texts)
    return {
        "query": query,
        "answer": f"Based on the Constitution: {context[:800]}..."
    }


def llm_synthesize_answer(query: str, hits: list):
    """Stub for LLM-based synthesis (extend with OpenAI/Anthropic/etc.)."""
    context = " ".join(
        hit["metadata"].get("Text_English") or hit["metadata"].get("Text_Kiswahili") or str(hit["metadata"])
        for hit in hits
    )
    return {
        "query": query,
        "answer": f"(LLM-Augmented) {context[:800]}..."
    }


if __name__ == "__main__":
    # Quick test
    test_query = "What is the role of the president?"
    results = retrieve(test_query, k=3)
    answer = synthesize_answer(test_query, results)
    print(answer)