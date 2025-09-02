import pickle
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path

# Paths
DATA_DIR = Path(_file_).resolve().parents[1] / "Data"
INDEX_PATH = DATA_DIR / "faiss_index.bin"
META_PATH = DATA_DIR / "id_to_metadata.pkl"
CSV_PATH = DATA_DIR / "kenya_constitution_prepared.csv"

# Load model once
MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index
faiss_index = faiss.read_index(str(INDEX_PATH))

# Load metadata
with open(META_PATH, "rb") as f:
    id_to_metadata = pickle.load(f)

# Load full constitution (for optional context synthesis)
constitution_df = pd.read_csv(CSV_PATH)


def embed_text(text: str) -> np.ndarray:
    """Convert text to embeddings."""
    return MODEL.encode([text])


def retrieve(query: str, k: int = 5):
    """Retrieve top-k passages related to the query."""
    query_vec = embed_text(query).astype("float32")
    distances, indices = faiss_index.search(query_vec, k)

    hits = []
    for i, idx in enumerate(indices[0]):
        if idx == -1:  # Skip empty slots
            continue
        metadata = id_to_metadata.get(idx, {"text": "", "section": "Unknown"})
        hits.append({
            "rank": i + 1,
            "score": float(distances[0][i]),
            "section": metadata.get("section"),
            "text": metadata.get("text"),
        })
    return hits


def synthesize_answer(query: str, hits: list) -> str:
    """Naive synthesis: concatenate retrieved text."""
    if not hits:
        return "No relevant information found in the constitution."
    context = " ".join(hit["text"] for hit in hits)
    return f"Based on the constitution, here’s what I found: {context}"


# ---------------------------------------------------
# Run standalone for testing
# ---------------------------------------------------
if _name_ == "_main_":
    test_query = "What is the role of the president?"
    print(f"Query: {test_query}\n")

    results = retrieve(test_query, k=3)
    print("Top results:")
    for r in results:
        print(f"- Section: {r['section']} | Score: {r['score']:.4f}")
        print(f"  Text: {r['text']}\n")

    answer = synthesize_answer(test_query, results)
    print("Synthesized Answer:\n", answer)