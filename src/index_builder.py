# src/index_builder.py
import pandas as pd
import numpy as np
import faiss
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# -----------------------------
# Paths
# -----------------------------
DATA_DIR = Path(__file__).resolve().parents[1] / "Data"
CSV_PATH = DATA_DIR / "Kenya_Constitution_Optimized.csv"
INDEX_PATH = DATA_DIR / "faiss_index.bin"
META_PATH = DATA_DIR / "id_to_metadata.pkl"

# -----------------------------
# Load CSV
# -----------------------------
if not CSV_PATH.exists():
    raise FileNotFoundError(f"CSV file not found at {CSV_PATH}")

df = pd.read_csv(CSV_PATH)
print(f"CSV loaded: {len(df)} rows")

# -----------------------------
# Load model
# -----------------------------
MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# Encode text
# -----------------------------
print("Encoding text...")
embeddings = []
for text in tqdm(df["text"].astype(str), desc="Encoding"):
    emb = MODEL.encode(text)
    embeddings.append(emb)
embeddings = np.array(embeddings).astype("float32")
print("Embeddings shape:", embeddings.shape)

# -----------------------------
# Build FAISS index
# -----------------------------
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)
print(f"FAISS index contains {index.ntotal} vectors")

# -----------------------------
# Save index and metadata
# -----------------------------
faiss.write_index(index, str(INDEX_PATH))
print(f"FAISS index saved to {INDEX_PATH}")

id_to_metadata = {i: {"section": df.loc[i, "section"], "text": df.loc[i, "text"]} for i in range(len(df))}
with open(META_PATH, "wb") as f:
    pickle.dump(id_to_metadata, f)
print(f"Metadata saved to {META_PATH}")
