# src/index_builder.py
import os
os.environ["USE_TF"] = "0"   # Disable TensorFlow in HuggingFace

import faiss
import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer

# ------------------------
# CONFIG
# ------------------------
DATA_PATH = "Data\kenya_constitution_structured.csv"   # <-- change if your CSV path is different
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # Lightweight, PyTorch-only
INDEX_PATH = "Data/faiss_index.bin"
MAPPING_PATH = "Data/id_to_metadata.pkl"

# ------------------------
# LOAD DATA
# ------------------------
def load_data():
    print("🔹 Loading data...")
    df = pd.read_csv(DATA_PATH)

    # Normalize column names (case-insensitive)
    df.columns = [c.strip().lower() for c in df.columns]

    # Map your dataset’s actual columns to expected ones
    column_map = {
        "article/section": "section",
        "text_english": "english_text",
        "text_kiswahili": "kiswahili_text"
    }
    df = df.rename(columns=column_map)

    if "english_text" not in df.columns:
        raise ValueError(f"CSV must contain 'Text_English'. Found: {df.columns}")

    return df, "english_text"

# ------------------------
# BUILD INDEX
# ------------------------
def build_index(df, text_col):
    print(f"🔹 Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print("🔹 Encoding sentences...")
    embeddings = model.encode(
        df[text_col].astype(str).tolist(),
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    print("🔹 Creating FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product = cosine similarity (after normalization)
    index.add(embeddings)

    print("✅ Index built successfully.")
    return index, embeddings

# ------------------------
# SAVE INDEX + METADATA
# ------------------------
def save_index(index, df):
    print(f"🔹 Saving FAISS index → {INDEX_PATH}")
    faiss.write_index(index, INDEX_PATH)

    print(f"🔹 Saving ID → metadata mapping → {MAPPING_PATH}")
    id_to_metadata = {
        i: {
            "section": row.get("section", ""),
            "english_text": row.get("english_text", ""),
            "kiswahili_text": row.get("kiswahili_text", "")
        }
        for i, row in df.iterrows()
    }

    with open(MAPPING_PATH, "wb") as f:
        pickle.dump(id_to_metadata, f)

    print("✅ Index and metadata saved.")

# ------------------------
# MAIN
# ------------------------
if __name__ == "__main__":
    df, text_col = load_data()
    index, embeddings = build_index(df, text_col)
    save_index(index, df)