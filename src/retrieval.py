# src/retrieval.py
from typing import List, Dict
import json
import re
from pathlib import Path

# -----------------------------
# Load Constitution Data
# -----------------------------
DATA_PATH = Path(__file__).resolve().parents[1] / "Data" / "constitution.json"

if DATA_PATH.exists():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        CONSTITUTION_SECTIONS: List[Dict[str, str]] = json.load(f)
else:
    CONSTITUTION_SECTIONS = []
    print(f"⚠️ Warning: Constitution file not found at {DATA_PATH}")


# -----------------------------
# Retrieval function
# -----------------------------
def retrieve(query: str, top_k: int = 5) -> List[Dict[str, str]]:
    """
    Retrieve top matching sections for a query.
    Returns a list of dicts: {"section": ..., "text": ..., "score": ...}
    """
    query_lower = query.lower()
    scored_sections = []

    for entry in CONSTITUTION_SECTIONS:
        section = entry.get("section", "Unknown Section")
        text = entry.get("text", "")
        text_lower = text.lower()

        # Simple keyword-based scoring
        score = sum(1 for word in query_lower.split() if word in text_lower)

        if score > 0:
            scored_sections.append({"section": section, "text": text, "score": score})

    # Sort by score (descending)
    scored_sections.sort(key=lambda x: x["score"], reverse=True)

    return scored_sections[:top_k]


# -----------------------------
# Answer synthesis
# -----------------------------
def synthesize_answer(query: str, top_k: int = 5) -> str:
    """
    Generate a concise answer from the top retrieved sections.
    """
    top_sections = retrieve(query, top_k)

    if not top_sections:
        return "No relevant information found in the Constitution."

    answers = []
    for entry in top_sections:
        section = entry.get("section", "Unknown Section")
        text = entry.get("text", "")
        clean_text = re.sub(r"\s+", " ", text).strip()
        answers.append(f"{clean_text} (Section: {section})")

    return " ".join(answers)
