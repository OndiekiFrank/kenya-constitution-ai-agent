# src/retrieval.py
from typing import List, Tuple
import re

# Dummy database of sections for demonstration. Replace with real embeddings/search later.
CONSTITUTION_SECTIONS = [
    {
        "section": "Article 136",
        "text": (
            "The President shall be elected by registered voters in a national election. "
            "A decision of the President in the performance of any function shall be in writing "
            "and bear the seal and signature of the President."
        )
    },
    {
        "section": "Article 147",
        "text": (
            "The Deputy President shall be the principal assistant of the President and shall "
            "deputise for the President in the execution of the President's functions. "
            "The President nominates or appoints judges, Cabinet Secretaries, ambassadors, "
            "and other public officers. The President may confer honours and exercise the power of mercy."
        )
    },
    {
        "section": "Article 155",
        "text": (
            "The President shall chair Cabinet meetings, direct and co-ordinate ministries and government departments, "
            "and assign responsibilities for the implementation and administration of any Act of Parliament to a Cabinet Secretary."
        )
    },
    {
        "section": "Article 217",
        "text": "The President exercises oversight over national revenue allocated to county governments."
    }
]

def retrieve(query: str, top_k: int = 5) -> List[Tuple[str, str]]:
    """
    Retrieve top matching sections for a query.
    Returns a list of tuples: (section, text)
    """
    # Simple keyword-based matching for demonstration
    query_lower = query.lower()
    scored_sections = []

    for entry in CONSTITUTION_SECTIONS:
        text_lower = entry["text"].lower()
        score = sum(1 for word in query_lower.split() if word in text_lower)
        if score > 0:
            scored_sections.append((score, entry["section"], entry["text"]))

    # Sort by score descending
    scored_sections.sort(reverse=True, key=lambda x: x[0])

    # Return top_k results
    return [(sec, txt) for _, sec, txt in scored_sections[:top_k]]


def synthesize_answer(query: str, top_k: int = 5) -> str:
    """
    Generate a concise answer from the top retrieved sections.
    """
    top_sections = retrieve(query, top_k)

    if not top_sections:
        return "No relevant information found in the Constitution."

    answers = []
    for section, text in top_sections:
        # Clean up the text for readability
        clean_text = re.sub(r"\s+", " ", text).strip()
        answers.append(f"{clean_text} (Section: {section})")

    return " ".join(answers)
