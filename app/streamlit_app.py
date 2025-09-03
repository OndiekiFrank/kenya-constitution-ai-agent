import streamlit as st
import sys
from pathlib import Path

# Add src/ to sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.append(str(SRC_DIR))

from retrieval import synthesize_answer

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🇰🇪 Constitution QA Agent")
st.write("Ask a question about the Kenyan Constitution and get relevant answers!")

# User input
query = st.text_input("Enter your question:")

# Number input for top_k
top_k = st.number_input(
    "Number of top results (k):",
    min_value=1,
    max_value=10,
    value=3,
    step=1
)

# Run retrieval
if st.button("Get Answer"):
    if query.strip():
        try:
            answer = synthesize_answer(query, top_k=int(top_k))
            st.write("### Answer:")
            st.write(answer)
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a question.")
