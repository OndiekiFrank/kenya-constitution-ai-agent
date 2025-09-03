# streamlit_app.py
import sys
from pathlib import Path
import streamlit as st

# -----------------------------
# Ensure src folder is in path
# -----------------------------
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

# -----------------------------
# Imports
# -----------------------------
from retrieval import retrieve, synthesize_answer

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="🇰🇪 Constitution QA Agent", layout="wide")
st.title("🇰🇪 Constitution QA Agent")
st.write("Ask a question about the Kenyan Constitution and get relevant answers!")

query = st.text_input("Enter your question:")
k = st.number_input("Number of top results (k):", min_value=1, max_value=20, value=5, step=1)

if st.button("Get Answer") and query:
    with st.spinner("Searching the Constitution..."):
        hits = retrieve(query, k=k)
        answer = synthesize_answer(query, hits)
    
    st.subheader("Answer:")
    st.write(answer)

    st.subheader("Top Hits:")
    for hit in hits:
        st.markdown(f"**Section:** {hit['section']} | **Score:** {hit['score']:.4f}")
        st.write(hit['text'])
        st.write("---")
