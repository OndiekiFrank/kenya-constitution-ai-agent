import streamlit as st
from src.retrieval import retrieve, synthesize_answer

st.set_page_config(page_title="Kenya Constitution AI Agent", layout="wide")

st.title("📜 Kenya Constitution AI Agent")

query = st.text_input("Ask a question about the Constitution:")

if query:
    hits = retrieve(query)
    resp = synthesize_answer(query, hits)
    
    st.subheader("Answer")
    st.write(resp["answer"])
    
    with st.expander("Retrieved Passages"):
        for _, row in hits.iterrows():
            st.markdown(f"- {row['text']}")