# src/api.py
from fastapi import FastAPI, Query
import pandas as pd
import uvicorn
from pathlib import Path
from src.retrieval import retrieve_answer

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "Data" / "kenya_constitution_structured.csv"

app = FastAPI(title="Kenya Constitution AI Agent")

def load_data(data_path: Path) -> pd.DataFrame:
    """Load and validate the constitution dataset."""
    df = pd.read_csv(data_path)
    expected_cols = {"Article/Section", "Text_English", "Text_Kiswahili"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}. Found: {list(df.columns)}")
    
    # Normalize column names for internal use
    df = df.rename(
        columns={
            "Article/Section": "section",
            "Text_English": "text_en",
            "Text_Kiswahili": "text_sw"
        }
    )
    return df

# Load data once
meta_df = load_data(DATA_PATH)

@app.get("/")
def root():
    return {"message": "Welcome to the Kenya Constitution AI Agent API. Use /query endpoint."}

@app.get("/query")
def query_agent(
    question: str = Query(..., description="Enter your question about the Constitution"),
    language: str = Query("en", description="Response language: 'en' or 'sw'")
):
    """
    Query the Constitution AI Agent.
    - `question`: your input question
    - `language`: 'en' for English or 'sw' for Kiswahili
    """
    if language not in ["en", "sw"]:
        return {"error": "Invalid language. Use 'en' or 'sw'."}

    try:
        answer, ref_section = retrieve_answer(question, meta_df, language)
        return {
            "question": question,
            "answer": answer,
            "reference_section": ref_section,
            "language": language
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("src.api:app", host="127.0.0.1", port=8000, reload=True)