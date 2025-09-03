# src/api.py
from fastapi import FastAPI
from pydantic import BaseModel
from .retrieval import retrieve, synthesize_answer

app = FastAPI(title="Kenya Constitution QA API")

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

@app.post("/query")
async def query_constitution(request: QueryRequest):
    hits = retrieve(request.query, request.top_k)
    answer = synthesize_answer(request.query, hits)
    return {"query": request.query, "answer": answer, "hits": hits}
