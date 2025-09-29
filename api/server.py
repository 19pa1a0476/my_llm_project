from fastapi import FastAPI, Query
from .rag_pipeline import get_answer

app = FastAPI()

@app.get("/query")
def query(q: str = Query(..., description="User question")):
    answer = get_answer(q)
    return {"query": q, "answer": answer}
