# app/api.py
from fastapi import FastAPI
from pydantic import BaseModel
import requests
import numpy as np
from app.search import search
import os

app = FastAPI()

JINA_API_KEY = os.getenv("JINA_API_KEY")
JINA_ENDPOINT = "https://api.jina.ai/v1/embeddings"
HEADERS = {"Authorization": f"Bearer {JINA_API_KEY}", "Content-Type": "application/json"}

class QueryRequest(BaseModel):
    question: str
    image: str | None = None

@app.get("/")
def read_root():
    return {"status": "TDS Virtual TA is running"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/api/")
async def query_api(request: QueryRequest):
    payload = {
        "input": [request.question],
        "model": "jina-embeddings-v2-base-en"
    }

    response = requests.post(JINA_ENDPOINT, json=payload, headers=HEADERS)
    if response.status_code != 200:
        return {"answer": "Embedding failed", "links": []}

    query_embedding = response.json()["data"][0]["embedding"]
    matches = search(query_embedding)

    return {
        "answer": matches[0],
        "links": [{"url": "https://example.com", "text": "Reference"}]
    }
