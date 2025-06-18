# app/api.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
from app.search import search
import requests
import os

from dotenv import load_dotenv
load_dotenv()
app = FastAPI()

JINA_API_KEY = os.getenv("JINA_API_KEY")
EMBEDDING_MODEL = "jina-embeddings-v2-base-en"

class Query(BaseModel):
    question: str

@app.post("/api/")
async def handle_question(query: Query):
    headers = {"Authorization": f"Bearer {JINA_API_KEY}"}
    payload = {
        "input": [query.question],
        "model": EMBEDDING_MODEL
    }
    res = requests.post("https://api.jina.ai/v1/embeddings", headers=headers, json=payload)
    embedding = res.json()["data"][0]["embedding"]
    results = search(embedding)
    return {"results": results}