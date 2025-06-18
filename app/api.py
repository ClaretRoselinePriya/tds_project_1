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

@app.get("/")
def health_check():
    return {"status": "running"}

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

    # Convert results into dummy links (you can improve formatting based on real data)
    links = []
    for r in results:
        links.append({
            "url": "https://discourse.onlinedegree.iitm.ac.in/t/example-thread",
            "text": r[:50]  # First 50 chars as link text
        })

    answer = "\n".join(results) if isinstance(results, list) else str(results)

    return {
        "answer": answer,
        "links": links
    }
