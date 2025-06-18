# app/api.py
from fastapi import FastAPI
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
    if not JINA_API_KEY:
        return {
            "answer": "API key missing. Please contact admin.",
            "links": []
        }

    headers = {"Authorization": f"Bearer {JINA_API_KEY}"}
    payload = {
        "input": [query.question],
        "model": EMBEDDING_MODEL
    }
    try:
        res = requests.post("https://api.jina.ai/v1/embeddings", headers=headers, json=payload)
        res.raise_for_status()
        embedding = res.json()["data"][0]["embedding"]
    except Exception as e:
        return {
            "answer": "Error generating embedding: " + str(e),
            "links": []
        }

    results = search(embedding)
    answer = "\n".join(results) if isinstance(results, list) else str(results)
    return {
        "answer": answer,
        "links": [
            {"url": "https://example.com", "text": "Example thread"}
        ]
    }
    
