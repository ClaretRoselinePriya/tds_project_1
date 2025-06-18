# app/api.py
from fastapi import FastAPI
from pydantic import BaseModel
import requests
import numpy as np
from app.search import search
import os
from dotenv import load_dotenv


load_dotenv()
JINA_API_KEY = os.getenv("JINA_API_KEY")
JINA_ENDPOINT = "https://api.jina.ai/v1/embeddings"
HEADERS = {"Authorization": f"Bearer {JINA_API_KEY}", "Content-Type": "application/json"}

app = FastAPI()
class QuestionRequest(BaseModel):
    question: str

@app.get("/")
def read_root():
    return {"status": "TDS Virtual TA is running"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/api/")
def answer_question(req: QuestionRequest):
    headers = {
        "Authorization": f"Bearer {JINA_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "input": [req.question],
        "model": "jina-embeddings-v2-base-en"
    }
    response = requests.post("https://api.jina.ai/v1/embeddings", headers=headers, json=payload)
    response.raise_for_status()
    query_embedding = response.json()["data"][0]["embedding"]

    results = search(query_embedding)
    return {
        "answer": results[0],
        "links": [{"url": "https://tds.s-anand.net/#/docker", "text": "Docker and Podman setup"}]
    }
