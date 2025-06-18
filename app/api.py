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
    headers = {"Authorization": f"Bearer {JINA_API_KEY}"}
    payload = {
        "input": [query.question],
        "model": EMBEDDING_MODEL
    }
    res = requests.post("https://api.jina.ai/v1/embeddings", headers=headers, json=payload)
    embedding = res.json()["data"][0]["embedding"]
    results = search(embedding)

    # Construct a meaningful answer and links (placeholder example below)
    answer = "You must use `gpt-3.5-turbo-0125`, even if the AI Proxy only supports `gpt-4o-mini`. Use the OpenAI API directly for this question."

    links = [
        {
            "url": "https://discourse.onlinedegree.iitm.ac.in/t/ga5-question-8-clarification/155939/4",
            "text": "Use the model that’s mentioned in the question."
        },
        {
            "url": "https://discourse.onlinedegree.iitm.ac.in/t/ga5-question-8-clarification/155939/3",
            "text": "My understanding is that you just have to use a tokenizer, similar to what Prof. Anand used, to get the number of tokens and multiply that by the given rate."
        }
    ]

    return {
        "answer": answer,
        "links": links
    }
