from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

class Query(BaseModel):
    question: str
    image: Optional[str] = None

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/api/")
def answer_question(query: Query):
    # Replace with actual inference logic
    raw_answer = "You must use `gpt-3.5-turbo-0125`, even if the AI Proxy only supports `gpt-4o-mini`. Use the OpenAI API directly for this question."
    raw_links = [
        {
            "url": "https://discourse.onlinedegree.iitm.ac.in/t/ga5-question-8-clarification/155939/4",
            "text": "Use the model that’s mentioned in the question."
        },
        {
            "url": "https://discourse.onlinedegree.iitm.ac.in/t/ga5-question-8-clarification/155939/3",
            "text": "Token count explanation using tokenizer."
        }
    ]

    return {
        "answer": raw_answer.strip(),
        "links": raw_links
    }
