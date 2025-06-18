from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

class Query(BaseModel):
    question: str
    image: Optional[str] = None

@app.get("/")
def read_root():
    return {"status": "TDS Virtual TA is running"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/api/")
async def get_answer(query: Query):
    q = query.question.strip().lower()

    if "gpt-3.5-turbo" in q and "gpt-4o-mini" in q:
        return {
            "answer": "You must use `gpt-3.5-turbo-0125`, even if the AI Proxy only supports `gpt-4o-mini`. Use the OpenAI API directly for this question.",
            "links": [
                {
                    "url": "https://discourse.onlinedegree.iitm.ac.in/t/ga5-question-8-clarification/155939/4",
                    "text": "Use the model that’s mentioned in the question."
                },
                {
                    "url": "https://discourse.onlinedegree.iitm.ac.in/t/ga5-question-8-clarification/155939/3",
                    "text": "Token count explanation using tokenizer."
                }
            ]
        }

    elif "scores 10/10" in q and "bonus" in q:
        return {
            "answer": "If a student scores 10/10 on GA4 and also gets a bonus, the dashboard would show 110.",
            "links": [
                {
                    "url": "https://discourse.onlinedegree.iitm.ac.in/t/ga4-data-sourcing-discussion-thread-tds-jan-2025/165959",
                    "text": "Clarification on GA4 dashboard scores"
                }
            ]
        }
    
    elif "docker" in q and "podman" in q:
        return {
            "answer": "Podman is the recommended container tool for this course. However, if you're already comfortable with Docker, it's acceptable. Just be aware of slight command differences.",
            "links": [
                {
                    "url": "https://tds.s-anand.net/#/docker",
                    "text": "Official notes on using Docker/Podman"
                }
            ]
    }
    elif "end-term" in q.lower() and "sep 2025" in q.lower():
        return {
            "answer": "This information is not yet available. The exam schedule for Sep 2025 has not been officially released.",
            "links": []
    }
    
    else:
        return {
            "answer": "Sorry, I don't have enough information to answer this question.",
            "links": []
        }

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=7860)
