
import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import requests

from langchain.text_splitter import MarkdownTextSplitter
from transformers import AutoTokenizer
from dotenv import load_dotenv
load_dotenv()

JINA_API_KEY = os.getenv("JINA_API_KEY")
EMBEDDING_MODEL = "jina-embeddings-v2-base-en"
HEADERS = {"Authorization": f"Bearer {JINA_API_KEY}"}

def get_embedding(text):
    payload = {"input": text, "model": EMBEDDING_MODEL}
    response = requests.post("https://api.jina.ai/v1/embeddings", json=payload, headers=HEADERS)
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]

# Load markdown files and split them
input_dir = Path("data")
splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = []
metadatas = []

for file in input_dir.glob("*.md"):
    with open(file, "r", encoding="utf-8") as f:
        raw_text = f.read()
    chunks = splitter.split_text(raw_text)
    for chunk in chunks:
        docs.append(chunk)
        metadatas.append({"file": file.name, "chunk": chunk})

# Generate and store embeddings
embeddings = []
for chunk in tqdm(docs, desc="Jina Embedding"):
    emb = get_embedding(chunk)
    embeddings.append(emb)

np.savez("embeddings.npz", embeddings=np.array(embeddings), chunks=np.array(chunks))
print("✅ Done. Saved full embeddings to embeddings.npz")
