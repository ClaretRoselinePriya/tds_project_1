import os
import json
import time
import numpy as np
import httpx
import random
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
from semantic_text_splitter import MarkdownSplitter

load_dotenv()
JINA_API_KEY = os.getenv("JINA_API_KEY")
HEADERS = {"Authorization": f"Bearer {JINA_API_KEY}"}
JINA_URL = "https://api.jina.ai/v1/embeddings"
OLD_FILE = "embeddings_jina.npz"  # your uploaded .npz
NEW_FILE = "embeddings_recovered.npz"

def get_chunks(file_path, chunk_size=1000):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return MarkdownSplitter(chunk_size).chunks(content)

def get_embedding(text, retries=5):
    for attempt in range(retries):
        try:
            response = httpx.post(
                JINA_URL,
                headers=HEADERS,
                json={"input": [text], "model": "jina-embeddings-v2-base-en"},
                timeout=30
            )
            response.raise_for_status()
            return response.json()["data"][0]["embedding"]
        except Exception as e:
            wait = min(90, 2 ** (attempt + 1) + random.uniform(1, 3))
            print(f"Retry {attempt+1} failed: {e} — wait {int(wait)}s")
            time.sleep(wait)
    raise RuntimeError("Max retries exceeded")

# === Load existing checkpoint ===
data = np.load(OLD_FILE, allow_pickle=True)
existing_chunks = set(data["chunks"])
existing_embeddings = list(data["embeddings"])

# === Get all chunks from data folder ===
all_chunks = []
file_paths = [*Path("data").glob("*.md"), *Path("data").rglob("*.md")]
for fp in file_paths:
    chunks = get_chunks(fp)
    all_chunks.extend(chunks)

# === Filter chunks that were skipped ===
missing_chunks = [c for c in all_chunks if c not in existing_chunks]
print(f"🧮 Total missing chunks to embed: {len(missing_chunks)}")

# === Embed only the missing chunks ===
new_chunks = []
new_embeddings = []
with tqdm(total=len(missing_chunks), desc="Recovering") as pbar:
    for chunk in missing_chunks:
        try:
            emb = get_embedding(chunk)
            new_chunks.append(chunk)
            new_embeddings.append(emb)
        except Exception as e:
            print(f"❌ Skipped chunk: {e}")
        pbar.update(1)
        time.sleep(1)

# === Combine all ===
final_chunks = list(data["chunks"]) + new_chunks
final_embeddings = existing_embeddings + new_embeddings

np.savez(NEW_FILE, chunks=final_chunks, embeddings=final_embeddings)
print(f"✅ Recovery complete. Saved to {NEW_FILE}")
