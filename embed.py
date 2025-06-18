import os
import time
import json
import numpy as np
import httpx
import random
from pathlib import Path
from semantic_text_splitter import MarkdownSplitter
from tqdm import tqdm
from dotenv import load_dotenv

# === Load API key ===
load_dotenv()
JINA_API_KEY = os.getenv("JINA_API_KEY")
JINA_API_URL = "https://api.jina.ai/v1/embeddings"
HEADERS = {"Authorization": f"Bearer {JINA_API_KEY}"}

CHECKPOINT_INTERVAL = 200
CHECKPOINT_FILE = "embeddings_jina_checkpoint.npz"
FINAL_OUTPUT = "embeddings_jina.npz"

# === Rate limiter ===
class RateLimiter:
    def __init__(self, per_minute=60, per_second=1):
        self.per_minute = per_minute
        self.per_second = per_second
        self.request_times = []
        self.last_time = 0

    def wait(self):
        now = time.time()
        gap = now - self.last_time
        if gap < 1.0 / self.per_second:
            time.sleep((1.0 / self.per_second) - gap)
        self.request_times = [t for t in self.request_times if now - t < 60]
        if len(self.request_times) >= self.per_minute:
            time.sleep(60 - (now - self.request_times[0]))
        self.request_times.append(time.time())
        self.last_time = time.time()

rate_limiter = RateLimiter()

# === Jina embedding API ===
def get_embedding(text, retries=5):
    for attempt in range(retries):
        try:
            rate_limiter.wait()
            response = httpx.post(
                JINA_API_URL,
                headers=HEADERS,
                json={"input": [text], "model": "jina-embeddings-v2-base-en"},
                timeout=30
            )
            response.raise_for_status()
            return response.json()["data"][0]["embedding"]
        except Exception as e:
            wait = min(90, 2 ** (attempt + 1) + random.uniform(1, 3))
            print(f"Retry {attempt + 1} failed: {e} — waiting {int(wait)}s...")
            time.sleep(wait)
    raise RuntimeError("Max retries exceeded")

# === Markdown splitter ===
def get_chunks(file_path, chunk_size=1000):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    splitter = MarkdownSplitter(chunk_size)
    return splitter.chunks(content)

# === Load previous checkpoint if exists ===
def load_checkpoint():
    if Path(CHECKPOINT_FILE).exists():
        data = np.load(CHECKPOINT_FILE, allow_pickle=True)
        return list(data["chunks"]), list(data["embeddings"])
    return [], []

# === Main ===
if __name__ == "__main__":
    files = [*Path("data").glob("*.md"), *Path("data").rglob("*.md")]
    all_chunks, all_embeddings = load_checkpoint()

    already_done = set(all_chunks)
    file_chunks = {}
    total_chunks = 0

    for file_path in files:
        chunks = get_chunks(file_path)
        file_chunks[file_path] = chunks
        total_chunks += len(chunks)

    with tqdm(total=total_chunks, desc="Jina Embedding") as pbar:
        pbar.update(len(all_chunks))  # resume from last checkpoint
        for file_path, chunks in file_chunks.items():
            for chunk in chunks:
                if chunk in already_done:
                    continue
                try:
                    emb = get_embedding(chunk)
                    all_chunks.append(chunk)
                    all_embeddings.append(emb)
                    pbar.set_postfix({"file": file_path.name, "chunks": len(all_chunks)})
                except Exception as e:
                    print(f"❌ Skipped: {file_path.name} — {e}")
                pbar.update(1)
                time.sleep(1)

                # save checkpoint
                if len(all_chunks) % CHECKPOINT_INTERVAL == 0:
                    np.savez(CHECKPOINT_FILE, chunks=all_chunks, embeddings=all_embeddings)
                    print(f"💾 Saved checkpoint at {len(all_chunks)} chunks")

    np.savez(FINAL_OUTPUT, chunks=all_chunks, embeddings=all_embeddings)
    print(f"✅ Done. Saved full embeddings to {FINAL_OUTPUT}")
