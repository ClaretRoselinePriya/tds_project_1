# app/search.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

data = np.load("embeddings.npz")
embeddings = data["embeddings"]
texts = data["chunks"]

def search(query_embedding, top_k=3):
    sims = cosine_similarity([query_embedding], embeddings)[0]
    top_indices = sims.argsort()[::-1][:top_k]
    return [texts[i] for i in top_indices]