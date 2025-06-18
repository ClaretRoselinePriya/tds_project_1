# search.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load embeddings
data = np.load("embeddings.npz", allow_pickle=True)
embeddings = data["embeddings"]
chunks = data["chunks"]

# Search function
def search(query_embedding, top_k=3):
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [chunks[i] for i in top_indices]
