import os
import json
import numpy as np
import faiss
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

OPENROUTER_EMBEDDING_URL = "https://openrouter.ai/api/v1/embeddings"

# Use a valid free embedding model on OpenRouter
EMBEDDING_MODEL = "nvidia/nemotron-3.5-embed-1b:free"

def get_embedding(text):
    """Fetches embedding vector from OpenRouter using a free embedding model."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": EMBEDDING_MODEL,
        "input": text
    }
    response = requests.post(OPENROUTER_EMBEDDING_URL, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]

# 1. Load your chunks
print("Loading chunks.json...")
if not os.path.exists("chunks.json"):
    raise FileNotFoundError("Error: 'chunks.json' not found in the current directory.")

with open("chunks.json", "r") as f:
    chunks = json.load(f)

print(f"Generating embeddings for {len(chunks)} chunks using OpenRouter ({EMBEDDING_MODEL})...")

# 2. Generate embeddings for all chunks
embeddings = []
for i, chunk in enumerate(chunks):
    print(f"Processing chunk {i+1}/{len(chunks)}...")
    emb = get_embedding(chunk)
    embeddings.append(emb)

# Convert to numpy array
embeddings_np = np.array(embeddings, dtype=np.float32)

# 3. Create FAISS index dynamically based on the model's dimension size
dimension = embeddings_np.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings_np)

# 4. Save the index to disk
faiss.write_index(index, "faiss_index.bin")
print(f"\nSuccess! 'faiss_index.bin' (Dimension: {dimension}) has been successfully created and saved.")