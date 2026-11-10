# import os
# import json
# import numpy as np
# import faiss
# import requests
# from dotenv import load_dotenv

# load_dotenv()
# api_key = os.getenv("OPENROUTER_API_KEY")

# OPENROUTER_EMBEDDING_URL = "https://openrouter.ai/api/v1/embeddings"
# # Correct free embedding model string on OpenRouter
# EMBEDDING_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2:free"

# def get_embedding(text):
#     headers = {
#         "Authorization": f"Bearer {api_key}",
#         "Content-Type": "application/json"
#     }
#     payload = {
#         "model": EMBEDDING_MODEL,
#         "input": text
#     }
#     response = requests.post(OPENROUTER_EMBEDDING_URL, headers=headers, json=payload)
#     response.raise_for_status()
#     return response.json()["data"][0]["embedding"]

# print("Loading chunks.json...")
# if not os.path.exists("chunks.json"):
#     raise FileNotFoundError("Error: 'chunks.json' not found in the current directory.")

# with open("chunks.json", "r") as f:
#     chunks = json.load(f)

# print(f"Generating embeddings for {len(chunks)} chunks using OpenRouter ({EMBEDDING_MODEL})...")

# embeddings = []
# for i, chunk in enumerate(chunks):
#     print(f"Processing chunk {i+1}/{len(chunks)}...")
#     emb = get_embedding(chunk)
#     embeddings.append(emb)

# embeddings_np = np.array(embeddings, dtype=np.float32)

# # Create and save FAISS index
# dimension = embeddings_np.shape[1]
# index = faiss.IndexFlatL2(dimension)
# index.add(embeddings_np)

# faiss.write_index(index, "faiss_index.bin")
# print(f"\nSuccess! 'faiss_index.bin' (Dimension: {dimension}) has been successfully created.")


import os
import json
import numpy as np
import faiss
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

OPENROUTER_EMBEDDING_URL = "https://openrouter.ai/api/v1/embeddings"
EMBEDDING_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2:free"

def get_embeddings_batch(texts):
    """Fetches embedding vectors for a list of texts in a single API call."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": EMBEDDING_MODEL,
        "input": texts  # Sending the entire list at once!
    }
    response = requests.post(OPENROUTER_EMBEDDING_URL, headers=headers, json=payload)
    response.raise_for_status()
    
    # OpenRouter returns a list of data objects sorted by index
    data = response.json()["data"]
    # Sort by 'index' to ensure order matches our chunks list
    sorted_data = sorted(data, key=lambda x: x["index"])
    return [item["embedding"] for item in sorted_data]

print("Loading chunks.json...")
if not os.path.exists("chunks.json"):
    raise FileNotFoundError("Error: 'chunks.json' not found in the current directory.")

with open("chunks.json", "r") as f:
    chunks = json.load(f)

print(f"Generating embeddings for all {len(chunks)} chunks in a single batch...")

# Get all embeddings in one request
embeddings = get_embeddings_batch(chunks)

embeddings_np = np.array(embeddings, dtype=np.float32)

# Create and save FAISS index
dimension = embeddings_np.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings_np)

faiss.write_index(index, "faiss_index.bin")
print(f"\n⚡ Success! 'faiss_index.bin' (Dimension: {dimension}) created instantly.")