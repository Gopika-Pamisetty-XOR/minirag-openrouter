import os
import re
import json
import numpy as np
import faiss
import requests
from rapidfuzz import process
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

# OpenRouter Endpoints & Models
OPENROUTER_EMBEDDING_URL = "https://openrouter.ai/api/v1/embeddings"
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"

# Must match the model used in build_index.py to ensure identical dimensions (2048)
EMBEDDING_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2:free"
# CHAT_MODEL = "nvidia/nemotron-3.5-lightning"
CHAT_MODEL = "nvidia/nemotron-3.5-lightning:free"

# Load FAISS index
if not os.path.exists("faiss_index.bin"):
    raise FileNotFoundError("Error: 'faiss_index.bin' not found. Run build_index.py first.")
index = faiss.read_index("faiss_index.bin")

# Load chunks
if not os.path.exists("chunks.json"):
    raise FileNotFoundError("Error: 'chunks.json' not found.")
with open("chunks.json", "r") as f:
    chunks = json.load(f)

COMMON_TERMS = [
    "pyATS", "Genie", "Git", "FAISS", "Docker", 
    "Kubernetes", "OSPF", "BGP", "SSH", "YAML"
]

def clean_query(query):
    return re.sub(r'[^a-zA-Z0-9\s]', '', query)

def get_embedding(text):
    """Fetches embedding vector from OpenRouter using the same model."""
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

def generate_answer(query, context_chunks):
    """Sends retrieved chunks and user query to Nemotron via OpenRouter chat completion."""
    context = "\n\n".join([f"- {chunk}" for chunk in context_chunks])
    
    prompt = f"""You are a helpful assistant. Use the following context to answer the user's question accurately. If you don't know the answer based on the context, say so clearly.

Context:
{context}

Question: {query}
"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }
    
    response = requests.post(OPENROUTER_CHAT_URL, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]

print("\n🚀 RAG Pipeline Ready (OpenRouter Free Embeddings + Nemotron)")
print("Type 'exit' to quit\n")

while True:
    query = input("Question: ")

    if query.lower() == "exit":
        break

    if not query.strip():
        print("Please enter a valid question\n")
        continue

    cleaned_query = clean_query(query)

    for word in cleaned_query.split():
        match = process.extractOne(
            word,
            COMMON_TERMS,
            score_cutoff=80
        )
        if match:
            cleaned_query = cleaned_query.replace(word, match[0])

    try:
        # 1. Convert user query to vector using OpenRouter free embeddings
        embedding_list = get_embedding(cleaned_query)
        query_vector = np.array([embedding_list], dtype=np.float32)

        # 2. Search FAISS index for top 3 matching chunks
        distances, indices = index.search(query_vector, k=3)
        retrieved_chunks = [chunks[idx] for idx in indices[0]]

        # 3. Generate answer using Nemotron chat completion
        print("\nThinking...")
        answer = generate_answer(query, retrieved_chunks)

        print("\nAnswer:\n")
        print(answer)
        print("\n" + "=" * 50 + "\n")

    except Exception as e:
        print(f"An error occurred: {e}\n")