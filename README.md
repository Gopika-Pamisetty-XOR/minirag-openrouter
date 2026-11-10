# minirag-openrouter
Here is the complete `README.md` text enclosed entirely inside a single code block, formatted so you can easily copy and paste it directly into your file:

```markdown
# minirag-openrouter 🚀

A simple, free Python RAG chatbot powered by OpenRouter, FAISS, and RapidFuzz.

## 🌟 Overview
This project is a lightweight, cost-effective Retrieval-Augmented Generation (RAG) pipeline built entirely in Python. It avoids expensive local LLM overhead or paid cloud databases by leveraging **OpenRouter's free-tier APIs** for both vector embeddings and LLM chat responses.

## 🛠️ Tech Stack & Services
* **Embeddings:** `nvidia/llama-nemotron-embed-vl-1b-v2:free` (OpenRouter API)
* **LLM Chat:** `nvidia/nemotron-3.5-lightning:free` (OpenRouter API)
* **Vector Indexing:** **FAISS** (Facebook AI Similarity Search)
* **Query Cleaning:** **RapidFuzz** (for handling typos and common terms)

---

## 📂 Project Structure
```text
rag_project/
├── .env                 # API keys (ignored by git)
├── chunks.json          # Knowledge base (text chunks)
├── build_index.py       # Generates the FAISS vector index
├── faiss_index.bin      # Compiled binary vector database (ignored by git)
└── rag_query.py         # Main interactive CLI chatbot script

```

---

## 🚀 Quickstart Guide

### 1. Clone and Setup Environment

```bash
git clone [https://github.com/YOUR_USERNAME/minirag-openrouter.git](https://github.com/YOUR_USERNAME/minirag-openrouter.git)
cd minirag-openrouter

# Create and activate virtual environment
python3 -m venv rag_env
source rag_env/bin/activate

# Install required dependencies
pip install numpy faiss-cpu requests python-dotenv rapidfuzz

```

### 2. Configure Your API Key

Create a `.env` file in the root directory and add your OpenRouter API key:

```env
OPENROUTER_API_KEY=your_actual_openrouter_api_key_here

```

### 3. Build the Vector Index

Run the indexer to process your `chunks.json` into `faiss_index.bin`:

```bash
python3 build_index.py

```

### 4. Run the Chatbot

Start asking questions against your local knowledge base:

```bash
python3 rag_query.py

```

```

```
