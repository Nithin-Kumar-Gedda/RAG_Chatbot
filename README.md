# 🔍 Insight Engine — RAG Document Assistant

A personal learning project to understand and implement a **Retrieval-Augmented Generation (RAG)** pipeline from scratch.

---

## 📌 Project Overview

Most LLMs don't know about your private documents. RAG solves this by retrieving relevant content from your own files and using it to generate accurate, grounded answers.

This project was built to understand each step of that process hands-on — from loading raw text files all the way to getting a natural language answer back through a chat interface.

### What it does

- Loads documents from a local directory (supports `.txt`, `.pdf`, `.csv`, `.docx`)
- Splits them into smaller chunks so retrieval is more precise
- Converts each chunk into a vector embedding using a local sentence transformer model
- Stores those embeddings in a persistent ChromaDB vector database
- When you ask a question, finds the most semantically similar chunks
- Passes those chunks as context to a Groq-hosted Llama 3.1 model to generate the final answer
- Serves everything through a clean Streamlit chat interface

### Why I built this

Instead of using a high-level framework like LlamaIndex or LangChain's full RAG chain — which hide all the details — this project builds each component as its own class: `DataLoader`, `TextSplitter`, `EmbeddingManager`, `VectorStore`, `RagRetriever`, and `ResponseGenerator`. The goal was to deeply understand what's happening at each stage rather than just calling a single function that does everything.

### Key concepts explored

- How embedding models turn text into vectors that capture semantic meaning
- How cosine similarity is used to find relevant chunks for a query
- Why chunk size and overlap matter for retrieval quality
- How to separate the retrieval step from the generation step
- How Streamlit's `session_state` and `cache_resource` work for stateful chat apps
