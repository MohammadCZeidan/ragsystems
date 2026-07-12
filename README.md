# RAG Systems

Production-oriented RAG playground with three tracks:

- Project 1: Hybrid Search RAG with citation verification
- Project 2: Multi-modal Document RAG platform
- Project 3: Agentic RAG with LangGraph

The initial implementation focuses on Project 1 and lays the backend/frontend structure for Projects 2 and 3.

## Stack

- Frontend: React, TypeScript, Vite
- Backend: FastAPI, Python
- Vector store: Qdrant
- Keyword search: BM25
- Reranking: Cross-encoder
- LLM and embeddings: OpenAI-compatible APIs, with local fallbacks for development

## Quick Start

1. Copy environment settings:

```bash
cp .env.example .env
```

2. Start Qdrant:

```bash
docker compose up -d qdrant
```

3. Start the backend:

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

For true cross-encoder reranking instead of the lightweight lexical fallback:

```bash
pip install -r requirements-reranker.txt
```

4. Start the frontend:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## Features Implemented

- Upload PDFs, Markdown, HTML, and text files
- Clean and normalize extracted document text
- Preserve source metadata
- Chunk documents with overlap
- Generate embeddings
- Store chunks in Qdrant
- Maintain a BM25 keyword index
- Hybrid retrieval with dense and BM25 scores
- Cross-encoder reranking when available
- Answer only from retrieved context
- Claim-level citation extraction and verification pass
- Insufficient-evidence response behavior
- Upload status, retrieval debugging, latency metrics, and evaluation endpoints

## Notes

The backend can run without an OpenAI key by using deterministic local embeddings and extractive answers. Add `OPENAI_API_KEY` in `.env` for generated answers and OpenAI embeddings.
