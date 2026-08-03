<img src="./readme/card-titles/title1.svg"/>
<br>

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details when a license file is added.

<br><br>
<!-- project overview -->
<img src="./readme/card-titles/title2.svg"/>

> RAG Systems is a production-oriented retrieval-augmented generation playground for testing hybrid search, citation verification, multi-modal document workflows, and agentic RAG patterns.<br>
> The initial implementation focuses on Hybrid Search RAG with a FastAPI backend, React/Vite frontend, Qdrant vector storage, BM25 keyword retrieval, reranking, and grounded answer generation.

<br>
<!-- System Design -->
<img src="./readme/card-titles/title3.svg"/>

### Project Tracks

| Track | Focus | Status |
|------|-------|--------|
| **Project 1** | Hybrid Search RAG with citation verification | Implemented first |
| **Project 2** | Multi-modal Document RAG platform | Structure planned |
| **Project 3** | Agentic RAG with LangGraph | Structure planned |

<br>

### Application Architecture

| Layer | Purpose |
|------|---------|
| **React + Vite Frontend** | Upload, query, retrieval debugging, latency metrics, and evaluation UI |
| **FastAPI Backend** | Document ingestion, chunking, retrieval, answer generation, verification, and API endpoints |
| **Document Loader** | Extracts and normalizes PDF, Markdown, HTML, and text content |
| **Hybrid Retrieval** | Combines Qdrant dense vector search with BM25 keyword search |
| **Reranking** | Uses a cross-encoder when installed, with a lightweight lexical fallback |
| **Generation Layer** | Answers only from retrieved context, with deterministic local fallback support |

<br>

### Repository Map

| Path | Description |
|------|-------------|
| `backend/app/main.py` | FastAPI application and route entrypoint |
| `backend/app/document_loader.py` | PDF, Markdown, HTML, and text extraction |
| `backend/app/chunking.py` | Chunking and overlap logic |
| `backend/app/embeddings.py` | OpenAI-compatible and deterministic local embeddings |
| `backend/app/retrieval.py` | Dense, BM25, hybrid retrieval, and reranking flow |
| `backend/app/generation.py` | Context-grounded answer generation |
| `backend/app/storage.py` | Qdrant storage integration |
| `frontend/` | React + TypeScript + Vite interface |
| `docker-compose.yml` | Qdrant service configuration |

<br><br>
<!-- Project Highlights -->
<img src="./readme/card-titles/title4.svg"/>

### Core Features

- **Document ingestion**: Upload PDFs, Markdown, HTML, and text files.<br>
- **Text normalization**: Clean extracted document text and preserve source metadata.<br>
- **Chunking pipeline**: Split documents with overlap for retrieval-friendly context windows.<br>
- **Vector storage**: Generate embeddings and store chunks in Qdrant.<br>
- **Hybrid search**: Combine dense vector scores with BM25 keyword relevance.<br>
- **Reranking**: Apply cross-encoder reranking when available, with a local lexical fallback.<br>
- **Grounded answers**: Answer only from retrieved context and return insufficient-evidence behavior when needed.<br>
- **Citation verification**: Extract answer claims and run a verification pass against supporting context.<br>
- **Debug visibility**: Show upload status, retrieval diagnostics, latency metrics, and evaluation endpoints.<br>

<br>

### Retrieval Pipeline

| Step | Behavior |
|------|----------|
| **Load** | Parse PDF, Markdown, HTML, or plain text |
| **Clean** | Normalize content and attach source metadata |
| **Chunk** | Split content into overlapping retrieval units |
| **Embed** | Use OpenAI-compatible embeddings or deterministic local vectors |
| **Store** | Persist chunks in Qdrant and maintain a BM25 index |
| **Retrieve** | Blend dense and keyword results |
| **Rerank** | Promote the strongest context passages |
| **Generate** | Produce grounded answers with citations |
| **Verify** | Check claim-level citation support |

<br>
<!-- Demo -->
<img src="./readme/card-titles/title5.svg"/>

### Quick Start

Copy environment settings:

```bash
cp .env.example .env
```

Start Qdrant:

```bash
docker compose up -d qdrant
```

Start the backend:

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

Start the frontend:

```bash
cd frontend
npm install
npm run dev
```

Open the app at:

```text
http://localhost:5173
```

<br>

### Local Fallback Mode

The backend can run without an OpenAI key by using deterministic local embeddings and extractive answers. Add `OPENAI_API_KEY` in `.env` for generated answers and OpenAI embeddings.

<br><br>
<!-- Development & Testing -->
<img src="./readme/card-titles/title6.svg"/>

### Backend Stack

| Tool | Purpose |
|------|---------|
| **FastAPI** | Python API server |
| **Qdrant Client** | Vector database integration |
| **rank-bm25** | Keyword retrieval index |
| **OpenAI SDK** | OpenAI-compatible embeddings and generation |
| **pypdf** | PDF extraction |
| **BeautifulSoup** | HTML parsing and cleanup |
| **httpx** | Async HTTP client support |
| **Uvicorn** | Local ASGI server |

<br>

### Frontend Stack

| Tool | Purpose |
|------|---------|
| **React** | Query and document workflow UI |
| **TypeScript** | Type-safe frontend code |
| **Vite** | Development server and build tooling |
| **lucide-react** | Interface icons |

<br>

### Development Commands

| Command | Purpose |
|---------|---------|
| `docker compose up -d qdrant` | Start the Qdrant vector store |
| `uvicorn app.main:app --reload --port 8000` | Run the FastAPI backend from `backend/` |
| `npm run dev` | Run the frontend from `frontend/` |
| `npm run build` | Build the frontend |
| `npm run preview` | Preview the built frontend |

<br><br>
<!-- Extras -->
<img src="./readme/card-titles/title7.svg"/>

### Additional Tools & Services

| Tool | Purpose |
|------|---------|
| **Qdrant** | Dense vector search and chunk storage |
| **BM25** | Lexical keyword retrieval |
| **Cross-Encoder Reranker** | Optional stronger reranking with `requirements-reranker.txt` |
| **Docker Compose** | Local vector database orchestration |
| **OpenAI-Compatible APIs** | Optional embeddings and generated answers |
| **Deterministic Local Embeddings** | Offline development fallback |

<br>

---

**RAG Systems** - Production-oriented playground for hybrid, verified, and extensible RAG workflows.

*Ground answers, verify citations, and make retrieval visible.*
