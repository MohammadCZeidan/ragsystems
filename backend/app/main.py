from __future__ import annotations

import tempfile
from pathlib import Path
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .chunking import chunk_document
from .config import get_settings
from .document_loader import extract_text
from .embeddings import EmbeddingService
from .generation import AnswerService
from .models import AskRequest, AskResponse, DocumentRecord, DocumentStatus, SearchResponse
from .retrieval import HybridRetriever
from .storage import catalog

settings = get_settings()
embeddings = EmbeddingService(settings)
retriever = HybridRetriever(settings, embeddings)
answers = AnswerService(settings)

app = FastAPI(title="RAG Systems API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.on_event("startup")
async def startup() -> None:
    await retriever.restore_catalog()


@app.post("/documents", response_model=DocumentRecord)
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)) -> DocumentRecord:
    suffix = Path(file.filename or "document.txt").suffix.lower()
    if suffix not in {".pdf", ".md", ".markdown", ".html", ".htm", ".txt"}:
        raise HTTPException(status_code=400, detail="Supported files: PDF, Markdown, HTML, and text.")

    document = DocumentRecord(id=str(uuid4()), filename=file.filename or "document", content_type=file.content_type or "")
    catalog.upsert_document(document)

    data = await file.read()
    temp_dir = Path(tempfile.gettempdir()) / "ragsystems_uploads"
    temp_dir.mkdir(exist_ok=True)
    temp_path = temp_dir / f"{document.id}{suffix}"
    temp_path.write_bytes(data)
    background_tasks.add_task(index_document, document.id, temp_path, document.filename, document.content_type)
    return document


async def index_document(document_id: str, path: Path, filename: str, content_type: str) -> None:
    try:
        catalog.set_status(document_id, DocumentStatus.indexing)
        text, metadata = extract_text(path, filename, content_type)
        chunks = chunk_document(document_id, text, metadata)
        catalog.add_chunks(chunks)
        await retriever.index_chunks(chunks)
        catalog.set_status(document_id, DocumentStatus.ready)
    except Exception as exc:
        catalog.set_status(document_id, DocumentStatus.failed, str(exc))


@app.get("/documents", response_model=list[DocumentRecord])
async def list_documents() -> list[DocumentRecord]:
    return catalog.documents()


@app.get("/search", response_model=SearchResponse)
async def search(q: str, top_k: int = 8) -> SearchResponse:
    return await retriever.search(q, top_k)


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    search_response = await retriever.search(request.question, request.top_k)
    return await answers.answer(request.question, search_response.hits, search_response.latency_ms)


@app.get("/metrics")
async def metrics() -> dict:
    docs = catalog.documents()
    chunks = catalog.chunks()
    return {
        "documents": len(docs),
        "chunks": len(chunks),
        "ready_documents": len([doc for doc in docs if doc.status == DocumentStatus.ready]),
        "failed_documents": len([doc for doc in docs if doc.status == DocumentStatus.failed]),
    }
