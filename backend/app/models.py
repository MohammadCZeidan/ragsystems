from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DocumentStatus(str, Enum):
    queued = "queued"
    indexing = "indexing"
    ready = "ready"
    failed = "failed"


class DocumentRecord(BaseModel):
    id: str
    filename: str
    content_type: str
    status: DocumentStatus = DocumentStatus.queued
    chunk_count: int = 0
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkRecord(BaseModel):
    id: str
    document_id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalHit(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    metadata: dict[str, Any]
    dense_score: float = 0.0
    bm25_score: float = 0.0
    rerank_score: float = 0.0
    combined_score: float = 0.0


class AskRequest(BaseModel):
    question: str
    top_k: int = Field(default=8, ge=1, le=30)


class Citation(BaseModel):
    chunk_id: str
    quote: str
    supported: bool
    reason: str


class Claim(BaseModel):
    text: str
    citations: list[Citation] = Field(default_factory=list)
    supported: bool = False


class AskResponse(BaseModel):
    answer: str
    claims: list[Claim]
    hits: list[RetrievalHit]
    latency_ms: float
    insufficient_evidence: bool
    confidence: float = 0.0
    assistant_note: str = ""
    suggested_questions: list[str] = Field(default_factory=list)
    evidence_gaps: list[str] = Field(default_factory=list)


class SearchResponse(BaseModel):
    hits: list[RetrievalHit]
    latency_ms: float
