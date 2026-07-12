from __future__ import annotations

from threading import RLock

from .models import ChunkRecord, DocumentRecord, DocumentStatus


class InMemoryCatalog:
    def __init__(self) -> None:
        self._documents: dict[str, DocumentRecord] = {}
        self._chunks: dict[str, ChunkRecord] = {}
        self._lock = RLock()

    def upsert_document(self, document: DocumentRecord) -> None:
        with self._lock:
            self._documents[document.id] = document

    def set_status(self, document_id: str, status: DocumentStatus, error: str | None = None) -> None:
        with self._lock:
            doc = self._documents[document_id]
            doc.status = status
            doc.error = error
            self._documents[document_id] = doc

    def add_chunks(self, chunks: list[ChunkRecord]) -> None:
        with self._lock:
            counts_by_document: dict[str, int] = {}
            for chunk in chunks:
                self._chunks[chunk.id] = chunk
                counts_by_document[chunk.document_id] = counts_by_document.get(chunk.document_id, 0) + 1
            for document_id, count in counts_by_document.items():
                doc = self._documents[document_id]
                doc.chunk_count += count
                self._documents[doc.id] = doc

    def documents(self) -> list[DocumentRecord]:
        with self._lock:
            return list(self._documents.values())

    def chunks(self) -> list[ChunkRecord]:
        with self._lock:
            return list(self._chunks.values())

    def get_chunk(self, chunk_id: str) -> ChunkRecord | None:
        with self._lock:
            return self._chunks.get(chunk_id)


catalog = InMemoryCatalog()
