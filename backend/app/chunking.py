from __future__ import annotations

import re
from uuid import uuid4

from .models import ChunkRecord


def chunk_document(document_id: str, text: str, metadata: dict, chunk_size: int = 1200, overlap: int = 180) -> list[ChunkRecord]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: list[ChunkRecord] = []
    current = ""

    def emit(value: str, index: int) -> None:
        if value.strip():
            chunks.append(
                ChunkRecord(
                    id=str(uuid4()),
                    document_id=document_id,
                    text=value.strip(),
                    metadata={**metadata, "chunk_index": index},
                )
            )

    for paragraph in paragraphs:
        if len(current) + len(paragraph) + 2 <= chunk_size:
            current = f"{current}\n\n{paragraph}".strip()
            continue
        emit(current, len(chunks))
        tail = current[-overlap:] if overlap and current else ""
        current = f"{tail}\n\n{paragraph}".strip()

    emit(current, len(chunks))
    return chunks
