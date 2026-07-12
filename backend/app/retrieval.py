from __future__ import annotations

import re
from time import perf_counter

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams
from rank_bm25 import BM25Okapi
from typing import Any

from .config import Settings
from .embeddings import EmbeddingService
from .models import ChunkRecord, DocumentRecord, DocumentStatus, RetrievalHit, SearchResponse
from .storage import catalog


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


class HybridRetriever:
    def __init__(self, settings: Settings, embeddings: EmbeddingService) -> None:
        self.settings = settings
        self.embeddings = embeddings
        self.qdrant = AsyncQdrantClient(url=settings.qdrant_url)
        self._bm25: BM25Okapi | None = None
        self._bm25_chunks: list[ChunkRecord] = []
        self._cross_encoder: Any | None = None

    async def ensure_collection(self) -> None:
        collections = await self.qdrant.get_collections()
        names = {collection.name for collection in collections.collections}
        if self.settings.qdrant_collection not in names:
            await self.qdrant.create_collection(
                collection_name=self.settings.qdrant_collection,
                vectors_config=VectorParams(size=self.settings.embedding_size, distance=Distance.COSINE),
            )

    async def index_chunks(self, chunks: list[ChunkRecord]) -> None:
        await self.ensure_collection()
        vectors = await self.embeddings.embed([chunk.text for chunk in chunks])
        points = [
            PointStruct(
                id=chunk.id,
                vector=vector,
                payload={"document_id": chunk.document_id, "text": chunk.text, "metadata": chunk.metadata},
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        await self.qdrant.upsert(collection_name=self.settings.qdrant_collection, points=points)
        self.rebuild_bm25()

    async def restore_catalog(self) -> None:
        await self.ensure_collection()
        chunks: list[ChunkRecord] = []
        offset = None
        while True:
            points, offset = await self.qdrant.scroll(
                collection_name=self.settings.qdrant_collection,
                limit=100,
                offset=offset,
                with_payload=True,
            )
            for point in points:
                payload = point.payload or {}
                metadata = dict(payload.get("metadata", {}))
                document_id = str(payload.get("document_id", ""))
                text = str(payload.get("text", ""))
                if not document_id or not text:
                    continue
                catalog.upsert_document(
                    DocumentRecord(
                        id=document_id,
                        filename=str(metadata.get("filename", "Restored document")),
                        content_type=str(metadata.get("content_type", "")),
                        status=DocumentStatus.ready,
                        metadata=metadata,
                    )
                )
                chunks.append(ChunkRecord(id=str(point.id), document_id=document_id, text=text, metadata=metadata))
            if offset is None:
                break
        catalog.add_chunks(chunks)
        self.rebuild_bm25()

    def rebuild_bm25(self) -> None:
        self._bm25_chunks = catalog.chunks()
        corpus = [tokenize(chunk.text) for chunk in self._bm25_chunks]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    async def search(self, query: str, top_k: int = 8) -> SearchResponse:
        started = perf_counter()
        await self.ensure_collection()
        query_vector = (await self.embeddings.embed([query]))[0]
        dense_results = await self.qdrant.search(
            collection_name=self.settings.qdrant_collection,
            query_vector=query_vector,
            limit=max(top_k * 3, 12),
            with_payload=True,
        )

        hits: dict[str, RetrievalHit] = {}
        for result in dense_results:
            payload = result.payload or {}
            hits[str(result.id)] = RetrievalHit(
                chunk_id=str(result.id),
                document_id=str(payload.get("document_id", "")),
                text=str(payload.get("text", "")),
                metadata=dict(payload.get("metadata", {})),
                dense_score=float(result.score or 0.0),
            )

        if self._bm25 is None:
            self.rebuild_bm25()
        if self._bm25:
            scores = self._bm25.get_scores(tokenize(query))
            ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[: max(top_k * 3, 12)]
            max_score = max([score for _, score in ranked], default=1.0) or 1.0
            for index, score in ranked:
                chunk = self._bm25_chunks[index]
                hit = hits.get(chunk.id) or RetrievalHit(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    text=chunk.text,
                    metadata=chunk.metadata,
                )
                hit.bm25_score = float(score / max_score)
                hits[chunk.id] = hit

        reranked = self._rerank(query, list(hits.values()))
        latency_ms = (perf_counter() - started) * 1000
        return SearchResponse(hits=reranked[:top_k], latency_ms=latency_ms)

    def _rerank(self, query: str, hits: list[RetrievalHit]) -> list[RetrievalHit]:
        if hits:
            try:
                from sentence_transformers import CrossEncoder

                if self._cross_encoder is None:
                    self._cross_encoder = CrossEncoder(self.settings.cross_encoder_model)
                scores = self._cross_encoder.predict([(query, hit.text) for hit in hits])
                max_score = max([float(score) for score in scores], default=1.0) or 1.0
                for hit, score in zip(hits, scores, strict=True):
                    hit.rerank_score = float(score) / max_score
                    hit.combined_score = 0.50 * hit.dense_score + 0.20 * hit.bm25_score + 0.30 * hit.rerank_score
                return sorted(hits, key=lambda hit: hit.combined_score, reverse=True)
            except Exception:
                self._cross_encoder = None

        query_terms = set(tokenize(query))
        for hit in hits:
            chunk_terms = set(tokenize(hit.text))
            overlap = len(query_terms & chunk_terms) / max(len(query_terms), 1)
            hit.rerank_score = overlap
            hit.combined_score = 0.55 * hit.dense_score + 0.25 * hit.bm25_score + 0.20 * hit.rerank_score
        return sorted(hits, key=lambda hit: hit.combined_score, reverse=True)
