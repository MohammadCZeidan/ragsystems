from __future__ import annotations

import hashlib

import numpy as np
from openai import AsyncOpenAI

from .config import Settings


class EmbeddingService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = AsyncOpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if self.client:
            response = await self.client.embeddings.create(model=self.settings.openai_embedding_model, input=texts)
            return [item.embedding for item in response.data]
        return [self._local_embedding(text) for text in texts]

    def _local_embedding(self, text: str) -> list[float]:
        vector = np.zeros(self.settings.embedding_size, dtype=np.float32)
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.settings.embedding_size
            vector[index] += 1.0
        norm = np.linalg.norm(vector)
        if norm:
            vector = vector / norm
        return vector.tolist()
