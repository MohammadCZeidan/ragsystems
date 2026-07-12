from __future__ import annotations

import re
from time import perf_counter

from openai import AsyncOpenAI

from .config import Settings
from .models import AskResponse, Citation, Claim, RetrievalHit
from .retrieval import tokenize


class AnswerService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = AsyncOpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None

    async def answer(self, question: str, hits: list[RetrievalHit], retrieval_latency_ms: float) -> AskResponse:
        started = perf_counter()
        if not hits:
            return AskResponse(
                answer="I do not have enough evidence in the indexed documents to answer that.",
                claims=[],
                hits=[],
                latency_ms=retrieval_latency_ms,
                insufficient_evidence=True,
            )

        answer = await self._generate(question, hits)
        claims = self._verify_claims(answer, hits)
        insufficient = not claims or any(not claim.supported for claim in claims)
        if insufficient:
            answer = f"{answer}\n\nSome claims could not be fully verified from the retrieved context."
        latency_ms = retrieval_latency_ms + (perf_counter() - started) * 1000
        return AskResponse(answer=answer, claims=claims, hits=hits, latency_ms=latency_ms, insufficient_evidence=insufficient)

    async def _generate(self, question: str, hits: list[RetrievalHit]) -> str:
        context = "\n\n".join(f"[{hit.chunk_id}] {hit.text}" for hit in hits)
        if self.client:
            response = await self.client.chat.completions.create(
                model=self.settings.openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Answer using only the provided context. Every factual sentence must end with one "
                            "or more citations in square brackets using chunk ids. If the context is insufficient, say so."
                        ),
                    },
                    {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}"},
                ],
                temperature=0.0,
            )
            return response.choices[0].message.content or ""

        return self._local_answer(question, hits[0])

    def _local_answer(self, question: str, hit: RetrievalHit) -> str:
        raw_text = hit.text.strip()
        text = re.sub(r"\s+", " ", raw_text).strip()
        citation = f"[{hit.chunk_id}]"
        lower_question = question.lower()

        if lower_question.startswith("who"):
            name = self._extract_name(raw_text)
            summary = self._extract_summary(raw_text)
            if summary:
                return f"{name} is {summary.rstrip('.')}. {citation}"
            first_sentence = self._first_supported_sentence(text)
            return f"{name} is described in the document as follows: {first_sentence}. {citation}"

        first_sentence = self._first_supported_sentence(text)
        return f"The strongest available evidence says: {first_sentence}. {citation}"

    def _first_supported_sentence(self, text: str) -> str:
        cleaned = re.sub(r"\s+", " ", text).strip()
        candidates = re.split(r"(?<=[.!?])\s+|\s+\|\s+", cleaned)
        for candidate in candidates:
            candidate = candidate.strip()
            if 40 <= len(candidate) <= 320:
                return candidate
        return cleaned[:280].strip()

    def _extract_name(self, text: str) -> str:
        ignored_title_words = {"Full", "Stack", "Engineer", "Specialist", "Systems", "AI", "RAG"}
        for line in text.splitlines():
            line = re.sub(r"^\[Page \d+\]\s*", "", line).strip()
            if not line:
                continue
            words = re.findall(r"\b[A-Z][a-z]+\b", line)
            name_words = [word for word in words if word not in ignored_title_words]
            if len(name_words) >= 2:
                return " ".join(name_words[:2])
        return "The person"

    def _extract_summary(self, text: str) -> str | None:
        match = re.search(r"Summary\s+(.*?)(?:\n\s*Education\b|\n\s*Experience\b|\n\s*Projects\b|$)", text, re.IGNORECASE | re.DOTALL)
        if not match:
            return None
        summary = re.sub(r"\s+", " ", match.group(1)).strip(" |")
        sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", summary) if part.strip()]
        if sentences:
            return " ".join(sentences[:2])
        return summary[:320].strip()

    def _verify_claims(self, answer: str, hits: list[RetrievalHit]) -> list[Claim]:
        hit_map = {hit.chunk_id: hit for hit in hits}
        sentences = [part.strip() for part in answer.splitlines() if "[" in part]
        claims: list[Claim] = []
        for sentence in sentences:
            citation_ids = re.findall(r"\[([0-9a-fA-F-]{32,36})\]", sentence)
            if not citation_ids:
                continue
            claim_text = re.sub(r"\[[^\]]+\]", "", sentence).strip()
            claim_terms = set(tokenize(claim_text))
            citations: list[Citation] = []
            for citation_id in citation_ids:
                hit = hit_map.get(citation_id)
                if not hit:
                    citations.append(Citation(chunk_id=citation_id, quote="", supported=False, reason="Citation was not retrieved."))
                    continue
                support_terms = set(tokenize(hit.text))
                overlap = len(claim_terms & support_terms) / max(len(claim_terms), 1)
                supported = overlap >= 0.35
                citations.append(
                    Citation(
                        chunk_id=citation_id,
                        quote=hit.text[:240],
                        supported=supported,
                        reason="Sufficient lexical support found." if supported else "Citation text does not sufficiently overlap the claim.",
                    )
                )
            claims.append(Claim(text=claim_text, citations=citations, supported=any(c.supported for c in citations)))
        return claims
