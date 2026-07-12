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
                confidence=0.0,
                assistant_note="I could not find matching document evidence. Try uploading more source material or asking with more specific names, dates, or terms.",
                suggested_questions=[
                    "Which documents should I upload to answer this?",
                    "What keywords should I search for?",
                    "Can you summarize the available documents first?",
                ],
                evidence_gaps=["No retrieved chunks were available for this question."],
            )

        answer = await self._generate(question, hits)
        claims = self._verify_claims(answer, hits)
        insufficient = not claims or any(not claim.supported for claim in claims)
        if insufficient:
            answer = f"{answer}\n\nSome claims could not be fully verified from the retrieved context."
        latency_ms = retrieval_latency_ms + (perf_counter() - started) * 1000
        confidence = self._confidence(claims, hits)
        return AskResponse(
            answer=answer,
            claims=claims,
            hits=hits,
            latency_ms=latency_ms,
            insufficient_evidence=insufficient,
            confidence=confidence,
            assistant_note=self._assistant_note(confidence, insufficient, hits),
            suggested_questions=self._suggested_questions(question, hits),
            evidence_gaps=self._evidence_gaps(claims, hits),
        )

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

    def _confidence(self, claims: list[Claim], hits: list[RetrievalHit]) -> float:
        if not claims or not hits:
            return 0.0
        supported_ratio = len([claim for claim in claims if claim.supported]) / len(claims)
        retrieval_strength = min(max(hits[0].combined_score, hits[0].dense_score, hits[0].bm25_score), 1.0)
        return round((0.7 * supported_ratio) + (0.3 * retrieval_strength), 2)

    def _assistant_note(self, confidence: float, insufficient: bool, hits: list[RetrievalHit]) -> str:
        if insufficient:
            return "I found related context, but at least one claim needs stronger support. Use the evidence gaps and retrieval cards before relying on the answer."
        if confidence >= 0.75:
            return "The answer is well supported by the retrieved document chunks. You can inspect the cited evidence below."
        if hits:
            return "The answer is supported, but confidence is moderate because the retrieved evidence is narrow or weakly scored."
        return "I could not find enough document evidence to answer confidently."

    def _suggested_questions(self, question: str, hits: list[RetrievalHit]) -> list[str]:
        if not hits:
            return [
                "What documents are currently indexed?",
                "Which source should I upload next?",
                "What exact terms should I search for?",
            ]

        filename = str(hits[0].metadata.get("filename", "this document"))
        lower_question = question.lower()
        if lower_question.startswith("who"):
            return [
                f"What are the strongest skills mentioned in {filename}?",
                f"What experience does {filename} describe?",
                f"Summarize {filename} in bullet points.",
            ]
        return [
            "What evidence supports this answer?",
            "What is missing from the documents?",
            "Give me a shorter summary with citations.",
        ]

    def _evidence_gaps(self, claims: list[Claim], hits: list[RetrievalHit]) -> list[str]:
        gaps: list[str] = []
        if not claims:
            gaps.append("No citation-bearing claims were produced.")
        unsupported = [claim.text for claim in claims if not claim.supported]
        if unsupported:
            gaps.append(f"{len(unsupported)} claim(s) did not pass citation verification.")
        if hits and max(hit.combined_score for hit in hits) < 0.25:
            gaps.append("Retrieval scores are low; ask a more specific question or add more source documents.")
        return gaps
