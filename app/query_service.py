from typing import Protocol

from app.models import Citation, EvidenceChunk, QueryRequest, QueryResponse
from app.retrieval import Retriever


class AnswerGenerator(Protocol):
    def generate(self, question: str, evidence: list[EvidenceChunk]) -> str: ...


class ExtractiveAnswerGenerator:
    """Deterministic development/test generator that never invents unsupported text."""

    def generate(self, question: str, evidence: list[EvidenceChunk]) -> str:
        del question
        return " ".join(chunk.text.strip() for chunk in evidence if chunk.text.strip())


class QueryService:
    """Citation-first RAG application service with explicit evidence abstention."""

    def __init__(
        self,
        retriever: Retriever,
        generator: AnswerGenerator,
        *,
        minimum_evidence_score: float = 0.25,
    ) -> None:
        if not 0.0 <= minimum_evidence_score <= 1.0:
            raise ValueError("minimum_evidence_score must be within [0, 1]")
        self._retriever = retriever
        self._generator = generator
        self._minimum_evidence_score = minimum_evidence_score

    def query(self, request: QueryRequest) -> QueryResponse:
        retrieved = self._retriever.search(request.question, request.top_k)
        evidence = [chunk for chunk in retrieved if chunk.score >= self._minimum_evidence_score]
        if not evidence:
            return QueryResponse(
                answer="I do not have enough retrieved evidence to answer this question.",
                citations=[],
                grounded=False,
                confidence=0.0,
            )

        answer = self._generator.generate(request.question, evidence).strip()
        if not answer:
            return QueryResponse(
                answer="I do not have enough grounded content to answer this question.",
                citations=[],
                grounded=False,
                confidence=0.0,
            )

        citations = [
            Citation(source=chunk.source, chunk_id=chunk.id, score=chunk.score)
            for chunk in evidence
        ]
        confidence = sum(chunk.score for chunk in evidence) / len(evidence)
        return QueryResponse(
            answer=answer,
            citations=citations,
            grounded=True,
            confidence=confidence,
        )
