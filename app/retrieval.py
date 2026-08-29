from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from app.models import EvidenceChunk


class Retriever(Protocol):
    def search(self, query: str, top_k: int) -> list[EvidenceChunk]: ...


@dataclass(frozen=True)
class SeedDocument:
    id: str
    text: str
    source: str


class InMemoryRetriever:
    """Deterministic lexical retriever used for local development and tests.

    The implementation is intentionally simple; its purpose is to establish a
    stable retrieval contract before swapping infrastructure for hybrid search.
    """

    def __init__(self, documents: Iterable[SeedDocument]) -> None:
        self._documents = list(documents)

    def search(self, query: str, top_k: int) -> list[EvidenceChunk]:
        query_terms = {term for term in query.lower().split() if term}
        ranked: list[EvidenceChunk] = []

        for document in self._documents:
            doc_terms = {term for term in document.text.lower().split() if term}
            overlap = len(query_terms & doc_terms)
            if overlap == 0:
                continue

            score = overlap / max(len(query_terms), 1)
            ranked.append(
                EvidenceChunk(
                    id=document.id,
                    text=document.text,
                    source=document.source,
                    score=min(score, 1.0),
                )
            )

        ranked.sort(key=lambda chunk: chunk.score, reverse=True)
        return ranked[:top_k]
