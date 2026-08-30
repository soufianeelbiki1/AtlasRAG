"""Hybrid retrieval orchestration with rank fusion and reranking.

The module combines independent retrieval backends without assuming their raw
scores are calibrated onto the same scale. Reciprocal-rank fusion uses rank
positions only; reranking remains a separate, replaceable stage.
"""

from dataclasses import dataclass
from typing import Protocol

from app.models import EvidenceChunk
from app.retrieval import Retriever


class Reranker(Protocol):
    def rerank(
        self,
        query: str,
        candidates: list[EvidenceChunk],
        top_k: int,
    ) -> list[EvidenceChunk]: ...


@dataclass(frozen=True, slots=True)
class FusionSource:
    name: str
    retriever: Retriever

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("fusion source name must not be empty")


class ReciprocalRankFusionRetriever:
    """Fuse heterogeneous retrievers using normalized reciprocal-rank scores."""

    def __init__(
        self,
        sources: tuple[FusionSource, ...],
        *,
        candidate_k: int = 20,
        rank_constant: int = 60,
    ) -> None:
        if len(sources) < 2:
            raise ValueError("hybrid retrieval requires at least two sources")
        if len({source.name for source in sources}) != len(sources):
            raise ValueError("fusion source names must be unique")
        if candidate_k < 1:
            raise ValueError("candidate_k must be at least 1")
        if rank_constant < 1:
            raise ValueError("rank_constant must be at least 1")
        self._sources = sources
        self._candidate_k = candidate_k
        self._rank_constant = rank_constant

    def search(self, query: str, top_k: int) -> list[EvidenceChunk]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        fused_scores: dict[str, float] = {}
        chunks: dict[str, EvidenceChunk] = {}
        first_seen: dict[str, int] = {}
        encounter = 0

        for source in self._sources:
            results = source.retriever.search(query, self._candidate_k)
            seen_in_source: set[str] = set()
            for rank, chunk in enumerate(results, start=1):
                if chunk.id in seen_in_source:
                    raise ValueError(
                        f"retriever {source.name} returned duplicate chunk id {chunk.id}"
                    )
                seen_in_source.add(chunk.id)

                existing = chunks.get(chunk.id)
                if existing is not None and (
                    existing.text != chunk.text or existing.source != chunk.source
                ):
                    raise ValueError(
                        f"chunk id {chunk.id} has conflicting content across retrieval sources"
                    )

                if chunk.id not in first_seen:
                    first_seen[chunk.id] = encounter
                    encounter += 1
                chunks[chunk.id] = chunk
                fused_scores[chunk.id] = fused_scores.get(chunk.id, 0.0) + 1.0 / (
                    self._rank_constant + rank
                )

        if not fused_scores:
            return []

        maximum = max(fused_scores.values())
        ranked_ids = sorted(
            fused_scores,
            key=lambda chunk_id: (
                -fused_scores[chunk_id],
                first_seen[chunk_id],
            ),
        )
        return [
            chunks[chunk_id].model_copy(
                update={"score": min(fused_scores[chunk_id] / maximum, 1.0)}
            )
            for chunk_id in ranked_ids[:top_k]
        ]


class TermCoverageReranker:
    """Credential-free reference reranker for deterministic regression tests.

    This is not presented as a production semantic reranker. It proves the
    reranking contract and keeps CI deterministic until a measured model-backed
    adapter is introduced.
    """

    def __init__(self, *, retrieval_weight: float = 0.7) -> None:
        if not 0.0 <= retrieval_weight <= 1.0:
            raise ValueError("retrieval_weight must be within [0, 1]")
        self._retrieval_weight = retrieval_weight

    def rerank(
        self,
        query: str,
        candidates: list[EvidenceChunk],
        top_k: int,
    ) -> list[EvidenceChunk]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        query_terms = {term for term in query.lower().split() if term}
        scored: list[tuple[float, int, EvidenceChunk]] = []
        for index, chunk in enumerate(candidates):
            chunk_terms = {term for term in chunk.text.lower().split() if term}
            coverage = (
                len(query_terms & chunk_terms) / len(query_terms)
                if query_terms
                else 0.0
            )
            combined = (
                self._retrieval_weight * chunk.score
                + (1.0 - self._retrieval_weight) * coverage
            )
            scored.append(
                (
                    combined,
                    index,
                    chunk.model_copy(update={"score": min(combined, 1.0)}),
                )
            )

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [item[2] for item in scored[:top_k]]


class HybridRetriever:
    """Compose rank fusion and an optional reranking stage."""

    def __init__(
        self,
        fusion: ReciprocalRankFusionRetriever,
        *,
        reranker: Reranker | None = None,
        rerank_candidates: int = 10,
    ) -> None:
        if rerank_candidates < 1:
            raise ValueError("rerank_candidates must be at least 1")
        self._fusion = fusion
        self._reranker = reranker
        self._rerank_candidates = rerank_candidates

    def search(self, query: str, top_k: int) -> list[EvidenceChunk]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        candidate_count = max(top_k, self._rerank_candidates)
        candidates = self._fusion.search(query, candidate_count)
        if self._reranker is None:
            return candidates[:top_k]
        return self._reranker.rerank(query, candidates, top_k)
