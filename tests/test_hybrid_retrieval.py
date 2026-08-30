from dataclasses import dataclass

import pytest

from app.hybrid_retrieval import (
    FusionSource,
    HybridRetriever,
    ReciprocalRankFusionRetriever,
    TermCoverageReranker,
)
from app.models import EvidenceChunk


@dataclass
class StaticRetriever:
    results: list[EvidenceChunk]

    def search(self, query: str, top_k: int) -> list[EvidenceChunk]:
        assert query
        return self.results[:top_k]


def chunk(id: str, text: str, score: float, source: str = "docs.md") -> EvidenceChunk:
    return EvidenceChunk(id=id, text=text, source=source, score=score)


def test_rrf_combines_rankings_without_comparing_raw_backend_scores() -> None:
    lexical = StaticRetriever(
        [
            chunk("a", "durable idempotency retries", 0.95),
            chunk("b", "payment ledger reconciliation", 0.80),
        ]
    )
    semantic = StaticRetriever(
        [
            chunk("b", "payment ledger reconciliation", 0.12),
            chunk("c", "network timeout reversal", 0.99),
        ]
    )
    retriever = ReciprocalRankFusionRetriever(
        (
            FusionSource("lexical", lexical),
            FusionSource("semantic", semantic),
        ),
        rank_constant=10,
    )

    results = retriever.search("payment reliability", top_k=3)

    assert [item.id for item in results] == ["b", "a", "c"]
    assert results[0].score == pytest.approx(1.0)
    assert 0.0 < results[1].score <= 1.0
    assert results[1].score == results[2].score


def test_rrf_rejects_conflicting_chunk_identity_across_sources() -> None:
    lexical = StaticRetriever([chunk("same", "first content", 1.0, "a.md")])
    semantic = StaticRetriever([chunk("same", "different content", 0.5, "b.md")])
    retriever = ReciprocalRankFusionRetriever(
        (
            FusionSource("lexical", lexical),
            FusionSource("semantic", semantic),
        )
    )

    with pytest.raises(ValueError, match="conflicting content"):
        retriever.search("query", top_k=2)


def test_rrf_rejects_duplicate_id_from_one_backend() -> None:
    repeated = chunk("same", "same content", 1.0)
    lexical = StaticRetriever([repeated, repeated])
    semantic = StaticRetriever([chunk("other", "other content", 0.5)])
    retriever = ReciprocalRankFusionRetriever(
        (
            FusionSource("lexical", lexical),
            FusionSource("semantic", semantic),
        )
    )

    with pytest.raises(ValueError, match="duplicate chunk id"):
        retriever.search("query", top_k=2)


def test_reference_reranker_can_promote_query_term_coverage() -> None:
    reranker = TermCoverageReranker(retrieval_weight=0.4)
    candidates = [
        chunk("a", "generic retrieval systems", 1.0),
        chunk("b", "hybrid retrieval reranking quality", 0.7),
    ]

    results = reranker.rerank("hybrid retrieval quality", candidates, top_k=2)

    assert [item.id for item in results] == ["b", "a"]
    assert results[0].score > results[1].score


def test_hybrid_retriever_applies_fusion_then_reranking() -> None:
    lexical = StaticRetriever(
        [
            chunk("a", "retrieval baseline", 0.9),
            chunk("b", "hybrid retrieval groundedness evaluation", 0.8),
        ]
    )
    semantic = StaticRetriever(
        [
            chunk("a", "retrieval baseline", 0.4),
            chunk("b", "hybrid retrieval groundedness evaluation", 0.3),
        ]
    )
    fusion = ReciprocalRankFusionRetriever(
        (
            FusionSource("lexical", lexical),
            FusionSource("semantic", semantic),
        )
    )
    hybrid = HybridRetriever(
        fusion,
        reranker=TermCoverageReranker(retrieval_weight=0.2),
        rerank_candidates=2,
    )

    results = hybrid.search("hybrid groundedness evaluation", top_k=1)

    assert [item.id for item in results] == ["b"]


def test_hybrid_configuration_fails_closed() -> None:
    source = FusionSource("one", StaticRetriever([]))
    with pytest.raises(ValueError, match="at least two"):
        ReciprocalRankFusionRetriever((source,))
    with pytest.raises(ValueError, match="unique"):
        ReciprocalRankFusionRetriever((source, source))
    with pytest.raises(ValueError, match="within"):
        TermCoverageReranker(retrieval_weight=1.1)
