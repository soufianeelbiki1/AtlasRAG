import pytest

from app.evaluation import RetrievalExample, evaluate_retrieval
from app.retrieval import InMemoryRetriever, SeedDocument


def test_evaluation_reports_ranking_metrics() -> None:
    retriever = InMemoryRetriever(
        [
            SeedDocument("a", "iso 8583 authorization", "a.md"),
            SeedDocument("b", "ledger reconciliation", "b.md"),
            SeedDocument("c", "unrelated text", "c.md"),
        ]
    )

    metrics = evaluate_retrieval(
        retriever,
        [RetrievalExample("iso authorization", frozenset({"a"}))],
        top_k=2,
    )

    assert metrics.evaluated == 1
    assert metrics.precision_at_k == pytest.approx(0.5)
    assert metrics.recall_at_k == pytest.approx(1.0)
    assert metrics.mean_reciprocal_rank == pytest.approx(1.0)


def test_evaluation_rejects_empty_dataset_and_invalid_k() -> None:
    retriever = InMemoryRetriever([])
    with pytest.raises(ValueError, match="at least one"):
        evaluate_retrieval(retriever, [], top_k=1)
    with pytest.raises(ValueError, match="top_k"):
        evaluate_retrieval(
            retriever,
            [RetrievalExample("query", frozenset({"missing"}))],
            top_k=0,
        )
