"""Evaluation primitives for deterministic retrieval regressions.

The evaluator is intentionally model- and infrastructure-independent: it scores
retriever output against a small, versionable relevance set so CI can catch
ranking regressions before introducing a vector database or LLM provider.
"""

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalExample:
    query: str
    relevant_ids: frozenset[str]

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("query must not be blank")
        if not self.relevant_ids:
            raise ValueError("relevant_ids must not be empty")


@dataclass(frozen=True)
class RetrievalMetrics:
    evaluated: int
    precision_at_k: float
    recall_at_k: float
    mean_reciprocal_rank: float


def evaluate_retrieval(
    retriever: object,
    examples: Iterable[RetrievalExample],
    *,
    top_k: int = 5,
) -> RetrievalMetrics:
    """Evaluate a retriever with precision/recall@k and MRR."""

    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    cases = list(examples)
    if not cases:
        raise ValueError("at least one retrieval example is required")

    precision_total = recall_total = mrr_total = 0.0
    for case in cases:
        results = retriever.search(case.query, top_k)
        ids = [chunk.id for chunk in results]
        hits = sum(result_id in case.relevant_ids for result_id in ids)
        precision_total += hits / top_k
        recall_total += hits / len(case.relevant_ids)

        first_relevant_rank = next(
            (rank for rank, result_id in enumerate(ids, start=1) if result_id in case.relevant_ids),
            None,
        )
        if first_relevant_rank is not None:
            mrr_total += 1 / first_relevant_rank

    count = len(cases)
    return RetrievalMetrics(
        evaluated=count,
        precision_at_k=precision_total / count,
        recall_at_k=recall_total / count,
        mean_reciprocal_rank=mrr_total / count,
    )
