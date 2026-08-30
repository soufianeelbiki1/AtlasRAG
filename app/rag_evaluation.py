"""Deterministic application-level RAG regression metrics.

Retrieval metrics remain in app.evaluation. This module evaluates query-service
behavior: citation correctness, abstention, answer support by cited evidence,
and expected-term relevance. It does not claim model-based semantic
groundedness or human-quality judgment.
"""

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from app.models import QueryRequest
from app.query_service import QueryService
from app.retrieval import SeedDocument


@dataclass(frozen=True, slots=True)
class RagRegressionExample:
    question: str
    relevant_ids: frozenset[str]
    expected_answer_terms: frozenset[str] = frozenset()
    should_abstain: bool = False

    def __post_init__(self) -> None:
        if not self.question.strip():
            raise ValueError("question must not be blank")
        if self.should_abstain and self.relevant_ids:
            raise ValueError("abstention cases must not declare relevant_ids")
        if not self.should_abstain and not self.relevant_ids:
            raise ValueError("answerable cases require relevant_ids")


@dataclass(frozen=True, slots=True)
class RagRegressionMetrics:
    evaluated: int
    citation_precision: float
    citation_recall: float
    abstention_accuracy: float
    supported_answer_rate: float
    answer_term_recall: float


def evaluate_rag_regression(
    service: QueryService,
    documents: Iterable[SeedDocument],
    examples: Iterable[RagRegressionExample],
    *,
    top_k: int = 4,
) -> RagRegressionMetrics:
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    cases = tuple(examples)
    if not cases:
        raise ValueError("at least one RAG regression example is required")

    document_by_id = _index_documents(documents)
    citation_precision_total = 0.0
    citation_recall_total = 0.0
    abstention_correct = 0
    supported_answers = 0
    answerable_responses = 0
    term_recall_total = 0.0
    term_cases = 0

    for case in cases:
        response = service.query(QueryRequest(question=case.question, top_k=top_k))
        cited_ids = [citation.chunk_id for citation in response.citations]

        if case.should_abstain:
            if not response.grounded and not cited_ids:
                abstention_correct += 1
            continue

        if response.grounded:
            abstention_correct += 1
        hits = sum(chunk_id in case.relevant_ids for chunk_id in cited_ids)
        if cited_ids:
            citation_precision_total += hits / len(cited_ids)
        citation_recall_total += hits / len(case.relevant_ids)

        if response.grounded:
            answerable_responses += 1
            evidence_text = " ".join(
                document_by_id[chunk_id].text
                for chunk_id in cited_ids
                if chunk_id in document_by_id
            )
            if _normalized(response.answer) in _normalized(evidence_text):
                supported_answers += 1

            if case.expected_answer_terms:
                term_cases += 1
                answer_terms = set(_tokens(response.answer))
                expected = {_normalized(term) for term in case.expected_answer_terms}
                term_recall_total += sum(term in answer_terms for term in expected) / len(expected)

    answerable_cases = sum(not case.should_abstain for case in cases)
    return RagRegressionMetrics(
        evaluated=len(cases),
        citation_precision=(
            citation_precision_total / answerable_cases if answerable_cases else 0.0
        ),
        citation_recall=citation_recall_total / answerable_cases if answerable_cases else 0.0,
        abstention_accuracy=abstention_correct / len(cases),
        supported_answer_rate=(
            supported_answers / answerable_responses if answerable_responses else 0.0
        ),
        answer_term_recall=term_recall_total / term_cases if term_cases else 0.0,
    )


def _index_documents(documents: Iterable[SeedDocument]) -> Mapping[str, SeedDocument]:
    indexed: dict[str, SeedDocument] = {}
    for document in documents:
        if document.id in indexed:
            raise ValueError(f"duplicate regression document id {document.id}")
        indexed[document.id] = document
    return indexed


def _normalized(text: str) -> str:
    return " ".join(_tokens(text))


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())
