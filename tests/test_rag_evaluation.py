import pytest

from app.query_service import ExtractiveAnswerGenerator, QueryService
from app.rag_evaluation import (
    RagRegressionExample,
    evaluate_rag_regression,
)
from app.regression_dataset import (
    RAG_REGRESSION_DATASET_PROVENANCE,
    RAG_REGRESSION_DATASET_VERSION,
    RAG_REGRESSION_DOCUMENTS,
    RAG_REGRESSION_EXAMPLES,
)
from app.retrieval import InMemoryRetriever


def test_reference_query_service_meets_regression_contract() -> None:
    service = QueryService(
        InMemoryRetriever(RAG_REGRESSION_DOCUMENTS),
        ExtractiveAnswerGenerator(),
        minimum_evidence_score=0.2,
    )

    metrics = evaluate_rag_regression(
        service,
        RAG_REGRESSION_DOCUMENTS,
        RAG_REGRESSION_EXAMPLES,
        top_k=2,
    )

    assert metrics.evaluated == 4
    assert metrics.citation_precision == pytest.approx(1.0)
    assert metrics.citation_recall == pytest.approx(1.0)
    assert metrics.abstention_accuracy == pytest.approx(1.0)
    assert metrics.supported_answer_rate == pytest.approx(1.0)
    assert metrics.answer_term_recall == pytest.approx(1.0)


def test_dataset_is_versioned_and_declares_synthetic_provenance() -> None:
    assert RAG_REGRESSION_DATASET_VERSION == "rag-regression-v1"
    assert "no proprietary" in RAG_REGRESSION_DATASET_PROVENANCE
    assert len({document.id for document in RAG_REGRESSION_DOCUMENTS}) == len(
        RAG_REGRESSION_DOCUMENTS
    )


def test_application_metrics_detect_irrelevant_citations() -> None:
    service = QueryService(
        InMemoryRetriever(RAG_REGRESSION_DOCUMENTS),
        ExtractiveAnswerGenerator(),
        minimum_evidence_score=0.0,
    )
    example = RagRegressionExample(
        question="rank payment",
        relevant_ids=frozenset({"retrieval-fusion"}),
        expected_answer_terms=frozenset({"rank"}),
    )

    metrics = evaluate_rag_regression(
        service,
        RAG_REGRESSION_DOCUMENTS,
        [example],
        top_k=2,
    )

    assert metrics.citation_precision < 1.0
    assert metrics.citation_recall == pytest.approx(1.0)


def test_regression_contract_rejects_invalid_cases_and_duplicate_documents() -> None:
    with pytest.raises(ValueError, match="abstention cases"):
        RagRegressionExample(
            question="unknown",
            relevant_ids=frozenset({"doc"}),
            should_abstain=True,
        )

    service = QueryService(InMemoryRetriever([]), ExtractiveAnswerGenerator())
    duplicate_documents = (
        RAG_REGRESSION_DOCUMENTS[0],
        RAG_REGRESSION_DOCUMENTS[0],
    )
    with pytest.raises(ValueError, match="duplicate regression document"):
        evaluate_rag_regression(
            service,
            duplicate_documents,
            [
                RagRegressionExample(
                    question="answerable",
                    relevant_ids=frozenset({"payments-idempotency"}),
                )
            ],
        )
