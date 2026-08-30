from app.models import QueryRequest
from app.query_service import ExtractiveAnswerGenerator, QueryService
from app.retrieval import InMemoryRetriever, SeedDocument


def service(*, threshold: float = 0.25) -> QueryService:
    retriever = InMemoryRetriever(
        [
            SeedDocument(
                id="doc-1",
                source="policy.md",
                text="Refund requests require a valid payment reference.",
            ),
            SeedDocument(
                id="doc-2",
                source="ops.md",
                text="Timeout events should be reconciled before operator replay.",
            ),
        ]
    )
    return QueryService(
        retriever,
        ExtractiveAnswerGenerator(),
        minimum_evidence_score=threshold,
    )


def test_grounded_query_returns_only_used_evidence_as_citations() -> None:
    response = service().query(QueryRequest(question="refund payment reference", top_k=4))

    assert response.grounded is True
    assert response.confidence > 0
    assert "valid payment reference" in response.answer
    assert [citation.chunk_id for citation in response.citations] == ["doc-1"]
    assert response.citations[0].source == "policy.md"


def test_query_abstains_when_retrieval_has_no_overlap() -> None:
    response = service().query(QueryRequest(question="weather in casablanca", top_k=4))

    assert response.grounded is False
    assert response.confidence == 0.0
    assert response.citations == []
    assert "not have enough retrieved evidence" in response.answer


def test_query_abstains_when_evidence_is_below_threshold() -> None:
    response = service(threshold=0.9).query(
        QueryRequest(question="refund payment reference additional unrelated terms", top_k=4)
    )

    assert response.grounded is False
    assert response.citations == []


def test_invalid_evidence_threshold_is_rejected() -> None:
    try:
        service(threshold=1.1)
    except ValueError as exc:
        assert "within [0, 1]" in str(exc)
    else:
        raise AssertionError("expected invalid threshold to be rejected")
