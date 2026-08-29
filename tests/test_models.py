import pytest
from pydantic import ValidationError

from app.models import Citation, EvidenceChunk, QueryRequest, QueryResponse


def test_query_request_enforces_bounded_question_and_top_k() -> None:
    request = QueryRequest(question="How does hybrid retrieval work?", top_k=8)

    assert request.top_k == 8

    with pytest.raises(ValidationError):
        QueryRequest(question="hi")

    with pytest.raises(ValidationError):
        QueryRequest(question="valid question", top_k=0)

    with pytest.raises(ValidationError):
        QueryRequest(question="valid question", top_k=21)


def test_evidence_and_citation_scores_are_probabilities() -> None:
    EvidenceChunk(id="chunk-1", text="evidence", source="doc.md", score=0.5)
    Citation(source="doc.md", chunk_id="chunk-1", score=1.0)

    with pytest.raises(ValidationError):
        EvidenceChunk(id="chunk-1", text="evidence", source="doc.md", score=-0.01)

    with pytest.raises(ValidationError):
        Citation(source="doc.md", chunk_id="chunk-1", score=1.01)


def test_query_response_confidence_is_bounded() -> None:
    response = QueryResponse(
        answer="Grounded answer",
        citations=[Citation(source="doc.md", chunk_id="chunk-1", score=0.9)],
        grounded=True,
        confidence=0.9,
    )

    assert response.grounded is True

    with pytest.raises(ValidationError):
        QueryResponse(answer="x", citations=[], grounded=False, confidence=2.0)
