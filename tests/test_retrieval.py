import pytest

from app.retrieval import InMemoryRetriever, SeedDocument


@pytest.fixture
def retriever() -> InMemoryRetriever:
    return InMemoryRetriever(
        [
            SeedDocument(
                id="payments-1",
                text="Payment retries require durable idempotency and explicit failure handling",
                source="payments.md",
            ),
            SeedDocument(
                id="retrieval-1",
                text="Hybrid retrieval combines lexical and semantic signals",
                source="retrieval.md",
            ),
            SeedDocument(
                id="retrieval-2",
                text="Retrieval evaluation measures ranking quality and grounded answers",
                source="evaluation.md",
            ),
        ]
    )


def test_search_ranks_by_query_term_overlap(retriever: InMemoryRetriever) -> None:
    results = retriever.search("retrieval evaluation quality", top_k=3)

    assert [chunk.id for chunk in results] == ["retrieval-2", "retrieval-1"]
    assert results[0].score == pytest.approx(1.0)
    assert results[1].score == pytest.approx(1 / 3)


def test_search_is_case_insensitive_and_respects_top_k(retriever: InMemoryRetriever) -> None:
    results = retriever.search("RETRIEVAL", top_k=1)

    assert len(results) == 1
    assert results[0].id == "retrieval-1"
    assert results[0].score == pytest.approx(1.0)


def test_search_drops_documents_without_overlap(retriever: InMemoryRetriever) -> None:
    assert retriever.search("passport illumination", top_k=5) == []


def test_empty_query_returns_no_results(retriever: InMemoryRetriever) -> None:
    assert retriever.search("   ", top_k=5) == []


def test_ties_preserve_document_order() -> None:
    retriever = InMemoryRetriever(
        [
            SeedDocument(id="first", text="alpha beta", source="a.md"),
            SeedDocument(id="second", text="alpha gamma", source="b.md"),
        ]
    )

    results = retriever.search("alpha", top_k=2)

    assert [chunk.id for chunk in results] == ["first", "second"]
