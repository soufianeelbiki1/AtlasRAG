import pytest

from app.ingestion import (
    DeterministicChunker,
    DocumentInput,
    IngestionConflictError,
    InMemoryIngestionStore,
    fingerprint_text,
)


def test_text_fingerprint_normalizes_whitespace() -> None:
    assert fingerprint_text("alpha  beta\n gamma") == fingerprint_text("alpha beta gamma")


def test_chunk_ids_are_deterministic_for_same_document_content() -> None:
    document = DocumentInput(
        document_id="policy-001",
        source="policy.md",
        text=" ".join(f"term-{index}" for index in range(120)),
    )
    chunker = DeterministicChunker(max_chars=120)

    first = chunker.chunk(document)
    second = chunker.chunk(document)

    assert len(first) > 1
    assert first == second
    assert [chunk.ordinal for chunk in first] == list(range(len(first)))
    assert len({chunk.id for chunk in first}) == len(first)


def test_exact_ingestion_replay_returns_same_chunks_without_duplication() -> None:
    store = InMemoryIngestionStore(DeterministicChunker(max_chars=100))
    document = DocumentInput(
        document_id="manual-001",
        source="manual.txt",
        text="A deterministic ingestion pipeline must make retries safe. " * 8,
    )

    first = store.ingest(document)
    replay = store.ingest(
        DocumentInput(
            document_id=document.document_id,
            source=document.source,
            text="  A deterministic ingestion pipeline must make retries safe.\n" * 8,
        )
    )

    assert first.replayed is False
    assert replay.replayed is True
    assert replay.document_fingerprint == first.document_fingerprint
    assert replay.chunks == first.chunks


def test_document_identity_conflict_is_explicit() -> None:
    store = InMemoryIngestionStore()
    store.ingest(DocumentInput("doc-1", "a.md", "original content"))

    with pytest.raises(IngestionConflictError, match="different normalized content"):
        store.ingest(DocumentInput("doc-1", "a.md", "changed content"))


def test_document_contract_rejects_empty_identity_source_and_text() -> None:
    with pytest.raises(ValueError, match="document_id"):
        DocumentInput("", "a.md", "text")
    with pytest.raises(ValueError, match="source"):
        DocumentInput("doc", "", "text")
    with pytest.raises(ValueError, match="text"):
        DocumentInput("doc", "a.md", "   ")
