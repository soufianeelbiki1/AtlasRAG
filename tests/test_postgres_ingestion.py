import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import psycopg
import pytest

from app.ingestion import DeterministicChunker, DocumentInput, IngestionConflictError
from app.postgres_ingestion import PostgresIngestionStore

DATABASE_URL = os.getenv("DATABASE_URL")
pytestmark = pytest.mark.skipif(DATABASE_URL is None, reason="DATABASE_URL is not configured")


@pytest.fixture(autouse=True)
def clean_database() -> None:
    assert DATABASE_URL is not None
    migration = Path("migrations/001_ingestion.sql").read_text(encoding="utf-8")
    with psycopg.connect(DATABASE_URL, autocommit=True) as connection:
        connection.execute(migration)
        connection.execute("TRUNCATE rag_chunks, rag_documents")


def test_postgres_ingestion_replay_is_durable_across_store_instances() -> None:
    assert DATABASE_URL is not None
    document = DocumentInput(
        "policy-001",
        "policy.md",
        "Durable ingestion retries must preserve deterministic chunk identity. " * 8,
    )
    first_store = PostgresIngestionStore(
        DATABASE_URL,
        DeterministicChunker(max_chars=120),
    )
    second_store = PostgresIngestionStore(
        DATABASE_URL,
        DeterministicChunker(max_chars=120),
    )

    first = first_store.ingest(document)
    replay = second_store.ingest(document)

    assert first.replayed is False
    assert replay.replayed is True
    assert replay.document_fingerprint == first.document_fingerprint
    assert replay.chunks == first.chunks


def test_postgres_ingestion_conflict_is_explicit() -> None:
    assert DATABASE_URL is not None
    store = PostgresIngestionStore(DATABASE_URL)
    store.ingest(DocumentInput("doc-1", "a.md", "original content"))

    with pytest.raises(IngestionConflictError, match="different normalized content"):
        store.ingest(DocumentInput("doc-1", "a.md", "changed content"))


def test_concurrent_replay_creates_one_document_and_one_chunk_set() -> None:
    assert DATABASE_URL is not None
    document = DocumentInput(
        "concurrent-001",
        "manual.md",
        "Concurrent retries should serialize by stable document identity. " * 10,
    )

    def ingest_once() -> bool:
        return (
            PostgresIngestionStore(
                DATABASE_URL,
                DeterministicChunker(max_chars=100),
            )
            .ingest(document)
            .replayed
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        replayed = list(executor.map(lambda _: ingest_once(), range(4)))

    assert replayed.count(False) == 1
    assert replayed.count(True) == 3

    with psycopg.connect(DATABASE_URL) as connection:
        document_count = connection.execute(
            "SELECT count(*) FROM rag_documents WHERE document_id = %s",
            (document.document_id,),
        ).fetchone()
        chunk_count = connection.execute(
            "SELECT count(*) FROM rag_chunks WHERE document_id = %s",
            (document.document_id,),
        ).fetchone()

    assert document_count is not None and document_count[0] == 1
    assert chunk_count is not None and chunk_count[0] > 1
