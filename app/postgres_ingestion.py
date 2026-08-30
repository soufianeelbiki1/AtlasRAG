"""Durable PostgreSQL ingestion persistence for AtlasRAG."""

from dataclasses import dataclass

import psycopg

from app.ingestion import (
    DeterministicChunker,
    DocumentChunk,
    DocumentInput,
    IngestionConflictError,
    IngestionResult,
    fingerprint_text,
)


@dataclass(frozen=True, slots=True)
class PostgresIngestionStore:
    database_url: str
    chunker: DeterministicChunker = DeterministicChunker()

    def ingest(self, document: DocumentInput) -> IngestionResult:
        document_fingerprint = fingerprint_text(document.text)
        candidate_chunks = self.chunker.chunk(document)

        with (
            psycopg.connect(self.database_url) as connection,
            connection.cursor() as cursor,
        ):
            cursor.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                (document.document_id,),
            )
            cursor.execute(
                """
                SELECT fingerprint
                FROM rag_documents
                WHERE document_id = %s
                FOR UPDATE
                """,
                (document.document_id,),
            )
            existing = cursor.fetchone()
            if existing is not None:
                if existing[0] != document_fingerprint:
                    raise IngestionConflictError(
                        "document_id was already ingested with different normalized content"
                    )
                return IngestionResult(
                    document_id=document.document_id,
                    document_fingerprint=document_fingerprint,
                    chunks=self._load_chunks(cursor, document.document_id),
                    replayed=True,
                )

            cursor.execute(
                """
                INSERT INTO rag_documents (document_id, source, fingerprint)
                VALUES (%s, %s, %s)
                """,
                (document.document_id, document.source, document_fingerprint),
            )
            cursor.executemany(
                """
                INSERT INTO rag_chunks (
                    id,
                    document_id,
                    source,
                    ordinal,
                    text,
                    fingerprint
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                [
                    (
                        chunk.id,
                        chunk.document_id,
                        chunk.source,
                        chunk.ordinal,
                        chunk.text,
                        chunk.fingerprint,
                    )
                    for chunk in candidate_chunks
                ],
            )

        return IngestionResult(
            document_id=document.document_id,
            document_fingerprint=document_fingerprint,
            chunks=candidate_chunks,
            replayed=False,
        )

    @staticmethod
    def _load_chunks(
        cursor: psycopg.Cursor[tuple[object, ...]],
        document_id: str,
    ) -> tuple[DocumentChunk, ...]:
        cursor.execute(
            """
            SELECT id, document_id, source, ordinal, text, fingerprint
            FROM rag_chunks
            WHERE document_id = %s
            ORDER BY ordinal
            """,
            (document_id,),
        )
        return tuple(
            DocumentChunk(
                id=str(row[0]),
                document_id=str(row[1]),
                source=str(row[2]),
                ordinal=int(row[3]),
                text=str(row[4]),
                fingerprint=str(row[5]),
            )
            for row in cursor.fetchall()
        )
