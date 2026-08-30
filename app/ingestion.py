import hashlib
import re
from dataclasses import dataclass


class IngestionConflictError(ValueError):
    """Raised when a stable document identity is reused for different content."""


@dataclass(frozen=True, slots=True)
class DocumentInput:
    document_id: str
    source: str
    text: str

    def __post_init__(self) -> None:
        if not self.document_id.strip():
            raise ValueError("document_id must not be empty")
        if not self.source.strip():
            raise ValueError("source must not be empty")
        if not self.text.strip():
            raise ValueError("text must not be empty")


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    id: str
    document_id: str
    source: str
    ordinal: int
    text: str
    fingerprint: str


@dataclass(frozen=True, slots=True)
class IngestionResult:
    document_id: str
    document_fingerprint: str
    chunks: tuple[DocumentChunk, ...]
    replayed: bool


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def fingerprint_text(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


class DeterministicChunker:
    """Paragraph-aware deterministic chunker with stable content-derived chunk IDs."""

    def __init__(self, *, max_chars: int = 700) -> None:
        if max_chars < 80:
            raise ValueError("max_chars must be at least 80")
        self._max_chars = max_chars

    def chunk(self, document: DocumentInput) -> tuple[DocumentChunk, ...]:
        normalized = normalize_text(document.text)
        words = normalized.split(" ")
        groups: list[str] = []
        current: list[str] = []
        current_chars = 0

        for word in words:
            projected = current_chars + len(word) + (1 if current else 0)
            if current and projected > self._max_chars:
                groups.append(" ".join(current))
                current = [word]
                current_chars = len(word)
            else:
                current.append(word)
                current_chars = projected
        if current:
            groups.append(" ".join(current))

        chunks: list[DocumentChunk] = []
        for ordinal, text in enumerate(groups):
            chunk_fingerprint = fingerprint_text(text)
            chunk_id_material = f"{document.document_id}:{ordinal}:{chunk_fingerprint}"
            chunk_id = "chk_" + hashlib.sha256(chunk_id_material.encode("utf-8")).hexdigest()[:24]
            chunks.append(
                DocumentChunk(
                    id=chunk_id,
                    document_id=document.document_id,
                    source=document.source,
                    ordinal=ordinal,
                    text=text,
                    fingerprint=chunk_fingerprint,
                )
            )
        return tuple(chunks)


class InMemoryIngestionStore:
    """Reference idempotency store for deterministic ingestion tests and local use."""

    def __init__(self, chunker: DeterministicChunker | None = None) -> None:
        self._chunker = chunker or DeterministicChunker()
        self._documents: dict[str, IngestionResult] = {}

    def ingest(self, document: DocumentInput) -> IngestionResult:
        document_fingerprint = fingerprint_text(document.text)
        existing = self._documents.get(document.document_id)
        if existing is not None:
            if existing.document_fingerprint != document_fingerprint:
                raise IngestionConflictError(
                    "document_id was already ingested with different normalized content"
                )
            return IngestionResult(
                document_id=existing.document_id,
                document_fingerprint=existing.document_fingerprint,
                chunks=existing.chunks,
                replayed=True,
            )

        result = IngestionResult(
            document_id=document.document_id,
            document_fingerprint=document_fingerprint,
            chunks=self._chunker.chunk(document),
            replayed=False,
        )
        self._documents[document.document_id] = result
        return result
