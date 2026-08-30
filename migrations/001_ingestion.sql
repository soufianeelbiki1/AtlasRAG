BEGIN;

CREATE TABLE IF NOT EXISTS rag_documents (
    document_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    fingerprint CHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT rag_documents_fingerprint_sha256
        CHECK (fingerprint ~ '^[0-9a-f]{64}$')
);

CREATE TABLE IF NOT EXISTS rag_chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES rag_documents(document_id) ON DELETE CASCADE,
    source TEXT NOT NULL,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    text TEXT NOT NULL CHECK (length(text) > 0),
    fingerprint CHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT rag_chunks_fingerprint_sha256
        CHECK (fingerprint ~ '^[0-9a-f]{64}$'),
    CONSTRAINT rag_chunks_document_ordinal_unique
        UNIQUE (document_id, ordinal)
);

CREATE INDEX IF NOT EXISTS rag_chunks_document_id_idx
    ON rag_chunks (document_id);

COMMIT;
