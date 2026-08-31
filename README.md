# AtlasRAG

**Evaluation-driven RAG engineering with durable ingestion, citation-first answers, abstention, hybrid rank fusion, reranking, and provider cost/latency accounting.**

AtlasRAG is the applied AI/LLM flagship in this portfolio. It focuses on the engineering around retrieval-augmented generation rather than a generic chat-with-PDFs demo: retrieval contracts, ingestion idempotency, measurable evaluation, citation support, weak-evidence abstention, provider boundaries, and operational cost/latency visibility.

> This repository is a reference implementation using deterministic test paths and synthetic/versioned evaluation data. It does not claim production traffic, proprietary corpora, semantic-vector quality, or compliance certification.

## What is implemented

- FastAPI `/health` and `/v1/query` endpoints with typed Pydantic contracts.
- Citation-first query service with configurable minimum-evidence threshold.
- Explicit abstention when evidence is absent, too weak, or cannot support a grounded response path.
- Deterministic lexical retriever behind a stable `Retriever` protocol.
- Hybrid retrieval orchestration using reciprocal-rank fusion across heterogeneous retriever ports.
- Cross-source chunk identity validation and a separate reranker port.
- Credential-free term-coverage reference reranker used to exercise the reranking contract in CI without pretending it is a production semantic model.
- Retrieval evaluation primitives including precision@k, recall@k, and MRR.
- Versioned RAG regression dataset with application-level citation precision/recall, abstention accuracy, expected-term recall, and answer-support checks.
- `AnswerGenerator` provider abstraction plus deterministic extractive reference generator.
- Deterministic ingestion with normalized SHA-256 document fingerprints, stable content-derived chunk IDs, provenance, and chunk ordinals.
- Durable PostgreSQL ingestion store with uniqueness constraints and transaction-scoped advisory locking for replay-safe concurrent retries.
- Exact document replay reconstructed from durable rows; changed content under a stable document ID is rejected rather than silently overwritten.
- Provider usage accounting for token consumption, estimated cost, and latency without embedding vendor logic into the application layer.
- Automated tests and GitHub Actions CI with no external AI credentials required.

## Architecture

```text
Documents
   |
   v
Normalization + fingerprinting
   |
   v
Deterministic chunking
   |
   +--> durable PostgreSQL document/chunk store
   |
   v
Retriever ports
   | lexical reference retrieval
   | future semantic/vector adapters
   |
   v
Reciprocal-rank fusion
   |
   v
Reranker port
   |
   v
Evidence threshold
   | weak evidence -> abstain
   | sufficient evidence
   v
AnswerGenerator provider port
   |
   v
Citation-first response
   |
   +--> evaluation metrics
   +--> usage / latency / cost accounting
```

The semantic/vector adapter is deliberately still an extension point. The repository does not claim pgvector quality merely because the architecture has a retrieval port for it.

## Trust and evaluation

AtlasRAG keeps different quality questions separate:

- **retrieval quality** — precision/recall@k and MRR;
- **citation quality** — whether cited chunks correspond to expected evidence;
- **abstention behavior** — whether weak/insufficient evidence produces a refusal to fabricate support;
- **answer support** — whether the generated answer content is supported by cited evidence under the repository's deterministic evaluation rules;
- **provider operations** — token use, estimated cost, and latency.

These application metrics are intentionally not labeled as model-based semantic groundedness scores.

## Durable ingestion semantics

Document identity is based on normalized content fingerprints and stable document IDs. The PostgreSQL implementation protects concurrent retries with an advisory lock and database constraints:

```text
same document ID + same normalized content
    -> replay-safe no-op / durable reconstruction

same document ID + changed normalized content
    -> explicit conflict
```

This is a bounded idempotency claim for repository ingestion, not an "exactly once" claim across arbitrary external systems.

## Run locally

```bash
python -m pip install -e '.[dev]'
ruff check .
ruff format --check .
pytest -q
```

The deterministic CI path requires no paid LLM or embedding-provider credentials.

## Portfolio signal

AtlasRAG demonstrates:

- RAG/LLM system design beyond API wrappers;
- durable ingestion and idempotency reasoning;
- information-retrieval evaluation;
- hybrid rank fusion and reranker abstractions;
- citation/abstention product semantics;
- provider-agnostic architecture;
- cost and latency accounting;
- explicit claim boundaries and reproducible regression testing.

## Next engineering milestone

Implement a measured semantic/vector retriever behind the existing port, evaluate it against the versioned regression dataset, and add durable background ingestion plus tenant isolation before claiming those capabilities.

## License

MIT
