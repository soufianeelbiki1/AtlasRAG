# AtlasRAG project context

AtlasRAG is the production AI/LLM engineering flagship. It should demonstrate measurable retrieval quality, citation-first responses, abstention under weak evidence, provider boundaries, deterministic evaluation, ingestion reliability, tenancy, cost/latency visibility, and explicit failure handling without claiming proprietary data or production traffic.

## Current state

- Deterministic lexical retriever behind a `Retriever` protocol.
- Pydantic query, evidence, citation, and response contracts.
- Retrieval evaluation primitives including precision/recall@k and MRR.
- Citation-first query application service with configurable minimum evidence threshold.
- Explicit abstention when retrieved evidence is absent/weak or grounded content is empty.
- `AnswerGenerator` provider port plus deterministic extractive reference generator.
- FastAPI `/health` and `/v1/query` reference endpoints.
- Tests cover retrieval, evaluation, grounded citations, abstention, and API validation.
- CI runs lint, formatting, compilation, and tests without external AI credentials.

## Guardrails

1. Never call a response grounded unless cited evidence actually supports the generated content path.
2. Retrieval metrics and generation metrics remain separate.
3. Provider adapters must sit behind stable ports; business logic must not depend on one LLM vendor.
4. Test and CI paths must remain deterministic and credential-free.
5. Multi-tenant identifiers must be enforced in persistence/retrieval boundaries before tenancy is claimed.
6. Cost/latency claims require measured traces rather than invented numbers.
7. Ingestion retries need durable idempotency before being described as production-grade.

## Priority sequence

- [x] deterministic retrieval contract
- [x] precision/recall@k and MRR evaluation primitives
- [x] citation-first query service
- [x] evidence-threshold abstention
- [x] deterministic generator/provider port
- [ ] document ingestion/chunking contracts with replay-safe document identity
- [ ] hybrid lexical/vector retrieval and reranking
- [ ] regression evaluation dataset with groundedness/answer-relevance scoring
- [ ] provider adapters with token/cost/latency accounting
- [ ] durable background ingestion jobs
- [ ] multi-tenant isolation
- [ ] OpenTelemetry/metrics and evaluation dashboard

## Next highest-value task

Add a deterministic ingestion/chunking pipeline with stable document fingerprints, chunk identities, duplicate/replay handling, and provenance metadata. Then add regression evaluation datasets that exercise retrieval and grounded-answer behavior without external model credentials.
