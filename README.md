# AtlasRAG

AtlasRAG is a FastAPI RAG backend with durable document ingestion, citation-aware responses, abstention on weak evidence, hybrid rank fusion, reranking hooks and regression evaluation.

The repository is built so retrieval, generation and evaluation can be tested independently. The default test path is deterministic and does not require paid model credentials.

## Query path

```text
question
  -> retriever(s)
  -> reciprocal-rank fusion
  -> reranker
  -> evidence threshold
  -> answer generator
  -> citations + usage metrics
```

If the retrieved evidence is missing or too weak, the query service abstains instead of producing an unsupported answer.

## Evaluation demo

Generate a standalone HTML regression report:

```bash
python -m app.demo_report --output build/atlasrag-evaluation.html
```

Open the generated file in a browser. It shows the versioned deterministic regression cases, expected evidence, grounded/abstained outcome, returned citations and the extractive answer alongside aggregate citation precision/recall, abstention accuracy and supported-answer rate.

The report is generated from the same credential-free regression path exercised in tests. Its metrics compare returned chunk IDs and deterministic answer support against hand-authored expectations; they are application regression checks, not model-based semantic groundedness scores.

## Retrieval and evaluation

- lexical reference retriever behind a `Retriever` protocol;
- reciprocal-rank fusion across multiple retriever ports;
- separate reranker interface;
- precision@k, recall@k and MRR;
- versioned regression data for citation precision/recall, abstention behavior, expected answer terms and evidence support.

A semantic/vector retriever is not implemented yet, so the repository does not present hybrid orchestration as measured vector-search quality.

## Durable ingestion

Documents are normalized and fingerprinted with SHA-256. Chunk IDs are derived deterministically from content and position.

The PostgreSQL ingestion store uses database constraints and a transaction-scoped advisory lock so concurrent retries for the same document ID are serialized. Replaying the same normalized content reconstructs the stored result; reusing a stable document ID with changed content returns a conflict.

## Provider accounting

The answer-generator boundary records provider-reported token usage, estimated cost and latency without coupling the application layer to one provider.

## API

```text
GET  /health
POST /v1/query
```

## Run and test

```bash
python -m pip install -e '.[dev]'
ruff check .
ruff format --check .
pytest -q
python -m app.demo_report --output build/atlasrag-evaluation.html
```

CI runs without external LLM or embedding credentials.

## Limitations

- semantic/vector retrieval is still an extension point;
- background ingestion jobs are not durable yet;
- tenant isolation is not implemented yet;
- regression metrics are application-level checks, not a claim of general semantic groundedness;
- the demo report currently shows deterministic regression usage rather than a live paid-model session.

## Roadmap

1. Add a measured semantic/vector retriever and compare it against the regression dataset.
2. Add durable background ingestion jobs.
3. Add tenant isolation at persistence and retrieval boundaries.
4. Add provider usage/cost/latency panels backed by a configured provider run.
5. Add OpenTelemetry metrics and traces around ingestion and query execution.

## License

MIT
