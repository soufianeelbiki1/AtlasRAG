"""Versioned credential-free regression dataset for AtlasRAG."""

from app.rag_evaluation import RagRegressionExample
from app.retrieval import SeedDocument

RAG_REGRESSION_DATASET_VERSION = "rag-regression-v1"
RAG_REGRESSION_DATASET_PROVENANCE = (
    "Hand-authored synthetic engineering facts for deterministic CI; no proprietary "
    "documents, user data, or production query logs."
)

RAG_REGRESSION_DOCUMENTS = (
    SeedDocument(
        id="payments-idempotency",
        source="atlasrag://regression/payments",
        text=(
            "Durable payment idempotency uses request fingerprints and database uniqueness "
            "constraints so exact retries can be replayed safely."
        ),
    ),
    SeedDocument(
        id="retrieval-fusion",
        source="atlasrag://regression/retrieval",
        text=(
            "Reciprocal rank fusion combines ranked retrieval lists without assuming raw "
            "backend relevance scores are calibrated."
        ),
    ),
    SeedDocument(
        id="grounded-abstention",
        source="atlasrag://regression/trust",
        text=(
            "A citation-first query service should abstain when retrieved evidence is too "
            "weak to support an answer."
        ),
    ),
)

RAG_REGRESSION_EXAMPLES = (
    RagRegressionExample(
        question="How do durable payment retries stay safe?",
        relevant_ids=frozenset({"payments-idempotency"}),
        expected_answer_terms=frozenset({"idempotency", "fingerprints", "uniqueness"}),
    ),
    RagRegressionExample(
        question="Why use reciprocal rank fusion?",
        relevant_ids=frozenset({"retrieval-fusion"}),
        expected_answer_terms=frozenset({"rank", "fusion", "calibrated"}),
    ),
    RagRegressionExample(
        question="When should a citation first service abstain?",
        relevant_ids=frozenset({"grounded-abstention"}),
        expected_answer_terms=frozenset({"abstain", "evidence"}),
    ),
    RagRegressionExample(
        question="What is the passport office opening time?",
        relevant_ids=frozenset(),
        should_abstain=True,
    ),
)
