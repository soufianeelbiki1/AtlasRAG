from fastapi import FastAPI

from app.models import QueryRequest, QueryResponse
from app.query_service import ExtractiveAnswerGenerator, QueryService
from app.retrieval import InMemoryRetriever, SeedDocument

app = FastAPI(title="AtlasRAG", version="0.1.0")

_reference_documents = (
    SeedDocument(
        id="retrieval-quality",
        source="atlasrag://reference/retrieval",
        text="AtlasRAG measures retrieval quality independently from answer generation.",
    ),
    SeedDocument(
        id="abstention",
        source="atlasrag://reference/trust",
        text="AtlasRAG abstains when retrieved evidence is too weak to support an answer.",
    ),
    SeedDocument(
        id="citations",
        source="atlasrag://reference/citations",
        text="AtlasRAG returns source citations for evidence used in grounded answers.",
    ),
)
_service = QueryService(
    InMemoryRetriever(_reference_documents),
    ExtractiveAnswerGenerator(),
    minimum_evidence_score=0.25,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "retrieval": "deterministic-reference"}


@app.post("/v1/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    return _service.query(request)
