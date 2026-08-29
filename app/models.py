from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=4, ge=1, le=20)


class EvidenceChunk(BaseModel):
    id: str
    text: str
    source: str
    score: float = Field(ge=0.0, le=1.0)


class Citation(BaseModel):
    source: str
    chunk_id: str
    score: float = Field(ge=0.0, le=1.0)


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    grounded: bool
    confidence: float = Field(ge=0.0, le=1.0)
