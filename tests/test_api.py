from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_identifies_reference_retrieval_mode() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "retrieval": "deterministic-reference"}


def test_query_endpoint_returns_grounded_citation_first_response() -> None:
    response = client.post(
        "/v1/query",
        json={"question": "How does AtlasRAG handle weak evidence?", "top_k": 4},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["grounded"] is True
    assert payload["citations"]
    assert all(citation["source"].startswith("atlasrag://") for citation in payload["citations"])


def test_query_endpoint_validates_request_contract() -> None:
    response = client.post("/v1/query", json={"question": "x", "top_k": 0})

    assert response.status_code == 422
