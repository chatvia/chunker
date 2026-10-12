import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app import __version__
from app.engines import CHUNKERS
from tests.conftest import ClientFactory, FakeEmbedder, make_settings


def test_health_is_public(make_client: ClientFactory) -> None:
    client = make_client(api_key="secret")

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__}


@pytest.mark.parametrize(
    ("headers", "expected_status"),
    [
        ({}, 401),
        ({"Authorization": "Bearer wrong"}, 401),
        ({"Authorization": "Bearer secret"}, 200),
    ],
)
def test_api_key_protects_v1(
    make_client: ClientFactory, headers: dict[str, str], expected_status: int
) -> None:
    client = make_client(api_key="secret")

    response = client.post("/v1/chunk", json={"text": "hello world"}, headers=headers)

    assert response.status_code == expected_status


def test_engines_lists_every_strategy_with_its_schema(client: TestClient) -> None:
    body = client.get("/v1/engines").json()

    strategies = {(c["engine"], c["strategy"]) for c in body["chunkers"]}
    assert len(strategies) == len(CHUNKERS) == 14
    assert {("chonkie", "semantic"), ("llamaindex", "semantic")} == {
        (c["engine"], c["strategy"]) for c in body["chunkers"] if c["requires_embeddings"]
    }
    assert all(c["description"] and c["options"]["properties"] for c in body["chunkers"])
    parsers = {p["engine"]: p for p in body["parsers"]}
    assert set(parsers) == {"unstructured", "llamaindex"}
    assert ".pdf" in parsers["unstructured"]["extensions"]
    assert body["embeddings"] == {"configured": True, "model": "fake-embedding"}
    assert "cl100k_base" in body["encodings"]


def test_engines_reports_missing_embeddings(bare_client: TestClient) -> None:
    body = bare_client.get("/v1/engines").json()

    assert body["embeddings"] == {"configured": False, "model": None}


def test_embed_returns_vectors_in_order(client: TestClient) -> None:
    response = client.post("/v1/embed", json={"input": ["a cat", "a rocket"]})

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "fake-embedding"
    assert [item["index"] for item in body["data"]] == [0, 1]
    assert body["data"][0]["embedding"] == FakeEmbedder._vector("a cat")


@pytest.mark.parametrize("payload", [{"input": []}, {"input": [""]}, {"input": "text"}])
def test_embed_validates_input(client: TestClient, payload: dict[str, object]) -> None:
    assert client.post("/v1/embed", json=payload).status_code == 422


def test_embed_requires_configuration(bare_client: TestClient) -> None:
    response = bare_client.post("/v1/embed", json={"input": ["hello"]})

    assert response.status_code == 400
    assert "EMBEDDINGS_MODEL" in response.json()["detail"]


def test_settings_reject_malformed_webhook_secret() -> None:
    with pytest.raises(ValidationError, match="base64"):
        make_settings(webhook_secret="whsec_not base64!")


def test_openapi_schema_builds(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()

    assert {"/v1/chunk", "/v1/chunk/file", "/v1/parse", "/v1/embed", "/v1/engines"} <= set(
        schema["paths"]
    )
