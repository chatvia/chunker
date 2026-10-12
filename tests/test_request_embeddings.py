"""Embeddings providers supplied per request, used when the server has none configured."""

from typing import Any, ClassVar

import pytest
from fastapi.testclient import TestClient

from app import embeddings
from tests.conftest import TWO_TOPICS, ClientFactory, FakeEmbedder

PROVIDER = {
    "base_url": "https://embeddings.example/v1",
    "api_key": "request-secret-key",
    "model": "request-model",
    "dimensions": 256,
    "batch_size": 8,
}
SEMANTIC = {"engine": "llamaindex", "strategy": "semantic", "buffer_size": 0}


class RecordingEmbedder(FakeEmbedder):
    instances: ClassVar[list[RecordingEmbedder]] = []

    def __init__(self, **options: Any) -> None:
        super().__init__()
        self.options = options
        self.model = str(options["model"])
        self.closed = False
        RecordingEmbedder.instances.append(self)

    async def aclose(self) -> None:
        self.closed = True


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch) -> type[RecordingEmbedder]:
    RecordingEmbedder.instances = []
    monkeypatch.setattr(embeddings, "OpenAICompatibleEmbedder", RecordingEmbedder)
    return RecordingEmbedder


def test_request_embeddings_power_semantic_chunking(
    bare_client: TestClient, provider: type[RecordingEmbedder]
) -> None:
    response = bare_client.post(
        "/v1/chunk",
        json={"text": TWO_TOPICS, "chunker": SEMANTIC, "embed": True, "embeddings": PROVIDER},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["embedding_model"] == "request-model"
    assert all(chunk["embedding"] for chunk in body["chunks"])
    [instance] = provider.instances
    assert instance.calls > 0
    assert instance.closed
    assert instance.options | {"timeout": None, "max_retries": None} == {
        "base_url": "https://embeddings.example/v1",
        "api_key": "request-secret-key",
        "model": "request-model",
        "dimensions": 256,
        "batch_size": 8,
        "timeout": None,
        "max_retries": None,
    }
    assert "request-secret-key" not in response.text


def test_embed_endpoint_accepts_request_embeddings(
    bare_client: TestClient, provider: type[RecordingEmbedder]
) -> None:
    response = bare_client.post("/v1/embed", json={"input": ["a cat"], "embeddings": PROVIDER})

    assert response.status_code == 200, response.text
    assert response.json()["model"] == "request-model"
    assert provider.instances[0].closed


def test_server_embeddings_take_precedence(client: TestClient) -> None:
    for path, payload in [
        ("/v1/chunk", {"text": TWO_TOPICS, "chunker": SEMANTIC, "embeddings": PROVIDER}),
        ("/v1/embed", {"input": ["a cat"], "embeddings": PROVIDER}),
    ]:
        response = client.post(path, json=payload)

        assert response.status_code == 400
        assert "configured on this server" in response.json()["detail"]


def test_request_embeddings_can_be_disabled(make_client: ClientFactory) -> None:
    client = make_client(allow_request_embeddings=False)

    response = client.post(
        "/v1/chunk", json={"text": TWO_TOPICS, "chunker": SEMANTIC, "embeddings": PROVIDER}
    )

    assert response.status_code == 400
    assert "ALLOW_REQUEST_EMBEDDINGS" in response.json()["detail"]


def test_semantic_without_any_embeddings_explains_both_options(bare_client: TestClient) -> None:
    response = bare_client.post("/v1/chunk", json={"text": TWO_TOPICS, "chunker": SEMANTIC})

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "EMBEDDINGS_MODEL" in detail
    assert "`embeddings`" in detail


@pytest.mark.parametrize(
    "provider_config",
    [
        {"base_url": "https://embeddings.example/v1"},
        {"model": "m", "base_url": "ftp://embeddings.example"},
        {"model": "m", "batch_size": 0},
        {"model": "m", "unexpected": True},
    ],
)
def test_invalid_request_embeddings_are_rejected(
    bare_client: TestClient, provider_config: dict[str, Any]
) -> None:
    response = bare_client.post(
        "/v1/chunk", json={"text": "hi", "chunker": SEMANTIC, "embeddings": provider_config}
    )

    assert response.status_code == 422


def test_base_url_defaults_to_openai(
    bare_client: TestClient, provider: type[RecordingEmbedder]
) -> None:
    bare_client.post("/v1/embed", json={"input": ["a"], "embeddings": {"model": "m"}})

    assert provider.instances[0].options["base_url"] == "https://api.openai.com/v1"
    assert provider.instances[0].options["api_key"] is None
