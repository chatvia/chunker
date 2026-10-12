import asyncio
import json

import httpx2
import openai
import pytest

from app.config import Settings
from app.embeddings import OpenAICompatibleEmbedder
from app.errors import UpstreamError
from tests.conftest import make_settings


class FakeEmbeddingsAPI:
    """Mimics `POST /embeddings`, returning items out of order to check re-sorting."""

    def __init__(self, status: int = 200) -> None:
        self.status = status
        self.bodies: list[dict[str, object]] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        if self.status != 200:
            return httpx2.Response(self.status, json={"error": {"message": "boom"}})
        data = [
            {"object": "embedding", "index": i, "embedding": [float(len(text)), float(i)]}
            for i, text in enumerate(body["input"])
        ]
        return httpx2.Response(
            200,
            json={
                "object": "list",
                "model": body["model"],
                "data": list(reversed(data)),
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
        )


def _embedder(api: FakeEmbeddingsAPI, dimensions: int | None = None) -> OpenAICompatibleEmbedder:
    base_url = "http://embeddings.test/v1"
    embedder = OpenAICompatibleEmbedder(
        model="test-model", base_url=base_url, dimensions=dimensions, batch_size=2
    )
    # Route the SDK clients through the fake API instead of the network.
    transport = httpx2.MockTransport(api)
    embedder._client = openai.OpenAI(
        base_url=base_url,
        api_key="k",
        max_retries=0,
        http_client=httpx2.Client(transport=transport),
    )
    embedder._async_client = openai.AsyncOpenAI(
        base_url=base_url,
        api_key="k",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=transport),
    )
    return embedder


@pytest.mark.parametrize("use_async", [False, True])
def test_embeds_in_batches_preserving_order(use_async: bool) -> None:
    api = FakeEmbeddingsAPI()
    embedder = _embedder(api, dimensions=256)
    texts = ["a", "bb", "ccc", "dddd", "eeeee"]

    vectors = asyncio.run(embedder.aembed(texts)) if use_async else embedder.embed(texts)

    assert [v[0] for v in vectors] == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert [len(body["input"]) for body in api.bodies] == [2, 2, 1]  # type: ignore[arg-type]
    assert all(body["dimensions"] == 256 for body in api.bodies)
    assert all(body["encoding_format"] == "float" for body in api.bodies)


def test_dimensions_are_omitted_unless_configured() -> None:
    api = FakeEmbeddingsAPI()

    _embedder(api).embed(["a"])

    assert "dimensions" not in api.bodies[0]


def test_provider_errors_become_upstream_errors() -> None:
    with pytest.raises(UpstreamError, match="HTTP 401"):
        _embedder(FakeEmbeddingsAPI(status=401)).embed(["a"])


def test_from_settings_requires_a_model() -> None:
    assert OpenAICompatibleEmbedder.from_settings(make_settings()) is None
    embedder = OpenAICompatibleEmbedder.from_settings(
        make_settings(embeddings_model="nomic-embed-text", embeddings_base_url="http://x/v1")
    )
    assert embedder is not None
    assert embedder.model == "nomic-embed-text"


def test_empty_env_values_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDINGS_MODEL", "")
    monkeypatch.setenv("API_KEY", "")

    settings = Settings(_env_file=None)

    assert settings.embeddings_model is None
    assert settings.api_key is None
