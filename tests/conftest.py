from collections.abc import Callable, Iterator, Sequence
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.embeddings import Embedder
from app.main import create_app

TOPICS = (
    {"cat", "cats", "kitten", "kittens", "feline", "whiskers", "meow", "purr"},
    {"rocket", "rockets", "orbit", "planet", "moon", "space", "nasa", "astronaut"},
)

TWO_TOPICS = (
    "Cats are small feline animals. A kitten will meow and purr. Cats groom their whiskers. "
    "Every cat loves a warm kitten nap. "
    "The rocket reached orbit around the planet. NASA sent an astronaut to the moon. "
    "Space travel needs a powerful rocket. The moon orbit took three days."
)


class FakeEmbedder:
    """Deterministic embeddings: one dimension per topic, so topic shifts are detectable."""

    model = "fake-embedding"

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        return [self._vector(text) for text in texts]

    async def aembed(self, texts: Sequence[str]) -> list[list[float]]:
        return self.embed(texts)

    @staticmethod
    def _vector(text: str) -> list[float]:
        words = {word.strip(".,!?").lower() for word in text.split()}
        return [len(words & topic) + 0.01 for topic in TOPICS] + [0.01]


def make_settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)


ClientFactory = Callable[..., TestClient]


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def make_client() -> Iterator[ClientFactory]:
    """Build clients with custom settings; lifespans are closed at teardown."""
    clients: list[TestClient] = []

    def factory(*, embedder: Embedder | None = None, **kwargs: Any) -> TestClient:
        transport = kwargs.pop("webhook_transport", None)
        app = create_app(make_settings(**kwargs), embedder=embedder, webhook_transport=transport)
        client = TestClient(app)
        client.__enter__()
        clients.append(client)
        return client

    yield factory
    for client in clients:
        client.__exit__(None, None, None)


@pytest.fixture
def client(make_client: ClientFactory, embedder: FakeEmbedder) -> TestClient:
    return make_client(embedder=embedder)


@pytest.fixture
def bare_client(make_client: ClientFactory) -> TestClient:
    """A client for a server without embeddings or webhooks configured."""
    return make_client()
