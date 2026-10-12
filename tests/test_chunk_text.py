from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.conftest import TWO_TOPICS, FakeEmbedder

ARTICLE = "\n\n".join(
    f"# Section {n}\n\nParagraph {n} talks about topic {n}. It has a few sentences. "
    f"Each sentence adds a little more detail about item {n}, so the text grows."
    for n in range(1, 13)
)

STRATEGIES: list[dict[str, Any]] = [
    {"engine": "langchain", "strategy": "recursive", "chunk_size": 40, "chunk_overlap": 8},
    {
        "engine": "langchain",
        "strategy": "recursive",
        "chunk_size": 200,
        "length_unit": "characters",
    },
    {"engine": "langchain", "strategy": "recursive", "chunk_size": 80, "language": "markdown"},
    {"engine": "langchain", "strategy": "character", "chunk_size": 40, "chunk_overlap": 0},
    {"engine": "langchain", "strategy": "token", "chunk_size": 40, "chunk_overlap": 8},
    {"engine": "langchain", "strategy": "markdown", "chunk_size": 40},
    {"engine": "llamaindex", "strategy": "sentence", "chunk_size": 40, "chunk_overlap": 8},
    {"engine": "llamaindex", "strategy": "token", "chunk_size": 40, "chunk_overlap": 8},
    {"engine": "llamaindex", "strategy": "markdown"},
    {"engine": "chonkie", "strategy": "token", "chunk_size": 40, "chunk_overlap": 8},
    {"engine": "chonkie", "strategy": "sentence", "chunk_size": 40},
    {"engine": "chonkie", "strategy": "recursive", "chunk_size": 40},
    {"engine": "unstructured", "strategy": "basic", "max_characters": 200},
    {"engine": "unstructured", "strategy": "by_title", "max_characters": 200},
]


def _id(config: dict[str, Any]) -> str:
    return "-".join(str(v) for v in config.values())


@pytest.mark.parametrize("chunker", STRATEGIES, ids=_id)
def test_every_strategy_chunks_text(client: TestClient, chunker: dict[str, Any]) -> None:
    response = client.post("/v1/chunk", json={"text": ARTICLE, "chunker": chunker})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["chunker"] == {"engine": chunker["engine"], "strategy": chunker["strategy"]}
    assert body["total_chunks"] == len(body["chunks"]) > 1
    assert body["document"]["characters"] == len(ARTICLE)
    for index, chunk in enumerate(body["chunks"]):
        assert chunk["index"] == index
        assert chunk["text"].strip()
        assert chunk["token_count"] > 0
        assert chunk["embedding"] is None
        if chunk["start_index"] is not None:
            assert ARTICLE[chunk["start_index"] : chunk["end_index"]] == chunk["text"]


@pytest.mark.parametrize(
    "chunker",
    [s for s in STRATEGIES if s["engine"] != "unstructured" and s["strategy"] != "markdown"],
    ids=_id,
)
def test_verbatim_strategies_report_offsets(client: TestClient, chunker: dict[str, Any]) -> None:
    body = client.post("/v1/chunk", json={"text": ARTICLE, "chunker": chunker}).json()

    starts = [chunk["start_index"] for chunk in body["chunks"]]
    assert None not in starts
    assert starts == sorted(starts)


def test_default_chunker_is_langchain_recursive(client: TestClient) -> None:
    body = client.post("/v1/chunk", json={"text": ARTICLE}).json()

    assert body["chunker"] == {"engine": "langchain", "strategy": "recursive"}


def test_markdown_chunks_carry_headers(client: TestClient) -> None:
    langchain = client.post(
        "/v1/chunk",
        json={"text": ARTICLE, "chunker": {"engine": "langchain", "strategy": "markdown"}},
    ).json()
    llamaindex = client.post(
        "/v1/chunk",
        json={"text": ARTICLE, "chunker": {"engine": "llamaindex", "strategy": "markdown"}},
    ).json()

    assert langchain["chunks"][0]["metadata"]["headers"] == {"h1": "Section 1"}
    assert llamaindex["chunks"][0]["metadata"]["header_path"] == "/"
    assert llamaindex["total_chunks"] == 12


def test_encoding_changes_token_counts(client: TestClient) -> None:
    counts = {
        encoding: client.post("/v1/chunk", json={"text": ARTICLE, "encoding": encoding}).json()[
            "document"
        ]["token_count"]
        for encoding in ("cl100k_base", "o200k_base", "p50k_base")
    }

    assert all(count > 0 for count in counts.values())
    assert len(set(counts.values())) > 1


def test_special_tokens_are_treated_as_text(client: TestClient) -> None:
    response = client.post(
        "/v1/chunk",
        json={
            "text": "Before <|endoftext|> after.",
            "chunker": {"engine": "langchain", "strategy": "token"},
        },
    )

    assert response.status_code == 200
    assert response.json()["chunks"][0]["text"] == "Before <|endoftext|> after."


@pytest.mark.parametrize("engine", ["llamaindex", "chonkie"])
def test_semantic_strategies_use_server_embeddings(
    client: TestClient, embedder: FakeEmbedder, engine: str
) -> None:
    chunker: dict[str, Any] = {"engine": engine, "strategy": "semantic"}
    if engine == "llamaindex":
        chunker |= {"buffer_size": 0, "breakpoint_percentile_threshold": 50}

    response = client.post("/v1/chunk", json={"text": TWO_TOPICS, "chunker": chunker})

    assert response.status_code == 200, response.text
    assert embedder.calls > 0
    chunks = response.json()["chunks"]
    assert chunks
    if engine == "llamaindex":
        assert len(chunks) == 2
        assert "rocket" not in chunks[0]["text"]
        assert "kitten" not in chunks[1]["text"]


def test_embed_adds_vectors(client: TestClient) -> None:
    body = client.post("/v1/chunk", json={"text": TWO_TOPICS, "embed": True}).json()

    assert body["embedding_model"] == "fake-embedding"
    assert all(
        chunk["embedding"] == FakeEmbedder._vector(chunk["text"]) for chunk in body["chunks"]
    )


def test_metadata_is_echoed(client: TestClient) -> None:
    body = client.post("/v1/chunk", json={"text": "hello", "metadata": {"file_id": 7}}).json()

    assert body["metadata"] == {"file_id": 7}


@pytest.mark.parametrize(
    "payload",
    [
        {"engine": "chonkie", "strategy": "semantic"},
        {"engine": "llamaindex", "strategy": "semantic"},
    ],
)
def test_semantic_requires_embeddings(bare_client: TestClient, payload: dict[str, str]) -> None:
    response = bare_client.post("/v1/chunk", json={"text": TWO_TOPICS, "chunker": payload})

    assert response.status_code == 400
    assert "EMBEDDINGS_MODEL" in response.json()["detail"]


def test_embed_requires_embeddings(bare_client: TestClient) -> None:
    assert bare_client.post("/v1/chunk", json={"text": "hi", "embed": True}).status_code == 400


@pytest.mark.parametrize(
    "payload",
    [
        {"text": ""},
        {"text": "hi", "chunker": {"engine": "nope", "strategy": "x"}},
        {"text": "hi", "chunker": {"engine": "chonkie", "strategy": "nope"}},
        {"text": "hi", "chunker": {"engine": "chonkie", "strategy": "token", "chunk_overlap": 600}},
        {"text": "hi", "chunker": {"engine": "langchain", "strategy": "token", "typo": 1}},
        {"text": "hi", "chunker": {"engine": "unstructured", "strategy": "basic", "overlap": 5000}},
        {"text": "hi", "encoding": "unknown"},
        {"text": "hi", "webhook_url": "not-a-url"},
        {"text": "hi", "unexpected": True},
    ],
)
def test_invalid_requests_are_rejected(client: TestClient, payload: dict[str, Any]) -> None:
    assert client.post("/v1/chunk", json=payload).status_code == 422
