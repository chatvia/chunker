from typing import Any, get_args

import pytest
from fastapi.testclient import TestClient

from app.engines import CHUNKERS
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
    {"engine": "langchain", "strategy": "nltk", "chunk_size": 40, "chunk_overlap": 0},
    {"engine": "langchain", "strategy": "js_framework", "chunk_size": 60, "chunk_overlap": 0},
    {"engine": "langchain", "strategy": "markdown_syntax"},
    {"engine": "llamaindex", "strategy": "sentence_window", "window_size": 1},
    {
        "engine": "llamaindex",
        "strategy": "hierarchical",
        "chunk_sizes": [64, 32],
        "chunk_overlap": 0,
    },
    {"engine": "chonkie", "strategy": "fast", "chunk_size": 200},
    {
        "engine": "chonkie",
        "strategy": "recursive",
        "chunk_size": 40,
        "levels": [{"delimiters": ["\n\n"]}, {"delimiters": [". "]}, {"whitespace": True}],
    },
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


def _verbatim(config: dict[str, Any]) -> bool:
    for model in CHUNKERS:
        engine = get_args(model.model_fields["engine"].annotation)
        strategy = get_args(model.model_fields["strategy"].annotation)
        if (engine, strategy) == ((config["engine"],), (config["strategy"],)):
            return model.verbatim
    raise AssertionError(f"unknown strategy {config}")


@pytest.mark.parametrize("chunker", [s for s in STRATEGIES if _verbatim(s)], ids=_id)
def test_verbatim_strategies_report_offsets(client: TestClient, chunker: dict[str, Any]) -> None:
    body = client.post("/v1/chunk", json={"text": ARTICLE, "chunker": chunker}).json()

    starts = [chunk["start_index"] for chunk in body["chunks"]]
    assert None not in starts
    if chunker["strategy"] != "hierarchical":  # emits every level, largest first
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
        # Only the topic shift is above the 90th percentile of sentence distances.
        chunker |= {"buffer_size": 0, "breakpoint_percentile_threshold": 90}

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
        {"text": "hi", "chunker": {"engine": "unstructured", "strategy": "basic"}},
        {"text": "hi", "encoding": "unknown"},
        {"text": "hi", "webhook_url": "not-a-url"},
        {"text": "hi", "unexpected": True},
    ],
)
def test_invalid_requests_are_rejected(client: TestClient, payload: dict[str, Any]) -> None:
    assert client.post("/v1/chunk", json=payload).status_code == 422


PAGES = [
    {"text": "Page one introduces the topic. It has two short sentences."},
    {"text": "   "},
    {"text": "Page three continues the topic. It also has two sentences."},
]
# Small enough that no chunk crosses a page break.
PAGE_CHUNKER = {
    "engine": "langchain",
    "strategy": "recursive",
    "length_unit": "characters",
    "chunk_size": 20,
    "chunk_overlap": 0,
}


def test_pages_map_chunks_to_page_numbers(client: TestClient) -> None:
    body = client.post("/v1/chunk", json={"pages": PAGES, "chunker": PAGE_CHUNKER}).json()

    assert body["document"]["pages"] == 3
    pages = [chunk["page_numbers"] for chunk in body["chunks"]]
    assert pages[0] == [1]
    assert pages[-1] == [3]  # the blank page keeps its position
    joined = "\n\n".join(page["text"] for page in PAGES if page["text"])
    for chunk in body["chunks"]:
        assert joined[chunk["start_index"] : chunk["end_index"]] == chunk["text"]


def test_chunks_spanning_pages_list_every_page(client: TestClient) -> None:
    chunker = {"engine": "langchain", "strategy": "token", "chunk_size": 500, "chunk_overlap": 0}

    body = client.post("/v1/chunk", json={"pages": PAGES, "chunker": chunker}).json()

    assert [chunk["page_numbers"] for chunk in body["chunks"]] == [[1, 2, 3]]


def test_explicit_page_numbers_are_kept(client: TestClient) -> None:
    pages = [{"text": "Fifth page text.", "page_number": 5}, {"text": "Sixth page text."}]

    body = client.post("/v1/chunk", json={"pages": pages, "chunker": PAGE_CHUNKER}).json()

    # Pages without a number fall back to their position.
    assert [chunk["page_numbers"] for chunk in body["chunks"]] == [[5], [2]]


def test_text_has_no_page_numbers(client: TestClient) -> None:
    body = client.post("/v1/chunk", json={"text": ARTICLE}).json()

    assert body["document"]["pages"] is None
    assert all(chunk["page_numbers"] == [] for chunk in body["chunks"])


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"text": "hi", "pages": [{"text": "hi"}]},
        {"pages": []},
        {"pages": [{"text": " "}, {"text": ""}]},
        {"pages": [{"text": "hi", "page_number": 0}]},
        {"pages": [{"text": "hi", "image": "x"}]},
    ],
)
def test_invalid_page_requests_are_rejected(client: TestClient, payload: dict[str, Any]) -> None:
    assert client.post("/v1/chunk", json=payload).status_code == 422
