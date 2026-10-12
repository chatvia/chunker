"""Strategies that need structured input (HTML, JSON, tables, code) and their options."""

import json
import os
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.engines import grammars

HTML = (
    "<html><body><h1>Intro</h1><p>Chunker splits text.</p>"
    "<h2>Details</h2><p>It is stateless. It has engines.</p>"
    "<table><tr><td>a</td><td>b</td></tr></table></body></html>"
)
JSON_DOC = json.dumps(
    {"user": {"name": "Ann", "roles": ["admin", "dev"]}, "items": [{"id": i} for i in range(30)]}
)
TABLE = "| name | qty |\n| --- | --- |\n" + "".join(f"| item{i} | {i} |\n" for i in range(7))
CODE = (
    "def add(a, b):\n    return a + b\n\n\n"
    "class Calc:\n    def mul(self, a, b):\n        return a * b\n\n"
    "    def div(self, a, b):\n        return a / b\n"
)


def _chunk(client: TestClient, text: str, chunker: dict[str, Any]) -> Any:
    response = client.post("/v1/chunk", json={"text": text, "chunker": chunker})
    assert response.status_code == 200, response.text
    return response.json()


@pytest.mark.parametrize("strategy", ["html_header", "html_section", "html_semantic"])
def test_langchain_html_strategies(client: TestClient, strategy: str) -> None:
    body = _chunk(client, HTML, {"engine": "langchain", "strategy": strategy})

    texts = " ".join(chunk["text"] for chunk in body["chunks"])
    assert "Chunker splits text." in texts
    assert "<p>" not in texts
    assert any(chunk["metadata"]["headers"] for chunk in body["chunks"])


def test_html_semantic_preserves_elements(client: TestClient) -> None:
    chunker = {
        "engine": "langchain",
        "strategy": "html_semantic",
        "elements_to_preserve": ["table"],
        "max_chunk_size": 200,
    }

    body = _chunk(client, HTML, chunker)

    assert any("a" in chunk["text"] and "b" in chunk["text"] for chunk in body["chunks"])


def test_llamaindex_html_tags(client: TestClient) -> None:
    body = _chunk(client, HTML, {"engine": "llamaindex", "strategy": "html", "tags": ["p"]})

    # Consecutive elements with the same tag are merged into one chunk.
    [chunk] = body["chunks"]
    assert chunk["text"] == "Chunker splits text.\nIt is stateless. It has engines."
    assert chunk["metadata"]["tag"] == "p"


def test_langchain_json_keeps_chunks_valid(client: TestClient) -> None:
    chunker = {"engine": "langchain", "strategy": "json", "max_chunk_size": 200}

    body = _chunk(client, JSON_DOC, chunker)

    assert body["total_chunks"] > 1
    for chunk in body["chunks"]:
        assert isinstance(json.loads(chunk["text"]), dict)


def test_llamaindex_json_flattens_paths(client: TestClient) -> None:
    body = _chunk(client, JSON_DOC, {"engine": "llamaindex", "strategy": "json"})

    assert "user name Ann" in body["chunks"][0]["text"]


@pytest.mark.parametrize("engine", ["langchain", "llamaindex"])
def test_json_strategies_reject_invalid_json(client: TestClient, engine: str) -> None:
    response = client.post(
        "/v1/chunk", json={"text": "{not json", "chunker": {"engine": engine, "strategy": "json"}}
    )

    assert response.status_code == 422
    assert "not valid JSON" in response.json()["detail"]


def test_markdown_syntax_tags_code_blocks(client: TestClient) -> None:
    text = "# Title\n\nSome text.\n\n```python\nprint('x')\n```\n"

    body = _chunk(client, text, {"engine": "langchain", "strategy": "markdown_syntax"})

    code = [chunk for chunk in body["chunks"] if chunk["metadata"].get("code") == "python"]
    assert code
    assert "print('x')" in code[0]["text"]


def test_sentence_window_metadata(client: TestClient) -> None:
    text = "Cats purr. Dogs bark. Birds sing. Fish swim."

    body = _chunk(
        client, text, {"engine": "llamaindex", "strategy": "sentence_window", "window_size": 1}
    )

    assert [chunk["text"].strip() for chunk in body["chunks"]][:2] == ["Cats purr.", "Dogs bark."]
    assert "Cats purr." in body["chunks"][1]["metadata"]["window"]
    assert "Birds sing." in body["chunks"][1]["metadata"]["window"]


def test_hierarchical_links_children_to_parents(client: TestClient) -> None:
    text = "Cats purr. Dogs bark. Birds sing in the morning. Fish swim quietly. " * 6
    chunker = {
        "engine": "llamaindex",
        "strategy": "hierarchical",
        "chunk_sizes": [64, 16],
        "chunk_overlap": 0,
    }

    chunks = _chunk(client, text, chunker)["chunks"]

    by_id = {chunk["metadata"]["node_id"]: chunk for chunk in chunks}
    roots = [chunk for chunk in chunks if chunk["metadata"]["parent_id"] is None]
    leaves = [chunk for chunk in chunks if chunk["metadata"]["level"] == 1]
    assert roots
    assert leaves
    assert all(chunk["metadata"]["level"] == 0 for chunk in roots)
    for leaf in leaves:
        parent = by_id[leaf["metadata"]["parent_id"]]
        assert leaf["text"].strip() in parent["text"]


@pytest.mark.parametrize(
    "chunker",
    [
        {"engine": "llamaindex", "strategy": "hierarchical", "chunk_sizes": [16, 64]},
        {"engine": "llamaindex", "strategy": "hierarchical", "chunk_sizes": [64, 16]},
        {"engine": "llamaindex", "strategy": "code", "language": "python", "chunk_lines": 2},
        {
            "engine": "chonkie",
            "strategy": "recursive",
            "levels": [{"delimiters": ["."], "whitespace": True}],
        },
        {"engine": "chonkie", "strategy": "semantic", "filter_window": 3, "filter_polyorder": 3},
        {"engine": "chonkie", "strategy": "token", "chunk_overlap": 1.5},
        {
            "engine": "langchain",
            "strategy": "html_semantic",
            "max_chunk_size": 10,
            "chunk_overlap": 10,
        },
    ],
)
def test_invalid_strategy_options_are_rejected(client: TestClient, chunker: dict[str, Any]) -> None:
    # The second case fails because the default overlap (20) is not smaller than 16.
    response = client.post("/v1/chunk", json={"text": "hello", "chunker": chunker})

    assert response.status_code == 422


def test_chonkie_token_accepts_fractional_overlap(client: TestClient) -> None:
    chunker = {"engine": "chonkie", "strategy": "token", "chunk_size": 10, "chunk_overlap": 0.5}

    body = _chunk(client, "word " * 50, chunker)

    assert body["total_chunks"] > 5


def test_chonkie_fast_chunker_options(client: TestClient) -> None:
    chunker = {"engine": "chonkie", "strategy": "fast", "chunk_size": 30, "delimiters": "."}

    body = _chunk(client, "One. Two. Three. Four. Five. Six. Seven. Eight.", chunker)

    assert body["total_chunks"] > 1
    assert all(len(chunk["text"].encode()) <= 30 for chunk in body["chunks"])


@pytest.mark.parametrize(("unit", "expected"), [("rows", 3), ("tokens", None)])
def test_chonkie_table_repeats_header(client: TestClient, unit: str, expected: int | None) -> None:
    chunker = {"engine": "chonkie", "strategy": "table", "unit": unit, "chunk_size": 3}
    if unit == "tokens":
        chunker["chunk_size"] = 20

    chunks = _chunk(client, TABLE, chunker)["chunks"]

    assert len(chunks) > 1
    assert all(chunk["text"].startswith("| name | qty |") for chunk in chunks)
    if expected is not None:
        assert len(chunks) == expected


def _require_grammars() -> None:
    if len(grammars.installed()) < grammars.CHONKIE_MIN_GRAMMARS:
        if os.environ.get("CI"):
            pytest.fail("CI must run `chunker warmup` before the tests")
        pytest.skip("tree-sitter grammars are not installed; run `uv run chunker warmup`")


@pytest.mark.parametrize("engine", ["llamaindex", "chonkie"])
def test_code_strategies_split_on_syntax(client: TestClient, engine: str) -> None:
    _require_grammars()
    chunker: dict[str, Any] = {"engine": engine, "strategy": "code", "language": "python"}
    chunker |= (
        {"chunk_lines": 4, "chunk_lines_overlap": 1, "max_chars": 60}
        if engine == "llamaindex"
        else {"chunk_size": 20}
    )

    chunks = _chunk(client, CODE, chunker)["chunks"]

    assert len(chunks) > 1
    assert chunks[0]["text"].startswith("def add")
    # Syntax-aware: no function is split across chunks.
    for function in (
        "def add(a, b):\n    return a + b",
        "def mul(self, a, b):",
        "def div(self, a, b):",
    ):
        assert sum(function in chunk["text"] for chunk in chunks) == 1


@pytest.mark.parametrize("engine", ["llamaindex", "chonkie"])
def test_code_strategies_reject_uninstalled_languages(client: TestClient, engine: str) -> None:
    chunker = {"engine": engine, "strategy": "code", "language": "brainfuck-not-installed"}

    response = client.post("/v1/chunk", json={"text": "x", "chunker": chunker})

    assert response.status_code == 400
    assert "CODE_LANGUAGES" in response.json()["detail"]
