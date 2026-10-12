import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.conftest import ClientFactory
from tests.documents import make_docx, make_pdf, make_pptx

PDF = make_pdf(
    [
        "Introduction\nChunker parses documents for retrieval.\nIt runs without a database.",
        "Second Page\nSemantic chunking groups related sentences.\nToken chunking uses counts.",
    ]
)
DOCX = make_docx(
    [
        ("Overview", "Chunker is a stateless API for RAG pipelines."),
        ("Engines", "It wraps LangChain, LlamaIndex, Unstructured and Chonkie."),
    ]
)
PPTX = make_pptx([("Welcome", "First slide body."), ("Agenda", "Second slide body.")])


def _upload(client: TestClient, path: str, filename: str, content: bytes, **fields: Any) -> Any:
    data = {k: json.dumps(v) if isinstance(v, dict) else str(v) for k, v in fields.items()}
    return client.post(path, files={"file": (filename, content)}, data=data)


@pytest.mark.parametrize("engine", ["unstructured", "llamaindex"])
def test_parse_pdf_keeps_page_numbers(client: TestClient, engine: str) -> None:
    response = _upload(client, "/v1/parse", "report.pdf", PDF, parser={"engine": engine})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["parser"] == engine
    assert body["document"]["filename"] == "report.pdf"
    assert {e["page_number"] for e in body["elements"]} == {1, 2}
    assert "Semantic chunking" in body["text"]
    assert all("file_path" not in e["metadata"] for e in body["elements"])


def test_unstructured_parser_returns_typed_elements(client: TestClient) -> None:
    body = _upload(client, "/v1/parse", "notes.docx", DOCX).json()

    assert [(e["type"], e["text"]) for e in body["elements"]][:2] == [
        ("Title", "Overview"),
        ("NarrativeText", "Chunker is a stateless API for RAG pipelines."),
    ]


@pytest.mark.parametrize(
    ("filename", "content"),
    [("notes.docx", DOCX), ("deck.pptx", PPTX), ("notes.md", b"# Title\n\nSome *markdown* text.")],
)
@pytest.mark.parametrize("engine", ["unstructured", "llamaindex"])
def test_parse_office_and_text_formats(
    client: TestClient, engine: str, filename: str, content: bytes
) -> None:
    response = _upload(client, "/v1/parse", filename, content, parser={"engine": engine})

    assert response.status_code == 200, response.text
    assert response.json()["text"].strip()


def test_chunk_file_defaults_to_unstructured(client: TestClient) -> None:
    response = _upload(client, "/v1/chunk/file", "report.pdf", PDF)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["parser"] == "unstructured"
    assert body["chunker"] == {"engine": "unstructured", "strategy": "by_title"}
    # Both short pages fit in one multi-page section.
    assert [chunk["page_numbers"] for chunk in body["chunks"]] == [[1, 2]]


def test_unstructured_chunks_can_stop_at_page_breaks(client: TestClient) -> None:
    chunker = {
        "engine": "unstructured",
        "strategy": "by_title",
        "multipage_sections": False,
        "combine_text_under_n_chars": 0,  # otherwise small sections merge across pages
    }

    body = _upload(client, "/v1/chunk/file", "report.pdf", PDF, chunker=chunker).json()

    assert [chunk["page_numbers"] for chunk in body["chunks"]] == [[1], [2]]


def test_chunk_file_with_mixed_engines_maps_pages(client: TestClient) -> None:
    response = _upload(
        client,
        "/v1/chunk/file",
        "report.pdf",
        PDF,
        parser={"engine": "llamaindex"},
        chunker={"engine": "chonkie", "strategy": "token", "chunk_size": 12},
        embed="true",
        metadata={"source_id": "abc"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["parser"] == "llamaindex"
    assert body["metadata"] == {"source_id": "abc"}
    assert body["embedding_model"] == "fake-embedding"
    pages = [chunk["page_numbers"] for chunk in body["chunks"]]
    assert pages[0] == [1]
    assert pages[-1] == [2]
    assert all(chunk["embedding"] for chunk in body["chunks"])


def test_unsupported_extension_is_415(client: TestClient) -> None:
    response = _upload(client, "/v1/chunk/file", "virus.exe", b"MZ")

    assert response.status_code == 415
    assert ".pdf" in response.json()["detail"]


def test_corrupt_file_is_422(client: TestClient) -> None:
    response = _upload(client, "/v1/parse", "broken.pdf", b"%PDF-1.4 definitely not a pdf")

    assert response.status_code == 422
    assert "broken.pdf" in response.json()["detail"]


def test_upload_size_limit(make_client: ClientFactory) -> None:
    client = make_client(max_upload_mb=1)

    response = _upload(client, "/v1/parse", "big.txt", b"a" * (1024 * 1024 + 1))

    assert response.status_code == 413
    assert "File exceeds" in response.json()["detail"]


def test_oversized_requests_are_rejected_before_reading(make_client: ClientFactory) -> None:
    client = make_client(max_upload_mb=1)

    response = _upload(client, "/v1/parse", "big.txt", b"a" * (3 * 1024 * 1024))

    assert response.status_code == 413
    assert "Request body exceeds" in response.json()["detail"]


@pytest.mark.parametrize(
    "fields",
    [
        {"parser": "{not json"},
        {"parser": {"engine": "mistral"}},
        {"chunker": {"engine": "langchain", "strategy": "token", "chunk_size": 0}},
        {"encoding": "nope"},
    ],
)
def test_invalid_form_fields_are_422(client: TestClient, fields: dict[str, Any]) -> None:
    response = _upload(client, "/v1/chunk/file", "notes.docx", DOCX, **fields)

    assert response.status_code == 422
