import asyncio
import base64
import json
from collections.abc import Sequence

import httpx
import pytest

from app.errors import UpstreamError
from app.webhooks import WebhookSender, decode_secret, sign
from tests.conftest import ClientFactory, FakeEmbedder

SECRET = "whsec_" + base64.b64encode(b"0123456789abcdef0123456789abcdef").decode()


def test_signature_matches_standard_webhooks_reference() -> None:
    # Test vector from the Standard Webhooks specification.
    key = decode_secret("whsec_MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw")
    body = b'{"test": 2432232314}'

    signature = sign(key, "msg_p5jXN8AQM9LWM0D4loKWxJek", 1614265330, body)

    assert signature == "v1,g0hM9SsE+OTPJTGt/tmIKtSyZlE3uFJELVlNIOLJ1OE="


@pytest.mark.parametrize("secret", ["whsec_***", "", "whsec_"])
def test_decode_secret_rejects_invalid_values(secret: str) -> None:
    with pytest.raises(ValueError, match="webhook secret"):
        decode_secret(secret)


class Recorder:
    def __init__(self, *statuses: int) -> None:
        self.statuses = list(statuses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(self.statuses.pop(0) if self.statuses else 200)

    def events(self) -> list[dict[str, object]]:
        return [json.loads(request.content) for request in self.requests]


def _assert_signed(request: httpx.Request) -> None:
    expected = sign(
        decode_secret(SECRET),
        request.headers["webhook-id"],
        int(request.headers["webhook-timestamp"]),
        request.content,
    )
    assert request.headers["webhook-signature"] == expected


def _send(recorder: Recorder, max_attempts: int = 3) -> bool:
    async def run() -> bool:
        async with httpx.AsyncClient(transport=httpx.MockTransport(recorder)) as client:
            sender = WebhookSender(
                client, decode_secret(SECRET), max_attempts=max_attempts, backoff_seconds=0
            )
            return await sender.send("https://example.test/hook", "test.event", {"ok": True})

    return asyncio.run(run())


def test_sender_retries_transient_failures_with_the_same_message_id() -> None:
    recorder = Recorder(503, 429, 200)

    assert _send(recorder) is True

    assert len(recorder.requests) == 3
    assert len({r.headers["webhook-id"] for r in recorder.requests}) == 1
    for request in recorder.requests:
        _assert_signed(request)


def test_sender_does_not_retry_client_errors() -> None:
    recorder = Recorder(400)

    assert _send(recorder) is False
    assert len(recorder.requests) == 1


def test_sender_gives_up_after_max_attempts() -> None:
    recorder = Recorder(500, 500, 500, 500)

    assert _send(recorder, max_attempts=2) is False
    assert len(recorder.requests) == 2


def test_async_text_job_posts_signed_result(
    make_client: ClientFactory, embedder: FakeEmbedder
) -> None:
    recorder = Recorder()
    client = make_client(
        embedder=embedder, webhook_secret=SECRET, webhook_transport=httpx.MockTransport(recorder)
    )

    response = client.post(
        "/v1/chunk",
        json={
            "text": "Hello world. " * 50,
            "webhook_url": "https://example.test/hook",
            "embed": True,
            "metadata": {"document_id": 42},
        },
    )

    assert response.status_code == 202
    job_id = response.json()["job_id"]
    [request] = recorder.requests
    _assert_signed(request)
    assert str(request.url) == "https://example.test/hook"
    [event] = recorder.events()
    assert event["type"] == "chunking.completed"
    data = event["data"]
    assert isinstance(data, dict)
    assert data["job_id"] == job_id
    assert data["metadata"] == {"document_id": 42}
    assert data["total_chunks"] == len(data["chunks"]) > 0
    assert data["chunks"][0]["embedding"]


class FailingEmbedder(FakeEmbedder):
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        raise UpstreamError("Embeddings provider returned HTTP 503: unavailable")


def test_async_job_reports_failures(make_client: ClientFactory) -> None:
    recorder = Recorder()
    client = make_client(
        embedder=FailingEmbedder(),
        webhook_secret=SECRET,
        webhook_transport=httpx.MockTransport(recorder),
    )

    response = client.post(
        "/v1/chunk",
        json={
            "text": "Hello world.",
            "embed": True,
            "webhook_url": "https://example.test/hook",
            "metadata": {"document_id": 7},
        },
    )

    assert response.status_code == 202
    [event] = recorder.events()
    assert event["type"] == "chunking.failed"
    assert event["data"] == {
        "job_id": response.json()["job_id"],
        "metadata": {"document_id": 7},
        "error": {
            "status_code": 502,
            "message": "Embeddings provider returned HTTP 503: unavailable",
        },
    }


def test_async_pages_job_keeps_page_numbers(make_client: ClientFactory) -> None:
    recorder = Recorder()
    client = make_client(webhook_secret=SECRET, webhook_transport=httpx.MockTransport(recorder))

    response = client.post(
        "/v1/chunk",
        json={
            "pages": [{"text": "Page one text."}, {"text": "Page two text."}],
            "chunker": {
                "engine": "langchain",
                "strategy": "recursive",
                "length_unit": "characters",
                "chunk_size": 20,
                "chunk_overlap": 0,
            },
            "webhook_url": "https://example.test/hook",
        },
    )

    assert response.status_code == 202
    [event] = recorder.events()
    assert event["type"] == "chunking.completed"
    data = event["data"]
    assert isinstance(data, dict)
    assert [chunk["page_numbers"] for chunk in data["chunks"]] == [[1], [2]]


def test_webhook_url_requires_secret(make_client: ClientFactory) -> None:
    client = make_client()

    response = client.post(
        "/v1/chunk", json={"text": "hello", "webhook_url": "https://example.test/hook"}
    )

    assert response.status_code == 400
    assert "WEBHOOK_SECRET" in response.json()["detail"]


def test_validation_happens_before_accepting_a_job(make_client: ClientFactory) -> None:
    recorder = Recorder()
    client = make_client(webhook_secret=SECRET, webhook_transport=httpx.MockTransport(recorder))

    response = client.post(
        "/v1/chunk",
        json={
            "text": "hello",
            "chunker": {"engine": "chonkie", "strategy": "semantic"},
            "webhook_url": "https://example.test/hook",
        },
    )

    assert response.status_code == 400  # no embeddings model configured
    assert recorder.requests == []
