import logging
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from app.embeddings import Embedder
from app.engines import ChunkContext, ChunkerBase, Document, RawChunk
from app.errors import ChunkerError, InvalidInputError, NotConfiguredError
from app.schemas import ChunkOut, ChunkResponse, DocumentInfo, StrategyInfo
from app.tokens import count_tokens
from app.webhooks import WebhookSender

logger = logging.getLogger(__name__)


def preflight(chunker: ChunkerBase, *, embed: bool, embeddings: bool) -> None:
    """Reject a request that cannot run, before any work (or a 202) happens."""
    if embed and not embeddings:
        raise NotConfiguredError(
            "`embed` needs embeddings: set EMBEDDINGS_MODEL on the server or send `embeddings` "
            "with the request."
        )
    chunker.preflight(embeddings=embeddings)


def _finalize(
    raw_chunks: list[RawChunk], document: Document, chunker: ChunkerBase, encoding: str
) -> list[ChunkOut]:
    text = document.text
    chunks: list[ChunkOut] = []
    cursor = 0
    for raw in raw_chunks:
        if not raw.text.strip():
            continue
        start: int | None = None
        end: int | None = None
        if (
            raw.start_index is not None
            and raw.end_index is not None
            and raw.start_index >= 0
            and text[raw.start_index : raw.end_index] == raw.text
        ):
            start, end = raw.start_index, raw.end_index
        elif chunker.verbatim:
            # Chunks come in document order and may overlap their predecessor, so search
            # forward from just after the previous chunk's start.
            found = text.find(raw.text, cursor)
            if found >= 0:
                start, end = found, found + len(raw.text)

        page_numbers: list[int] = []
        if start is not None and end is not None:
            cursor = start + 1
            page_numbers = document.page_numbers(start, end)

        chunks.append(
            ChunkOut(
                index=len(chunks),
                text=raw.text,
                token_count=count_tokens(raw.text, encoding),
                start_index=start,
                end_index=end,
                page_numbers=page_numbers,
                metadata=raw.metadata,
            )
        )
    return chunks


async def chunk(
    document: Document,
    chunker: ChunkerBase,
    *,
    encoding: str,
    embed: bool,
    embedder: Embedder | None,
    metadata: dict[str, Any],
    page_count: int | None = None,
) -> ChunkResponse:
    context = ChunkContext(encoding=encoding, embedder=embedder)
    try:
        raw_chunks = await run_in_threadpool(chunker.chunk, document, context)
    except ChunkerError:
        raise
    except (ValueError, ValidationError) as exc:
        # Library-level validation of options or input that our models do not mirror.
        raise InvalidInputError(
            f"The {chunker.engine} {chunker.strategy} chunker rejected the request: {exc}"
        ) from exc
    chunks = _finalize(raw_chunks, document, chunker, encoding)

    embedding_model = None
    if embed and chunks:
        embedder = context.require_embedder()
        vectors = await embedder.aembed([c.text for c in chunks])
        for item, vector in zip(chunks, vectors, strict=True):
            item.embedding = vector
        embedding_model = embedder.model

    return ChunkResponse(
        chunker=StrategyInfo(engine=chunker.engine, strategy=chunker.strategy),
        document=DocumentInfo(
            characters=len(document.text),
            token_count=count_tokens(document.text, encoding),
            pages=page_count,
        ),
        total_chunks=len(chunks),
        chunks=chunks,
        embedding_model=embedding_model,
        metadata=metadata,
    )


async def run_job(
    sender: WebhookSender,
    url: str,
    job_id: str,
    job: Callable[[], Awaitable[ChunkResponse]],
    metadata: dict[str, Any],
) -> None:
    """Run a chunking job in the background and report its outcome to the webhook."""
    try:
        result = await job()
    except Exception as exc:
        if isinstance(exc, ChunkerError):
            error = {"status_code": exc.status_code, "message": exc.detail}
        else:
            logger.exception("Job %s failed", job_id)
            error = {"status_code": 500, "message": "Internal error while processing the job"}
        data = {"job_id": job_id, "metadata": metadata, "error": error}
        await sender.send(url, "chunking.failed", data)
        return
    await sender.send(
        url, "chunking.completed", {"job_id": job_id, **result.model_dump(mode="json")}
    )
