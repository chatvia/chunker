import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from fastapi import UploadFile
from starlette.concurrency import run_in_threadpool

from app.embeddings import Embedder
from app.engines import ChunkContext, ChunkerBase, ParsedDocument, ParserBase, RawChunk
from app.errors import (
    ChunkerError,
    FileTooLargeError,
    NotConfiguredError,
    ParseError,
    UnsupportedFileTypeError,
)
from app.schemas import ChunkOut, ChunkResponse, DocumentInfo, StrategyInfo
from app.tokens import count_tokens
from app.webhooks import WebhookSender

logger = logging.getLogger(__name__)

_READ_CHUNK_BYTES = 1024 * 1024


async def read_upload(file: UploadFile, max_bytes: int) -> bytes:
    buffer = bytearray()
    while chunk := await file.read(_READ_CHUNK_BYTES):
        buffer.extend(chunk)
        if len(buffer) > max_bytes:
            raise FileTooLargeError(f"File exceeds the {max_bytes // (1024 * 1024)} MB limit")
    return bytes(buffer)


def check_supported(parser: ParserBase, filename: str) -> None:
    extension = Path(filename).suffix.lower()
    if extension not in parser.extensions:
        supported = ", ".join(sorted(parser.extensions))
        raise UnsupportedFileTypeError(
            f"The {parser.engine!r} parser does not support {extension or 'extensionless'!r} "
            f"files. Supported: {supported}"
        )


def check_embeddings(chunker: ChunkerBase, embed: bool, embedder: Embedder | None) -> None:
    """Fail fast, before any work (or a 202) happens, when embeddings are needed but missing."""
    if embedder is None and (embed or chunker.requires_embeddings):
        raise NotConfiguredError(
            "Embeddings are not configured on this server; set EMBEDDINGS_MODEL to use "
            "`embed` or semantic strategies."
        )


async def parse(
    parser: ParserBase, data: bytes, filename: str, content_type: str | None
) -> ParsedDocument:
    check_supported(parser, filename)
    try:
        document = await run_in_threadpool(parser.parse, data, filename)
    except ChunkerError:
        raise
    except Exception as exc:
        logger.exception("Failed to parse %r with %s", filename, parser.engine)
        raise ParseError(f"Could not parse {filename!r}: {exc}") from exc
    if not document.text.strip():
        raise ParseError(
            f"No text could be extracted from {filename!r}. The file may be corrupt, or it "
            "may be scanned: try the unstructured parser with strategy 'hi_res' or 'ocr_only'."
        )
    document.content_type = content_type
    return document


def document_info(document: ParsedDocument, encoding: str) -> DocumentInfo:
    return DocumentInfo(
        filename=document.filename,
        content_type=document.content_type,
        characters=len(document.text),
        token_count=count_tokens(document.text, encoding),
    )


def _finalize(
    raw_chunks: list[RawChunk], document: ParsedDocument, chunker: ChunkerBase, encoding: str
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

        if start is not None and end is not None:
            cursor = start + 1
        if raw.page_numbers is not None:
            page_numbers = raw.page_numbers
        elif start is not None and end is not None:
            page_numbers = document.page_numbers(start, end)
        else:
            page_numbers = []

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
    document: ParsedDocument,
    chunker: ChunkerBase,
    *,
    encoding: str,
    embed: bool,
    embedder: Embedder | None,
    metadata: dict[str, Any],
    parser: ParserBase | None = None,
) -> ChunkResponse:
    context = ChunkContext(encoding=encoding, embedder=embedder)
    raw_chunks = await run_in_threadpool(chunker.chunk, document, context)
    chunks = _finalize(raw_chunks, document, chunker, encoding)

    embedding_model = None
    if embed and chunks:
        embedder = context.require_embedder()
        vectors = await embedder.aembed([c.text for c in chunks])
        for item, vector in zip(chunks, vectors, strict=True):
            item.embedding = vector
        embedding_model = embedder.model

    return ChunkResponse(
        parser=parser.engine if parser else None,
        chunker=StrategyInfo(engine=chunker.engine, strategy=chunker.strategy),
        document=document_info(document, encoding),
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
