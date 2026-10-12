import uuid
from collections.abc import Awaitable, Callable
from typing import Annotated, Any

from fastapi import APIRouter, BackgroundTasks, File, Form, UploadFile
from fastapi.responses import JSONResponse
from pydantic import HttpUrl, Json

from app import pipeline
from app.dependencies import EmbedderDep, SettingsDep, WebhookSenderDep
from app.engines import (
    DEFAULT_FILE_CHUNKER,
    DEFAULT_PARSER,
    ChunkerConfig,
    ParsedDocument,
    ParserConfig,
)
from app.errors import NotConfiguredError
from app.schemas import (
    CHUNKER_FORM_HELP,
    PARSER_FORM_HELP,
    ChunkResponse,
    ChunkTextRequest,
    JobAccepted,
)
from app.tokens import DEFAULT_ENCODING, Encoding
from app.webhooks import WebhookSender

router = APIRouter(prefix="/v1", tags=["chunking"])

_RESPONSES: dict[int | str, dict[str, Any]] = {
    202: {"model": JobAccepted, "description": "Accepted; the result is POSTed to `webhook_url`."}
}


async def _run_or_schedule(
    job: Callable[[], Awaitable[ChunkResponse]],
    *,
    webhook_url: HttpUrl | None,
    metadata: dict[str, Any],
    background: BackgroundTasks,
    sender: WebhookSender | None,
) -> ChunkResponse | JSONResponse:
    if webhook_url is None:
        return await job()
    if sender is None:
        raise NotConfiguredError("Set WEBHOOK_SECRET on the server to use webhook_url.")
    job_id = str(uuid.uuid4())
    background.add_task(pipeline.run_job, sender, str(webhook_url), job_id, job, metadata)
    return JSONResponse(status_code=202, content=JobAccepted(job_id=job_id).model_dump())


@router.post(
    "/chunk",
    response_model=ChunkResponse,
    responses=_RESPONSES,
    summary="Chunk text",
)
async def chunk_text(
    body: ChunkTextRequest,
    background: BackgroundTasks,
    embedder: EmbedderDep,
    sender: WebhookSenderDep,
) -> ChunkResponse | JSONResponse:
    pipeline.check_embeddings(body.chunker, body.embed, embedder)
    document = ParsedDocument.from_text(body.text)

    async def job() -> ChunkResponse:
        return await pipeline.chunk(
            document,
            body.chunker,
            encoding=body.encoding,
            embed=body.embed,
            embedder=embedder,
            metadata=body.metadata,
        )

    return await _run_or_schedule(
        job,
        webhook_url=body.webhook_url,
        metadata=body.metadata,
        background=background,
        sender=sender,
    )


@router.post(
    "/chunk/file",
    response_model=ChunkResponse,
    responses=_RESPONSES,
    summary="Parse a file and chunk it",
)
async def chunk_file(
    file: Annotated[UploadFile, File(description="Document to parse and chunk.")],
    background: BackgroundTasks,
    settings: SettingsDep,
    embedder: EmbedderDep,
    sender: WebhookSenderDep,
    parser: Annotated[
        Json[ParserConfig] | None,
        Form(description=PARSER_FORM_HELP),
    ] = None,
    chunker: Annotated[
        Json[ChunkerConfig] | None,
        Form(description=CHUNKER_FORM_HELP),
    ] = None,
    encoding: Annotated[Encoding, Form()] = DEFAULT_ENCODING,
    embed: Annotated[bool, Form(description="Embed each chunk with the server's model.")] = False,
    webhook_url: Annotated[
        HttpUrl | None, Form(description="Process asynchronously and POST the result here.")
    ] = None,
    metadata: Annotated[
        Json[dict[str, Any]] | None, Form(description="JSON object echoed back in the result.")
    ] = None,
) -> ChunkResponse | JSONResponse:
    parser_config = parser or DEFAULT_PARSER
    chunker_config = chunker or DEFAULT_FILE_CHUNKER
    filename = file.filename or "upload"
    metadata = metadata or {}

    # Validate everything we can before reading the body or answering 202.
    pipeline.check_supported(parser_config, filename)
    pipeline.check_embeddings(chunker_config, embed, embedder)
    data = await pipeline.read_upload(file, settings.max_upload_bytes)
    content_type = file.content_type

    async def job() -> ChunkResponse:
        document = await pipeline.parse(parser_config, data, filename, content_type)
        return await pipeline.chunk(
            document,
            chunker_config,
            encoding=encoding,
            embed=embed,
            embedder=embedder,
            metadata=metadata,
            parser=parser_config,
        )

    return await _run_or_schedule(
        job, webhook_url=webhook_url, metadata=metadata, background=background, sender=sender
    )
