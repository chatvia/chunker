import uuid
from typing import Any

from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import JSONResponse

from app import pipeline
from app.dependencies import EmbedderDep, SettingsDep, WebhookSenderDep
from app.embeddings import check_request_embeddings, resolve_embedder
from app.errors import NotConfiguredError
from app.schemas import ChunkRequest, ChunkResponse, JobAccepted

router = APIRouter(prefix="/v1", tags=["chunking"])

_RESPONSES: dict[int | str, dict[str, Any]] = {
    202: {"model": JobAccepted, "description": "Accepted; the result is POSTed to `webhook_url`."}
}


@router.post("/chunk", response_model=ChunkResponse, responses=_RESPONSES, summary="Chunk text")
async def chunk(
    body: ChunkRequest,
    background: BackgroundTasks,
    settings: SettingsDep,
    embedder: EmbedderDep,
    sender: WebhookSenderDep,
) -> ChunkResponse | JSONResponse:
    check_request_embeddings(body.embeddings, embedder, settings)
    pipeline.preflight(
        body.chunker,
        embed=body.embed,
        embeddings=embedder is not None or body.embeddings is not None,
    )
    document = body.document()

    async def job() -> ChunkResponse:
        # Created here so a per-request client lives as long as the (possibly async) job.
        async with resolve_embedder(body.embeddings, embedder, settings) as job_embedder:
            return await pipeline.chunk(
                document,
                body.chunker,
                encoding=body.encoding,
                embed=body.embed,
                embedder=job_embedder,
                metadata=body.metadata,
                page_count=len(body.pages) if body.pages is not None else None,
            )

    if body.webhook_url is None:
        return await job()
    if sender is None:
        raise NotConfiguredError("Set WEBHOOK_SECRET on the server to use webhook_url.")
    job_id = str(uuid.uuid4())
    background.add_task(pipeline.run_job, sender, str(body.webhook_url), job_id, job, body.metadata)
    return JSONResponse(status_code=202, content=JobAccepted(job_id=job_id).model_dump())
