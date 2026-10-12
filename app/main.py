import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI

from app import __version__
from app.config import Settings, get_settings
from app.dependencies import require_api_key
from app.embeddings import Embedder, OpenAICompatibleEmbedder
from app.errors import register_exception_handlers
from app.middleware import BodySizeLimitMiddleware
from app.routes import chunk, embed, engines, health, parse
from app.webhooks import WebhookSender, decode_secret

DESCRIPTION = """
Stateless HTTP API for RAG pipelines: parse documents, chunk text and compute embeddings
with **LangChain**, **LlamaIndex**, **Unstructured** and **Chonkie** from any language.

Nothing is stored. Pass `webhook_url` to process asynchronously; results are POSTed back,
signed per the [Standard Webhooks](https://standardwebhooks.com) spec.
"""


def create_app(
    settings: Settings | None = None,
    *,
    embedder: Embedder | None = None,
    webhook_transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    """Build the application. `embedder` and `webhook_transport` are injectable for tests."""
    settings = settings or get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owned_embedder = None if embedder else OpenAICompatibleEmbedder.from_settings(settings)
        app.state.embedder = embedder or owned_embedder
        async with httpx.AsyncClient(
            transport=webhook_transport,
            timeout=settings.webhook_timeout,
            verify=settings.webhook_verify_tls,
        ) as http_client:
            secret = settings.webhook_secret
            app.state.webhook_sender = (
                WebhookSender(
                    http_client,
                    decode_secret(secret.get_secret_value()),
                    max_attempts=settings.webhook_max_attempts,
                    user_agent=f"chunker/{__version__}",
                )
                if secret
                else None
            )
            yield
        if owned_embedder is not None:
            await owned_embedder.aclose()

    app = FastAPI(title="Chunker", version=__version__, description=DESCRIPTION, lifespan=lifespan)
    app.state.settings = settings
    register_exception_handlers(app)

    app.include_router(health.router)
    protected = [Depends(require_api_key)]
    for module in (engines, parse, chunk, embed):
        app.include_router(module.router, dependencies=protected)

    # Allow some headroom over the file limit for multipart framing and form fields.
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.max_upload_bytes + 1024 * 1024)
    return app


app = create_app()
