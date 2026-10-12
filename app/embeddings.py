import asyncio
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from itertools import batched
from typing import Protocol, Self

import openai
from openai.types import CreateEmbeddingResponse
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, PositiveInt, SecretStr

from app.config import Settings
from app.errors import ChunkerError, NotConfiguredError, UpstreamError

_MAX_CONCURRENT_BATCHES = 4


class Embedder(Protocol):
    """Embeds texts with the configured model. Shared by every engine adapter."""

    @property
    def model(self) -> str: ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def aembed(self, texts: Sequence[str]) -> list[list[float]]: ...


class OpenAICompatibleEmbedder:
    """Embeddings from any server implementing the OpenAI `/embeddings` API."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str | None = None,
        dimensions: int | None = None,
        batch_size: int = 64,
        timeout: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        # Local servers (Ollama, vLLM, TEI, ...) often need no key, but the SDK requires one.
        api_key = api_key or "not-needed"
        self._client = openai.OpenAI(
            base_url=base_url, api_key=api_key, timeout=timeout, max_retries=max_retries
        )
        self._async_client = openai.AsyncOpenAI(
            base_url=base_url, api_key=api_key, timeout=timeout, max_retries=max_retries
        )
        self._model = model
        self._dimensions = dimensions
        self._batch_size = batch_size

    @classmethod
    def from_settings(cls, settings: Settings) -> Self | None:
        if not settings.embeddings_model:
            return None
        api_key = settings.embeddings_api_key
        return cls(
            model=settings.embeddings_model,
            base_url=settings.embeddings_base_url,
            api_key=api_key.get_secret_value() if api_key else None,
            dimensions=settings.embeddings_dimensions,
            batch_size=settings.embeddings_batch_size,
            timeout=settings.embeddings_timeout,
            max_retries=settings.embeddings_max_retries,
        )

    @property
    def model(self) -> str:
        return self._model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for batch in batched(texts, self._batch_size, strict=False):
            try:
                response = self._client.embeddings.create(
                    model=self._model,
                    input=list(batch),
                    dimensions=self._dimensions or openai.omit,
                    encoding_format="float",
                )
            except openai.APIError as exc:
                raise _upstream_error(exc) from exc
            vectors.extend(_vectors(response))
        return vectors

    async def aembed(self, texts: Sequence[str]) -> list[list[float]]:
        semaphore = asyncio.Semaphore(_MAX_CONCURRENT_BATCHES)

        async def embed_batch(batch: tuple[str, ...]) -> list[list[float]]:
            async with semaphore:
                try:
                    response = await self._async_client.embeddings.create(
                        model=self._model,
                        input=list(batch),
                        dimensions=self._dimensions or openai.omit,
                        encoding_format="float",
                    )
                except openai.APIError as exc:
                    raise _upstream_error(exc) from exc
            return _vectors(response)

        results = await asyncio.gather(
            *(embed_batch(batch) for batch in batched(texts, self._batch_size, strict=False))
        )
        return [vector for batch_vectors in results for vector in batch_vectors]

    async def aclose(self) -> None:
        await self._async_client.close()
        self._client.close()


def _vectors(response: CreateEmbeddingResponse) -> list[list[float]]:
    return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]


def _upstream_error(exc: openai.APIError) -> UpstreamError:
    status = getattr(exc, "status_code", None)
    prefix = (
        f"Embeddings provider returned HTTP {status}" if status else "Embeddings request failed"
    )
    return UpstreamError(f"{prefix}: {exc.message}")


class EmbeddingsConfig(BaseModel):
    """An OpenAI-compatible embeddings provider supplied with the request.

    Only accepted when the server has no EMBEDDINGS_MODEL of its own."""

    model_config = ConfigDict(extra="forbid")

    base_url: HttpUrl = Field(
        default=HttpUrl("https://api.openai.com/v1"),
        description="OpenAI-compatible API root, e.g. https://api.mistral.ai/v1.",
    )
    api_key: SecretStr | None = Field(default=None, description="Sent as a Bearer token.")
    model: str = Field(min_length=1, examples=["text-embedding-3-small"])
    dimensions: PositiveInt | None = Field(
        default=None, description="Request shorter vectors, if the model supports it."
    )
    batch_size: int = Field(default=64, ge=1, le=2048, description="Texts per API call.")


def check_request_embeddings(
    config: EmbeddingsConfig | None, server: Embedder | None, settings: Settings
) -> None:
    """Server configuration wins; requests may only bring a provider when there is none."""
    if config is None:
        return
    if server is not None:
        raise ChunkerError(
            "Embeddings are configured on this server; remove `embeddings` from the request."
        )
    if not settings.allow_request_embeddings:
        raise NotConfiguredError(
            "This server does not accept `embeddings` in requests (ALLOW_REQUEST_EMBEDDINGS)."
        )


@asynccontextmanager
async def resolve_embedder(
    config: EmbeddingsConfig | None, server: Embedder | None, settings: Settings
) -> AsyncIterator[Embedder | None]:
    """Yield the request's embedder (closed afterwards) or the server's."""
    if config is None:
        yield server
        return
    embedder = OpenAICompatibleEmbedder(
        model=config.model,
        base_url=str(config.base_url),
        api_key=config.api_key.get_secret_value() if config.api_key else None,
        dimensions=config.dimensions,
        batch_size=config.batch_size,
        timeout=settings.embeddings_timeout,
        max_retries=settings.embeddings_max_retries,
    )
    try:
        yield embedder
    finally:
        await embedder.aclose()
