from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

from app.embeddings import EmbeddingsConfig
from app.engines import DEFAULT_CHUNKER, ChunkerConfig, Document
from app.tokens import DEFAULT_ENCODING, Encoding

MAX_TEXT_LENGTH = 10_000_000
MAX_PAGES = 10_000
MAX_EMBED_INPUTS = 2048


class PageIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(description="The page's text or Markdown, e.g. from an OCR service.")
    page_number: int | None = Field(
        default=None, ge=1, description="Defaults to the page's position in `pages` (1-based)."
    )


class ChunkRequest(BaseModel):
    """Send either `text`, or `pages` to know which pages each chunk comes from."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "text": "Long text to split...",
                    "chunker": {"engine": "llamaindex", "strategy": "sentence", "chunk_size": 256},
                },
                {
                    "pages": [{"text": "# Page one\n\n..."}, {"text": "Page two..."}],
                    "chunker": {"engine": "chonkie", "strategy": "semantic", "threshold": 0.75},
                    "embed": True,
                    "embeddings": {
                        "base_url": "https://api.openai.com/v1",
                        "api_key": "sk-...",
                        "model": "text-embedding-3-small",
                    },
                },
            ]
        },
    )

    text: str | None = Field(default=None, min_length=1, max_length=MAX_TEXT_LENGTH)
    pages: list[PageIn] | None = Field(default=None, min_length=1, max_length=MAX_PAGES)
    chunker: ChunkerConfig = Field(default=DEFAULT_CHUNKER)
    encoding: Encoding = Field(
        default=DEFAULT_ENCODING,
        description="tiktoken encoding used by token-based strategies and for `token_count`.",
    )
    embed: bool = Field(default=False, description="Return an embedding for every chunk.")
    embeddings: EmbeddingsConfig | None = Field(
        default=None,
        description="Embeddings provider for semantic strategies and `embed`, when the server "
        "has none configured.",
    )
    webhook_url: HttpUrl | None = Field(
        default=None,
        description="Process asynchronously: respond 202 with a `job_id` and POST the result here.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Opaque data echoed back in the response and webhook, e.g. your own IDs.",
    )

    @model_validator(mode="after")
    def _check_input(self) -> Self:
        if (self.text is None) == (self.pages is None):
            raise ValueError("send exactly one of `text` or `pages`")
        if self.pages is not None:
            if not any(page.text.strip() for page in self.pages):
                raise ValueError("`pages` contain no text")
            if sum(len(page.text) for page in self.pages) > MAX_TEXT_LENGTH:
                raise ValueError(f"`pages` exceed {MAX_TEXT_LENGTH} characters in total")
        return self

    def document(self) -> Document:
        if self.pages is None:
            return Document.from_text(self.text or "")
        return Document.from_pages(
            (page.text, page.page_number or position)
            for position, page in enumerate(self.pages, start=1)
        )


class ChunkOut(BaseModel):
    index: int
    text: str
    token_count: int
    start_index: int | None = Field(
        default=None, description="Character offset in the input text (pages joined by \\n\\n)."
    )
    end_index: int | None = None
    page_numbers: list[int] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None


class DocumentInfo(BaseModel):
    characters: int
    token_count: int
    pages: int | None = None


class StrategyInfo(BaseModel):
    engine: str
    strategy: str


class ChunkResponse(BaseModel):
    chunker: StrategyInfo
    document: DocumentInfo
    total_chunks: int
    chunks: list[ChunkOut]
    embedding_model: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class JobAccepted(BaseModel):
    job_id: str
    status: Literal["accepted"] = "accepted"


class EmbedRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: list[Annotated[str, Field(min_length=1)]] = Field(
        min_length=1, max_length=MAX_EMBED_INPUTS
    )
    embeddings: EmbeddingsConfig | None = Field(
        default=None, description="Embeddings provider, when the server has none configured."
    )


class EmbeddingOut(BaseModel):
    index: int
    embedding: list[float]


class EmbedResponse(BaseModel):
    model: str
    data: list[EmbeddingOut]


class ChunkerInfo(BaseModel):
    engine: str
    strategy: str
    description: str
    requires_embeddings: bool
    options: dict[str, Any] = Field(description="JSON Schema of the strategy's options.")


class TokensRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: list[str] = Field(min_length=1, max_length=MAX_EMBED_INPUTS)
    encoding: Encoding = DEFAULT_ENCODING

    @model_validator(mode="after")
    def _check_length(self) -> Self:
        if sum(len(text) for text in self.input) > MAX_TEXT_LENGTH:
            raise ValueError(f"`input` exceeds {MAX_TEXT_LENGTH} characters in total")
        return self


class TokenCount(BaseModel):
    index: int
    tokens: int


class TokensResponse(BaseModel):
    encoding: str
    data: list[TokenCount]


class EmbeddingsInfo(BaseModel):
    configured: bool = Field(description="An embeddings model is configured on the server.")
    model: str | None = None
    per_request: bool = Field(description="Requests may bring their own `embeddings` provider.")


class EnginesResponse(BaseModel):
    chunkers: list[ChunkerInfo]
    encodings: list[str]
    embeddings: EmbeddingsInfo


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    version: str
