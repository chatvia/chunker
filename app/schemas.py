from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from app.engines import DEFAULT_TEXT_CHUNKER, ChunkerConfig
from app.tokens import DEFAULT_ENCODING, Encoding

MAX_TEXT_LENGTH = 10_000_000
MAX_EMBED_INPUTS = 2048

PARSER_FORM_HELP = 'Parser as JSON. Default: {"engine": "unstructured", "strategy": "fast"}'
CHUNKER_FORM_HELP = 'Chunker as JSON. Default: {"engine": "unstructured", "strategy": "by_title"}'


class ChunkTextRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "text": "Long text to split...",
                    "chunker": {"engine": "llamaindex", "strategy": "sentence", "chunk_size": 256},
                    "embed": False,
                }
            ]
        },
    )

    text: str = Field(min_length=1, max_length=MAX_TEXT_LENGTH)
    chunker: ChunkerConfig = Field(default=DEFAULT_TEXT_CHUNKER)
    encoding: Encoding = Field(
        default=DEFAULT_ENCODING,
        description="tiktoken encoding used by token-based strategies and for `token_count`.",
    )
    embed: bool = Field(default=False, description="Embed each chunk with the server's model.")
    webhook_url: HttpUrl | None = Field(
        default=None,
        description="Process asynchronously: respond 202 with a `job_id` and POST the result here.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Opaque data echoed back in the response and webhook, e.g. your own IDs.",
    )


class ChunkOut(BaseModel):
    index: int
    text: str
    token_count: int
    start_index: int | None = Field(
        default=None, description="Character offset in the (parsed) document text."
    )
    end_index: int | None = None
    page_numbers: list[int] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None


class DocumentInfo(BaseModel):
    filename: str | None = None
    content_type: str | None = None
    characters: int
    token_count: int


class StrategyInfo(BaseModel):
    engine: str
    strategy: str


class ChunkResponse(BaseModel):
    parser: str | None = Field(default=None, description="Parser engine, for file requests.")
    chunker: StrategyInfo
    document: DocumentInfo
    total_chunks: int
    chunks: list[ChunkOut]
    embedding_model: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class JobAccepted(BaseModel):
    job_id: str
    status: Literal["accepted"] = "accepted"


class ElementOut(BaseModel):
    text: str
    type: str
    page_number: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParseResponse(BaseModel):
    parser: str
    document: DocumentInfo
    text: str = Field(description="All element texts joined by blank lines.")
    elements: list[ElementOut]


class EmbedRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: list[Annotated[str, Field(min_length=1)]] = Field(
        min_length=1, max_length=MAX_EMBED_INPUTS
    )


class EmbeddingOut(BaseModel):
    index: int
    embedding: list[float]


class EmbedResponse(BaseModel):
    model: str
    data: list[EmbeddingOut]


class ParserInfo(BaseModel):
    engine: str
    description: str
    extensions: list[str]
    options: dict[str, Any] = Field(description="JSON Schema of the parser's options.")


class ChunkerInfo(BaseModel):
    engine: str
    strategy: str
    description: str
    requires_embeddings: bool
    options: dict[str, Any] = Field(description="JSON Schema of the strategy's options.")


class EmbeddingsInfo(BaseModel):
    configured: bool
    model: str | None = None


class EnginesResponse(BaseModel):
    parsers: list[ParserInfo]
    chunkers: list[ChunkerInfo]
    encodings: list[str]
    embeddings: EmbeddingsInfo


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    version: str
