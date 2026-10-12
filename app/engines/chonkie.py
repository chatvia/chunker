"""Chunkers backed by Chonkie."""

from typing import Annotated, Any, ClassVar, Literal, Self

import numpy as np
from chonkie import (
    CodeChunker,
    FastChunker,
    RecursiveChunker,
    SemanticChunker,
    SentenceChunker,
    TableChunker,
    TokenChunker,
)
from chonkie.embeddings import BaseEmbeddings
from chonkie.types import Chunk
from chonkie.types.recursive import RecursiveLevel, RecursiveRules
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.embeddings import Embedder
from app.engines import grammars
from app.engines.base import ChunkContext, ChunkerBase, Document, RawChunk
from app.tokens import get_encoding

IncludeDelim = Literal["prev", "next"] | None
DEFAULT_DELIMITERS = [". ", "! ", "? ", "\n"]


class _EmbedderAdapter(BaseEmbeddings):
    """Exposes the configured embedder as a Chonkie embedding model."""

    def __init__(self, embedder: Embedder, encoding: str) -> None:
        super().__init__()
        self._embedder = embedder
        self._encoding = encoding
        self._dimension: int | None = None

    def embed(self, text: str) -> np.ndarray:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        return [np.asarray(vector, dtype=np.float32) for vector in self._embedder.embed(texts)]

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._dimension = len(self._embedder.embed(["dimension probe"])[0])
        return self._dimension

    def get_tokenizer(self) -> Any:
        return get_encoding(self._encoding)


def _to_raw(chunks: list[Chunk]) -> list[RawChunk]:
    return [RawChunk(chunk.text, chunk.start_index, chunk.end_index) for chunk in chunks]


class _ChonkieChunker(ChunkerBase):
    engine: Literal["chonkie"]
    chunk_size: int = Field(default=512, ge=1, description="Maximum chunk length in tokens.")


class _Sentences(_ChonkieChunker):
    min_sentences_per_chunk: int = Field(default=1, ge=1)
    delim: list[str] = Field(
        default_factory=lambda: list(DEFAULT_DELIMITERS),
        min_length=1,
        description="Sentence delimiters.",
    )
    include_delim: IncludeDelim = Field(
        default="prev", description="Attach delimiters to the previous or next sentence."
    )


class ChonkieToken(_ChonkieChunker):
    """Split into fixed windows of `chunk_size` tokens (`TokenChunker`)."""

    strategy: Literal["token"]
    chunk_overlap: Annotated[int, Field(ge=0)] | Annotated[float, Field(gt=0, lt=1)] = Field(
        default=0, description="Overlap in tokens, or as a fraction of `chunk_size` (0-1)."
    )

    @model_validator(mode="after")
    def _check_overlap(self) -> Self:
        if isinstance(self.chunk_overlap, int) and self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self

    def chunk(self, document: Document, context: ChunkContext) -> list[RawChunk]:
        chunker = TokenChunker(
            tokenizer=context.encoding,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        return _to_raw(chunker.chunk(document.text))


class ChonkieSentence(_Sentences):
    """Split into chunks of whole sentences up to `chunk_size` tokens (`SentenceChunker`)."""

    strategy: Literal["sentence"]
    chunk_overlap: int = Field(default=0, ge=0, description="Token overlap between chunks.")
    min_characters_per_sentence: int = Field(default=12, ge=1)

    @model_validator(mode="after")
    def _check_overlap(self) -> Self:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self

    def chunk(self, document: Document, context: ChunkContext) -> list[RawChunk]:
        chunker = SentenceChunker(
            tokenizer=context.encoding,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            min_sentences_per_chunk=self.min_sentences_per_chunk,
            min_characters_per_sentence=self.min_characters_per_sentence,
            delim=self.delim,
            include_delim=self.include_delim,
        )
        return _to_raw(chunker.chunk(document.text))


class RecursiveLevelOptions(BaseModel):
    """One level of the recursive hierarchy: split on `delimiters`, on whitespace, or on a
    regex `pattern`; a level with none of them splits on token boundaries."""

    model_config = ConfigDict(extra="forbid")

    delimiters: list[str] | None = Field(default=None, min_length=1)
    whitespace: bool = False
    include_delim: IncludeDelim = "prev"
    pattern: str | None = None
    pattern_mode: Literal["split", "extract"] = "split"

    @model_validator(mode="after")
    def _check(self) -> Self:
        try:
            self.to_level()  # Chonkie validates the combination
        except (NotImplementedError, ValueError) as exc:
            raise ValueError(str(exc)) from exc
        return self

    def to_level(self) -> RecursiveLevel:
        return RecursiveLevel(**self.model_dump())


class ChonkieRecursive(_ChonkieChunker):
    """Recursively split on paragraphs, sentences, punctuation and words (`RecursiveChunker`).

    Set `levels` to define your own hierarchy of delimiters."""

    strategy: Literal["recursive"]
    min_characters_per_chunk: int = Field(default=24, ge=1)
    levels: list[RecursiveLevelOptions] | None = Field(
        default=None, min_length=1, description="Custom rules, tried in order."
    )

    def chunk(self, document: Document, context: ChunkContext) -> list[RawChunk]:
        options: dict[str, Any] = {}
        if self.levels is not None:
            options["rules"] = RecursiveRules(levels=[level.to_level() for level in self.levels])
        chunker = RecursiveChunker(
            tokenizer=context.encoding,
            chunk_size=self.chunk_size,
            min_characters_per_chunk=self.min_characters_per_chunk,
            **options,
        )
        return _to_raw(chunker.chunk(document.text))


class ChonkieSemantic(_Sentences):
    """Group consecutive sentences while their embedding similarity stays above `threshold`,
    smoothing the similarity curve with a Savitzky-Golay filter (`SemanticChunker`).
    Needs embeddings."""

    strategy: Literal["semantic"]
    threshold: float = Field(default=0.8, gt=0, le=1, description="Similarity threshold.")
    similarity_window: int = Field(default=3, ge=1)
    min_characters_per_sentence: int = Field(default=24, ge=1)
    skip_window: int = Field(default=0, ge=0, description="Merge similar non-adjacent groups.")
    filter_window: int = Field(default=5, ge=2)
    filter_polyorder: int = Field(default=3, ge=0)
    filter_tolerance: float = Field(default=0.2, ge=0, le=1)

    requires_embeddings: ClassVar[bool] = True

    @model_validator(mode="after")
    def _check_filter(self) -> Self:
        if self.filter_polyorder >= self.filter_window:
            raise ValueError("filter_polyorder must be smaller than filter_window")
        return self

    def chunk(self, document: Document, context: ChunkContext) -> list[RawChunk]:
        options = self.model_dump(exclude={"engine", "strategy"})
        chunker = SemanticChunker(
            embedding_model=_EmbedderAdapter(context.require_embedder(), context.encoding),
            **options,
        )
        return _to_raw(chunker.chunk(document.text))


class ChonkieFast(_ChonkieChunker):
    """Very fast split at delimiter bytes, up to `chunk_size` bytes (`FastChunker`)."""

    strategy: Literal["fast"]
    chunk_size: int = Field(default=4096, ge=1, description="Maximum chunk length in bytes.")
    delimiters: str = Field(default="\n.?", min_length=1, description="Single-byte delimiters.")
    pattern: str | None = Field(default=None, description="Multi-byte delimiter instead.")
    prefix: bool = Field(default=False, description="Put delimiters at the start of chunks.")
    consecutive: bool = Field(default=False, description="Split at runs of delimiters.")
    forward_fallback: bool = Field(
        default=False, description="Search forward when no delimiter fits."
    )

    def chunk(self, document: Document, context: ChunkContext) -> list[RawChunk]:
        chunker = FastChunker(**self.model_dump(exclude={"engine", "strategy"}))
        return _to_raw(chunker.chunk(document.text))


class ChonkieTable(_ChonkieChunker):
    """Split a Markdown table into smaller tables, repeating the header in each
    (`TableChunker`). The input is a Markdown table."""

    strategy: Literal["table"]
    unit: Literal["rows", "tokens"] = Field(default="rows", description="Unit of `chunk_size`.")
    chunk_size: int = Field(default=10, ge=1, description="Data rows, or tokens, per chunk.")

    verbatim: ClassVar[bool] = False  # headers are repeated in every chunk

    def chunk(self, document: Document, context: ChunkContext) -> list[RawChunk]:
        tokenizer = "row" if self.unit == "rows" else context.encoding
        chunker = TableChunker(tokenizer=tokenizer, chunk_size=self.chunk_size)
        return _to_raw(chunker.chunk(document.text))


class ChonkieCode(_ChonkieChunker):
    """Split source code along its syntax tree with tree-sitter (`CodeChunker`).
    `language` must be one of the server's CODE_LANGUAGES."""

    strategy: Literal["code"]
    language: str = Field(min_length=1, examples=["python"])

    def preflight(self, *, embeddings: bool) -> None:
        super().preflight(embeddings=embeddings)
        grammars.require(self.language, minimum_installed=grammars.CHONKIE_MIN_GRAMMARS)

    def chunk(self, document: Document, context: ChunkContext) -> list[RawChunk]:
        # Checked again here: below the minimum, Chonkie would download every grammar.
        grammars.require(self.language, minimum_installed=grammars.CHONKIE_MIN_GRAMMARS)
        chunker = CodeChunker(
            tokenizer=context.encoding, chunk_size=self.chunk_size, language=self.language
        )
        return _to_raw(chunker.chunk(document.text))


ChonkieChunker = Annotated[
    ChonkieToken
    | ChonkieSentence
    | ChonkieRecursive
    | ChonkieSemantic
    | ChonkieFast
    | ChonkieTable
    | ChonkieCode,
    Field(discriminator="strategy"),
]
