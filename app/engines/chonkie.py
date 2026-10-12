"""Chunkers backed by Chonkie."""

from typing import Annotated, Any, ClassVar, Literal, Self

import numpy as np
from chonkie import RecursiveChunker, SemanticChunker, SentenceChunker, TokenChunker
from chonkie.embeddings import BaseEmbeddings
from chonkie.types import Chunk
from pydantic import Field, model_validator

from app.embeddings import Embedder
from app.engines.base import ChunkContext, ChunkerBase, ParsedDocument, RawChunk
from app.tokens import get_encoding


class _EmbedderAdapter(BaseEmbeddings):
    """Exposes the server-configured embedder as a Chonkie embedding model."""

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


class _OverlappingChunker(_ChonkieChunker):
    chunk_overlap: int = Field(default=0, ge=0, description="Token overlap between chunks.")

    @model_validator(mode="after")
    def _check_overlap(self) -> Self:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


class ChonkieToken(_OverlappingChunker):
    """Split into fixed windows of `chunk_size` tokens (`TokenChunker`)."""

    strategy: Literal["token"]

    def chunk(self, document: ParsedDocument, context: ChunkContext) -> list[RawChunk]:
        chunker = TokenChunker(
            tokenizer=context.encoding,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        return _to_raw(chunker.chunk(document.text))


class ChonkieSentence(_OverlappingChunker):
    """Split into chunks of whole sentences up to `chunk_size` tokens (`SentenceChunker`)."""

    strategy: Literal["sentence"]
    min_sentences_per_chunk: int = Field(default=1, ge=1)

    def chunk(self, document: ParsedDocument, context: ChunkContext) -> list[RawChunk]:
        chunker = SentenceChunker(
            tokenizer=context.encoding,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            min_sentences_per_chunk=self.min_sentences_per_chunk,
        )
        return _to_raw(chunker.chunk(document.text))


class ChonkieRecursive(_ChonkieChunker):
    """Recursively split on paragraphs, sentences, punctuation and words (`RecursiveChunker`)."""

    strategy: Literal["recursive"]
    min_characters_per_chunk: int = Field(default=24, ge=1)

    def chunk(self, document: ParsedDocument, context: ChunkContext) -> list[RawChunk]:
        chunker = RecursiveChunker(
            tokenizer=context.encoding,
            chunk_size=self.chunk_size,
            min_characters_per_chunk=self.min_characters_per_chunk,
        )
        return _to_raw(chunker.chunk(document.text))


class ChonkieSemantic(_ChonkieChunker):
    """Group consecutive sentences while their embedding similarity stays above `threshold`
    (`SemanticChunker`). Requires the server's embeddings model."""

    strategy: Literal["semantic"]
    threshold: float = Field(default=0.8, gt=0, le=1, description="Similarity threshold.")
    similarity_window: int = Field(default=3, ge=1)
    min_sentences_per_chunk: int = Field(default=1, ge=1)

    requires_embeddings: ClassVar[bool] = True

    def chunk(self, document: ParsedDocument, context: ChunkContext) -> list[RawChunk]:
        chunker = SemanticChunker(
            embedding_model=_EmbedderAdapter(context.require_embedder(), context.encoding),
            threshold=self.threshold,
            chunk_size=self.chunk_size,
            similarity_window=self.similarity_window,
            min_sentences_per_chunk=self.min_sentences_per_chunk,
        )
        return _to_raw(chunker.chunk(document.text))


ChonkieChunker = Annotated[
    ChonkieToken | ChonkieSentence | ChonkieRecursive | ChonkieSemantic,
    Field(discriminator="strategy"),
]
