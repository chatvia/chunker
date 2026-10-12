"""Chunking engines.

Each strategy is a Pydantic model that both validates its options and executes itself. The
models are combined into discriminated unions (by `engine`, then `strategy`), which gives
precise request validation and an OpenAPI schema clients can generate code from.
"""

from typing import Annotated, Any, Union, get_args, get_origin

from pydantic import Discriminator, Tag

from app.engines.base import ChunkContext, ChunkerBase, Document, Page, RawChunk
from app.engines.chonkie import ChonkieChunker
from app.engines.langchain import LangchainChunker, LangchainRecursive
from app.engines.llamaindex import LlamaIndexChunker


def _engine(value: Any) -> str | None:
    engine = value.get("engine") if isinstance(value, dict) else getattr(value, "engine", None)
    return engine if isinstance(engine, str) else None


# The outer union uses a callable discriminator: a field discriminator would emit a nested
# `discriminator.mapping`, which OpenAPI does not allow. Inner unions discriminate on `strategy`.
ChunkerConfig = Annotated[
    Annotated[LangchainChunker, Tag("langchain")]
    | Annotated[LlamaIndexChunker, Tag("llamaindex")]
    | Annotated[ChonkieChunker, Tag("chonkie")],
    Discriminator(
        _engine,
        custom_error_type="invalid_engine",
        custom_error_message="Unknown chunker engine; see GET /v1/engines",
    ),
]


def _members(tp: Any) -> list[type]:
    """Flatten (possibly nested, Annotated) unions into their model classes."""
    if get_origin(tp) is Annotated:
        return _members(get_args(tp)[0])
    if get_origin(tp) is Union or type(tp).__name__ == "UnionType":
        return [model for arg in get_args(tp) for model in _members(arg)]
    return [tp]


CHUNKERS: list[type[ChunkerBase]] = _members(ChunkerConfig)

DEFAULT_CHUNKER = LangchainRecursive(engine="langchain", strategy="recursive")

__all__ = [
    "CHUNKERS",
    "DEFAULT_CHUNKER",
    "ChunkContext",
    "ChunkerBase",
    "ChunkerConfig",
    "Document",
    "Page",
    "RawChunk",
]
