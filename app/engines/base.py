from bisect import bisect_right
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, ClassVar, Self

from pydantic import BaseModel, ConfigDict

from app.embeddings import Embedder
from app.errors import NotConfiguredError

ELEMENT_SEPARATOR = "\n\n"


@dataclass(slots=True)
class Element:
    """A piece of a parsed document: an Unstructured element or a LlamaIndex document/page."""

    text: str
    type: str = "Text"
    page_number: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedDocument:
    elements: list[Element]
    filename: str | None = None
    content_type: str | None = None
    # Engine-native objects (e.g. Unstructured elements) so same-engine chunkers can reuse them.
    native_engine: str | None = None
    native: list[Any] | None = None

    @classmethod
    def from_text(cls, text: str) -> Self:
        return cls(elements=[Element(text=text)])

    @cached_property
    def _layout(self) -> tuple[str, list[int], list[int | None]]:
        parts: list[str] = []
        starts: list[int] = []
        pages: list[int | None] = []
        offset = 0
        for element in self.elements:
            if not element.text:
                continue
            if parts:
                parts.append(ELEMENT_SEPARATOR)
                offset += len(ELEMENT_SEPARATOR)
            parts.append(element.text)
            starts.append(offset)
            pages.append(element.page_number)
            offset += len(element.text)
        return "".join(parts), starts, pages

    @property
    def text(self) -> str:
        """All element texts joined by blank lines; chunk offsets refer to this string."""
        return self._layout[0]

    def page_numbers(self, start: int, end: int) -> list[int]:
        """Page numbers of the elements overlapping the character range [start, end)."""
        _, starts, pages = self._layout
        first = max(bisect_right(starts, start) - 1, 0)
        last = bisect_right(starts, max(end - 1, start))
        return sorted({page for page in pages[first:last] if page is not None})


@dataclass(slots=True)
class RawChunk:
    text: str
    start_index: int | None = None
    end_index: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    # Set by engines that know the pages directly; otherwise derived from the offsets.
    page_numbers: list[int] | None = None


@dataclass(frozen=True, slots=True)
class ChunkContext:
    encoding: str
    embedder: Embedder | None = None

    def require_embedder(self) -> Embedder:
        if self.embedder is None:
            raise NotConfiguredError(
                "This strategy needs an embeddings model; set EMBEDDINGS_MODEL on the server."
            )
        return self.embedder


class EngineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ChunkerBase(EngineConfig):
    engine: str
    strategy: str

    requires_embeddings: ClassVar[bool] = False
    # Chunk texts are verbatim slices of the document, so missing offsets can be located.
    verbatim: ClassVar[bool] = True

    def chunk(self, document: ParsedDocument, context: ChunkContext) -> list[RawChunk]:
        raise NotImplementedError


class ParserBase(EngineConfig):
    engine: str

    extensions: ClassVar[frozenset[str]]

    def parse(self, data: bytes, filename: str) -> ParsedDocument:
        raise NotImplementedError
