from bisect import bisect_right
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, ClassVar, Self

from pydantic import BaseModel, ConfigDict

from app.embeddings import Embedder
from app.errors import NotConfiguredError

PAGE_SEPARATOR = "\n\n"


@dataclass(slots=True)
class Page:
    text: str
    number: int | None = None


@dataclass
class Document:
    """The text to chunk: plain text is a single page without a number."""

    pages: list[Page]

    @classmethod
    def from_text(cls, text: str) -> Self:
        return cls(pages=[Page(text)])

    @classmethod
    def from_pages(cls, pages: Iterable[tuple[str, int]]) -> Self:
        return cls(pages=[Page(text, number) for text, number in pages])

    @cached_property
    def _layout(self) -> tuple[str, list[int], list[int | None]]:
        parts: list[str] = []
        starts: list[int] = []
        numbers: list[int | None] = []
        offset = 0
        for page in self.pages:
            if not page.text:
                continue
            if parts:
                parts.append(PAGE_SEPARATOR)
                offset += len(PAGE_SEPARATOR)
            parts.append(page.text)
            starts.append(offset)
            numbers.append(page.number)
            offset += len(page.text)
        return "".join(parts), starts, numbers

    @property
    def text(self) -> str:
        """All page texts joined by blank lines; chunk offsets refer to this string."""
        return self._layout[0]

    def page_numbers(self, start: int, end: int) -> list[int]:
        """Numbers of the pages overlapping the character range [start, end)."""
        _, starts, numbers = self._layout
        first = max(bisect_right(starts, start) - 1, 0)
        last = bisect_right(starts, max(end - 1, start))
        return sorted({number for number in numbers[first:last] if number is not None})


@dataclass(slots=True)
class RawChunk:
    text: str
    start_index: int | None = None
    end_index: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ChunkContext:
    encoding: str
    embedder: Embedder | None = None

    def require_embedder(self) -> Embedder:
        if self.embedder is None:
            raise NotConfiguredError("This strategy needs embeddings, but none are configured.")
        return self.embedder


class ChunkerBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    engine: str
    strategy: str

    requires_embeddings: ClassVar[bool] = False
    # Chunk texts are verbatim slices of the document, so missing offsets can be located.
    verbatim: ClassVar[bool] = True

    def preflight(self, *, embeddings: bool) -> None:
        """Reject a request that cannot run, before any work (or a 202) happens."""
        if self.requires_embeddings and not embeddings:
            raise NotConfiguredError(
                "This strategy needs embeddings: set EMBEDDINGS_MODEL on the server or send "
                "`embeddings` with the request."
            )

    def chunk(self, document: Document, context: ChunkContext) -> list[RawChunk]:
        raise NotImplementedError
