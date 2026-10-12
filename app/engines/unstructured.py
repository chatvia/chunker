"""Parser and chunkers backed by the open-source `unstructured` library (runs in-process)."""

import io
from typing import Annotated, Any, ClassVar, Literal, Self

from pydantic import Field, model_validator
from unstructured.chunking.basic import chunk_elements
from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import Element as UnstructuredElement
from unstructured.partition.auto import partition
from unstructured.partition.text import partition_text

from app.engines.base import (
    ChunkContext,
    ChunkerBase,
    Element,
    ParsedDocument,
    ParserBase,
    RawChunk,
)

ENGINE = "unstructured"


def _metadata(element: UnstructuredElement) -> dict[str, Any]:
    metadata: dict[str, Any] = element.metadata.to_dict()
    metadata.pop("orig_elements", None)
    return metadata


class UnstructuredParser(ParserBase):
    """Partition files into typed elements (Title, NarrativeText, Table, ...) with page numbers.

    `fast` extracts embedded text with no extra system packages. `hi_res`, `ocr_only` and `auto`
    run layout detection/OCR and need poppler and tesseract installed on the server."""

    engine: Literal["unstructured"]
    strategy: Literal["fast", "hi_res", "ocr_only", "auto"] = "fast"
    languages: list[str] | None = Field(
        default=None, description="Tesseract language codes for OCR, e.g. ['eng', 'ara']."
    )

    extensions: ClassVar[frozenset[str]] = frozenset(
        {
            ".pdf", ".docx", ".pptx", ".xlsx", ".csv", ".tsv", ".md", ".txt",
            ".html", ".htm", ".xml", ".eml", ".msg",
        }
    )  # fmt: skip

    def parse(self, data: bytes, filename: str) -> ParsedDocument:
        native = partition(
            file=io.BytesIO(data),
            metadata_filename=filename,
            strategy=self.strategy,
            languages=self.languages,
        )
        elements = [
            Element(
                text=element.text,
                type=element.category,
                page_number=element.metadata.page_number,
                metadata=_metadata(element),
            )
            for element in native
        ]
        return ParsedDocument(
            elements=elements, filename=filename, native_engine=ENGINE, native=native
        )


class _UnstructuredChunker(ChunkerBase):
    engine: Literal["unstructured"]
    max_characters: int = Field(default=1000, ge=1, description="Hard maximum chunk length.")
    new_after_n_chars: int | None = Field(
        default=None, ge=1, description="Soft maximum: start a new chunk after this length."
    )
    overlap: int = Field(default=0, ge=0, description="Characters of overlap between chunks.")
    overlap_all: bool = Field(
        default=False, description="Also overlap chunks that were not split mid-element."
    )

    verbatim: ClassVar[bool] = False  # element texts are cleaned and re-joined

    @model_validator(mode="after")
    def _check_sizes(self) -> Self:
        if self.overlap >= self.max_characters:
            raise ValueError("overlap must be smaller than max_characters")
        if self.new_after_n_chars is not None and self.new_after_n_chars > self.max_characters:
            raise ValueError("new_after_n_chars must not exceed max_characters")
        return self

    def _common_options(self) -> dict[str, Any]:
        return {
            "max_characters": self.max_characters,
            "new_after_n_chars": self.new_after_n_chars,
            "overlap": self.overlap,
            "overlap_all": self.overlap_all,
            "include_orig_elements": True,  # used for page numbers, then dropped
        }

    @staticmethod
    def _elements(document: ParsedDocument) -> list[UnstructuredElement]:
        if document.native_engine == ENGINE and document.native is not None:
            return document.native
        return partition_text(text=document.text)

    @staticmethod
    def _to_raw(chunks: list[UnstructuredElement]) -> list[RawChunk]:
        raw_chunks = []
        for chunk in chunks:
            pages = {
                element.metadata.page_number
                for element in chunk.metadata.orig_elements or []
                if element.metadata.page_number is not None
            }
            raw_chunks.append(
                RawChunk(
                    text=chunk.text,
                    metadata={"type": chunk.category, **_metadata(chunk)},
                    page_numbers=sorted(pages),
                )
            )
        return raw_chunks


class UnstructuredBasic(_UnstructuredChunker):
    """Fill chunks with whole elements up to `max_characters`, splitting oversized elements."""

    strategy: Literal["basic"]

    def chunk(self, document: ParsedDocument, context: ChunkContext) -> list[RawChunk]:
        return self._to_raw(chunk_elements(self._elements(document), **self._common_options()))


class UnstructuredByTitle(_UnstructuredChunker):
    """Like `basic`, but every Title element starts a new chunk so sections stay separate."""

    strategy: Literal["by_title"]
    combine_text_under_n_chars: int | None = Field(
        default=None, ge=0, description="Merge small consecutive sections up to this length."
    )
    multipage_sections: bool = Field(
        default=True,
        description="Allow sections to span pages. Small sections are still merged across pages "
        "unless `combine_text_under_n_chars` is 0.",
    )

    def chunk(self, document: ParsedDocument, context: ChunkContext) -> list[RawChunk]:
        chunks = chunk_by_title(
            self._elements(document),
            combine_text_under_n_chars=self.combine_text_under_n_chars,
            multipage_sections=self.multipage_sections,
            **self._common_options(),
        )
        return self._to_raw(chunks)


UnstructuredChunker = Annotated[
    UnstructuredBasic | UnstructuredByTitle, Field(discriminator="strategy")
]
