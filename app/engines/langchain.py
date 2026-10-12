"""Chunkers backed by `langchain-text-splitters`."""

from typing import Annotated, Any, ClassVar, Literal, Self

from langchain_text_splitters import (
    CharacterTextSplitter,
    Language,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
    TextSplitter,
    TokenTextSplitter,
)
from pydantic import Field, model_validator

from app.engines.base import ChunkContext, ChunkerBase, ParsedDocument, RawChunk

LengthUnit = Literal["tokens", "characters"]


class _LangchainChunker(ChunkerBase):
    engine: Literal["langchain"]


class _SizedChunker(_LangchainChunker):
    chunk_size: int = Field(default=512, ge=1, description="Maximum chunk length.")
    chunk_overlap: int = Field(default=64, ge=0, description="Overlap between chunks.")

    @model_validator(mode="after")
    def _check_overlap(self) -> Self:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self

    def _sized_splitter[T: TextSplitter](
        self, cls: type[T], unit: LengthUnit, context: ChunkContext, **kwargs: Any
    ) -> T:
        kwargs |= {"chunk_size": self.chunk_size, "chunk_overlap": self.chunk_overlap}
        if unit == "tokens":
            return cls.from_tiktoken_encoder(
                encoding_name=context.encoding, disallowed_special=(), **kwargs
            )
        return cls(**kwargs)


def _split(splitter: TextSplitter, document: ParsedDocument) -> list[RawChunk]:
    # LangChain's `add_start_index` subtracts the overlap as characters even when lengths are
    # measured in tokens, producing wrong offsets; the pipeline locates the chunks instead.
    return [RawChunk(text=text) for text in splitter.split_text(document.text)]


class LangchainRecursive(_SizedChunker):
    """Recursively split on paragraphs, lines, sentences and words until chunks fit `chunk_size`.

    Set `language` for code-aware separators (python, js, markdown, html, ...)."""

    strategy: Literal["recursive"]
    length_unit: LengthUnit = "tokens"
    separators: list[str] | None = Field(default=None, min_length=1)
    language: Language | None = None

    def chunk(self, document: ParsedDocument, context: ChunkContext) -> list[RawChunk]:
        kwargs: dict[str, Any] = {}
        if self.language is not None:
            kwargs["separators"] = RecursiveCharacterTextSplitter.get_separators_for_language(
                self.language
            )
            kwargs["is_separator_regex"] = True
        elif self.separators is not None:
            kwargs["separators"] = self.separators
        splitter = self._sized_splitter(
            RecursiveCharacterTextSplitter, self.length_unit, context, **kwargs
        )
        return _split(splitter, document)


class LangchainCharacter(_SizedChunker):
    """Split on a single separator, then merge pieces up to `chunk_size`."""

    strategy: Literal["character"]
    length_unit: LengthUnit = "tokens"
    separator: str = "\n\n"

    def chunk(self, document: ParsedDocument, context: ChunkContext) -> list[RawChunk]:
        splitter = self._sized_splitter(
            CharacterTextSplitter, self.length_unit, context, separator=self.separator
        )
        return _split(splitter, document)


class LangchainToken(_SizedChunker):
    """Split into fixed windows of `chunk_size` tokens, ignoring text structure."""

    strategy: Literal["token"]

    def chunk(self, document: ParsedDocument, context: ChunkContext) -> list[RawChunk]:
        splitter = TokenTextSplitter(
            encoding_name=context.encoding,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            disallowed_special=(),
        )
        return _split(splitter, document)


class LangchainMarkdown(_LangchainChunker):
    """Split Markdown into sections by headers; each chunk carries its header hierarchy.

    Set `chunk_size` to further split long sections (in tokens)."""

    strategy: Literal["markdown"]
    max_header_level: int = Field(default=3, ge=1, le=6)
    strip_headers: bool = False
    chunk_size: int | None = Field(default=None, ge=1)
    chunk_overlap: int = Field(default=0, ge=0)

    verbatim: ClassVar[bool] = False  # the splitter normalizes whitespace

    @model_validator(mode="after")
    def _check_overlap(self) -> Self:
        if self.chunk_size is not None and self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self

    def chunk(self, document: ParsedDocument, context: ChunkContext) -> list[RawChunk]:
        headers = [("#" * level, f"h{level}") for level in range(1, self.max_header_level + 1)]
        sections = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers, strip_headers=self.strip_headers
        ).split_text(document.text)
        if self.chunk_size is not None:
            sections = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                encoding_name=context.encoding,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                disallowed_special=(),
            ).split_documents(sections)
        return [
            RawChunk(text=section.page_content, metadata={"headers": section.metadata})
            for section in sections
        ]


LangchainChunker = Annotated[
    LangchainRecursive | LangchainCharacter | LangchainToken | LangchainMarkdown,
    Field(discriminator="strategy"),
]
