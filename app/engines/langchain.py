"""Chunkers backed by `langchain-text-splitters`."""

import json
import re
from typing import Annotated, Any, ClassVar, Literal, Self

from langchain_core.documents import Document as LangchainDocument
from langchain_text_splitters import (
    CharacterTextSplitter,
    ExperimentalMarkdownSyntaxTextSplitter,
    HTMLHeaderTextSplitter,
    HTMLSectionSplitter,
    HTMLSemanticPreservingSplitter,
    JSFrameworkTextSplitter,
    Language,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
    RecursiveJsonSplitter,
    TextSplitter,
    TokenTextSplitter,
)
from langchain_text_splitters.nltk import NLTKTextSplitter
from pydantic import Field, model_validator

from app.engines.base import ChunkContext, ChunkerBase, Document, RawChunk
from app.errors import InvalidInputError

LengthUnit = Literal["tokens", "characters"]
KeepSeparator = bool | Literal["start", "end"]

_HEADER_KEY = re.compile(r"h[1-6]")


class _LangchainChunker(ChunkerBase):
    engine: Literal["langchain"]


def _check_overlap(chunk_size: int | None, chunk_overlap: int) -> None:
    if chunk_size is not None and chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")


class _Sized(_LangchainChunker):
    chunk_size: int = Field(default=512, ge=1, description="Maximum chunk length.")
    chunk_overlap: int = Field(default=64, ge=0, description="Overlap between chunks.")
    strip_whitespace: bool = Field(default=True, description="Trim whitespace around chunks.")

    @model_validator(mode="after")
    def _check_sizes(self) -> Self:
        _check_overlap(self.chunk_size, self.chunk_overlap)
        return self

    def _size_options(self) -> dict[str, Any]:
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "strip_whitespace": self.strip_whitespace,
        }


class _Measured(_Sized):
    length_unit: LengthUnit = Field(
        default="tokens", description="Unit of `chunk_size`/`chunk_overlap`."
    )

    def _splitter[T: TextSplitter](self, cls: type[T], context: ChunkContext, **kwargs: Any) -> T:
        kwargs |= self._size_options()
        if self.length_unit == "tokens":
            return cls.from_tiktoken_encoder(
                encoding_name=context.encoding, disallowed_special=(), **kwargs
            )
        return cls(**kwargs)


def _split(splitter: TextSplitter, document: Document) -> list[RawChunk]:
    # LangChain's `add_start_index` subtracts the overlap as characters even when lengths are
    # measured in tokens, producing wrong offsets; the pipeline locates the chunks instead.
    return [RawChunk(text=text) for text in splitter.split_text(document.text)]


class LangchainRecursive(_Measured):
    """Recursively split on paragraphs, lines, sentences and words until chunks fit `chunk_size`.

    Set `language` for code- or markup-aware separators (python, js, markdown, latex, html, ...)."""

    strategy: Literal["recursive"]
    separators: list[str] | None = Field(default=None, min_length=1)
    is_separator_regex: bool = False
    keep_separator: KeepSeparator = Field(
        default=True,
        description="Keep separators, attached to the next (`start`) or previous chunk.",
    )
    language: Language | None = Field(default=None, description="Overrides `separators`.")

    def chunk(self, document: Document, context: ChunkContext) -> list[RawChunk]:
        kwargs: dict[str, Any] = {
            "keep_separator": self.keep_separator,
            "is_separator_regex": self.is_separator_regex,
        }
        if self.language is not None:
            kwargs["separators"] = RecursiveCharacterTextSplitter.get_separators_for_language(
                self.language
            )
            kwargs["is_separator_regex"] = True
        elif self.separators is not None:
            kwargs["separators"] = self.separators
        return _split(self._splitter(RecursiveCharacterTextSplitter, context, **kwargs), document)


class LangchainCharacter(_Measured):
    """Split on a single separator, then merge pieces up to `chunk_size`."""

    strategy: Literal["character"]
    separator: str = "\n\n"
    is_separator_regex: bool = False
    keep_separator: KeepSeparator = False

    def chunk(self, document: Document, context: ChunkContext) -> list[RawChunk]:
        splitter = self._splitter(
            CharacterTextSplitter,
            context,
            separator=self.separator,
            is_separator_regex=self.is_separator_regex,
            keep_separator=self.keep_separator,
        )
        return _split(splitter, document)


class LangchainToken(_Sized):
    """Split into fixed windows of `chunk_size` tokens, ignoring text structure."""

    strategy: Literal["token"]

    def chunk(self, document: Document, context: ChunkContext) -> list[RawChunk]:
        splitter = TokenTextSplitter(
            encoding_name=context.encoding, disallowed_special=(), **self._size_options()
        )
        return _split(splitter, document)


class LangchainNLTK(_Measured):
    """Split into sentences with NLTK, then merge them up to `chunk_size`.

    Sentences are re-joined with `separator`, so chunks have no offsets."""

    strategy: Literal["nltk"]
    separator: str = "\n\n"
    language: str = Field(default="english", description="NLTK Punkt language.")

    verbatim: ClassVar[bool] = False

    def chunk(self, document: Document, context: ChunkContext) -> list[RawChunk]:
        splitter = self._splitter(
            NLTKTextSplitter, context, separator=self.separator, language=self.language
        )
        return _split(splitter, document)


class LangchainJSFramework(_Measured):
    """Split React, Vue and Svelte code on component boundaries, then recursively."""

    strategy: Literal["js_framework"]
    separators: list[str] | None = Field(
        default=None, description="Extra separators, tried before the defaults."
    )

    def chunk(self, document: Document, context: ChunkContext) -> list[RawChunk]:
        splitter = self._splitter(JSFrameworkTextSplitter, context, separators=self.separators)
        return _split(splitter, document)


class _Sections(_LangchainChunker):
    """Splitters that cut a document into header-delimited sections."""

    max_header_level: int = Field(default=3, ge=1, le=6, description="Deepest header to split on.")
    chunk_size: int | None = Field(
        default=None, ge=1, description="Further split sections longer than this many tokens."
    )
    chunk_overlap: int = Field(default=0, ge=0, description="Token overlap for `chunk_size`.")

    verbatim: ClassVar[bool] = False  # sections are extracted and re-assembled

    @model_validator(mode="after")
    def _check_sizes(self) -> Self:
        _check_overlap(self.chunk_size, self.chunk_overlap)
        return self

    def _sections(self, document: Document) -> list[LangchainDocument]:
        raise NotImplementedError

    def chunk(self, document: Document, context: ChunkContext) -> list[RawChunk]:
        sections = self._sections(document)
        if self.chunk_size is not None:
            sections = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                encoding_name=context.encoding,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                disallowed_special=(),
            ).split_documents(sections)
        return [
            RawChunk(text=section.page_content, metadata=_section_metadata(section.metadata))
            for section in sections
        ]


def _section_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    headers = {key: value for key, value in metadata.items() if _HEADER_KEY.fullmatch(key)}
    others = {key.lower(): value for key, value in metadata.items() if key not in headers}
    return {"headers": headers, **others}


def _markdown_headers(level: int) -> list[tuple[str, str]]:
    return [("#" * n, f"h{n}") for n in range(1, level + 1)]


def _html_headers(level: int) -> list[tuple[str, str]]:
    return [(f"h{n}", f"h{n}") for n in range(1, level + 1)]


class LangchainMarkdown(_Sections):
    """Split Markdown into sections by headers; each chunk carries its header hierarchy."""

    strategy: Literal["markdown"]
    strip_headers: bool = False
    return_each_line: bool = False

    def _sections(self, document: Document) -> list[LangchainDocument]:
        return MarkdownHeaderTextSplitter(
            headers_to_split_on=_markdown_headers(self.max_header_level),
            strip_headers=self.strip_headers,
            return_each_line=self.return_each_line,
        ).split_text(document.text)


class LangchainMarkdownSyntax(_Sections):
    """Split Markdown by headers while preserving whitespace, code blocks and rules;
    code blocks become their own chunks tagged with their language. Experimental in LangChain."""

    strategy: Literal["markdown_syntax"]
    strip_headers: bool = True
    return_each_line: bool = False

    def _sections(self, document: Document) -> list[LangchainDocument]:
        return ExperimentalMarkdownSyntaxTextSplitter(
            headers_to_split_on=_markdown_headers(self.max_header_level),
            strip_headers=self.strip_headers,
            return_each_line=self.return_each_line,
        ).split_text(document.text)


class LangchainHTMLHeader(_Sections):
    """Split HTML into text sections by heading tags; each chunk carries its headings."""

    strategy: Literal["html_header"]
    return_each_element: bool = False

    def _sections(self, document: Document) -> list[LangchainDocument]:
        return HTMLHeaderTextSplitter(
            headers_to_split_on=_html_headers(self.max_header_level),
            return_each_element=self.return_each_element,
        ).split_text(document.text)


class LangchainHTMLSection(_Sections):
    """Split HTML into sections, one per heading together with its content."""

    strategy: Literal["html_section"]

    def _sections(self, document: Document) -> list[LangchainDocument]:
        return HTMLSectionSplitter(
            headers_to_split_on=_html_headers(self.max_header_level)
        ).split_text(document.text)


class LangchainHTMLSemantic(_LangchainChunker):
    """Split HTML by headings while keeping tables, lists and chosen elements intact,
    optionally turning links and media into Markdown. Beta in LangChain."""

    strategy: Literal["html_semantic"]
    max_header_level: int = Field(default=3, ge=1, le=6)
    max_chunk_size: int = Field(default=1000, ge=1, description="In characters.")
    chunk_overlap: int = Field(default=0, ge=0, description="In characters.")
    separators: list[str] | None = None
    keep_separator: KeepSeparator = True
    elements_to_preserve: list[str] | None = Field(
        default=None, description="Tags never split, e.g. ['table', 'ul', 'ol']."
    )
    preserve_links: bool = False
    preserve_images: bool = False
    preserve_videos: bool = False
    preserve_audio: bool = False
    allowlist_tags: list[str] | None = None
    denylist_tags: list[str] | None = None
    normalize_text: bool = Field(default=False, description="Lowercase and strip punctuation.")
    stopword_removal: bool = False
    stopword_lang: str = "english"
    preserve_parent_metadata: bool = False

    verbatim: ClassVar[bool] = False

    @model_validator(mode="after")
    def _check_sizes(self) -> Self:
        _check_overlap(self.max_chunk_size, self.chunk_overlap)
        return self

    def chunk(self, document: Document, context: ChunkContext) -> list[RawChunk]:
        options = self.model_dump(exclude={"engine", "strategy", "max_header_level"})
        splitter = HTMLSemanticPreservingSplitter(
            headers_to_split_on=_html_headers(self.max_header_level), **options
        )
        return [
            RawChunk(text=section.page_content, metadata=_section_metadata(section.metadata))
            for section in splitter.split_text(document.text)
        ]


class LangchainJSON(_LangchainChunker):
    """Split a JSON document into smaller valid JSON documents, keeping nesting intact.

    The input text must be JSON."""

    strategy: Literal["json"]
    max_chunk_size: int = Field(default=2000, ge=1, description="In characters.")
    min_chunk_size: int | None = Field(default=None, ge=1, description="In characters.")
    convert_lists: bool = Field(default=False, description="Split inside lists too.")
    ensure_ascii: bool = False

    verbatim: ClassVar[bool] = False

    def chunk(self, document: Document, context: ChunkContext) -> list[RawChunk]:
        data = _load_json(document.text)
        if isinstance(data, list):
            data = {str(index): item for index, item in enumerate(data)}
        if not isinstance(data, dict):
            raise InvalidInputError("The json strategy needs a JSON object or array.")
        splitter = RecursiveJsonSplitter(
            max_chunk_size=self.max_chunk_size, min_chunk_size=self.min_chunk_size
        )
        texts = splitter.split_text(
            data, convert_lists=self.convert_lists, ensure_ascii=self.ensure_ascii
        )
        return [RawChunk(text=text) for text in texts]


def _load_json(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise InvalidInputError(f"The input is not valid JSON: {exc}") from exc


LangchainChunker = Annotated[
    LangchainRecursive
    | LangchainCharacter
    | LangchainToken
    | LangchainNLTK
    | LangchainJSFramework
    | LangchainMarkdown
    | LangchainMarkdownSyntax
    | LangchainHTMLHeader
    | LangchainHTMLSection
    | LangchainHTMLSemantic
    | LangchainJSON,
    Field(discriminator="strategy"),
]
