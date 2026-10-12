"""Chunkers backed by LlamaIndex node parsers (`llama-index-core`)."""

import json
from typing import Annotated, Any, ClassVar, Literal, Self

from llama_index.core import Document as LlamaDocument
from llama_index.core.base.embeddings.base import BaseEmbedding, Embedding
from llama_index.core.node_parser import (
    CodeSplitter,
    HierarchicalNodeParser,
    HTMLNodeParser,
    JSONNodeParser,
    MarkdownNodeParser,
    NodeParser,
    SemanticSplitterNodeParser,
    SentenceSplitter,
    SentenceWindowNodeParser,
    TokenTextSplitter,
)
from llama_index.core.node_parser.text.sentence import CHUNKING_REGEX
from llama_index.core.schema import NodeRelationship, TextNode
from pydantic import Field, PrivateAttr, model_validator

from app.embeddings import Embedder
from app.engines import grammars
from app.engines.base import ChunkContext, ChunkerBase, Document, RawChunk
from app.errors import InvalidInputError
from app.tokens import get_encoding


class _EmbedderAdapter(BaseEmbedding):
    """Exposes the configured embedder as a LlamaIndex embedding model."""

    _embedder: Embedder = PrivateAttr()

    def __init__(self, embedder: Embedder, **kwargs: Any) -> None:
        super().__init__(model_name=embedder.model, **kwargs)
        self._embedder = embedder

    def _get_query_embedding(self, query: str) -> Embedding:
        return self._embedder.embed([query])[0]

    def _get_text_embedding(self, text: str) -> Embedding:
        return self._embedder.embed([text])[0]

    def _get_text_embeddings(self, texts: list[str]) -> list[Embedding]:
        return self._embedder.embed(texts)

    async def _aget_query_embedding(self, query: str) -> Embedding:
        return (await self._embedder.aembed([query]))[0]


class _LlamaIndexChunker(ChunkerBase):
    engine: Literal["llamaindex"]

    def _node_parser(self, context: ChunkContext) -> NodeParser:
        raise NotImplementedError

    def _nodes(self, document: Document, context: ChunkContext) -> list[TextNode]:
        nodes = self._node_parser(context).get_nodes_from_documents(
            [LlamaDocument(text=document.text)]
        )
        return [node for node in nodes if isinstance(node, TextNode)]

    def chunk(self, document: Document, context: ChunkContext) -> list[RawChunk]:
        return [
            RawChunk(
                text=node.get_content(),
                start_index=node.start_char_idx,
                end_index=node.end_char_idx,
                metadata=dict(node.metadata),
            )
            for node in self._nodes(document, context)
        ]


class _SizedChunker(_LlamaIndexChunker):
    chunk_size: int = Field(default=512, ge=1, description="Maximum chunk length in tokens.")
    chunk_overlap: int = Field(default=64, ge=0, description="Token overlap between chunks.")

    @model_validator(mode="after")
    def _check_overlap(self) -> Self:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


class LlamaIndexSentence(_SizedChunker):
    """Split into chunks of whole sentences up to `chunk_size` tokens, preferring paragraph
    boundaries (`SentenceSplitter`)."""

    strategy: Literal["sentence"]
    separator: str = Field(default=" ", description="Word separator.")
    paragraph_separator: str = "\n\n\n"
    secondary_chunking_regex: str | None = Field(
        default=CHUNKING_REGEX,
        description="Fallback split for sentences longer than `chunk_size`.",
    )

    def _node_parser(self, context: ChunkContext) -> NodeParser:
        return SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separator=self.separator,
            paragraph_separator=self.paragraph_separator,
            secondary_chunking_regex=self.secondary_chunking_regex,
            tokenizer=get_encoding(context.encoding).encode,
            include_prev_next_rel=False,
        )


class LlamaIndexToken(_SizedChunker):
    """Split into chunks of `chunk_size` tokens on word boundaries (`TokenTextSplitter`)."""

    strategy: Literal["token"]
    separator: str = " "
    backup_separators: list[str] = Field(default_factory=lambda: ["\n"])

    def _node_parser(self, context: ChunkContext) -> NodeParser:
        return TokenTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separator=self.separator,
            backup_separators=self.backup_separators,
            tokenizer=get_encoding(context.encoding).encode,
            include_prev_next_rel=False,
        )


class LlamaIndexSemantic(_LlamaIndexChunker):
    """Start a new chunk where the embedding similarity between sentence groups drops
    (`SemanticSplitterNodeParser`). Needs embeddings."""

    strategy: Literal["semantic"]
    buffer_size: int = Field(
        default=1, ge=0, description="Neighbouring sentences grouped when comparing similarity."
    )
    breakpoint_percentile_threshold: int = Field(
        default=95, ge=1, le=99, description="Dissimilarity percentile that starts a new chunk."
    )

    requires_embeddings: ClassVar[bool] = True

    def _node_parser(self, context: ChunkContext) -> NodeParser:
        return SemanticSplitterNodeParser(
            embed_model=_EmbedderAdapter(context.require_embedder()),
            buffer_size=self.buffer_size,
            breakpoint_percentile_threshold=self.breakpoint_percentile_threshold,
            include_prev_next_rel=False,
        )


class LlamaIndexMarkdown(_LlamaIndexChunker):
    """Split Markdown by headers (`MarkdownNodeParser`); each chunk carries its `header_path`."""

    strategy: Literal["markdown"]
    header_path_separator: str = "/"

    def _node_parser(self, context: ChunkContext) -> NodeParser:
        return MarkdownNodeParser(
            header_path_separator=self.header_path_separator, include_prev_next_rel=False
        )


class LlamaIndexSentenceWindow(_LlamaIndexChunker):
    """One chunk per sentence; `metadata.window` holds the surrounding sentences, for
    sentence-window retrieval (`SentenceWindowNodeParser`)."""

    strategy: Literal["sentence_window"]
    window_size: int = Field(default=3, ge=0, description="Sentences on each side of the chunk.")

    def _node_parser(self, context: ChunkContext) -> NodeParser:
        return SentenceWindowNodeParser.from_defaults(
            window_size=self.window_size, include_prev_next_rel=False
        )


class LlamaIndexHierarchical(_LlamaIndexChunker):
    """Chunk at several sizes at once (largest first) for auto-merging retrieval. Each chunk
    has `metadata.node_id`, `parent_id` and `level` (0 = largest)."""

    strategy: Literal["hierarchical"]
    chunk_sizes: list[int] = Field(
        default_factory=lambda: [2048, 512, 128], min_length=1, description="In tokens."
    )
    chunk_overlap: int = Field(default=20, ge=0, description="Token overlap at every level.")

    @model_validator(mode="after")
    def _check_sizes(self) -> Self:
        if any(size < 1 for size in self.chunk_sizes):
            raise ValueError("chunk_sizes must be positive")
        if self.chunk_sizes != sorted(self.chunk_sizes, reverse=True):
            raise ValueError("chunk_sizes must be in decreasing order")
        if self.chunk_overlap >= min(self.chunk_sizes):
            raise ValueError("chunk_overlap must be smaller than every chunk size")
        return self

    def _node_parser(self, context: ChunkContext) -> NodeParser:
        tokenizer = get_encoding(context.encoding).encode
        ids = [f"level-{level}" for level in range(len(self.chunk_sizes))]
        parsers: dict[str, NodeParser] = {
            parser_id: SentenceSplitter(
                chunk_size=size,
                chunk_overlap=self.chunk_overlap,
                tokenizer=tokenizer,
                include_prev_next_rel=False,
            )
            for parser_id, size in zip(ids, self.chunk_sizes, strict=True)
        }
        return HierarchicalNodeParser.from_defaults(
            node_parser_ids=ids, node_parser_map=parsers, include_prev_next_rel=False
        )

    def chunk(self, document: Document, context: ChunkContext) -> list[RawChunk]:
        nodes = self._nodes(document, context)
        parents = {
            node.node_id: parent.node_id
            if (parent := node.relationships.get(NodeRelationship.PARENT)) is not None
            and not isinstance(parent, list)
            else None
            for node in nodes
        }

        def level(node_id: str) -> int:
            depth = 0
            while (parent_id := parents.get(node_id)) is not None:
                node_id, depth = parent_id, depth + 1
            return depth

        return [
            RawChunk(
                text=node.get_content(),
                start_index=node.start_char_idx,
                end_index=node.end_char_idx,
                metadata={
                    "node_id": node.node_id,
                    "parent_id": parents[node.node_id],
                    "level": level(node.node_id),
                },
            )
            for node in nodes
        ]


class LlamaIndexHTML(_LlamaIndexChunker):
    """One chunk per HTML element of the given `tags` (`HTMLNodeParser`); the input is HTML."""

    strategy: Literal["html"]
    tags: list[str] = Field(
        default_factory=lambda: [
            "p",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "li",
            "b",
            "i",
            "u",
            "section",
        ],
        min_length=1,
    )

    verbatim: ClassVar[bool] = False  # text is extracted from the markup

    def _node_parser(self, context: ChunkContext) -> NodeParser:
        return HTMLNodeParser(tags=self.tags, include_prev_next_rel=False)


class LlamaIndexJSON(_LlamaIndexChunker):
    """Flatten JSON into `path value` lines, one chunk per top-level item (`JSONNodeParser`);
    the input must be JSON."""

    strategy: Literal["json"]

    verbatim: ClassVar[bool] = False

    def _node_parser(self, context: ChunkContext) -> NodeParser:
        return JSONNodeParser(include_prev_next_rel=False)

    def chunk(self, document: Document, context: ChunkContext) -> list[RawChunk]:
        try:
            json.loads(document.text)
        except json.JSONDecodeError as exc:
            raise InvalidInputError(f"The input is not valid JSON: {exc}") from exc
        return super().chunk(document, context)


class LlamaIndexCode(_LlamaIndexChunker):
    """Split source code along its syntax tree (functions, classes, ...) with tree-sitter
    (`CodeSplitter`). `language` must be one of the server's CODE_LANGUAGES."""

    strategy: Literal["code"]
    language: str = Field(min_length=1, examples=["python"])
    chunk_lines: int = Field(default=40, ge=1)
    chunk_lines_overlap: int = Field(default=15, ge=1)
    max_chars: int = Field(default=1500, ge=1, description="Used when `count_mode` is char.")
    count_mode: Literal["char", "token"] = "char"
    max_tokens: int = Field(default=512, ge=1, description="Used when `count_mode` is token.")

    @model_validator(mode="after")
    def _check_overlap(self) -> Self:
        if self.chunk_lines_overlap >= self.chunk_lines:
            raise ValueError("chunk_lines_overlap must be smaller than chunk_lines")
        return self

    def preflight(self, *, embeddings: bool) -> None:
        super().preflight(embeddings=embeddings)
        grammars.require(self.language)

    def _node_parser(self, context: ChunkContext) -> NodeParser:
        grammars.require(self.language)
        return CodeSplitter(
            language=self.language,
            chunk_lines=self.chunk_lines,
            chunk_lines_overlap=self.chunk_lines_overlap,
            max_chars=self.max_chars,
            count_mode=self.count_mode,
            max_tokens=self.max_tokens,
            tokenizer=get_encoding(context.encoding).encode,
            include_prev_next_rel=False,
        )


LlamaIndexChunker = Annotated[
    LlamaIndexSentence
    | LlamaIndexToken
    | LlamaIndexSemantic
    | LlamaIndexMarkdown
    | LlamaIndexSentenceWindow
    | LlamaIndexHierarchical
    | LlamaIndexHTML
    | LlamaIndexJSON
    | LlamaIndexCode,
    Field(discriminator="strategy"),
]
