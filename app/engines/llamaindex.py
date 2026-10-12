"""Parser and chunkers backed by LlamaIndex (`llama-index-core`, `llama-index-readers-file`)."""

import tempfile
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal, Self

from llama_index.core import Document, SimpleDirectoryReader
from llama_index.core.base.embeddings.base import BaseEmbedding, Embedding
from llama_index.core.node_parser import (
    MarkdownNodeParser,
    NodeParser,
    SemanticSplitterNodeParser,
    SentenceSplitter,
    TokenTextSplitter,
)
from llama_index.core.schema import TextNode
from pydantic import Field, PrivateAttr, model_validator

from app.embeddings import Embedder
from app.engines.base import (
    ChunkContext,
    ChunkerBase,
    Element,
    ParsedDocument,
    ParserBase,
    RawChunk,
)
from app.tokens import get_encoding

# Reader metadata that leaks server paths or is meaningless for an uploaded file.
_DROPPED_METADATA = frozenset(
    {"file_path", "file_name", "creation_date", "last_modified_date", "last_accessed_date"}
)


class LlamaIndexParser(ParserBase):
    """Parse files with LlamaIndex's `SimpleDirectoryReader` (one element per page for PDFs)."""

    engine: Literal["llamaindex"]

    extensions: ClassVar[frozenset[str]] = frozenset(
        {".pdf", ".docx", ".pptx", ".xlsx", ".xls", ".csv", ".md", ".txt", ".ipynb"}
    )

    def parse(self, data: bytes, filename: str) -> ParsedDocument:
        suffix = Path(filename).suffix.lower()
        with tempfile.TemporaryDirectory(prefix="chunker-") as tmp:
            path = Path(tmp) / f"upload{suffix}"
            path.write_bytes(data)
            documents = SimpleDirectoryReader(input_files=[path], raise_on_error=True).load_data()

        elements = []
        for document in documents:
            metadata = {k: v for k, v in document.metadata.items() if k not in _DROPPED_METADATA}
            page_label = str(metadata.get("page_label", ""))
            elements.append(
                Element(
                    text=document.text,
                    type="Document",
                    page_number=int(page_label) if page_label.isdigit() else None,
                    metadata=metadata,
                )
            )
        return ParsedDocument(elements=elements, filename=filename)


class _EmbedderAdapter(BaseEmbedding):
    """Exposes the server-configured embedder as a LlamaIndex embedding model."""

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

    def chunk(self, document: ParsedDocument, context: ChunkContext) -> list[RawChunk]:
        nodes = self._node_parser(context).get_nodes_from_documents([Document(text=document.text)])
        return [
            RawChunk(
                text=node.get_content(),
                start_index=node.start_char_idx,
                end_index=node.end_char_idx,
                metadata=dict(node.metadata),
            )
            for node in nodes
            if isinstance(node, TextNode)
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
    """Split into chunks of whole sentences up to `chunk_size` tokens (`SentenceSplitter`)."""

    strategy: Literal["sentence"]

    def _node_parser(self, context: ChunkContext) -> NodeParser:
        return SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            tokenizer=get_encoding(context.encoding).encode,
            include_prev_next_rel=False,
        )


class LlamaIndexToken(_SizedChunker):
    """Split into chunks of `chunk_size` tokens on word boundaries (`TokenTextSplitter`)."""

    strategy: Literal["token"]

    def _node_parser(self, context: ChunkContext) -> NodeParser:
        return TokenTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            tokenizer=get_encoding(context.encoding).encode,
            include_prev_next_rel=False,
        )


class LlamaIndexSemantic(_LlamaIndexChunker):
    """Start a new chunk where the embedding similarity between sentence groups drops
    (`SemanticSplitterNodeParser`). Requires the server's embeddings model."""

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

    def _node_parser(self, context: ChunkContext) -> NodeParser:
        return MarkdownNodeParser(include_prev_next_rel=False)


LlamaIndexChunker = Annotated[
    LlamaIndexSentence | LlamaIndexToken | LlamaIndexSemantic | LlamaIndexMarkdown,
    Field(discriminator="strategy"),
]
