"""Chunker: a stateless HTTP API for RAG text chunking and embeddings."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("chunker")
except PackageNotFoundError:  # pragma: no cover - only when running from an uninstalled checkout
    __version__ = "0.0.0"
