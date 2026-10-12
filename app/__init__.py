"""Chunker: a stateless HTTP API for RAG document parsing, chunking and embeddings."""

import os
from importlib.metadata import PackageNotFoundError, version

# Unstructured sends anonymous usage analytics on import unless explicitly opted out.
os.environ.setdefault("SCARF_NO_ANALYTICS", "true")
os.environ.setdefault("DO_NOT_TRACK", "true")

try:
    __version__ = version("chunker")
except PackageNotFoundError:  # pragma: no cover - only when running from an uninstalled checkout
    __version__ = "0.0.0"
