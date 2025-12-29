"""Pydantic models for the Chunker API."""
from pydantic import BaseModel, Field
from typing import List


class TextChunkRequest(BaseModel):
    """Request model for text chunking."""
    text: str = Field(..., min_length=1, description="Text to be chunked")
    chunk_size: int = Field(default=800, ge=100, le=8000, description="Target chunk size in tokens")
    chunk_overlap: int = Field(default=200, ge=0, le=1000, description="Overlap between chunks")
    model: str = Field(default="gpt-4", description="Model to use for token counting")


class Chunk(BaseModel):
    index: int
    content: str
    tokens: int
    images: List[str] = []
    tables: List[str] = []
    prev_content: str = ""
    next_content: str = ""

class LlmChunk(BaseModel):
    index: int
    content: str
    tokens: int
    images: List[str] = []
    tables: List[str] = []

class ChunkResponse(BaseModel):
    """Response model containing chunks and metadata."""
    total_chunks: int
    total_tokens: int
    chunks: List[Chunk]
    metadata: dict

