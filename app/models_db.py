"""Database models for the Chunker API."""
from sqlalchemy import Column, Integer, String, Text, DateTime, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base
from app.database import Base


class ChunkerChunk(Base):
    """Model for chunker_chunks table."""
    __tablename__ = "chunker_chunks"

    id = Column(Integer, primary_key=True)
    index_id = Column(Integer, nullable=False)
    checksum = Column(String, nullable=True)
    original_chunk = Column(Text, nullable=False)
    llm_chunk = Column(Text, nullable=False)
    webhook_id = Column(UUID(as_uuid=True), nullable=True)
    webhook_sent_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)

