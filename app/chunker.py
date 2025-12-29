import asyncio
import hashlib
import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List
import tiktoken
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import SecretStr
from sqlalchemy import text
from app.models import Chunk, LlmChunk
from app.database import get_session
from app.webhook import send_chunk_webhook
from app.scaleway import embed, vision, describe_tables
from app.storage import upload_image_to_s3, get_s3_url

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_MODEL = "gpt-4"
DEFAULT_LLM_MODEL = "llama-3.3-70b-instruct"
DEFAULT_EMBEDDING_MODEL = "qwen3-embedding-8b"
EMBEDDING_DIMENSION = 1024

# Pricing constants (per million tokens)
PRICE_PER_INPUT_MIL = 0.90
PRICE_PER_OUTPUT_MIL = 0.90

# Processing constants
DEFAULT_BATCH_SIZE = 5
BATCH_DELAY_SECONDS = 0.5
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 1


def count_tokens(text: str, model: str = DEFAULT_MODEL) -> int:
    """Count tokens in text using tiktoken."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def create_chunks_from_ocr_text(ocr_response: dict, model: str = DEFAULT_MODEL) -> List[Chunk]:
    pages = ocr_response.get("pages", [])
    chunks = []
    
    for i, page in enumerate(pages):
        # Get current page content
        content = page.get("markdown", "")
        
        # Get prev page content (empty if first page)
        prev_content = ""
        if i > 0:
            prev_content = pages[i - 1].get("markdown", "")
        
        # Get next page content (empty if last page)
        next_content = ""
        if i < len(pages) - 1:
            next_content = pages[i + 1].get("markdown", "")
        
        # Extract image base64 strings (images are objects with 'image_base64' field)
        images = []
        for image in page.get("images", []):
            if isinstance(image, dict) and "image_base64" in image:
                images.append(image["image_base64"])
        
        # Extract table contents (tables are objects with 'content' field)
        tables = []
        for table in page.get("tables", []):
            if isinstance(table, dict) and "content" in table:
                tables.append(table["content"])
        
        # Create chunk
        chunk = Chunk(
            index=page.get("index", i),
            content=content,
            tokens=count_tokens(content, model),
            images=images,
            tables=tables,
            prev_content=prev_content,
            next_content=next_content
        )
        
        chunks.append(chunk)

    return chunks

def _parse_xml_response(response: str) -> dict:
    """Parse XML response to extract Chunk, Summary, and Questions."""
    result = {
        "chunk": "",
        "summary": "",
        "questions": []
    }
    
    # Extract Chunk content
    chunk_match = re.search(r'<Chunk>(.*?)</Chunk>', response, re.DOTALL)
    if chunk_match:
        result["chunk"] = chunk_match.group(1).strip()
    
    # Extract Summary content
    summary_match = re.search(r'<Summary>(.*?)</Summary>', response, re.DOTALL)
    if summary_match:
        result["summary"] = summary_match.group(1).strip()
    
    # Extract Questions content
    questions_match = re.search(r'<Questions>(.*?)</Questions>', response, re.DOTALL)
    if questions_match:
        questions_text = questions_match.group(1).strip()
        # Split by newlines and filter empty lines
        result["questions"] = [q.strip() for q in questions_text.split('\n') if q.strip()]
    
    return result

def calculate_cost_in_cents(
    input_tokens: int,
    output_tokens: int,
    price_per_input_mil: float = PRICE_PER_INPUT_MIL,
    price_per_output_mil: float = PRICE_PER_OUTPUT_MIL
) -> float:
    """
    Calculate cost in cents based on token counts and pricing.

    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        price_per_input_mil: Price per million input tokens
        price_per_output_mil: Price per million output tokens

    Returns:
        Cost in cents (rounded to 2 decimal places)
    """
    input_rate = price_per_input_mil * 100 / 1_000_000
    output_rate = price_per_output_mil * 100 / 1_000_000

    input_cost = input_tokens * input_rate
    output_cost = output_tokens * output_rate

    return round(input_cost + output_cost, 2)

def _load_prompts() -> tuple[str, str]:
    """Load system and user prompts from files."""
    system_prompt_path = Path(__file__).parent.parent / "prompts" / "chunking" / "system.md"
    if not system_prompt_path.exists():
        raise FileNotFoundError(f"System prompt not found at: {system_prompt_path}")
    
    with open(system_prompt_path, 'r', encoding='utf-8') as f:
        system_prompt = f.read()
    
    user_prompt_path = Path(__file__).parent.parent / "prompts" / "chunking" / "user.md"
    if not user_prompt_path.exists():
        raise FileNotFoundError(f"User prompt template not found at: {user_prompt_path}")
    
    with open(user_prompt_path, 'r', encoding='utf-8') as f:
        user_prompt_template = f.read()
    
    return system_prompt, user_prompt_template


def _setup_llm_chain(model: str, system_prompt: str, user_prompt_template: str):
    """Set up LangChain chain with Scaleway LLM."""
    api_key = os.environ.get("SCALEWAYAI_API_KEY")
    base_url = os.environ.get("SCALEWAYAI_ENDPOINT", "https://api.scaleway.ai/ce82c941-d642-4fe0-94d7-b89bee83dd9d/v1")
    
    llm = ChatOpenAI(
        model=model,
        api_key=SecretStr(api_key),
        base_url=base_url,
        temperature=0,
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", user_prompt_template)
    ])
    
    return prompt | llm | StrOutputParser()


def _calculate_checksum(content: str) -> str:
    """Calculate MD5 checksum of content (matching Laravel implementation)."""
    return hashlib.md5(content.encode('utf-8')).hexdigest()


async def _check_existing_chunk(
    checksum: str,
    chunkable_type: str,
    chunkable_id: int
) -> str | None:
    """Check if chunk already exists in database. Returns chunk content if found, None otherwise."""
    async with get_session() as session:
        try:
            check_query = text("""
                SELECT chunk FROM chunks 
                WHERE checksum = :checksum 
                AND chunkable_type = :chunkable_type
                AND chunkable_id = :chunkable_id
                LIMIT 1
            """)
            result = await session.execute(check_query, {
                "checksum": checksum,
                "chunkable_type": chunkable_type,
                "chunkable_id": chunkable_id,
            })
            existing_row = result.fetchone()
            return existing_row[0] if existing_row else None
        except Exception as e:
            logger.error(f"Failed to check for existing chunk: {e}", exc_info=True)
            return None


def _calculate_chunk_costs(
    system_prompt: str,
    user_prompt_template: str,
    input_dict: dict,
    response: str,
    enhanced_content: str
) -> float:
    """Calculate total cost in cents for chunker and embedding."""
    # Chunker input cost (system prompt + user prompt)
    user_prompt = user_prompt_template.format(
        text=input_dict["text"],
        previous_context=input_dict["previous_context"],
        next_context=input_dict["next_context"]
    )
    chunker_input_tokens = count_tokens(system_prompt + user_prompt)
    chunker_output_tokens = count_tokens(response)

    chunker_cost = calculate_cost_in_cents(
        input_tokens=chunker_input_tokens,
        output_tokens=chunker_output_tokens
    )

    # Embedding cost
    embedding_tokens = count_tokens(enhanced_content)
    embedding_cost = calculate_cost_in_cents(
        input_tokens=embedding_tokens,
        output_tokens=0
    )

    return chunker_cost + embedding_cost


async def _prepare_embedding(content: str) -> str:
    """Generate and format embedding vector for PostgreSQL."""
    embedding_vector = await embed(content)
    return "[" + ",".join(str(x) for x in embedding_vector) + "]"


def _prepare_metadata(
    described_images: dict[str, str],
    tables_markdown: list[str],
    tables_descriptions: str
) -> dict:
    """
    Prepare metadata dictionary for chunk.

    Args:
        described_images: Dictionary mapping S3 image paths to descriptions
        tables_markdown: List of table markdown strings
        tables_descriptions: Natural language description of tables
    """
    return {
        "images": described_images,
        "tables": {
            "markdown": tables_markdown,
            "descriptions": tables_descriptions,
        },
    }


async def _process_images_and_tables(
    chunk: Chunk, user_id: int, agent_id: int, file_id: int
) -> tuple[dict[str, str], str]:
    described_images = {}

    # Process images: upload to S3 and get descriptions
    if chunk.images:
        for image_base64 in chunk.images:
            try:
                # Generate unique filename
                unique_id = uuid.uuid4().hex[:12]
                filename = f"{unique_id}.jpg"
                
                # Build S3 path: users/{id}/agents/{id}/files/{id}/{filename}
                s3_path = f"users/{user_id}/agents/{agent_id}/files/{file_id}/{filename}"
                
                # Upload image to S3
                s3_key = upload_image_to_s3(image_base64, s3_path)
                s3_url = get_s3_url(s3_key)

                # Get image description using vision
                image_description = await vision(image_base64)

                if image_description:
                    described_images[s3_url] = image_description
            except Exception as e:
                logger.warning(f"Failed to process image: {e}")
                # Continue with next image if one fails
                continue

    # Process tables: get descriptions
    tables_description = ""
    if chunk.tables:
        try:
            tables_description = await describe_tables(chunk.tables)
        except Exception as e:
            logger.warning(f"Failed to describe tables: {e}")

    return described_images, tables_description


async def _save_chunk_to_database(
    chunk: Chunk,
    enhanced_content: str,
    summary: str,
    questions: list,
    embedding_str: str,
    metadata: dict,
    chunk_checksum: str,
    total_cost_in_cents: float,
    user_id: int,
    chunkable_type: str,
    chunkable_id: int,
    model: str
) -> int | None:
    """Save chunk to database and return the inserted chunk ID."""
    external_id = uuid.uuid4()
    webhook_uuid_obj = uuid.uuid4()
    webhook_uuid_str = str(webhook_uuid_obj)

    async with get_session() as session:
        try:
            # Use parameterized query for the BM25 tokenization
            insert_query = text("""
                INSERT INTO chunks
                (user_id, chunkable_type, chunkable_id, type, chunk_index, chunk,
                 original_chunk, questions, summary, chunked_using_model, embedding_model,
                 cost_in_cents, tokens, embedding, embedding_bm25, metadata, checksum,
                 retrieval_count, external_id, created_at, updated_at)
                VALUES (:user_id, :chunkable_type, :chunkable_id, :type, :chunk_index, :chunk,
                        :original_chunk, CAST(:questions AS json), :summary, :chunked_using_model, :embedding_model,
                        :cost_in_cents, :tokens, CAST(:embedding AS vector(1024)),
                        tokenizer_catalog.tokenize(:bm25_content, 'llmlingua2'),
                        CAST(:metadata AS jsonb), :checksum, :retrieval_count, :external_id, :created_at, :updated_at)
                RETURNING id
            """)

            result = await session.execute(insert_query, {
                "user_id": user_id,
                "chunkable_type": chunkable_type,
                "chunkable_id": chunkable_id,
                "type": "text",
                "chunk_index": chunk.index,
                "chunk": enhanced_content,
                "original_chunk": chunk.content,
                "questions": json.dumps(questions) if questions else None,
                "summary": summary if summary else None,
                "chunked_using_model": model,
                "embedding_model": DEFAULT_EMBEDDING_MODEL,
                "cost_in_cents": total_cost_in_cents,
                "tokens": count_tokens(enhanced_content),
                "embedding": embedding_str,
                "bm25_content": enhanced_content,
                "metadata": json.dumps(metadata),
                "checksum": chunk_checksum,
                "retrieval_count": 0,
                "external_id": external_id,
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
            })
            await session.commit()

            chunk_id = result.scalar()

            # Send webhook after successful persistence
            await send_chunk_webhook(webhook_uuid_str, chunk_id, chunk.index)

            return chunk_id
        except Exception as e:
            await session.rollback()
            logger.error(f"Failed to persist chunk {chunk.index}: {e}", exc_info=True)
            return None


async def chunk_using_llm(
    chunks: List[Chunk],
    user_id: int,
    agent_id: int,
    chunkable_type: str,
    chunkable_id: int,
    model: str = DEFAULT_LLM_MODEL,
    persist: bool = True,
    batch_size: int = DEFAULT_BATCH_SIZE
) -> List[LlmChunk]:
    if not chunks:
        return []

    # Load prompts and set up chain
    system_prompt, user_prompt_template = _load_prompts()
    chain = _setup_llm_chain(model, system_prompt, user_prompt_template)

    # Prepare inputs for parallel processing
    inputs = [
        {
            "text": chunk.content,
            "previous_context": chunk.prev_content,
            "next_context": chunk.next_content,
        }
        for chunk in chunks
    ]

    # Helper function to process a single chunk
    async def process_chunk(chunk: Chunk, input_dict: dict) -> LlmChunk:
        chunk_checksum = _calculate_checksum(chunk.content)
        
        # Check if chunk already exists in database
        if persist:
            existing_content = await _check_existing_chunk(chunk_checksum, chunkable_type, chunkable_id)
            if existing_content:
                return LlmChunk(
                    index=chunk.index,
                    content=existing_content,
                    tokens=count_tokens(existing_content),
                    images=chunk.images,
                    tables=chunk.tables,
                )
        
        # Process images and tables: upload to S3 and get descriptions
        described_images, tables_description = await _process_images_and_tables(
            chunk, user_id, agent_id, chunkable_id
        )
        
        # Build enhanced content with image descriptions and table descriptions
        content_parts = [chunk.content]
        
        # Add image descriptions
        if described_images:
            image_descriptions = "\n\n".join(described_images.values())
            content_parts.append(image_descriptions)
        
        # Add table descriptions
        if tables_description:
            content_parts.append(tables_description)
        
        # Update input_dict with enhanced content for chunking
        enhanced_input_dict = input_dict.copy()
        enhanced_input_dict["text"] = "\n\n".join(content_parts)
        
        # Process with LLM if not found in database
        response = await chain.ainvoke(enhanced_input_dict)
        parsed = _parse_xml_response(response)

        enhanced_content = parsed["chunk"] if parsed["chunk"] else enhanced_input_dict["text"]
        summary = parsed.get("summary", "")
        questions = parsed.get("questions", [])

        llm_chunk = LlmChunk(
            index=chunk.index,
            content=enhanced_content,
            tokens=count_tokens(enhanced_content),
            images=chunk.images,
            tables=chunk.tables,
        )

        # Persist to database if requested
        if persist:
            total_cost_in_cents = _calculate_chunk_costs(
                system_prompt, user_prompt_template, enhanced_input_dict, response, enhanced_content
            )
            embedding_str = await _prepare_embedding(enhanced_content)
            metadata = _prepare_metadata(
                described_images=described_images,
                tables_markdown=chunk.tables,
                tables_descriptions=tables_description
            )
            
            await _save_chunk_to_database(
                chunk=chunk,
                enhanced_content=enhanced_content,
                summary=summary,
                questions=questions,
                embedding_str=embedding_str,
                metadata=metadata,
                chunk_checksum=chunk_checksum,
                total_cost_in_cents=total_cost_in_cents,
                user_id=user_id,
                chunkable_type=chunkable_type,
                chunkable_id=chunkable_id,
                model=model
            )

        return llm_chunk

    # Process chunks in batches to avoid rate limits
    llm_chunks = []
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        batch_inputs = inputs[i:i + batch_size]
        
        # Process batch in parallel
        tasks = [process_chunk(chunk, input_dict) for chunk, input_dict in zip(batch_chunks, batch_inputs)]
        batch_results = await asyncio.gather(*tasks)
        llm_chunks.extend(batch_results)
        
        # Small delay between batches to avoid hitting rate limits
        if i + batch_size < len(chunks):
            await asyncio.sleep(BATCH_DELAY_SECONDS)

    return llm_chunks