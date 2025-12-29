from fastapi import FastAPI, File, UploadFile, HTTPException, Form
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
import hashlib
import json
from pathlib import Path
import app.ocr as ocr
from app.chunker import create_chunks_from_ocr_text, chunk_using_llm
from app.scaleway import completions, embed
from app.models import TextChunkRequest, Chunk, ChunkResponse

CACHE_DIR = Path("var/cache")

app = FastAPI(
    title="Chunker API",
    description="Intelligent text chunking service using LangChain",
    version="0.1.0"
)


def calculate_checksum(content: bytes) -> str:
    """Calculate SHA256 checksum of file content."""
    return hashlib.sha256(content).hexdigest()


def get_cached_ocr(checksum: str) -> dict | None:
    """Retrieve cached OCR result if it exists."""
    cache_file = CACHE_DIR / f"{checksum}.json"
    if cache_file.exists():
        with open(cache_file, 'r') as f:
            return json.load(f)
    return None


def save_ocr_to_cache(checksum: str, ocr_result: dict) -> None:
    cache_file = CACHE_DIR / f"{checksum}.json"
    with open(cache_file, 'w') as f:
        json.dump(ocr_result, f, indent=2)


def count_tokens(text: str, model: str = "gpt-4") -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def extract_text_from_file(file_content: bytes) -> str:
    return file_content.decode('utf-8')

def split_text_into_chunks(text: str, chunk_size: int, chunk_overlap: int, model: str) -> tuple[list, int]:
    """Split text into chunks and return chunks with metadata."""

    # Count total tokens
    total_tokens = count_tokens(text, model)

    
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator=True
    )
    
    # Split text
    splits = text_splitter.split_text(text)
    
    # Create chunks with metadata
    chunks = []
    char_position = 0
    
    for idx, split in enumerate(splits, start=1):
        token_count = count_tokens(split, model)
        start_pos = text.find(split, char_position)
        end_pos = start_pos + len(split)
        
        chunks.append(Chunk(
            index=idx,
            content=split,
            tokens=token_count,
            start_char=start_pos,
            end_char=end_pos
        ))
        
        char_position = start_pos + len(split)
    
    return chunks, total_tokens


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "chunker", "version": "0.1.0"}


@app.post("/chunk/file")
async def chunk_file(
    file: UploadFile = File(...),
    user_id: int = Form(...),
    agent_id: int = Form(...),
    file_id: int = Form(...),
    # chunk_size: int = Form(default=800),
    # chunk_overlap: int = Form(default=200),
    # model: str = Form(default="gpt-4"),
):
    file_content = await file.read()
    checksum = calculate_checksum(file_content)
    
    # Check if result is cached
    oc_result = get_cached_ocr(checksum)
    if oc_result is None:
        await file.seek(0)

        # Upload file and process OCR
        file_ocr = await ocr.upload_file(file)
        signed_url = ocr.signed_url(file_ocr.id)
        ocr_response = ocr.ocr(signed_url.url)

        oc_result = ocr_response.model_dump()

        save_ocr_to_cache(checksum, oc_result)

    chunks = create_chunks_from_ocr_text(oc_result)

    return await chunk_using_llm(
        chunks,
        user_id=user_id,
        chunkable_type="App\\Models\\SourceFile",
        chunkable_id=file_id
    )


@app.post("/chunk/text", response_model=ChunkResponse)
async def chunk_text(request: TextChunkRequest):
    """
    Split text into chunks based on token count.
    
    Request body:
    - **text**: Text to be chunked
    - **chunk_size**: Target size for each chunk in tokens (default: 800)
    - **chunk_overlap**: Number of overlapping tokens between chunks (default: 200)
    - **model**: Model name for token counting (default: gpt-4)
    """
    
    # Validate overlap
    if request.chunk_overlap >= request.chunk_size:
        raise HTTPException(status_code=400, detail="chunk_overlap must be less than chunk_size")
    
    # Split text into chunks
    chunks, total_tokens = split_text_into_chunks(
        request.text, 
        request.chunk_size, 
        request.chunk_overlap, 
        request.model
    )
    
    return ChunkResponse(
        total_chunks=len(chunks),
        total_tokens=total_tokens,
        chunks=chunks,
        metadata={
            "original_size": len(request.text),
            "chunk_size": request.chunk_size,
            "chunk_overlap": request.chunk_overlap,
            "model": request.model
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
