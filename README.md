# Chunker API

A FastAPI service for intelligent text chunking using LangChain and tiktoken.

## Features

- 📄 File upload and chunking
- 📝 Direct text chunking
- 🎯 Token-based splitting with configurable size
- 🔄 Overlap support for context preservation
- 🤖 Multiple model support for token counting

## Installation

Using `uv` (recommended):

```bash
# Install dependencies
uv pip install -e .

# Or install in development mode with dev dependencies
uv pip install -e ".[dev]"
```

## Usage

### Start the server

```bash
# Development mode (auto-reload)
python main.py

# Or using uvicorn directly
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

Interactive API docs: `http://localhost:8000/docs`

### API Endpoints

#### 1. Health Check

```bash
curl http://localhost:8000/
```

#### 2. Chunk File

Upload a file and split it into chunks:

```bash
curl -X POST "http://localhost:8000/chunk/file" \
  -F "file=@document.txt" \
  -F "chunk_size=800" \
  -F "chunk_overlap=200" \
  -F "model=gpt-4"
```

#### 3. Chunk Text

Send text directly for chunking:

```bash
curl -X POST "http://localhost:8000/chunk/text" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your long text here...",
    "chunk_size": 800,
    "chunk_overlap": 200,
    "model": "gpt-4"
  }'
```

### Parameters

- **chunk_size** (default: 800): Target size for each chunk in tokens
- **chunk_overlap** (default: 200): Number of overlapping tokens between chunks
- **model** (default: "gpt-4"): Model name for token counting (gpt-4, gpt-3.5-turbo, etc.)

### Response Format

```json
{
  "total_chunks": 5,
  "total_tokens": 3456,
  "chunks": [
    {
      "index": 1,
      "content": "Chunk text here...",
      "tokens": 723,
      "start_char": 0,
      "end_char": 2891
    }
  ],
  "metadata": {
    "filename": "document.txt",
    "content_type": "text/plain",
    "original_size": 12456,
    "chunk_size": 800,
    "chunk_overlap": 200,
    "model": "gpt-4"
  }
}
```

## Integration with Laravel

You can call this service from your Laravel application:

```php
use Illuminate\Support\Facades\Http;

$response = Http::attach(
    'file', file_get_contents($filePath), 'document.txt'
)->post('http://localhost:8000/chunk/file', [
    'chunk_size' => 800,
    'chunk_overlap' => 200,
    'model' => 'gpt-4'
]);

$chunks = $response->json('chunks');
```

## Development

### Run tests

```bash
pytest
```

### Code formatting

```bash
ruff check --fix .
```

## Next Steps

- [ ] Add PDF parsing support
- [ ] Add DOCX parsing support
- [ ] Add HTML cleaning and parsing
- [ ] Add semantic chunking strategies
- [ ] Add embedding-based chunking
- [ ] Add authentication
- [ ] Add rate limiting

