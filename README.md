# Chunker

A stateless HTTP API that puts the RAG toolkits of **LangChain**, **LlamaIndex**, **Unstructured**
and **Chonkie** behind one JSON interface, so any language (PHP, JS, Go, ...) can parse documents,
chunk text and compute embeddings without embedding a Python library.

- **Stateless:** no database, no storage. Uploads are processed in memory (or short-lived temp files) and discarded.
- **Pluggable engines:** pick the parser and chunking strategy per request.
- **Semantic chunking** with any OpenAI-compatible embeddings API (OpenAI, Mistral, Ollama, vLLM, TEI, ...).
- **Async jobs:** pass `webhook_url` and get the result POSTed back, signed per [Standard Webhooks](https://standardwebhooks.com).
- **Typed:** every option is validated; the OpenAPI schema at `/docs` can generate clients.

## Engines

| Engine       | Parses files                                                                 | Chunking strategies                             |
| ------------ | ---------------------------------------------------------------------------- | ----------------------------------------------- |
| Unstructured | pdf, docx, pptx, xlsx, csv, tsv, md, txt, html, htm, xml, eml, msg           | `basic`, `by_title`                             |
| LlamaIndex   | pdf, docx, pptx, xlsx, xls, csv, md, txt, ipynb                              | `sentence`, `token`, `semantic`\*, `markdown`   |
| LangChain    | –                                                                            | `recursive`, `character`, `token`, `markdown`   |
| Chonkie      | –                                                                            | `token`, `sentence`, `recursive`, `semantic`\*  |

\* Needs `EMBEDDINGS_MODEL`. LangChain's document loaders live in `langchain-community`, which LangChain
has sunset, so LangChain is used for splitting only. `GET /v1/engines` returns every strategy with
the JSON Schema of its options.

## Quick start

Requires [uv](https://docs.astral.sh/uv/). Python 3.13 is installed automatically
(`unstructured` does not support 3.14 yet).

```bash
uv sync
cp .env.example .env   # optional: embeddings, API key, webhook secret
uv run chunker serve --reload
```

Open http://localhost:8000/docs for the interactive API reference.

## API

| Method | Path             | Purpose                                                         |
| ------ | ---------------- | --------------------------------------------------------------- |
| GET    | `/health`        | Liveness check (no auth)                                        |
| GET    | `/v1/engines`    | Parsers, chunking strategies and their option schemas          |
| POST   | `/v1/parse`      | File → text + elements (with page numbers)                     |
| POST   | `/v1/chunk`      | Text → chunks                                                   |
| POST   | `/v1/chunk/file` | File → parse → chunks                                           |
| POST   | `/v1/embed`      | Texts → embeddings with the configured model                    |

### Chunk text

```bash
curl -s localhost:8000/v1/chunk -H 'content-type: application/json' -d '{
  "text": "Your long text...",
  "chunker": {"engine": "llamaindex", "strategy": "sentence", "chunk_size": 512, "chunk_overlap": 64},
  "embed": false
}'
```

The default chunker is `{"engine": "langchain", "strategy": "recursive"}` (512 tokens, 64 overlap).
Token counts use `encoding` (`cl100k_base` by default; also `o200k_base`, `p50k_base`, `r50k_base`).

### Chunk a file

Multipart form; `parser`, `chunker` and `metadata` are JSON strings.

```bash
curl -s localhost:8000/v1/chunk/file \
  -F file=@report.pdf \
  -F 'parser={"engine": "unstructured", "strategy": "fast"}' \
  -F 'chunker={"engine": "chonkie", "strategy": "semantic", "threshold": 0.75}' \
  -F embed=true
```

Defaults: Unstructured parser (`fast`) and Unstructured `by_title` chunking. The `fast` strategy reads
embedded text. Scanned documents need `hi_res` or `ocr_only`, which require `poppler` and `tesseract`
on the server (e.g. `brew install poppler tesseract` or `apt install poppler-utils tesseract-ocr`);
`hi_res` also downloads its layout model from Hugging Face on first use.

### Response

```json
{
  "parser": "unstructured",
  "chunker": {"engine": "chonkie", "strategy": "semantic"},
  "document": {"filename": "report.pdf", "content_type": "application/pdf", "characters": 18234, "token_count": 4102},
  "total_chunks": 12,
  "chunks": [
    {
      "index": 0,
      "text": "Chunk text...",
      "token_count": 341,
      "start_index": 0,
      "end_index": 1520,
      "page_numbers": [1, 2],
      "metadata": {},
      "embedding": [0.0123, -0.0456]
    }
  ],
  "embedding_model": "text-embedding-3-small",
  "metadata": {}
}
```

`start_index`/`end_index` are offsets into the parsed text. They are `null` for strategies that
rewrite whitespace (Unstructured, Markdown header splitting). `page_numbers` is filled for paginated
files.

## Asynchronous jobs and webhooks

Large files or semantic chunking can take a while. Add `webhook_url` (and set `WEBHOOK_SECRET`) to
get `202 {"job_id": "..."}` immediately; the result is POSTed when done:

```json
{"type": "chunking.completed", "timestamp": "2026-10-12T09:00:00.123456Z", "data": {"job_id": "...", "chunks": [], "metadata": {}}}
```

On failure the event is `chunking.failed` with `data.error = {"status_code": 422, "message": "..."}`.
Pass your own IDs in `metadata`; they are echoed back. Failed deliveries (network errors, 408/425/429, 5xx) are
retried with exponential backoff. Jobs run in-process, so a restart drops in-flight jobs.

Each delivery carries `webhook-id`, `webhook-timestamp` and `webhook-signature` headers. Verify them with
any [Standard Webhooks library](https://github.com/standard-webhooks/standard-webhooks), or by hand, e.g. in Laravel:

```php
$secret = base64_decode(Str::after(config('services.chunker.webhook_secret'), 'whsec_'));
$signed = $request->header('webhook-id').'.'.$request->header('webhook-timestamp').'.'.$request->getContent();
$expected = 'v1,'.base64_encode(hash_hmac('sha256', $signed, $secret, true));

abort_unless(
    abs(time() - (int) $request->header('webhook-timestamp')) < 300
        && collect(explode(' ', $request->header('webhook-signature')))->contains(fn ($s) => hash_equals($expected, $s)),
    401
);
```

### Calling from Laravel

```php
$response = Http::withToken(config('services.chunker.key'))
    ->attach('file', Storage::get($path), $filename)
    ->post(config('services.chunker.url').'/v1/chunk/file', [
        'chunker' => json_encode(['engine' => 'llamaindex', 'strategy' => 'semantic']),
        'embed' => 'true',
        'webhook_url' => route('chunker.webhook'),
        'metadata' => json_encode(['source_file_id' => $file->id]),
    ]);

$jobId = $response->json('job_id');
```

## Configuration

Environment variables (or `.env`); see [.env.example](.env.example).

| Variable                | Default                     | Description                                              |
| ----------------------- | --------------------------- | -------------------------------------------------------- |
| `API_KEY`               | –                           | Require `Authorization: Bearer <key>` on `/v1/*`         |
| `MAX_UPLOAD_MB`         | `50`                        | Upload size limit                                        |
| `EMBEDDINGS_BASE_URL`   | `https://api.openai.com/v1` | Any OpenAI-compatible embeddings API                     |
| `EMBEDDINGS_API_KEY`    | –                           | Optional for local servers such as Ollama                |
| `EMBEDDINGS_MODEL`      | –                           | Enables semantic chunking, `embed` and `/v1/embed`       |
| `EMBEDDINGS_DIMENSIONS` | –                           | Request shorter vectors (if the model supports it)       |
| `WEBHOOK_SECRET`        | –                           | `whsec_<base64>`; enables `webhook_url`                  |
| `WEBHOOK_VERIFY_TLS`    | `true`                      | Set `false` for local receivers with self-signed certs   |
| `HOST` / `PORT`         | `127.0.0.1` / `8000`        | Defaults for `chunker serve`                             |
| `LOG_LEVEL`             | `INFO`                      |                                                          |

## Deployment

```bash
uv sync --locked --no-dev
uv run chunker warmup                       # pre-download tokenizer files
uv run chunker serve --host 0.0.0.0 --port 8000 --workers 4
```

Run it behind a reverse proxy (TLS) under a process manager such as systemd or supervisor. On Linux,
PyTorch (required by `unstructured[pdf]`) is installed from the CPU-only index to keep the install small.

## Development

```bash
uv run pytest                  # tests
uv run ruff format .           # format
uv run ruff check .            # lint
uv run mypy app tests          # type check (strict)
```

## Releases

CI runs on every push and pull request. Pushing a `v*.*.*` tag runs CI, checks the tag matches
`pyproject.toml`, builds the sdist and wheel, attests their provenance and publishes a GitHub Release
with generated notes (tags with a suffix such as `v1.0.0-rc.1` become pre-releases).

```bash
uv version --bump minor        # e.g. 0.2.0 -> 0.3.0
git commit -am "Release v$(uv version --short)"
git tag -a "v$(uv version --short)" -m "v$(uv version --short)"
git push --follow-tags
```
