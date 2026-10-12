<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo-dark.svg">
    <img src="docs/assets/logo-light.svg" alt="Chunker" width="340">
  </picture>
</p>

<p align="center">
  <strong>Chunk text for RAG over HTTP, from any language.</strong><br>
  LangChain, LlamaIndex and Chonkie behind one stateless JSON API.
</p>

<p align="center">
  <a href="https://github.com/chatvia/chunker/actions/workflows/ci.yml"><img src="https://github.com/chatvia/chunker/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI"></a>
  <a href="https://github.com/chatvia/chunker/releases"><img src="https://img.shields.io/github/v/release/chatvia/chunker?sort=semver&amp;display_name=tag" alt="Release"></a>
  <a href="https://github.com/chatvia/chunker/pkgs/container/chunker"><img src="https://img.shields.io/badge/ghcr.io-chatvia%2Fchunker-2496ED?logo=docker&amp;logoColor=white" alt="Docker image"></a>
  <img src="https://img.shields.io/badge/python-3.14-3776AB?logo=python&amp;logoColor=white" alt="Python 3.14">
  <a href="https://github.com/astral-sh/uv"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json" alt="uv"></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a>
  <img src="https://img.shields.io/badge/types-mypy%20strict-2A6DB2" alt="mypy strict">
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> ·
  <a href="#docker">Docker</a> ·
  <a href="#api">API</a> ·
  <a href="#webhooks">Webhooks</a> ·
  <a href="#configuration">Configuration</a> ·
  <a href="#development">Development</a>
</p>

---

Chunker exposes the text splitters of the Python RAG ecosystem as a small, typed HTTP API, so a
PHP, JavaScript or Go application can use them without embedding Python.

It does one thing: **chunking**. Documents are parsed elsewhere (Mistral OCR, Docling, the
Unstructured API, ...) and their text, or their pages, is sent to Chunker.

- **Stateless:** no database, no storage, no files.
- **Pluggable:** choose the engine and strategy per request.
- **Page-aware:** send pages and every chunk reports the pages it came from.
- **27 strategies** across three engines, each with its full set of options.
- **Semantic chunking** with any OpenAI-compatible embeddings API (OpenAI, Mistral, Ollama, vLLM,
  TEI, ...), configured on the server or sent with each request.
- **Async jobs:** pass `webhook_url` and the result is POSTed back, signed per [Standard Webhooks](https://standardwebhooks.com).
- **Typed:** every option is validated, and the OpenAPI schema at `/docs` can generate clients.
- **Small image:** under 200 MB to download, multi-arch (amd64/arm64), non-root, health-checked.

## Chunking strategies

Pick one per request with `{"engine": ..., "strategy": ..., ...options}`. `GET /v1/engines`
returns every strategy with a description and the JSON Schema of all its options.

**LangChain**

| Strategy          | Splits                                                                       |
| ----------------- | ---------------------------------------------------------------------------- |
| `recursive`       | On paragraphs, lines, sentences, words; code-aware with `language` (default)  |
| `character`       | On one separator, merged up to `chunk_size`                                    |
| `token`           | Into fixed token windows                                                       |
| `nltk`            | Into NLTK sentences, merged up to `chunk_size`                                 |
| `js_framework`    | React, Vue and Svelte code on component boundaries                             |
| `markdown`        | Markdown by headers; chunks carry their header hierarchy                       |
| `markdown_syntax` | Markdown by headers, keeping code blocks and formatting intact                  |
| `html_header`     | HTML into text sections by heading tags                                         |
| `html_section`    | HTML into sections, one per heading with its content                            |
| `html_semantic`   | HTML by headings, keeping tables, lists, links and media intact                  |
| `json`            | A JSON document into smaller valid JSON documents                               |

**LlamaIndex**

| Strategy          | Splits                                                                       |
| ----------------- | ---------------------------------------------------------------------------- |
| `sentence`        | Into whole sentences up to `chunk_size` tokens                                  |
| `token`           | Into token windows on word boundaries                                           |
| `semantic`\*      | Where the embedding similarity between sentence groups drops                    |
| `markdown`        | Markdown by headers; chunks carry their `header_path`                           |
| `sentence_window` | One sentence per chunk, with the surrounding sentences in `metadata.window`      |
| `hierarchical`    | At several sizes at once, linked by `node_id`/`parent_id`, for auto-merging      |
| `html`            | HTML into one chunk per element of the chosen `tags`                            |
| `json`            | JSON into flattened `path value` lines                                           |
| `code`†           | Source code along its syntax tree (tree-sitter)                                 |

**Chonkie**

| Strategy          | Splits                                                                       |
| ----------------- | ---------------------------------------------------------------------------- |
| `token`           | Into token windows; overlap in tokens or as a fraction                          |
| `sentence`        | Into whole sentences up to `chunk_size` tokens                                  |
| `recursive`       | Recursively, with your own delimiter `levels` if you like                       |
| `semantic`\*      | Into groups of similar sentences, with a smoothing filter                       |
| `fast`            | At delimiter bytes, very fast                                                   |
| `table`           | A Markdown table into smaller tables, repeating the header                      |
| `code`†           | Source code along its syntax tree (tree-sitter)                                 |

\* Needs embeddings: configured on the server or sent with the request (see
[Embeddings](#embeddings)).
† `language` must be one of `CODE_LANGUAGES` (24 popular languages by default; their grammars are
installed by `chunker warmup` and baked into the Docker image).

The input is plain text or Markdown for most strategies, HTML for the `html*` strategies, JSON for
`json`, a Markdown table for `table`, and source code for `code` and `js_framework`.

## Quick start

With Docker:

```bash
docker run --rm -p 8000:8000 ghcr.io/chatvia/chunker:latest
```

Or from source, with [uv](https://docs.astral.sh/uv/) (Python 3.14 is installed automatically):

```bash
uv sync
cp .env.example .env   # optional: embeddings, API key, webhook secret
uv run chunker serve --reload
```

Then open http://localhost:8000/docs for the interactive API reference.

## Docker

Images are published to the GitHub Container Registry for `linux/amd64` and `linux/arm64` on every
release.

| Tag                      | Points to                                    |
| ------------------------ | -------------------------------------------- |
| `latest`                 | The newest stable release                    |
| `0.2.0`                  | An exact release                             |
| `0.2`                    | The newest patch release of a minor version  |
| `1` (from 1.0.0 onwards) | The newest release of a major version        |

### Configuration through environment variables

Every [setting](#configuration) is an environment variable, so models, API keys and secrets are
passed at run time and never baked into the image:

```bash
# Individual variables
docker run -d -p 8000:8000 \
  -e API_KEY=change-me \
  -e EMBEDDINGS_BASE_URL=https://api.openai.com/v1 \
  -e EMBEDDINGS_API_KEY=sk-... \
  -e EMBEDDINGS_MODEL=text-embedding-3-small \
  ghcr.io/chatvia/chunker:latest

# Or a whole .env file (see .env.example)
docker run -d -p 8000:8000 --env-file .env ghcr.io/chatvia/chunker:latest
```

With Docker Compose, [`compose.yaml`](compose.yaml) loads `.env` automatically when it exists:

```bash
docker compose up -d            # published image
docker compose up -d --build    # build from this checkout
```

To use an embeddings server running on the Docker host (e.g. Ollama), point
`EMBEDDINGS_BASE_URL` at `http://host.docker.internal:11434/v1` (on Linux, add
`--add-host=host.docker.internal:host-gateway`).

The image runs as an unprivileged user (UID `10001`), exposes port `8000` with a `/health` check,
and starts `WORKERS` server processes (default `1`). Every release image has a signed build
provenance attestation:
`gh attestation verify oci://ghcr.io/chatvia/chunker:latest --repo chatvia/chunker`.

## API

| Method | Path          | Purpose                                                |
| ------ | ------------- | ------------------------------------------------------ |
| GET    | `/health`     | Liveness check (no auth)                               |
| GET    | `/v1/engines` | Chunking strategies and their option schemas           |
| POST   | `/v1/chunk`   | Text or pages → chunks (optionally with embeddings)    |
| POST   | `/v1/embed`   | Texts → embeddings                                     |
| POST   | `/v1/tokens`  | Texts → token counts for an `encoding`                 |

When `API_KEY` is set, `/v1` endpoints require `Authorization: Bearer <API_KEY>`.

### Chunk text

```bash
curl -s localhost:8000/v1/chunk -H 'content-type: application/json' -d '{
  "text": "Your long text...",
  "chunker": {"engine": "llamaindex", "strategy": "sentence", "chunk_size": 512, "chunk_overlap": 64}
}'
```

The default chunker is `{"engine": "langchain", "strategy": "recursive"}` (512 tokens, 64 overlap).
Token counts use `encoding` (`cl100k_base` by default; also `o200k_base`, `p50k_base`, `r50k_base`).
Set `"embed": true` to get an embedding for every chunk.

### Chunk pages from a parser or OCR service

Send `pages` instead of `text` to keep track of where each chunk comes from. `page_number` is
optional and defaults to the page's position. With [Mistral OCR](https://docs.mistral.ai/capabilities/document_ai/basic_ocr/),
map each page's `markdown` and its zero-based `index`:

```bash
curl -s localhost:8000/v1/chunk -H 'content-type: application/json' -d '{
  "pages": [
    {"text": "# Annual report\n\nRevenue grew 12%...", "page_number": 1},
    {"text": "## Outlook\n\nWe expect...", "page_number": 2}
  ],
  "chunker": {"engine": "chonkie", "strategy": "semantic", "threshold": 0.75},
  "embed": true
}'
```

### Embeddings

Semantic strategies and `"embed": true` need an OpenAI-compatible embeddings API. Configure it on
the server with `EMBEDDINGS_BASE_URL`, `EMBEDDINGS_API_KEY` and `EMBEDDINGS_MODEL`. When the server
has no `EMBEDDINGS_MODEL`, each request can bring its own provider instead:

```bash
curl -s localhost:8000/v1/chunk -H 'content-type: application/json' -d '{
  "text": "Your long text...",
  "chunker": {"engine": "llamaindex", "strategy": "semantic", "breakpoint_percentile_threshold": 90},
  "embed": true,
  "embeddings": {
    "base_url": "https://api.mistral.ai/v1",
    "api_key": "...",
    "model": "mistral-embed",
    "dimensions": null,
    "batch_size": 64
  }
}'
```

The server configuration always wins: a request that sends `embeddings` to a server with
`EMBEDDINGS_MODEL` set gets a 400, so the server's key is never sent to a client-chosen URL.
Set `ALLOW_REQUEST_EMBEDDINGS=false` to forbid per-request providers. `/v1/embed` accepts the same
`embeddings` object, and `GET /v1/engines` reports which mode the server is in.

### Count tokens

```bash
curl -s localhost:8000/v1/tokens -H 'content-type: application/json' \
  -d '{"input": ["Hello world", "Another text"], "encoding": "o200k_base"}'
# {"encoding": "o200k_base", "data": [{"index": 0, "tokens": 2}, {"index": 1, "tokens": 2}]}
```

### Response

```json
{
  "chunker": {"engine": "chonkie", "strategy": "semantic"},
  "document": {"characters": 18234, "token_count": 4102, "pages": 2},
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

`start_index`/`end_index` are offsets into the input text (pages are joined with a blank line).
They are `null` when a strategy rewrites the text (Markdown, HTML and JSON strategies, `nltk`,
`table`). `page_numbers` is empty when the input is `text`. `metadata` holds strategy-specific
data such as `headers`, `window` or `node_id`/`parent_id`/`level`.

## Webhooks

Long documents or semantic chunking can take a while. Add `webhook_url` (and set `WEBHOOK_SECRET`)
to get `202 {"job_id": "..."}` immediately; the result is POSTed when done:

```json
{"type": "chunking.completed", "timestamp": "2026-10-12T09:00:00.123456Z", "data": {"job_id": "...", "chunks": [], "metadata": {}}}
```

On failure the event is `chunking.failed` with `data.error = {"status_code": 502, "message": "..."}`.
Pass your own IDs in `metadata`; they are echoed back. Deliveries that fail with a network error,
408, 425, 429 or 5xx are retried with exponential backoff. Jobs run in-process, so a restart drops
in-flight jobs.

Each delivery carries `webhook-id`, `webhook-timestamp` and `webhook-signature` headers. Verify them
with any [Standard Webhooks library](https://github.com/standard-webhooks/standard-webhooks), or by
hand, e.g. in Laravel:

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
// $ocr is the response of Mistral OCR (or any parser that returns pages).
$pages = collect($ocr['pages'])->map(fn (array $page) => [
    'text' => $page['markdown'],
    'page_number' => $page['index'] + 1,
]);

$response = Http::withToken(config('services.chunker.key'))
    ->post(config('services.chunker.url').'/v1/chunk', [
        'pages' => $pages->all(),
        'chunker' => ['engine' => 'llamaindex', 'strategy' => 'semantic'],
        'embed' => true,
        'webhook_url' => route('chunker.webhook'),
        'metadata' => ['source_file_id' => $file->id],
    ]);

$jobId = $response->json('job_id');
```

## Configuration

Set environment variables or use a `.env` file; see [.env.example](.env.example).

| Variable                | Default                     | Description                                             |
| ----------------------- | --------------------------- | ------------------------------------------------------- |
| `API_KEY`               | –                           | Require `Authorization: Bearer <key>` on `/v1/*`        |
| `MAX_REQUEST_MB`        | `50`                        | Maximum request body size                               |
| `EMBEDDINGS_BASE_URL`   | `https://api.openai.com/v1` | Any OpenAI-compatible embeddings API                    |
| `EMBEDDINGS_API_KEY`    | –                           | Optional for local servers such as Ollama               |
| `EMBEDDINGS_MODEL`      | –                           | Enables semantic chunking, `embed` and `/v1/embed`      |
| `EMBEDDINGS_DIMENSIONS` | –                           | Request shorter vectors (if the model supports it)      |
| `ALLOW_REQUEST_EMBEDDINGS` | `true`                   | Accept `embeddings` in requests when no model is set    |
| `CODE_LANGUAGES`        | 24 popular languages        | Grammars for the `code` strategies (comma-separated)    |
| `WEBHOOK_SECRET`        | –                           | `whsec_<base64>`; enables `webhook_url`                 |
| `WEBHOOK_VERIFY_TLS`    | `true`                      | Set `false` for local receivers with self-signed certs  |
| `HOST` / `PORT`         | `127.0.0.1` / `8000`        | Bind address (`0.0.0.0` in the Docker image)            |
| `WORKERS`               | `1`                         | Server processes                                        |
| `LOG_LEVEL`             | `INFO`                      | `DEBUG`, `INFO`, `WARNING` or `ERROR`                   |

## Running without Docker

```bash
uv sync --locked --no-dev
uv run chunker warmup                       # download tokenizer files and code grammars
uv run chunker serve --host 0.0.0.0 --port 8000 --workers 4
```

Run it behind a reverse proxy (TLS) under a process manager such as systemd.

## Development

```bash
uv sync                                      # install with dev tools
uvx pre-commit install                       # run the checks on every commit
uv run pytest --cov                          # tests with coverage (90% minimum)
uvx pre-commit run --all-files               # every static check, as in CI
```

The pre-commit hooks run Ruff (format and lint), mypy in strict mode, a `uv.lock` freshness check,
hadolint, ShellCheck, actionlint and zizmor. `scripts/smoke-test.sh IMAGE` starts an image and
checks it end to end.

### Continuous integration

- **[CI](.github/workflows/ci.yml)** (pushes to `main`, pull requests, weekly): pre-commit hooks,
  tests with coverage, `uv audit` for vulnerable dependencies, and a Docker build with a smoke test.
- **[Release](.github/workflows/release.yml)** (`v*` tags): checks the tag matches the version, runs
  CI, builds and smoke-tests the image on native amd64 and arm64 runners, publishes the multi-arch
  image to GHCR with a provenance attestation, and creates the GitHub Release.
- **[Dependabot](.github/dependabot.yml)** (weekly): updates Python packages, base images and
  actions after a 7-day cooldown.

All actions are pinned to commit SHAs, and workflows run with read-only permissions unless a job
needs more.

### Releasing

```bash
uv version --bump minor                      # e.g. 0.2.0 -> 0.3.0
git commit -am "Release v$(uv version --short)"
git tag -a "v$(uv version --short)" -m "v$(uv version --short)"
git push --follow-tags
```

Tags with a suffix, such as `v1.0.0-rc.1`, are published as pre-releases and do not move `latest`.

> [!IMPORTANT]
> GitHub creates new container packages as **private**, and visibility cannot be changed through the
> API. After the first release, an organization owner must open **Packages → chunker → Package
> settings → Change visibility** and choose **Public** (the organization must allow public packages
> under **Settings → Packages**). This is needed only once; the release workflow warns until it is done.
