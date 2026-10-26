# syntax=docker/dockerfile:1

FROM ghcr.io/astral-sh/uv:0.13.5@sha256:32a7bb39b496babbd997b0a39a4d66c126bdaf5661b302bf6f06abda8deca6ff AS uv

# --- Build: resolve the locked dependencies into a virtualenv -------------------
FROM python:3.14-slim-trixie@sha256:a2b82f3c48559aa0a8446d9af49826b6e2b2016f4cd2afabfe6013ec53729170 AS builder

COPY --from=uv /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /src

# Dependencies first, so code changes do not invalidate this layer.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=.python-version,target=.python-version \
    uv sync --locked --no-dev --no-install-project

COPY pyproject.toml uv.lock .python-version README.md ./
COPY app ./app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

# --- Runtime -------------------------------------------------------------------
# Debian slim rather than Alpine: Chonkie's compiled components publish no musl wheels.
FROM python:3.14-slim-trixie@sha256:a2b82f3c48559aa0a8446d9af49826b6e2b2016f4cd2afabfe6013ec53729170 AS runtime

RUN groupadd --system --gid 10001 chunker \
    && useradd --system --uid 10001 --gid chunker --create-home --home-dir /home/chunker chunker

# DL3064 is a false positive: "TIKTOKEN_CACHE_DIR" is a path, not a secret.
# hadolint ignore=DL3064
ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    TIKTOKEN_CACHE_DIR=/home/chunker/.cache/tiktoken

COPY --from=builder /app/.venv /app/.venv
WORKDIR /app
USER 10001:10001

# Bake tokenizer files and the CODE_LANGUAGES grammars into the image: requests never download.
RUN chunker warmup

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen(f\"http://127.0.0.1:{os.environ.get('PORT', '8000')}/health\", timeout=4)"]

CMD ["chunker", "serve"]
