#!/usr/bin/env bash
# Start a Chunker image and check that it serves requests.
# Usage: scripts/smoke-test.sh IMAGE
set -euo pipefail

image="${1:?usage: $0 IMAGE}"
name="chunker-smoke-$$"
key="smoke-test-key"

cleanup() {
  if [ "${passed:-false}" != true ]; then
    docker logs "$name" 2>&1 | tail -50 || true
  fi
  docker rm -f "$name" >/dev/null 2>&1 || true
}
trap cleanup EXIT

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

docker run -d --name "$name" -p 127.0.0.1::8000 -e API_KEY="$key" "$image" >/dev/null
port="$(docker port "$name" 8000/tcp | head -n1 | cut -d: -f2)"
base="http://127.0.0.1:$port"

status=starting
for _ in $(seq 1 90); do
  status="$(docker inspect -f '{{.State.Health.Status}}' "$name")"
  [ "$status" = starting ] || break
  sleep 2
done
[ "$status" = healthy ] || fail "container is $status"
echo "ok: container is healthy"

[ "$(docker exec "$name" id -u)" != 0 ] || fail "container runs as root"
echo "ok: runs as a non-root user"

curl -fsS "$base/health" | grep -q '"status":"ok"' || fail "/health"
echo "ok: /health"

code="$(curl -s -o /dev/null -w '%{http_code}' "$base/v1/engines")"
[ "$code" = 401 ] || fail "expected 401 without API key, got $code"
echo "ok: API_KEY from the environment is enforced"

curl -fsS "$base/v1/chunk" \
  -H "Authorization: Bearer $key" \
  -H "Content-Type: application/json" \
  -d '{"text": "First sentence here. Second sentence there.", "chunker": {"engine": "chonkie", "strategy": "sentence", "chunk_size": 8}}' \
  | grep -q '"total_chunks":2' || fail "/v1/chunk"
echo "ok: /v1/chunk"

grammars="$(docker exec "$name" python -c 'import tree_sitter_language_pack as t; print(len(t.downloaded_languages()))')"
[ "$grammars" -ge 20 ] || fail "expected the CODE_LANGUAGES grammars in the image, found $grammars"
echo "ok: $grammars code grammars installed"

curl -fsS "$base/v1/chunk" \
  -H "Authorization: Bearer $key" \
  -H "Content-Type: application/json" \
  -d '{"text": "def add(a, b):\n    return a + b\n\n\nclass Calc:\n    def mul(self, a, b):\n        return a * b\n", "chunker": {"engine": "chonkie", "strategy": "code", "language": "python", "chunk_size": 20}}' \
  | grep -q '"total_chunks":2' || fail "/v1/chunk with code"
echo "ok: code chunking"

curl -fsS "$base/v1/tokens" \
  -H "Authorization: Bearer $key" \
  -H "Content-Type: application/json" \
  -d '{"input": ["hello world"]}' \
  | grep -q '"tokens":2' || fail "/v1/tokens"
echo "ok: /v1/tokens"

# Pre-parsed pages (e.g. OCR output) exercise LlamaIndex and the page-number mapping.
curl -fsS "$base/v1/chunk" \
  -H "Authorization: Bearer $key" \
  -H "Content-Type: application/json" \
  -d '{"pages": [{"text": "Page one has a sentence."}, {"text": "Page two has another sentence."}], "chunker": {"engine": "llamaindex", "strategy": "sentence", "chunk_size": 8, "chunk_overlap": 0}}' \
  | grep -q '"page_numbers":\[2\]' || fail "/v1/chunk with pages"
echo "ok: /v1/chunk with pages"

passed=true
echo "Smoke test passed for $image"
