"""Outgoing webhooks, signed according to the Standard Webhooks spec (https://standardwebhooks.com).

Receivers can verify deliveries with any Standard Webhooks library (PHP, Node, Go, Python, ...)
using the same `WEBHOOK_SECRET`.
"""

import asyncio
import base64
import binascii
import hashlib
import hmac
import json
import logging
import time
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import httpx

logger = logging.getLogger(__name__)

SECRET_PREFIX = "whsec_"  # noqa: S105 - a format prefix, not a secret
_RETRYABLE_STATUS = frozenset({408, 425, 429})


def decode_secret(secret: str) -> bytes:
    try:
        key = base64.b64decode(secret.removeprefix(SECRET_PREFIX), validate=True)
    except binascii.Error as exc:
        raise ValueError(
            "webhook secret must be base64-encoded, optionally prefixed with 'whsec_' "
            "(generate one with: echo whsec_$(openssl rand -base64 32))"
        ) from exc
    if not key:
        raise ValueError("webhook secret must not be empty")
    return key


def sign(key: bytes, msg_id: str, timestamp: int, body: bytes) -> str:
    signed_content = f"{msg_id}.{timestamp}.".encode() + body
    digest = hmac.new(key, signed_content, hashlib.sha256).digest()
    return f"v1,{base64.b64encode(digest).decode()}"


class WebhookSender:
    def __init__(
        self,
        client: httpx.AsyncClient,
        key: bytes,
        *,
        max_attempts: int = 3,
        backoff_seconds: float = 1.0,
        user_agent: str = "chunker",
    ) -> None:
        self._client = client
        self._key = key
        self._max_attempts = max_attempts
        self._backoff_seconds = backoff_seconds
        self._user_agent = user_agent

    async def send(self, url: str, event_type: str, data: Mapping[str, Any]) -> bool:
        """Deliver an event, retrying transient failures. Returns whether it was accepted."""
        msg_id = f"msg_{uuid.uuid4().hex}"
        created_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        payload = {"type": event_type, "timestamp": created_at, "data": data}
        body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode()

        for attempt in range(1, self._max_attempts + 1):
            timestamp = int(time.time())
            headers = {
                "content-type": "application/json",
                "user-agent": self._user_agent,
                "webhook-id": msg_id,
                "webhook-timestamp": str(timestamp),
                "webhook-signature": sign(self._key, msg_id, timestamp, body),
            }
            try:
                response = await self._client.post(url, content=body, headers=headers)
            except httpx.HTTPError as exc:
                retryable = True
                logger.warning("Webhook %s attempt %d failed: %s", msg_id, attempt, exc)
            else:
                if response.is_success:
                    logger.info("Webhook %s (%s) delivered", msg_id, event_type)
                    return True
                retryable = response.status_code >= 500 or response.status_code in _RETRYABLE_STATUS
                logger.warning(
                    "Webhook %s attempt %d got HTTP %d", msg_id, attempt, response.status_code
                )

            if not retryable or attempt == self._max_attempts:
                break
            await asyncio.sleep(self._backoff_seconds * 2 ** (attempt - 1))

        logger.error("Webhook %s (%s) could not be delivered to %s", msg_id, event_type, url)
        return False
