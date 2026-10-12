import secrets
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import Settings
from app.embeddings import Embedder
from app.webhooks import WebhookSender

_bearer = HTTPBearer(auto_error=False, description="Required when the server sets API_KEY.")


def get_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def get_embedder(request: Request) -> Embedder | None:
    embedder: Embedder | None = request.app.state.embedder
    return embedder


def get_webhook_sender(request: Request) -> WebhookSender | None:
    sender: WebhookSender | None = request.app.state.webhook_sender
    return sender


SettingsDep = Annotated[Settings, Depends(get_settings)]
EmbedderDep = Annotated[Embedder | None, Depends(get_embedder)]
WebhookSenderDep = Annotated[WebhookSender | None, Depends(get_webhook_sender)]


async def require_api_key(
    settings: SettingsDep,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> None:
    if settings.api_key is None:
        return
    expected = settings.api_key.get_secret_value().encode()
    if credentials is None or not secrets.compare_digest(
        credentials.credentials.encode(), expected
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
