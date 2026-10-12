from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class ChunkerError(Exception):
    """An error with a client-facing message, rendered as `{"detail": ...}`."""

    status_code = 400

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class InvalidInputError(ChunkerError):
    """The input cannot be processed by the chosen strategy (e.g. invalid JSON)."""

    status_code = 422


class NotConfiguredError(ChunkerError):
    """The request needs a feature that is not configured on this server."""


class UpstreamError(ChunkerError):
    status_code = 502


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ChunkerError)
    async def handle_chunker_error(_: Request, exc: ChunkerError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
