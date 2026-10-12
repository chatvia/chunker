from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send


class BodySizeLimitMiddleware:
    """Reject requests whose declared Content-Length exceeds a limit before reading the body."""

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            length = dict(scope["headers"]).get(b"content-length", b"")
            if length.isdigit() and int(length) > self.max_bytes:
                limit_mb = self.max_bytes // (1024 * 1024)
                response = JSONResponse(
                    {"detail": f"Request body exceeds the {limit_mb} MB limit"}, status_code=413
                )
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)
