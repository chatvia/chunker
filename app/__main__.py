"""Command line entry point: `chunker serve` (default) or `chunker warmup`."""

import argparse
import logging

import uvicorn

from app.config import get_settings
from app.tokens import ENCODINGS, get_encoding

logger = logging.getLogger("app")


def warmup() -> None:
    """Fetch data that libraries otherwise download lazily on the first request."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for name in ENCODINGS:
        logger.info("Loading tiktoken encoding %s", name)
        get_encoding(name)
    logger.info("Warmup complete")


def main(argv: list[str] | None = None) -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(prog="chunker", description=__doc__)
    commands = parser.add_subparsers(dest="command")

    serve = commands.add_parser("serve", help="Run the HTTP server (default)")
    serve.add_argument("--host", default=settings.host)
    serve.add_argument("--port", type=int, default=settings.port)
    serve.add_argument("--workers", type=int, default=1)
    serve.add_argument("--reload", action="store_true", help="Reload on code changes (dev)")

    commands.add_parser("warmup", help="Pre-download tokenizer data used at runtime")

    args = parser.parse_args(argv)
    if args.command == "warmup":
        warmup()
        return

    uvicorn.run(
        "app.main:app",
        host=getattr(args, "host", settings.host),
        port=getattr(args, "port", settings.port),
        workers=getattr(args, "workers", 1),
        reload=getattr(args, "reload", False),
        log_level=settings.log_level.lower(),
        proxy_headers=True,
    )


if __name__ == "__main__":
    main()
