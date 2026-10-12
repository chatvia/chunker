import inspect
from typing import get_args

from fastapi import APIRouter
from pydantic import BaseModel

from app.dependencies import EmbedderDep
from app.engines import CHUNKERS, PARSERS
from app.schemas import ChunkerInfo, EmbeddingsInfo, EnginesResponse, ParserInfo
from app.tokens import ENCODINGS

router = APIRouter(prefix="/v1", tags=["engines"])


def _literal(model: type[BaseModel], field: str) -> str:
    value: str = get_args(model.model_fields[field].annotation)[0]
    return value


def _description(model: type[BaseModel]) -> str:
    return inspect.cleandoc(model.__doc__ or "")


@router.get("/engines", summary="List parsers, chunking strategies and their options")
async def list_engines(embedder: EmbedderDep) -> EnginesResponse:
    return EnginesResponse(
        parsers=[
            ParserInfo(
                engine=_literal(parser, "engine"),
                description=_description(parser),
                extensions=sorted(parser.extensions),
                options=parser.model_json_schema(),
            )
            for parser in PARSERS
        ],
        chunkers=[
            ChunkerInfo(
                engine=_literal(chunker, "engine"),
                strategy=_literal(chunker, "strategy"),
                description=_description(chunker),
                requires_embeddings=chunker.requires_embeddings,
                options=chunker.model_json_schema(),
            )
            for chunker in CHUNKERS
        ],
        encodings=list(ENCODINGS),
        embeddings=EmbeddingsInfo(
            configured=embedder is not None, model=embedder.model if embedder else None
        ),
    )
