import inspect
from typing import get_args

from fastapi import APIRouter
from pydantic import BaseModel

from app.dependencies import EmbedderDep, SettingsDep
from app.engines import CHUNKERS
from app.schemas import ChunkerInfo, EmbeddingsInfo, EnginesResponse
from app.tokens import ENCODINGS

router = APIRouter(prefix="/v1", tags=["engines"])


def _literal(model: type[BaseModel], field: str) -> str:
    value: str = get_args(model.model_fields[field].annotation)[0]
    return value


def _description(model: type[BaseModel]) -> str:
    return inspect.cleandoc(model.__doc__ or "")


@router.get("/engines", summary="List chunking strategies and their options")
async def list_engines(settings: SettingsDep, embedder: EmbedderDep) -> EnginesResponse:
    return EnginesResponse(
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
            configured=embedder is not None,
            model=embedder.model if embedder else None,
            per_request=embedder is None and settings.allow_request_embeddings,
        ),
    )
