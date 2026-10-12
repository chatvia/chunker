from fastapi import APIRouter

from app.dependencies import EmbedderDep, SettingsDep
from app.embeddings import check_request_embeddings, resolve_embedder
from app.errors import NotConfiguredError
from app.schemas import EmbeddingOut, EmbedRequest, EmbedResponse

router = APIRouter(prefix="/v1", tags=["embeddings"])


@router.post("/embed", summary="Embed texts")
async def embed(body: EmbedRequest, settings: SettingsDep, embedder: EmbedderDep) -> EmbedResponse:
    check_request_embeddings(body.embeddings, embedder, settings)
    async with resolve_embedder(body.embeddings, embedder, settings) as active:
        if active is None:
            raise NotConfiguredError(
                "Set EMBEDDINGS_MODEL on the server or send `embeddings` with the request."
            )
        vectors = await active.aembed(body.input)
        model = active.model
    return EmbedResponse(
        model=model,
        data=[EmbeddingOut(index=i, embedding=vector) for i, vector in enumerate(vectors)],
    )
