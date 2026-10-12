from fastapi import APIRouter

from app.dependencies import EmbedderDep
from app.errors import NotConfiguredError
from app.schemas import EmbeddingOut, EmbedRequest, EmbedResponse

router = APIRouter(prefix="/v1", tags=["embeddings"])


@router.post("/embed", summary="Embed texts with the server's embeddings model")
async def embed(body: EmbedRequest, embedder: EmbedderDep) -> EmbedResponse:
    if embedder is None:
        raise NotConfiguredError("Set EMBEDDINGS_MODEL on the server to use /v1/embed.")
    vectors = await embedder.aembed(body.input)
    return EmbedResponse(
        model=embedder.model,
        data=[EmbeddingOut(index=i, embedding=vector) for i, vector in enumerate(vectors)],
    )
