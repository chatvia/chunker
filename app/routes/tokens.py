from fastapi import APIRouter

from app.schemas import TokenCount, TokensRequest, TokensResponse
from app.tokens import count_tokens

router = APIRouter(prefix="/v1", tags=["tokens"])


@router.post("/tokens", summary="Count tokens")
async def tokens(body: TokensRequest) -> TokensResponse:
    return TokensResponse(
        encoding=body.encoding,
        data=[
            TokenCount(index=i, tokens=count_tokens(text, body.encoding))
            for i, text in enumerate(body.input)
        ],
    )
