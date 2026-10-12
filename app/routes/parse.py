from typing import Annotated

from fastapi import APIRouter, File, Form, UploadFile
from pydantic import Json

from app import pipeline
from app.dependencies import SettingsDep
from app.engines import DEFAULT_PARSER, ParserConfig
from app.schemas import PARSER_FORM_HELP, ElementOut, ParseResponse
from app.tokens import DEFAULT_ENCODING, Encoding

router = APIRouter(prefix="/v1", tags=["parsing"])


@router.post("/parse", summary="Extract text and elements from a file")
async def parse_file(
    file: Annotated[UploadFile, File(description="Document to parse.")],
    settings: SettingsDep,
    parser: Annotated[
        Json[ParserConfig] | None,
        Form(description=PARSER_FORM_HELP),
    ] = None,
    encoding: Annotated[Encoding, Form()] = DEFAULT_ENCODING,
) -> ParseResponse:
    parser_config = parser or DEFAULT_PARSER
    filename = file.filename or "upload"
    pipeline.check_supported(parser_config, filename)
    data = await pipeline.read_upload(file, settings.max_upload_bytes)
    document = await pipeline.parse(parser_config, data, filename, file.content_type)
    return ParseResponse(
        parser=parser_config.engine,
        document=pipeline.document_info(document, encoding),
        text=document.text,
        elements=[
            ElementOut(
                text=element.text,
                type=element.type,
                page_number=element.page_number,
                metadata=element.metadata,
            )
            for element in document.elements
        ],
    )
