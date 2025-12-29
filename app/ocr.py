import os

from mistralai import Mistral, UploadFileOut, DeleteFileOut, RetrieveFileOut, FileSignedURL, OCRResponse
from mistralai.models import FileTypedDict
from dotenv import load_dotenv
from starlette.datastructures import UploadFile

load_dotenv()
api_key = os.environ["MISTRAL_API_KEY"]
client = Mistral(api_key=api_key)

async def upload_file(file: UploadFile) -> UploadFileOut:
    file_content = await file.read()
    
    file_dict: FileTypedDict = {
        "file_name": str(file.filename),
        "content": file_content,
    }
    
    uploaded_file = client.files.upload(
        file=file_dict,
        purpose="ocr"
    )
    
    return uploaded_file

def delete_file(file_id: str) -> DeleteFileOut:
    return client.files.delete(file_id=file_id)

def get_file(file_id: str) -> RetrieveFileOut:
    return client.files.retrieve(file_id=file_id)

def signed_url(file_id: str, expires_in: int = 2) -> FileSignedURL:
    return client.files.get_signed_url(file_id=file_id, expiry=expires_in)

def ocr(document_url: str) -> OCRResponse:
    return client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "document_url",
            "document_url": document_url,
        },
        table_format="markdown",
        include_image_base64=True
    )
