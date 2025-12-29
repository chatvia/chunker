import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr

from app.utils import load_prompt

load_dotenv()
api_key = os.environ["SCALEWAYAI_API_KEY"]

_client = ChatOpenAI(
    api_key=SecretStr(api_key),
    base_url=os.environ["SCALEWAYAI_ENDPOINT"],
)


async def completions(model, messages, **kwargs):
    return await _client.chat.completions.create(
        model=model,
        messages=messages,
        **kwargs
    )


async def embed(text):
    embeddings = OpenAIEmbeddings(
        api_key=SecretStr(api_key),
        base_url=os.environ["SCALEWAYAI_ENDPOINT"],
        model="qwen3-embedding-8b",
        dimensions=1024,
    )

    return embeddings.embed_query(text)


async def vision(image_base64: str) -> str | bool:
    """
    Describe an image using vision model. Returns description or False if no_info.

    Args:
        image_base64: Base64-encoded image string (with or without data URI prefix)

    Returns:
        Image description string, or False if image contains no_info
    """
    # Remove data URI prefix if present
    if "," in image_base64:
        image_base64 = image_base64.split(",", 1)[1]

    # Construct data URI for the image
    image_url = f"data:image/jpeg;base64,{image_base64}"

    # Load vision prompt
    prompt = load_prompt("image-vision-describe.md")

    # Prepare messages for vision API (OpenAI-compatible format)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt,
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url,
                    },
                },
            ],
        }
    ]

    try:
        # Use the same client as completions function
        response = await _client.chat.completions.create(
            model="gemma-3-27b-it",
            messages=messages,
        )

        result = response.choices[0].message.content or "no_info"

        # Check if result indicates no information
        if "no_info" in result.lower():
            return False

        return result
    except Exception as e:
        print(f"Vision API error: {e}")
        return False


async def describe_tables(tables: list[str]) -> str:
    """
    Describe table(s) in natural language using LLM.

    Args:
        tables: List of table markdown strings

    Returns:
        Natural language description of the tables
    """
    if not tables:
        return ""

    # Load table description prompt
    prompt_template = load_prompt("describe-table.md")

    # Join tables with newlines
    tables_text = "\n\n".join(tables)

    # Format prompt with tables
    prompt = prompt_template.format(tables=tables_text)

    # Prepare messages
    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    try:
        response = await _client.chat.completions.create(
            model="llama-3.3-70b-instruct",  # Use default model for text processing
            messages=messages,
            temperature=0,
        )

        result = response.choices[0].message.content or ""
        return result.strip()
    except Exception as e:
        print(f"Table description API error: {e}")
        return ""
