import base64
import io
import os

import boto3
from botocore.config import Config
from dotenv import load_dotenv
from PIL import Image

load_dotenv()


def get_s3_client():
    """Create and return a configured S3 client for Scaleway."""
    access_key_id = os.environ.get("SCALEWAY_ACCESS_KEY_ID")
    secret_access_key = os.environ.get("SCALEWAY_SECRET_ACCESS_KEY")
    region = os.environ.get("SCALEWAY_DEFAULT_REGION", "fr-par")
    endpoint = os.environ.get("SCALEWAY_ENDPOINT")
    use_path_style = os.environ.get("SCALEWAY_USE_PATH_STYLE_ENDPOINT", "true").lower() == "true"

    if not access_key_id or not secret_access_key:
        raise ValueError("SCALEWAY_ACCESS_KEY_ID and SCALEWAY_SECRET_ACCESS_KEY must be set")

    config = Config(
        s3={
            "addressing_style": "path" if use_path_style else "virtual"
        }
    )

    s3_client = boto3.client(
        "s3",
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        region_name=region,
        endpoint_url=endpoint,
        config=config,
    )

    return s3_client


def upload_image_to_s3(image_base64: str, s3_path: str) -> str:
    """
    Upload a base64-encoded image to Scaleway S3 and return the S3 path.

    Args:
        image_base64: Base64-encoded image string (with or without data URI prefix)
        s3_path: Full S3 path (key) where the image should be uploaded

    Returns:
        S3 path (key) where the image was uploaded
    """
    # Remove data URI prefix if present (e.g., "data:image/png;base64,")
    if "," in image_base64:
        image_base64 = image_base64.split(",", 1)[1]

    # Decode base64 to bytes
    try:
        image_bytes = base64.b64decode(image_base64)
    except Exception as e:
        raise ValueError(f"Failed to decode base64 image: {e}")

    # Open image with Pillow and convert to JPEG
    try:
        image = Image.open(io.BytesIO(image_bytes))

        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Convert to JPEG bytes
        jpeg_buffer = io.BytesIO()
        image.save(jpeg_buffer, format="JPEG", quality=95)
        image_bytes = jpeg_buffer.getvalue()
    except Exception as e:
        raise ValueError(f"Failed to process image with Pillow: {e}")

    # Get S3 client and bucket
    s3_client = get_s3_client()
    bucket_name = os.environ.get("SCALEWAY_BUCKET")

    if not bucket_name:
        raise ValueError("SCALEWAY_BUCKET must be set")

    # Upload to S3
    try:
        s3_client.put_object(
            Bucket=bucket_name,
            Key=s3_path,
            Body=image_bytes,
            ContentType="image/jpeg",
        )
    except Exception as e:
        raise RuntimeError(f"Failed to upload image to S3: {e}")

    # Return S3 path (key)
    return s3_path


def get_s3_url(s3_key: str) -> str:
    """
    Generate a public URL for an S3 object.

    Args:
        s3_key: S3 key/path of the object

    Returns:
        Public URL to access the object
    """
    base_url = os.environ.get("SCALEWAY_URL")
    bucket_name = os.environ.get("SCALEWAY_BUCKET")

    if not base_url or not bucket_name:
        raise ValueError("SCALEWAY_URL and SCALEWAY_BUCKET must be set")

    # Construct URL based on path style or virtual host style
    use_path_style = os.environ.get("SCALEWAY_USE_PATH_STYLE_ENDPOINT", "true").lower() == "true"

    if use_path_style:
        # Path-style: https://endpoint/bucket/key
        return f"{base_url}/{bucket_name}/{s3_key}"
    else:
        # Virtual-host-style: https://bucket.endpoint/key
        return f"{base_url}/{s3_key}"

