import os
import hmac
import hashlib
from dotenv import load_dotenv

load_dotenv()


def generate_auth_hash(webhook_id: str, chunk_id: int, index: int) -> str:
    """
    Generate HMAC hash for webhook authentication.
    
    Args:
        webhook_id: UUID string of the webhook
        chunk_id: Database ID of the chunk
        index: Index of the chunk
        
    Returns:
        HMAC SHA256 hash as hexadecimal string
    """
    hash_key = os.environ.get("HASH_KEY")
    
    if not hash_key:
        raise ValueError("HASH_KEY environment variable is not set")
    
    # Create message from webhook_id, chunk_id, and index
    message = f"{webhook_id}:{chunk_id}:{index}"
    
    # Generate HMAC SHA256 hash
    hash_obj = hmac.new(
        hash_key.encode('utf-8'),
        message.encode('utf-8'),
        hashlib.sha256
    )
    
    return hash_obj.hexdigest()


