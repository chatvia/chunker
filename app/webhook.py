import os
import httpx
from dotenv import load_dotenv
from app.hash import generate_auth_hash

load_dotenv()


async def send_chunk_webhook(webhook_id: str, chunk_id: int, index: int) -> bool:
    """
    Send webhook POST request with webhook_id and hashed auth key to the configured endpoint.
    
    Args:
        webhook_id: UUID string to send in the webhook
        chunk_id: Database ID of the chunk
        index: Index of the chunk
        
    Returns:
        True if webhook was sent successfully, False otherwise
    """
    webhook_endpoint = os.environ.get("WEBHOOK_CHUNK_ENDPOINT")
    
    if not webhook_endpoint:
        return False
    
    try:
        # Generate HMAC hash for authentication
        auth_hash = generate_auth_hash(webhook_id, chunk_id, index)
        
        async with httpx.AsyncClient(timeout=30.0, verify=False) as client:
            response = await client.post(
                webhook_endpoint,
                json={
                    "webhook_id": webhook_id,
                    "auth_hash": auth_hash
                }
            )
            response.raise_for_status()
            return True
    except Exception as e:
        print(f"Failed to send webhook: {e}")
        return False

