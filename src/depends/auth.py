from hmac import compare_digest

from fastapi import Security, HTTPException
from fastapi.security.api_key import APIKeyHeader
from starlette.status import HTTP_403_FORBIDDEN

from config.settings import settings

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


async def check_api_key(api_key: str = Security(api_key_header)) -> None:
    """Check if the provided API key is valid using constant-time comparison.

    Args:
        api_key: API key from request header

    Raises:
        HTTPException: If API key is invalid or missing
    """
    expected_key = settings.api_access_key
    if not api_key or not compare_digest(api_key, expected_key):
        raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="Could not validate API key")
