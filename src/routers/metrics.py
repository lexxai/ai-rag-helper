from fastapi import APIRouter, status
from fastapi.responses import Response

try:
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
except ImportError:
    generate_latest = None
    CONTENT_TYPE_LATEST = None

from config.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/metrics", tags=["metrics"])


# Prometheus metrics endpoint
@router.get("/")
async def metrics():
    if generate_latest is None:
        return Response(status_code=status.HTTP_501_NOT_IMPLEMENTED)
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
