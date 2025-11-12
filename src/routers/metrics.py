from fastapi import APIRouter
from fastapi.responses import Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from logger_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/metrics", tags=["metrics"])


# Prometheus metrics endpoint
@router.get("/")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
