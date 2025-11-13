from fastapi import APIRouter, Depends

from config.settings import settings
from depends.databases import get_redis_session
from depends.model_manager import get_model_manager
from handlers.ebbeding import handler_embedding_cache
from logger_config import get_logger
from model_manager import ModelManager
from schemas.embedding import EmbeddingRequest, EmbeddingResponse

logger = get_logger(__name__)

router = APIRouter(prefix="/embed", tags=["Embedding"])


@router.post("/")
async def embed(
    req: EmbeddingRequest, manager: ModelManager = Depends(get_model_manager), redis=Depends(get_redis_session)
) -> EmbeddingResponse:
    model_name = req.model or getattr(settings, "default_model_names", {}).get("embed")
    if not model_name:
        raise ValueError("Model name is required")
    results = await handler_embedding_cache(model_name, req.texts, manager, redis) or {}
    return EmbeddingResponse(**results)
