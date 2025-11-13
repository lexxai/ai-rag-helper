from fastapi import APIRouter, Depends

from config.settings import settings
from depends.model_manager import get_model_manager
from handlers.ebbeding import handler_embedding
from logger_config import get_logger
from model_manager import ModelManager
from schemas.embedding import EmbeddingRequest, EmbeddingResponse

logger = get_logger(__name__)

router = APIRouter(prefix="/embed", tags=["Embedding"])


@router.post("/")
async def embed(req: EmbeddingRequest, manager: ModelManager = Depends(get_model_manager)) -> EmbeddingResponse:
    model_name = req.model or getattr(settings, "default_model_names", {}).get("embed")
    if not model_name:
        raise ValueError("Model name is required")
    results = await handler_embedding(model_name, req.texts, manager) or {}
    return EmbeddingResponse(**results)
