from fastapi import APIRouter, Depends

from depends.databases import get_redis_session
from depends.model_manager import get_model_manager
from handlers.rerank import handler_rerank
from logger_config import get_logger
from model_manager import ModelManager
from schemas.rerank import RerankRequest, RerankResponse

logger = get_logger(__name__)

router = APIRouter(prefix="/rerank", tags=["Rerank"])


@router.post("/")
async def rerank(
    req: RerankRequest, manager: ModelManager = Depends(get_model_manager), redis=Depends(get_redis_session)
) -> RerankResponse:
    rerank_result = await handler_rerank(req.model, req.query, req.candidates, manager, redis)
    return RerankResponse(**rerank_result)
