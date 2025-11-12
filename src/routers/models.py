from fastapi import APIRouter, Depends, Body

from depends import get_model_manager
from handlers.models import handler_load_model, handler_unload_model
from logger_config import get_logger
from model_manager import ModelManager
from schemas.models import ModelLoadResponse

logger = get_logger(__name__)

router = APIRouter(prefix="/models", tags=["Models"])


@router.post("/load", response_model=ModelLoadResponse)
async def load_model(model_name: str = Body(..., embed=True), manager: ModelManager = Depends(get_model_manager)):
    await handler_load_model(model_name, manager)
    return {"status": "ok", "message": f"Loaded {model_name}"}


@router.post("/unload", response_model=ModelLoadResponse)
async def unload_model(model_name: str = Body(..., embed=True), manager: ModelManager = Depends(get_model_manager)):
    await handler_unload_model(model_name, manager)
    return {"status": "ok", "message": f"Unloaded {model_name}"}
