from fastapi import APIRouter, Depends, Body

from depends import get_model_manager
from handlers.models import handler_load_model, handler_unload_model, handler_list_loaded_models, handler_list_available
from logger_config import get_logger
from model_manager import ModelManager
from schemas.models import ModelLoadResponse, ModelListItems

logger = get_logger(__name__)

router = APIRouter(prefix="/models", tags=["Models"])


@router.post("/load")
async def load_model(
    model_name: str = Body(..., embed=True), manager: ModelManager = Depends(get_model_manager)
) -> ModelLoadResponse:
    model_instance = await handler_load_model(model_name, manager)
    if model_instance is not None:
        return ModelLoadResponse(status="ok", message=f"Loaded {model_name}")
    return ModelLoadResponse(status="error", message=f"Not loaded {model_name}")


#
@router.post("/unload")
async def unload_model(
    model_name: str = Body(..., embed=True), manager: ModelManager = Depends(get_model_manager)
) -> ModelLoadResponse:
    result = await handler_unload_model(model_name, manager)
    if result:
        return ModelLoadResponse(status="ok", message=f"Unloaded {model_name}")
    return ModelLoadResponse(status="error", message=f"Not unloaded {model_name}")


@router.post("/loaded")
async def unload_model(manager: ModelManager = Depends(get_model_manager)) -> list[ModelListItems]:
    result = await handler_list_loaded_models(manager)
    return result


@router.post("/available")
async def unload_model(manager: ModelManager = Depends(get_model_manager)) -> list[str]:
    result = await handler_list_available(manager)
    return result
