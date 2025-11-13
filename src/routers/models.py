from fastapi import APIRouter, Depends, HTTPException, Query
from starlette import status

from depends.model_manager import get_model_manager
from handlers.models import (
    handler_load_model,
    handler_unload_model,
    handler_list_loaded_models,
    handler_list_available,
    handler_retrieve_properties,
)
from logger_config import get_logger
from model_manager import ModelManager
from schemas.models import ModelLoadResponse, ModelListItems, FilterParamsModelName

logger = get_logger(__name__)

router = APIRouter(prefix="/models", tags=["Models"])


@router.get("/load")
async def load_model(
    model_name: str = Query(description="Model name"), manager: ModelManager = Depends(get_model_manager)
) -> ModelLoadResponse:
    model_instance = await handler_load_model(model_name, manager)
    if model_instance is not None:
        return ModelLoadResponse(status="ok", message=f"Loaded {model_name}")
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to load model {model_name}")


#
@router.get("/unload")
async def unload_model(
    model_name: str = Query(description="Model name"), manager: ModelManager = Depends(get_model_manager)
) -> ModelLoadResponse:
    result = await handler_unload_model(model_name, manager)
    if result:
        return ModelLoadResponse(status="ok", message=f"Unloaded {model_name}")
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to unload model {model_name}")


@router.get("/loaded")
async def list_loaded_models(manager: ModelManager = Depends(get_model_manager)) -> list[ModelListItems]:
    result = await handler_list_loaded_models(manager)
    return result


@router.get("/available")
async def list_available_models(manager: ModelManager = Depends(get_model_manager)) -> list[str]:
    result = await handler_list_available(manager)
    return result


@router.get("/properties")
async def retrieve_properties(
    model_name: str = Query(description="Model name"), manager: ModelManager = Depends(get_model_manager)
) -> dict | None:
    result = await handler_retrieve_properties(model_name, manager)
    return result
