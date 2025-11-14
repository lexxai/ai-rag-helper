from fastapi import APIRouter, Depends, HTTPException, Query
from starlette import status

from depends.auth import check_api_key
from depends.model_manager import get_model_manager
from handlers.models import (
    handler_load_model,
    handler_unload_model,
    handler_list_loaded_models,
    handler_list_available_names,
    handler_retrieve_properties,
    handler_preload_models,
    handler_list_available_types,
)
from logger_config import get_logger
from model_manager import ModelManager
from schemas.models import ModelLoadResponse, ModelListItems

logger = get_logger(__name__)

router = APIRouter(prefix="/models", tags=["Models"])


@router.get("/load")
async def load_model(
    model_name: str = Query(description="Model name can be in format: 'type:name'"),
    manager: ModelManager = Depends(get_model_manager),
) -> ModelLoadResponse:
    model_name = model_name.strip('"').strip("'").strip()
    model_instance = await handler_load_model(model_name, manager)
    if model_instance is not None:
        return ModelLoadResponse(status="ok", message=f"Loaded {model_name}")
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to load model {model_name}")


#
@router.get("/unload")
async def unload_model(
    model_name: str = Query(description="Model name can be in format: 'type:name'"),
    manager: ModelManager = Depends(get_model_manager),
) -> ModelLoadResponse:
    model_name = model_name.strip('"').strip("'").strip()
    result = await handler_unload_model(model_name, manager)
    if result:
        return ModelLoadResponse(status="ok", message=f"Unloaded {model_name}")
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to unload model {model_name}")


@router.get("/loaded")
async def list_loaded_models(manager: ModelManager = Depends(get_model_manager)) -> list[ModelListItems]:
    result = await handler_list_loaded_models(manager)
    return result


@router.get("/available_names")
def list_available_models(
    model_types: str | None = None, manager: ModelManager = Depends(get_model_manager)
) -> list[str]:
    model_types = model_types.strip('"').strip("'").strip()
    result = handler_list_available_names(manager, model_types)
    return result


@router.get("/available_types")
def list_available_types(manager: ModelManager = Depends(get_model_manager)) -> list[str]:
    result = handler_list_available_types(manager)
    return result


@router.get("/properties")
def retrieve_properties(
    model_name: str = Query(description="Model name can be in format: 'type:name'"),
    manager: ModelManager = Depends(get_model_manager),
) -> dict | None:
    result = handler_retrieve_properties(model_name, manager)
    return result


@router.get("/preload", dependencies=[Depends(check_api_key)])
async def preload_models(manager: ModelManager = Depends(get_model_manager)) -> list[str]:
    """Forced to preload all available models to the disk cache."""
    result = await handler_preload_models(manager)
    return result
