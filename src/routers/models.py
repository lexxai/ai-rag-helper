from fastapi import APIRouter, Depends, Body

from depends import get_model_manager
from handlers.models import handler_load_model, handler_unload_model
from logger_config import get_logger
from model_manager import ModelManager
from schemas.models import ModelLoadResponse

logger = get_logger(__name__)

router = APIRouter(prefix="/models", tags=["Models"])


# @router.post("/load")
# async def load_model(
#     model_name: str = Body(..., embed=True), manager: ModelManager = Depends(get_model_manager)
# ) -> dict:
#     await handler_load_model(model_name, manager)
#     return {"status": "ok", "message": f"Loaded {model_name}"}


@router.post("/load")
async def load_model(
    model_name: str = Body(..., embed=True), manager: ModelManager = Depends(get_model_manager)
) -> ModelLoadResponse:
    logger.info(f"load_model {manager=}")
    # await handler_load_model(model_name)
    return ModelLoadResponse(status="ok", message=f"Loaded {model_name}")


#
# @router.post("/unload")
# async def unload_model(model_name: str = Body(..., embed=True), manager: ModelManager = Depends(get_model_manager)) -> ModelLoadResponse:
#     await handler_unload_model(model_name, manager)
#     return ModelLoadResponse(status="ok", message=f"Unloaded {model_name}")
