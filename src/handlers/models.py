from model_manager import ModelManager
from schemas.models import ModelListItems


async def handler_load_model(model_name: str, manager: ModelManager):
    return await manager.get_model(model_name)


async def handler_unload_model(model_name: str, manager: ModelManager):
    return await manager.unload_model(model_name)


async def handler_list_loaded_models(manager: ModelManager) -> list[ModelListItems]:
    return await manager.list_loaded()


async def handler_list_available(manager: ModelManager) -> list[str]:
    return await manager.list_available()
