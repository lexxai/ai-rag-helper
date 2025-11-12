from model_manager import ModelManager


async def handler_load_model(model_name: str, manager: ModelManager):
    return await manager.get_model(model_name)


async def handler_unload_model(model_name: str, manager: ModelManager):
    return await manager.unload_model(model_name)
