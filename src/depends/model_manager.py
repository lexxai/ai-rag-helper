from fastapi import Request

from model_manager import ModelManager


async def get_model_manager(arg: Request) -> ModelManager:
    """Get model manager from app state via request object.

    Args:
        arg: Request object containing app state

    Returns:
        ModelManager instance from app state
    """
    if isinstance(arg, Request):
        return arg.app.state.manager

    return arg.state.manager  # noqa
