from typing import overload

from fastapi import Request, FastAPI

from model_manager import ModelManager


async def get_redis_session(request: Request):
    """Get Redis session from app state via request object."""
    yield request.app.state.redis


@overload
async def get_model_manager(request: Request) -> ModelManager: ...


@overload
async def get_model_manager(app: FastAPI) -> ModelManager: ...


async def get_model_manager(arg: Request | FastAPI) -> ModelManager:
    """Get model manager from app state via request object or FastAPI app.

    Args:
        arg: Either Request or FastAPI app instance

    Returns:
        ModelManager instance from app state
    """
    if isinstance(arg, Request):
        yield arg.app.state.manager  # noqa
    else:
        yield arg.state.manager  # noqa
