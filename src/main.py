from contextlib import asynccontextmanager

import redis.asyncio as redis
from fastapi import FastAPI

from logger_config import setup_root_logger, get_logger
from model_manager import ModelManager
from routers import cache, models
from config.settings import settings

# Setup root logger for the application
setup_root_logger(level=settings.log_level)

# Get logger for this module
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context to initialize Redis"""
    redis_url = settings.redis_url

    try:
        redis_client = redis.from_url(redis_url)
        await redis_client.ping()
        logger.debug("Connected to Redis")
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")
        redis_client = None

    # Store in app state
    app.state.redis = redis_client  # noqa
    try:
        manager = ModelManager(timeout=settings.model_manager_timeout, check_gpu=settings.model_manager_check_gpu)
        app.state.manager = manager  # noqa
    except Exception as e:
        logger.error(f"ModelManager Creating failed: {e}")
        raise

    yield

    # Cleanup
    if manager is not None:
        await manager.close()
        logger.debug("Model Manager closed")

    if redis_client is not None:
        await redis_client.close()
        await redis_client.connection_pool.disconnect()
        logger.debug("Redis disconnected")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    debug=settings.debug,
    lifespan=lifespan,
)

# Include routers with api_prefix from settings
app.include_router(cache.router, prefix=settings.api_prefix)
app.include_router(models.router, prefix=settings.api_prefix)
# app.include_router(ws.router_ws, prefix=settings.api_prefix)
