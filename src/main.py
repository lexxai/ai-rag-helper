from contextlib import asynccontextmanager

from fastapi import FastAPI
from redis.asyncio import Redis

from logger_config import setup_root_logger, get_logger
from settings import settings
from routers import cache

# Setup root logger for the application
setup_root_logger(level=settings.log_level)

# Get logger for this module
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context to initialize Redis"""
    redis_url = settings.redis_url

    try:
        redis_client = Redis.from_url(redis_url)
        await redis_client.ping()
        logger.debug("Connected to Redis")
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")
        redis_client = None

    # Store in app state
    app.state.redis = redis_client  # noqa
    yield
    # Cleanup
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
