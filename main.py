from contextlib import asynccontextmanager
from os import environ

from fastapi import FastAPI, Depends, Body
from redis.asyncio import Redis

from logger_config import setup_root_logger, get_logger
from settings import settings

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
        logger.error("Redis disconnected")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    debug=settings.debug,
    lifespan=lifespan,
)


async def get_redis_session():
    yield app.state.redis  # noqa


@app.post("/cache-set")
async def cache_set(
    key: str = Body(...),
    value: str = Body(...),
    redis: Redis | None = Depends(get_redis_session),
):
    if redis is None:
        logger.warning("Redis not available — skipping cache_set")
        return {"status": "skipped", "reason": "redis unavailable"}

    await redis.set(key, value)
    return {"status": "ok", "key": key, "value": value}


@app.get("/cache-get")
async def cache_get(
    key: str,
    redis: Redis | None = Depends(get_redis_session),
):
    if redis is None:
        logger.warning("Redis not available — skipping cache_get")
        return {"status": "skipped", "reason": "redis unavailable"}

    value = await redis.get(key)
    return {"key": key, "value": value.decode() if value else None}
