from fastapi import APIRouter, Depends, Body
from redis.asyncio import Redis

from depends.databases import get_redis_session
from depends.auth import check_api_key
from logger_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/cache", tags=["cache"], dependencies=[Depends(check_api_key)])


@router.post("/set")
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


@router.get("/get")
async def cache_get(key: str, redis: Redis | None = Depends(get_redis_session)):
    if redis is None:
        logger.warning("Redis not available — skipping cache_get")
        return {"status": "skipped", "reason": "redis unavailable"}

    value = await redis.get(key)
    return {"key": key, "value": value.decode() if value else None}
