from typing import Any

try:
    import orjson as json

except ImportError:
    import json


from redis.asyncio.client import Redis

from config.settings import settings
from logger_config import get_logger
from model_manager import ModelManager

logger = get_logger(__name__)


async def handler_rerank(
    model_name: str, query: str, candidates: list[str], manager: ModelManager, redis: Redis | None
) -> dict[str, Any]:
    # Prepare input pairs
    pairs = [[query, c] for c in candidates]
    model = await manager.get_model(model_name)
    if not model:
        raise ValueError("Model not found")
    try:
        scores = await model.infer("predict", pairs, show_progress_bar=False)
        if scores is None:
            return {"scores": [], "sorted_candidates": [], "model_name": model_name}
        sorted_candidates = [c for _, c in sorted(zip(scores, candidates), reverse=True)]
        return {"scores": scores.tolist(), "sorted_candidates": sorted_candidates, "model_name": model_name}
    except Exception as e:
        raise ValueError(f"Failed to generate scores of rerank: {e}")


async def handler_rerank_cache(
    model_name: str, query: str, candidates: list[str], manager: ModelManager, redis: Redis | None
) -> dict[str, Any]:
    pairs = [[query, c] for c in candidates]
    pairs_keys = [f"{q}||{c}" for q, c in pairs]
    key = None
    if settings.embedding_cache_results and redis is not None:
        key = await manager.generate_hash_key(pairs_keys, "rerank", model_name)
        cached = await redis.get(key)
        if cached is not None:
            return {"scores": json.loads(cached), "model_name": model_name}
    model = await manager.get_model(model_name)
    if not model:
        raise ValueError("Model not found")
    try:
        scores = await model.infer("predict", pairs, show_progress_bar=False)
        if scores is None:
            return {"scores": [], "sorted_candidates": [], "model_name": model_name}
        sorted_candidates = [c for _, c in sorted(zip(scores, candidates), reverse=True)]
        if key is not None:
            await redis.set(key, json.dumps(sorted_candidates), ex=settings.embedding_cache_ttl)
        return {"scores": scores.tolist(), "sorted_candidates": sorted_candidates, "model_name": model_name}
    except Exception as e:
        raise ValueError(f"Failed to generate scores of rerank: {e}")
