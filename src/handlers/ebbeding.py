import hashlib
from typing import Any

import numpy as np
from redis.asyncio.client import Redis

from config.settings import settings
from model_manager import ModelManager


def get_text_key(text: str):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


async def handler_embedding(
    model_name: str, texts: list[str], manager: ModelManager, redis: Redis | None
) -> dict[str, Any]:
    key = None
    if settings.embedding_cache_results and redis is not None:
        key = await manager.generate_hash_key(texts, "embeddings", model_name)
        cached = await redis.get(key)
        if cached is not None:
            embeddings = np.frombuffer(cached, dtype=np.float32).reshape(len(texts), -1)
            dimensions = embeddings.shape[-1]
            return {"embeddings": embeddings, "dimensions": dimensions, "model_name": model_name}
    model = await manager.get_model(model_name)
    if not model:
        raise ValueError("Model not found")
    try:
        embeddings = await model.infer("encode", texts, convert_to_tensor=True, show_progress_bar=False)
        if embeddings is None:
            return {"embeddings": [], "dimensions": 0, "model_name": model_name}
        dimensions = embeddings.shape[-1]
        arr = embeddings.cpu().numpy()
        if key is not None:
            await redis.set(key, arr.tobytes(), ex=settings.embedding_cache_ttl)
        return {"embeddings": arr, "dimensions": dimensions, "model_name": model_name}
    except Exception as e:
        raise ValueError(f"Failed to generate embeddings: {e}")
