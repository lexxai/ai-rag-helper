import hashlib
from typing import Any

import numpy as np
from redis.asyncio.client import Redis

from config.settings import settings
from logger_config import get_logger
from model_manager import ModelManager

logger = get_logger(__name__)


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


async def handler_embedding_cache(
    model_name: str,
    texts: list[str],
    manager: ModelManager,
    redis: Redis | None,
    max_batch_size: int = None,
) -> dict[str, Any]:
    if not texts:
        return {"embeddings": [], "dimensions": 0, "model_name": model_name}

    keys = None
    to_encode: list[str] = []
    encode_indices: list[int] = []
    cached_embeddings: list[np.ndarray | None] = [None] * len(texts)
    dimensions = 0

    # ============================================================
    # 1. Check Redis cache (if enabled)
    # ============================================================
    if settings.embedding_cache_results and redis is not None:
        keys = [await manager.generate_hash_key(t, "embeddings", model_name) for t in texts]
        cached_bytes = await redis.mget(keys)

        for i, (text, cache_hit) in enumerate(zip(texts, cached_bytes)):
            if cache_hit is not None:
                emb = np.frombuffer(cache_hit, dtype=np.float32)
                cached_embeddings[i] = emb
                if dimensions == 0:
                    dimensions = emb.shape[-1]
            else:
                to_encode.append(text)
                encode_indices.append(i)

        # All cached → skip model entirely
        if not to_encode:
            return {
                "embeddings": [e for e in cached_embeddings],
                "dimensions": dimensions,
                "model_name": model_name,
            }

    # ============================================================
    # 2. Load model
    # ============================================================
    model = await manager.get_model(model_name)
    if not model:
        raise ValueError("Model not found")

    max_batch_size = max_batch_size or manager.get_batch_size(model_name)
    logger.debug(f"Using batch size {max_batch_size} for model {model_name}")

    # ============================================================
    # 3. Encode in chunks (batched)
    # ============================================================
    new_embeddings: list[np.ndarray] = []
    new_keys_to_cache: list[str] = []
    # TODO bug with max_batch_size if less
    try:
        for start_idx in range(0, len(to_encode), max_batch_size):
            logger.debug(f"Using batch idx: {start_idx} for embedding")
            end_idx = start_idx + max_batch_size
            batch_texts: list[str] = to_encode[start_idx:end_idx]
            batch_indices = encode_indices[start_idx:end_idx]
            batch_keys: list[str] | None = [keys[i] for i in batch_indices] if keys else None

            # Model inference (one batch)
            batch_tensor = await model.infer("encode", batch_texts, convert_to_tensor=True, show_progress_bar=False)

            if batch_tensor is None or batch_tensor.numel() == 0:
                batch_arr = np.array([], dtype=np.float32).reshape(0, 0)
            else:
                batch_arr = batch_tensor.cpu().numpy()  # (batch_size, dim)
                if dimensions == 0:
                    dimensions = batch_arr.shape[-1]
                elif batch_arr.shape[-1] != dimensions:
                    raise ValueError("Embedding dimension mismatch across batches")

            # Store results in correct positions
            for emb, idx, key in zip(batch_arr, batch_indices, batch_keys or []):
                cached_embeddings[idx] = emb
                new_embeddings.append(emb)
                if key:
                    new_keys_to_cache.append(key)

            # Optional: cache this batch immediately (or defer)
            if batch_keys and redis is not None:
                to_redis = {k: a.tobytes() for k, a in zip(batch_keys, batch_arr)}
                await redis.mset(to_redis)
                async with redis.pipeline(transaction=True) as pipe:
                    for k in batch_keys:
                        await pipe.expire(k, settings.embedding_cache_ttl)
                    await pipe.execute()

        # ============================================================
        # 4. Final assembly
        # ============================================================
        final_embeddings = []
        for emb in cached_embeddings:
            if emb is None:
                # Fallback: zero vector of correct dim
                emb = np.zeros(dimensions, dtype=np.float32) if dimensions > 0 else np.array([])
            final_embeddings.append(emb)

        return {
            "embeddings": final_embeddings,
            "dimensions": dimensions,
            "model_name": model_name,
        }

    except Exception as e:
        raise ValueError(f"Failed to generate embeddings: {e}")
