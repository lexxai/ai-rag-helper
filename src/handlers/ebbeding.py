from typing import Any

from model_manager import ModelManager


async def handler_embedding(model_name: str, texts: list[str], manager: ModelManager) -> dict[str, Any]:
    model = await manager.get_model(model_name)
    if not model:
        raise ValueError("Model not found")
    try:
        embeddings = await model.infer("encode", texts, convert_to_tensor=True, show_progress_bar=False)
        if embeddings is None:
            return {"embeddings": [], "dimensions": 0, "model_name": model_name}
        dimensions = embeddings.shape[-1]
        return {"embeddings": embeddings.cpu().tolist(), "dimensions": dimensions, "model_name": model_name}
    except Exception as e:
        raise ValueError(f"Failed to generate embeddings: {e}")
