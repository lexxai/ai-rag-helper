from model_manager import ModelManager


async def handler_embedding(model_name: str, texts: list[str], manager: ModelManager) -> dict[str, list[list[float]]]:
    model = await manager.get_model(model_name)
    if not model:
        raise ValueError("Model not found")
    embeddings = await model.infer("encode", texts, convert_to_tensor=True, show_progress_bar=False)
    return {"embeddings": embeddings.cpu().tolist()}
