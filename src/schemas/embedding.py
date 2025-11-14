from pydantic import BaseModel, Field

from config.models_list_config import models_list_config
from config.settings import settings

example_model_name = settings.default_model_names["embed"]
example_model_dimensions = models_list_config.get_model_properties(example_model_name).get("dimensions", 384)


class EmbeddingRequest(BaseModel):
    texts: list[str] = Field(examples=[["hello world", "this is a test"]])
    model: str | None = Field(default=None, examples=[example_model_name])
    dimensions: int | None = Field(default=None, description="Number of dimensions to return")
    batch_size: int | None = Field(default=None, description="Batch size to use for inference")


class EmbeddingResponse(BaseModel):
    embeddings: list[list[float]] = Field(
        examples=[
            [
                [-0.03447727486491203, 0.03102317824959755, 0.006734970025718212, "...", 0.030206844210624695],
                [0.0306123998016119, 0.013831381686031818, -0.020843738690018654, "...", -0.04484280198812485],
            ]
        ]
    )
    dimensions: int = Field(examples=[example_model_dimensions])
    model_name: str = Field(examples=[example_model_name])
