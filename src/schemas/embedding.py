from pydantic import BaseModel, Field

from config.settings import settings


class EmbeddingRequest(BaseModel):
    texts: list[str] = Field(examples=[["hello world", "this is a test"]])
    model: str | None = Field(default=None, examples=[settings.default_model_names["embed"]])
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
    dimensions: int = Field(examples=[384])
    model_name: str = Field(examples=["sentence-transformers/all-MiniLM-L6-v2"])
