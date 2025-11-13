from pydantic import BaseModel, Field

from config.settings import settings


class EmbeddingRequest(BaseModel):
    texts: list[str] = Field(examples=[["hello world", "this is a test"]])
    model: str | None = Field(default=None, examples=[settings.default_model_names["embed"]])


class EmbeddingResponse(BaseModel):
    embeddings: list[list[float]] = Field(
        examples=[[[0.000001, 0.000002, "...", 0.002001], [0.000003, 0.000004, "...", 0.005001]]]
    )
    dimensions: int = Field(examples=[384])
    model_name: str = Field(examples=["sentence-transformers/all-MiniLM-L6-v2"])
