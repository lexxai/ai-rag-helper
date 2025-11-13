from pydantic import BaseModel


class EmbeddingRequest(BaseModel):
    texts: list[str]
    model: str | None = None  # optional override
