from pydantic import BaseModel


class ModelLoadResponse(BaseModel):
    status: str
    message: str
