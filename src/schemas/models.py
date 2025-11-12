from pydantic import BaseModel, ConfigDict


class ModelLoadResponse(BaseModel):

    status: str
    message: str
