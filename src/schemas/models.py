from pydantic import BaseModel, ConfigDict


class ModelLoadResponse(BaseModel):

    status: str
    message: str


class ModelListItems(BaseModel):
    model: str
    device: str
    last_used: float
    last_infer_time: float
    gpu_used_gb: float
