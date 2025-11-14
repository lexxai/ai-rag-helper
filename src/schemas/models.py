from pydantic import BaseModel, Field


class FilterParamsModelName(BaseModel):

    model_name: str = Field(description="Name of the model")


class ModelLoadResponse(BaseModel):

    status: str
    message: str


class ModelListItems(BaseModel):
    model_type: str
    model_name: str
    device: str
    last_used: float
    last_infer_time: float
    gpu_used_gb: float
    timeout: int = None
