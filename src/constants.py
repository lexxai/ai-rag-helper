from enum import StrEnum


class GpuTool(StrEnum):
    NVIDIA = "nvidia"
    ROCM = "rocm"


class GpuDevice(StrEnum):
    CUDA = "cuda"
    CPU = "cpu"


APPROVED_MODELS = {"hf": ["sentence-transformers/paraphrase-multilingual-mpnet-base-v2"]}
