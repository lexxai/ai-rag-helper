from enum import StrEnum


class GpuTool(StrEnum):
    NVIDIA = "nvidia"
    ROCM = "rocm"


class GpuDevice(StrEnum):
    CUDA = "cuda"
    CPU = "cpu"
    MPS = "mps"
    NPU = "npu"


class GpuToolSMI(StrEnum):
    NVIDIA = "nvidia-smi"
    ROCM = "rocm-smi"
