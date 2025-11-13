from enum import StrEnum, IntEnum


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


class BatchSize(IntEnum):
    CUDA = 128
    CPU = 32
    MPS = 64  # Apple Silicon
    NPU = 64  # e.g. Ascend, etc.

    @classmethod
    def from_device(cls, device: GpuDevice) -> int:
        try:
            return cls[device.name].value
        except KeyError:
            # Fallback for unknown devices
            return 32  # safe default
