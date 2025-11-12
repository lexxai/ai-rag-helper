import shutil

from constants import GpuToolSMI, GpuTool


# Singleton GPU detection
def detect_gpu_tool() -> str | None:
    """Detect GPU platform once."""
    if hasattr(detect_gpu_tool, "_cached"):
        return detect_gpu_tool._cached  # noqa
    if shutil.which(GpuToolSMI.ROCM):
        detect_gpu_tool._cached = GpuTool.ROCM
    elif shutil.which(GpuToolSMI.NVIDIA):
        detect_gpu_tool._cached = GpuTool.NVIDIA
    else:
        detect_gpu_tool._cached = None
    return detect_gpu_tool._cached  # noqa
