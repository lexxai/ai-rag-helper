import os
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


try:
    import psutil

    def get_app_memory_usage() -> float:
        """Returns the memory usage of the current process in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)

except ImportError:

    def get_app_memory_usage() -> float:
        return 0
