import asyncio
import gc
import shutil
import subprocess
import time
from asyncio import Task
from functools import lru_cache

from prometheus_client import Gauge

from constants import GpuTool, GpuDevice, APPROVED_MODELS
from settings import settings


from logger_config import get_logger

logger = get_logger(__name__)


class EventBus:
    """Simple async pub/sub bus."""

    def __init__(self):
        self.listeners: set[asyncio.Queue] = set()

    async def subscribe(self) -> asyncio.Queue:
        q = asyncio.Queue()
        self.listeners.add(q)
        return q

    async def unsubscribe(self, q: asyncio.Queue):
        self.listeners.discard(q)

    async def publish(self, event: dict):
        for q in list(self.listeners):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass


# Singleton GPU detection
def detect_gpu_tool() -> str | None:
    """Detect GPU platform once."""
    if hasattr(detect_gpu_tool, "_cached"):
        return detect_gpu_tool._cached  # noqa
    if shutil.which("nvidia-smi"):
        detect_gpu_tool._cached = GpuTool.NVIDIA
    elif shutil.which("rocm-smi"):
        detect_gpu_tool._cached = GpuTool.ROCM
    else:
        detect_gpu_tool._cached = None
    return detect_gpu_tool._cached  # noqa


class ModelInstance:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.last_used = time.time()
        self.last_infer_time = 0.0
        self.lock = asyncio.Lock()
        self.gpu_mem_gb = 0.0
        self.load_model()

    @property
    @lru_cache(maxsize=1)
    def device(self):
        try:
            import torch

        except ImportError:
            raise "Install torch dependency package"

        return GpuDevice.CUDA if torch.cuda.is_available() else GpuDevice.CPU

    def load_model(self):
        if self.model is None:
            device = self.device
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(
                self.model_name, cache_folder=settings.model_cache_folder, device=device, local_files_only=True
            )

    async def infer(self, func, *args, **kwargs):
        async with self.lock:
            start = time.time()
            result = await asyncio.to_thread(func, *args, **kwargs)
            self.last_used = time.time()
            self.last_infer_time = time.time() - start
            return result

    def unload(self):
        try:
            del self.model
            gc.collect()
            if self.device == GpuDevice.CUDA:
                torch.cuda.empty_cache()  # noqa
            logger.debug(f"Unloaded model {self.model_name}")
        except Exception as e:
            logger.error(f"Error unloading {self.model_name}: {e}")


class ModelManager:
    def __init__(self, timeout: int = None, check_gpu: bool = True):
        self.models: dict[str, ModelInstance] = {}
        self.timeout = timeout or settings.model_manager_timeout
        self.lock = asyncio.Lock()
        self.bus = EventBus()
        self.check_gpu = check_gpu
        self.gpu_tool = detect_gpu_tool()

        # Prometheus metrics.py
        self.model_count_gauge = Gauge("loaded_models_total", "Number of currently loaded models")
        self.model_gpu_gauge = Gauge("model_gpu_usage_gb", "Per-model GPU memory usage in GB", ["model"])
        self.model_infer_gauge = Gauge("model_last_infer_seconds", "Last inference duration per model", ["model"])
        self.total_gpu_gauge = Gauge("gpu_total_usage_gb", "Total GPU memory usage in GB")

        # Background tasks
        self.tasks: dict[str, Task] = {"cleanup_loop": asyncio.create_task(self._cleanup_loop())}
        if self.check_gpu:
            self.tasks["gpu_monitor_loop"] = asyncio.create_task(self._gpu_monitor_loop())
        self.tasks["prometheus_loop"] = asyncio.create_task(self._prometheus_loop())
        self._shutdown_event = asyncio.Event()

        # logger.debug(f"{self.gpu_tool=}")

    @staticmethod
    def check_model_name(model_name: str, model_type: str = "hf") -> bool:
        if model_name:
            return model_name in APPROVED_MODELS.get(model_type, [])
        return False

    async def get_model(self, model_name: str, model_type: str = "hf") -> ModelInstance | None:
        if not self.check_model_name(model_name, model_type):
            logger.error(f"Loading model '{model_name}' is not approved for '{model_type}'")
            return None

        async with self.lock:
            if model_name not in self.models:
                logger.debug(f"Loading model {model_name}")
                try:
                    self.models[model_name] = ModelInstance(model_name)
                    self.models[model_name].last_used = time.time()
                    await self.bus.publish({"action": "loaded", "model": model_name})
                except Exception as e:
                    logger.error(f"get_model[{model_name}] is unsuccessfully. {e} ")
                    return None
            else:
                self.models[model_name].last_used = time.time()
            return self.models[model_name]

    async def unload_model(self, model_name: str) -> bool:
        async with self.lock:
            if model_name in list(self.models.keys()):
                print("unload_model model_name", model_name)
                self.models[model_name].unload()
                del self.models[model_name]
                logger.info(f"Model '{model_name}' is unloaded.")
                await self.bus.publish({"action": "unloaded", "model": model_name})
                return True
        return False

    async def list_loaded(self):
        data = []
        for name, inst in self.models.items():
            data.append(
                {
                    "model": name,
                    "device": inst.device,
                    "last_used": inst.last_used,
                    "last_infer_time": inst.last_infer_time,
                    "gpu_used_gb": inst.gpu_mem_gb,
                }
            )
        return data

    async def close(self):
        # Signal shutdown to all loops
        self._shutdown_event.set()

        # Cancel all background tasks
        for task_name, task in self.tasks.items():
            if not task.done():
                task.cancel()
                logger.debug(f"Cancelling task: {task_name}")

        # Wait for all tasks to complete (with cancellation)
        if self.tasks:
            await asyncio.gather(*self.tasks.values(), return_exceptions=True)
            logger.debug("All background tasks stopped")

        # Unload all models
        for name in list(self.models.keys()):
            await self.unload_model(name)

        await self.bus.publish({"action": "model manager closed"})

    async def _cleanup_loop(self):
        logger.debug(f"Cleanup monitor is starting [{self.timeout}s] ...")
        while not self._shutdown_event.is_set():
            await asyncio.sleep(self.timeout)
            now = time.time()
            async with self.lock:
                for name in list(self.models.keys()):
                    # TODO check it
                    print("_cleanup_loop", name)
                    m = self.models[name]
                    if (now - m.last_used) > self.timeout:
                        print("_cleanup_loop unload_model", name)
                        await self.unload_model(name)
        logger.debug("Cleanup monitor finished")

    async def _gpu_monitor_loop(self):
        logger.debug(f"GPU monitor is starting [{settings.gpu_monitor_loop_delay}s] ...")
        while not self._shutdown_event.is_set():
            await asyncio.sleep(settings.gpu_monitor_loop_delay)
            async with self.lock:
                if self.gpu_tool is not None and self.gpu_tool == GpuTool.NVIDIA:
                    try:
                        output = subprocess.check_output(
                            [
                                "nvidia-smi",
                                "--query-gpu=memory.used",
                                "--format=csv,noheader,nounits",
                            ],
                            text=True,
                        )
                        used = float(output.strip().splitlines()[0]) / 1024  # MB → GB
                        for m in self.models.values():
                            m.gpu_mem_gb = used / max(1, len(self.models))
                    except Exception as e:
                        logger.error(f"NVIDIA GPU monitor failed: {e}")
                elif self.gpu_tool is not None and self.gpu_tool == GpuTool.ROCM:
                    try:
                        output = subprocess.check_output(["rocm-smi", "--showuse", "--json"], text=True)
                        import json

                        rocm_data = json.loads(output)
                        total_mem = sum(int(gpu["VRAMUse"]) for gpu in rocm_data["card"]) / 1024
                        for m in self.models.values():
                            m.gpu_mem_gb = total_mem / max(1, len(self.models))
                    except Exception as e:
                        logger.error(f"ROCm GPU monitor failed: {e}")
                else:
                    for m in self.models.values():
                        m.gpu_mem_gb = 0.0
                    logger.debug("GPU monitor value of 'gpu_tool' is unsupported, break")
                    break

        logger.debug("GPU monitor finished")

    async def _prometheus_loop(self):
        logger.debug(f"Prometheus monitor is starting [{settings.prometheus_loop_delay}s] ...")
        while not self._shutdown_event.is_set():
            await asyncio.sleep(settings.prometheus_loop_delay)
            async with self.lock:
                self.model_count_gauge.set(len(self.models))
                total_gpu = 0.0
                for name, m in self.models.items():
                    self.model_gpu_gauge.labels(model=name).set(m.gpu_mem_gb)
                    self.model_infer_gauge.labels(model=name).set(m.last_infer_time)
                    total_gpu += m.gpu_mem_gb
                self.total_gpu_gauge.set(total_gpu)
        logger.debug("Prometheus monitor finished")
