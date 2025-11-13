import asyncio
import gc
import hashlib
import subprocess
import time
from asyncio import Task
from functools import lru_cache
from typing import Literal

from prometheus_client import Gauge
from starlette.concurrency import run_in_threadpool

from constants import GpuTool, GpuDevice, GpuToolSMI
from config.models_list_config import APPROVED_MODELS, models_list_config
from events import EventBus
from config.settings import settings


from logger_config import get_logger
from utils import detect_gpu_tool

logger = get_logger(__name__)


class ModelInstance:
    def __init__(self, model_name: str, local_files_only: bool = None):
        self.model_name = model_name
        self.model = None
        self.last_used = time.time()
        self.last_infer_time = 0.0
        self.lock = asyncio.Lock()
        self.gpu_mem_gb = 0.0
        self.local_files_only = settings.model_cache_folder_only_local if local_files_only is None else local_files_only
        self.load_model()

    @classmethod
    @lru_cache(maxsize=1)
    def get_device(cls):
        try:
            logger.debug(f"Import torch & SentenceTransformer ...")
            import torch
            from sentence_transformers import SentenceTransformer

            return GpuDevice.CUDA if torch.cuda.is_available() else GpuDevice.CPU
        except ImportError:
            raise "Install torch dependency package"

    @property
    def device(self):
        return self.get_device()

    def load_model(self):
        if self.model is None:
            device = self.device
            from sentence_transformers import SentenceTransformer

            hf_token = getattr(settings, "hf_token", None)
            backend = "torch"
            truncate_dim = None
            self.model: SentenceTransformer = SentenceTransformer(
                self.model_name,
                cache_folder=settings.model_cache_folder,
                device=device,
                token=hf_token,
                local_files_only=self.local_files_only,
                backend=backend,  # noqa
                truncate_dim=truncate_dim,
            )

    async def infer(self, func, *args, **kwargs):
        async with self.lock:
            if isinstance(func, str):
                func = getattr(self.model, func, None)
            if not func or not callable(func):
                raise ValueError("Invalid function name")
            start = time.time()
            # result = await asyncio.to_thread(func, *args, **kwargs)
            result = await run_in_threadpool(func, *args, **kwargs)
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
    def __init__(self, timeout: int = None, model_type: str = None):
        self.models: dict[str, ModelInstance] = {}
        self.model_type = model_type or "hf"
        timeout = timeout or settings.model_manager_timeout
        self.lock = asyncio.Lock()
        self.bus = EventBus()
        self.gpu_tool = detect_gpu_tool()
        if settings.pre_import_on_boot:
            device = ModelInstance.get_device()
            logger.debug(f"Pre-imported modules. Detected: {device.name} ...")

        # Prometheus metrics.py
        self.model_count_gauge = Gauge("loaded_models_total", "Number of currently loaded models")
        self.model_gpu_gauge = Gauge("model_gpu_usage_gb", "Per-model GPU memory usage in GB", ["model"])
        self.model_infer_gauge = Gauge("model_last_infer_seconds", "Last inference duration per model", ["model"])
        self.total_gpu_gauge = Gauge("gpu_total_usage_gb", "Total GPU memory usage in GB")

        # Background tasks
        self.tasks: dict[str, Task] = {}
        if timeout:
            self.tasks["cleanup_loop"] = asyncio.create_task(self._cleanup_loop(timeout))
        if settings.gpu_monitor_loop_delay:
            self.tasks["gpu_monitor_loop"] = asyncio.create_task(
                self._gpu_monitor_loop(settings.gpu_monitor_loop_delay)
            )
        if settings.prometheus_loop_delay:
            self.tasks["prometheus_loop"] = asyncio.create_task(self._prometheus_loop(settings.prometheus_loop_delay))
        self._shutdown_event = asyncio.Event()

        # logger.debug(f"{self.gpu_tool=}")

    @staticmethod
    def check_model_name(model_name: str, model_type: str = "hf") -> bool:
        if model_name:
            return model_name in APPROVED_MODELS.get(model_type, {}).keys()
        return False

    @staticmethod
    def get_model_properties(model_name: str, model_type: str = "hf") -> dict | None:
        if not model_name:
            return None
        return APPROVED_MODELS.get(model_type, {}).get(model_name)

    async def get_model(
        self, model_name: str, model_type: str = None, local_files_only: bool = None
    ) -> ModelInstance | None:
        model_type = model_type or self.model_type
        if not self.check_model_name(model_name, model_type):
            logger.error(f"Loading model '{model_name}' is not approved for '{model_type}'")
            return None

        async with self.lock:
            if model_name not in self.models:
                logger.debug(f"Loading model {model_name}")
                try:
                    self.models[model_name] = ModelInstance(model_name, local_files_only)
                    self.models[model_name].last_used = time.time()
                    await self.bus.publish({"action": "loaded", "model": model_name})
                except Exception as e:
                    logger.error(f"get_model[{model_name}] is unsuccessfully. {e} ")
                    return None
            else:
                self.models[model_name].last_used = time.time()
            return self.models[model_name]

    async def unload_model(self, model_name: str) -> bool:
        if not (model_names := list(self.models.keys())):
            return False
        if model_name in model_names:
            self.models[model_name].unload()
            del self.models[model_name]
            logger.info(f"The model '{model_name}' is unloaded.")
            await self.bus.publish({"action": "unloaded", "model": model_name})
            return True
        return False

    async def list_loaded(self):
        data = []
        for name, inst in self.models.items():
            data.append(
                {
                    "model": name,
                    "device": str(inst.device),
                    "last_used": inst.last_used,
                    "last_infer_time": inst.last_infer_time,
                    "gpu_used_gb": inst.gpu_mem_gb,
                }
            )
        return data

    def list_available_models_name(self) -> list[str]:
        return models_list_config.get_model_names(self.model_type)

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

    async def _cleanup_loop(self, timeout: int):
        logger.info(f"Cleanup monitor is starting [{timeout}s] ...")
        while not self._shutdown_event.is_set():
            await asyncio.sleep(timeout)
            now = time.time()
            async with self.lock:
                for model_name in list(self.models.keys()):
                    m = self.models[model_name]
                    if (now - m.last_used) > timeout:
                        logger.info(f"Cleanup unloading the model '{model_name}' by timeout inactivity.")
                        await self.unload_model(model_name)
        logger.info("Cleanup monitor finished")

    async def _gpu_monitor_loop(self, gpu_monitor_loop_delay: int):
        logger.info(f"GPU monitor is starting [{gpu_monitor_loop_delay}s] ...")
        while not self._shutdown_event.is_set():
            await asyncio.sleep(gpu_monitor_loop_delay)
            async with self.lock:
                if self.gpu_tool is not None and self.gpu_tool == GpuTool.NVIDIA:
                    try:
                        output = subprocess.check_output(
                            [
                                str(GpuToolSMI.NVIDIA),
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
                        output = subprocess.check_output([str(GpuToolSMI.ROCM), "--showuse", "--json"], text=True)
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

        logger.info("GPU monitor finished")

    async def _prometheus_loop(self, prometheus_loop_delay: int):
        logger.info(f"Prometheus monitor is starting [{prometheus_loop_delay}s] ...")
        while not self._shutdown_event.is_set():
            await asyncio.sleep(prometheus_loop_delay)
            async with self.lock:
                self.model_count_gauge.set(len(self.models))
                total_gpu = 0.0
                for name, m in self.models.items():
                    self.model_gpu_gauge.labels(model=name).set(m.gpu_mem_gb)
                    self.model_infer_gauge.labels(model=name).set(m.last_infer_time)
                    total_gpu += m.gpu_mem_gb
                self.total_gpu_gauge.set(total_gpu)
        logger.info("Prometheus monitor finished")

    async def preload_available_models(self) -> list[str]:
        result = []
        for model_name in self.list_available_models_name():
            model = await self.get_model(model_name, local_files_only=True)
            if model:
                result.append(model_name)
                await self.unload_model(model_name)
        return result

    @staticmethod
    def get_key(texts: list[str] | str, prefix: str = "hf") -> str:
        joined = "||".join(texts) if isinstance(texts, list) else texts
        hashed = hashlib.sha256(joined.encode("utf-8")).hexdigest()
        return f"{prefix}:{hashed}"
