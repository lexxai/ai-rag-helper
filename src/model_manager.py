import asyncio
import gc
import hashlib
import subprocess
import time
from asyncio import Task
from functools import lru_cache

try:
    from prometheus_client import Gauge
except ImportError:
    Gauge = None
from starlette.concurrency import run_in_threadpool

from config.models_list_config import APPROVED_MODELS, models_list_config
from config.settings import settings
from constants import GpuTool, GpuDevice, GpuToolSMI, BatchSize
from events import EventBus
from logger_config import get_logger
from utils import detect_gpu_tool, get_app_memory_usage

logger = get_logger(__name__)


class ModelInstance:
    def __init__(self, model_name: str, model_class_name: str, local_files_only: bool = None):
        self.model_name = model_name
        self.model_class_name = model_class_name
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
            logger.debug("Import torch ...")
            import torch

            # from sentence_transformers import SentenceTransformer, CrossEncoder  # noqa F401

            return GpuDevice.CUDA if torch.cuda.is_available() else GpuDevice.CPU
        except ImportError:
            raise "Install [torch, sentence_transformers] dependency package"

    @property
    def device(self):
        return self.get_device()

    def load_model(self):
        if self.model is None:
            device = self.device
            model_class = None
            hf_token = getattr(settings, "hf_token", None)
            backend = "torch"
            truncate_dim = None
            model_params = {
                "model_name_or_path": self.model_name,
                "cache_folder": settings.model_cache_folder,
                "device": device,
                "token": hf_token,
                "local_files_only": self.local_files_only,
                "backend": backend,  # noqa
            }
            match self.model_class_name:
                case "sentence-transformers":
                    from sentence_transformers import SentenceTransformer

                    model_class = SentenceTransformer
                    model_params["truncate_dim"] = truncate_dim

                case "cross-encoder":
                    from sentence_transformers import CrossEncoder

                    model_class = CrossEncoder

            if model_class is None:
                raise ValueError(f"Invalid model class name: {model_class}/{self.model_class_name}")

            self.model: SentenceTransformer | CrossEncoder = model_class(**model_params)

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
        self.model_count_gauge = Gauge("loaded_models_total", "Number of currently loaded models") if Gauge else None
        self.model_gpu_gauge = (
            Gauge("model_gpu_usage_gb", "Per-model GPU memory usage in GB", ["model"]) if Gauge else None
        )
        self.model_infer_gauge = (
            Gauge("model_last_infer_seconds", "Last inference duration per model", ["model"]) if Gauge else None
        )
        self.total_gpu_gauge = Gauge("gpu_total_usage_gb", "Total GPU memory usage in GB") if Gauge else None

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
        self.app_memory_usage = get_app_memory_usage() if get_app_memory_usage is not None else 0.0

        # logger.debug(f"{self.gpu_tool=}")

    @staticmethod
    def check_model_name(model_name: str) -> bool:
        """
        Validates the provided model name by stripping any surrounding quotes or
        extra spaces and checking it against the configuration for allowed model
        names.

        :param model_name: The name of the model to validate, provided as a string.
        :return: True if the model name is valid and matches the configuration,
            otherwise False.
        """
        model_name = model_name.strip('"').strip("'").strip()
        if not model_name:
            return False
        return models_list_config.validate_model_name(model_name)

    @staticmethod
    def decode_model_name(model_name: str) -> tuple[str, str, str]:
        """
        Decodes a given model name string into its components by splitting it based on a
        specified delimiter and extracting the model type and model class name.

        If the given model name contains the delimiter ':', it splits the name into a
        model type and a model name. Otherwise, it uses a default model type. This method
        also decodes the model class name using an internal decoder method.

        :param model_name: The model name string to be decoded.
        :type model_name: str
        :return: A tuple containing the model type, the model name, and the decoded
                 model class name.
        :rtype: tuple[str, str, str]
        """
        model_name = model_name.strip('"').strip("'").strip()
        return models_list_config.decode_model_name(model_name)

    @staticmethod
    def get_model_properties(model_name: str, model_type: str = "hf") -> dict | None:
        if not model_name:
            return None
        return APPROVED_MODELS.get(model_type, {}).get(model_name)

    async def get_model(self, model_name: str, local_files_only: bool = None) -> ModelInstance | None:
        if not self.check_model_name(model_name):
            logger.error(f"Loading model '{model_name}' is not approved")
            return None

        model_name = model_name.strip('"').strip("'").strip()
        async with self.lock:
            if model_name not in self.models:
                logger.debug(f"Loading model {model_name}")
                try:
                    model_type, model_name_strip, model_class_name = self.decode_model_name(model_name)
                    self.models[model_name] = ModelInstance(
                        model_name=model_name_strip,
                        model_class_name=model_class_name,
                        local_files_only=local_files_only,
                    )
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
        if not self.check_model_name(model_name):
            return False

        model_name = model_name.strip('"').strip("'").strip()
        if model_name in model_names:
            self.models[model_name].unload()
            del self.models[model_name]
            logger.info(f"The model '{model_name}' is unloaded.")
            await self.bus.publish({"action": "unloaded", "model": model_name})
            return True
        return False

    async def list_loaded(self):
        data = []
        for model_name, inst in self.models.items():
            model_type, model_name_strip = self.decode_model_name(model_name)
            information = {
                "model_type": model_type,
                "model_name": model_name_strip,
                "device": str(inst.device),
                "last_used": inst.last_used,
                "last_infer_time": inst.last_infer_time,
                "gpu_used_gb": inst.gpu_mem_gb,
            }
            if settings.model_manager_timeout > 0:
                information["timeout"] = int(settings.model_manager_timeout - (time.time() - inst.last_used))
            data.append(information)
        return data

    def list_available_model_names(self, model_type: str = None) -> list[str]:
        model_type = model_type or self.model_type
        return models_list_config.get_model_names(model_type)

    @staticmethod
    def list_available_model_types():
        return models_list_config.get_model_types()

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
                if get_app_memory_usage is not None:
                    self.app_memory_usage = get_app_memory_usage()
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
                    logger.info("No supported GPU monitoring device (NVIDIA/ROCm) was found")
                    break

        logger.info("GPU monitor finished")

    async def _prometheus_loop(self, prometheus_loop_delay: int):
        if Gauge is None and get_app_memory_usage is None:
            logger.info(
                "Prometheus monitor finished: required monitoring packages (prometheus_client and psutil) are not installed"
            )
            return
        logger.info(f"Prometheus monitor is starting [{prometheus_loop_delay}s] ...")
        while not self._shutdown_event.is_set():
            await asyncio.sleep(prometheus_loop_delay)
            self.app_memory_usage = get_app_memory_usage()
            if Gauge is None:
                continue
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
        for model_name in self.list_available_model_names():
            model = await self.get_model(model_name, local_files_only=True)
            if model:
                result.append(model_name)
                await self.unload_model(model_name)
        return result

    @staticmethod
    async def generate_hash_key(texts: list[str] | str, prefix: str = "hf", model_name: str = "") -> str:
        joined = model_name + ("||".join(texts) if isinstance(texts, list) else texts)
        hashed = await run_in_threadpool(hashlib.sha256, joined.encode("utf-8"))
        return f"{prefix}:{hashed.hexdigest()}"

    @staticmethod
    def get_batch_size(model_name: str | None = None) -> int:
        # 1. Model-specific override
        if model_name is not None:
            props = models_list_config.get_model_properties(model_name)
            model_bs = props.get("batch_size")
            if model_bs is not None:
                return int(model_bs)  # ensure int

        # 2. Device-based default
        device = ModelInstance.get_device()  # returns GpuDevice
        return BatchSize.from_device(device)
