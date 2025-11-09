import asyncio
import gc
import time
import shutil
import subprocess
from sentence_transformers import SentenceTransformer
from typing import Optional
from prometheus_client import Gauge

try:
    import torch
except ImportError:
    raise "Install torch dependecy package"
from contextlib import asynccontextmanager
from redis.asyncio import Redis

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
def detect_gpu_tool() -> Optional[str]:
    """Detect GPU platform once."""
    if hasattr(detect_gpu_tool, "_cached"):
        return detect_gpu_tool._cached
    if shutil.which("nvidia-smi"):
        detect_gpu_tool._cached = "nvidia"
    elif shutil.which("rocm-smi"):
        detect_gpu_tool._cached = "rocm"
    else:
        detect_gpu_tool._cached = None
    return detect_gpu_tool._cached


class ModelInstance:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.last_used = time.time()
        self.last_infer_time = 0.0
        self.lock = asyncio.Lock()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gpu_mem_gb = 0.0

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
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"✅ Unloaded model {self.model_name}")
        except Exception as e:
            print(f"⚠️ Error unloading {self.model_name}: {e}")


class ModelManager:
    def __init__(self, timeout: int = 600, check_gpu: bool = True):
        self.models: dict[str, ModelInstance] = {}
        self.timeout = timeout
        self.lock = asyncio.Lock()
        self.bus = EventBus()
        self.check_gpu = check_gpu
        self.gpu_tool = detect_gpu_tool()

        # Prometheus metrics
        self.model_count_gauge = Gauge(
            "loaded_models_total", "Number of currently loaded models"
        )
        self.model_gpu_gauge = Gauge(
            "model_gpu_usage_gb", "Per-model GPU memory usage in GB", ["model"]
        )
        self.model_infer_gauge = Gauge(
            "model_last_infer_seconds", "Last inference duration per model", ["model"]
        )
        self.total_gpu_gauge = Gauge(
            "gpu_total_usage_gb", "Total GPU memory usage in GB"
        )

        asyncio.create_task(self._cleanup_loop())
        if self.check_gpu:
            asyncio.create_task(self._gpu_monitor_loop())
        asyncio.create_task(self._prometheus_loop())

    async def get_model(self, model_name: str) -> ModelInstance:
        async with self.lock:
            if model_name not in self.models:
                print(f"🚀 Loading model {model_name}")
                self.models[model_name] = ModelInstance(model_name)
                await self.bus.publish({"action": "loaded", "model": model_name})
            else:
                self.models[model_name].last_used = time.time()
            return self.models[model_name]

    async def unload_model(self, model_name: str):
        async with self.lock:
            if model_name in self.models:
                self.models[model_name].unload()
                del self.models[model_name]
                await self.bus.publish({"action": "unloaded", "model": model_name})

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

    async def _cleanup_loop(self):
        while True:
            await asyncio.sleep(60)
            now = time.time()
            async with self.lock:
                for name in list(self.models.keys()):
                    m = self.models[name]
                    if now - m.last_used > self.timeout:
                        await self.unload_model(name)

    async def _gpu_monitor_loop(self):
        while True:
            await asyncio.sleep(5)
            async with self.lock:
                if self.gpu_tool == "nvidia":
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
                        print(f"⚠️ NVIDIA GPU monitor failed: {e}")
                elif self.gpu_tool == "rocm":
                    try:
                        output = subprocess.check_output(
                            ["rocm-smi", "--showuse", "--json"], text=True
                        )
                        import json

                        rocm_data = json.loads(output)
                        total_mem = (
                            sum(int(gpu["VRAMUse"]) for gpu in rocm_data["card"]) / 1024
                        )
                        for m in self.models.values():
                            m.gpu_mem_gb = total_mem / max(1, len(self.models))
                    except Exception as e:
                        print(f"⚠️ ROCm GPU monitor failed: {e}")
                else:
                    for m in self.models.values():
                        m.gpu_mem_gb = 0.0

    async def _prometheus_loop(self):
        while True:
            await asyncio.sleep(5)
            async with self.lock:
                self.model_count_gauge.set(len(self.models))
                total_gpu = 0.0
                for name, m in self.models.items():
                    self.model_gpu_gauge.labels(model=name).set(m.gpu_mem_gb)
                    self.model_infer_gauge.labels(model=name).set(m.last_infer_time)
                    total_gpu += m.gpu_mem_gb
                self.total_gpu_gauge.set(total_gpu)
