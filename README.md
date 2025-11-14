AI RAG Helper
=================

High-level FastAPI service to help with RAG (Retrieval Augmented Generation) pipelines. It manages approved embedding models, provides an API to preload/unload models, compute embeddings with optional Redis caching, and exposes basic monitoring hooks. The project is designed to run locally with Python/uv or via Docker Compose and supports CPU, CUDA, and ROCm PyTorch builds via optional dependency extras.

## Contents
- Overview
- Features
- Quickstart
  - Prerequisites
  - Installation (uv)
  - Configuration (.env and models list)
  - Run locally
  - Run with Docker
- API Reference (summary)
- Settings (environment variables)
- Development

## Overview
The service wraps sentence-transformers/Hugging Face embedding models behind a simple API. It can: list approved models, preload them to a local cache, compute embeddings (optionally cached in Redis), and manage model lifecycle in memory.

## Features
- FastAPI-based API with JSON or ORJSON responses
- Embedding endpoint with batch support and Redis result caching
- Approved model allowlist via YAML file
- Model lifecycle management: load, unload, list loaded, list available, get properties
- Works with CPU/CUDA/ROCm PyTorch wheels (choose via extras)
- Docker Compose setup with Redis

## ModelManager

The `ModelManager` is the core component responsible for managing embedding and reranking models throughout their lifecycle. It provides:

### Key Responsibilities
- **Model Loading & Unloading**: Dynamically loads models on-demand and manages memory by unloading inactive models
- **Automatic Cleanup**: Background task that unloads models after a configurable timeout period of inactivity
- **GPU Monitoring**: Tracks GPU memory usage for NVIDIA (via nvidia-smi) and AMD ROCm devices
- **Prometheus Metrics**: Exposes model usage metrics including loaded model count, GPU memory per model, and inference times
- **Thread-Safe Operations**: Uses async locks to ensure safe concurrent access to models
- **Device Detection**: Automatically detects and uses available hardware (CPU, CUDA, or ROCm)

### How It Works
The ModelManager instantiates models using either `SentenceTransformer` (for embedding models) or `CrossEncoder` (for reranking models) from the sentence-transformers library. Each model is wrapped in a `ModelInstance` that tracks usage statistics and handles inference requests. Models are loaded lazily when first requested and can be preloaded to disk cache via the `/models/preload` endpoint.

### Configuration
ModelManager behavior is controlled by environment variables:
- `MODEL_MANAGER_TIMEOUT`: Seconds of inactivity before auto-unloading (default: 600)
- `GPU_MONITOR_LOOP_DELAY`: GPU monitoring interval in seconds (default: 5)
- `PROMETHEUS_LOOP_DELAY`: Prometheus metrics update interval (default: 15)
- `PRE_IMPORT_ON_BOOT`: Whether to import model libraries at startup (default: false)

The manager is injected into route handlers via FastAPI's dependency injection system using `get_model_manager()`.

## Quickstart

### Prerequisites
- Python 3.12+ (tested with 3.14 in Docker args)
- uv (Python package/dependency manager): https://docs.astral.sh/uv/
- Redis (local or via Docker; docker-compose.yaml provides one)

### Installation (uv)
1) Clone the repo and change directory into it.
2) Choose one extra for PyTorch (CPU is default):
   - cpu (default)
   - cu128 (CUDA 12.8)
   - rocm (ROCm 6.4; not supported on Windows)
   - docker (skip installing torch; useful when you rely on container-used with 'rocm-pytorch' image)

By default, the project’s uv configuration installs the `default_extras = ["ai-rag-helper[cpu]"]` group.

Sync dependencies:
```
uv sync
```

To switch extras explicitly:
```
# CPU
uv sync --extra cpu

# CUDA 12.8
uv sync --extra cu128

# ROCm (non-Windows)
uv sync --extra rocm

# No torch  (use Docker instead with 'rocm-pytorch' image)
uv sync --extra docker
```

### Configuration
1) Environment variables
   - Copy `dot_env.example` to `.env` and adjust values.
   - Important keys: `APP_PORT`, `API_ACCESS_KEY`, `REDIS_URL`, `HF_TOKEN`, etc. See Settings section below.

2) Approved models list
   - The service reads model allowlist and properties from `src/config/.models.yaml`.
   - An example file is provided: `src/config/dot.models_example.yaml`.
   - Create your config by copying and editing:
```
cp src/config/dot.models_example.yaml src/config/.models.yaml
```

### Run locally
Start the API with uvicorn:
```
uv run uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```
Then open the docs:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Run with Docker
The repository includes a Dockerfile and docker-compose.yaml.

Basic run (CPU by default):
```
docker compose up -d --build
docker compose logs -f api
```

### Notes
- The compose file starts a Redis service and the API. The API service mounts `./src` and `./data/models` into the container for live development and persisted model cache.
- Torch extra can be controlled via build-arg `EXTRA` (defaults to `cpu`).
- `.env` is passed into the container; set `APP_PORT` there (default 8000). The API will be available on http://localhost:8000
- For ROCm, see `docker-compose.rocm.yaml` and related comments under `dockers/` if present on your system/hardware.

## API Reference (summary)
Base prefix: `/api/v1`

### Models
- `GET /models/available` → list[str] of approved model names
- `GET /models/loaded` → list of loaded models
- `GET /models/load?model_name=...` → load a model
- `GET /models/unload?model_name=...` → unload a model
- `GET /models/properties?model_name=...` → return model properties (dimensions, max_tokens, batch_size, …)
- `GET /models/preload` → preload all available models to disk cache (requires API key; see auth dependency)

### Embedding
- `POST /embed/` → compute embeddings
  - Request body (`schemas/embedding.py`):
    ```json
    {
      "texts": ["hello", "world"],
      "model": "sentence-transformers/all-MiniLM-L6-v2",
      "batch_size": 32
    }
    ```
  - Response (`EmbeddingResponse`): vectors and metadata

### Rerank
- `POST /rerank/` → rerank documents based on query relevance
  - Request body (`schemas/rerank.py`):
    ```json
    {
      "query": "What is machine learning?",
      "candidates": ["ML is a subset of AI", "The sky is blue", "Neural networks learn patterns"],
      "model": "sentence-transformers/all-MiniLM-L6-v2"
    }
    ```
  - Response (`RerankResponse`): ranked candidates with relevance scores
  - Uses cross-encoder models to score query-document pairs for better relevance ranking in RAG pipelines

### Cache (requires API key dependency on router or endpoint)
- `POST /cache/set` with body `{ "key": "k", "value": "v" }`
- `GET /cache/get?key=...`

## Settings (environment variables)
Defined in `src/config/settings.py` (Pydantic BaseSettings). Key values include:
- `REDIS_URL` (env: `redis_url`) default: `redis://redis:6379/0`
- `LOG_LEVEL` (env: `log_level`) one of: DEBUG, INFO, WARNING, ERROR, CRITICAL
- `APP_NAME` (env: `app_name`) default: `AI RAG Helper`
- `APP_VERSION` (env: `app_version`) default: `0.0.1`
- `DEBUG` (env: `debug`) default: `false`
- `API_PREFIX` (env: `api_prefix`) default: `/api/v1`
- `CORS_ORIGINS` (env: `cors_origins`) default: `[*]`
- `API_ACCESS_KEY` (env: `api_access_key`) optional; when set, certain endpoints require it
- `MODEL_MANAGER_TIMEOUT` (env: `model_manager_timeout`) default: `600` seconds
- `GPU_MONITOR_LOOP_DELAY` (env: `gpu_monitor_loop_delay`) default: `5` seconds
- `PROMETHEUS_LOOP_DELAY` (env: `prometheus_loop_delay`) default: `15` seconds
- `PRE_IMPORT_ON_BOOT` (env: `pre_import_on_boot`) default: `false`
- `APPROVED_MODELS_CONFIG_PATH` (env: `approved_models_config_path`) default: `config/.models.yaml` (resolved under `src/`)
- `MODEL_CACHE_FOLDER` (env: `model_cache_folder`) default: `models` (resolved to `data/models` under repo root)
- `MODEL_CACHE_FOLDER_ONLY_LOCAL` (env: `model_cache_folder_only_local`) default: `false`
- `EMBEDDING_CACHE_RESULTS` (env: `embedding_cache_results`) default: `true`
- `EMBEDDING_CACHE_TTL` (env: `embedding_cache_ttl`) default: one week
- `HF_TOKEN` (env: `hf_token`) optional; set for private models or rate limits
- `DEFAULT_MODEL_NAMES` (env: `default_model_names`) JSON mapping, e.g. `{ "embed": "…", "rerank": "…" }`

## Development
- Code style: black, isort, and flake8 settings in `pyproject.toml` (line length 120)
- Linting/formatting: install dev tools with `uv sync --group dev`
- Pre-commit: `pre-commit install && pre-commit run -a`

### Testing
- There is an example HTTP test file: `tests/test_main.http` you can run with IDE HTTP client or REST tools.

### Project Layout
```
ai-rag-helper/
├─ src/
│  ├─ main.py                # FastAPI app, CORS, exception handlers, routers
│  ├─ lifespan.py            # app lifespan hooks
│  ├─ logger_config.py       # logging setup helpers
│  ├─ routers/
│  │  ├─ embedding.py        # /api/v1/embed endpoint
│  │  ├─ models.py           # /api/v1/models endpoints
│  │  └─ cache.py            # /api/v1/cache endpoints
│  ├─ config/
│  │  ├─ settings.py         # Pydantic BaseSettings and paths
│  │  ├─ models_list_config.py # YAML loader/validator for approved models
│  │  ├─ dot.models_example.yaml # example for .models.yaml
│  │  └─ .models.yaml        # your models config (ignored if not committed)
│  ├─ model_manager.py       # model lifecycle and cache mgmt
│  ├─ schemas/               # request/response models
│  └─ handlers/              # business logic for routers
├─ data/models/              # on-disk model cache (created on first run)
├─ docker-compose.yaml       # API + Redis
├─ docker-compose.rocm.yaml  # ROCm-oriented compose (if using AMD)
├─ Dockerfile
├─ dot_env.example           # copy to .env
├─ pyproject.toml            # dependencies & tooling; uv configuration
└─ uv.lock
```

## License
This repository’s license is not specified in this README. If you intend to open-source it, consider adding a LICENSE file.
