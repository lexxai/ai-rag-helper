try:
    import orjson as json

except ImportError:
    import json

from pathlib import Path
from typing import Literal, Annotated, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_PATH = Path(__file__).parent.parent


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables and .env file.
    """

    model_config = SettingsConfigDict(
        env_file=BASE_PATH.parent / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow",
        arbitrary_types_allowed=True,
    )

    # Redis Configuration
    redis_url: str = "redis://redis:6379/0"

    # Logging Configuration
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - " "%(filename)s:%(lineno)d - %(message)s"

    # Application Configuration
    app_name: str = "AI RAG Helper"
    app_version: str = "0.0.1"
    debug: bool = False

    # API Configuration
    api_prefix: str = "/api/v1"
    cors_origins: list[str] = ["*"]
    api_access_key: str | None = None

    # These settings accept either None or positive integers from environment variables
    model_manager_timeout: int = Field(
        default=600,
        description="Delay in seconds to check inactive models. Use 0 to disable.",
        ge=0,
    )
    gpu_monitor_loop_delay: int = Field(
        default=5,
        description="Delay in seconds to gpu monitoring. Use 0 to disable.",
        ge=0,
    )
    prometheus_loop_delay: int = Field(
        default=15,
        description="Delay in seconds to prometheus monitoring. Use 0 to disable.",
        ge=0,
    )

    pre_import_on_boot: bool = False  # For preload pytorch module on boot
    approved_models_config_path: Path = Path("config/.models.yaml")
    model_cache_folder: Path = Path("models")
    model_cache_only_local: bool = False

    hf_token: str | None = None

    default_model_names: dict[str, str] = {
        "embed": "sentence-transformers/all-MiniLM-L6-v2",
        "rerank": "sentence-transformers/all-MiniLM-L6-v2",
    }

    @field_validator("default_model_names", mode="before")
    @classmethod
    def parse(cls, v):
        if not v:
            return {"embed": "all-MiniLM-L6-v2", "rerank": "all-MiniLM-L6-v2"}
        if isinstance(v, dict):
            return v
        try:
            return json.loads(v)
        except:
            return {"embed": "all-MiniLM-L6-v2"}

    def __init__(self):
        super().__init__()
        cache_path = (BASE_PATH.parent / "data" / self.model_cache_folder).resolve()
        if not cache_path.is_relative_to(BASE_PATH.parent):
            raise ValueError("Model cache folder must be within the base directory")
        self.model_cache_folder = cache_path
        self.model_cache_folder.mkdir(exist_ok=True, parents=True)
        approved_models_config_path = (BASE_PATH / self.approved_models_config_path).resolve()
        if not approved_models_config_path.is_relative_to(BASE_PATH):
            raise ValueError("Model approved_models_config_path folder must be within the base directory")
        self.approved_models_config_path = approved_models_config_path
        print(self.default_model_names)


# Singleton instance
settings = Settings()
