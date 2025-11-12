from pathlib import Path
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_PATH = Path(__file__).parent


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables and .env file.
    """

    model_config = SettingsConfigDict(
        env_file=BASE_PATH.parent / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Redis Configuration
    redis_url: str = "redis://redis:6379/0"
    redis_max_connections: int = 10

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

    model_manager_timeout: int = 600
    model_manager_check_gpu: bool = True

    gpu_monitor_loop_delay: int = 5
    prometheus_loop_delay: int = 15

    pre_import_on_boot: bool = True
    approved_models_config_path: Path = Path("models.yaml")
    model_cache_folder: Path = Path("models")

    huggingface_api_key: str = None

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


# Singleton instance
settings = Settings()
