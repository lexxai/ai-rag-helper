from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables and .env file.
    """

    model_config = SettingsConfigDict(
        env_file="../.env",
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

    gpu_monitor_loop_delay: int = 15
    prometheus_loop_delay: int = 15


# Singleton instance
settings = Settings()
