from pathlib import Path
from typing import Dict, List

import yaml

from config.settings import settings


class YamlConfigException(Exception):
    """Exception raised for errors in YAML configuration."""

    ...


class ModelsListConfig:
    """Configuration class for approved models."""

    def __init__(self, config_path: str | Path = None):
        self.config_path = Path(config_path) if config_path else settings.approved_models_config_path
        self.approved_models = self.load_config()
        self.model_type = "hf"

    def get_model_names(self, model_type: str = None) -> list[str]:
        model_type = model_type or self.model_type
        return list(self.approved_models.get(model_type, {}).keys())

    def load_config(self) -> dict[str, dict[str, dict]]:
        """Load approved models from YAML configuration file."""
        try:
            if not self.config_path.exists():
                return {self.model_type: {}}

            with self.config_path.open("r") as file:
                config = yaml.safe_load(file)
                if not isinstance(config, dict):
                    raise YamlConfigException("Invalid config format")
                return config
        except yaml.YAMLError as e:
            raise YamlConfigException(f"Error parsing YAML file: {e}")
        except Exception as e:
            raise YamlConfigException(f"Error loading config: {e}")


models_list_config = ModelsListConfig()
APPROVED_MODELS = models_list_config.approved_models

# print(APPROVED_MODELS)
