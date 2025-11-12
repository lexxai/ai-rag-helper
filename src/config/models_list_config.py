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

    def load_config(self) -> Dict[str, List[str]]:
        """Load approved models from YAML configuration file."""
        try:
            if not self.config_path.exists():
                return {"hf": []}

            with self.config_path.open("r") as file:
                config = yaml.safe_load(file)
                if not isinstance(config, dict):
                    raise YamlConfigException("Invalid config format")
                return config
        except yaml.YAMLError as e:
            raise YamlConfigException(f"Error parsing YAML file: {e}")
        except Exception as e:
            raise YamlConfigException(f"Error loading config: {e}")


APPROVED_MODELS = ModelsListConfig().approved_models
