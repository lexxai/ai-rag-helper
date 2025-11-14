from pathlib import Path

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

    def get_model_types(self) -> list[str]:
        return list(self.approved_models.keys())

    def get_model_names(self, model_type: str = None) -> list[str]:
        model_type = model_type or self.model_type
        return [f"{model_type}:{name}" for name in self.approved_models.get(model_type, {}).keys()]

    def get_models(self, model_type: str = None) -> dict:
        return self.approved_models.get(model_type, {})

    def decode_model_name(self, model_name: str) -> tuple[str, str]:
        items = model_name.split(":", maxsplit=1)
        if len(items) == 2:
            model_type, model_name = items
            return model_type, model_name
        return self.model_type, model_name

    def get_model_properties(self, model_name: str) -> dict:
        model_type, model_name = self.decode_model_name(model_name)
        return self.approved_models.get(model_type, {}).get(model_name, {})

    def validate_model_name(self, model_name: str) -> bool:
        model_type, model_name = self.decode_model_name(model_name)
        return model_type in self.approved_models and model_name in self.approved_models[model_type]

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
