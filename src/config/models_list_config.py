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

    @staticmethod
    def decode_model_class_name(model_name: str) -> str:
        """
        Decode and extract the model class name from a given string representation of a model name.
        This static method processes the input by removing leading/trailing whitespace, converting
        it to lowercase, and splitting it by a slash to extract the first component. If the split
        does not contain two components, it returns a default value.

        :param model_name: Input string representing the model name.
        :type model_name: str
        :return: The extracted model class name or the default value "sentence-transformers".
        :rtype: str
        """
        items = model_name.strip().lower().split("/", maxsplit=1)
        if len(items) == 2:
            model_class_name = items[0]
            return model_class_name
        return "sentence-transformers"

    def decode_model_name(self, model_name: str) -> tuple[str, str, str]:
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
        items = model_name.split(":", maxsplit=1)
        if len(items) == 2:
            model_type, model_name = items
            return model_type, model_name, self.decode_model_class_name(model_name)
        return self.model_type, model_name, self.decode_model_class_name(model_name)

    def get_model_properties(self, model_name: str) -> dict:
        model_type, model_name, _ = self.decode_model_name(model_name)
        return self.approved_models.get(model_type, {}).get(model_name, {})

    def validate_model_name(self, model_name: str) -> bool:
        model_type, model_name, _ = self.decode_model_name(model_name)
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
