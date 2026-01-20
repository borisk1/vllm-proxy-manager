#!/usr/bin/env python3
"""
Configuration Management

Loads and validates configuration from JSON file and environment variables.
"""

import json
import os
import logging
from typing import List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a single vLLM model"""
    name: str
    model_path: str
    port: int
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 4096
    served_model_name: Optional[str] = None
    dtype: str = "auto"
    quantization: Optional[str] = None
    trust_remote_code: bool = False

    @classmethod
    def from_dict(cls, data: dict):
        """Create ModelConfig from dictionary"""
        return cls(
            name=data["name"],
            model_path=data["model_path"],
            port=data["port"],
            gpu_memory_utilization=data.get("gpu_memory_utilization", 0.9),
            max_model_len=data.get("max_model_len", 4096),
            served_model_name=data.get("served_model_name"),
            dtype=data.get("dtype", "auto"),
            quantization=data.get("quantization"),
            trust_remote_code=data.get("trust_remote_code", False)
        )


class Config:
    """Main application configuration"""

    def __init__(self, config_path: str = "data/models.json"):
        """
        Load configuration from JSON file and environment variables

        Args:
            config_path: Path to models.json configuration file
        """
        self.config_path = config_path
        self.models: List[ModelConfig] = []
        self.exclusive_mode: bool = True
        self.health_check_interval: int = 30
        self.port: int = 8080
        self.vllm_host: Optional[str] = None

        self._load_config()
        self._load_env_overrides()

    def _load_config(self):
        """Load configuration from JSON file"""
        try:
            with open(self.config_path, 'r') as f:
                config_data = json.load(f)

            # Load models
            for model_data in config_data.get("models", []):
                self.models.append(ModelConfig.from_dict(model_data))

            # Load settings
            self.exclusive_mode = config_data.get("exclusive_mode", True)
            self.health_check_interval = config_data.get("health_check_interval", 30)

            logger.info(f"Loaded configuration from {self.config_path}")
            logger.info(f"Found {len(self.models)} models")

        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            # Create default config
            self._create_default_config()
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in configuration file: {e}")
            raise

    def _load_env_overrides(self):
        """Load configuration overrides from environment variables"""
        if os.getenv("VLLM_HOST"):
            self.vllm_host = os.getenv("VLLM_HOST")
            logger.info(f"Using vLLM host from environment: {self.vllm_host}")

        if os.getenv("PORT"):
            self.port = int(os.getenv("PORT"))
            logger.info(f"Using port from environment: {self.port}")

        if os.getenv("EXCLUSIVE_MODE"):
            self.exclusive_mode = os.getenv("EXCLUSIVE_MODE").lower() == "true"

    def _create_default_config(self):
        """Create default configuration file if none exists"""
        default_config = {
            "models": [
                {
                    "name": "example-model",
                    "model_path": "/path/to/model",
                    "port": 8001,
                    "gpu_memory_utilization": 0.9,
                    "max_model_len": 4096,
                    "served_model_name": "example-model",
                    "dtype": "auto",
                    "quantization": null,
                    "trust_remote_code": false
                }
            ],
            "exclusive_mode": true,
            "health_check_interval": 30
        }

        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)

        with open(self.config_path, 'w') as f:
            json.dump(default_config, f, indent=2)

        logger.info(f"Created default configuration at {self.config_path}")
        logger.warning("Please edit the configuration file and restart the application")

    def validate(self) -> bool:
        """
        Validate configuration

        Returns:
            True if configuration is valid, False otherwise
        """
        if not self.models:
            logger.error("No models configured")
            return False

        # Check for duplicate ports
        ports = [m.port for m in self.models]
        if len(ports) != len(set(ports)):
            logger.error("Duplicate ports detected in model configuration")
            return False

        # Check for duplicate model names
        names = [m.name for m in self.models]
        if len(names) != len(set(names)):
            logger.error("Duplicate model names detected in configuration")
            return False

        return True
