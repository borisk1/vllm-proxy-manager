#!/usr/bin/env python3
"""
Model Manager - Docker Container Lifecycle Management

Handles starting, stopping, pausing, and monitoring vLLM containers.
Implements exclusive mode for GPU resource optimization.
"""

import docker
import asyncio
import logging
from typing import Dict, Optional
from datetime import datetime
import httpx

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages lifecycle of vLLM model containers"""

    def __init__(self, config):
        """
        Initialize model manager

        Args:
            config: Configuration object with models and settings
        """
        self.config = config
        self.docker_client = docker.from_env()
        self.model_locks = {}  # Locks for thread-safe model operations
        self.last_used = {}    # Track last usage time for each model

        # Initialize locks for each model
        for model in config.models:
            self.model_locks[model.name] = asyncio.Lock()

    async def ensure_model_running(self, model_config):
        """
        Ensure a model is running and ready to serve requests

        Args:
            model_config: ModelConfig object
        """
        async with self.model_locks[model_config.name]:
            container_name = f"vllm_{model_config.name}"

            try:
                container = self.docker_client.containers.get(container_name)

                # Check if paused
                if container.status == "paused":
                    logger.info(f"Unpausing model: {model_config.name}")
                    container.unpause()
                    await self._wait_for_model_ready(model_config)

                # Check if stopped
                elif container.status != "running":
                    logger.info(f"Starting stopped model: {model_config.name}")
                    container.start()
                    await self._wait_for_model_ready(model_config)

                # Update last used time
                self.last_used[model_config.name] = datetime.now()

                # Handle exclusive mode
                if self.config.exclusive_mode:
                    await self._sleep_other_models(model_config.name)

            except docker.errors.NotFound:
                # Container doesn't exist, create and start it
                logger.info(f"Creating new container for model: {model_config.name}")
                await self.start_model(model_config)

    async def start_model(self, model_config):
        """
        Start a new vLLM container for the specified model

        Args:
            model_config: ModelConfig object
        """
        container_name = f"vllm_{model_config.name}"

        # Build vLLM command arguments
        cmd_args = [
            "/root/model",  # Model path inside container
            "--served-model-name", model_config.served_model_name or model_config.name,
            "--gpu-memory-utilization", str(model_config.gpu_memory_utilization),
            "--max-model-len", str(model_config.max_model_len),
        ]

        if model_config.dtype:
            cmd_args.extend(["--dtype", model_config.dtype])

        if model_config.quantization:
            cmd_args.extend(["--quantization", model_config.quantization])

        if model_config.trust_remote_code:
            cmd_args.append("--trust-remote-code")

        # Container configuration
        container_config = {
            "name": container_name,
            "image": "vllm/vllm-openai:latest",
            "command": ["vllm", "serve"] + cmd_args,
            "runtime": "nvidia",
            "detach": True,
            "ports": {
                '8000/tcp': model_config.port
            },
            "volumes": {
                model_config.model_path: {
                    'bind': '/root/model',
                    'mode': 'ro'
                }
            },
            "environment": {
                "NVIDIA_VISIBLE_DEVICES": "all"
            },
            "shm_size": "2g"
        }

        try:
            # Remove existing container if present
            try:
                old_container = self.docker_client.containers.get(container_name)
                old_container.remove(force=True)
                logger.info(f"Removed old container: {container_name}")
            except docker.errors.NotFound:
                pass

            # Create and start new container
            container = self.docker_client.containers.run(**container_config)
            logger.info(f"Started container: {container_name} on port {model_config.port}")

            # Wait for model to be ready
            await self._wait_for_model_ready(model_config)

            # Update last used time
            self.last_used[model_config.name] = datetime.now()

        except Exception as e:
            logger.error(f"Failed to start model {model_config.name}: {e}", exc_info=True)
            raise

    async def stop_model(self, model_name: str):
        """
        Stop and remove a model container

        Args:
            model_name: Name of the model to stop
        """
        container_name = f"vllm_{model_name}"

        try:
            container = self.docker_client.containers.get(container_name)
            container.stop(timeout=10)
            container.remove()
            logger.info(f"Stopped and removed container: {container_name}")

            # Clean up tracking
            if model_name in self.last_used:
                del self.last_used[model_name]

        except docker.errors.NotFound:
            logger.warning(f"Container {container_name} not found")
        except Exception as e:
            logger.error(f"Failed to stop model {model_name}: {e}")
            raise

    async def pause_model(self, model_name: str):
        """
        Pause a model container to free GPU memory

        Args:
            model_name: Name of the model to pause
        """
        container_name = f"vllm_{model_name}"

        try:
            container = self.docker_client.containers.get(container_name)
            if container.status == "running":
                container.pause()
                logger.info(f"Paused container: {container_name}")
        except docker.errors.NotFound:
            logger.warning(f"Container {container_name} not found")
        except Exception as e:
            logger.error(f"Failed to pause model {model_name}: {e}")

    async def get_model_status(self, model_name: str) -> str:
        """
        Get current status of a model container

        Args:
            model_name: Name of the model

        Returns:
            Status string: 'running', 'paused', 'stopped', or 'not_found'
        """
        container_name = f"vllm_{model_name}"

        try:
            container = self.docker_client.containers.get(container_name)
            return container.status
        except docker.errors.NotFound:
            return "not_found"

    async def _wait_for_model_ready(self, model_config, max_wait: int = 300):
        """
        Wait for model to be ready to serve requests

        Args:
            model_config: ModelConfig object
            max_wait: Maximum seconds to wait
        """
        vllm_host = self.config.vllm_host or "172.17.0.1"
        health_url = f"http://{vllm_host}:{model_config.port}/v1/models"

        logger.info(f"Waiting for model {model_config.name} to be ready...")

        for i in range(max_wait):
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(health_url, timeout=5.0)
                    if response.status_code == 200:
                        logger.info(f"Model {model_config.name} is ready!")
                        return
            except Exception:
                pass

            await asyncio.sleep(1)

            if i % 10 == 0 and i > 0:
                logger.info(f"Still waiting for {model_config.name}... ({i}s)")

        raise TimeoutError(f"Model {model_config.name} did not become ready within {max_wait} seconds")

    async def _sleep_other_models(self, active_model_name: str):
        """
        Pause all models except the active one (exclusive mode)

        Args:
            active_model_name: Name of the model to keep running
        """
        for model in self.config.models:
            if model.name != active_model_name:
                await self.pause_model(model.name)

    async def check_all_models_health(self):
        """Periodic health check for all running models"""
        for model in self.config.models:
            status = await self.get_model_status(model.name)

            if status == "running":
                # Check if model is responsive
                vllm_host = self.config.vllm_host or "172.17.0.1"
                health_url = f"http://{vllm_host}:{model.port}/v1/models"

                try:
                    async with httpx.AsyncClient() as client:
                        response = await client.get(health_url, timeout=5.0)
                        if response.status_code != 200:
                            logger.warning(f"Model {model.name} is not responding correctly")
                except Exception as e:
                    logger.error(f"Health check failed for {model.name}: {e}")

    def cleanup(self):
        """Cleanup resources on shutdown"""
        logger.info("Cleaning up model manager...")
        # Optionally stop all containers here
        # for model in self.config.models:
        #     asyncio.create_task(self.stop_model(model.name))
