#!/usr/bin/env python3
"""
vLLM Proxy Manager - Main Application Entry Point

Provides OpenAI-compatible API endpoints for managing and routing requests
to multiple vLLM model instances with automatic lifecycle management.
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import logging
from typing import Dict, Any, Optional
import asyncio
from contextlib import asynccontextmanager

from model_manager import ModelManager
from config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global instances
model_manager: Optional[ModelManager] = None
config: Optional[Config] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager - startup and shutdown events"""
    global model_manager, config

    # Startup
    logger.info("Starting vLLM Proxy Manager...")
    config = Config()
    model_manager = ModelManager(config)

    # Start health check background task
    asyncio.create_task(health_check_loop())

    logger.info(f"Proxy manager ready on port {config.port}")
    logger.info(f"Exclusive mode: {config.exclusive_mode}")
    logger.info(f"Loaded {len(config.models)} models")

    yield

    # Shutdown
    logger.info("Shutting down vLLM Proxy Manager...")
    if model_manager:
        model_manager.cleanup()


app = FastAPI(
    title="vLLM Proxy Manager",
    description="Multi-model vLLM proxy with automatic lifecycle management",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def health_check_loop():
    """Background task for periodic health checks"""
    while True:
        try:
            await asyncio.sleep(config.health_check_interval)
            if model_manager:
                await model_manager.check_all_models_health()
        except Exception as e:
            logger.error(f"Health check error: {e}")


@app.get("/")
async def root():
    """Root endpoint - serves web UI"""
    return FileResponse("static/index.html")


@app.get("/v1/models")
async def list_models():
    """OpenAI-compatible endpoint - list available models"""
    models = []

    for model_config in config.models:
        models.append({
            "id": model_config.served_model_name or model_config.name,
            "object": "model",
            "created": 1700000000,
            "owned_by": "vllm-proxy-manager",
            "permission": [],
            "root": model_config.name,
            "parent": None
        })

    return {
        "object": "list",
        "data": models
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OpenAI-compatible endpoint - chat completions"""
    try:
        body = await request.json()
        model_name = body.get("model")

        if not model_name:
            raise HTTPException(status_code=400, detail="Model name required")

        # Find model config
        model_config = None
        for mc in config.models:
            if mc.name == model_name or mc.served_model_name == model_name:
                model_config = mc
                break

        if not model_config:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

        # Ensure model is running
        logger.info(f"Chat request for model: {model_name}")
        await model_manager.ensure_model_running(model_config)

        # Proxy request to vLLM
        vllm_host = config.vllm_host or "172.17.0.1"
        target_url = f"http://{vllm_host}:{model_config.port}/v1/chat/completions"

        # Replace model name with served_model_name if configured
        if model_config.served_model_name:
            body["model"] = model_config.served_model_name

        async with httpx.AsyncClient(timeout=300.0) as client:
            if body.get("stream", False):
                # Streaming response
                async def stream_proxy():
                    async with client.stream("POST", target_url, json=body) as response:
                        async for chunk in response.aiter_bytes():
                            yield chunk

                return StreamingResponse(
                    stream_proxy(),
                    media_type="text/event-stream"
                )
            else:
                # Non-streaming response
                response = await client.post(target_url, json=body)
                return JSONResponse(content=response.json(), status_code=response.status_code)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat completion error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/completions")
async def completions(request: Request):
    """OpenAI-compatible endpoint - text completions"""
    # Similar to chat_completions but for /v1/completions endpoint
    try:
        body = await request.json()
        model_name = body.get("model")

        if not model_name:
            raise HTTPException(status_code=400, detail="Model name required")

        model_config = None
        for mc in config.models:
            if mc.name == model_name or mc.served_model_name == model_name:
                model_config = mc
                break

        if not model_config:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

        await model_manager.ensure_model_running(model_config)

        vllm_host = config.vllm_host or "172.17.0.1"
        target_url = f"http://{vllm_host}:{model_config.port}/v1/completions"

        if model_config.served_model_name:
            body["model"] = model_config.served_model_name

        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(target_url, json=body)
            return JSONResponse(content=response.json(), status_code=response.status_code)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Completion error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/models")
async def get_models_status():
    """Get status of all managed models"""
    models = []

    for model_config in config.models:
        status = await model_manager.get_model_status(model_config.name)
        models.append({
            "name": model_config.name,
            "port": model_config.port,
            "status": status,
            "served_model_name": model_config.served_model_name,
            "model_path": model_config.model_path
        })

    return {"models": models}


@app.post("/api/models/{model_name}/start")
async def start_model(model_name: str):
    """Manually start a specific model"""
    model_config = None
    for mc in config.models:
        if mc.name == model_name:
            model_config = mc
            break

    if not model_config:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

    try:
        await model_manager.start_model(model_config)
        return {"status": "success", "message": f"Model {model_name} started"}
    except Exception as e:
        logger.error(f"Failed to start model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/models/{model_name}/stop")
async def stop_model(model_name: str):
    """Manually stop a specific model"""
    try:
        await model_manager.stop_model(model_name)
        return {"status": "success", "message": f"Model {model_name} stopped"}
    except Exception as e:
        logger.error(f"Failed to stop model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/models/{model_name}/status")
async def get_model_status(model_name: str):
    """Get status of a specific model"""
    status = await model_manager.get_model_status(model_name)
    return {"name": model_name, "status": status}


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "service": "vllm-proxy-manager"}


if __name__ == "__main__":
    import uvicorn

    # Load config to get port
    temp_config = Config()

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=temp_config.port,
        log_level="info"
    )
