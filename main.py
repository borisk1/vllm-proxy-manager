from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import docker
import json
import time
import httpx
from typing import Optional, List, Dict
from pydantic import BaseModel
import asyncio
from datetime import datetime
import os
import random

app = FastAPI(title="LLM Proxy Manager")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

docker_client = docker.from_env()
CONFIG_FILE = "/app/data/models_config.json"
STATS_FILE = "/app/data/model_stats.json"
VLLM_HOST = os.getenv("VLLM_HOST", "host.docker.internal")

# Lock to prevent auto-sleep during active requests
model_locks = {}

# Funny loading messages displayed during model initialization
LOADING_MESSAGES = [
    "🧠 Waking up the neurons, they had a good nap...",
    "🔥 Heating up the GPUs, winter is coming...",
    "🎯 Loading weights, they're heavier than you think...",
    "🚀 Initializing hyperspace jump sequence...",
    "🤖 Teaching the model to think, it's a slow learner...",
    "☕ Brewing some AI coffee, this might take a while...",
    "🧪 Mixing the secret sauce of intelligence...",
    "🎪 Juggling tensors, don't drop them!",
    "🎨 Painting the neural pathways with gradients...",
    "🎭 The model is getting into character...",
    "🏋️ Lifting those matrix multiplications...",
    "🎸 Tuning the attention heads, they're a bit off-key...",
    "🍕 Ordering pizza for the GPUs, they're hungry...",
    "🎯 Calculating the meaning of life... still at 41.99...",
    "🌊 Surfing the waves of probability distributions...",
    "🎲 Rolling the dice on random initialization...",
    "🔮 Consulting the Oracle of Backpropagation...",
    "🎺 Jazz hands while loading embeddings...",
    "🏃 Running marathons through vector spaces...",
    "🎪 Performing circus tricks with attention masks...",
    "🧙 Casting spells on the weight matrices...",
    "🎬 Directing the blockbuster: 'Return of the Gradients'...",
    "🌮 Feeding tacos to the transformer layers...",
    "🎹 Playing a symphony on the activation functions...",
    "🏰 Building castles in the latent space...",
    "🎨 Bob Ross is painting happy little neurons...",
    "🚁 Helicopter parents hovering over the model...",
    "🎪 Herding cats through the decoder layers...",
    "🍿 Popping kernels of knowledge...",
    "🎯 Throwing darts at the loss function...",
    "🏊 Swimming through seas of backpropagation...",
    "🎸 Shredding guitar solos on the attention mechanism...",
    "🌈 Finding the pot of gold at the end of training...",
    "🎭 Method acting as a large language model...",
    "🎪 Walking the tightrope of numerical stability...",
    "🍔 Assembling the perfect burger of intelligence...",
    "🎨 Mixing colors in the RGB space of embeddings...",
    "🏋️ Bench pressing the entire vocabulary...",
    "🎯 Aiming for the bullseye of convergence...",
    "🌊 Catching waves in the gradient descent ocean...",
    "🎺 Blowing bubbles through the MLP layers...",
    "🏃 Sprinting through the hallways of hidden states...",
    "🎪 Balancing plates on the attention heads...",
    "🍕 Slicing through dimensions like pizza...",
    "🎬 The model is in hair and makeup...",
    "🧙 Brewing potions in the cauldron of computation...",
    "🎹 Tickling the ivories of inference...",
    "🏰 Defending the fortress of floating point precision...",
    "🎨 Sculpting masterpieces from raw tensors...",
    "🚁 Dropping supplies to stranded gradients...",
]

class ModelConfig(BaseModel):
    name: str
    backend: str = "vllm"  # vllm or llamacpp
    model_path: str
    is_local: bool = False
    gpu_memory_utilization: Optional[float] = 0.9
    max_model_len: Optional[int] = None
    quantization: Optional[str] = None
    tensor_parallel_size: Optional[int] = 1
    context_size: Optional[int] = 4096
    gpu_layers: Optional[int] = 35
    parallel_requests: Optional[int] = 4
    port: int
    exclusive: bool = False
    auto_sleep: bool = True
    sleep_timeout: int = 300
    always_visible: bool = False
    docker_image: Optional[str] = None
    custom_docker_args: Optional[str] = None
    custom_vllm_args: Optional[str] = None
    custom_llamacpp_args: Optional[str] = None
    custom_env_vars: Optional[str] = None
    preload: bool = False

def load_configs():
    """Load model configurations from JSON file"""
    os.makedirs("/app/data", exist_ok=True)
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_configs(configs):
    """Save model configurations to JSON file"""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(configs, f, indent=2)

def load_stats():
    """Load model statistics from JSON file"""
    if os.path.exists(STATS_FILE):
        try:
            with open(STATS_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_stats(stats):
    """Save model statistics to JSON file"""
    try:
        with open(STATS_FILE, 'w') as f:
            json.dump(stats, f, indent=2)
    except Exception as e:
        print(f"[STATS] Error: {e}")

configs = load_configs()
stats = load_stats()
last_request_time = {}

def get_container_name(model_name):
    """Generate Docker container name from model name"""
    return f"llm_{model_name.replace('/', '_').replace(':', '_').replace('.', '_')}"

def get_served_model_name(config: dict) -> str:
    """Extract served model name from vLLM custom arguments or use model path"""
    backend = config.get("backend", "vllm")
    if backend == "vllm" and config.get("custom_vllm_args") and "--served-model-name" in config.get("custom_vllm_args", ""):
        args_text = config["custom_vllm_args"]
        if '\\n' in args_text:
            args_text = args_text.replace('\\n', '\n')
        lines = args_text.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith("--served-model-name"):
                parts = line.split(None, 1)
                if len(parts) >= 2:
                    return parts[1].strip()
    return config["model_path"]

def get_vllm_command(config: ModelConfig) -> List[str]:
    """Build vLLM command line arguments"""
    container_model_path = "/root/model" if config.is_local else config.model_path
    cmd_parts = [container_model_path, "--gpu-memory-utilization", str(config.gpu_memory_utilization),
                 "--tensor-parallel-size", str(config.tensor_parallel_size),
                 "--port", "8000", "--host", "0.0.0.0", "--trust-remote-code"]

    if config.max_model_len:
        cmd_parts.extend(["--max-model-len", str(config.max_model_len)])
    if config.quantization:
        cmd_parts.extend(["--quantization", config.quantization])

    has_served_model_name = False
    if config.custom_vllm_args:
        args_text = config.custom_vllm_args
        if '\\n' in args_text:
            args_text = args_text.replace('\\n', '\n')

        if '--served-model-name' in args_text:
            has_served_model_name = True

        lines = args_text.split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split(None, 1)
                if parts:
                    cmd_parts.extend(parts)

    if not has_served_model_name:
        cmd_parts.extend(["--served-model-name", config.name])

    return cmd_parts

def get_llamacpp_command(config: ModelConfig) -> List[str]:
    """Build llama.cpp command line arguments"""
    container_model_path = "/root/model" if config.is_local else config.model_path

    cmd_parts = [
        "--server",
        "--model", container_model_path,
        "--host", "0.0.0.0",
        "--port", "8000",
        "--ctx-size", str(config.context_size or 4096),
        "-ngl", str(config.gpu_layers if config.gpu_layers is not None else 35),
        "--parallel", str(config.parallel_requests or 4),
        "--chat-template", "chatml"
    ]

    if config.custom_llamacpp_args:
        args_text = config.custom_llamacpp_args
        if '\\n' in args_text:
            args_text = args_text.replace('\\n', '\n')

        lines = args_text.split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split(None, 1)
                if parts:
                    cmd_parts.extend(parts)

    return cmd_parts

def parse_custom_docker_args(custom_args_str: Optional[str]) -> Dict:
    """Parse custom Docker arguments from string to dictionary"""
    if not custom_args_str:
        return {}
    result = {}
    try:
        args_text = custom_args_str.strip()
        if '\\n' in args_text:
            args_text = args_text.replace('\\n', '\n')
        lines = args_text.split('\n')
        for line in lines:
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                if value.lower() in ['true', 'false']:
                    value = value.lower() == 'true'
                elif value.isdigit():
                    value = int(value)
                result[key] = value
    except Exception as e:
        print(f"[DOCKER] Error parsing Docker args: {e}")
    return result

def parse_custom_env_vars(env_vars_str: Optional[str]) -> Dict[str, str]:
    """Parse custom environment variables from string to dictionary"""
    if not env_vars_str:
        return {}
    result = {}
    try:
        env_text = env_vars_str.strip()
        if '\\n' in env_text:
            env_text = env_text.replace('\\n', '\n')
        lines = env_text.split('\n')
        for line in lines:
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                key, value = line.split('=', 1)
                result[key.strip()] = value.strip()
    except Exception as e:
        print(f"[ENV] Error parsing env vars: {e}")
    return result

def init_model_stats(model_name: str):
    """Initialize statistics for a model if not exists"""
    if model_name not in stats:
        stats[model_name] = {
            "total_requests": 0, "total_prompt_tokens": 0, "total_completion_tokens": 0,
            "total_tokens": 0, "total_generation_time": 0.0, "average_tokens_per_second": 0.0,
            "started_at": None, "last_request_at": None
        }

def update_model_stats(model_name: str, prompt_tokens: int, completion_tokens: int, generation_time: float):
    """Update model statistics after each request"""
    init_model_stats(model_name)
    stats[model_name]["total_requests"] += 1
    stats[model_name]["total_prompt_tokens"] += prompt_tokens
    stats[model_name]["total_completion_tokens"] += completion_tokens
    stats[model_name]["total_tokens"] += prompt_tokens + completion_tokens
    stats[model_name]["total_generation_time"] += generation_time
    stats[model_name]["last_request_at"] = datetime.now().isoformat()
    if stats[model_name]["total_generation_time"] > 0:
        stats[model_name]["average_tokens_per_second"] = (
            stats[model_name]["total_completion_tokens"] / stats[model_name]["total_generation_time"]
        )
    last_request_time[model_name] = time.time()
    save_stats(stats)

def get_model_lock(model_name: str):
    """Get or create asyncio lock for model to prevent concurrent operations"""
    if model_name not in model_locks:
        model_locks[model_name] = asyncio.Lock()
    return model_locks[model_name]

async def ensure_model_running(model_name: str, config: dict):
    """Ensure model is running with retry logic"""
    container_name = get_container_name(model_name)
    max_retries = 3

    for attempt in range(max_retries):
        try:
            container = docker_client.containers.get(container_name)
            if container.status == "running":
                return
            elif container.status == "paused":
                print(f"[ENSURE] Model {model_name} is paused, waking up...")
                container.unpause()
                await asyncio.sleep(1)
                last_request_time[model_name] = time.time()
                return
            else:
                print(f"[ENSURE] Model {model_name} status: {container.status}, restarting...")
                await start_model(model_name)
                return

        except docker.errors.NotFound:
            print(f"[ENSURE] Model {model_name} not found, starting...")
            await start_model(model_name)
            return
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"[ENSURE] Attempt {attempt + 1} failed: {e}, retrying...")
                await asyncio.sleep(1)
            else:
                raise

@app.get("/")
async def root():
    """Serve the main web interface"""
    try:
        with open("/app/static/index.html", 'r') as f:
            return HTMLResponse(content=f.read())
    except:
        return HTMLResponse(content="<h1>LLM Proxy Manager</h1>")

app.mount("/static", StaticFiles(directory="/app/static"), name="static")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse(content={"status": "OK"})

@app.get("/running")
async def list_running_models():
    """List all currently running models"""
    running = []
    for name, config in configs.items():
        container_name = get_container_name(name)
        try:
            container = docker_client.containers.get(container_name)
            if container.status == "running":
                running.append({
                    "name": name,
                    "port": config["port"],
                    "status": "running",
                    "backend": config.get("backend", "vllm"),
                    "docker_image": config.get("docker_image", "vllm/vllm-openai:v0.13.0"),
                    "uptime": (datetime.now() - datetime.fromisoformat(stats.get(name, {}).get("started_at", datetime.now().isoformat()))).total_seconds() if stats.get(name, {}).get("started_at") else 0
                })
        except:
            pass
    return JSONResponse(content={"models": running, "count": len(running)})

@app.post("/models/unload/{model_name}")
async def unload_model(model_name: str):
    """Unload (stop) a model - compatibility endpoint"""
    return await stop_model(model_name)

@app.get("/v1/models")
async def list_openai_models():
    """OpenAI-compatible endpoint to list available models"""
    model_list = []
    for name, config in configs.items():
        container_name = get_container_name(name)
        status = "not_running"
        try:
            container = docker_client.containers.get(container_name)
            status = container.status
        except:
            pass
        always_visible = config.get("always_visible", False)
        is_running = status == "running"
        if is_running or always_visible:
            model_entry = {
                "id": name, "object": "model", "created": int(datetime.now().timestamp()),
                "owned_by": "llm-proxy", "permission": [], "root": name, "parent": None
            }
            if config.get("max_model_len"):
                model_entry["context_length"] = config["max_model_len"]
            elif config.get("context_size"):
                model_entry["context_length"] = config["context_size"]
            model_list.append(model_entry)
    return JSONResponse(content={"object": "list", "data": model_list})

async def stream_response_with_loading(vllm_request: dict, port: int, model_name: str):
    """Stream response from backend with funny loading progress updates"""
    start_time = time.time()
    prompt_tokens = 0
    completion_tokens = 0
    max_retries = 2

    # Shuffle messages for variety
    messages = LOADING_MESSAGES.copy()
    random.shuffle(messages)
    message_index = 0

    for attempt in range(max_retries):
        async with httpx.AsyncClient(timeout=600.0) as client:
            try:
                target_url = f"http://{VLLM_HOST}:{port}/v1/chat/completions"

                async with client.stream("POST", target_url, json=vllm_request) as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        error_msg = error_text.decode()
                        print(f"[STREAM] Error {response.status_code}: {error_msg}")
                        yield f"data: {json.dumps({'error': error_msg})}\n\n"
                        return

                    async for chunk in response.aiter_bytes():
                        if chunk:
                            try:
                                chunk_str = chunk.decode('utf-8')
                                if '"usage"' in chunk_str:
                                    for line in chunk_str.split('\n'):
                                        if line.startswith('data: ') and line.strip() != 'data: [DONE]':
                                            try:
                                                data = json.loads(line[6:])
                                                if 'usage' in data:
                                                    prompt_tokens = data['usage'].get('prompt_tokens', 0)
                                                    completion_tokens = data['usage'].get('completion_tokens', 0)
                                            except:
                                                pass
                            except:
                                pass
                            yield chunk

                    generation_time = time.time() - start_time
                    if completion_tokens > 0:
                        update_model_stats(model_name, prompt_tokens, completion_tokens, generation_time)
                    return

            except (httpx.ConnectError, httpx.RemoteProtocolError) as e:
                if attempt < max_retries - 1:
                    print(f"[STREAM] Connection error (attempt {attempt + 1}), loading model...")

                    # Send initial loading message
                    loading_msg = {
                        "id": f"chatcmpl-loading-{int(time.time())}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model_name,
                        "choices": [{
                            "index": 0,
                            "delta": {"role": "assistant", "content": f"⏳ Loading model {model_name}...\n\n"},
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(loading_msg)}\n\n"

                    async with get_model_lock(model_name):
                        await ensure_model_running(model_name, configs[model_name])

                    # Wait and send funny progress updates every 10 seconds
                    for i in range(60):  # Wait up to 10 minutes
                        await asyncio.sleep(10)

                        # Send funny message
                        funny_message = messages[message_index % len(messages)]
                        message_index += 1

                        progress_msg = {
                            "id": f"chatcmpl-loading-{int(time.time())}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model_name,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": f"{funny_message}\n"},
                                "finish_reason": None
                            }]
                        }
                        yield f"data: {json.dumps(progress_msg)}\n\n"

                        # Check if model is ready
                        try:
                            test_response = await client.get(f"http://{VLLM_HOST}:{port}/v1/models", timeout=5.0)
                            if test_response.status_code == 200:
                                ready_msg = {
                                    "id": f"chatcmpl-loading-{int(time.time())}",
                                    "object": "chat.completion.chunk",
                                    "created": int(time.time()),
                                    "model": model_name,
                                    "choices": [{
                                        "index": 0,
                                        "delta": {"content": "\n✅ Model is ready! Let me think...\n\n"},
                                        "finish_reason": None
                                    }]
                                }
                                yield f"data: {json.dumps(ready_msg)}\n\n"
                                break
                        except:
                            pass
                else:
                    error_msg = {
                        "error": {
                            "message": f"Failed to load model after {max_retries} attempts: {str(e)}",
                            "type": "service_unavailable",
                            "code": 503
                        }
                    }
                    yield f"data: {json.dumps(error_msg)}\n\n"
                    return
            except Exception as e:
                print(f"[STREAM] Exception: {str(e)}")
                error_msg = {"error": {"message": str(e), "type": "internal_error", "code": 500}}
                yield f"data: {json.dumps(error_msg)}\n\n"
                return

@app.post("/v1/chat/completions")
async def proxy_chat_completion(request: dict):
    """OpenAI-compatible chat completion endpoint with automatic model loading"""
    model_name = request.get("model")

    if model_name not in configs:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

    config = configs[model_name]

    print(f"[REQUEST] Received request for model: {model_name}")

    async with get_model_lock(model_name):
        last_request_time[model_name] = time.time()

        container_name = get_container_name(model_name)
        model_needs_loading = False
        try:
            container = docker_client.containers.get(container_name)
            if container.status != "running":
                model_needs_loading = True
                print(f"[REQUEST] Model {model_name} not running, status: {container.status}")
        except docker.errors.NotFound:
            model_needs_loading = True
            print(f"[REQUEST] Model {model_name} not found, needs to start")

        if model_needs_loading:
            print(f"[REQUEST] Starting model {model_name}...")

        await ensure_model_running(model_name, config)

        # Handle exclusive mode - stop all other models
        if config.get("exclusive", False):
            for other_model in configs.keys():
                if other_model != model_name:
                    await sleep_model(other_model)

    port = config["port"]
    served_model_name = get_served_model_name(config)

    vllm_request = {
        "model": served_model_name,
        "messages": request.get("messages"),
        "temperature": request.get("temperature", 0.7),
        "max_tokens": request.get("max_tokens", 2000),
        "stream": request.get("stream", False)
    }

    for key in ["top_p", "frequency_penalty", "presence_penalty", "stop"]:
        if key in request:
            vllm_request[key] = request[key]

    if request.get("stream"):
        return StreamingResponse(
            stream_response_with_loading(vllm_request, port, model_name),
            media_type="text/event-stream"
        )

    # For non-streaming requests
    start_time = time.time()
    max_retries = 3

    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=600.0) as client:
                target_url = f"http://{VLLM_HOST}:{port}/v1/chat/completions"
                print(f"[REQUEST] Sending request to {target_url} (attempt {attempt + 1})")

                response = await client.post(target_url, json=vllm_request)
                generation_time = time.time() - start_time

                if response.status_code != 200:
                    error_text = response.text
                    print(f"[REQUEST] Backend error: {error_text}")
                    raise HTTPException(status_code=response.status_code, detail=f"Backend error: {error_text}")

                response_data = response.json()
                usage = response_data.get("usage", {})
                update_model_stats(model_name, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0), generation_time)

                print(f"[REQUEST] Success! Response generated in {generation_time:.2f}s")
                return JSONResponse(content=response_data)

        except (httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadTimeout) as e:
            if attempt < max_retries - 1:
                wait_time = 10 if attempt == 0 else 20
                print(f"[REQUEST] Connection error (attempt {attempt + 1}): {e}, waiting {wait_time}s before retry...")

                async with get_model_lock(model_name):
                    await ensure_model_running(model_name, config)

                await asyncio.sleep(wait_time)
            else:
                print(f"[REQUEST] Failed after {max_retries} attempts: {e}")
                raise HTTPException(status_code=503, detail=f"Service unavailable after {max_retries} attempts: {str(e)}")

@app.get("/api/models")
async def list_models():
    """List all configured models with their status and statistics"""
    result = []
    for name, config in configs.items():
        container_name = get_container_name(name)
        try:
            container = docker_client.containers.get(container_name)
            status = container.status
        except:
            status = "not_created"
        init_model_stats(name)
        model_stats = stats.get(name, {})
        result.append({
            "name": name, "config": config, "status": status, "running": status == "running",
            "stats": {
                "total_requests": model_stats.get("total_requests", 0),
                "total_prompt_tokens": model_stats.get("total_prompt_tokens", 0),
                "total_completion_tokens": model_stats.get("total_completion_tokens", 0),
                "total_tokens": model_stats.get("total_tokens", 0),
                "average_tokens_per_second": round(model_stats.get("average_tokens_per_second", 0), 2),
                "total_generation_time": round(model_stats.get("total_generation_time", 0), 2),
                "started_at": model_stats.get("started_at"),
                "last_request_at": model_stats.get("last_request_at")
            }
        })
    return JSONResponse(content=result)

@app.post("/api/models")
async def add_model(config: ModelConfig):
    """Add a new model configuration"""
    if config.name in configs:
        raise HTTPException(status_code=400, detail="Model exists")
    configs[config.name] = config.dict()
    save_configs(configs)
    init_model_stats(config.name)
    return {"message": "Model added"}

@app.put("/api/models/{model_name}")
async def update_model(model_name: str, config: ModelConfig):
    """Update an existing model configuration"""
    if model_name not in configs:
        raise HTTPException(status_code=404, detail="Not found")
    await stop_model(model_name)
    if model_name != config.name and config.name in configs:
        raise HTTPException(status_code=400, detail="Name exists")
    if model_name != config.name:
        del configs[model_name]
    configs[config.name] = config.dict()
    save_configs(configs)
    return {"message": "Updated"}

@app.delete("/api/models/{model_name}")
async def delete_model(model_name: str):
    """Delete a model configuration and stop its container"""
    if model_name not in configs:
        raise HTTPException(status_code=404, detail="Not found")
    await stop_model(model_name)
    del configs[model_name]
    save_configs(configs)
    return {"message": "Deleted"}

@app.post("/api/models/{model_name}/start")
async def start_model(model_name: str):
    """Start a model container"""
    if model_name not in configs:
        raise HTTPException(status_code=404, detail="Not found")
    config = ModelConfig(**configs[model_name])
    container_name = get_container_name(model_name)
    backend = config.backend or "vllm"

    # Handle exclusive mode - stop other models first
    if config.exclusive:
        for other_model in configs.keys():
            if other_model != model_name:
                other_container_name = get_container_name(other_model)
                try:
                    other_container = docker_client.containers.get(other_container_name)
                    if other_container.status == "paused":
                        print(f"[EXCLUSIVE] Unpausing {other_model} before stopping...")
                        other_container.unpause()
                        await asyncio.sleep(0.5)
                    if other_container.status in ["running", "paused"]:
                        print(f"[EXCLUSIVE] Stopping {other_model}...")
                        await stop_model(other_model)
                except docker.errors.NotFound:
                    pass
                except Exception as e:
                    print(f"[EXCLUSIVE] Error handling {other_model}: {e}")
        await asyncio.sleep(2)

    try:
        try:
            container = docker_client.containers.get(container_name)
            if container.status == "running":
                return {"message": "Already running"}
            elif container.status == "paused":
                container.unpause()
                return {"message": "Resumed"}
            else:
                container.remove()
        except docker.errors.NotFound:
            pass

        # Select appropriate Docker image and command based on backend
        if backend == "vllm":
            docker_image = config.docker_image if config.docker_image else "vllm/vllm-openai:v0.13.0"
            command = get_vllm_command(config)
        else:
            docker_image = config.docker_image if config.docker_image else "ghcr.io/ggerganov/llama.cpp:server"
            command = get_llamacpp_command(config)

        custom_docker_kwargs = parse_custom_docker_args(config.custom_docker_args)
        volumes = {"/root/.cache/huggingface": {"bind": "/root/.cache/huggingface", "mode": "rw"}}
        if config.is_local:
            volumes[config.model_path.rstrip('/')] = {"bind": "/root/model", "mode": "ro"}

        environment = {
            "CUDA_VISIBLE_DEVICES": "0" if (config.tensor_parallel_size or 1) == 1 else ",".join(str(i) for i in range(config.tensor_parallel_size or 1)),
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID"
        }

        if backend == "vllm":
            environment["VLLM_TARGET_DEVICE"] = "cuda"

        custom_env = parse_custom_env_vars(config.custom_env_vars)
        environment.update(custom_env)

        docker_kwargs = {
            "image": docker_image, "command": command, "name": container_name,
            "detach": True, "runtime": "nvidia",
            "device_requests": [docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])],
            "ports": {'8000/tcp': config.port}, "environment": environment, "volumes": volumes,
            "shm_size": "32g", "ipc_mode": "host", "remove": False
        }
        docker_kwargs.update(custom_docker_kwargs)
        container = docker_client.containers.run(**docker_kwargs)

        print(f"[START] Waiting for model {model_name} to be ready...")
        await wait_for_model_ready(config.port, timeout=600)

        init_model_stats(model_name)
        stats[model_name]["started_at"] = datetime.now().isoformat()
        last_request_time[model_name] = time.time()
        save_stats(stats)

        print(f"[START] Model {model_name} is ready!")
        return {"message": "Started"}
    except Exception as e:
        print(f"[START] Error starting {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/models/{model_name}/stop")
async def stop_model(model_name: str):
    """Stop a model container"""
    container_name = get_container_name(model_name)
    try:
        container = docker_client.containers.get(container_name)
        if container.status == "paused":
            print(f"[STOP] Container {model_name} is paused, unpausing before stop...")
            container.unpause()
            await asyncio.sleep(0.5)
        container.stop(timeout=10)
        container.remove()
        if model_name in last_request_time:
            del last_request_time[model_name]
        return {"message": "Stopped"}
    except docker.errors.NotFound:
        return {"message": "Not running"}
    except Exception as e:
        print(f"[STOP] Error stopping {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/models/{model_name}/sleep")
async def sleep_model(model_name: str):
    """Pause a model container to free GPU memory"""
    container_name = get_container_name(model_name)
    try:
        container = docker_client.containers.get(container_name)
        if container.status == "running":
            container.pause()
        return {"message": "Paused"}
    except docker.errors.NotFound:
        return {"message": "Not found"}

@app.post("/api/models/{model_name}/wake")
async def wake_model(model_name: str):
    """Resume a paused model container"""
    container_name = get_container_name(model_name)
    try:
        container = docker_client.containers.get(container_name)
        if container.status == "paused":
            container.unpause()
            last_request_time[model_name] = time.time()
        return {"message": "Resumed"}
    except docker.errors.NotFound:
        raise HTTPException(status_code=404, detail="Not found")

async def wait_for_model_ready(port: int, timeout: int = 600):
    """Wait for model to be ready by polling the /v1/models endpoint"""
    start_time = time.time()
    async with httpx.AsyncClient() as client:
        while time.time() - start_time < timeout:
            try:
                response = await client.get(f"http://{VLLM_HOST}:{port}/v1/models", timeout=5.0)
                if response.status_code == 200:
                    print(f"[READY] Model on port {port} is ready!")
                    return True
            except Exception as e:
                elapsed = int(time.time() - start_time)
                if elapsed % 10 == 0:  # Log every 10 seconds
                    print(f"[READY] Waiting for model on port {port}... ({elapsed}s elapsed)")
            await asyncio.sleep(5)
    raise TimeoutError(f"Model failed to start within {timeout} seconds")

@app.post("/api/config/export")
async def export_config():
    """Export all model configurations"""
    return configs

@app.post("/api/config/import")
async def import_config(new_configs: Dict):
    """Import model configurations from backup"""
    global configs
    configs = new_configs
    save_configs(configs)
    return {"message": "Imported"}

async def auto_sleep_task():
    """Background task to automatically pause idle models"""
    print("[AUTO-SLEEP] Started")
    while True:
        try:
            await asyncio.sleep(60)
            current_time = time.time()
            for model_name, config in configs.items():
                if not config.get("auto_sleep", False):
                    continue

                lock = get_model_lock(model_name)
                if lock.locked():
                    continue

                sleep_timeout = config.get("sleep_timeout", 300)
                last_used = last_request_time.get(model_name, 0)
                if last_used == 0:
                    continue
                idle_time = current_time - last_used
                if idle_time > sleep_timeout:
                    container_name = get_container_name(model_name)
                    try:
                        container = docker_client.containers.get(container_name)
                        if container.status == "running":
                            print(f"[AUTO-SLEEP] {model_name} idle {int(idle_time)}s - sleeping")
                            container.pause()
                    except:
                        pass
        except Exception as e:
            print(f"[AUTO-SLEEP] Error: {e}")

async def preload_models_task():
    """Background task to preload models marked for startup"""
    print("[PRELOAD] Starting preload task...")
    await asyncio.sleep(5)
    for model_name, config in configs.items():
        if config.get("preload", False):
            try:
                print(f"[PRELOAD] Starting {model_name}...")
                await start_model(model_name)
                print(f"[PRELOAD] {model_name} started successfully")
            except Exception as e:
                print(f"[PRELOAD] Failed to start {model_name}: {e}")

@app.on_event("startup")
async def startup_event():
    """Initialize background tasks on application startup"""
    asyncio.create_task(auto_sleep_task())
    asyncio.create_task(preload_models_task())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)

