# 🚀 LLM Proxy Manager

A powerful Docker-based management system for running multiple Large Language Models (LLMs) with automatic resource management, load balancing, and an intuitive web interface.

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## ✨ Features

### 🎯 Core Features
- **Multi-Model Management**: Run multiple LLM models simultaneously with isolated Docker containers
- **Dual Backend Support**: Both vLLM and llama.cpp inference backends
- **Auto-Sleep Mode**: Automatically pause idle models to save GPU memory
- **Exclusive Mode**: Run resource-intensive models with exclusive GPU access
- **OpenAI-Compatible API**: Drop-in replacement for OpenAI API endpoints
- **Hot-Swapping**: Seamlessly switch between models without downtime

### 🛠️ Advanced Features
- **Tensor Parallelism**: Distribute models across multiple GPUs
- **Custom Quantization**: Support for AWQ, GPTQ, MXFP4, INT8, FP8
- **Request Caching**: Intelligent caching during model loading
- **Funny Loading Messages**: 50+ entertaining messages during model initialization
- **Real-time Statistics**: Track usage, tokens/sec, and request counts
- **Docker Integration**: Full Docker API integration for container management

### 🎨 User Interface
- **Modern Web UI**: Beautiful gradient interface with real-time updates
- **Model Cards**: Visual representation of each model's status and stats
- **One-Click Actions**: Start, stop, sleep, wake models with single clicks
- **Config Import/Export**: Backup and restore your model configurations
- **Live Monitoring**: Health checks and running model overview

## 📋 Requirements

- Docker with NVIDIA GPU support
- Python 3.11+
- NVIDIA GPU with CUDA support
- At least 16GB GPU VRAM (recommended 24GB+)

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/llm-proxy-manager.git
cd llm-proxy-manager
```

2. Create data directory:
```bash
mkdir -p data
```

3. Build and start the container:
```bash
docker-compose up -d --build
```

4. Access the web interface:
```
http://localhost:8080
```

### Adding Your First Model

1. Click the **+** button in the bottom-right corner
2. Fill in the model configuration:
   - **Name**: A unique identifier for your model
   - **Backend**: Choose vLLM or llama.cpp
   - **Model Path**: HuggingFace model ID or local path
   - **Port**: Unique port for this model (e.g., 8001)
3. Configure advanced options (optional):
   - GPU memory utilization
   - Tensor parallel size
   - Quantization method
   - Auto-sleep timeout
4. Click **Save** and then **Start** to launch the model

## 📚 Usage

### OpenAI-Compatible API

The proxy exposes an OpenAI-compatible API endpoint:

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="your-model-name",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)

print(response.choices[0].message.content)
```

### Streaming Responses

```python
stream = client.chat.completions.create(
    model="your-model-name",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### Model Management API

#### List all models
```bash
curl http://localhost:8080/api/models
```

#### Start a model
```bash
curl -X POST http://localhost:8080/api/models/your-model-name/start
```

#### Stop a model
```bash
curl -X POST http://localhost:8080/api/models/your-model-name/stop
```

#### Sleep a model (pause to free GPU memory)
```bash
curl -X POST http://localhost:8080/api/models/your-model-name/sleep
```

#### Wake a model
```bash
curl -X POST http://localhost:8080/api/models/your-model-name/wake
```

## ⚙️ Configuration

### Model Configuration Options

#### Basic Settings
- **name**: Unique model identifier
- **backend**: `vllm` or `llamacpp`
- **model_path**: HuggingFace ID or local filesystem path
- **is_local**: Set to `true` for local models
- **port**: Unique port number (8001-9000 recommended)

#### vLLM Settings
- **gpu_memory_utilization**: GPU memory fraction (0.0-1.0)
- **max_model_len**: Maximum context length
- **quantization**: Quantization method (awq, gptq, etc.)
- **tensor_parallel_size**: Number of GPUs for tensor parallelism

#### llama.cpp Settings
- **context_size**: Context window size
- **gpu_layers**: Number of layers offloaded to GPU
- **parallel_requests**: Number of parallel inference requests

#### Advanced Options
- **exclusive**: Stop all other models when this one starts
- **auto_sleep**: Automatically pause after idle timeout
- **sleep_timeout**: Seconds before auto-sleep (default: 300)
- **always_visible**: Show in model list even when stopped
- **preload**: Auto-start on proxy manager startup

#### Custom Arguments
- **custom_env_vars**: Environment variables for Docker container
- **custom_vllm_args**: Additional vLLM command-line arguments
- **custom_llamacpp_args**: Additional llama.cpp arguments
- **custom_docker_args**: Additional Docker runtime arguments
- **docker_image**: Override default Docker image

### Example Configuration

```json
{
  "my-model": {
    "name": "my-model",
    "backend": "vllm",
    "model_path": "meta-llama/Llama-2-7b-chat-hf",
    "is_local": false,
    "gpu_memory_utilization": 0.9,
    "max_model_len": 4096,
    "tensor_parallel_size": 2,
    "port": 8001,
    "exclusive": false,
    "auto_sleep": true,
    "sleep_timeout": 300,
    "preload": true
  }
}
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Web Browser                         │
│              (http://localhost:8080)                    │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              LLM Proxy Manager (FastAPI)                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   Web UI     │  │  OpenAI API  │  │  Management  │ │
│  │   Server     │  │   Endpoint   │  │     API      │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
│  ┌──────────────────────────────────────────────────┐  │
│  │         Docker Container Manager                 │  │
│  └──────────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────────────┘
                     │
          ┌──────────┼──────────┐
          │          │          │
          ▼          ▼          ▼
    ┌─────────┐ ┌─────────┐ ┌─────────┐
    │ vLLM    │ │ vLLM    │ │llama.cpp│
    │ Model 1 │ │ Model 2 │ │ Model 3 │
    │ (GPU 0) │ │(GPU 0,1)│ │ (GPU 2) │
    └─────────┘ └─────────┘ └─────────┘
```

## 🔧 Troubleshooting

### Model fails to start
- Check GPU memory availability: `nvidia-smi`
- Verify Docker has GPU access: `docker run --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi`
- Check container logs: `docker logs llm_<model_name>`

### Connection errors
- Ensure ports aren't conflicting with other services
- Check firewall settings
- Verify `VLLM_HOST` environment variable points to correct address

### Out of memory errors
- Reduce `gpu_memory_utilization`
- Enable auto-sleep for unused models
- Use quantization (AWQ/GPTQ)
- Reduce `max_model_len`

### Slow loading times
- Use local models instead of downloading from HuggingFace
- Enable model preloading for frequently used models
- Check disk I/O performance

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [vLLM](https://github.com/vllm-project/vllm) - Fast LLM inference engine
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - Efficient LLM inference in C++
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [Docker](https://www.docker.com/) - Containerization platform

## 📧 Support

For issues, questions, or suggestions, please open an issue on GitHub.

---

Made with ❤️ by the LLM Proxy Manager team

