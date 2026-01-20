# vLLM Proxy Manager

A Docker-based proxy manager for vLLM models with automatic model lifecycle management, GPU resource optimization, and OpenAI-compatible API.

## Features

- 🚀 **Multi-Model Support** - Manage multiple vLLM models simultaneously
- 🔄 **Automatic Model Lifecycle** - Start, stop, pause models on demand
- 💾 **Resource Optimization** - Exclusive mode with automatic model sleeping
- 🔌 **OpenAI API Compatible** - Drop-in replacement for OpenAI endpoints
- 🐳 **Docker-based** - Easy deployment with Docker Compose
- 🎛️ **Web UI** - Simple web interface for model management
- 📊 **Health Monitoring** - Real-time model status tracking

## Architecture

```
┌─────────────┐
│ Open WebUI  │
└──────┬──────┘
       │ API calls
       ↓
┌──────────────────┐
│ Proxy Manager    │  (Port 8080)
│ - Routes requests│
│ - Manages models │
└────┬─────────────┘
     │ Docker API
     ↓
┌─────────────────────┐
│ vLLM Containers     │
│ - Model A (8001)    │
│ - Model B (8002)    │
└─────────────────────┘
```

## Prerequisites

- Docker & Docker Compose
- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/vllm-proxy-manager.git
cd vllm-proxy-manager
```

### 2. Configure Models

Edit `data/models.json`:

```json
{
  "models": [
    {
      "name": "glm-45-airko",
      "model_path": "/path/to/model",
      "port": 8001,
      "gpu_memory_utilization": 0.9,
      "max_model_len": 4096,
      "served_model_name": "GLM-4.5-Air-Derestricted"
    }
  ],
  "exclusive_mode": true,
  "health_check_interval": 30
}
```

### 3. Start Proxy Manager

```bash
docker compose up -d
```

### 4. Access Web UI

Open http://localhost:8080 in your browser.

## Configuration

### Environment Variables

- `VLLM_HOST` - Host IP for vLLM containers (default: `172.17.0.1` for Docker bridge network)
- `PORT` - Proxy manager port (default: `8080`)

### Model Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `name` | Unique model identifier | Required |
| `model_path` | Path to model files | Required |
| `port` | External port mapping | Required |
| `gpu_memory_utilization` | GPU memory fraction | `0.9` |
| `max_model_len` | Maximum context length | `4096` |
| `served_model_name` | API model name | Same as `name` |
| `dtype` | Model precision | `auto` |
| `quantization` | Quantization method | `null` |

### Exclusive Mode

When enabled, only one model runs at a time. Inactive models are automatically paused to free GPU memory.

## API Endpoints

### Chat Completions

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-45-airko",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### List Models

```bash
curl http://localhost:8080/v1/models
```

### Model Management

```bash
# Start model
curl -X POST http://localhost:8080/api/models/{model_name}/start

# Stop model
curl -X POST http://localhost:8080/api/models/{model_name}/stop

# Get status
curl http://localhost:8080/api/models/{model_name}/status
```

## Troubleshooting

### Connection Errors

If you see `Connection error: All connection attempts failed`:

1. Check if vLLM container is running:
   ```bash
   docker ps | grep vllm
   ```

2. If container is paused, unpause it:
   ```bash
   docker unpause vllm_{model_name}
   ```

3. Check network connectivity:
   ```bash
   docker exec -it vllm_proxy_manager curl http://172.17.0.1:{port}/v1/models
   ```

### Model Won't Start

- Verify model path exists and is mounted correctly
- Check GPU availability: `nvidia-smi`
- Review container logs: `docker logs vllm_{model_name}`

### Memory Issues

- Reduce `gpu_memory_utilization` in model config
- Enable `exclusive_mode` to run one model at a time
- Reduce `max_model_len`

## Integration with Open WebUI

1. In Open WebUI, go to **Admin Panel** → **Settings** → **Connections**
2. Add new OpenAI API connection:
   - **API Base URL**: `http://vllm_proxy_manager:8080/v1`
   - **API Key**: (leave empty or set in proxy config)
3. Models will appear automatically in the model selector

## Development

### Project Structure

```
vllm-proxy-manager/
├── main.py                 # FastAPI application entry point
├── model_manager.py        # Docker container lifecycle management
├── config.py              # Configuration loader
├── Dockerfile             # Proxy manager container image
├── docker-compose.yml     # Deployment configuration
├── requirements.txt       # Python dependencies
├── data/
│   └── models.json       # Model definitions
└── static/
    └── index.html        # Web UI
```

### Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run proxy manager
python main.py
```

## License

MIT License - See LICENSE file for details

## Contributing

Pull requests are welcome! Please open an issue first to discuss proposed changes.

## Acknowledgments

- [vLLM](https://github.com/vllm-project/vllm) - High-performance LLM inference engine
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework for APIs
