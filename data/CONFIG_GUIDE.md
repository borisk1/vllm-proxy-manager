# models.json Configuration Guide

## Basic Structure

```json
{
  "models": [
    {
      "name": "model-name",
      "model_path": "/path/to/model",
      "port": 8001,
      "gpu_memory_utilization": 0.9,
      "max_model_len": 4096,
      "served_model_name": "model-name",
      "dtype": "auto",
      "quantization": null,
      "trust_remote_code": false
    }
  ],
  "exclusive_mode": true,
  "health_check_interval": 30
}
```

## Parameters Explained

### Model Configuration

- **name**: Unique identifier for the model (used internally)
- **model_path**: Absolute path to model directory on host
- **port**: External port to expose (8001, 8002, 8003, etc.)
- **gpu_memory_utilization**: GPU memory fraction (0.0-1.0)
- **max_model_len**: Maximum context length
- **served_model_name**: Name used in API requests (what clients will see)
- **dtype**: Data type ("auto", "float16", "bfloat16", "float32")
- **quantization**: Quantization method (null, "awq", "gptq", "squeezellm")
- **trust_remote_code**: Allow custom model code execution (true/false)

### Global Settings

- **exclusive_mode**: Only one model runs at a time (saves GPU memory)
- **health_check_interval**: Seconds between health checks

## Example Configurations

### Single Model (Basic)

```json
{
  "models": [
    {
      "name": "llama-3-8b",
      "model_path": "/root/modeli/meta-llama/Llama-3-8B-Instruct",
      "port": 8001,
      "served_model_name": "llama-3-8b-instruct",
      "gpu_memory_utilization": 0.9,
      "max_model_len": 8192
    }
  ],
  "exclusive_mode": true,
  "health_check_interval": 30
}
```

### Multiple Models

```json
{
  "models": [
    {
      "name": "glm-45-airko",
      "model_path": "/root/modeli/GLM-4.5-Air-AWQ",
      "port": 8001,
      "served_model_name": "GLM-4.5-Air-Derestricted",
      "gpu_memory_utilization": 0.8,
      "max_model_len": 4096,
      "trust_remote_code": true
    },
    {
      "name": "gpt-oss-120b",
      "model_path": "/root/.cache/huggingface/hub/models--justinjja--gpt-oss-120b-Derestricted-MXFP4/snapshots/a7fddca0dc0da89b367b97bb0f118bfc2179bee1",
      "port": 8002,
      "served_model_name": "gpt-oss-120b-Derestricted-MXFP4",
      "gpu_memory_utilization": 0.75,
      "max_model_len": 131072,
      "trust_remote_code": true
    }
  ],
  "exclusive_mode": true,
  "health_check_interval": 30
}
```

### Custom Quantized Model

```json
{
  "models": [
    {
      "name": "mistral-7b-awq",
      "model_path": "/root/modeli/mistral-7b-instruct-awq",
      "port": 8001,
      "served_model_name": "mistral-7b-instruct",
      "gpu_memory_utilization": 0.9,
      "max_model_len": 32768,
      "quantization": "awq",
      "dtype": "float16"
    }
  ],
  "exclusive_mode": false,
  "health_check_interval": 30
}
```

## Tips

1. **Port Selection**: Each model needs a unique port (8001, 8002, 8003, etc.)

2. **GPU Memory**: Sum of all `gpu_memory_utilization` should not exceed 1.0 if `exclusive_mode` is false

3. **Model Paths**: 
   - Use absolute paths
   - For HuggingFace cache: `/root/.cache/huggingface/hub/models--{org}--{model}/snapshots/{hash}`
   - For local models: `/root/modeli/{model-name}`

4. **Trust Remote Code**: Required for:
   - Custom model architectures
   - Models with custom tokenizers
   - Models like GPT-OSS, GLM, etc.

5. **Exclusive Mode**: 
   - `true`: Only one model active at a time (saves memory)
   - `false`: Multiple models can run simultaneously

## Troubleshooting

**Model not found error**: Check that `model_path` exists and is readable

**Port already in use**: Change port number to unused one

**Out of memory**: Reduce `gpu_memory_utilization` or enable `exclusive_mode`

**Model architecture error**: Set `trust_remote_code: true`
