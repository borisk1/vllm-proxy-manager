# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-01-24

### Added
- Initial release of LLM Proxy Manager
- Multi-model management with Docker containers
- Support for vLLM and llama.cpp backends
- OpenAI-compatible API endpoints
- Auto-sleep mode for idle models
- Exclusive mode for resource-intensive models
- Tensor parallelism support
- Custom quantization support (AWQ, GPTQ, MXFP4, INT8, FP8)
- Web-based management interface
- Real-time statistics tracking
- Config import/export functionality
- 50+ funny loading messages during model initialization
- Health check and monitoring endpoints
- Model preloading on startup
- Request caching during model loading
- Custom environment variable support
- Custom Docker arguments support

### Features
- Beautiful gradient web UI with real-time updates
- One-click model start/stop/sleep/wake actions
- Visual model cards with status badges
- Live token/sec statistics
- Model usage tracking
- Automatic GPU memory management
- Hot-swapping between models
- Extended timeout handling (600 seconds)
- Retry logic with exponential backoff
- Streaming response support with progress updates

### Documentation
- Comprehensive README with examples
- API documentation
- Configuration guide
- Troubleshooting section
- Contributing guidelines

## [Unreleased]

### Planned Features
- Multi-user support with authentication
- Model performance benchmarking
- Automatic model selection based on query
- Cost tracking and budgeting
- Model fine-tuning integration
- Kubernetes deployment support
- Prometheus metrics export
- Grafana dashboard templates

