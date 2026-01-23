# Contributing to LLM Proxy Manager

First off, thank you for considering contributing to LLM Proxy Manager! It's people like you that make this tool better for everyone.

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the existing issues to avoid duplicates. When you create a bug report, include as many details as possible:

- **Use a clear and descriptive title**
- **Describe the exact steps to reproduce the problem**
- **Provide specific examples** (code snippets, configuration files)
- **Describe the behavior you observed and what you expected**
- **Include screenshots** if relevant
- **Include your environment details** (OS, Docker version, GPU model, CUDA version)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion:

- **Use a clear and descriptive title**
- **Provide a step-by-step description** of the suggested enhancement
- **Provide specific examples** to demonstrate the steps
- **Describe the current behavior** and **explain the behavior you expected**
- **Explain why this enhancement would be useful**

### Pull Requests

- Fill in the required template
- Follow the Python style guide (PEP 8)
- Include comments in your code where necessary
- Update documentation if you change functionality
- Add tests for new features
- Ensure all tests pass before submitting

## Development Setup

1. Fork and clone the repository
```bash
git clone https://github.com/yourusername/llm-proxy-manager.git
cd llm-proxy-manager
```

2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install fastapi uvicorn docker httpx pydantic
```

4. Make your changes and test locally
```bash
python main.py
```

## Style Guidelines

### Python Style Guide

- Follow PEP 8
- Use meaningful variable names
- Add docstrings to functions and classes
- Keep functions focused and small
- Use type hints where appropriate

### Git Commit Messages

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters
- Reference issues and pull requests liberally

Example:
```
Add support for Mistral models

- Implement custom tokenizer handling
- Add Mistral-specific configuration options
- Update documentation with Mistral examples

Fixes #123
```

## Testing

Before submitting a pull request:

1. Test your changes with at least one model
2. Verify the web UI works correctly
3. Check that existing functionality still works
4. Test both vLLM and llama.cpp backends if applicable

## Documentation

- Update README.md if you change functionality
- Add inline comments for complex code
- Update API documentation if you change endpoints
- Include examples for new features

## Questions?

Feel free to open an issue with your question or reach out to the maintainers.

Thank you for contributing! 🎉

