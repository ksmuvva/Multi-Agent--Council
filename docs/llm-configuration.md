# LLM Configuration Guide

This guide explains how to configure different LLM providers for the Multi-Agent Reasoning System.

## Overview

The system supports multiple LLM providers through a unified configuration interface. You can switch between providers by setting environment variables in your `.env` file, without changing any code.

## Supported Providers

| Provider | Description | Environment Variable |
|----------|-------------|----------------------|
| **Anthropic** | Claude models (default) | `MAS_LLM_PROVIDER=anthropic` |
| **OpenAI** | GPT models | `MAS_LLM_PROVIDER=openai` |
| **Azure OpenAI** | Azure-hosted GPT models | `MAS_LLM_PROVIDER=azure_openai` |
| **Google** | Gemini models | `MAS_LLM_PROVIDER=google` |
| **Mistral AI** | Mistral models | `MAS_LLM_PROVIDER=mistral` |
| **Cohere** | Command models | `MAS_LLM_PROVIDER=cohere` |
| **Together AI** | Open-source models | `MAS_LLM_PROVIDER=together` |
| **Custom** | Any OpenAI-compatible endpoint | `MAS_LLM_PROVIDER=custom` |

## Quick Start

### Step 1: Copy the Environment Template

```bash
cp .env.example .env
```

### Step 2: Choose Your Provider

Edit `.env` and set your preferred provider:

```bash
# For Anthropic (Claude)
MAS_LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-your-key-here

# For OpenAI (GPT)
# MAS_LLM_PROVIDER=openai
# OPENAI_API_KEY=sk-your-key-here
```

### Step 3: Run the System

```bash
# CLI
mas query "Hello, world!"

# Streamlit UI
streamlit run src/ui/app.py
```

## Provider-Specific Configuration

### Anthropic (Claude)

```bash
MAS_LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-your-key-here

# Optional: For API gateway or proxy
# ANTHROPIC_BASE_URL=https://your-gateway.com
```

**Default Models:**
- Orchestrator/Council: `claude-3-5-opus-20240507`
- Operational Agents: `claude-3-5-sonnet-20241022`
- Light Agents: `claude-3-5-haiku-20241022`

### OpenAI (GPT)

```bash
MAS_LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here

# Optional: For custom endpoint
# OPENAI_BASE_URL=https://your-proxy.com/v1

# Optional: Organization ID
# OPENAI_ORGANIZATION=org-your-id
```

**Default Models:**
- Orchestrator/Council: `gpt-4o`
- Operational Agents: `gpt-4o`
- Light Agents: `gpt-4o-mini`

### Azure OpenAI

```bash
MAS_LLM_PROVIDER=azure_openai
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_API_VERSION=2024-02-15-preview

# Optional: Default deployment name
# AZURE_OPENAI_DEPLOYMENT=your-deployment
```

**Note:** You may need to set model overrides to match your Azure deployment names:

```bash
MAS_ORCHESTRATOR_MODEL=your-gpt4-deployment
MAS_ANALYST_MODEL=your-gpt4-deployment
```

### Google (Gemini)

```bash
MAS_LLM_PROVIDER=google
GOOGLE_API_KEY=your-api-key

# Optional: For Vertex AI
# GOOGLE_PROJECT_ID=your-project-id
# GOOGLE_LOCATION=us-central1
```

**Default Models:**
- Orchestrator/Council: `gemini-1.5-pro`
- Operational Agents: `gemini-1.5-pro`
- Light Agents: `gemini-1.5-flash`

### Mistral AI

```bash
MAS_LLM_PROVIDER=mistral
MISTRAL_API_KEY=your-api-key

# Optional: For custom endpoint
# MISTRAL_BASE_URL=https://your-proxy.com
```

**Default Models:**
- Orchestrator/Council: `mistral-large-latest`
- Operational Agents: `mistral-large-latest`
- Light Agents: `mistral-medium-latest`

### Cohere

```bash
MAS_LLM_PROVIDER=cohere
COHERE_API_KEY=your-api-key

# Optional: For custom endpoint
# COHERE_BASE_URL=https://your-proxy.com
```

**Default Models:**
- Orchestrator/Council: `command-r-plus`
- Operational Agents: `command-r-plus`
- Light Agents: `command-r`

### Together AI

```bash
MAS_LLM_PROVIDER=together
TOGETHER_API_KEY=your-api-key

# Optional: For custom endpoint
# TOGETHER_BASE_URL=https://your-proxy.com
```

**Default Models:**
- Orchestrator/Council: `meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo`
- Operational Agents: `meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo`
- Light Agents: `meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo`

### Custom (OpenAI-Compatible)

Use this for:
- Local models (Ollama, LM Studio, vLLM)
- Other cloud providers
- Custom endpoints

```bash
MAS_LLM_PROVIDER=custom
CUSTOM_API_KEY=your-api-key-or-dummy
CUSTOM_BASE_URL=https://your-endpoint.com/v1

# Optional: Prefix for model names
# CUSTOM_MODEL_PREFIX=
```

**Example for Ollama:**
```bash
MAS_LLM_PROVIDER=custom
CUSTOM_API_KEY=ollama
CUSTOM_BASE_URL=http://localhost:11434/v1
MAS_DEFAULT_MODEL=llama3.1
```

**Example for vLLM:**
```bash
MAS_LLM_PROVIDER=custom
CUSTOM_API_KEY=vllm
CUSTOM_BASE_URL=http://localhost:8000/v1
MAS_DEFAULT_MODEL=meta-llama/Llama-3.1-70B-Instruct
```

## Model Override Configuration

You can override the default model for any specific agent:

```bash
# Override specific agent models
MAS_ORCHESTRATOR_MODEL=claude-3-5-opus-20240507
MAS_COUNCIL_MODEL=claude-3-5-opus-20240507
MAS_ANALYST_MODEL=claude-3-5-sonnet-20241022
MAS_PLANNER_MODEL=claude-3-5-sonnet-20241022
MAS_CLARIFIER_MODEL=claude-3-5-haiku-20241022
MAS_RESEARCHER_MODEL=claude-3-5-sonnet-20241022
MAS_EXECUTOR_MODEL=claude-3-5-sonnet-20241022
MAS_CODE_REVIEWER_MODEL=claude-3-5-sonnet-20241022
MAS_VERIFIER_MODEL=claude-3-5-opus-20240507
MAS_CRITIC_MODEL=claude-3-5-opus-20240507
MAS_REVIEWER_MODEL=claude-3-5-opus-20240507
MAS_FORMATTER_MODEL=claude-3-5-sonnet-20241022
MAS_MEMORY_CURATOR_MODEL=claude-3-5-sonnet-20241022
MAS_SME_MODEL=claude-3-5-sonnet-20241022
```

## Switching Providers

To switch providers:

1. **Change the provider setting:**
   ```bash
   MAS_LLM_PROVIDER=openai  # Switch from Anthropic to OpenAI
   ```

2. **Add the new API key:**
   ```bash
   OPENAI_API_KEY=sk-your-key-here
   ```

3. **(Optional) Remove old API key:**
   ```bash
   # ANTHROPIC_API_KEY=sk-ant-old-key  # Comment out or remove
   ```

4. **Restart your application:**
   ```bash
   # CLI - just run again
   mas query "Test query"

   # Streamlit - restart the server
   # Press Ctrl+C and run: streamlit run src/ui/app.py
   ```

## Validating Configuration

Test your configuration using the CLI:

```bash
# Check system status (shows configured provider)
mas status

# Run a simple test query
mas query "What is 2 + 2?" --verbose

# List available tools
mas tools --verbose
```

## Common Issues

### Issue: "API key not configured"

**Solution:** Make sure you've set the correct API key for your provider:

```bash
# For Anthropic
ANTHROPIC_API_KEY=sk-ant-your-key

# For OpenAI
OPENAI_API_KEY=sk-your-key
```

### Issue: "Model not found"

**Solution:** Some providers use different model IDs. You may need to set model overrides:

```bash
# For Azure OpenAI, use deployment names
MAS_ORCHESTRATOR_MODEL=your-deployment-name

# For Together AI, use full model paths
MAS_ORCHESTRATOR_MODEL=meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo
```

### Issue: "Connection timeout" or "Base URL error"

**Solution:** Check your base URL configuration:

```bash
# For custom endpoints
CUSTOM_BASE_URL=https://correct-endpoint.com/v1
```

## Cost Optimization

Different providers have different costs. Use lighter models for less critical tasks:

```bash
# Use cheaper models for simple agents
MAS_CLARIFIER_MODEL=claude-3-5-haiku-20241022
MAS_FORMATTER_MODEL=claude-3-5-haiku-20241022

# Use premium models for critical tasks
MAS_VERIFIER_MODEL=claude-3-5-opus-20240507
MAS_CRITIC_MODEL=claude-3-5-opus-20240507
```

## Programmatic Configuration

You can also configure the system programmatically:

```python
from src.config import get_settings, reload_settings

# Get current settings
settings = get_settings()
print(f"Provider: {settings.llm_provider}")
print(f"API Key configured: {settings.validate_api_key()}")

# Get model for specific agent
from src.config import get_model_for_agent
model = get_model_for_agent("orchestrator")
print(f"Orchestrator model: {model}")

# Reload settings after changing environment
import os
os.environ["MAS_LLM_PROVIDER"] = "openai"
os.environ["OPENAI_API_KEY"] = "sk-new-key"
reload_settings()
```

## Security Best Practices

1. **Never commit `.env` files** to version control
2. **Use different API keys** for development and production
3. **Rotate API keys regularly**
4. **Set budget limits** to prevent unexpected charges:
   ```bash
   MAS_MAX_BUDGET=5.00
   MAS_BUDGET_WARNING_THRESHOLD=0.8
   ```

## Additional Configuration

See `.env.example` for all available configuration options, including:

- Budget and cost control
- Agent configuration (max turns, SME count, etc.)
- Logging configuration
- UI configuration
- Feature flags
- Development settings
