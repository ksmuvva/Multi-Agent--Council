# Configuration Quick Reference

Quick reference for common LLM provider configurations.

## Anthropic (Claude) - Default

```bash
MAS_LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-xxxxx
```

| Agent | Default Model |
|-------|---------------|
| Orchestrator | claude-3-5-opus-20240507 |
| Council | claude-3-5-opus-20240507 |
| Analyst | claude-3-5-sonnet-20241022 |
| Verifier | claude-3-5-opus-20240507 |

## OpenAI (GPT)

```bash
MAS_LLM_PROVIDER=openai
OPENAI_API_KEY=sk-xxxxx
```

| Agent | Default Model |
|-------|---------------|
| Orchestrator | gpt-4o |
| Council | gpt-4o |
| Analyst | gpt-4o |
| Verifier | gpt-4o |

## Google (Gemini)

```bash
MAS_LLM_PROVIDER=google
GOOGLE_API_KEY=xxxxx
```

| Agent | Default Model |
|-------|---------------|
| Orchestrator | gemini-1.5-pro |
| Council | gemini-1.5-pro |
| Analyst | gemini-1.5-pro |
| Verifier | gemini-1.5-pro |

## Azure OpenAI

```bash
MAS_LLM_PROVIDER=azure_openai
AZURE_OPENAI_API_KEY=xxxxx
AZURE_OPENAI_ENDPOINT=https://xxxxx.openai.azure.com
```

Note: Set model overrides to match your deployment names.

## Local Models (Ollama)

```bash
MAS_LLM_PROVIDER=custom
CUSTOM_API_KEY=ollama
CUSTOM_BASE_URL=http://localhost:11434/v1
MAS_DEFAULT_MODEL=llama3.1
```

## Local Models (vLLM)

```bash
MAS_LLM_PROVIDER=custom
CUSTOM_API_KEY=vllm
CUSTOM_BASE_URL=http://localhost:8000/v1
MAS_DEFAULT_MODEL=meta-llama/Llama-3.1-70B-Instruct
```

## Together AI

```bash
MAS_LLM_PROVIDER=together
TOGETHER_API_KEY=xxxxx
```

| Agent | Default Model |
|-------|---------------|
| Orchestrator | meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo |
| Council | meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo |
| Analyst | meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo |
