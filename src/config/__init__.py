"""
Configuration Module for Multi-Agent Reasoning System

Provides centralized configuration management supporting multiple
LLM providers through a unified interface.

Usage:
    from src.config import get_settings, get_model_for_agent

    settings = get_settings()
    model = get_model_for_agent("orchestrator")
    api_key = settings.get_api_key()
"""

from .settings import (
    LLMProvider,
    ModelMapping,
    Settings,
    DEFAULT_MODEL_MAPPINGS,
    get_settings,
    reload_settings,
    get_api_key,
    get_model_for_agent,
    get_provider,
    get_provider_config,
)

__all__ = [
    "LLMProvider",
    "ModelMapping",
    "Settings",
    "DEFAULT_MODEL_MAPPINGS",
    "get_settings",
    "reload_settings",
    "get_api_key",
    "get_model_for_agent",
    "get_provider",
    "get_provider_config",
]
