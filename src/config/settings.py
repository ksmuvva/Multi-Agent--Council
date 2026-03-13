"""
Centralized Configuration Management for Multi-Agent Reasoning System

This module provides a unified configuration system that supports multiple
LLM providers through a single interface. Uses pydantic-settings for
type-safe environment variable loading with validation.

Supported LLM Providers:
- Anthropic (Claude models)
- OpenAI (GPT models)
- Azure OpenAI
- Google (Gemini models)
- Mistral AI
- Cohere
- Together AI
- Any OpenAI-compatible endpoint

Usage:
    from src.config.settings import get_settings, settings

    # Access current settings
    settings = get_settings()
    api_key = settings.anthropic_api_key

    # Get model for specific agent
    model = settings.get_model_for_agent("orchestrator")

    # Switch provider (requires restart)
    # Set MAS_LLM_PROVIDER=openai in .env file
"""

import os
from typing import Optional, Dict, List, Literal, Any
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
    from pydantic import Field, AliasChoices
    PYDANTIC_SETTINGS_AVAILABLE = True
except ImportError:
    from pydantic import BaseSettings, Field
    PYDANTIC_SETTINGS_AVAILABLE = False


# =============================================================================
# LLM Provider Enumeration
# =============================================================================

class LLMProvider(str, Enum):
    """Supported LLM providers."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    GOOGLE = "google"
    MISTRAL = "mistral"
    COHERE = "cohere"
    TOGETHER = "together"
    GLM = "glm"  # Zhipu AI GLM models (ChatGLM/GLM-4)
    CUSTOM = "custom"  # For OpenAI-compatible endpoints


# =============================================================================
# Model Registry
# =============================================================================

@dataclass
class ModelMapping:
    """Maps logical agent names to provider-specific model IDs."""
    provider: LLMProvider
    models: Dict[str, str]  # Logical name -> Model ID

    def get_model(self, logical_name: str, default: Optional[str] = None) -> Optional[str]:
        """Get model ID for logical name."""
        return self.models.get(logical_name, default)


# Default model mappings for each provider
DEFAULT_MODEL_MAPPINGS: Dict[LLMProvider, ModelMapping] = {
    LLMProvider.ANTHROPIC: ModelMapping(
        provider=LLMProvider.ANTHROPIC,
        models={
            "default": "claude-3-5-sonnet-20241022",
            "orchestrator": "claude-3-5-opus-20240507",
            "council": "claude-3-5-opus-20240507",
            "analyst": "claude-3-5-sonnet-20241022",
            "planner": "claude-3-5-sonnet-20241022",
            "clarifier": "claude-3-5-haiku-20241022",
            "researcher": "claude-3-5-sonnet-20241022",
            "executor": "claude-3-5-sonnet-20241022",
            "code_reviewer": "claude-3-5-sonnet-20241022",
            "verifier": "claude-3-5-opus-20240507",
            "critic": "claude-3-5-opus-20240507",
            "reviewer": "claude-3-5-opus-20240507",
            "formatter": "claude-3-5-sonnet-20241022",
            "memory_curator": "claude-3-5-sonnet-20241022",
            "sme": "claude-3-5-sonnet-20241022",
        },
    ),
    LLMProvider.OPENAI: ModelMapping(
        provider=LLMProvider.OPENAI,
        models={
            "default": "gpt-4o",
            "orchestrator": "gpt-4o",
            "council": "gpt-4o",
            "analyst": "gpt-4o",
            "planner": "gpt-4o",
            "clarifier": "gpt-4o-mini",
            "researcher": "gpt-4o",
            "executor": "gpt-4o",
            "code_reviewer": "gpt-4o",
            "verifier": "gpt-4o",
            "critic": "gpt-4o",
            "reviewer": "gpt-4o",
            "formatter": "gpt-4o",
            "memory_curator": "gpt-4o",
            "sme": "gpt-4o",
        },
    ),
    LLMProvider.AZURE_OPENAI: ModelMapping(
        provider=LLMProvider.AZURE_OPENAI,
        models={
            "default": "gpt-4o",
            "orchestrator": "gpt-4o",
            "council": "gpt-4o",
            "analyst": "gpt-4o",
            "planner": "gpt-4o",
            "clarifier": "gpt-4o-mini",
            "researcher": "gpt-4o",
            "executor": "gpt-4o",
            "code_reviewer": "gpt-4o",
            "verifier": "gpt-4o",
            "critic": "gpt-4o",
            "reviewer": "gpt-4o",
            "formatter": "gpt-4o",
            "memory_curator": "gpt-4o",
            "sme": "gpt-4o",
        },
    ),
    LLMProvider.GOOGLE: ModelMapping(
        provider=LLMProvider.GOOGLE,
        models={
            "default": "gemini-1.5-pro",
            "orchestrator": "gemini-1.5-pro",
            "council": "gemini-1.5-pro",
            "analyst": "gemini-1.5-flash",
            "planner": "gemini-1.5-pro",
            "clarifier": "gemini-1.5-flash",
            "researcher": "gemini-1.5-pro",
            "executor": "gemini-1.5-pro",
            "code_reviewer": "gemini-1.5-pro",
            "verifier": "gemini-1.5-pro",
            "critic": "gemini-1.5-pro",
            "reviewer": "gemini-1.5-pro",
            "formatter": "gemini-1.5-flash",
            "memory_curator": "gemini-1.5-flash",
            "sme": "gemini-1.5-pro",
        },
    ),
    LLMProvider.MISTRAL: ModelMapping(
        provider=LLMProvider.MISTRAL,
        models={
            "default": "mistral-large-latest",
            "orchestrator": "mistral-large-latest",
            "council": "mistral-large-latest",
            "analyst": "mistral-large-latest",
            "planner": "mistral-large-latest",
            "clarifier": "mistral-medium-latest",
            "researcher": "mistral-large-latest",
            "executor": "mistral-large-latest",
            "code_reviewer": "mistral-large-latest",
            "verifier": "mistral-large-latest",
            "critic": "mistral-large-latest",
            "reviewer": "mistral-large-latest",
            "formatter": "mistral-medium-latest",
            "memory_curator": "mistral-medium-latest",
            "sme": "mistral-large-latest",
        },
    ),
    LLMProvider.COHERE: ModelMapping(
        provider=LLMProvider.COHERE,
        models={
            "default": "command-r-plus",
            "orchestrator": "command-r-plus",
            "council": "command-r-plus",
            "analyst": "command-r",
            "planner": "command-r-plus",
            "clarifier": "command-r",
            "researcher": "command-r-plus",
            "executor": "command-r-plus",
            "code_reviewer": "command-r-plus",
            "verifier": "command-r-plus",
            "critic": "command-r-plus",
            "reviewer": "command-r-plus",
            "formatter": "command-r",
            "memory_curator": "command-r",
            "sme": "command-r-plus",
        },
    ),
    LLMProvider.TOGETHER: ModelMapping(
        provider=LLMProvider.TOGETHER,
        models={
            "default": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            "orchestrator": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
            "council": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
            "analyst": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            "planner": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            "clarifier": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            "researcher": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            "executor": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            "code_reviewer": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            "verifier": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            "critic": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            "reviewer": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            "formatter": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            "memory_curator": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            "sme": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        },
    ),
    LLMProvider.GLM: ModelMapping(
        provider=LLMProvider.GLM,
        models={
            "default": "glm-4-plus",
            "orchestrator": "glm-4-plus",
            "council": "glm-4-plus",
            "analyst": "glm-4-plus",
            "planner": "glm-4-plus",
            "clarifier": "glm-4-flash",
            "researcher": "glm-4-plus",
            "executor": "glm-4-plus",
            "code_reviewer": "glm-4-plus",
            "verifier": "glm-4-plus",
            "critic": "glm-4-plus",
            "reviewer": "glm-4-plus",
            "formatter": "glm-4-flash",
            "memory_curator": "glm-4-flash",
            "sme": "glm-4-plus",
        },
    ),
    LLMProvider.CUSTOM: ModelMapping(
        provider=LLMProvider.CUSTOM,
        models={
            "default": "default-model",
            "orchestrator": "orchestrator-model",
            "council": "council-model",
            "analyst": "analyst-model",
            "planner": "planner-model",
            "clarifier": "clarifier-model",
            "researcher": "researcher-model",
            "executor": "executor-model",
            "code_reviewer": "code_reviewer-model",
            "verifier": "verifier-model",
            "critic": "critic-model",
            "reviewer": "reviewer-model",
            "formatter": "formatter-model",
            "memory_curator": "memory_curator-model",
            "sme": "sme-model",
        },
    ),
}


# =============================================================================
# Settings Configuration
# =============================================================================

class Settings(BaseSettings):
    """
    Centralized configuration for the Multi-Agent Reasoning System.

    Loads configuration from environment variables with sensible defaults.
    Supports multiple LLM providers through a unified interface.

    Environment variables:
        MAS_LLM_PROVIDER: Which LLM provider to use (anthropic, openai, etc.)
        {PROVIDER}_API_KEY: API key for the provider (e.g., ANTHROPIC_API_KEY)
        MAS_DEFAULT_MODEL: Default model to use (fallback)
        MAS_{AGENT}_MODEL: Override model for specific agent
    """

    if PYDANTIC_SETTINGS_AVAILABLE:
        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            env_prefix="",
            case_sensitive=False,
            extra="ignore",
        )
    else:
        # Fallback for older pydantic
        class Config:
            env_file = ".env"
            env_file_encoding = "utf-8"
            case_sensitive = False
            extra = "ignore"

    # =========================================================================
    # LLM Provider Selection
    # =========================================================================
    llm_provider: LLMProvider = Field(
        default=LLMProvider.ANTHROPIC,
        validation_alias=AliasChoices("llm_provider", "MAS_LLM_PROVIDER", "LLM_PROVIDER"),
    )

    # =========================================================================
    # Anthropic Configuration
    # =========================================================================
    anthropic_api_key: Optional[str] = None
    anthropic_base_url: Optional[str] = None  # For proxy/API gateway

    # =========================================================================
    # OpenAI Configuration
    # =========================================================================
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None  # For custom endpoint
    openai_organization: Optional[str] = None

    # =========================================================================
    # Azure OpenAI Configuration
    # =========================================================================
    azure_openai_api_key: Optional[str] = None
    azure_openai_endpoint: Optional[str] = None
    azure_openai_api_version: str = "2024-02-15-preview"
    azure_openai_deployment: Optional[str] = None  # Default deployment name

    # =========================================================================
    # Google Configuration
    # =========================================================================
    google_api_key: Optional[str] = None
    google_project_id: Optional[str] = None
    google_location: Optional[str] = None  # e.g., "us-central1"

    # =========================================================================
    # Mistral Configuration
    # =========================================================================
    mistral_api_key: Optional[str] = None
    mistral_base_url: Optional[str] = None

    # =========================================================================
    # Cohere Configuration
    # =========================================================================
    cohere_api_key: Optional[str] = None
    cohere_base_url: Optional[str] = None

    # =========================================================================
    # Together AI Configuration
    # =========================================================================
    together_api_key: Optional[str] = None
    together_base_url: Optional[str] = None

    # =========================================================================
    # GLM (Zhipu AI) Configuration
    # =========================================================================
    glm_api_key: Optional[str] = None
    glm_base_url: Optional[str] = None  # Default: https://open.bigmodel.cn/api/paas/v4

    # =========================================================================
    # Custom/OpenAI-Compatible Configuration
    # =========================================================================
    custom_api_key: Optional[str] = None
    custom_base_url: str = "https://api.openai.com/v1"
    custom_model_prefix: str = ""  # Prefix for model names if needed

    # =========================================================================
    # Model Configuration (Legacy/Override Support)
    # =========================================================================
    # These are used if MAS_LLM_PROVIDER is not set or for overrides
    default_model: Optional[str] = Field(
        default=None, validation_alias=AliasChoices("default_model", "MAS_DEFAULT_MODEL", "DEFAULT_MODEL"))
    orchestrator_model: Optional[str] = Field(
        default=None, validation_alias=AliasChoices("orchestrator_model", "MAS_ORCHESTRATOR_MODEL", "ORCHESTRATOR_MODEL"))
    council_model: Optional[str] = Field(
        default=None, validation_alias=AliasChoices("council_model", "MAS_COUNCIL_MODEL", "COUNCIL_MODEL"))
    analyst_model: Optional[str] = Field(
        default=None, validation_alias=AliasChoices("analyst_model", "MAS_ANALYST_MODEL", "ANALYST_MODEL"))
    planner_model: Optional[str] = Field(
        default=None, validation_alias=AliasChoices("planner_model", "MAS_PLANNER_MODEL", "PLANNER_MODEL"))
    clarifier_model: Optional[str] = Field(
        default=None, validation_alias=AliasChoices("clarifier_model", "MAS_CLARIFIER_MODEL", "CLARIFIER_MODEL"))
    researcher_model: Optional[str] = Field(
        default=None, validation_alias=AliasChoices("researcher_model", "MAS_RESEARCHER_MODEL", "RESEARCHER_MODEL"))
    executor_model: Optional[str] = Field(
        default=None, validation_alias=AliasChoices("executor_model", "MAS_EXECUTOR_MODEL", "EXECUTOR_MODEL"))
    code_reviewer_model: Optional[str] = Field(
        default=None, validation_alias=AliasChoices("code_reviewer_model", "MAS_CODE_REVIEWER_MODEL", "CODE_REVIEWER_MODEL"))
    verifier_model: Optional[str] = Field(
        default=None, validation_alias=AliasChoices("verifier_model", "MAS_VERIFIER_MODEL", "VERIFIER_MODEL"))
    critic_model: Optional[str] = Field(
        default=None, validation_alias=AliasChoices("critic_model", "MAS_CRITIC_MODEL", "CRITIC_MODEL"))
    reviewer_model: Optional[str] = Field(
        default=None, validation_alias=AliasChoices("reviewer_model", "MAS_REVIEWER_MODEL", "REVIEWER_MODEL"))
    formatter_model: Optional[str] = Field(
        default=None, validation_alias=AliasChoices("formatter_model", "MAS_FORMATTER_MODEL", "FORMATTER_MODEL"))
    memory_curator_model: Optional[str] = Field(
        default=None, validation_alias=AliasChoices("memory_curator_model", "MAS_MEMORY_CURATOR_MODEL", "MEMORY_CURATOR_MODEL"))
    sme_model: Optional[str] = Field(
        default=None, validation_alias=AliasChoices("sme_model", "MAS_SME_MODEL", "SME_MODEL"))

    # =========================================================================
    # Budget and Cost Control
    # =========================================================================
    max_budget: float = Field(
        default=5.0, validation_alias=AliasChoices("max_budget", "MAS_MAX_BUDGET", "MAX_BUDGET"))
    budget_warning_threshold: float = Field(
        default=0.8, validation_alias=AliasChoices("budget_warning_threshold", "MAS_BUDGET_WARNING_THRESHOLD", "BUDGET_WARNING_THRESHOLD"))

    # =========================================================================
    # Agent Configuration
    # =========================================================================
    max_turns_orchestrator: int = Field(
        default=200, validation_alias=AliasChoices("max_turns_orchestrator", "MAS_MAX_TURNS_ORCHESTRATOR", "MAX_TURNS_ORCHESTRATOR"))
    max_turns_subagent: int = Field(
        default=30, validation_alias=AliasChoices("max_turns_subagent", "MAS_MAX_TURNS_SUBAGENT", "MAX_TURNS_SUBAGENT"))
    max_turns_executor: int = Field(
        default=50, validation_alias=AliasChoices("max_turns_executor", "MAS_MAX_TURNS_EXECUTOR", "MAX_TURNS_EXECUTOR"))
    max_sme_count: int = 3
    sme_auto_spawn: bool = True

    # =========================================================================
    # Logging Configuration
    # =========================================================================
    log_level: str = "INFO"
    log_format: str = "json"
    log_file: Optional[str] = None

    # =========================================================================
    # UI Configuration
    # =========================================================================
    streamlit_server_headless: bool = True
    streamlit_server_port: int = 8501
    streamlit_server_address: str = "localhost"

    # =========================================================================
    # Session Management
    # =========================================================================
    session_timeout: int = 60
    session_persistence: bool = True

    # =========================================================================
    # Feature Flags
    # =========================================================================
    enable_council: bool = True
    enable_sme: bool = True
    enable_code_reviewer: bool = True
    enable_memory_curator: bool = True
    enable_debate: bool = True
    enable_ensemble: bool = True
    enable_cost_tracking: bool = True

    # =========================================================================
    # Development Settings
    # =========================================================================
    debug: bool = False
    dev_mode: bool = False
    run_integration: bool = False

    # =========================================================================
    # Provider-Agnostic Methods
    # =========================================================================

    def get_api_key(self) -> str:
        """
        Get the API key for the current provider.

        Returns:
            API key string

        Raises:
            ValueError: If no API key is configured for the provider
        """
        provider = self.llm_provider

        # Mapping of providers to their API key attributes
        api_key_map = {
            LLMProvider.ANTHROPIC: ("anthropic_api_key", "ANTHROPIC_API_KEY"),
            LLMProvider.OPENAI: ("openai_api_key", "OPENAI_API_KEY"),
            LLMProvider.AZURE_OPENAI: ("azure_openai_api_key", "AZURE_OPENAI_API_KEY"),
            LLMProvider.GOOGLE: ("google_api_key", "GOOGLE_API_KEY"),
            LLMProvider.MISTRAL: ("mistral_api_key", "MISTRAL_API_KEY"),
            LLMProvider.COHERE: ("cohere_api_key", "COHERE_API_KEY"),
            LLMProvider.TOGETHER: ("together_api_key", "TOGETHER_API_KEY"),
            LLMProvider.GLM: ("glm_api_key", "GLM_API_KEY"),
            LLMProvider.CUSTOM: ("custom_api_key", "CUSTOM_API_KEY"),
        }

        if provider not in api_key_map:
            raise ValueError(f"Unknown LLM provider: {provider}")

        attr_name, env_var = api_key_map[provider]
        api_key = getattr(self, attr_name)

        if not api_key:
            # Try fallback to ANTHROPIC_API_KEY for backward compatibility
            if provider != LLMProvider.ANTHROPIC and self.anthropic_api_key:
                return self.anthropic_api_key

            raise ValueError(
                f"API key not configured for provider '{provider.value}'. "
                f"Set environment variable '{env_var}' or '{attr_name}' in .env file."
            )

        return api_key

    def get_base_url(self) -> Optional[str]:
        """
        Get the base URL for the current provider (if custom).

        Returns:
            Base URL or None for default provider endpoints
        """
        provider = self.llm_provider

        base_url_map = {
            LLMProvider.OPENAI: "openai_base_url",
            LLMProvider.AZURE_OPENAI: "azure_openai_endpoint",
            LLMProvider.MISTRAL: "mistral_base_url",
            LLMProvider.COHERE: "cohere_base_url",
            LLMProvider.TOGETHER: "together_base_url",
            LLMProvider.GLM: "glm_base_url",
            LLMProvider.CUSTOM: "custom_base_url",
        }

        attr_name = base_url_map.get(provider)
        if attr_name:
            return getattr(self, attr_name, None)

        return None

    def get_model_for_agent(self, agent_name: str) -> str:
        """
        Get the model ID for a specific agent.

        Args:
            agent_name: Logical agent name (e.g., "orchestrator", "analyst")

        Returns:
            Model ID for the current provider

        Example:
            settings = get_settings()
            model = settings.get_model_for_agent("orchestrator")
            # Returns "claude-3-5-opus-20240507" for Anthropic provider
            # Returns "gpt-4o" for OpenAI provider
        """
        # Normalize agent name
        agent_key = agent_name.lower().replace(" ", "_").replace("-", "_")

        # Check for explicit per-agent override first (e.g., MAS_ORCHESTRATOR_MODEL)
        override_attr = f"{agent_key}_model"
        if hasattr(self, override_attr):
            override = getattr(self, override_attr)
            if override:
                return override

        # Check for default model override (MAS_DEFAULT_MODEL)
        # This takes precedence over provider mappings when explicitly set
        if self.default_model:
            return self.default_model

        # Get default model mapping for current provider
        mapping = DEFAULT_MODEL_MAPPINGS.get(self.llm_provider)
        if not mapping:
            # Fall back to Anthropic if provider not in registry
            mapping = DEFAULT_MODEL_MAPPINGS[LLMProvider.ANTHROPIC]

        # Try to get specific model for agent
        model = mapping.get_model(agent_key)
        if model:
            return model

        # Fall back to default model from provider mapping
        default_model = mapping.get_model("default")
        if default_model:
            return default_model

        # Ultimate fallback
        return "claude-3-5-sonnet-20241022"

    def get_provider_config(self) -> Dict[str, Any]:
        """
        Get complete configuration for the current provider.

        Returns:
            Dictionary with provider-specific configuration

        Example:
            settings = get_settings()
            config = settings.get_provider_config()
            # Returns {
            #     "provider": "anthropic",
            #     "api_key": "sk-...",
            #     "base_url": None,
            #     "model": "claude-3-5-opus-20240507",
            # }
        """
        provider = self.llm_provider

        config = {
            "provider": provider.value,
            "api_key": self.get_api_key(),
            "base_url": self.get_base_url(),
        }

        # Add provider-specific config
        if provider == LLMProvider.AZURE_OPENAI:
            config.update({
                "endpoint": self.azure_openai_endpoint,
                "api_version": self.azure_openai_api_version,
                "deployment": self.azure_openai_deployment,
            })
        elif provider == LLMProvider.GOOGLE:
            config.update({
                "project_id": self.google_project_id,
                "location": self.google_location,
            })
        elif provider == LLMProvider.OPENAI:
            config.update({
                "organization": self.openai_organization,
            })

        return config

    def validate_api_key(self) -> bool:
        """
        Validate that an API key is configured for the current provider.

        Returns:
            True if API key is configured, False otherwise
        """
        try:
            self.get_api_key()
            return True
        except ValueError:
            return False

    def get_all_models(self) -> Dict[str, str]:
        """
        Get all model mappings for the current provider.

        Returns:
            Dictionary mapping logical agent names to model IDs
        """
        mapping = DEFAULT_MODEL_MAPPINGS.get(self.llm_provider)
        if mapping:
            return mapping.models
        return {}

    def list_supported_providers(self) -> List[str]:
        """
        List all supported LLM providers.

        Returns:
            List of provider names
        """
        return [p.value for p in LLMProvider]

    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """
        Get information about a specific model.

        Args:
            model_id: Model identifier

        Returns:
            Dictionary with model information
        """
        return {
            "model_id": model_id,
            "provider": self.llm_provider.value,
            "api_key_configured": self.validate_api_key(),
            "base_url": self.get_base_url(),
        }


# =============================================================================
# Global Settings Instance
# =============================================================================

_global_settings: Optional[Settings] = None


def get_settings(env_file: Optional[str] = None) -> Settings:
    """
    Get the global settings instance.

    Args:
        env_file: Optional path to .env file (for testing)

    Returns:
        Settings instance

    Example:
        from src.config.settings import get_settings

        settings = get_settings()
        api_key = settings.get_api_key()
        model = settings.get_model_for_agent("orchestrator")
    """
    global _global_settings

    if _global_settings is None:
        if env_file:
            _global_settings = Settings(_env_file=env_file)
        else:
            _global_settings = Settings()

    return _global_settings


def reload_settings(env_file: Optional[str] = None) -> Settings:
    """
    Reload settings from environment.

    Args:
        env_file: Optional path to .env file

    Returns:
        New Settings instance
    """
    global _global_settings
    _global_settings = None
    return get_settings(env_file)


# =============================================================================
# Convenience Functions
# =============================================================================

def get_api_key() -> str:
    """Convenience function to get the API key for current provider."""
    return reload_settings().get_api_key()


def get_model_for_agent(agent_name: str) -> str:
    """Convenience function to get model ID for an agent."""
    return reload_settings().get_model_for_agent(agent_name)


def get_provider() -> LLMProvider:
    """Convenience function to get current LLM provider."""
    return reload_settings().llm_provider


def get_provider_config() -> Dict[str, Any]:
    """Convenience function to get provider configuration."""
    return reload_settings().get_provider_config()


# Export main classes
__all__ = [
    "LLMProvider",
    "ModelMapping",
    "Settings",
    "get_settings",
    "reload_settings",
    "get_api_key",
    "get_model_for_agent",
    "get_provider",
    "get_provider_config",
]
