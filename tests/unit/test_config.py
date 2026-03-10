"""
Unit Tests for Configuration Management

Tests for centralized settings, multi-provider support,
model mapping, and API key validation.
"""

import pytest
import os
from pathlib import Path
from unittest.mock import patch, Mock

from src.config.settings import (
    LLMProvider,
    ModelMapping,
    Settings,
    get_settings,
    reload_settings,
    get_api_key,
    get_model_for_agent,
    get_provider,
    DEFAULT_MODEL_MAPPINGS,
    _global_settings,
)
import src.config.settings as _settings_module


@pytest.fixture(autouse=True)
def reset_settings_singleton():
    """Reset the global settings singleton before each test."""
    _settings_module._global_settings = None
    yield
    _settings_module._global_settings = None


# =============================================================================
# LLM Provider Enum Tests
# =============================================================================

class TestLLMProvider:
    """Tests for LLM provider enumeration."""

    def test_provider_values(self):
        """Test that all expected providers are defined."""
        expected_providers = [
            "anthropic",
            "openai",
            "azure_openai",
            "google",
            "mistral",
            "cohere",
            "together",
            "custom",
        ]

        for provider in expected_providers:
            assert LLMProvider(provider) is not None

    def test_provider_comparison(self):
        """Test provider enum comparison."""
        assert LLMProvider.ANTHROPIC == "anthropic"
        assert LLMProvider.OPENAI == "openai"
        assert LLMProvider.ANTHROPIC != LLMProvider.OPENAI


# =============================================================================
# Model Mapping Tests
# =============================================================================

class TestModelMapping:
    """Tests for ModelMapping dataclass."""

    def test_anthropic_model_mapping(self):
        """Test Anthropic model mapping."""
        mapping = DEFAULT_MODEL_MAPPINGS[LLMProvider.ANTHROPIC]

        assert mapping.provider == LLMProvider.ANTHROPIC
        assert "orchestrator" in mapping.models
        assert "council" in mapping.models
        assert "analyst" in mapping.models

    def test_get_model_existing(self):
        """Test getting existing model from mapping."""
        mapping = DEFAULT_MODEL_MAPPINGS[LLMProvider.ANTHROPIC]

        model = mapping.get_model("orchestrator")
        assert model == "claude-3-5-opus-20240507"

    def test_get_model_missing_with_default(self):
        """Test getting missing model with default fallback."""
        mapping = DEFAULT_MODEL_MAPPINGS[LLMProvider.ANTHROPIC]

        model = mapping.get_model("nonexistent", "default-model")
        assert model == "default-model"

    def test_get_model_missing_no_default(self):
        """Test getting missing model without default."""
        mapping = DEFAULT_MODEL_MAPPINGS[LLMProvider.ANTHROPIC]

        model = mapping.get_model("nonexistent")
        assert model is None

    def test_openai_model_mapping(self):
        """Test OpenAI model mapping."""
        mapping = DEFAULT_MODEL_MAPPINGS[LLMProvider.OPENAI]

        assert mapping.provider == LLMProvider.OPENAI
        assert mapping.get_model("orchestrator") == "gpt-4o"
        assert mapping.get_model("clarifier") == "gpt-4o-mini"

    def test_all_providers_have_mappings(self):
        """Test that all providers have model mappings."""
        for provider in LLMProvider:
            assert provider in DEFAULT_MODEL_MAPPINGS
            mapping = DEFAULT_MODEL_MAPPINGS[provider]
            assert "default" in mapping.models
            assert "orchestrator" in mapping.models


# =============================================================================
# Settings Tests
# =============================================================================

class TestSettingsDefaults:
    """Tests for Settings default values."""

    @patch.dict(os.environ, {}, clear=True)
    def test_default_provider(self):
        """Test default LLM provider."""
        settings = Settings(_env_file=None)
        assert settings.llm_provider == LLMProvider.ANTHROPIC

    @patch.dict(os.environ, {}, clear=True)
    def test_default_budget(self):
        """Test default budget settings."""
        settings = Settings(_env_file=None)
        assert settings.max_budget == 5.0
        assert settings.budget_warning_threshold == 0.8

    @patch.dict(os.environ, {}, clear=True)
    def test_default_max_turns(self):
        """Test default max turns settings."""
        settings = Settings(_env_file=None)
        assert settings.max_turns_orchestrator == 200
        assert settings.max_turns_subagent == 30
        assert settings.max_turns_executor == 50

    @patch.dict(os.environ, {}, clear=True)
    def test_default_feature_flags(self):
        """Test default feature flags."""
        settings = Settings(_env_file=None)
        assert settings.enable_council is True
        assert settings.enable_sme is True
        assert settings.enable_code_reviewer is True
        assert settings.enable_debate is True


class TestSettingsFromEnvironment:
    """Tests for Settings loaded from environment variables."""

    @patch.dict(os.environ, {
        "MAS_LLM_PROVIDER": "openai",
        "OPENAI_API_KEY": "sk-test-key",
        "MAS_MAX_BUDGET": "10.0",
        "MAS_MAX_TURNS_ORCHESTRATOR": "300",
    }, clear=True)
    def test_load_from_environment(self):
        """Test loading settings from environment."""
        settings = Settings(_env_file=None)

        assert settings.llm_provider == LLMProvider.OPENAI
        assert settings.openai_api_key == "sk-test-key"
        assert settings.max_budget == 10.0
        assert settings.max_turns_orchestrator == 300

    @patch.dict(os.environ, {
        "MAS_LLM_PROVIDER": "anthropic",
        "ANTHROPIC_API_KEY": "sk-ant-test-key",
    }, clear=True)
    def test_anthropic_provider_config(self):
        """Test Anthropic provider configuration."""
        settings = Settings(_env_file=None)

        assert settings.llm_provider == LLMProvider.ANTHROPIC
        assert settings.anthropic_api_key == "sk-ant-test-key"

    @patch.dict(os.environ, {
        "MAS_LLM_PROVIDER": "azure_openai",
        "AZURE_OPENAI_API_KEY": "azure-test-key",
        "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
    }, clear=True)
    def test_azure_provider_config(self):
        """Test Azure OpenAI provider configuration."""
        settings = Settings(_env_file=None)

        assert settings.llm_provider == LLMProvider.AZURE_OPENAI
        assert settings.azure_openai_api_key == "azure-test-key"
        assert settings.azure_openai_endpoint == "https://test.openai.azure.com"

    @patch.dict(os.environ, {
        "MAS_LLM_PROVIDER": "custom",
        "CUSTOM_API_KEY": "custom-key",
        "CUSTOM_BASE_URL": "https://custom-endpoint.com/v1",
    }, clear=True)
    def test_custom_provider_config(self):
        """Test custom provider configuration."""
        settings = Settings(_env_file=None)

        assert settings.llm_provider == LLMProvider.CUSTOM
        assert settings.custom_api_key == "custom-key"
        assert settings.custom_base_url == "https://custom-endpoint.com/v1"


# =============================================================================
# Settings Methods Tests
# =============================================================================

class TestSettingsMethods:
    """Tests for Settings instance methods."""

    @patch.dict(os.environ, {
        "MAS_LLM_PROVIDER": "anthropic",
        "ANTHROPIC_API_KEY": "sk-ant-test",
    }, clear=True)
    def test_get_api_key_anthropic(self):
        """Test getting API key for Anthropic provider."""
        settings = Settings(_env_file=None)
        api_key = settings.get_api_key()

        assert api_key == "sk-ant-test"

    @patch.dict(os.environ, {
        "MAS_LLM_PROVIDER": "openai",
        "OPENAI_API_KEY": "sk-openai-test",
    }, clear=True)
    def test_get_api_key_openai(self):
        """Test getting API key for OpenAI provider."""
        settings = Settings(_env_file=None)
        api_key = settings.get_api_key()

        assert api_key == "sk-openai-test"

    @patch.dict(os.environ, {
        "MAS_LLM_PROVIDER": "anthropic",
    }, clear=True)
    def test_get_api_key_missing(self):
        """Test getting API key when not configured."""
        settings = Settings(_env_file=None)

        with pytest.raises(ValueError, match="API key not configured"):
            settings.get_api_key()

    @patch.dict(os.environ, {
        "MAS_LLM_PROVIDER": "anthropic",
        "ANTHROPIC_API_KEY": "sk-ant-test",
    }, clear=True)
    def test_get_base_url_anthropic(self):
        """Test getting base URL for Anthropic (should be None)."""
        settings = Settings(_env_file=None)
        base_url = settings.get_base_url()

        assert base_url is None

    @patch.dict(os.environ, {
        "MAS_LLM_PROVIDER": "custom",
        "CUSTOM_API_KEY": "test",
        "CUSTOM_BASE_URL": "https://custom.com/v1",
    }, clear=True)
    def test_get_base_url_custom(self):
        """Test getting base URL for custom provider."""
        settings = Settings(_env_file=None)
        base_url = settings.get_base_url()

        assert base_url == "https://custom.com/v1"

    @patch.dict(os.environ, {
        "MAS_LLM_PROVIDER": "anthropic",
        "ANTHROPIC_API_KEY": "sk-test",
    }, clear=True)
    def test_get_model_for_agent(self):
        """Test getting model ID for agent."""
        settings = Settings(_env_file=None)

        # Test various agents
        assert "claude" in settings.get_model_for_agent("orchestrator").lower()
        assert "claude" in settings.get_model_for_agent("analyst").lower()
        assert "claude" in settings.get_model_for_agent("council").lower()

    @patch.dict(os.environ, {
        "MAS_LLM_PROVIDER": "openai",
        "OPENAI_API_KEY": "sk-test",
    }, clear=True)
    def test_get_model_for_agent_openai(self):
        """Test getting model ID for agent with OpenAI provider."""
        settings = Settings(_env_file=None)

        model = settings.get_model_for_agent("orchestrator")
        assert "gpt" in model.lower()

    @patch.dict(os.environ, {
        "MAS_LLM_PROVIDER": "openai",
        "OPENAI_API_KEY": "sk-test",
        "MAS_ORCHESTRATOR_MODEL": "custom-model-override",
    }, clear=True)
    def test_get_model_with_override(self):
        """Test that model overrides work."""
        settings = Settings(_env_file=None)

        model = settings.get_model_for_agent("orchestrator")
        assert model == "custom-model-override"

    @patch.dict(os.environ, {
        "MAS_LLM_PROVIDER": "anthropic",
        "ANTHROPIC_API_KEY": "sk-test",
    }, clear=True)
    def test_get_provider_config(self):
        """Test getting complete provider configuration."""
        settings = Settings(_env_file=None)
        config = settings.get_provider_config()

        assert config["provider"] == "anthropic"
        assert "api_key" in config
        assert "base_url" in config

    @patch.dict(os.environ, {
        "MAS_LLM_PROVIDER": "anthropic",
        "ANTHROPIC_API_KEY": "sk-test",
    }, clear=True)
    def test_validate_api_key_valid(self):
        """Test API key validation with valid key."""
        settings = Settings(_env_file=None)
        assert settings.validate_api_key() is True

    @patch.dict(os.environ, {
        "MAS_LLM_PROVIDER": "anthropic",
    }, clear=True)
    def test_validate_api_key_invalid(self):
        """Test API key validation with missing key."""
        settings = Settings(_env_file=None)
        assert settings.validate_api_key() is False

    @patch.dict(os.environ, {
        "MAS_LLM_PROVIDER": "anthropic",
        "ANTHROPIC_API_KEY": "sk-test",
    }, clear=True)
    def test_get_all_models(self):
        """Test getting all model mappings."""
        settings = Settings(_env_file=None)
        models = settings.get_all_models()

        assert isinstance(models, dict)
        assert "orchestrator" in models
        assert "analyst" in models
        assert "council" in models

    @patch.dict(os.environ, {}, clear=True)
    def test_list_supported_providers(self):
        """Test listing all supported providers."""
        settings = Settings(_env_file=None)
        providers = settings.list_supported_providers()

        assert "anthropic" in providers
        assert "openai" in providers
        assert "google" in providers


# =============================================================================
# Global Settings Tests
# =============================================================================

class TestGlobalSettings:
    """Tests for global settings instance."""

    def test_get_settings_returns_instance(self):
        """Test that get_settings returns Settings instance."""
        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_get_settings_singleton(self):
        """Test that get_settings returns same instance."""
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2

    @patch.dict(os.environ, {"MAS_LLM_PROVIDER": "openai"}, clear=True)
    def test_reload_settings(self):
        """Test reloading settings."""
        # Get initial settings
        settings1 = get_settings()

        # Change environment and reload
        with patch.dict(os.environ, {"MAS_LLM_PROVIDER": "anthropic"}):
            settings2 = reload_settings()

        # Should be different instance
        assert settings2 is not settings1


# =============================================================================
# Convenience Functions Tests
# =============================================================================

class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    @patch.dict(os.environ, {
        "MAS_LLM_PROVIDER": "anthropic",
        "ANTHROPIC_API_KEY": "sk-test",
    }, clear=True)
    def test_get_api_key_function(self):
        """Test get_api_key convenience function."""
        from src.config import get_api_key
        api_key = get_api_key()
        assert api_key == "sk-test"

    def test_get_model_for_agent_function(self):
        """Test get_model_for_agent convenience function returns a non-empty model."""
        from src.config.settings import reload_settings
        reload_settings()
        from src.config import get_model_for_agent
        model = get_model_for_agent("orchestrator")
        assert isinstance(model, str) and len(model) > 0

    @patch.dict(os.environ, {
        "MAS_LLM_PROVIDER": "anthropic",
    }, clear=True)
    def test_get_provider_function(self):
        """Test get_provider convenience function."""
        from src.config import get_provider
        provider = get_provider()
        assert provider == LLMProvider.ANTHROPIC

    @patch.dict(os.environ, {
        "MAS_LLM_PROVIDER": "anthropic",
        "ANTHROPIC_API_KEY": "sk-test",
    }, clear=True)
    def test_get_provider_config_function(self):
        """Test get_provider_config convenience function."""
        from src.config import get_provider_config
        config = get_provider_config()
        assert "provider" in config
        assert "api_key" in config


# =============================================================================
# Model Override Tests
# =============================================================================

class TestModelOverrides:
    """Tests for per-agent model overrides."""

    @patch.dict(os.environ, {
        "MAS_LLM_PROVIDER": "anthropic",
        "ANTHROPIC_API_KEY": "sk-test",
        "MAS_ORCHESTRATOR_MODEL": "claude-3-5-haiku-20241022",
        "MAS_ANALYST_MODEL": "claude-3-5-haiku-20241022",
    }, clear=True)
    def test_per_agent_overrides(self):
        """Test that per-agent model overrides work."""
        settings = Settings(_env_file=None)

        orchestrator_model = settings.get_model_for_agent("orchestrator")
        analyst_model = settings.get_model_for_agent("analyst")
        planner_model = settings.get_model_for_agent("planner")

        assert orchestrator_model == "claude-3-5-haiku-20241022"
        assert analyst_model == "claude-3-5-haiku-20241022"
        # Planner should use default (no override set)
        assert "haiku" not in planner_model

    @patch.dict(os.environ, {
        "MAS_LLM_PROVIDER": "openai",
        "OPENAI_API_KEY": "sk-test",
        "MAS_DEFAULT_MODEL": "gpt-4o-mini",
    }, clear=True)
    def test_default_model_override(self):
        """Test that default model override affects agents without specific overrides."""
        settings = Settings(_env_file=None)

        # Agent without specific override should use default
        formatter_model = settings.get_model_for_agent("formatter")
        assert formatter_model == "gpt-4o-mini"


# =============================================================================
# Provider Switching Tests
# =============================================================================

class TestProviderSwitching:
    """Tests for switching between LLM providers."""

    @patch.dict(os.environ, {
        "MAS_LLM_PROVIDER": "anthropic",
        "ANTHROPIC_API_KEY": "sk-ant-test",
    }, clear=True)
    def test_switch_to_anthropic(self):
        """Test switching to Anthropic provider."""
        settings = Settings(_env_file=None)

        assert settings.llm_provider == LLMProvider.ANTHROPIC
        model = settings.get_model_for_agent("orchestrator")
        assert "claude" in model.lower()

    @patch.dict(os.environ, {
        "MAS_LLM_PROVIDER": "openai",
        "OPENAI_API_KEY": "sk-openai-test",
    }, clear=True)
    def test_switch_to_openai(self):
        """Test switching to OpenAI provider."""
        settings = Settings(_env_file=None)

        assert settings.llm_provider == LLMProvider.OPENAI
        model = settings.get_model_for_agent("orchestrator")
        assert "gpt" in model.lower()

    @patch.dict(os.environ, {
        "MAS_LLM_PROVIDER": "google",
        "GOOGLE_API_KEY": "google-test-key",
    }, clear=True)
    def test_switch_to_google(self):
        """Test switching to Google provider."""
        settings = Settings(_env_file=None)

        assert settings.llm_provider == LLMProvider.GOOGLE
        model = settings.get_model_for_agent("orchestrator")
        assert "gemini" in model.lower()

    @patch.dict(os.environ, {
        "MAS_LLM_PROVIDER": "mistral",
        "MISTRAL_API_KEY": "mistral-test-key",
    }, clear=True)
    def test_switch_to_mistral(self):
        """Test switching to Mistral provider."""
        settings = Settings(_env_file=None)

        assert settings.llm_provider == LLMProvider.MISTRAL
        model = settings.get_model_for_agent("orchestrator")
        assert "mistral" in model.lower()
