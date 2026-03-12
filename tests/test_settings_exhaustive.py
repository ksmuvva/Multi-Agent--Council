"""
Exhaustive Tests for Configuration Settings Module

Tests LLM providers, model mappings, settings class,
API key handling, model resolution, and convenience functions.
"""

import sys
import os
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from src.config.settings import (
    LLMProvider,
    ModelMapping,
    DEFAULT_MODEL_MAPPINGS,
    Settings,
    get_settings,
    reload_settings,
    get_api_key,
    get_model_for_agent,
    get_provider,
    get_provider_config,
)

import src.config.settings as settings_module


@pytest.fixture(autouse=True)
def reset_global_settings():
    """Reset global settings singleton between tests."""
    settings_module._global_settings = None
    yield
    settings_module._global_settings = None


# =============================================================================
# LLMProvider Enum Tests
# =============================================================================

class TestLLMProvider:
    def test_all_providers(self):
        assert LLMProvider.ANTHROPIC == "anthropic"
        assert LLMProvider.OPENAI == "openai"
        assert LLMProvider.AZURE_OPENAI == "azure_openai"
        assert LLMProvider.GOOGLE == "google"
        assert LLMProvider.MISTRAL == "mistral"
        assert LLMProvider.COHERE == "cohere"
        assert LLMProvider.TOGETHER == "together"
        assert LLMProvider.GLM == "glm"
        assert LLMProvider.CUSTOM == "custom"

    def test_count(self):
        assert len(LLMProvider) == 9


# =============================================================================
# ModelMapping Tests
# =============================================================================

class TestModelMapping:
    def test_create(self):
        mm = ModelMapping(
            provider=LLMProvider.ANTHROPIC,
            models={"default": "claude-3-5-sonnet", "orchestrator": "claude-3-5-opus"},
        )
        assert mm.get_model("default") == "claude-3-5-sonnet"
        assert mm.get_model("orchestrator") == "claude-3-5-opus"

    def test_get_model_missing(self):
        mm = ModelMapping(provider=LLMProvider.ANTHROPIC, models={"default": "model"})
        assert mm.get_model("nonexistent") is None

    def test_get_model_with_default(self):
        mm = ModelMapping(provider=LLMProvider.ANTHROPIC, models={})
        assert mm.get_model("missing", "fallback") == "fallback"


# =============================================================================
# DEFAULT_MODEL_MAPPINGS Tests
# =============================================================================

class TestDefaultModelMappings:
    def test_all_providers_have_mappings(self):
        for provider in LLMProvider:
            assert provider in DEFAULT_MODEL_MAPPINGS

    def test_all_mappings_have_default(self):
        for provider, mapping in DEFAULT_MODEL_MAPPINGS.items():
            assert mapping.get_model("default") is not None

    def test_all_mappings_have_key_agents(self):
        key_agents = ["orchestrator", "executor", "verifier", "critic"]
        for provider, mapping in DEFAULT_MODEL_MAPPINGS.items():
            for agent in key_agents:
                assert mapping.get_model(agent) is not None, \
                    f"Provider {provider} missing agent {agent}"

    def test_anthropic_uses_claude_models(self):
        mapping = DEFAULT_MODEL_MAPPINGS[LLMProvider.ANTHROPIC]
        for model in mapping.models.values():
            assert "claude" in model.lower()

    def test_openai_uses_gpt_models(self):
        mapping = DEFAULT_MODEL_MAPPINGS[LLMProvider.OPENAI]
        for model in mapping.models.values():
            assert "gpt" in model.lower()


# =============================================================================
# Settings Tests
# =============================================================================

class TestSettings:
    def test_default_settings(self):
        settings = Settings()
        assert settings.llm_provider == LLMProvider.ANTHROPIC
        assert settings.max_budget == 5.0
        assert settings.max_turns_orchestrator == 200
        assert settings.max_turns_subagent == 30
        assert settings.max_sme_count == 3

    def test_feature_flags_defaults(self):
        settings = Settings()
        assert settings.enable_council is True
        assert settings.enable_sme is True
        assert settings.enable_code_reviewer is True
        assert settings.enable_memory_curator is True
        assert settings.enable_debate is True
        assert settings.enable_ensemble is True
        assert settings.enable_cost_tracking is True

    def test_development_defaults(self):
        settings = Settings()
        assert settings.debug is False
        assert settings.dev_mode is False

    # --------- API Key ---------

    def test_get_api_key_anthropic(self):
        settings = Settings(anthropic_api_key="test_key")
        assert settings.get_api_key() == "test_key"

    def test_get_api_key_openai(self):
        settings = Settings(llm_provider=LLMProvider.OPENAI, openai_api_key="openai_key")
        assert settings.get_api_key() == "openai_key"

    def test_get_api_key_missing_raises(self):
        settings = Settings(llm_provider=LLMProvider.GOOGLE, anthropic_api_key=None)
        with pytest.raises(ValueError):
            settings.get_api_key()

    def test_get_api_key_fallback_to_anthropic(self):
        settings = Settings(
            llm_provider=LLMProvider.OPENAI,
            openai_api_key=None,
            anthropic_api_key="fallback_key",
        )
        assert settings.get_api_key() == "fallback_key"

    # --------- Base URL ---------

    def test_get_base_url_anthropic(self):
        settings = Settings(llm_provider=LLMProvider.ANTHROPIC)
        assert settings.get_base_url() is None

    def test_get_base_url_custom(self):
        settings = Settings(
            llm_provider=LLMProvider.CUSTOM,
            custom_base_url="https://custom.api.com",
        )
        assert settings.get_base_url() == "https://custom.api.com"

    def test_get_base_url_openai(self):
        settings = Settings(
            llm_provider=LLMProvider.OPENAI,
            openai_base_url="https://openai.proxy.com",
        )
        assert settings.get_base_url() == "https://openai.proxy.com"

    # --------- Model for Agent ---------

    def test_get_model_for_agent_default(self):
        settings = Settings()
        model = settings.get_model_for_agent("executor")
        assert "claude" in model.lower()

    def test_get_model_for_agent_orchestrator(self):
        settings = Settings()
        model = settings.get_model_for_agent("orchestrator")
        assert "opus" in model.lower()

    def test_get_model_for_agent_override(self):
        settings = Settings(executor_model="custom-model")
        model = settings.get_model_for_agent("executor")
        assert model == "custom-model"

    def test_get_model_for_agent_unknown(self):
        settings = Settings()
        model = settings.get_model_for_agent("unknown_agent_xyz")
        # Should fall back to default
        assert model is not None

    def test_get_model_for_agent_normalized(self):
        settings = Settings()
        model = settings.get_model_for_agent("Code Reviewer")
        assert model is not None

    # --------- Provider Config ---------

    def test_get_provider_config_anthropic(self):
        settings = Settings(anthropic_api_key="key123")
        config = settings.get_provider_config()
        assert config["provider"] == "anthropic"
        assert config["api_key"] == "key123"

    def test_get_provider_config_azure(self):
        settings = Settings(
            llm_provider=LLMProvider.AZURE_OPENAI,
            azure_openai_api_key="azure_key",
            azure_openai_endpoint="https://azure.endpoint",
        )
        config = settings.get_provider_config()
        assert "endpoint" in config
        assert "api_version" in config

    def test_get_provider_config_google(self):
        settings = Settings(
            llm_provider=LLMProvider.GOOGLE,
            google_api_key="google_key",
        )
        config = settings.get_provider_config()
        assert "project_id" in config

    # --------- Validate API Key ---------

    def test_validate_api_key_present(self):
        settings = Settings(anthropic_api_key="key")
        assert settings.validate_api_key() is True

    def test_validate_api_key_missing(self):
        settings = Settings(
            llm_provider=LLMProvider.GOOGLE,
            anthropic_api_key=None,
        )
        assert settings.validate_api_key() is False

    # --------- Get All Models ---------

    def test_get_all_models(self):
        settings = Settings()
        models = settings.get_all_models()
        assert "default" in models
        assert "orchestrator" in models
        assert len(models) > 10

    # --------- List Providers ---------

    def test_list_supported_providers(self):
        settings = Settings()
        providers = settings.list_supported_providers()
        assert "anthropic" in providers
        assert "openai" in providers
        assert len(providers) == 9

    # --------- Model Info ---------

    def test_get_model_info(self):
        settings = Settings(anthropic_api_key="key")
        info = settings.get_model_info("claude-3-5-sonnet-20241022")
        assert info["model_id"] == "claude-3-5-sonnet-20241022"
        assert info["provider"] == "anthropic"
        assert info["api_key_configured"] is True


# =============================================================================
# Global Settings Functions Tests
# =============================================================================

class TestGlobalSettings:
    def test_get_settings_creates_instance(self):
        settings = get_settings()
        assert settings is not None

    def test_get_settings_singleton(self):
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_reload_settings(self):
        s1 = get_settings()
        s2 = reload_settings()
        assert s1 is not s2

    def test_convenience_get_provider(self):
        provider = get_provider()
        assert isinstance(provider, LLMProvider)

    def test_convenience_get_model_for_agent(self):
        model = get_model_for_agent("executor")
        assert model is not None
