"""
Exhaustive Tests for SME Registry Module

Tests all personas, query functions, interaction modes,
keyword matching, display formatting, and statistics.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from src.core.sme_registry import (
    InteractionMode,
    SMEPersona,
    SME_REGISTRY,
    get_persona,
    get_all_personas,
    find_personas_by_keywords,
    find_personas_by_domain,
    get_persona_ids,
    validate_interaction_mode,
    get_persona_for_display,
    get_registry_stats,
)


# =============================================================================
# InteractionMode Enum Tests
# =============================================================================

class TestInteractionMode:
    """Tests for InteractionMode enum."""

    def test_values(self):
        assert InteractionMode.ADVISOR == "advisor"
        assert InteractionMode.CO_EXECUTOR == "co_executor"
        assert InteractionMode.DEBATER == "debater"

    def test_count(self):
        assert len(InteractionMode) == 3

    def test_is_string(self):
        for mode in InteractionMode:
            assert isinstance(mode.value, str)


# =============================================================================
# SMEPersona Dataclass Tests
# =============================================================================

class TestSMEPersona:
    """Tests for SMEPersona dataclass."""

    def test_create_persona(self):
        persona = SMEPersona(
            name="Test SME",
            persona_id="test_sme",
            domain="Testing",
            trigger_keywords=["test"],
            skill_files=["test-skill"],
            system_prompt_template="config/sme/test.md",
            interaction_modes=[InteractionMode.ADVISOR],
        )
        assert persona.name == "Test SME"
        assert persona.persona_id == "test_sme"
        assert persona.default_model == "sonnet"
        assert persona.description == ""

    def test_defaults(self):
        persona = SMEPersona(
            name="Test", persona_id="test", domain="Test",
            trigger_keywords=[], skill_files=[],
            system_prompt_template="test.md",
            interaction_modes=[InteractionMode.ADVISOR],
        )
        assert persona.default_model == "sonnet"
        assert persona.description == ""


# =============================================================================
# Registry Content Tests
# =============================================================================

class TestRegistryContent:
    """Tests for the SME_REGISTRY contents."""

    def test_registry_has_10_personas(self):
        assert len(SME_REGISTRY) == 10

    EXPECTED_PERSONAS = [
        "iam_architect", "cloud_architect", "security_analyst",
        "data_engineer", "ai_ml_engineer", "test_engineer",
        "business_analyst", "technical_writer", "devops_engineer",
        "frontend_developer",
    ]

    @pytest.mark.parametrize("persona_id", EXPECTED_PERSONAS)
    def test_persona_exists(self, persona_id):
        assert persona_id in SME_REGISTRY

    @pytest.mark.parametrize("persona_id", EXPECTED_PERSONAS)
    def test_persona_has_required_fields(self, persona_id):
        persona = SME_REGISTRY[persona_id]
        assert persona.name
        assert persona.persona_id == persona_id
        assert persona.domain
        assert len(persona.trigger_keywords) > 0
        assert len(persona.skill_files) > 0
        assert persona.system_prompt_template
        assert len(persona.interaction_modes) > 0

    @pytest.mark.parametrize("persona_id", EXPECTED_PERSONAS)
    def test_persona_has_valid_modes(self, persona_id):
        persona = SME_REGISTRY[persona_id]
        for mode in persona.interaction_modes:
            assert isinstance(mode, InteractionMode)

    def test_all_personas_have_advisor_mode(self):
        for persona in SME_REGISTRY.values():
            assert InteractionMode.ADVISOR in persona.interaction_modes

    def test_debater_mode_availability(self):
        debater_personas = [
            pid for pid, p in SME_REGISTRY.items()
            if InteractionMode.DEBATER in p.interaction_modes
        ]
        assert "cloud_architect" in debater_personas
        assert "security_analyst" in debater_personas
        assert "ai_ml_engineer" in debater_personas

    def test_iam_architect_keywords(self):
        persona = SME_REGISTRY["iam_architect"]
        assert "sailpoint" in persona.trigger_keywords
        assert "cyberark" in persona.trigger_keywords
        assert "identity" in persona.trigger_keywords

    def test_cloud_architect_keywords(self):
        persona = SME_REGISTRY["cloud_architect"]
        assert "azure" in persona.trigger_keywords
        assert "aws" in persona.trigger_keywords
        assert "kubernetes" in persona.trigger_keywords

    def test_security_analyst_keywords(self):
        persona = SME_REGISTRY["security_analyst"]
        assert "owasp" in persona.trigger_keywords
        assert "vulnerability" in persona.trigger_keywords


# =============================================================================
# get_persona Tests
# =============================================================================

class TestGetPersona:
    """Tests for get_persona function."""

    def test_existing_persona(self):
        persona = get_persona("cloud_architect")
        assert persona is not None
        assert persona.persona_id == "cloud_architect"

    def test_nonexistent_persona(self):
        assert get_persona("nonexistent") is None

    def test_empty_id(self):
        assert get_persona("") is None

    def test_all_personas_retrievable(self):
        for pid in SME_REGISTRY:
            persona = get_persona(pid)
            assert persona is not None
            assert persona.persona_id == pid


# =============================================================================
# get_all_personas Tests
# =============================================================================

class TestGetAllPersonas:
    """Tests for get_all_personas function."""

    def test_returns_dict(self):
        result = get_all_personas()
        assert isinstance(result, dict)

    def test_returns_copy(self):
        result = get_all_personas()
        result["new_persona"] = None
        assert "new_persona" not in SME_REGISTRY

    def test_count(self):
        assert len(get_all_personas()) == 10


# =============================================================================
# find_personas_by_keywords Tests
# =============================================================================

class TestFindPersonasByKeywords:
    """Tests for find_personas_by_keywords function."""

    def test_single_keyword(self):
        result = find_personas_by_keywords(["azure"])
        assert len(result) > 0
        persona_ids = [p.persona_id for p in result]
        assert "cloud_architect" in persona_ids

    def test_multiple_keywords(self):
        result = find_personas_by_keywords(["security", "owasp"])
        assert len(result) > 0
        assert result[0].persona_id == "security_analyst"

    def test_no_matching_keywords(self):
        result = find_personas_by_keywords(["zzzznonexistent"])
        assert len(result) == 0

    def test_empty_keywords(self):
        result = find_personas_by_keywords([])
        assert len(result) == 0

    def test_sorted_by_match_count(self):
        result = find_personas_by_keywords(["docker", "kubernetes", "container"])
        if len(result) > 1:
            # cloud_architect or devops_engineer should be first (more matches)
            top_ids = [p.persona_id for p in result[:2]]
            assert "cloud_architect" in top_ids or "devops_engineer" in top_ids

    def test_case_insensitive(self):
        result = find_personas_by_keywords(["AZURE"])
        persona_ids = [p.persona_id for p in result]
        assert "cloud_architect" in persona_ids

    def test_partial_keyword_match(self):
        result = find_personas_by_keywords(["sail"])
        persona_ids = [p.persona_id for p in result]
        assert "iam_architect" in persona_ids

    def test_rag_keyword(self):
        result = find_personas_by_keywords(["rag"])
        persona_ids = [p.persona_id for p in result]
        assert "ai_ml_engineer" in persona_ids

    def test_multiple_personas_match_security(self):
        result = find_personas_by_keywords(["security"])
        assert len(result) >= 1


# =============================================================================
# find_personas_by_domain Tests
# =============================================================================

class TestFindPersonasByDomain:
    """Tests for find_personas_by_domain function."""

    def test_cloud_domain(self):
        result = find_personas_by_domain(["Cloud"])
        persona_ids = [p.persona_id for p in result]
        assert "cloud_architect" in persona_ids

    def test_security_domain(self):
        result = find_personas_by_domain(["Security"])
        persona_ids = [p.persona_id for p in result]
        assert "security_analyst" in persona_ids

    def test_no_matching_domain(self):
        result = find_personas_by_domain(["Astrobiology"])
        assert len(result) == 0

    def test_case_insensitive(self):
        result = find_personas_by_domain(["cloud"])
        assert len(result) > 0

    def test_partial_domain_match(self):
        result = find_personas_by_domain(["Data"])
        persona_ids = [p.persona_id for p in result]
        assert "data_engineer" in persona_ids

    def test_empty_keywords(self):
        result = find_personas_by_domain([])
        assert len(result) == 0


# =============================================================================
# get_persona_ids Tests
# =============================================================================

class TestGetPersonaIds:
    """Tests for get_persona_ids function."""

    def test_returns_list(self):
        result = get_persona_ids()
        assert isinstance(result, list)

    def test_count(self):
        assert len(get_persona_ids()) == 10

    def test_contains_expected_ids(self):
        ids = get_persona_ids()
        assert "cloud_architect" in ids
        assert "security_analyst" in ids


# =============================================================================
# validate_interaction_mode Tests
# =============================================================================

class TestValidateInteractionMode:
    """Tests for validate_interaction_mode function."""

    def test_valid_advisor_mode(self):
        assert validate_interaction_mode("cloud_architect", InteractionMode.ADVISOR) is True

    def test_valid_debater_mode(self):
        assert validate_interaction_mode("cloud_architect", InteractionMode.DEBATER) is True

    def test_invalid_debater_for_iam(self):
        assert validate_interaction_mode("iam_architect", InteractionMode.DEBATER) is False

    def test_nonexistent_persona(self):
        assert validate_interaction_mode("nonexistent", InteractionMode.ADVISOR) is False

    def test_all_personas_support_advisor(self):
        for pid in SME_REGISTRY:
            assert validate_interaction_mode(pid, InteractionMode.ADVISOR) is True

    def test_co_executor_mode(self):
        assert validate_interaction_mode("cloud_architect", InteractionMode.CO_EXECUTOR) is True


# =============================================================================
# get_persona_for_display Tests
# =============================================================================

class TestGetPersonaForDisplay:
    """Tests for get_persona_for_display function."""

    def test_existing_persona(self):
        result = get_persona_for_display("cloud_architect")
        assert result is not None
        assert result["id"] == "cloud_architect"
        assert result["name"] == "Cloud Architect"
        assert "domain" in result
        assert "trigger_keywords" in result
        assert "skill_files" in result
        assert "interaction_modes" in result
        assert "default_model" in result

    def test_nonexistent_persona(self):
        assert get_persona_for_display("nonexistent") is None

    def test_interaction_modes_are_strings(self):
        result = get_persona_for_display("cloud_architect")
        for mode in result["interaction_modes"]:
            assert isinstance(mode, str)

    def test_all_personas_display(self):
        for pid in SME_REGISTRY:
            result = get_persona_for_display(pid)
            assert result is not None
            assert result["id"] == pid


# =============================================================================
# get_registry_stats Tests
# =============================================================================

class TestGetRegistryStats:
    """Tests for get_registry_stats function."""

    def test_returns_dict(self):
        stats = get_registry_stats()
        assert isinstance(stats, dict)

    def test_total_personas(self):
        stats = get_registry_stats()
        assert stats["total_personas"] == 10

    def test_persona_ids_list(self):
        stats = get_registry_stats()
        assert len(stats["persona_ids"]) == 10

    def test_domains_unique(self):
        stats = get_registry_stats()
        assert len(stats["domains"]) > 0

    def test_total_trigger_keywords_positive(self):
        stats = get_registry_stats()
        assert stats["total_trigger_keywords"] > 0

    def test_available_skills_not_empty(self):
        stats = get_registry_stats()
        assert len(stats["available_skills"]) > 0
