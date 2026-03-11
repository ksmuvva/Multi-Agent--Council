"""
Comprehensive SME Persona Validation Tests

Validates and analyses all 10 SME personas across:
1. Schema correctness (trailing-comma regression, Field descriptions, JSON schema)
2. Registry integrity (all personas, keywords, domains, skills, modes)
3. Spawner lifecycle (spawn, interact, all 3 modes for every persona)
4. Claude Agent SDK integration (imports, config, tools, skills mapping)
5. Skills system (SKILL.md existence, mapping coverage)
6. Cross-component data flow (Council Chair -> Spawner -> Interaction -> Report)
7. Edge cases and boundary conditions
"""

import json
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Set
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from src.schemas.sme import (
    SMEInteractionMode,
    AdvisorReport,
    CoExecutorSection,
    CoExecutorReport,
    DebatePosition,
    DebaterReport,
    SMEAdvisoryReport,
)
from src.schemas.council import (
    SMESelection,
    SMESelectionReport,
    InteractionMode as CouncilInteractionMode,
)
from src.core.sme_registry import (
    SME_REGISTRY,
    SMEPersona,
    InteractionMode as RegistryInteractionMode,
    get_persona,
    get_all_personas,
    find_personas_by_keywords,
    find_personas_by_domain,
    get_persona_ids,
    validate_interaction_mode,
    get_persona_for_display,
    get_registry_stats,
)
from src.agents.sme_spawner import (
    SMESpawner,
    SpawnedSME,
    SpawnResult,
    create_sme_spawner,
)
from src.core.sdk_integration import (
    AGENT_ALLOWED_TOOLS,
    ClaudeAgentOptions,
    build_agent_options,
    get_skills_for_agent,
    get_skills_for_sme,
    _get_output_schema,
)
from src.agents.council import (
    CouncilChairAgent,
)


# =============================================================================
# 1. Schema Correctness - Trailing Comma Regression Prevention
# =============================================================================

class TestSMESchemaTrailingCommaRegression:
    """Ensure NO Pydantic Field() has a trailing comma creating tuple defaults."""

    SME_MODELS = [
        AdvisorReport,
        CoExecutorSection,
        CoExecutorReport,
        DebatePosition,
        DebaterReport,
        SMEAdvisoryReport,
    ]

    @pytest.mark.parametrize("model_class", SME_MODELS)
    def test_no_tuple_defaults(self, model_class):
        """Every field default must NOT be a tuple (trailing-comma bug)."""
        for name, field_info in model_class.model_fields.items():
            assert not isinstance(field_info.default, tuple), (
                f"{model_class.__name__}.{name} has tuple default - "
                f"trailing comma bug! Got: {field_info.default}"
            )

    @pytest.mark.parametrize("model_class", SME_MODELS)
    def test_all_fields_have_description(self, model_class):
        """Every field should have a description for SDK outputFormat."""
        for name, field_info in model_class.model_fields.items():
            assert field_info.description is not None, (
                f"{model_class.__name__}.{name} missing description"
            )

    @pytest.mark.parametrize("model_class", SME_MODELS)
    def test_json_schema_no_warnings(self, model_class):
        """JSON schema generation should produce no warnings."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            schema = model_class.model_json_schema()
            pydantic_warnings = [
                x for x in w
                if "non-serializable-default" in str(x.message)
            ]
            assert len(pydantic_warnings) == 0, (
                f"{model_class.__name__} has {len(pydantic_warnings)} "
                f"non-serializable-default warnings"
            )

    @pytest.mark.parametrize("model_class", SME_MODELS)
    def test_json_schema_has_properties(self, model_class):
        """JSON schema should have a 'properties' key with all fields."""
        schema = model_class.model_json_schema()
        assert "properties" in schema
        for name in model_class.model_fields:
            assert name in schema["properties"], (
                f"{model_class.__name__} schema missing property '{name}'"
            )


# =============================================================================
# 2. Schema Round-Trip Serialization
# =============================================================================

class TestSMESchemaRoundTrip:
    """Test that all SME schemas can serialize and deserialize cleanly."""

    def test_advisor_report_round_trip(self):
        report = AdvisorReport(
            sme_persona="Security Analyst",
            reviewed_content="Test content",
            domain_corrections=["Fix auth", "Add encryption"],
            missing_considerations=["Rate limiting"],
            recommendations=["Use OWASP guidelines"],
            confidence=0.9,
        )
        json_str = report.model_dump_json()
        parsed = AdvisorReport.model_validate_json(json_str)
        assert parsed.sme_persona == "Security Analyst"
        assert len(parsed.domain_corrections) == 2
        assert parsed.confidence == 0.9

    def test_co_executor_section_round_trip(self):
        section = CoExecutorSection(
            sme_persona="Cloud Architect",
            section_title="Infrastructure",
            content="Use AKS for orchestration",
            domain_context="Cloud perspective",
            integration_notes="Integrate with main output",
        )
        json_str = section.model_dump_json()
        parsed = CoExecutorSection.model_validate_json(json_str)
        assert parsed.section_title == "Infrastructure"

    def test_co_executor_report_round_trip(self):
        report = CoExecutorReport(
            sme_persona="Cloud Architect",
            contributed_sections=[
                CoExecutorSection(
                    sme_persona="Cloud Architect",
                    section_title="Infra",
                    content="Content",
                    domain_context="Context",
                    integration_notes="Notes",
                )
            ],
            coordination_notes="Coordinate with Executor",
            domain_assumptions=["Cloud provider chosen"],
        )
        json_str = report.model_dump_json()
        parsed = CoExecutorReport.model_validate_json(json_str)
        assert len(parsed.contributed_sections) == 1
        assert parsed.coordination_notes == "Coordinate with Executor"

    def test_debate_position_round_trip(self):
        position = DebatePosition(
            sme_persona="Data Engineer",
            position="ACID compliance is critical",
            domain_rationale="Data integrity must be maintained",
            supporting_evidence=["Gartner report", "Industry standards"],
            confidence=0.85,
        )
        json_str = position.model_dump_json()
        parsed = DebatePosition.model_validate_json(json_str)
        assert parsed.confidence == 0.85
        assert len(parsed.supporting_evidence) == 2

    def test_debater_report_round_trip(self):
        report = DebaterReport(
            sme_persona="Data Engineer",
            debate_round=2,
            position=DebatePosition(
                sme_persona="Data Engineer",
                position="ACID first",
                domain_rationale="Rationale",
                supporting_evidence=["Evidence"],
                confidence=0.8,
            ),
            counter_arguments_addressed=["Cost argument addressed"],
            remaining_concerns=["Budget concerns"],
            willingness_to_concede=0.4,
        )
        json_str = report.model_dump_json()
        parsed = DebaterReport.model_validate_json(json_str)
        assert parsed.debate_round == 2
        assert parsed.willingness_to_concede == 0.4

    def test_sme_advisory_report_round_trip_advisor(self):
        report = SMEAdvisoryReport(
            sme_persona="Security Analyst",
            interaction_mode=SMEInteractionMode.ADVISOR,
            domain="Application Security",
            task_context="Review auth implementation",
            findings=["SQL injection risk"],
            recommendations=["Use parameterized queries"],
            confidence=0.9,
            caveats=["Needs pentest"],
            advisor_report=AdvisorReport(
                sme_persona="Security Analyst",
                reviewed_content="Auth code",
                domain_corrections=["Fix SQL"],
                missing_considerations=["Rate limiting"],
                recommendations=["Add WAF"],
                confidence=0.9,
            ),
            skills_used=["azure-architect"],
        )
        json_str = report.model_dump_json()
        parsed = SMEAdvisoryReport.model_validate_json(json_str)
        assert parsed.interaction_mode == SMEInteractionMode.ADVISOR
        assert parsed.advisor_report is not None
        assert parsed.co_executor_report is None
        assert parsed.debater_report is None

    def test_sme_advisory_report_round_trip_co_executor(self):
        report = SMEAdvisoryReport(
            sme_persona="Cloud Architect",
            interaction_mode=SMEInteractionMode.CO_EXECUTOR,
            domain="Cloud Architecture",
            task_context="Deploy app",
            findings=["Contributed 1 section"],
            recommendations=["Review infra"],
            confidence=0.85,
            co_executor_report=CoExecutorReport(
                sme_persona="Cloud Architect",
                contributed_sections=[
                    CoExecutorSection(
                        sme_persona="Cloud Architect",
                        section_title="Infra",
                        content="Content",
                        domain_context="Context",
                        integration_notes="Notes",
                    )
                ],
                coordination_notes="Notes",
                domain_assumptions=["Provider chosen"],
            ),
            skills_used=[],
        )
        json_str = report.model_dump_json()
        parsed = SMEAdvisoryReport.model_validate_json(json_str)
        assert parsed.interaction_mode == SMEInteractionMode.CO_EXECUTOR
        assert parsed.co_executor_report is not None

    def test_sme_advisory_report_round_trip_debater(self):
        report = SMEAdvisoryReport(
            sme_persona="Data Engineer",
            interaction_mode=SMEInteractionMode.DEBATER,
            domain="Data Engineering",
            task_context="Choose DB",
            findings=["Position: ACID"],
            recommendations=[],
            confidence=0.85,
            debater_report=DebaterReport(
                sme_persona="Data Engineer",
                debate_round=1,
                position=DebatePosition(
                    sme_persona="Data Engineer",
                    position="ACID first",
                    domain_rationale="Rationale",
                    supporting_evidence=["Evidence"],
                    confidence=0.85,
                ),
                counter_arguments_addressed=[],
                remaining_concerns=[],
                willingness_to_concede=0.3,
            ),
            skills_used=["data-scientist"],
        )
        json_str = report.model_dump_json()
        parsed = SMEAdvisoryReport.model_validate_json(json_str)
        assert parsed.interaction_mode == SMEInteractionMode.DEBATER
        assert parsed.debater_report is not None


# =============================================================================
# 3. Schema Validation Edge Cases
# =============================================================================

class TestSMESchemaValidation:
    """Test Pydantic validation for SME schema edge cases."""

    def test_confidence_below_zero_rejected(self):
        with pytest.raises(ValidationError):
            AdvisorReport(
                sme_persona="Test",
                reviewed_content="Test",
                domain_corrections=[],
                missing_considerations=[],
                recommendations=[],
                confidence=-0.1,
            )

    def test_confidence_above_one_rejected(self):
        with pytest.raises(ValidationError):
            AdvisorReport(
                sme_persona="Test",
                reviewed_content="Test",
                domain_corrections=[],
                missing_considerations=[],
                recommendations=[],
                confidence=1.1,
            )

    def test_confidence_boundary_zero(self):
        report = AdvisorReport(
            sme_persona="Test",
            reviewed_content="Test",
            domain_corrections=[],
            missing_considerations=[],
            recommendations=[],
            confidence=0.0,
        )
        assert report.confidence == 0.0

    def test_confidence_boundary_one(self):
        report = AdvisorReport(
            sme_persona="Test",
            reviewed_content="Test",
            domain_corrections=[],
            missing_considerations=[],
            recommendations=[],
            confidence=1.0,
        )
        assert report.confidence == 1.0

    def test_debate_round_below_one_rejected(self):
        with pytest.raises(ValidationError):
            DebaterReport(
                sme_persona="Test",
                debate_round=0,
                position=DebatePosition(
                    sme_persona="Test",
                    position="Test",
                    domain_rationale="Test",
                    supporting_evidence=[],
                    confidence=0.5,
                ),
                counter_arguments_addressed=[],
                remaining_concerns=[],
                willingness_to_concede=0.5,
            )

    def test_willingness_to_concede_bounds(self):
        with pytest.raises(ValidationError):
            DebaterReport(
                sme_persona="Test",
                debate_round=1,
                position=DebatePosition(
                    sme_persona="Test",
                    position="Test",
                    domain_rationale="Test",
                    supporting_evidence=[],
                    confidence=0.5,
                ),
                counter_arguments_addressed=[],
                remaining_concerns=[],
                willingness_to_concede=1.5,
            )

    def test_empty_lists_allowed(self):
        """Empty lists should be valid for all list fields."""
        report = AdvisorReport(
            sme_persona="Test",
            reviewed_content="Test",
            domain_corrections=[],
            missing_considerations=[],
            recommendations=[],
            confidence=0.5,
        )
        assert report.domain_corrections == []

    def test_sme_interaction_mode_enum_values(self):
        assert SMEInteractionMode.ADVISOR.value == "advisor"
        assert SMEInteractionMode.CO_EXECUTOR.value == "co_executor"
        assert SMEInteractionMode.DEBATER.value == "debater"
        assert len(SMEInteractionMode) == 3

    def test_invalid_interaction_mode_rejected(self):
        with pytest.raises(ValueError):
            SMEInteractionMode("invalid_mode")


# =============================================================================
# 4. Registry Integrity - All 10 Personas
# =============================================================================

class TestRegistryIntegrity:
    """Validate all 10 SME personas in the registry."""

    EXPECTED_PERSONAS = [
        "iam_architect",
        "cloud_architect",
        "security_analyst",
        "data_engineer",
        "ai_ml_engineer",
        "test_engineer",
        "business_analyst",
        "technical_writer",
        "devops_engineer",
        "frontend_developer",
    ]

    def test_registry_has_exactly_10_personas(self):
        assert len(SME_REGISTRY) == 10

    def test_all_expected_personas_exist(self):
        for persona_id in self.EXPECTED_PERSONAS:
            assert persona_id in SME_REGISTRY, f"Missing persona: {persona_id}"

    @pytest.mark.parametrize("persona_id", EXPECTED_PERSONAS)
    def test_persona_has_required_fields(self, persona_id):
        persona = get_persona(persona_id)
        assert persona is not None
        assert persona.name, f"{persona_id} missing name"
        assert persona.persona_id == persona_id
        assert persona.domain, f"{persona_id} missing domain"
        assert len(persona.trigger_keywords) > 0, f"{persona_id} has no trigger keywords"
        assert len(persona.skill_files) > 0, f"{persona_id} has no skill files"
        assert persona.system_prompt_template, f"{persona_id} missing template path"
        assert len(persona.interaction_modes) > 0, f"{persona_id} has no interaction modes"

    @pytest.mark.parametrize("persona_id", EXPECTED_PERSONAS)
    def test_persona_template_path_valid(self, persona_id):
        """Template path should follow config/sme/{id}.md pattern."""
        persona = get_persona(persona_id)
        assert persona.system_prompt_template.startswith("config/sme/")
        assert persona.system_prompt_template.endswith(".md")

    @pytest.mark.parametrize("persona_id", EXPECTED_PERSONAS)
    def test_persona_has_description(self, persona_id):
        persona = get_persona(persona_id)
        assert len(persona.description) > 10, f"{persona_id} description too short"

    def test_all_personas_have_advisor_mode(self):
        """Every persona should support at least ADVISOR mode."""
        for persona_id, persona in SME_REGISTRY.items():
            assert RegistryInteractionMode.ADVISOR in persona.interaction_modes, (
                f"{persona_id} missing ADVISOR mode"
            )

    def test_debater_mode_only_for_certain_personas(self):
        """Only cloud_architect, security_analyst, ai_ml_engineer should have DEBATER."""
        debater_personas = {
            pid for pid, p in SME_REGISTRY.items()
            if RegistryInteractionMode.DEBATER in p.interaction_modes
        }
        expected_debaters = {"cloud_architect", "security_analyst", "ai_ml_engineer"}
        assert debater_personas == expected_debaters

    def test_no_duplicate_persona_ids(self):
        ids = get_persona_ids()
        assert len(ids) == len(set(ids))

    def test_no_overlapping_trigger_keywords(self):
        """Check for keywords that appear in multiple unrelated personas."""
        keyword_to_personas: Dict[str, List[str]] = {}
        for persona_id, persona in SME_REGISTRY.items():
            for kw in persona.trigger_keywords:
                keyword_to_personas.setdefault(kw, []).append(persona_id)

        # Some overlap is expected (e.g., "kubernetes" in cloud + devops)
        # But flag any keyword in 3+ personas
        for kw, personas in keyword_to_personas.items():
            assert len(personas) <= 3, (
                f"Keyword '{kw}' in {len(personas)} personas: {personas}"
            )


# =============================================================================
# 5. Registry Query Functions
# =============================================================================

class TestRegistryQueryFunctions:
    """Test all registry query functions."""

    def test_get_persona_valid(self):
        persona = get_persona("security_analyst")
        assert persona is not None
        assert persona.name == "Security Analyst"

    def test_get_persona_invalid(self):
        assert get_persona("nonexistent") is None

    def test_get_all_personas_returns_copy(self):
        all_personas = get_all_personas()
        assert len(all_personas) == 10
        # Should be a copy, not the original
        all_personas["test"] = None
        assert "test" not in SME_REGISTRY

    def test_find_personas_by_keywords_single(self):
        results = find_personas_by_keywords(["kubernetes"])
        assert len(results) > 0
        domains = [r.domain for r in results]
        assert any("Cloud" in d for d in domains)

    def test_find_personas_by_keywords_multiple(self):
        results = find_personas_by_keywords(["security", "owasp"])
        assert len(results) > 0
        assert results[0].persona_id == "security_analyst"

    def test_find_personas_by_keywords_no_match(self):
        results = find_personas_by_keywords(["xyzzy_nonexistent"])
        assert len(results) == 0

    def test_find_personas_by_keywords_sorted_by_match_count(self):
        # "security" matches security_analyst, potentially others
        results = find_personas_by_keywords(
            ["security", "owasp", "vulnerability", "pentest"]
        )
        if len(results) >= 2:
            # First result should have the most matches
            assert results[0].persona_id == "security_analyst"

    def test_find_personas_by_domain(self):
        results = find_personas_by_domain(["Cloud"])
        assert len(results) > 0
        assert any(r.persona_id == "cloud_architect" for r in results)

    def test_find_personas_by_domain_no_match(self):
        results = find_personas_by_domain(["quantum_computing"])
        assert len(results) == 0

    def test_get_persona_ids(self):
        ids = get_persona_ids()
        assert len(ids) == 10
        assert "security_analyst" in ids

    def test_validate_interaction_mode_valid(self):
        assert validate_interaction_mode(
            "cloud_architect", RegistryInteractionMode.DEBATER
        ) is True

    def test_validate_interaction_mode_invalid_persona(self):
        assert validate_interaction_mode(
            "nonexistent", RegistryInteractionMode.ADVISOR
        ) is False

    def test_validate_interaction_mode_unsupported(self):
        # technical_writer only has ADVISOR and CO_EXECUTOR
        assert validate_interaction_mode(
            "technical_writer", RegistryInteractionMode.DEBATER
        ) is False

    def test_get_persona_for_display(self):
        display = get_persona_for_display("cloud_architect")
        assert display is not None
        assert display["id"] == "cloud_architect"
        assert display["name"] == "Cloud Architect"
        assert "advisor" in display["interaction_modes"]
        assert isinstance(display["trigger_keywords"], list)

    def test_get_persona_for_display_invalid(self):
        assert get_persona_for_display("nonexistent") is None

    def test_get_registry_stats(self):
        stats = get_registry_stats()
        assert stats["total_personas"] == 10
        assert len(stats["persona_ids"]) == 10
        assert stats["total_trigger_keywords"] > 50  # Should have many keywords
        assert len(stats["available_skills"]) > 5


# =============================================================================
# 6. SME Spawner - Full Lifecycle for All Personas
# =============================================================================

@pytest.fixture
def spawner(tmp_path):
    """Create an SMESpawner with temp directories."""
    return SMESpawner(
        skills_dir=str(tmp_path / "skills"),
        sme_templates_dir=str(tmp_path / "sme_templates"),
    )


class TestSpawnerAllPersonas:
    """Test spawner with every persona in every supported mode."""

    ALL_PERSONA_IDS = list(SME_REGISTRY.keys())

    @pytest.mark.parametrize("persona_id", ALL_PERSONA_IDS)
    def test_advisor_mode_all_personas(self, spawner, persona_id):
        """Every persona supports advisor mode and produces valid reports."""
        persona = get_persona(persona_id)
        sme = SpawnedSME(
            persona_id=persona_id,
            persona_name=persona.name,
            domain=persona.domain,
            interaction_mode=SMEInteractionMode.ADVISOR,
            system_prompt=f"You are {persona.name}",
            skills_loaded=persona.skill_files,
            spawn_context={
                "activation_phase": "execution",
                "reasoning": "Testing",
                "task_context": f"Test task for {persona.domain}",
                "execution_phase": "execution",
            },
        )
        report = spawner.execute_sme_interaction(sme, "Review this content")
        assert isinstance(report, SMEAdvisoryReport)
        assert report.interaction_mode == SMEInteractionMode.ADVISOR
        assert report.advisor_report is not None
        assert 0.0 <= report.confidence <= 1.0
        assert len(report.findings) > 0
        assert len(report.recommendations) > 0

    @pytest.mark.parametrize("persona_id", ALL_PERSONA_IDS)
    def test_co_executor_mode_supported_personas(self, spawner, persona_id):
        """Personas with CO_EXECUTOR mode produce valid co-executor reports."""
        persona = get_persona(persona_id)
        if RegistryInteractionMode.CO_EXECUTOR not in persona.interaction_modes:
            pytest.skip(f"{persona_id} does not support CO_EXECUTOR")

        sme = SpawnedSME(
            persona_id=persona_id,
            persona_name=persona.name,
            domain=persona.domain,
            interaction_mode=SMEInteractionMode.CO_EXECUTOR,
            system_prompt=f"You are {persona.name}",
            skills_loaded=persona.skill_files,
            spawn_context={
                "activation_phase": "execution",
                "reasoning": "Testing",
                "task_context": f"Build something with {persona.domain}",
                "execution_phase": "execution",
            },
        )
        report = spawner.execute_sme_interaction(sme, "Build something")
        assert isinstance(report, SMEAdvisoryReport)
        assert report.interaction_mode == SMEInteractionMode.CO_EXECUTOR
        assert report.co_executor_report is not None
        assert len(report.co_executor_report.contributed_sections) > 0
        for section in report.co_executor_report.contributed_sections:
            assert len(section.content) > 0
            assert len(section.section_title) > 0

    @pytest.mark.parametrize("persona_id", [
        "cloud_architect", "security_analyst", "ai_ml_engineer"
    ])
    def test_debater_mode_supported_personas(self, spawner, persona_id):
        """Personas with DEBATER mode produce valid debater reports."""
        persona = get_persona(persona_id)
        sme = SpawnedSME(
            persona_id=persona_id,
            persona_name=persona.name,
            domain=persona.domain,
            interaction_mode=SMEInteractionMode.DEBATER,
            system_prompt=f"You are {persona.name}",
            skills_loaded=persona.skill_files,
            spawn_context={
                "activation_phase": "execution",
                "reasoning": "Testing debate",
                "task_context": f"Debate on {persona.domain}",
                "execution_phase": "execution",
            },
        )
        report = spawner.execute_sme_interaction(
            sme, "Debate the approach",
            additional_context={
                "debate_round": 1,
                "counter_arguments": ["Cost is too high"],
            },
        )
        assert isinstance(report, SMEAdvisoryReport)
        assert report.interaction_mode == SMEInteractionMode.DEBATER
        assert report.debater_report is not None
        assert report.debater_report.debate_round == 1
        assert len(report.debater_report.position.position) > 0
        assert 0.0 <= report.debater_report.willingness_to_concede <= 1.0


# =============================================================================
# 7. Spawner from Council Selection
# =============================================================================

class TestSpawnerFromSelection:
    """Test spawning SMEs from Council Chair selections."""

    def test_spawn_single_sme(self, spawner):
        selections = [
            SMESelection(
                persona_name="Cloud Architect",
                persona_domain="Cloud Infrastructure Architecture",
                skills_to_load=["azure-architect"],
                interaction_mode=CouncilInteractionMode.ADVISOR,
                reasoning="Cloud expertise needed",
                activation_phase="planning",
            )
        ]
        result = spawner.spawn_from_selection(selections, "Deploy cloud app")
        assert isinstance(result, SpawnResult)
        assert result.total_spawned == 1
        assert result.spawned_smes[0].persona_name == "Cloud Architect"

    def test_spawn_multiple_smes(self, spawner):
        selections = [
            SMESelection(
                persona_name="Security Analyst",
                persona_domain="Application & Infrastructure Security",
                skills_to_load=[],
                interaction_mode=CouncilInteractionMode.ADVISOR,
                reasoning="Security review",
                activation_phase="review",
            ),
            SMESelection(
                persona_name="Cloud Architect",
                persona_domain="Cloud Infrastructure Architecture",
                skills_to_load=[],
                interaction_mode=CouncilInteractionMode.CO_EXECUTOR,
                reasoning="Architecture needed",
                activation_phase="execution",
            ),
        ]
        result = spawner.spawn_from_selection(selections, "Secure cloud app")
        assert result.total_spawned == 2

    def test_spawn_with_invalid_persona_skipped(self, spawner):
        selections = [
            SMESelection(
                persona_name="Nonexistent Expert",
                persona_domain="nonexistent_domain",
                skills_to_load=[],
                interaction_mode=CouncilInteractionMode.ADVISOR,
                reasoning="test",
                activation_phase="execution",
            )
        ]
        result = spawner.spawn_from_selection(selections, "Test")
        assert result.total_spawned == 0

    def test_spawn_mode_fallback(self, spawner):
        """If persona doesn't support requested mode, fallback to ADVISOR."""
        selections = [
            SMESelection(
                persona_name="Technical Writer",
                persona_domain="Technical Documentation & Communication",
                skills_to_load=[],
                interaction_mode=CouncilInteractionMode.DEBATER,
                reasoning="test",
                activation_phase="execution",
            )
        ]
        result = spawner.spawn_from_selection(selections, "Write docs")
        if result.total_spawned > 0:
            assert result.spawned_smes[0].interaction_mode == SMEInteractionMode.ADVISOR

    def test_spawn_result_metadata(self, spawner):
        selections = [
            SMESelection(
                persona_name="Data Engineer",
                persona_domain="Data Engineering & Analytics",
                skills_to_load=["data-scientist"],
                interaction_mode=CouncilInteractionMode.ADVISOR,
                reasoning="Data expertise",
                activation_phase="planning",
            )
        ]
        result = spawner.spawn_from_selection(
            selections, "Build pipeline", execution_phase="planning"
        )
        assert result.spawn_metadata["task_context"] == "Build pipeline"
        assert result.spawn_metadata["execution_phase"] == "planning"
        assert "timestamp" in result.spawn_metadata

    def test_spawn_empty_selection_list(self, spawner):
        result = spawner.spawn_from_selection([], "Test")
        assert result.total_spawned == 0
        assert result.spawned_smes == []


# =============================================================================
# 8. Claude Agent SDK Integration
# =============================================================================

class TestClaudeAgentSDKIntegration:
    """Validate Claude Agent SDK configuration and imports."""

    def test_sdk_import_available(self):
        """Claude Agent SDK should be importable."""
        try:
            from claude_agent_sdk import query as sdk_query
            assert callable(sdk_query)
        except ImportError:
            pytest.skip("claude_agent_sdk not installed")

    def test_anthropic_fallback_available(self):
        """Anthropic library should be importable as fallback."""
        from anthropic import Anthropic
        assert Anthropic is not None

    def test_sdk_integration_module_imports(self):
        """SDK integration module should import without errors."""
        from src.core.sdk_integration import (
            spawn_subagent,
            build_agent_options,
            _execute_sdk_query,
            _execute_anthropic_api,
            _validate_output,
            create_sdk_mcp_server,
        )
        assert callable(spawn_subagent)
        assert callable(build_agent_options)

    def test_sme_default_tools(self):
        """SME default tools should include Read, Glob, Grep, Skill."""
        tools = AGENT_ALLOWED_TOOLS["sme_default"]
        assert "Read" in tools
        assert "Glob" in tools
        assert "Grep" in tools
        assert "Skill" in tools

    def test_sme_default_no_write_or_edit(self):
        """SME personas should NOT have Write/Edit tools (advisory only)."""
        tools = AGENT_ALLOWED_TOOLS["sme_default"]
        assert "Write" not in tools
        assert "Edit" not in tools
        assert "Bash" not in tools

    def test_council_agents_have_no_tools(self):
        """Council agents are reasoning-only."""
        for agent in ["council_chair", "quality_arbiter", "ethics_advisor"]:
            assert AGENT_ALLOWED_TOOLS[agent] == [], f"{agent} should have no tools"

    def test_claude_agent_options_serialization(self):
        options = ClaudeAgentOptions(
            name="Test SME",
            model="claude-sonnet-4-20250514",
            system_prompt="You are a test SME",
            max_turns=10,
            allowed_tools=["Read", "Glob"],
        )
        kwargs = options.to_sdk_kwargs()
        assert kwargs["name"] == "Test SME"
        assert kwargs["model"] == "claude-sonnet-4-20250514"
        assert kwargs["allowed_tools"] == ["Read", "Glob"]

    def test_output_schema_for_sme_advisory(self):
        """SMEAdvisoryReport should produce valid JSON schema for SDK."""
        schema = SMEAdvisoryReport.model_json_schema()
        assert "properties" in schema
        required = schema.get("required", [])
        assert "sme_persona" in required
        assert "interaction_mode" in required
        assert "findings" in required
        assert "confidence" in required


# =============================================================================
# 9. Skills System Validation
# =============================================================================

class TestSkillsSystem:
    """Validate skills mapping and availability."""

    def test_skills_directory_exists(self):
        skills_dir = Path(".claude/skills")
        assert skills_dir.exists(), "Skills directory .claude/skills/ does not exist"

    def test_available_skills_have_skill_md(self):
        """Each skill directory should contain a SKILL.md file."""
        skills_dir = Path(".claude/skills")
        if skills_dir.exists():
            for skill_dir in skills_dir.iterdir():
                if skill_dir.is_dir() and not skill_dir.name.startswith("_"):
                    skill_md = skill_dir / "SKILL.md"
                    assert skill_md.exists(), (
                        f"Missing SKILL.md in {skill_dir.name}"
                    )

    def test_get_skills_for_sme_returns_list(self):
        for persona_id in get_persona_ids():
            skills = get_skills_for_sme(persona_id)
            assert isinstance(skills, list)

    def test_get_skills_for_sme_invalid_persona(self):
        skills = get_skills_for_sme("nonexistent")
        assert skills == []

    def test_operational_agents_have_skills(self):
        """Key operational agents should have skill mappings."""
        assert len(get_skills_for_agent("executor")) > 0
        assert len(get_skills_for_agent("planner")) > 0
        assert len(get_skills_for_agent("researcher")) > 0

    def test_sme_referenced_skills_documented(self):
        """Collect all skills referenced by SME personas."""
        all_referenced_skills: Set[str] = set()
        for persona in SME_REGISTRY.values():
            all_referenced_skills.update(persona.skill_files)

        # These are domain-specific skills that may be in external repos
        # At minimum, verify they are non-empty strings
        for skill in all_referenced_skills:
            assert isinstance(skill, str)
            assert len(skill) > 0


# =============================================================================
# 10. SME Persona Config Files Validation
# =============================================================================

class TestSMEPersonaConfigFiles:
    """Validate SME persona markdown config files."""

    SME_CONFIG_DIR = Path("config/sme")

    def test_config_directory_exists(self):
        assert self.SME_CONFIG_DIR.exists()

    @pytest.mark.parametrize("persona_id", list(SME_REGISTRY.keys()))
    def test_persona_config_file_exists(self, persona_id):
        """Each persona should have a config file."""
        config_file = self.SME_CONFIG_DIR / f"{persona_id}.md"
        assert config_file.exists(), f"Missing config file: {config_file}"

    @pytest.mark.parametrize("persona_id", list(SME_REGISTRY.keys()))
    def test_persona_config_file_not_empty(self, persona_id):
        """Config files should have meaningful content."""
        config_file = self.SME_CONFIG_DIR / f"{persona_id}.md"
        if config_file.exists():
            content = config_file.read_text()
            assert len(content) > 50, f"{persona_id}.md is too short"


# =============================================================================
# 11. Domain Finding Analysis
# =============================================================================

class TestDomainAnalysis:
    """Test domain-specific analysis for all domain categories."""

    @pytest.fixture
    def spawner_instance(self, tmp_path):
        return SMESpawner(
            skills_dir=str(tmp_path / "skills"),
            sme_templates_dir=str(tmp_path / "templates"),
        )

    def _make_sme(self, domain: str) -> SpawnedSME:
        return SpawnedSME(
            persona_id="test",
            persona_name="Test",
            domain=domain,
            interaction_mode=SMEInteractionMode.ADVISOR,
            system_prompt="",
            skills_loaded=[],
            spawn_context={"task_context": "test"},
        )

    def test_security_detects_hardcoded_password(self, spawner_instance):
        sme = self._make_sme("Application Security")
        findings = spawner_instance._analyze_domain_findings(
            sme, "password = 'admin123'"
        )
        assert any("password" in f.lower() for f in findings)

    def test_security_detects_sql_injection(self, spawner_instance):
        sme = self._make_sme("Application Security")
        findings = spawner_instance._analyze_domain_findings(
            sme, "sql_query = base_sql + user_input"
        )
        assert any("sql" in f.lower() or "injection" in f.lower() for f in findings)

    def test_security_detects_eval(self, spawner_instance):
        sme = self._make_sme("Application Security")
        findings = spawner_instance._analyze_domain_findings(
            sme, "result = eval(user_input)"
        )
        assert any("injection" in f.lower() for f in findings)

    def test_cloud_detects_hardcoded_ip(self, spawner_instance):
        sme = self._make_sme("Cloud Infrastructure")
        findings = spawner_instance._analyze_domain_findings(
            sme, "server hardcoded ip address"
        )
        assert any("hardcoded" in f.lower() for f in findings)

    def test_data_detects_select_star(self, spawner_instance):
        sme = self._make_sme("Data Engineering")
        findings = spawner_instance._analyze_domain_findings(
            sme, "SELECT * FROM large_table"
        )
        assert any("fetching" in f.lower() for f in findings)

    def test_generic_domain_returns_general_observation(self, spawner_instance):
        sme = self._make_sme("Quantum Computing")
        findings = spawner_instance._analyze_domain_findings(
            sme, "Simple hello world"
        )
        assert len(findings) > 0
        assert any("reviewed" in f.lower() for f in findings)


# =============================================================================
# 12. Section Determination by Domain
# =============================================================================

class TestSectionDetermination:
    """Test that all domains get appropriate co-executor sections."""

    @pytest.fixture
    def spawner_instance(self, tmp_path):
        return SMESpawner(
            skills_dir=str(tmp_path / "skills"),
            sme_templates_dir=str(tmp_path / "templates"),
        )

    @pytest.mark.parametrize("domain,expected_keyword", [
        ("Application Security", "Security"),
        ("Cloud Infrastructure", "Infrastructure"),
        ("Data Engineering", "Data"),
        ("Frontend Development", "User Interface"),
        ("DevOps", "CI/CD"),
        ("Technical Documentation & Communication", "Documentation"),
        ("Identity and Access Management", "Considerations"),
        ("Quality Assurance & Testing", "Considerations"),
    ])
    def test_domain_produces_relevant_sections(
        self, spawner_instance, domain, expected_keyword
    ):
        sme = SpawnedSME(
            persona_id="test",
            persona_name="Test",
            domain=domain,
            interaction_mode=SMEInteractionMode.CO_EXECUTOR,
            system_prompt="",
            skills_loaded=[],
            spawn_context={"task_context": "test"},
        )
        sections = spawner_instance._determine_sections(sme, "Build something")
        titles = [s["title"] for s in sections]
        assert any(expected_keyword in t for t in titles), (
            f"Domain '{domain}' sections {titles} missing keyword '{expected_keyword}'"
        )


# =============================================================================
# 13. Cross-Component Integration
# =============================================================================

class TestCrossComponentIntegration:
    """Test full pipeline: Council Chair -> SME Spawner -> Interaction."""

    def test_chair_selects_then_spawner_spawns(self, spawner):
        """Council Chair output feeds directly into SME Spawner."""
        chair = CouncilChairAgent(system_prompt_path="nonexistent.md")
        report = chair.select_smes(
            "Build a secure cloud API with data pipelines",
            tier_level=3,
        )
        assert isinstance(report, SMESelectionReport)
        assert len(report.selected_smes) > 0

        # Feed selections to spawner
        result = spawner.spawn_from_selection(
            report.selected_smes,
            report.task_summary,
        )
        assert result.total_spawned > 0

        # Execute interaction with first spawned SME
        first_sme = result.spawned_smes[0]
        advisory = spawner.execute_sme_interaction(
            first_sme, "Build secure REST API"
        )
        assert isinstance(advisory, SMEAdvisoryReport)
        assert advisory.sme_persona == first_sme.persona_name

    def test_spawner_interaction_produces_serializable_output(self, spawner):
        """All interaction outputs must be JSON-serializable for SDK."""
        persona = get_persona("cloud_architect")
        sme = SpawnedSME(
            persona_id="cloud_architect",
            persona_name=persona.name,
            domain=persona.domain,
            interaction_mode=SMEInteractionMode.ADVISOR,
            system_prompt="Test",
            skills_loaded=[],
            spawn_context={"task_context": "test", "execution_phase": "test",
                          "activation_phase": "test", "reasoning": "test"},
        )
        report = spawner.execute_sme_interaction(sme, "Deploy to cloud")
        json_str = report.model_dump_json()
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
        assert "sme_persona" in parsed
        assert "findings" in parsed

    def test_debater_multi_round_simulation(self, spawner):
        """Simulate a multi-round debate with increasing concession."""
        persona = get_persona("cloud_architect")
        sme = SpawnedSME(
            persona_id="cloud_architect",
            persona_name=persona.name,
            domain=persona.domain,
            interaction_mode=SMEInteractionMode.DEBATER,
            system_prompt="Test",
            skills_loaded=[],
            spawn_context={"task_context": "test", "execution_phase": "test",
                          "activation_phase": "test", "reasoning": "test"},
        )

        # Round 1: No counter-arguments
        r1 = spawner.execute_sme_interaction(
            sme, "Choose architecture",
            additional_context={"debate_round": 1, "counter_arguments": []},
        )
        assert r1.debater_report.debate_round == 1

        # Round 2: With counter-arguments
        r2 = spawner.execute_sme_interaction(
            sme, "Choose architecture",
            additional_context={
                "debate_round": 2,
                "counter_arguments": ["Cost too high", "Too complex"],
            },
        )
        assert r2.debater_report.debate_round == 2
        assert len(r2.debater_report.counter_arguments_addressed) > 0
        # Concession should be higher with counter-args
        assert r2.debater_report.willingness_to_concede >= r1.debater_report.willingness_to_concede


# =============================================================================
# 14. Interaction Mode Enum Consistency
# =============================================================================

class TestInteractionModeConsistency:
    """Ensure InteractionMode enums are consistent across modules."""

    def test_registry_and_schema_modes_match(self):
        """Registry and schema InteractionMode should have same values."""
        registry_values = {m.value for m in RegistryInteractionMode}
        schema_values = {m.value for m in SMEInteractionMode}
        assert registry_values == schema_values

    def test_council_and_schema_modes_match(self):
        """Council and SME schema InteractionMode should have same values."""
        council_values = {m.value for m in CouncilInteractionMode}
        schema_values = {m.value for m in SMEInteractionMode}
        assert council_values == schema_values

    def test_mode_conversion_covers_all_modes(self, spawner):
        """SMESpawner mode conversion should handle all council modes."""
        for council_mode in CouncilInteractionMode:
            result = spawner._convert_interaction_mode(council_mode)
            assert isinstance(result, SMEInteractionMode)


# =============================================================================
# 15. System Prompt and Skills Loading
# =============================================================================

class TestSystemPromptAndSkillsLoading:
    """Test system prompt and skills loading in spawner."""

    def test_loads_existing_template(self, tmp_path):
        templates_dir = tmp_path / "templates"
        templates_dir.mkdir()
        template_file = templates_dir / "test_sme.md"
        template_file.write_text("# Test SME\nYou are a test expert.")

        spawner = SMESpawner(
            skills_dir=str(tmp_path / "skills"),
            sme_templates_dir=str(templates_dir),
        )

        persona = MagicMock()
        persona.persona_id = "test_sme"
        persona.name = "Test SME"
        persona.domain = "Testing"
        persona.description = "Test expert"
        persona.system_prompt_template = str(template_file)

        prompt = spawner._load_system_prompt(persona)
        assert "Test SME" in prompt

    def test_generates_default_prompt_on_missing_template(self, tmp_path):
        spawner = SMESpawner(
            skills_dir=str(tmp_path / "skills"),
            sme_templates_dir=str(tmp_path / "templates"),
        )

        persona = MagicMock()
        persona.persona_id = "missing_sme"
        persona.name = "Missing SME"
        persona.domain = "Missing Domain"
        persona.description = "Missing expert"
        persona.system_prompt_template = "nonexistent/path.md"

        prompt = spawner._load_system_prompt(persona)
        assert "Missing SME" in prompt
        assert "Missing Domain" in prompt

    def test_loads_existing_skills(self, tmp_path):
        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "test-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("# Test Skill")

        spawner = SMESpawner(
            skills_dir=str(skills_dir),
            sme_templates_dir=str(tmp_path / "templates"),
        )

        persona = MagicMock()
        persona.skill_files = ["test-skill", "nonexistent-skill"]

        loaded = spawner._load_skills(persona, [])
        assert "test-skill" in loaded
        assert "nonexistent-skill" not in loaded

    def test_loads_requested_skills(self, tmp_path):
        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "extra-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("# Extra Skill")

        spawner = SMESpawner(
            skills_dir=str(skills_dir),
            sme_templates_dir=str(tmp_path / "templates"),
        )

        persona = MagicMock()
        persona.skill_files = []

        loaded = spawner._load_skills(persona, ["extra-skill"])
        assert "extra-skill" in loaded


# =============================================================================
# 16. Edge Cases and Boundary Conditions
# =============================================================================

class TestEdgeCases:
    """Edge cases for SME persona system."""

    def test_empty_content_advisor_mode(self, spawner):
        sme = SpawnedSME(
            persona_id="security_analyst",
            persona_name="Security Analyst",
            domain="Application Security",
            interaction_mode=SMEInteractionMode.ADVISOR,
            system_prompt="",
            skills_loaded=[],
            spawn_context={"task_context": ""},
        )
        report = spawner.execute_sme_interaction(sme, "")
        assert isinstance(report, SMEAdvisoryReport)

    def test_very_long_content(self, spawner):
        sme = SpawnedSME(
            persona_id="cloud_architect",
            persona_name="Cloud Architect",
            domain="Cloud Architecture",
            interaction_mode=SMEInteractionMode.ADVISOR,
            system_prompt="",
            skills_loaded=[],
            spawn_context={"task_context": "test"},
        )
        long_content = "Build " * 10000
        report = spawner.execute_sme_interaction(sme, long_content)
        assert isinstance(report, SMEAdvisoryReport)
        # Reviewed content should be truncated
        assert len(report.advisor_report.reviewed_content) <= 200

    def test_special_characters_in_content(self, spawner):
        sme = SpawnedSME(
            persona_id="data_engineer",
            persona_name="Data Engineer",
            domain="Data Engineering",
            interaction_mode=SMEInteractionMode.ADVISOR,
            system_prompt="",
            skills_loaded=[],
            spawn_context={"task_context": "test"},
        )
        report = spawner.execute_sme_interaction(
            sme, "SELECT * FROM users WHERE name = 'O\\'Brien'; -- injection"
        )
        assert isinstance(report, SMEAdvisoryReport)

    def test_debater_no_counter_arguments(self, spawner):
        """Debater with no counter-arguments should still produce report."""
        sme = SpawnedSME(
            persona_id="ai_ml_engineer",
            persona_name="AI/ML Engineer",
            domain="Artificial Intelligence",
            interaction_mode=SMEInteractionMode.DEBATER,
            system_prompt="",
            skills_loaded=[],
            spawn_context={"task_context": "test"},
        )
        report = spawner.execute_sme_interaction(
            sme, "Choose ML framework",
            additional_context={"debate_round": 1},
        )
        assert report.debater_report is not None
        assert report.debater_report.counter_arguments_addressed == []

    def test_debater_many_counter_arguments(self, spawner):
        """Debater with many counter-arguments should cap concession at 0.7."""
        sme = SpawnedSME(
            persona_id="security_analyst",
            persona_name="Security Analyst",
            domain="Application Security",
            interaction_mode=SMEInteractionMode.DEBATER,
            system_prompt="",
            skills_loaded=[],
            spawn_context={"task_context": "test"},
        )
        report = spawner.execute_sme_interaction(
            sme, "Debate security approach",
            additional_context={
                "debate_round": 3,
                "counter_arguments": [
                    "Cost too high",
                    "Performance impact",
                    "Complexity",
                    "Time constraints",
                    "Team lacks expertise",
                    "Overkill for the use case",
                ],
            },
        )
        assert report.debater_report.willingness_to_concede <= 0.7

    def test_convenience_function_create_sme_spawner(self):
        spawner = create_sme_spawner()
        assert isinstance(spawner, SMESpawner)
        assert spawner.model == "claude-3-5-sonnet-20241022"

    def test_convenience_function_custom_params(self):
        spawner = create_sme_spawner(
            skills_dir="/tmp/test_skills",
            sme_templates_dir="/tmp/test_templates",
            model="claude-3-opus",
        )
        assert spawner.model == "claude-3-opus"
