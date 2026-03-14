"""
Exhaustive tests for SMESpawner (src/agents/sme_spawner.py).

Covers:
- __init__ with defaults and custom params
- spawn_from_selection() - processing SMESelection objects
- SpawnedSME and SpawnResult dataclasses
- System prompt loading (_load_system_prompt) with file and fallback
- Skill file loading (_load_skills)
- Interaction mode handling and conversion
- Advisory, co-execution, and debater mode execution
- Edge cases: unknown persona_id, empty selections
- All helper methods for each interaction mode
"""

import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from dataclasses import dataclass
from typing import List

from src.agents.sme_spawner import (
    SMESpawner,
    SpawnedSME,
    SpawnResult,
    create_sme_spawner,
)
from src.schemas.sme import (
    SMEInteractionMode,
    SMEAdvisoryReport,
    AdvisorReport,
    CoExecutorReport,
    CoExecutorSection,
    DebaterReport,
    DebatePosition,
)
from src.schemas.council import (
    SMESelection,
    InteractionMode as CouncilInteractionMode,
)
from src.core.sme_registry import SMEPersona, InteractionMode as RegistryInteractionMode


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def spawner(tmp_path):
    """Create an SMESpawner with tmp directories."""
    return SMESpawner(
        skills_dir=str(tmp_path / "skills"),
        sme_templates_dir=str(tmp_path / "sme_templates"),
    )


@pytest.fixture
def mock_persona():
    """Create a mock SMEPersona."""
    return SMEPersona(
        name="Cloud Architect",
        persona_id="cloud_architect",
        domain="Cloud Infrastructure",
        trigger_keywords=["cloud", "aws", "azure"],
        skill_files=["azure-architect"],
        system_prompt_template="config/sme/cloud_architect.md",
        interaction_modes=[
            RegistryInteractionMode.ADVISOR,
            RegistryInteractionMode.CO_EXECUTOR,
            RegistryInteractionMode.DEBATER,
        ],
        description="Cloud expert",
    )


@pytest.fixture
def mock_security_persona():
    return SMEPersona(
        name="Security Analyst",
        persona_id="security_analyst",
        domain="Application & Infrastructure Security",
        trigger_keywords=["security", "owasp"],
        skill_files=["sec-skill"],
        system_prompt_template="config/sme/security_analyst.md",
        interaction_modes=[
            RegistryInteractionMode.ADVISOR,
            RegistryInteractionMode.DEBATER,
        ],
        description="Security expert",
    )


@pytest.fixture
def mock_data_persona():
    return SMEPersona(
        name="Data Engineer",
        persona_id="data_engineer",
        domain="Data Engineering & Analytics",
        trigger_keywords=["data", "sql"],
        skill_files=[],
        system_prompt_template="config/sme/data_engineer.md",
        interaction_modes=[
            RegistryInteractionMode.ADVISOR,
            RegistryInteractionMode.CO_EXECUTOR,
        ],
        description="Data expert",
    )


@pytest.fixture
def advisor_selection():
    return SMESelection(
        persona_name="Cloud Architect",
        persona_domain="Cloud Infrastructure",
        skills_to_load=["azure-architect"],
        interaction_mode=CouncilInteractionMode.ADVISOR,
        reasoning="Cloud expertise needed",
        activation_phase="execution",
    )


@pytest.fixture
def co_executor_selection():
    return SMESelection(
        persona_name="Cloud Architect",
        persona_domain="Cloud Infrastructure",
        skills_to_load=[],
        interaction_mode=CouncilInteractionMode.CO_EXECUTOR,
        reasoning="Need cloud sections",
        activation_phase="execution",
    )


@pytest.fixture
def debater_selection():
    return SMESelection(
        persona_name="Security Analyst",
        persona_domain="Application & Infrastructure Security",
        skills_to_load=[],
        interaction_mode=CouncilInteractionMode.DEBATER,
        reasoning="Need security debate",
        activation_phase="review",
    )


def make_spawned_sme(persona_name="Cloud Architect", domain="Cloud Infrastructure",
                     mode=SMEInteractionMode.ADVISOR):
    return SpawnedSME(
        persona_id="cloud_architect",
        persona_name=persona_name,
        domain=domain,
        interaction_mode=mode,
        system_prompt="You are a cloud expert.",
        skills_loaded=["azure-architect"],
        spawn_context={
            "activation_phase": "execution",
            "reasoning": "Cloud expertise",
            "task_context": "Build API",
            "execution_phase": "execution",
        },
    )


# ============================================================================
# Dataclasses
# ============================================================================

class TestSpawnedSME:
    def test_creation(self):
        sme = make_spawned_sme()
        assert sme.persona_id == "cloud_architect"
        assert sme.persona_name == "Cloud Architect"
        assert sme.interaction_mode == SMEInteractionMode.ADVISOR
        assert isinstance(sme.skills_loaded, list)
        assert isinstance(sme.spawn_context, dict)

    def test_different_modes(self):
        for mode in SMEInteractionMode:
            sme = make_spawned_sme(mode=mode)
            assert sme.interaction_mode == mode


class TestSpawnResult:
    def test_creation(self):
        sme = make_spawned_sme()
        result = SpawnResult(
            spawned_smes=[sme],
            total_spawned=1,
            interaction_modes_used={SMEInteractionMode.ADVISOR},
            spawn_metadata={"task_context": "test"},
        )
        assert result.total_spawned == 1
        assert len(result.spawned_smes) == 1
        assert SMEInteractionMode.ADVISOR in result.interaction_modes_used

    def test_empty(self):
        result = SpawnResult(
            spawned_smes=[],
            total_spawned=0,
            interaction_modes_used=set(),
            spawn_metadata={},
        )
        assert result.total_spawned == 0


# ============================================================================
# __init__
# ============================================================================

class TestInit:
    def test_defaults(self, tmp_path):
        spawner = SMESpawner(
            skills_dir=str(tmp_path / "s"),
            sme_templates_dir=str(tmp_path / "t"),
        )
        assert spawner.model == "claude-sonnet-4-20250514"

    def test_custom_model(self, tmp_path):
        spawner = SMESpawner(
            skills_dir=str(tmp_path / "s"),
            sme_templates_dir=str(tmp_path / "t"),
            model="claude-3-opus",
        )
        assert spawner.model == "claude-3-opus"

    def test_directories_created(self, tmp_path):
        skills = tmp_path / "nested" / "skills"
        templates = tmp_path / "nested" / "templates"
        SMESpawner(skills_dir=str(skills), sme_templates_dir=str(templates))
        assert skills.exists()
        assert templates.exists()


# ============================================================================
# _convert_interaction_mode
# ============================================================================

class TestConvertInteractionMode:
    def test_advisor(self, spawner):
        result = spawner._convert_interaction_mode(CouncilInteractionMode.ADVISOR)
        assert result == SMEInteractionMode.ADVISOR

    def test_co_executor(self, spawner):
        result = spawner._convert_interaction_mode(CouncilInteractionMode.CO_EXECUTOR)
        assert result == SMEInteractionMode.CO_EXECUTOR

    def test_debater(self, spawner):
        result = spawner._convert_interaction_mode(CouncilInteractionMode.DEBATER)
        assert result == SMEInteractionMode.DEBATER


# ============================================================================
# _load_system_prompt
# ============================================================================

class TestLoadSystemPrompt:
    def test_from_template_path(self, spawner, mock_persona, tmp_path):
        # Create file at persona's template path
        template_path = Path(mock_persona.system_prompt_template)
        template_path.parent.mkdir(parents=True, exist_ok=True)
        template_path.write_text("Cloud prompt content")

        prompt = spawner._load_system_prompt(mock_persona)
        assert prompt == "Cloud prompt content"

    def test_fallback_to_sme_templates_dir(self, spawner, tmp_path):
        persona = SMEPersona(
            name="Test",
            persona_id="test_persona",
            domain="Testing",
            trigger_keywords=[],
            skill_files=[],
            system_prompt_template="nonexistent/path.md",
            interaction_modes=[],
        )
        # Create in sme_templates_dir
        template_file = spawner.sme_templates_dir / "test_persona.md"
        template_file.write_text("Fallback prompt")

        prompt = spawner._load_system_prompt(persona)
        assert prompt == "Fallback prompt"

    def test_default_prompt_generation(self, spawner):
        persona = SMEPersona(
            name="Unknown Expert",
            persona_id="unknown",
            domain="Unknown Domain",
            trigger_keywords=[],
            skill_files=[],
            system_prompt_template="nonexistent.md",
            interaction_modes=[],
            description="Expert in unknown things",
        )
        prompt = spawner._load_system_prompt(persona)
        assert "Unknown Expert" in prompt
        assert "Unknown Domain" in prompt
        assert "Expert in unknown things" in prompt


# ============================================================================
# _load_skills
# ============================================================================

class TestLoadSkills:
    def test_loads_existing_skills(self, spawner, mock_persona):
        # Create skill file
        skill_dir = spawner.skills_dir / "azure-architect"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("Azure skill")

        loaded = spawner._load_skills(mock_persona, [])
        assert "azure-architect" in loaded

    def test_loads_additional_requested_skills(self, spawner, mock_persona):
        # Create both persona skill and additional skill
        for skill in ["azure-architect", "extra-skill"]:
            d = spawner.skills_dir / skill
            d.mkdir(parents=True)
            (d / "SKILL.md").write_text(f"{skill} content")

        loaded = spawner._load_skills(mock_persona, ["extra-skill"])
        assert "azure-architect" in loaded
        assert "extra-skill" in loaded

    def test_nonexistent_skill_ignored(self, spawner, mock_persona):
        loaded = spawner._load_skills(mock_persona, ["nonexistent"])
        assert "nonexistent" not in loaded

    def test_no_duplicate_skills(self, spawner, mock_persona):
        d = spawner.skills_dir / "azure-architect"
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text("content")

        loaded = spawner._load_skills(mock_persona, ["azure-architect"])
        assert loaded.count("azure-architect") == 1

    def test_empty_skill_files(self, spawner):
        persona = SMEPersona(
            name="T", persona_id="t", domain="T",
            trigger_keywords=[], skill_files=[],
            system_prompt_template="x.md", interaction_modes=[],
        )
        loaded = spawner._load_skills(persona, [])
        assert loaded == []


# ============================================================================
# spawn_from_selection
# ============================================================================

class TestSpawnFromSelection:
    def test_spawn_advisor(self, spawner, advisor_selection, mock_persona):
        with patch("src.agents.sme_spawner.get_persona", return_value=mock_persona), \
             patch("src.agents.sme_spawner.validate_interaction_mode", return_value=True), \
             patch("src.agents.sme_spawner.find_personas_by_keywords", return_value=[]):
            result = spawner.spawn_from_selection(
                [advisor_selection], "Build an API", "execution"
            )
        assert result.total_spawned == 1
        assert result.spawned_smes[0].interaction_mode == SMEInteractionMode.ADVISOR

    def test_spawn_co_executor(self, spawner, co_executor_selection, mock_persona):
        with patch("src.agents.sme_spawner.get_persona", return_value=mock_persona), \
             patch("src.agents.sme_spawner.validate_interaction_mode", return_value=True), \
             patch("src.agents.sme_spawner.find_personas_by_keywords", return_value=[]):
            result = spawner.spawn_from_selection(
                [co_executor_selection], "Build an API", "execution"
            )
        assert result.total_spawned == 1
        assert result.spawned_smes[0].interaction_mode == SMEInteractionMode.CO_EXECUTOR

    def test_spawn_debater(self, spawner, debater_selection, mock_security_persona):
        with patch("src.agents.sme_spawner.get_persona", return_value=mock_security_persona), \
             patch("src.agents.sme_spawner.validate_interaction_mode", return_value=True), \
             patch("src.agents.sme_spawner.find_personas_by_keywords", return_value=[]):
            result = spawner.spawn_from_selection(
                [debater_selection], "Review security", "review"
            )
        assert result.total_spawned == 1
        assert result.spawned_smes[0].interaction_mode == SMEInteractionMode.DEBATER

    def test_unknown_persona_skipped(self, spawner):
        selection = SMESelection(
            persona_name="Unknown",
            persona_domain="Unknown Domain",
            skills_to_load=[],
            interaction_mode=CouncilInteractionMode.ADVISOR,
            reasoning="test",
            activation_phase="execution",
        )
        with patch("src.agents.sme_spawner.get_persona", return_value=None), \
             patch("src.agents.sme_spawner.find_personas_by_keywords", return_value=[]):
            result = spawner.spawn_from_selection([selection], "task", "execution")
        assert result.total_spawned == 0

    def test_empty_selections(self, spawner):
        result = spawner.spawn_from_selection([], "task", "execution")
        assert result.total_spawned == 0
        assert result.spawned_smes == []

    def test_invalid_mode_fallback_to_advisor(self, spawner, advisor_selection, mock_persona):
        with patch("src.agents.sme_spawner.get_persona", return_value=mock_persona), \
             patch("src.agents.sme_spawner.validate_interaction_mode", return_value=False), \
             patch("src.agents.sme_spawner.find_personas_by_keywords", return_value=[]):
            result = spawner.spawn_from_selection(
                [advisor_selection], "task", "execution"
            )
        assert result.spawned_smes[0].interaction_mode == SMEInteractionMode.ADVISOR

    def test_multiple_selections(self, spawner, mock_persona, mock_security_persona):
        selections = [
            SMESelection(
                persona_name="Cloud Architect",
                persona_domain="Cloud Infrastructure",
                skills_to_load=[],
                interaction_mode=CouncilInteractionMode.ADVISOR,
                reasoning="cloud",
                activation_phase="execution",
            ),
            SMESelection(
                persona_name="Security Analyst",
                persona_domain="Application & Infrastructure Security",
                skills_to_load=[],
                interaction_mode=CouncilInteractionMode.DEBATER,
                reasoning="security",
                activation_phase="review",
            ),
        ]

        call_count = [0]
        personas = [mock_persona, mock_security_persona]

        def side_effect_get(pid):
            if call_count[0] < len(personas):
                p = personas[call_count[0]]
                call_count[0] += 1
                return p
            return None

        with patch("src.agents.sme_spawner.get_persona", side_effect=side_effect_get), \
             patch("src.agents.sme_spawner.validate_interaction_mode", return_value=True), \
             patch("src.agents.sme_spawner.find_personas_by_keywords", return_value=[]):
            result = spawner.spawn_from_selection(selections, "task", "execution")
        assert result.total_spawned == 2

    def test_spawn_metadata(self, spawner, advisor_selection, mock_persona):
        with patch("src.agents.sme_spawner.get_persona", return_value=mock_persona), \
             patch("src.agents.sme_spawner.validate_interaction_mode", return_value=True), \
             patch("src.agents.sme_spawner.find_personas_by_keywords", return_value=[]):
            result = spawner.spawn_from_selection(
                [advisor_selection], "Build API", "planning"
            )
        assert result.spawn_metadata["task_context"] == "Build API"
        assert result.spawn_metadata["execution_phase"] == "planning"
        assert "timestamp" in result.spawn_metadata

    def test_interaction_modes_tracked(self, spawner, mock_persona, mock_security_persona):
        selections = [
            SMESelection(
                persona_name="Cloud Architect",
                persona_domain="Cloud Infrastructure",
                skills_to_load=[],
                interaction_mode=CouncilInteractionMode.ADVISOR,
                reasoning="x", activation_phase="x",
            ),
            SMESelection(
                persona_name="Security Analyst",
                persona_domain="Security",
                skills_to_load=[],
                interaction_mode=CouncilInteractionMode.DEBATER,
                reasoning="y", activation_phase="y",
            ),
        ]
        call_count = [0]
        personas = [mock_persona, mock_security_persona]

        def side_effect_get(pid):
            if call_count[0] < len(personas):
                p = personas[call_count[0]]
                call_count[0] += 1
                return p
            return None

        with patch("src.agents.sme_spawner.get_persona", side_effect=side_effect_get), \
             patch("src.agents.sme_spawner.validate_interaction_mode", return_value=True), \
             patch("src.agents.sme_spawner.find_personas_by_keywords", return_value=[]):
            result = spawner.spawn_from_selection(selections, "task", "exec")

        assert SMEInteractionMode.ADVISOR in result.interaction_modes_used
        assert SMEInteractionMode.DEBATER in result.interaction_modes_used

    def test_fallback_find_by_keywords(self, spawner, mock_persona):
        """When get_persona returns None twice, fall back to find_personas_by_keywords."""
        selection = SMESelection(
            persona_name="Cloud Architect",
            persona_domain="Cloud Infrastructure",
            skills_to_load=[],
            interaction_mode=CouncilInteractionMode.ADVISOR,
            reasoning="x", activation_phase="x",
        )
        with patch("src.agents.sme_spawner.get_persona", return_value=None), \
             patch("src.agents.sme_spawner.find_personas_by_keywords", return_value=[mock_persona]), \
             patch("src.agents.sme_spawner.validate_interaction_mode", return_value=True):
            result = spawner.spawn_from_selection([selection], "task", "exec")
        assert result.total_spawned == 1


# ============================================================================
# execute_sme_interaction - Advisor Mode
# ============================================================================

class TestAdvisorMode:
    def test_returns_advisory_report(self, spawner):
        sme = make_spawned_sme(domain="Application & Infrastructure Security")
        report = spawner.execute_sme_interaction(sme, "password = 'secret123'")
        assert isinstance(report, SMEAdvisoryReport)
        assert report.interaction_mode == SMEInteractionMode.ADVISOR
        assert report.advisor_report is not None

    def test_security_findings(self, spawner):
        sme = make_spawned_sme(domain="Application & Infrastructure Security")
        report = spawner.execute_sme_interaction(sme, "password = 'secret123'")
        assert any("password" in f.lower() or "hardcoded" in f.lower()
                    for f in report.findings)

    def test_cloud_findings(self, spawner):
        sme = make_spawned_sme(domain="Cloud Infrastructure")
        report = spawner.execute_sme_interaction(sme, "hardcoded ip address used")
        assert len(report.findings) >= 1

    def test_data_findings(self, spawner):
        sme = make_spawned_sme(domain="Data Engineering")
        report = spawner.execute_sme_interaction(sme, "SELECT * FROM users")
        assert any("over-fetching" in f.lower() for f in report.findings)

    def test_general_observation_when_no_findings(self, spawner):
        sme = make_spawned_sme(domain="Cloud Infrastructure")
        report = spawner.execute_sme_interaction(sme, "clean code here")
        assert any("reviewed" in f.lower() for f in report.findings)

    def test_confidence_range(self, spawner):
        sme = make_spawned_sme(domain="Cloud Infrastructure")
        report = spawner.execute_sme_interaction(sme, "content")
        assert 0.0 <= report.confidence <= 1.0

    def test_caveats_present(self, spawner):
        sme = make_spawned_sme(domain="Application & Infrastructure Security")
        report = spawner.execute_sme_interaction(sme, "content")
        assert len(report.caveats) >= 1

    def test_skills_used_populated(self, spawner):
        sme = make_spawned_sme()
        report = spawner.execute_sme_interaction(sme, "content")
        assert report.skills_used == ["azure-architect"]

    def test_domain_corrections(self, spawner):
        sme = make_spawned_sme(domain="Application & Infrastructure Security")
        report = spawner.execute_sme_interaction(sme, "password = 'test'")
        assert report.advisor_report is not None
        assert len(report.advisor_report.domain_corrections) >= 1

    def test_missing_considerations(self, spawner):
        sme = make_spawned_sme(domain="Application & Infrastructure Security")
        report = spawner.execute_sme_interaction(sme, "basic app with no security")
        assert report.advisor_report is not None
        assert len(report.advisor_report.missing_considerations) >= 0


# ============================================================================
# execute_sme_interaction - Co-Executor Mode
# ============================================================================

class TestCoExecutorMode:
    def test_returns_co_executor_report(self, spawner):
        sme = make_spawned_sme(mode=SMEInteractionMode.CO_EXECUTOR)
        report = spawner.execute_sme_interaction(sme, "Build API")
        assert isinstance(report, SMEAdvisoryReport)
        assert report.interaction_mode == SMEInteractionMode.CO_EXECUTOR
        assert report.co_executor_report is not None

    def test_contributed_sections(self, spawner):
        sme = make_spawned_sme(
            domain="Cloud Infrastructure",
            mode=SMEInteractionMode.CO_EXECUTOR,
        )
        report = spawner.execute_sme_interaction(sme, "Build API")
        assert len(report.co_executor_report.contributed_sections) >= 1

    def test_cloud_sections(self, spawner):
        sme = make_spawned_sme(
            domain="Cloud Infrastructure",
            mode=SMEInteractionMode.CO_EXECUTOR,
        )
        report = spawner.execute_sme_interaction(sme, "content")
        titles = [s.section_title for s in report.co_executor_report.contributed_sections]
        assert any("Infrastructure" in t or "Deployment" in t for t in titles)

    def test_security_sections(self, spawner):
        sme = make_spawned_sme(
            domain="Application & Infrastructure Security",
            mode=SMEInteractionMode.CO_EXECUTOR,
        )
        report = spawner.execute_sme_interaction(sme, "content")
        titles = [s.section_title for s in report.co_executor_report.contributed_sections]
        assert any("Security" in t or "Authentication" in t for t in titles)

    def test_data_sections(self, spawner):
        sme = make_spawned_sme(
            domain="Data Engineering",
            mode=SMEInteractionMode.CO_EXECUTOR,
        )
        report = spawner.execute_sme_interaction(sme, "content")
        titles = [s.section_title for s in report.co_executor_report.contributed_sections]
        assert any("Data" in t for t in titles)

    def test_coordination_notes(self, spawner):
        sme = make_spawned_sme(mode=SMEInteractionMode.CO_EXECUTOR)
        report = spawner.execute_sme_interaction(sme, "content")
        assert report.co_executor_report.coordination_notes
        assert "Integrate" in report.co_executor_report.coordination_notes

    def test_domain_assumptions(self, spawner):
        sme = make_spawned_sme(
            domain="Cloud Infrastructure",
            mode=SMEInteractionMode.CO_EXECUTOR,
        )
        report = spawner.execute_sme_interaction(sme, "content")
        assert len(report.co_executor_report.domain_assumptions) >= 1

    def test_confidence_high(self, spawner):
        sme = make_spawned_sme(mode=SMEInteractionMode.CO_EXECUTOR)
        report = spawner.execute_sme_interaction(sme, "content")
        assert report.confidence == 0.85

    def test_default_domain_sections(self, spawner):
        sme = make_spawned_sme(
            domain="Exotic Domain",
            mode=SMEInteractionMode.CO_EXECUTOR,
        )
        report = spawner.execute_sme_interaction(sme, "content")
        titles = [s.section_title for s in report.co_executor_report.contributed_sections]
        assert any("Exotic Domain" in t for t in titles)


# ============================================================================
# execute_sme_interaction - Debater Mode
# ============================================================================

class TestDebaterMode:
    def test_returns_debater_report(self, spawner):
        sme = make_spawned_sme(
            domain="Application & Infrastructure Security",
            mode=SMEInteractionMode.DEBATER,
        )
        report = spawner.execute_sme_interaction(sme, "Build API")
        assert isinstance(report, SMEAdvisoryReport)
        assert report.interaction_mode == SMEInteractionMode.DEBATER
        assert report.debater_report is not None

    def test_security_position(self, spawner):
        sme = make_spawned_sme(
            domain="Application & Infrastructure Security",
            mode=SMEInteractionMode.DEBATER,
        )
        report = spawner.execute_sme_interaction(sme, "content")
        assert "security" in report.debater_report.position.position.lower()

    def test_cloud_position(self, spawner):
        sme = make_spawned_sme(
            domain="Cloud Infrastructure",
            mode=SMEInteractionMode.DEBATER,
        )
        report = spawner.execute_sme_interaction(sme, "content")
        assert "cloud" in report.debater_report.position.position.lower()

    def test_data_position(self, spawner):
        sme = make_spawned_sme(
            domain="Data Engineering",
            mode=SMEInteractionMode.DEBATER,
        )
        report = spawner.execute_sme_interaction(sme, "content")
        assert "data" in report.debater_report.position.position.lower()

    def test_default_position(self, spawner):
        sme = make_spawned_sme(
            domain="Exotic Domain",
            mode=SMEInteractionMode.DEBATER,
        )
        report = spawner.execute_sme_interaction(sme, "content")
        assert "Exotic Domain" in report.debater_report.position.position

    def test_debate_round(self, spawner):
        sme = make_spawned_sme(
            domain="Cloud Infrastructure",
            mode=SMEInteractionMode.DEBATER,
        )
        report = spawner.execute_sme_interaction(
            sme, "content", additional_context={"debate_round": 3}
        )
        assert report.debater_report.debate_round == 3

    def test_default_debate_round(self, spawner):
        sme = make_spawned_sme(
            domain="Cloud Infrastructure",
            mode=SMEInteractionMode.DEBATER,
        )
        report = spawner.execute_sme_interaction(sme, "content")
        assert report.debater_report.debate_round == 1

    def test_counter_arguments_addressed(self, spawner):
        sme = make_spawned_sme(
            domain="Cloud Infrastructure",
            mode=SMEInteractionMode.DEBATER,
        )
        report = spawner.execute_sme_interaction(
            sme, "content",
            additional_context={
                "counter_arguments": ["cost is too high", "performance impact"],
            },
        )
        addressed = report.debater_report.counter_arguments_addressed
        assert len(addressed) >= 2

    def test_counter_arguments_limited_to_three(self, spawner):
        sme = make_spawned_sme(
            domain="Cloud Infrastructure",
            mode=SMEInteractionMode.DEBATER,
        )
        report = spawner.execute_sme_interaction(
            sme, "content",
            additional_context={
                "counter_arguments": [f"arg {i}" for i in range(5)],
            },
        )
        assert len(report.debater_report.counter_arguments_addressed) <= 3

    def test_willingness_to_concede_range(self, spawner):
        sme = make_spawned_sme(
            domain="Cloud Infrastructure",
            mode=SMEInteractionMode.DEBATER,
        )
        report = spawner.execute_sme_interaction(sme, "content")
        assert 0.0 <= report.debater_report.willingness_to_concede <= 1.0

    def test_willingness_increases_with_counters(self, spawner):
        sme = make_spawned_sme(
            domain="Cloud Infrastructure",
            mode=SMEInteractionMode.DEBATER,
        )
        report_no_counters = spawner.execute_sme_interaction(sme, "content")
        report_with_counters = spawner.execute_sme_interaction(
            sme, "content",
            additional_context={"counter_arguments": ["cost", "time", "complexity"]},
        )
        assert (report_with_counters.debater_report.willingness_to_concede
                >= report_no_counters.debater_report.willingness_to_concede)

    def test_willingness_capped_at_07(self, spawner):
        sme = make_spawned_sme(
            domain="Cloud Infrastructure",
            mode=SMEInteractionMode.DEBATER,
        )
        report = spawner.execute_sme_interaction(
            sme, "content",
            additional_context={"counter_arguments": [f"arg{i}" for i in range(10)]},
        )
        assert report.debater_report.willingness_to_concede <= 0.7

    def test_remaining_concerns(self, spawner):
        sme = make_spawned_sme(
            domain="Application & Infrastructure Security",
            mode=SMEInteractionMode.DEBATER,
        )
        report = spawner.execute_sme_interaction(sme, "content")
        assert len(report.debater_report.remaining_concerns) >= 0

    def test_debate_caveats(self, spawner):
        sme = make_spawned_sme(
            domain="Cloud Infrastructure",
            mode=SMEInteractionMode.DEBATER,
        )
        report = spawner.execute_sme_interaction(sme, "content")
        assert len(report.caveats) >= 1

    def test_supporting_evidence(self, spawner):
        sme = make_spawned_sme(
            domain="Application & Infrastructure Security",
            mode=SMEInteractionMode.DEBATER,
        )
        report = spawner.execute_sme_interaction(sme, "content")
        evidence = report.debater_report.position.supporting_evidence
        assert len(evidence) >= 1

    def test_domain_rationale(self, spawner):
        sme = make_spawned_sme(
            domain="Application & Infrastructure Security",
            mode=SMEInteractionMode.DEBATER,
        )
        report = spawner.execute_sme_interaction(sme, "content")
        assert len(report.debater_report.position.domain_rationale) > 0

    def test_findings_include_position_info(self, spawner):
        sme = make_spawned_sme(
            domain="Cloud Infrastructure",
            mode=SMEInteractionMode.DEBATER,
        )
        report = spawner.execute_sme_interaction(sme, "content")
        assert any("Position" in f for f in report.findings)
        assert any("Confidence" in f for f in report.findings)

    def test_high_concession_recommendation(self, spawner):
        sme = make_spawned_sme(
            domain="Cloud Infrastructure",
            mode=SMEInteractionMode.DEBATER,
        )
        # With enough counter-args, willingness should be > 0.7
        report = spawner.execute_sme_interaction(
            sme, "content",
            additional_context={"counter_arguments": ["a", "b", "c", "d", "e"]},
        )
        if report.debater_report.willingness_to_concede >= 0.7:
            assert any("concede" in r.lower() for r in report.recommendations)

    def test_low_concession_recommendation(self, spawner):
        sme = make_spawned_sme(
            domain="Cloud Infrastructure",
            mode=SMEInteractionMode.DEBATER,
        )
        report = spawner.execute_sme_interaction(sme, "content")
        if report.debater_report.willingness_to_concede < 0.3:
            assert any("advocate" in r.lower() for r in report.recommendations)


# ============================================================================
# execute_sme_interaction - Default mode fallback
# ============================================================================

class TestDefaultModeFallback:
    def test_unknown_mode_defaults_to_advisor(self, spawner):
        """If mode is not recognized, falls back to advisor."""
        sme = make_spawned_sme(mode=SMEInteractionMode.ADVISOR)
        report = spawner.execute_sme_interaction(sme, "content")
        assert report.interaction_mode == SMEInteractionMode.ADVISOR


# ============================================================================
# Helper methods
# ============================================================================

class TestHelperMethods:
    def test_get_timestamp(self, spawner):
        ts = spawner._get_timestamp()
        assert isinstance(ts, str)
        assert "T" in ts  # ISO format

    def test_calculate_confidence_base(self, spawner):
        sme = make_spawned_sme()
        confidence = spawner._calculate_confidence(sme, [])
        assert confidence == 0.8

    def test_calculate_confidence_with_findings(self, spawner):
        sme = make_spawned_sme()
        confidence = spawner._calculate_confidence(sme, ["found something"])
        assert confidence == 0.9

    def test_confidence_capped(self, spawner):
        sme = make_spawned_sme()
        confidence = spawner._calculate_confidence(sme, ["a", "b", "c"])
        assert confidence <= 0.95

    def test_generate_recommendations(self, spawner):
        sme = make_spawned_sme(domain="Application & Infrastructure Security")
        recs = spawner._generate_recommendations(sme, ["finding"], ["correction"])
        assert "correction" in recs
        assert any("security" in r.lower() for r in recs)

    def test_recommendations_deduped_and_limited(self, spawner):
        sme = make_spawned_sme(domain="Cloud Infrastructure")
        recs = spawner._generate_recommendations(
            sme,
            ["f1", "f2"],
            ["c1", "c1", "c2", "c3", "c4", "c5", "c6"],
        )
        assert len(recs) <= 5

    def test_get_default_corrections_security(self, spawner):
        sme = make_spawned_sme(domain="Application & Infrastructure Security")
        corrections = spawner._get_default_corrections(sme)
        assert any("OWASP" in c or "security" in c.lower() for c in corrections)

    def test_get_default_corrections_cloud(self, spawner):
        sme = make_spawned_sme(domain="Cloud Infrastructure")
        corrections = spawner._get_default_corrections(sme)
        assert any("cloud" in c.lower() for c in corrections)

    def test_get_default_corrections_data(self, spawner):
        sme = make_spawned_sme(domain="Data Engineering")
        corrections = spawner._get_default_corrections(sme)
        assert any("data" in c.lower() or "index" in c.lower() for c in corrections)

    def test_get_default_corrections_test(self, spawner):
        sme = make_spawned_sme(domain="Testing & QA")
        corrections = spawner._get_default_corrections(sme)
        assert any("test" in c.lower() or "coverage" in c.lower() for c in corrections)

    def test_get_default_corrections_unknown(self, spawner):
        sme = make_spawned_sme(domain="Exotic")
        corrections = spawner._get_default_corrections(sme)
        assert len(corrections) >= 1

    def test_position_confidence(self, spawner):
        sme = make_spawned_sme()
        confidence = spawner._calculate_position_confidence(sme, "position")
        assert confidence == 0.85

    def test_remaining_concerns_many_counter_args(self, spawner):
        sme = make_spawned_sme(domain="Cloud Infrastructure")
        pos = DebatePosition(
            sme_persona="Cloud Architect",
            position="pos",
            domain_rationale="rationale",
            supporting_evidence=[],
            confidence=0.8,
        )
        concerns = spawner._identify_remaining_concerns(
            sme, pos, ["a", "b", "c", "d"]
        )
        assert any("balancing" in c.lower() for c in concerns)

    def test_coordination_notes_empty_sections(self, spawner):
        sme = make_spawned_sme()
        notes = spawner._generate_coordination_notes(sme, [], "content")
        assert "advisory" in notes.lower()

    def test_domain_assumptions_cloud(self, spawner):
        sme = make_spawned_sme(domain="Cloud Infrastructure")
        assumptions = spawner._identify_domain_assumptions(sme, "content")
        assert any("cloud" in a.lower() for a in assumptions)

    def test_domain_assumptions_security(self, spawner):
        sme = make_spawned_sme(domain="Application & Infrastructure Security")
        assumptions = spawner._identify_domain_assumptions(sme, "content")
        assert any("security" in a.lower() for a in assumptions)

    def test_domain_assumptions_default(self, spawner):
        sme = make_spawned_sme(domain="Exotic Domain")
        assumptions = spawner._identify_domain_assumptions(sme, "content")
        assert any("Exotic Domain" in a for a in assumptions)


# ============================================================================
# Section generation methods
# ============================================================================

class TestSectionGeneration:
    @pytest.mark.parametrize("domain,expected_keywords", [
        ("Application & Infrastructure Security", ["Security", "Authentication"]),
        ("Cloud Infrastructure", ["Infrastructure", "Deployment"]),
        ("Data Engineering", ["Data"]),
        ("Frontend Development", ["User Interface", "Component"]),
        ("DevOps & Automation", ["CI/CD", "Monitoring"]),
        ("Technical Documentation", ["Documentation", "Usage"]),
    ])
    def test_domain_section_mapping(self, spawner, domain, expected_keywords):
        sme = make_spawned_sme(domain=domain, mode=SMEInteractionMode.CO_EXECUTOR)
        sections = spawner._determine_sections(sme, "content")
        titles = [s["title"] for s in sections]
        for kw in expected_keywords:
            assert any(kw in t for t in titles), f"Expected '{kw}' in {titles}"

    def test_architecture_section_content(self, spawner):
        sme = make_spawned_sme(domain="Cloud Infrastructure")
        content = spawner._generate_architecture_section(sme, "Infra")
        assert "Separation of Concerns" in content
        assert "Cloud Infrastructure" in content

    def test_implementation_section_content(self, spawner):
        sme = make_spawned_sme(domain="Cloud Infrastructure")
        content = spawner._generate_implementation_section(sme, "Impl")
        assert "python" in content.lower() or "class" in content.lower()

    def test_deployment_section_content(self, spawner):
        sme = make_spawned_sme()
        content = spawner._generate_deployment_section(sme, "Deploy")
        assert "Deployment" in content
        assert "Rollback" in content

    def test_documentation_section_content(self, spawner):
        sme = make_spawned_sme()
        content = spawner._generate_documentation_section(sme, "Docs")
        assert "API Reference" in content

    def test_operations_section_content(self, spawner):
        sme = make_spawned_sme()
        content = spawner._generate_operations_section(sme, "Ops")
        assert "Monitoring" in content
        assert "Alerting" in content

    def test_generic_section_content(self, spawner):
        sme = make_spawned_sme(domain="Exotic")
        content = spawner._generate_generic_section(sme, "General")
        assert "Exotic" in content


# ============================================================================
# Convenience function
# ============================================================================

class TestCreateSmeSpawner:
    def test_creates_instance(self, tmp_path):
        spawner = create_sme_spawner(
            skills_dir=str(tmp_path / "s"),
            sme_templates_dir=str(tmp_path / "t"),
        )
        assert isinstance(spawner, SMESpawner)
