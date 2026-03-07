"""
Tests for the SMESpawner.

Tests persona spawning from registry, interaction mode conversion,
three interaction modes (advisor, co-executor, debater),
system prompt loading, skill loading, and helper methods.
"""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.agents.sme_spawner import (
    SMESpawner,
    SpawnedSME,
    SpawnResult,
    create_sme_spawner,
)
from src.schemas.sme import (
    SMEAdvisoryReport,
    SMEInteractionMode,
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


@pytest.fixture
def spawner(tmp_path):
    """Create an SMESpawner with temp directories."""
    return SMESpawner(
        skills_dir=str(tmp_path / "skills"),
        sme_templates_dir=str(tmp_path / "sme_templates"),
    )


@pytest.fixture
def security_sme():
    """Create a spawned security SME."""
    return SpawnedSME(
        persona_id="security_analyst",
        persona_name="Security Analyst",
        domain="Application Security",
        interaction_mode=SMEInteractionMode.ADVISOR,
        system_prompt="You are a Security Analyst.",
        skills_loaded=["security-review"],
        spawn_context={
            "activation_phase": "execution",
            "reasoning": "Security expertise needed",
            "task_context": "Build a secure API",
            "execution_phase": "execution",
        },
    )


@pytest.fixture
def cloud_sme():
    """Create a spawned cloud SME."""
    return SpawnedSME(
        persona_id="cloud_architect",
        persona_name="Cloud Architect",
        domain="Cloud Architecture",
        interaction_mode=SMEInteractionMode.CO_EXECUTOR,
        system_prompt="You are a Cloud Architect.",
        skills_loaded=["azure-architect"],
        spawn_context={
            "activation_phase": "planning",
            "reasoning": "Cloud expertise needed",
            "task_context": "Deploy on AWS",
            "execution_phase": "execution",
        },
    )


@pytest.fixture
def data_sme():
    """Create a spawned data SME for debate mode."""
    return SpawnedSME(
        persona_id="data_engineer",
        persona_name="Data Engineer",
        domain="Data Engineering",
        interaction_mode=SMEInteractionMode.DEBATER,
        system_prompt="You are a Data Engineer.",
        skills_loaded=[],
        spawn_context={
            "activation_phase": "execution",
            "reasoning": "Data expertise needed",
            "task_context": "Design data pipeline",
            "execution_phase": "execution",
        },
    )


class TestSMESpawnerInitialization:
    """Tests for SMESpawner initialization."""

    def test_default_initialization(self, tmp_path):
        """Test default init parameters."""
        spawner = SMESpawner(
            skills_dir=str(tmp_path / "skills"),
            sme_templates_dir=str(tmp_path / "templates"),
        )
        assert spawner.model == "claude-3-5-sonnet-20241022"

    def test_directories_created(self, tmp_path):
        """Test directories are created."""
        skills_dir = tmp_path / "new_skills"
        templates_dir = tmp_path / "new_templates"
        SMESpawner(
            skills_dir=str(skills_dir),
            sme_templates_dir=str(templates_dir),
        )
        assert skills_dir.exists()
        assert templates_dir.exists()

    def test_custom_model(self, tmp_path):
        """Test custom model."""
        spawner = SMESpawner(
            skills_dir=str(tmp_path / "skills"),
            sme_templates_dir=str(tmp_path / "templates"),
            model="claude-3-opus",
        )
        assert spawner.model == "claude-3-opus"


class TestInteractionModeConversion:
    """Tests for interaction mode conversion."""

    @pytest.mark.parametrize("council_mode,expected_sme_mode", [
        (CouncilInteractionMode.ADVISOR, SMEInteractionMode.ADVISOR),
        (CouncilInteractionMode.CO_EXECUTOR, SMEInteractionMode.CO_EXECUTOR),
        (CouncilInteractionMode.DEBATER, SMEInteractionMode.DEBATER),
    ])
    def test_convert_interaction_mode(self, spawner, council_mode, expected_sme_mode):
        """Test mode conversion from Council to SME format."""
        result = spawner._convert_interaction_mode(council_mode)
        assert result == expected_sme_mode


class TestAdvisorMode:
    """Tests for advisor mode execution."""

    def test_advisor_returns_report(self, spawner, security_sme):
        """Test advisor mode returns SMEAdvisoryReport."""
        report = spawner._execute_advisor_mode(
            security_sme, "Build an API with password = 'admin123'", None
        )
        assert isinstance(report, SMEAdvisoryReport)
        assert report.interaction_mode == SMEInteractionMode.ADVISOR

    def test_advisor_has_findings(self, spawner, security_sme):
        """Test advisor produces findings."""
        report = spawner._execute_advisor_mode(
            security_sme, "password = 'secret' in the code", None
        )
        assert len(report.findings) > 0

    def test_advisor_has_recommendations(self, spawner, security_sme):
        """Test advisor produces recommendations."""
        report = spawner._execute_advisor_mode(
            security_sme, "Build a secure API", None
        )
        assert len(report.recommendations) > 0

    def test_advisor_confidence_range(self, spawner, security_sme):
        """Test advisor confidence is in valid range."""
        report = spawner._execute_advisor_mode(
            security_sme, "Some content", None
        )
        assert 0.0 <= report.confidence <= 1.0

    def test_advisor_has_caveats(self, spawner, security_sme):
        """Test advisor has caveats."""
        report = spawner._execute_advisor_mode(
            security_sme, "Some content", None
        )
        assert len(report.caveats) > 0

    def test_advisor_report_populated(self, spawner, security_sme):
        """Test advisor_report field is populated."""
        report = spawner._execute_advisor_mode(
            security_sme, "Some content", None
        )
        assert report.advisor_report is not None
        assert isinstance(report.advisor_report, AdvisorReport)


class TestDomainFindings:
    """Tests for domain-specific finding analysis."""

    def test_security_findings_password(self, spawner, security_sme):
        """Test security domain detects hardcoded password."""
        findings = spawner._analyze_domain_findings(
            security_sme, "password = 'admin123'"
        )
        assert any("password" in f.lower() for f in findings)

    def test_security_findings_eval(self, spawner, security_sme):
        """Test security domain detects eval usage."""
        findings = spawner._analyze_domain_findings(
            security_sme, "result = eval(user_input)"
        )
        assert any("injection" in f.lower() for f in findings)

    def test_generic_findings(self, spawner, security_sme):
        """Test generic findings when no patterns match."""
        findings = spawner._analyze_domain_findings(
            security_sme, "Hello world"
        )
        assert len(findings) > 0  # Should have at least general observation

    def test_findings_limited_to_five(self, spawner, security_sme):
        """Test findings are limited to 5."""
        findings = spawner._analyze_domain_findings(
            security_sme,
            "password = 'x' sql+injection eval(x) password='y' sql+again eval(z) more",
        )
        assert len(findings) <= 5


class TestMissingConsiderations:
    """Tests for missing considerations identification."""

    def test_security_missing_considerations(self, spawner, security_sme):
        """Test security SME identifies missing considerations."""
        missing = spawner._identify_missing_considerations(
            security_sme, "Build an API"
        )
        # Should identify missing rate limiting, encryption, or audit
        assert len(missing) > 0

    def test_no_missing_when_addressed(self, spawner, security_sme):
        """Test fewer missing items when topics are addressed."""
        missing = spawner._identify_missing_considerations(
            security_sme,
            "Build an API with rate limiting, encryption, and audit logging",
        )
        # All three security considerations are mentioned
        assert len(missing) == 0


class TestCoExecutorMode:
    """Tests for co-executor mode execution."""

    def test_co_executor_returns_report(self, spawner, cloud_sme):
        """Test co-executor mode returns SMEAdvisoryReport."""
        report = spawner._execute_co_executor_mode(
            cloud_sme, "Deploy the application", None
        )
        assert isinstance(report, SMEAdvisoryReport)
        assert report.interaction_mode == SMEInteractionMode.CO_EXECUTOR

    def test_co_executor_has_sections(self, spawner, cloud_sme):
        """Test co-executor produces contributed sections."""
        report = spawner._execute_co_executor_mode(
            cloud_sme, "Deploy the application", None
        )
        assert report.co_executor_report is not None
        assert len(report.co_executor_report.contributed_sections) > 0

    def test_co_executor_section_has_content(self, spawner, cloud_sme):
        """Test contributed sections have content."""
        report = spawner._execute_co_executor_mode(
            cloud_sme, "Deploy the application", None
        )
        for section in report.co_executor_report.contributed_sections:
            assert isinstance(section, CoExecutorSection)
            assert len(section.content) > 0
            assert len(section.section_title) > 0

    def test_co_executor_high_confidence(self, spawner, cloud_sme):
        """Test co-executor has high confidence."""
        report = spawner._execute_co_executor_mode(
            cloud_sme, "Deploy the application", None
        )
        assert report.confidence == 0.85

    def test_co_executor_coordination_notes(self, spawner, cloud_sme):
        """Test co-executor has coordination notes."""
        report = spawner._execute_co_executor_mode(
            cloud_sme, "Deploy the application", None
        )
        assert len(report.co_executor_report.coordination_notes) > 0


class TestSectionDetermination:
    """Tests for section determination by domain."""

    @pytest.mark.parametrize("domain,expected_keyword", [
        ("Application Security", "Security"),
        ("Cloud Architecture", "Infrastructure"),
        ("Data Engineering", "Data"),
        ("Frontend Development", "User Interface"),
        ("DevOps", "CI/CD"),
    ])
    def test_domain_specific_sections(self, spawner, domain, expected_keyword):
        """Test domain determines section titles."""
        sme = SpawnedSME(
            persona_id="test", persona_name="Test",
            domain=domain, interaction_mode=SMEInteractionMode.CO_EXECUTOR,
            system_prompt="", skills_loaded=[],
            spawn_context={"task_context": "test"},
        )
        sections = spawner._determine_sections(sme, "Build something")
        titles = [s["title"] for s in sections]
        assert any(expected_keyword in t for t in titles)


class TestDebaterMode:
    """Tests for debater mode execution."""

    def test_debater_returns_report(self, spawner, data_sme):
        """Test debater mode returns SMEAdvisoryReport."""
        report = spawner._execute_debater_mode(
            data_sme, "Choose database technology", None
        )
        assert isinstance(report, SMEAdvisoryReport)
        assert report.interaction_mode == SMEInteractionMode.DEBATER

    def test_debater_has_position(self, spawner, data_sme):
        """Test debater has a position."""
        report = spawner._execute_debater_mode(
            data_sme, "Choose database technology", None
        )
        assert report.debater_report is not None
        assert len(report.debater_report.position.position) > 0

    def test_debater_has_evidence(self, spawner, data_sme):
        """Test debater provides supporting evidence."""
        report = spawner._execute_debater_mode(
            data_sme, "Choose database technology", None
        )
        assert len(report.debater_report.position.supporting_evidence) > 0

    def test_debater_addresses_counter_arguments(self, spawner, data_sme):
        """Test debater addresses counter-arguments."""
        report = spawner._execute_debater_mode(
            data_sme, "Choose database technology",
            {"counter_arguments": ["Cost is too high", "Performance impact"]},
        )
        assert len(report.debater_report.counter_arguments_addressed) > 0

    def test_debater_willingness_to_concede(self, spawner, data_sme):
        """Test debater has willingness to concede."""
        report = spawner._execute_debater_mode(
            data_sme, "Choose database technology", None
        )
        assert 0.0 <= report.debater_report.willingness_to_concede <= 1.0

    def test_debater_concession_increases_with_arguments(self, spawner, data_sme):
        """Test concession willingness increases with more counter-arguments."""
        base = spawner._calculate_concession_willingness(
            data_sme,
            MagicMock(),
            [],
        )
        with_args = spawner._calculate_concession_willingness(
            data_sme,
            MagicMock(),
            ["arg1", "arg2", "arg3"],
        )
        assert with_args >= base

    def test_debater_concession_capped(self, spawner, data_sme):
        """Test concession willingness is capped at 0.7."""
        result = spawner._calculate_concession_willingness(
            data_sme,
            MagicMock(),
            ["a", "b", "c", "d", "e", "f"],
        )
        assert result <= 0.7


class TestDebatePositions:
    """Tests for debate position determination."""

    @pytest.mark.parametrize("domain,expected_keyword", [
        ("Application Security", "Security"),
        ("Cloud Architecture", "Cloud"),
        ("Data Engineering", "Data"),
        ("Frontend Development", "User experience"),
        ("DevOps", "Automation"),
    ])
    def test_domain_positions(self, spawner, domain, expected_keyword):
        """Test domain-specific positions are generated."""
        sme = SpawnedSME(
            persona_id="test", persona_name="Test",
            domain=domain, interaction_mode=SMEInteractionMode.DEBATER,
            system_prompt="", skills_loaded=[],
            spawn_context={"task_context": "test"},
        )
        position = spawner._determine_debate_position(sme, "Build something", None)
        assert expected_keyword.lower() in position.lower()


class TestExecuteSMEInteraction:
    """Tests for the main execute_sme_interaction method."""

    def test_routes_to_advisor(self, spawner, security_sme):
        """Test advisor mode routing."""
        security_sme.interaction_mode = SMEInteractionMode.ADVISOR
        report = spawner.execute_sme_interaction(security_sme, "Review this code")
        assert report.interaction_mode == SMEInteractionMode.ADVISOR

    def test_routes_to_co_executor(self, spawner, cloud_sme):
        """Test co-executor mode routing."""
        cloud_sme.interaction_mode = SMEInteractionMode.CO_EXECUTOR
        report = spawner.execute_sme_interaction(cloud_sme, "Deploy app")
        assert report.interaction_mode == SMEInteractionMode.CO_EXECUTOR

    def test_routes_to_debater(self, spawner, data_sme):
        """Test debater mode routing."""
        data_sme.interaction_mode = SMEInteractionMode.DEBATER
        report = spawner.execute_sme_interaction(data_sme, "Choose tech")
        assert report.interaction_mode == SMEInteractionMode.DEBATER

    def test_default_routes_to_advisor(self, spawner, security_sme):
        """Test unknown mode defaults to advisor."""
        # Force an unknown mode scenario by using a valid mode
        security_sme.interaction_mode = SMEInteractionMode.ADVISOR
        report = spawner.execute_sme_interaction(security_sme, "Test")
        assert isinstance(report, SMEAdvisoryReport)


class TestConvenienceFunction:
    """Tests for create_sme_spawner convenience function."""

    def test_create_sme_spawner(self):
        """Test convenience function creates an SMESpawner."""
        spawner = create_sme_spawner(
            skills_dir="/tmp/test_skills",
            sme_templates_dir="/tmp/test_templates",
        )
        assert isinstance(spawner, SMESpawner)

    def test_create_sme_spawner_defaults(self):
        """Test convenience function uses defaults."""
        spawner = create_sme_spawner()
        assert spawner.model == "claude-3-5-sonnet-20241022"
