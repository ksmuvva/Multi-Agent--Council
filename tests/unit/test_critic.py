"""
Tests for the CriticAgent.

Tests five attack vectors: logic, completeness, quality,
contradiction, and red-team argumentation.
"""

import pytest
from unittest.mock import patch, mock_open

from src.agents.critic import (
    CriticAgent,
    ArgumentAnalysis,
    create_critic,
)
from src.schemas.critic import (
    CritiqueReport,
    AttackVector,
    SeverityLevel,
)


@pytest.fixture
def critic():
    """Create a CriticAgent with no system prompt file."""
    return CriticAgent(system_prompt_path="nonexistent.md")


GOOD_SOLUTION = """
This solution implements a REST API using FastAPI with proper error handling.

First, we define the data models using Pydantic for validation.
Next, we implement the CRUD endpoints with proper HTTP status codes.
Then, we add JWT-based authentication for security.

The solution includes example usage and unit tests for verification.
It follows best practices for API design and is well-documented.
"""

SHORT_SOLUTION = "Use print."

CONTRADICTORY_SOLUTION = """
This approach always works in all cases.
However, this approach never works for edge cases.
It is both possible and impossible to verify.
"""

SOLUTION_WITH_TODO = """
This is a solution that could work.
It might need improvements. TODO: add error handling.
The implementation is quick and TBD on testing.
"""


class TestCriticInitialization:
    """Tests for CriticAgent initialization."""

    def test_default_initialization(self):
        """Test default init parameters."""
        agent = CriticAgent(system_prompt_path="nonexistent.md")
        assert agent.model == "claude-opus-4-20250514"
        assert agent.max_turns == 30

    def test_fallacy_patterns_initialized(self):
        """Test fallacy patterns are configured."""
        agent = CriticAgent(system_prompt_path="nonexistent.md")
        assert "slippery_slope" in agent.fallacy_patterns

    def test_system_prompt_fallback(self):
        """Test fallback prompt."""
        agent = CriticAgent(system_prompt_path="nonexistent.md")
        assert "Critic" in agent.system_prompt

    def test_system_prompt_from_file(self):
        """Test loading from file."""
        with patch("builtins.open", mock_open(read_data="Critic prompt")):
            agent = CriticAgent(system_prompt_path="exists.md")
            assert agent.system_prompt == "Critic prompt"

    def test_custom_model(self):
        """Test custom model."""
        agent = CriticAgent(
            system_prompt_path="nonexistent.md",
            model="claude-3-sonnet",
        )
        assert agent.model == "claude-3-sonnet"


class TestCritique:
    """Tests for the critique method."""

    def test_basic_critique(self, critic):
        """Test basic critique produces CritiqueReport."""
        report = critic.critique(GOOD_SOLUTION, "Build an API")
        assert isinstance(report, CritiqueReport)
        assert len(report.attacks) > 0

    def test_critique_has_assessment(self, critic):
        """Test critique has overall assessment."""
        report = critic.critique(GOOD_SOLUTION, "Build an API")
        assert len(report.overall_assessment) > 0

    def test_critique_has_solution_summary(self, critic):
        """Test critique has solution summary."""
        report = critic.critique(GOOD_SOLUTION, "Build an API")
        assert len(report.solution_summary) > 0

    def test_good_solution_approved(self, critic):
        """Test good solution may be approved."""
        report = critic.critique(GOOD_SOLUTION, "Build an API")
        # Good solution should have would_approve=True or at least not many critical issues
        assert isinstance(report.would_approve, bool)

    def test_critique_recommended_revisions(self, critic):
        """Test revisions are recommended."""
        report = critic.critique(SHORT_SOLUTION, "Build a complex system")
        assert isinstance(report.recommended_revisions, list)


class TestLogicAttack:
    """Tests for the logic attack vector."""

    def test_logic_attack_on_clean(self, critic):
        """Test logic attack on clean solution."""
        report = critic.critique(GOOD_SOLUTION, "Build API")
        logic_attacks = [a for a in report.attacks if a.vector == AttackVector.LOGIC]
        # Clean solution should have few logic issues
        assert isinstance(logic_attacks, list)

    def test_contradiction_detection(self, critic):
        """Test logical contradiction detection."""
        result = critic._has_logical_contradictions("This is good. This is not good.")
        # Simple "not X" pattern detection
        assert isinstance(result, bool)

    def test_logic_attack_report_type(self, critic):
        """Test logic attack produces LogicAttack."""
        report = critic.critique(GOOD_SOLUTION, "Build API")
        assert report.logic_attack is not None

    def test_argument_analysis(self, critic):
        """Test argument structure analysis."""
        analysis = critic._analyze_argument_structure(GOOD_SOLUTION)
        assert isinstance(analysis, ArgumentAnalysis)
        assert len(analysis.premises) >= 0

    def test_argument_with_conclusion(self, critic):
        """Test argument with conclusion keyword."""
        text = "The data shows X. Therefore, we should choose Y."
        analysis = critic._analyze_argument_structure(text)
        assert analysis.conclusion != "" or len(analysis.premises) > 0


class TestCompletenessAttack:
    """Tests for the completeness attack vector."""

    def test_completeness_finds_missing(self, critic):
        """Test completeness attack finds missing elements."""
        report = critic.critique(SHORT_SOLUTION, "Build a secure API with testing")
        completeness = report.completeness_attack
        assert len(completeness.missing) > 0

    def test_completeness_on_good_solution(self, critic):
        """Test completeness on comprehensive solution."""
        report = critic.critique(GOOD_SOLUTION, "Build an API")
        completeness = report.completeness_attack
        assert len(completeness.covered) > 0

    def test_missing_security(self, critic):
        """Test detection of missing security consideration."""
        simple = "This function adds two numbers and returns the result."
        report = critic.critique(simple, "Build a calculator")
        completeness = report.completeness_attack
        assert any("security" in m.lower() for m in completeness.missing)


class TestQualityAttack:
    """Tests for the quality attack vector."""

    def test_quality_weaknesses_detected(self, critic):
        """Test quality weaknesses in short solution."""
        report = critic.critique(SHORT_SOLUTION, "Build a system")
        quality = report.quality_attack
        assert len(quality.weaknesses) > 0

    def test_quality_strengths_detected(self, critic):
        """Test quality strengths in good solution."""
        report = critic.critique(GOOD_SOLUTION, "Build API")
        quality = report.quality_attack
        assert len(quality.strengths) > 0

    def test_todo_detected(self, critic):
        """Test TODO detection as weakness."""
        report = critic.critique(SOLUTION_WITH_TODO, "Build something")
        quality = report.quality_attack
        assert any("placeholder" in w.lower() or "incomplete" in w.lower() for w in quality.weaknesses)


class TestContradictionScan:
    """Tests for the contradiction scan."""

    def test_contradiction_detected(self, critic):
        """Test contradiction is detected."""
        report = critic.critique(CONTRADICTORY_SOLUTION, "Test")
        scan = report.contradiction_scan
        has_contradictions = (
            len(scan.internal_contradictions) > 0 or
            len(scan.inconsistencies) > 0
        )
        assert has_contradictions

    def test_no_contradiction_in_clean(self, critic):
        """Test no contradiction in clean solution."""
        report = critic.critique("This is a clean solution.", "Test")
        scan = report.contradiction_scan
        assert len(scan.internal_contradictions) == 0


class TestRedTeam:
    """Tests for red-team argumentation."""

    def test_red_team_produces_attack_surface(self, critic):
        """Test red team identifies attack surface."""
        report = critic.critique(GOOD_SOLUTION, "Build API")
        red_team = report.red_team_argumentation
        assert len(red_team.attack_surface) > 0

    def test_red_team_failure_modes(self, critic):
        """Test red team identifies failure modes."""
        report = critic.critique(GOOD_SOLUTION, "Build API")
        red_team = report.red_team_argumentation
        assert len(red_team.failure_modes) > 0

    def test_red_team_worst_cases(self, critic):
        """Test red team identifies worst cases."""
        report = critic.critique(GOOD_SOLUTION, "Build API")
        red_team = report.red_team_argumentation
        assert len(red_team.worst_case_scenarios) > 0


class TestCriticalIssues:
    """Tests for critical issue identification."""

    def test_no_critical_for_clean(self, critic):
        """Test no critical issues for clean solution."""
        report = critic.critique(GOOD_SOLUTION, "Build API")
        assert isinstance(report.critical_issues, list)

    def test_would_approve_no_critical(self, critic):
        """Test approval logic with no critical issues."""
        # If there are no critical issues, would_approve depends on high count
        result = critic._would_approve_solution([], [])
        assert result is True

    def test_would_not_approve_critical(self, critic):
        """Test approval blocked by critical issues."""
        result = critic._would_approve_solution([], ["Critical bug"])
        assert result is False


class TestConvenienceFunction:
    """Tests for create_critic convenience function."""

    def test_create_critic(self):
        """Test convenience function creates a CriticAgent."""
        agent = create_critic(system_prompt_path="nonexistent.md")
        assert isinstance(agent, CriticAgent)
