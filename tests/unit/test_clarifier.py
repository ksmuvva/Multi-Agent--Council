"""
Tests for the ClarifierAgent.

Tests question formulation, priority ranking, workflow determination,
and quality assessment with defaults.
"""

import pytest
from unittest.mock import patch, mock_open

from src.agents.clarifier import ClarifierAgent, create_clarifier
from src.schemas.clarifier import (
    ClarificationRequest,
    ClarificationQuestion,
    QuestionPriority,
)
from src.schemas.analyst import (
    TaskIntelligenceReport,
    SubTask,
    MissingInfo,
    SeverityLevel,
    ModalityType,
)


@pytest.fixture
def clarifier():
    """Create a ClarifierAgent with no system prompt file."""
    return ClarifierAgent(system_prompt_path="nonexistent.md")


@pytest.fixture
def report_no_missing():
    """Report with no missing info."""
    return TaskIntelligenceReport(
        literal_request="Hello",
        inferred_intent="Greeting",
        sub_tasks=[SubTask(description="Greet", dependencies=[])],
        missing_info=[],
        assumptions=[],
        modality=ModalityType.TEXT,
        recommended_approach="Respond",
        escalation_needed=False,
    )


@pytest.fixture
def report_critical_missing():
    """Report with critical missing info."""
    return TaskIntelligenceReport(
        literal_request="Build an API",
        inferred_intent="Create API",
        sub_tasks=[SubTask(description="Build", dependencies=[])],
        missing_info=[
            MissingInfo(
                requirement="Authentication method",
                severity=SeverityLevel.CRITICAL,
                impact="Security architecture depends on auth",
                default_assumption="JWT-based authentication",
            ),
            MissingInfo(
                requirement="Database technology",
                severity=SeverityLevel.CRITICAL,
                impact="Schema design depends on DB choice",
                default_assumption="PostgreSQL",
            ),
        ],
        assumptions=["JWT auth", "PostgreSQL"],
        modality=ModalityType.CODE,
        recommended_approach="Clarify first",
        escalation_needed=False,
    )


@pytest.fixture
def report_mixed_severity():
    """Report with mixed severity missing info."""
    return TaskIntelligenceReport(
        literal_request="Build a web app",
        inferred_intent="Create web application",
        sub_tasks=[SubTask(description="Build", dependencies=[])],
        missing_info=[
            MissingInfo(
                requirement="Authentication method",
                severity=SeverityLevel.CRITICAL,
                impact="Security depends on auth",
                default_assumption="JWT",
            ),
            MissingInfo(
                requirement="Deployment target",
                severity=SeverityLevel.IMPORTANT,
                impact="Deployment config varies",
                default_assumption="Docker",
            ),
            MissingInfo(
                requirement="Testing requirements",
                severity=SeverityLevel.NICE_TO_HAVE,
                impact="Could add test coverage",
                default_assumption="Unit tests",
            ),
        ],
        assumptions=[],
        modality=ModalityType.CODE,
        recommended_approach="Clarify",
        escalation_needed=False,
    )


class TestClarifierInitialization:
    """Tests for ClarifierAgent initialization."""

    def test_default_initialization(self):
        """Test default init parameters."""
        agent = ClarifierAgent(system_prompt_path="nonexistent.md")
        assert agent.model == "claude-3-5-sonnet-20241022"
        assert agent.max_turns == 30

    def test_priority_weights_set(self):
        """Test priority weights are initialized."""
        agent = ClarifierAgent(system_prompt_path="nonexistent.md")
        assert "quality_impact" in agent.priority_weights
        assert "reversibility" in agent.priority_weights

    def test_system_prompt_fallback(self):
        """Test fallback prompt when file not found."""
        agent = ClarifierAgent(system_prompt_path="nonexistent.md")
        assert "Clarifier" in agent.system_prompt

    def test_system_prompt_from_file(self):
        """Test loading system prompt from file."""
        with patch("builtins.open", mock_open(read_data="Custom clarifier")):
            agent = ClarifierAgent(system_prompt_path="exists.md")
            assert agent.system_prompt == "Custom clarifier"

    def test_custom_model(self):
        """Test custom model setting."""
        agent = ClarifierAgent(
            system_prompt_path="nonexistent.md",
            model="claude-3-opus",
        )
        assert agent.model == "claude-3-opus"


class TestFormulateQuestions:
    """Tests for question formulation."""

    def test_no_questions_when_no_missing_info(self, clarifier, report_no_missing):
        """Test empty request when no missing info."""
        result = clarifier.formulate_questions(report_no_missing)
        assert isinstance(result, ClarificationRequest)
        assert result.total_questions == 0
        assert result.can_proceed_with_defaults is True

    def test_generates_questions_for_missing_info(self, clarifier, report_critical_missing):
        """Test questions generated for missing info."""
        result = clarifier.formulate_questions(report_critical_missing)
        assert result.total_questions > 0
        assert len(result.questions) > 0

    def test_respects_max_questions(self, clarifier, report_mixed_severity):
        """Test max_questions limit is respected."""
        result = clarifier.formulate_questions(report_mixed_severity, max_questions=2)
        assert result.total_questions <= 2

    def test_questions_have_default_answers(self, clarifier, report_critical_missing):
        """Test all questions have default answers."""
        result = clarifier.formulate_questions(report_critical_missing)
        for question in result.questions:
            assert question.default_answer is not None

    def test_questions_have_impact_assessment(self, clarifier, report_critical_missing):
        """Test questions have impact assessment."""
        result = clarifier.formulate_questions(report_critical_missing)
        for question in result.questions:
            assert question.impact_if_unanswered is not None
            assert question.impact_if_unanswered.quality_impact is not None


class TestQuestionRanking:
    """Tests for question ranking."""

    def test_critical_questions_first(self, clarifier, report_mixed_severity):
        """Test critical questions appear first."""
        result = clarifier.formulate_questions(report_mixed_severity)
        if len(result.questions) >= 2:
            first_priority = result.questions[0].priority
            assert first_priority == QuestionPriority.CRITICAL

    def test_severity_to_priority_mapping(self, clarifier):
        """Test severity-to-priority mapping."""
        assert clarifier._severity_to_priority(SeverityLevel.CRITICAL) == QuestionPriority.CRITICAL
        assert clarifier._severity_to_priority(SeverityLevel.IMPORTANT) == QuestionPriority.HIGH
        assert clarifier._severity_to_priority(SeverityLevel.NICE_TO_HAVE) == QuestionPriority.MEDIUM

    def test_ranking_preserves_all_questions(self, clarifier, report_mixed_severity):
        """Test ranking doesn't lose questions."""
        result = clarifier.formulate_questions(report_mixed_severity, max_questions=10)
        assert result.total_questions == 3  # Three missing info items

    def test_answer_options_for_auth(self, clarifier):
        """Test answer options are provided for authentication."""
        options = clarifier._get_answer_options("Authentication method")
        assert options is not None
        assert "JWT" in options

    def test_no_answer_options_for_unknown(self, clarifier):
        """Test no answer options for unknown requirements."""
        options = clarifier._get_answer_options("random requirement xyz")
        assert options is None


class TestWorkflowDetermination:
    """Tests for workflow and quality assessment."""

    def test_critical_workflow_waits(self, clarifier, report_critical_missing):
        """Test workflow waits for critical questions."""
        result = clarifier.formulate_questions(report_critical_missing)
        assert "wait" in result.recommended_workflow.lower() or "critical" in result.recommended_workflow.lower()

    def test_can_proceed_with_defaults_no_critical(self, clarifier):
        """Test can proceed when no critical questions."""
        report = TaskIntelligenceReport(
            literal_request="Build something",
            inferred_intent="Build",
            sub_tasks=[SubTask(description="Build", dependencies=[])],
            missing_info=[
                MissingInfo(
                    requirement="Testing requirements",
                    severity=SeverityLevel.NICE_TO_HAVE,
                    impact="Could add tests",
                    default_assumption="Unit tests",
                ),
            ],
            assumptions=[],
            modality=ModalityType.TEXT,
            recommended_approach="Build",
            escalation_needed=False,
        )
        result = clarifier.formulate_questions(report)
        assert result.can_proceed_with_defaults is True

    def test_quality_assessment_with_critical(self, clarifier, report_critical_missing):
        """Test quality assessment mentions low quality for critical missing."""
        result = clarifier.formulate_questions(report_critical_missing)
        assert "low" in result.expected_quality_with_defaults.lower()

    def test_quality_assessment_no_missing(self, clarifier, report_no_missing):
        """Test quality assessment is high when no missing info."""
        result = clarifier.formulate_questions(report_no_missing)
        assert "high" in result.expected_quality_with_defaults.lower()


class TestConvenienceFunction:
    """Tests for create_clarifier convenience function."""

    def test_create_clarifier(self):
        """Test convenience function creates a ClarifierAgent."""
        agent = create_clarifier(system_prompt_path="nonexistent.md")
        assert isinstance(agent, ClarifierAgent)
