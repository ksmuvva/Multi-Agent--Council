"""
Exhaustive Tests for ClarifierAgent

Tests all methods, edge cases, and branch paths for the Clarifier subagent.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import patch, mock_open

from src.agents.clarifier import ClarifierAgent, create_clarifier
from src.schemas.clarifier import (
    ClarificationRequest,
    ClarificationQuestion,
    QuestionPriority,
    ImpactAssessment,
)
from src.schemas.analyst import (
    TaskIntelligenceReport,
    SubTask,
    MissingInfo,
    SeverityLevel,
    ModalityType,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_system_prompt():
    """Provide a mock system prompt for tests."""
    return "You are the Clarifier. Formulate questions for missing requirements."


@pytest.fixture
def clarifier(mock_system_prompt):
    """Create a ClarifierAgent with mocked file I/O."""
    with patch("builtins.open", mock_open(read_data=mock_system_prompt)):
        return ClarifierAgent()


@pytest.fixture
def empty_report():
    """A TaskIntelligenceReport with no missing info."""
    return TaskIntelligenceReport(
        literal_request="Create a hello world script",
        inferred_intent="Generate a simple Python script",
        sub_tasks=[
            SubTask(description="Write script", dependencies=[], estimated_complexity="low"),
        ],
        missing_info=[],
        assumptions=["Using Python"],
        modality=ModalityType.CODE,
        recommended_approach="Direct implementation",
        escalation_needed=False,
        suggested_tier=1,
        confidence=0.95,
    )


@pytest.fixture
def report_with_missing():
    """A TaskIntelligenceReport with various missing info severities."""
    return TaskIntelligenceReport(
        literal_request="Build a REST API with authentication",
        inferred_intent="Create a production REST API",
        sub_tasks=[
            SubTask(description="Design API", dependencies=[], estimated_complexity="medium"),
            SubTask(description="Implement endpoints", dependencies=["Design API"], estimated_complexity="high"),
        ],
        missing_info=[
            MissingInfo(
                requirement="authentication method",
                severity=SeverityLevel.CRITICAL,
                impact="Cannot design security layer",
                default_assumption="JWT-based authentication",
            ),
            MissingInfo(
                requirement="database technology",
                severity=SeverityLevel.IMPORTANT,
                impact="Affects schema design and queries",
                default_assumption="PostgreSQL",
            ),
            MissingInfo(
                requirement="documentation format",
                severity=SeverityLevel.NICE_TO_HAVE,
                impact="Affects output presentation",
                default_assumption="Markdown",
            ),
        ],
        assumptions=["Python/FastAPI"],
        modality=ModalityType.CODE,
        recommended_approach="Start with design",
        escalation_needed=False,
        suggested_tier=2,
        confidence=0.7,
    )


@pytest.fixture
def report_critical_only():
    """A TaskIntelligenceReport with only critical missing info."""
    return TaskIntelligenceReport(
        literal_request="Deploy a kubernetes cluster",
        inferred_intent="Set up k8s infrastructure",
        sub_tasks=[],
        missing_info=[
            MissingInfo(
                requirement="deployment target",
                severity=SeverityLevel.CRITICAL,
                impact="Cannot determine infrastructure",
                default_assumption="Docker",
            ),
            MissingInfo(
                requirement="programming language",
                severity=SeverityLevel.CRITICAL,
                impact="Cannot write code",
                default_assumption="Python",
            ),
        ],
        assumptions=[],
        modality=ModalityType.CODE,
        recommended_approach="Design then implement",
        escalation_needed=False,
        suggested_tier=3,
        confidence=0.5,
    )


@pytest.fixture
def report_many_missing():
    """A TaskIntelligenceReport with more than 5 missing info items."""
    return TaskIntelligenceReport(
        literal_request="Build a full-stack app",
        inferred_intent="Create a complete application",
        sub_tasks=[],
        missing_info=[
            MissingInfo(requirement="authentication method", severity=SeverityLevel.CRITICAL, impact="security"),
            MissingInfo(requirement="database technology", severity=SeverityLevel.CRITICAL, impact="storage"),
            MissingInfo(requirement="deployment target", severity=SeverityLevel.IMPORTANT, impact="hosting"),
            MissingInfo(requirement="programming language", severity=SeverityLevel.IMPORTANT, impact="implementation"),
            MissingInfo(requirement="testing framework", severity=SeverityLevel.IMPORTANT, impact="quality"),
            MissingInfo(requirement="documentation format", severity=SeverityLevel.NICE_TO_HAVE, impact="docs"),
            MissingInfo(requirement="color scheme", severity=SeverityLevel.NICE_TO_HAVE, impact="aesthetics"),
        ],
        assumptions=[],
        modality=ModalityType.CODE,
        recommended_approach="Iterative",
        escalation_needed=False,
        suggested_tier=2,
        confidence=0.6,
    )


# ============================================================================
# __init__ Tests
# ============================================================================

class TestClarifierInit:
    """Tests for ClarifierAgent.__init__."""

    def test_default_params(self):
        """Test initialization with default parameters."""
        with patch("builtins.open", mock_open(read_data="prompt content")):
            agent = ClarifierAgent()
        assert agent.system_prompt_path == "config/agents/clarifier/CLAUDE.md"
        assert agent.model == "claude-sonnet-4-20250514"
        assert agent.max_turns == 30
        assert agent.system_prompt == "prompt content"

    def test_custom_params(self):
        """Test initialization with custom parameters."""
        with patch("builtins.open", mock_open(read_data="custom prompt")):
            agent = ClarifierAgent(
                system_prompt_path="custom/clarifier.md",
                model="claude-3-opus-20240229",
                max_turns=15,
            )
        assert agent.system_prompt_path == "custom/clarifier.md"
        assert agent.model == "claude-3-opus-20240229"
        assert agent.max_turns == 15
        assert agent.system_prompt == "custom prompt"

    def test_system_prompt_file_not_found(self):
        """Test fallback when system prompt file does not exist."""
        with patch("builtins.open", side_effect=FileNotFoundError):
            agent = ClarifierAgent()
        assert "Clarifier" in agent.system_prompt
        assert "questions" in agent.system_prompt

    def test_priority_weights_populated(self):
        """Test that priority_weights dict is populated."""
        with patch("builtins.open", mock_open(read_data="")):
            agent = ClarifierAgent()
        assert "quality_impact" in agent.priority_weights
        assert "reversibility" in agent.priority_weights
        assert "user_burden" in agent.priority_weights
        assert agent.priority_weights["quality_impact"] == 0.5
        assert agent.priority_weights["reversibility"] == 0.3
        assert agent.priority_weights["user_burden"] == 0.2


# ============================================================================
# formulate_questions() Tests
# ============================================================================

class TestFormulateQuestions:
    """Tests for ClarifierAgent.formulate_questions."""

    def test_empty_missing_info(self, clarifier, empty_report):
        """Test formulate_questions with no missing info."""
        result = clarifier.formulate_questions(empty_report)
        assert isinstance(result, ClarificationRequest)
        assert result.total_questions == 0
        assert result.questions == []
        assert result.can_proceed_with_defaults is True
        assert "No clarifications needed" in result.recommended_workflow
        assert "High quality" in result.expected_quality_with_defaults

    def test_with_missing_info(self, clarifier, report_with_missing):
        """Test formulate_questions with various missing info."""
        result = clarifier.formulate_questions(report_with_missing)
        assert isinstance(result, ClarificationRequest)
        assert result.total_questions == 3
        assert len(result.questions) == 3

    def test_max_questions_limit(self, clarifier, report_many_missing):
        """Test that max_questions limits the output."""
        result = clarifier.formulate_questions(report_many_missing, max_questions=3)
        assert result.total_questions == 3
        assert len(result.questions) == 3

    def test_default_max_questions_is_5(self, clarifier, report_many_missing):
        """Test that default max_questions is 5."""
        result = clarifier.formulate_questions(report_many_missing)
        assert result.total_questions == 5

    def test_questions_ranked_by_priority(self, clarifier, report_with_missing):
        """Test that questions are ranked by priority (critical first)."""
        result = clarifier.formulate_questions(report_with_missing)
        priorities = [q.priority for q in result.questions]
        # Critical should come before HIGH, which comes before MEDIUM
        critical_indices = [i for i, p in enumerate(priorities) if p == QuestionPriority.CRITICAL]
        high_indices = [i for i, p in enumerate(priorities) if p == QuestionPriority.HIGH]
        medium_indices = [i for i, p in enumerate(priorities) if p == QuestionPriority.MEDIUM]
        if critical_indices and high_indices:
            assert max(critical_indices) < min(high_indices)
        if high_indices and medium_indices:
            assert max(high_indices) < min(medium_indices)

    def test_with_context(self, clarifier, report_with_missing):
        """Test formulate_questions with additional context."""
        context = {"previous_clarifications": ["auth was discussed"]}
        result = clarifier.formulate_questions(report_with_missing, context=context)
        assert isinstance(result, ClarificationRequest)

    def test_has_critical_affects_workflow(self, clarifier, report_critical_only):
        """Test that critical questions affect recommended workflow."""
        result = clarifier.formulate_questions(report_critical_only)
        assert "critical" in result.recommended_workflow.lower()

    def test_can_proceed_with_defaults_set(self, clarifier, report_with_missing):
        """Test that can_proceed_with_defaults is determined."""
        result = clarifier.formulate_questions(report_with_missing)
        assert isinstance(result.can_proceed_with_defaults, bool)


# ============================================================================
# _generate_questions() Tests
# ============================================================================

class TestGenerateQuestions:
    """Tests for ClarifierAgent._generate_questions."""

    def test_generates_one_per_missing_info(self, clarifier, report_with_missing):
        """Test that one question is generated per missing info item."""
        questions = clarifier._generate_questions(
            report_with_missing.missing_info,
            report_with_missing.literal_request,
        )
        assert len(questions) == len(report_with_missing.missing_info)

    def test_question_has_all_fields(self, clarifier, report_with_missing):
        """Test that each question has all required fields."""
        questions = clarifier._generate_questions(
            report_with_missing.missing_info,
            report_with_missing.literal_request,
        )
        for q in questions:
            assert isinstance(q, ClarificationQuestion)
            assert q.question
            assert q.priority
            assert q.reason
            assert q.context
            assert q.default_answer
            assert q.impact_if_unanswered

    def test_default_assumption_used(self, clarifier):
        """Test that MissingInfo.default_assumption is used as default_answer."""
        missing = [
            MissingInfo(
                requirement="language",
                severity=SeverityLevel.IMPORTANT,
                impact="code generation",
                default_assumption="Python 3.11",
            ),
        ]
        questions = clarifier._generate_questions(missing, "Build an app")
        assert questions[0].default_answer == "Python 3.11"

    def test_no_default_assumption_uses_fallback(self, clarifier):
        """Test fallback when default_assumption is None."""
        missing = [
            MissingInfo(
                requirement="something",
                severity=SeverityLevel.NICE_TO_HAVE,
                impact="minor",
                default_assumption=None,
            ),
        ]
        questions = clarifier._generate_questions(missing, "Do something")
        assert questions[0].default_answer == "Will use standard best practices"

    def test_with_context_param(self, clarifier):
        """Test that context param is accepted (used for future extension)."""
        missing = [
            MissingInfo(requirement="x", severity=SeverityLevel.IMPORTANT, impact="y"),
        ]
        context = {"user_preference": "minimal"}
        questions = clarifier._generate_questions(missing, "task", context)
        assert len(questions) == 1


# ============================================================================
# _build_question_text() Tests
# ============================================================================

class TestBuildQuestionText:
    """Tests for ClarifierAgent._build_question_text."""

    @pytest.mark.parametrize("severity,expected_prefix", [
        (SeverityLevel.CRITICAL, "What"),
        (SeverityLevel.IMPORTANT, "Which"),
        (SeverityLevel.NICE_TO_HAVE, "Any preference for"),
    ])
    def test_question_template_by_severity(self, clarifier, severity, expected_prefix):
        """Test that each severity level uses the correct question template."""
        text = clarifier._build_question_text("authentication method", severity)
        assert text.startswith(expected_prefix)

    def test_requirement_lowercased(self, clarifier):
        """Test that the requirement is lowercased in the question."""
        text = clarifier._build_question_text("Authentication Method", SeverityLevel.CRITICAL)
        assert "authentication method" in text

    def test_critical_template(self, clarifier):
        """Test exact critical template format."""
        text = clarifier._build_question_text("database", SeverityLevel.CRITICAL)
        assert text == "What database should be used?"

    def test_important_template(self, clarifier):
        """Test exact important template format."""
        text = clarifier._build_question_text("framework", SeverityLevel.IMPORTANT)
        assert text == "Which framework do you prefer?"

    def test_nice_to_have_template(self, clarifier):
        """Test exact nice_to_have template format."""
        text = clarifier._build_question_text("color scheme", SeverityLevel.NICE_TO_HAVE)
        assert text == "Any preference for color scheme?"


# ============================================================================
# _build_reason() Tests
# ============================================================================

class TestBuildReason:
    """Tests for ClarifierAgent._build_reason."""

    def test_reason_contains_impact(self, clarifier):
        """Test that reason contains the impact from MissingInfo."""
        info = MissingInfo(
            requirement="auth",
            severity=SeverityLevel.CRITICAL,
            impact="Cannot design security layer",
        )
        reason = clarifier._build_reason(info)
        assert "Cannot design security layer" in reason

    def test_reason_format(self, clarifier):
        """Test the exact format of the reason string."""
        info = MissingInfo(
            requirement="db",
            severity=SeverityLevel.IMPORTANT,
            impact="Affects queries",
        )
        reason = clarifier._build_reason(info)
        assert reason == "This affects: Affects queries"


# ============================================================================
# _build_context() Tests
# ============================================================================

class TestBuildContext:
    """Tests for ClarifierAgent._build_context."""

    def test_context_contains_requirement(self, clarifier):
        """Test that context mentions the requirement."""
        info = MissingInfo(
            requirement="authentication method",
            severity=SeverityLevel.CRITICAL,
            impact="security",
        )
        context = clarifier._build_context(info, "Build an API")
        assert "authentication method" in context

    def test_context_format(self, clarifier):
        """Test the exact format of the context string."""
        info = MissingInfo(
            requirement="database",
            severity=SeverityLevel.IMPORTANT,
            impact="storage",
        )
        context = clarifier._build_context(info, "Build an app")
        assert context == "To complete your request, I need to know about database."


# ============================================================================
# _assess_impact() Tests
# ============================================================================

class TestAssessImpact:
    """Tests for ClarifierAgent._assess_impact."""

    def test_critical_impact(self, clarifier):
        """Test impact assessment for critical severity."""
        info = MissingInfo(
            requirement="auth",
            severity=SeverityLevel.CRITICAL,
            impact="security",
        )
        impact = clarifier._assess_impact(info)
        assert impact["quality"] == "Severe impact - output may be unusable"
        assert impact["risk"] == "high"
        assert len(impact["revisions"]) == 2
        assert "completely redo" in impact["revisions"][0]

    def test_important_impact(self, clarifier):
        """Test impact assessment for important severity."""
        info = MissingInfo(
            requirement="db",
            severity=SeverityLevel.IMPORTANT,
            impact="queries",
        )
        impact = clarifier._assess_impact(info)
        assert impact["quality"] == "Moderate impact - output quality degraded"
        assert impact["risk"] == "medium"
        assert len(impact["revisions"]) == 2
        assert "significant revisions" in impact["revisions"][0]

    def test_nice_to_have_impact(self, clarifier):
        """Test impact assessment for nice_to_have severity."""
        info = MissingInfo(
            requirement="color",
            severity=SeverityLevel.NICE_TO_HAVE,
            impact="aesthetics",
        )
        impact = clarifier._assess_impact(info)
        assert impact["quality"] == "Minor impact - output should be acceptable"
        assert impact["risk"] == "low"
        assert len(impact["revisions"]) == 2
        assert "Minor enhancements" in impact["revisions"][0]


# ============================================================================
# _get_answer_options() Tests
# ============================================================================

class TestGetAnswerOptions:
    """Tests for ClarifierAgent._get_answer_options."""

    @pytest.mark.parametrize("requirement,expected_options", [
        ("authentication method", ["JWT", "OAuth 2.0", "API Key", "Session-based"]),
        ("database technology", ["PostgreSQL", "MySQL", "MongoDB", "SQLite", "None"]),
        ("deployment target", ["Docker", "Kubernetes", "Cloud (AWS/Azure/GCP)", "Local"]),
        ("programming language", ["Python", "JavaScript/TypeScript", "Java", "Go", "C++"]),
        ("testing framework", ["pytest", "unittest", "Jest", "JUnit", "None"]),
        ("documentation format", ["Markdown", "HTML", "PDF", "DocX"]),
    ])
    def test_known_requirements_return_options(self, clarifier, requirement, expected_options):
        """Test that known requirements return predefined options."""
        options = clarifier._get_answer_options(requirement)
        assert options == expected_options

    def test_unknown_requirement_returns_none(self, clarifier):
        """Test that unknown requirements return None."""
        options = clarifier._get_answer_options("color scheme")
        assert options is None

    def test_case_insensitive_matching(self, clarifier):
        """Test that matching is case-insensitive."""
        options = clarifier._get_answer_options("Authentication Method")
        assert options is not None
        assert "JWT" in options

    def test_partial_match(self, clarifier):
        """Test that partial matches work (substring matching)."""
        options = clarifier._get_answer_options("preferred database technology for this project")
        assert options is not None
        assert "PostgreSQL" in options


# ============================================================================
# _rank_questions() Tests
# ============================================================================

class TestRankQuestions:
    """Tests for ClarifierAgent._rank_questions."""

    def _make_question(self, priority, risk_level="medium"):
        """Helper to create a ClarificationQuestion."""
        return ClarificationQuestion(
            question="Test?",
            priority=priority,
            reason="test",
            context="test",
            default_answer="default",
            impact_if_unanswered=ImpactAssessment(
                quality_impact="test",
                risk_level=risk_level,
                potential_revisions=["test"],
            ),
            answer_options=None,
        )

    def test_critical_before_high(self, clarifier):
        """Test that critical questions come before high."""
        questions = [
            self._make_question(QuestionPriority.HIGH),
            self._make_question(QuestionPriority.CRITICAL),
        ]
        ranked = clarifier._rank_questions(questions)
        assert ranked[0].priority == QuestionPriority.CRITICAL
        assert ranked[1].priority == QuestionPriority.HIGH

    def test_high_before_medium(self, clarifier):
        """Test that high questions come before medium."""
        questions = [
            self._make_question(QuestionPriority.MEDIUM),
            self._make_question(QuestionPriority.HIGH),
        ]
        ranked = clarifier._rank_questions(questions)
        assert ranked[0].priority == QuestionPriority.HIGH
        assert ranked[1].priority == QuestionPriority.MEDIUM

    def test_medium_before_low(self, clarifier):
        """Test that medium questions come before low."""
        questions = [
            self._make_question(QuestionPriority.LOW),
            self._make_question(QuestionPriority.MEDIUM),
        ]
        ranked = clarifier._rank_questions(questions)
        assert ranked[0].priority == QuestionPriority.MEDIUM
        assert ranked[1].priority == QuestionPriority.LOW

    def test_full_ordering(self, clarifier):
        """Test full priority ordering: critical > high > medium > low."""
        questions = [
            self._make_question(QuestionPriority.LOW),
            self._make_question(QuestionPriority.CRITICAL),
            self._make_question(QuestionPriority.MEDIUM),
            self._make_question(QuestionPriority.HIGH),
        ]
        ranked = clarifier._rank_questions(questions)
        priorities = [q.priority for q in ranked]
        assert priorities == [
            QuestionPriority.CRITICAL,
            QuestionPriority.HIGH,
            QuestionPriority.MEDIUM,
            QuestionPriority.LOW,
        ]

    def test_same_priority_sorted_by_risk(self, clarifier):
        """Test that within same priority, higher risk comes first."""
        questions = [
            self._make_question(QuestionPriority.HIGH, risk_level="low"),
            self._make_question(QuestionPriority.HIGH, risk_level="high"),
            self._make_question(QuestionPriority.HIGH, risk_level="medium"),
        ]
        ranked = clarifier._rank_questions(questions)
        risks = [q.impact_if_unanswered.risk_level for q in ranked]
        assert risks == ["high", "medium", "low"]

    def test_empty_list(self, clarifier):
        """Test ranking an empty list."""
        ranked = clarifier._rank_questions([])
        assert ranked == []


# ============================================================================
# _severity_to_priority() Tests
# ============================================================================

class TestSeverityToPriority:
    """Tests for ClarifierAgent._severity_to_priority."""

    @pytest.mark.parametrize("severity,expected_priority", [
        (SeverityLevel.CRITICAL, QuestionPriority.CRITICAL),
        (SeverityLevel.IMPORTANT, QuestionPriority.HIGH),
        (SeverityLevel.NICE_TO_HAVE, QuestionPriority.MEDIUM),
    ])
    def test_mapping(self, clarifier, severity, expected_priority):
        """Test severity to priority mapping."""
        assert clarifier._severity_to_priority(severity) == expected_priority

    def test_unknown_severity_returns_low(self, clarifier):
        """Test that an unknown severity defaults to LOW."""
        # Create a mock severity that won't match
        result = clarifier._severity_to_priority("unknown_value")
        assert result == QuestionPriority.LOW


# ============================================================================
# _determine_workflow() Tests
# ============================================================================

class TestDetermineWorkflow:
    """Tests for ClarifierAgent._determine_workflow."""

    def _make_question(self, priority):
        return ClarificationQuestion(
            question="Test?", priority=priority, reason="r", context="c",
            default_answer="d",
            impact_if_unanswered=ImpactAssessment(
                quality_impact="q", risk_level="low", potential_revisions=["r"],
            ),
        )

    def test_has_critical(self, clarifier):
        """Test workflow when critical questions exist."""
        questions = [self._make_question(QuestionPriority.CRITICAL)]
        result = clarifier._determine_workflow(questions, has_critical=True)
        assert "critical questions first" in result.lower()
        assert "Wait for answers" in result

    def test_few_questions_no_critical(self, clarifier):
        """Test workflow with <=3 questions and no critical."""
        questions = [
            self._make_question(QuestionPriority.HIGH),
            self._make_question(QuestionPriority.MEDIUM),
        ]
        result = clarifier._determine_workflow(questions, has_critical=False)
        assert "all questions together" in result.lower()
        assert "defaults" in result.lower()

    def test_many_questions_no_critical(self, clarifier):
        """Test workflow with >3 questions and no critical."""
        questions = [self._make_question(QuestionPriority.HIGH) for _ in range(5)]
        result = clarifier._determine_workflow(questions, has_critical=False)
        assert "priority groups" in result.lower()
        assert "skip" in result.lower()

    def test_exactly_3_questions_no_critical(self, clarifier):
        """Test workflow with exactly 3 questions and no critical (boundary)."""
        questions = [self._make_question(QuestionPriority.MEDIUM) for _ in range(3)]
        result = clarifier._determine_workflow(questions, has_critical=False)
        assert "all questions together" in result.lower()

    def test_4_questions_no_critical(self, clarifier):
        """Test workflow with 4 questions and no critical (boundary)."""
        questions = [self._make_question(QuestionPriority.MEDIUM) for _ in range(4)]
        result = clarifier._determine_workflow(questions, has_critical=False)
        assert "priority groups" in result.lower()


# ============================================================================
# _can_proceed_with_defaults() Tests
# ============================================================================

class TestCanProceedWithDefaults:
    """Tests for ClarifierAgent._can_proceed_with_defaults."""

    def _make_question(self, priority, default_answer="default"):
        return ClarificationQuestion(
            question="Test?", priority=priority, reason="r", context="c",
            default_answer=default_answer,
            impact_if_unanswered=ImpactAssessment(
                quality_impact="q", risk_level="low", potential_revisions=["r"],
            ),
        )

    def test_no_critical_can_proceed(self, clarifier):
        """Test that no critical questions allows proceeding."""
        questions = [
            self._make_question(QuestionPriority.HIGH),
            self._make_question(QuestionPriority.MEDIUM),
        ]
        assert clarifier._can_proceed_with_defaults(questions) is True

    def test_critical_with_defaults_can_proceed(self, clarifier):
        """Test that critical with all defaults allows proceeding."""
        questions = [
            self._make_question(QuestionPriority.CRITICAL, default_answer="JWT"),
            self._make_question(QuestionPriority.HIGH, default_answer="PostgreSQL"),
        ]
        assert clarifier._can_proceed_with_defaults(questions) is True

    def test_critical_without_defaults_cannot_proceed(self, clarifier):
        """Test that critical without defaults blocks proceeding."""
        questions = [
            self._make_question(QuestionPriority.CRITICAL, default_answer=""),
        ]
        assert clarifier._can_proceed_with_defaults(questions) is False

    def test_empty_questions_can_proceed(self, clarifier):
        """Test that empty questions list allows proceeding."""
        assert clarifier._can_proceed_with_defaults([]) is True

    def test_all_low_priority_can_proceed(self, clarifier):
        """Test that all low priority allows proceeding."""
        questions = [self._make_question(QuestionPriority.LOW) for _ in range(5)]
        assert clarifier._can_proceed_with_defaults(questions) is True

    def test_critical_with_mixed_defaults(self, clarifier):
        """Test critical present but one question has empty default."""
        questions = [
            self._make_question(QuestionPriority.CRITICAL, default_answer="JWT"),
            self._make_question(QuestionPriority.HIGH, default_answer=""),
        ]
        # has_critical=True, all_have_defaults=False => not (True) or False => False
        assert clarifier._can_proceed_with_defaults(questions) is False


# ============================================================================
# _assess_quality_with_defaults() Tests
# ============================================================================

class TestAssessQualityWithDefaults:
    """Tests for ClarifierAgent._assess_quality_with_defaults."""

    def _make_question(self, priority):
        return ClarificationQuestion(
            question="Test?", priority=priority, reason="r", context="c",
            default_answer="d",
            impact_if_unanswered=ImpactAssessment(
                quality_impact="q", risk_level="low", potential_revisions=["r"],
            ),
        )

    def _make_report(self):
        return TaskIntelligenceReport(
            literal_request="test",
            inferred_intent="test",
            sub_tasks=[],
            missing_info=[],
            assumptions=[],
            modality=ModalityType.TEXT,
            recommended_approach="direct",
            escalation_needed=False,
            suggested_tier=1,
            confidence=0.9,
        )

    def test_critical_questions_low_quality(self, clarifier):
        """Test that critical questions result in low quality assessment."""
        questions = [self._make_question(QuestionPriority.CRITICAL)]
        result = clarifier._assess_quality_with_defaults(questions, self._make_report())
        assert "Low" in result

    def test_many_high_questions_medium_quality(self, clarifier):
        """Test that >2 high questions result in medium quality."""
        questions = [self._make_question(QuestionPriority.HIGH) for _ in range(3)]
        result = clarifier._assess_quality_with_defaults(questions, self._make_report())
        assert "Medium" in result

    def test_few_high_questions_good_quality(self, clarifier):
        """Test that <=2 high questions result in good quality."""
        questions = [
            self._make_question(QuestionPriority.HIGH),
            self._make_question(QuestionPriority.MEDIUM),
        ]
        result = clarifier._assess_quality_with_defaults(questions, self._make_report())
        assert "Good" in result

    def test_no_questions_high_quality(self, clarifier):
        """Test that no questions results in high quality."""
        result = clarifier._assess_quality_with_defaults([], self._make_report())
        assert "High" in result

    def test_only_medium_low_questions_good_quality(self, clarifier):
        """Test that only medium/low questions result in good quality."""
        questions = [
            self._make_question(QuestionPriority.MEDIUM),
            self._make_question(QuestionPriority.LOW),
        ]
        result = clarifier._assess_quality_with_defaults(questions, self._make_report())
        assert "Good" in result

    def test_exactly_2_high_good_quality(self, clarifier):
        """Test that exactly 2 high questions (boundary) result in good quality."""
        questions = [self._make_question(QuestionPriority.HIGH) for _ in range(2)]
        result = clarifier._assess_quality_with_defaults(questions, self._make_report())
        assert "Good" in result

    def test_critical_overrides_high_count(self, clarifier):
        """Test that critical takes precedence over high count."""
        questions = [
            self._make_question(QuestionPriority.CRITICAL),
            self._make_question(QuestionPriority.HIGH),
            self._make_question(QuestionPriority.HIGH),
            self._make_question(QuestionPriority.HIGH),
        ]
        result = clarifier._assess_quality_with_defaults(questions, self._make_report())
        assert "Low" in result


# ============================================================================
# create_clarifier() Convenience Function Tests
# ============================================================================

class TestCreateClarifierFunction:
    """Tests for the create_clarifier() convenience function."""

    def test_default_params(self):
        """Test create_clarifier with default parameters."""
        with patch("builtins.open", mock_open(read_data="prompt")):
            agent = create_clarifier()
        assert isinstance(agent, ClarifierAgent)
        assert agent.system_prompt_path == "config/agents/clarifier/CLAUDE.md"
        assert agent.model == "claude-sonnet-4-20250514"

    def test_custom_params(self):
        """Test create_clarifier with custom parameters."""
        with patch("builtins.open", mock_open(read_data="custom")):
            agent = create_clarifier(
                system_prompt_path="custom/path.md",
                model="custom-model",
            )
        assert agent.system_prompt_path == "custom/path.md"
        assert agent.model == "custom-model"

    def test_returns_clarifier_agent(self):
        """Test that create_clarifier returns a ClarifierAgent instance."""
        with patch("builtins.open", mock_open(read_data="")):
            agent = create_clarifier()
        assert isinstance(agent, ClarifierAgent)


# ============================================================================
# Schema Integration Tests
# ============================================================================

class TestSchemaIntegration:
    """Test that ClarifierAgent output conforms to schemas."""

    def test_clarification_request_schema_valid(self, clarifier, report_with_missing):
        """Test that formulate_questions output is valid ClarificationRequest."""
        result = clarifier.formulate_questions(report_with_missing)
        assert isinstance(result, ClarificationRequest)
        data = result.model_dump()
        assert "total_questions" in data
        assert "questions" in data
        assert "recommended_workflow" in data

    def test_question_schema_valid(self, clarifier, report_with_missing):
        """Test that generated questions are valid ClarificationQuestion instances."""
        result = clarifier.formulate_questions(report_with_missing)
        for q in result.questions:
            assert isinstance(q, ClarificationQuestion)
            assert q.question
            assert q.priority in QuestionPriority
            assert isinstance(q.impact_if_unanswered, ImpactAssessment)

    def test_json_roundtrip(self, clarifier, report_with_missing):
        """Test that result can be serialized and deserialized."""
        result = clarifier.formulate_questions(report_with_missing)
        json_str = result.model_dump_json()
        restored = ClarificationRequest.model_validate_json(json_str)
        assert restored.total_questions == result.total_questions
        assert len(restored.questions) == len(result.questions)


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_missing_info(self, clarifier):
        """Test with exactly one missing info item."""
        report = TaskIntelligenceReport(
            literal_request="Do something",
            inferred_intent="Do it",
            sub_tasks=[],
            missing_info=[
                MissingInfo(
                    requirement="something",
                    severity=SeverityLevel.IMPORTANT,
                    impact="needed",
                    default_assumption="fallback",
                ),
            ],
            assumptions=[],
            modality=ModalityType.TEXT,
            recommended_approach="Direct",
            escalation_needed=False,
            suggested_tier=1,
            confidence=0.9,
        )
        result = clarifier.formulate_questions(report)
        assert result.total_questions == 1

    def test_all_critical_missing_info(self, clarifier, report_critical_only):
        """Test with all critical missing info."""
        result = clarifier.formulate_questions(report_critical_only)
        for q in result.questions:
            assert q.priority == QuestionPriority.CRITICAL

    def test_max_questions_zero(self, clarifier, report_with_missing):
        """Test with max_questions=0."""
        result = clarifier.formulate_questions(report_with_missing, max_questions=0)
        assert result.total_questions == 0
        assert result.questions == []

    def test_max_questions_exceeds_available(self, clarifier, report_with_missing):
        """Test with max_questions larger than available questions."""
        result = clarifier.formulate_questions(report_with_missing, max_questions=100)
        assert result.total_questions == len(report_with_missing.missing_info)

    def test_impact_assessment_structure(self, clarifier, report_with_missing):
        """Test that ImpactAssessment has correct structure."""
        result = clarifier.formulate_questions(report_with_missing)
        for q in result.questions:
            impact = q.impact_if_unanswered
            assert isinstance(impact, ImpactAssessment)
            assert impact.quality_impact
            assert impact.risk_level in ("low", "medium", "high")
            assert isinstance(impact.potential_revisions, list)
            assert len(impact.potential_revisions) > 0

    def test_answer_options_for_known_requirement(self, clarifier):
        """Test that known requirements get answer options."""
        report = TaskIntelligenceReport(
            literal_request="Build API",
            inferred_intent="Build it",
            sub_tasks=[],
            missing_info=[
                MissingInfo(
                    requirement="authentication method",
                    severity=SeverityLevel.CRITICAL,
                    impact="security",
                ),
            ],
            assumptions=[],
            modality=ModalityType.CODE,
            recommended_approach="Direct",
            escalation_needed=False,
            suggested_tier=1,
            confidence=0.9,
        )
        result = clarifier.formulate_questions(report)
        assert result.questions[0].answer_options is not None
        assert "JWT" in result.questions[0].answer_options

    def test_answer_options_none_for_unknown(self, clarifier):
        """Test that unknown requirements get no answer options."""
        report = TaskIntelligenceReport(
            literal_request="Build API",
            inferred_intent="Build it",
            sub_tasks=[],
            missing_info=[
                MissingInfo(
                    requirement="preferred color",
                    severity=SeverityLevel.NICE_TO_HAVE,
                    impact="aesthetics",
                ),
            ],
            assumptions=[],
            modality=ModalityType.TEXT,
            recommended_approach="Direct",
            escalation_needed=False,
            suggested_tier=1,
            confidence=0.9,
        )
        result = clarifier.formulate_questions(report)
        assert result.questions[0].answer_options is None
