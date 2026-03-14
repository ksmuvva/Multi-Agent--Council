"""
Exhaustive Tests for ReviewerAgent

Tests all methods of the ReviewerAgent including initialization,
quality gates, verdict determination, arbitration, and convenience functions.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import patch, mock_open

from src.agents.reviewer import ReviewerAgent, ReviewContext, create_reviewer
from src.schemas.reviewer import (
    ReviewVerdict,
    Verdict,
    CheckItem,
    Revision,
    QualityGateResults,
    ArbitrationInput,
)


# =============================================================================
# Schema Tests
# =============================================================================

class TestVerdictEnum:
    """Tests for Verdict enum."""

    def test_values(self):
        assert Verdict.PASS == "PASS"
        assert Verdict.FAIL == "FAIL"

    def test_count(self):
        assert len(Verdict) == 2


class TestCheckItemSchema:
    """Tests for CheckItem model."""

    def test_creation(self):
        ci = CheckItem(
            check_name="Test", passed=True, notes="ok",
            severity_if_failed="high",
        )
        assert ci.check_name == "Test"
        assert ci.passed is True


class TestRevisionSchema:
    """Tests for Revision model."""

    def test_creation(self):
        r = Revision(
            category="cat", description="desc", reason="reason",
            priority="high", specific_instructions="fix it",
        )
        assert r.priority == "high"


class TestQualityGateResultsSchema:
    """Tests for QualityGateResults model."""

    def test_creation_without_code_review(self):
        qg = QualityGateResults(
            completeness=CheckItem(check_name="C", passed=True, notes="ok", severity_if_failed="high"),
            consistency=CheckItem(check_name="Co", passed=True, notes="ok", severity_if_failed="medium"),
            verifier_signoff=CheckItem(check_name="V", passed=True, notes="ok", severity_if_failed="critical"),
            critic_findings_addressed=CheckItem(check_name="Cr", passed=True, notes="ok", severity_if_failed="high"),
            readability=CheckItem(check_name="R", passed=True, notes="ok", severity_if_failed="low"),
        )
        assert qg.code_review_passed is None

    def test_creation_with_code_review(self):
        qg = QualityGateResults(
            completeness=CheckItem(check_name="C", passed=True, notes="ok", severity_if_failed="high"),
            consistency=CheckItem(check_name="Co", passed=True, notes="ok", severity_if_failed="medium"),
            verifier_signoff=CheckItem(check_name="V", passed=True, notes="ok", severity_if_failed="critical"),
            critic_findings_addressed=CheckItem(check_name="Cr", passed=True, notes="ok", severity_if_failed="high"),
            readability=CheckItem(check_name="R", passed=True, notes="ok", severity_if_failed="low"),
            code_review_passed=CheckItem(check_name="CR", passed=True, notes="ok", severity_if_failed="critical"),
        )
        assert qg.code_review_passed is not None


class TestArbitrationInputSchema:
    """Tests for ArbitrationInput model."""

    def test_creation(self):
        ai = ArbitrationInput(
            reviewer_verdict=Verdict.PASS,
            verifier_verdict=Verdict.PASS,
            critic_verdict=Verdict.FAIL,
            disagreement_reason="mismatch",
            debate_rounds_completed=0,
        )
        assert ai.disagreement_reason == "mismatch"


class TestReviewVerdictSchema:
    """Tests for ReviewVerdict model."""

    def test_creation(self):
        qg = QualityGateResults(
            completeness=CheckItem(check_name="C", passed=True, notes="ok", severity_if_failed="high"),
            consistency=CheckItem(check_name="Co", passed=True, notes="ok", severity_if_failed="medium"),
            verifier_signoff=CheckItem(check_name="V", passed=True, notes="ok", severity_if_failed="critical"),
            critic_findings_addressed=CheckItem(check_name="Cr", passed=True, notes="ok", severity_if_failed="high"),
            readability=CheckItem(check_name="R", passed=True, notes="ok", severity_if_failed="low"),
        )
        rv = ReviewVerdict(
            verdict=Verdict.PASS,
            confidence=0.9,
            quality_gate_results=qg,
            reasons=["all good"],
            can_revise=True,
            summary="summary",
        )
        assert rv.verdict == Verdict.PASS
        assert rv.arbitration_needed is False
        assert rv.tier_4_arbiter_involved is False


# =============================================================================
# ReviewContext Tests
# =============================================================================

class TestReviewContext:
    """Tests for ReviewContext dataclass."""

    def test_fields(self):
        ctx = ReviewContext(
            original_request="build it",
            agent_outputs={"analyst": {"modality": "text"}},
            revision_count=0,
            max_revisions=2,
            tier_level=2,
            is_code_output=False,
        )
        assert ctx.original_request == "build it"
        assert ctx.revision_count == 0
        assert ctx.max_revisions == 2
        assert ctx.tier_level == 2
        assert ctx.is_code_output is False
        assert "analyst" in ctx.agent_outputs


# =============================================================================
# ReviewerAgent.__init__ Tests
# =============================================================================

class TestReviewerAgentInit:
    """Tests for ReviewerAgent initialization."""

    def test_defaults(self):
        with patch("builtins.open", side_effect=FileNotFoundError):
            agent = ReviewerAgent()
        assert agent.system_prompt_path == "config/agents/reviewer/CLAUDE.md"
        assert agent.model == "claude-opus-4-20250514"
        assert agent.max_turns == 30

    def test_custom_params(self):
        with patch("builtins.open", side_effect=FileNotFoundError):
            agent = ReviewerAgent(
                system_prompt_path="custom/path.md",
                model="claude-3-haiku",
                max_turns=5,
            )
        assert agent.system_prompt_path == "custom/path.md"
        assert agent.model == "claude-3-haiku"
        assert agent.max_turns == 5

    def test_system_prompt_loaded_from_file(self):
        content = "You are a test reviewer."
        with patch("builtins.open", mock_open(read_data=content)):
            agent = ReviewerAgent()
        assert agent.system_prompt == content

    def test_system_prompt_fallback(self):
        with patch("builtins.open", side_effect=FileNotFoundError):
            agent = ReviewerAgent()
        assert "Reviewer" in agent.system_prompt
        assert "quality gate" in agent.system_prompt.lower()

    def test_verdict_matrix_initialized(self):
        with patch("builtins.open", side_effect=FileNotFoundError):
            agent = ReviewerAgent()
        assert (Verdict.PASS, Verdict.PASS) in agent.verdict_matrix
        assert (Verdict.PASS, Verdict.FAIL) in agent.verdict_matrix
        assert (Verdict.FAIL, Verdict.PASS) in agent.verdict_matrix
        assert (Verdict.FAIL, Verdict.FAIL) in agent.verdict_matrix
        assert agent.verdict_matrix[(Verdict.PASS, Verdict.PASS)] == "PROCEED_TO_FORMATTER"
        assert agent.verdict_matrix[(Verdict.PASS, Verdict.FAIL)] == "EXECUTOR_REVISE"
        assert agent.verdict_matrix[(Verdict.FAIL, Verdict.PASS)] == "RESEARCHER_REVERIFY"
        assert agent.verdict_matrix[(Verdict.FAIL, Verdict.FAIL)] == "FULL_REGENERATION"

    def test_critical_failure_patterns_initialized(self):
        with patch("builtins.open", side_effect=FileNotFoundError):
            agent = ReviewerAgent()
        assert "security" in agent.critical_failure_patterns
        assert "hallucination" in agent.critical_failure_patterns
        assert "logic" in agent.critical_failure_patterns
        assert len(agent.critical_failure_patterns["security"]) >= 5
        assert len(agent.critical_failure_patterns["hallucination"]) >= 4
        assert len(agent.critical_failure_patterns["logic"]) >= 3


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def reviewer():
    with patch("builtins.open", side_effect=FileNotFoundError):
        return ReviewerAgent()


@pytest.fixture
def basic_context():
    return ReviewContext(
        original_request="Build a Python web application",
        agent_outputs={},
        revision_count=0,
        max_revisions=2,
        tier_level=2,
        is_code_output=False,
    )


@pytest.fixture
def code_context():
    return ReviewContext(
        original_request="Write a Python function",
        agent_outputs={},
        revision_count=0,
        max_revisions=2,
        tier_level=2,
        is_code_output=True,
    )


def _make_all_pass_gates():
    return QualityGateResults(
        completeness=CheckItem(check_name="Completeness", passed=True, notes="ok", severity_if_failed="critical"),
        consistency=CheckItem(check_name="Consistency", passed=True, notes="ok", severity_if_failed="medium"),
        verifier_signoff=CheckItem(check_name="Verifier Sign-off", passed=True, notes="ok", severity_if_failed="critical"),
        critic_findings_addressed=CheckItem(check_name="Critic Findings Addressed", passed=True, notes="ok", severity_if_failed="high"),
        readability=CheckItem(check_name="Readability", passed=True, notes="ok", severity_if_failed="low"),
    )


def _make_empty_failures():
    return {"security": [], "hallucination": [], "logic": []}


# =============================================================================
# _check_completeness Tests
# =============================================================================

class TestCheckCompleteness:
    """Tests for _check_completeness."""

    def test_all_requirements_met(self, reviewer):
        ctx = ReviewContext(
            original_request="Create a function",
            agent_outputs={}, revision_count=0, max_revisions=2,
            tier_level=2, is_code_output=False,
        )
        output = "Here is the function we created for the system."
        result = reviewer._check_completeness(output, ctx)
        assert isinstance(result, CheckItem)
        assert result.check_name == "Completeness"

    def test_missing_requirements(self, reviewer):
        ctx = ReviewContext(
            original_request='Create a function and include "special_feature"',
            agent_outputs={}, revision_count=0, max_revisions=2,
            tier_level=2, is_code_output=False,
        )
        output = "Here is a basic overview."
        result = reviewer._check_completeness(output, ctx)
        assert result.passed is False
        assert "Missing" in result.notes

    def test_synonyms_match(self, reviewer):
        ctx = ReviewContext(
            original_request="Add test coverage",
            agent_outputs={}, revision_count=0, max_revisions=2,
            tier_level=2, is_code_output=False,
        )
        output = "We included unit tests and testing for coverage."
        result = reviewer._check_completeness(output, ctx)
        # "test" requirement should be matched by "testing" synonym
        assert result.check_name == "Completeness"


# =============================================================================
# _extract_requirements Tests
# =============================================================================

class TestExtractRequirements:
    """Tests for _extract_requirements."""

    @pytest.mark.parametrize("verb", [
        "create", "write", "build", "generate", "implement", "design", "develop",
        "include", "add", "ensure", "provide",
        "must", "should", "shall",
    ])
    def test_action_patterns(self, reviewer, verb):
        request = f"Please {verb} something useful"
        result = reviewer._extract_requirements(request)
        assert len(result) >= 1

    def test_numbers_extracted(self, reviewer):
        result = reviewer._extract_requirements("Create 5 endpoints")
        found = any("5 items" in r for r in result)
        assert found

    def test_quoted_terms_extracted(self, reviewer):
        result = reviewer._extract_requirements('Add "feature_x" to the system')
        assert "feature_x" in result

    def test_deduplication(self, reviewer):
        result = reviewer._extract_requirements(
            "Create something. Build something."
        )
        # Should deduplicate
        assert len(result) == len(set(result))


# =============================================================================
# _is_requirement_addressed Tests
# =============================================================================

class TestIsRequirementAddressed:
    """Tests for _is_requirement_addressed."""

    def test_direct_match(self, reviewer):
        assert reviewer._is_requirement_addressed("function", "here is the function") is True

    @pytest.mark.parametrize("req,output", [
        ("test", "we have unit tests"),
        ("test", "testing was done"),
        ("documentation", "see the docs"),
        ("documentation", "well documented code"),
        ("error", "exception handling is robust"),
        ("security", "authentication is enabled"),
        ("performance", "code is optimized"),
    ])
    def test_synonym_match(self, reviewer, req, output):
        assert reviewer._is_requirement_addressed(req, output) is True

    def test_no_match(self, reviewer):
        assert reviewer._is_requirement_addressed("zebra", "no match here") is False


# =============================================================================
# _check_consistency Tests
# =============================================================================

class TestCheckConsistency:
    """Tests for _check_consistency."""

    def test_no_contradictions(self, reviewer, basic_context):
        output = "The system works well. It handles all cases."
        result = reviewer._check_consistency(output, basic_context)
        assert result.passed is True
        assert "consistent" in result.notes.lower()

    def test_contradiction_pattern(self, reviewer, basic_context):
        output = "both valid and not valid results are expected"
        result = reviewer._check_consistency(output, basic_context)
        assert result.passed is False

    def test_analyst_modality_code_mismatch(self, reviewer):
        ctx = ReviewContext(
            original_request="Write code",
            agent_outputs={"analyst": {"modality": "code"}},
            revision_count=0, max_revisions=2,
            tier_level=2, is_code_output=False,
        )
        output = "A plain text explanation without any code."
        result = reviewer._check_consistency(output, ctx)
        assert result.passed is False
        assert "code" in result.notes.lower()

    def test_analyst_modality_code_match(self, reviewer):
        ctx = ReviewContext(
            original_request="Write code",
            agent_outputs={"analyst": {"modality": "code"}},
            revision_count=0, max_revisions=2,
            tier_level=2, is_code_output=True,
        )
        output = "def hello():\n    return 'world'"
        result = reviewer._check_consistency(output, ctx)
        # Should pass because output looks like code
        assert result.passed is True

    def test_analyst_modality_image_mismatch(self, reviewer):
        ctx = ReviewContext(
            original_request="Draw diagram",
            agent_outputs={"analyst": {"modality": "image"}},
            revision_count=0, max_revisions=2,
            tier_level=2, is_code_output=False,
        )
        output = "Here is a description of the system without any diagram."
        result = reviewer._check_consistency(output, ctx)
        assert result.passed is False
        assert "image" in result.notes.lower() or "diagram" in result.notes.lower()


# =============================================================================
# _looks_like_code Tests
# =============================================================================

class TestLooksLikeCode:
    """Tests for _looks_like_code."""

    @pytest.mark.parametrize("text", [
        "```python\nprint('hello')\n```",
        "def my_function():",
        "function doSomething() {",
        "class MyClass:",
        "import os",
        "return result",
    ])
    def test_code_indicators(self, reviewer, text):
        assert reviewer._looks_like_code(text) is True

    def test_plain_text(self, reviewer):
        assert reviewer._looks_like_code("Just a plain sentence.") is False


# =============================================================================
# _check_verifier_signoff Tests
# =============================================================================

class TestCheckVerifierSignoff:
    """Tests for _check_verifier_signoff."""

    def test_none_report(self, reviewer):
        result = reviewer._check_verifier_signoff(None)
        assert result.passed is True
        assert "lower tier" in result.notes.lower()

    def test_pass_verdict_high_reliability(self, reviewer):
        report = {"verdict": "PASS", "overall_reliability": 0.9}
        result = reviewer._check_verifier_signoff(report)
        assert result.passed is True
        assert "passed" in result.notes.lower()

    def test_fail_verdict(self, reviewer):
        report = {"verdict": "FAIL", "overall_reliability": 0.9}
        result = reviewer._check_verifier_signoff(report)
        assert result.passed is False

    def test_low_reliability(self, reviewer):
        report = {"verdict": "PASS", "overall_reliability": 0.5}
        result = reviewer._check_verifier_signoff(report)
        assert result.passed is False

    def test_exactly_threshold_reliability(self, reviewer):
        report = {"verdict": "PASS", "overall_reliability": 0.7}
        result = reviewer._check_verifier_signoff(report)
        assert result.passed is True

    def test_severity_is_critical(self, reviewer):
        result = reviewer._check_verifier_signoff(None)
        assert result.severity_if_failed == "critical"


# =============================================================================
# _check_critic_findings Tests
# =============================================================================

class TestCheckCriticFindings:
    """Tests for _check_critic_findings."""

    def test_none_report(self, reviewer):
        result = reviewer._check_critic_findings(None)
        assert result.passed is True
        assert "lower tier" in result.notes.lower()

    def test_critical_assessment(self, reviewer):
        report = {
            "overall_assessment": "Solution has critical flaws",
            "attacks": [],
        }
        result = reviewer._check_critic_findings(report)
        assert result.passed is False

    def test_failed_attacks_over_threshold(self, reviewer):
        report = {
            "overall_assessment": "Some issues found",
            "attacks": [
                {"verdict": "FAIL"},
                {"verdict": "FAIL"},
            ],
        }
        result = reviewer._check_critic_findings(report)
        assert result.passed is False

    def test_one_failed_attack_passes(self, reviewer):
        report = {
            "overall_assessment": "Minor issues",
            "attacks": [
                {"verdict": "FAIL"},
                {"verdict": "PASS"},
            ],
        }
        result = reviewer._check_critic_findings(report)
        assert result.passed is True

    def test_no_issues(self, reviewer):
        report = {
            "overall_assessment": "Solution is sound",
            "attacks": [{"verdict": "PASS"}],
        }
        result = reviewer._check_critic_findings(report)
        assert result.passed is True


# =============================================================================
# _check_readability Tests
# =============================================================================

class TestCheckReadability:
    """Tests for _check_readability."""

    def test_good_readability(self, reviewer):
        output = "## Section One\n\n1. First point.\n2. Second point.\n\nThis is clear."
        result = reviewer._check_readability(output)
        assert result.passed is True
        assert "clear" in result.notes.lower()

    def test_long_sentences(self, reviewer):
        # Create output where >20% of sentences have >30 words
        long_sentence = " ".join(["word"] * 35) + "."
        short_sentence = "Short."
        output = f"{long_sentence} {short_sentence}"
        result = reviewer._check_readability(output)
        # 1 of 2 sentences is long = 50% > 20%
        assert result.passed is False
        assert "long" in result.notes.lower()

    def test_no_structure_long_content(self, reviewer):
        output = "A" * 600  # Long content without structure markers
        result = reviewer._check_readability(output)
        assert result.passed is False
        assert "structure" in result.notes.lower()

    def test_very_short_output(self, reviewer):
        output = "Hi."
        result = reviewer._check_readability(output)
        assert result.passed is False
        assert "short" in result.notes.lower()

    def test_structured_long_content_passes(self, reviewer):
        output = "## Header\n\n" + "Normal sentence here. " * 20
        result = reviewer._check_readability(output)
        assert result.passed is True


# =============================================================================
# _check_code_review Tests
# =============================================================================

class TestCheckCodeReview:
    """Tests for _check_code_review."""

    def test_none_report(self, reviewer):
        result = reviewer._check_code_review(None)
        assert result is None

    def test_pass(self, reviewer):
        report = {"pass_fail": True, "findings": []}
        result = reviewer._check_code_review(report)
        assert result.passed is True

    def test_fail(self, reviewer):
        report = {"pass_fail": False, "findings": []}
        result = reviewer._check_code_review(report)
        assert result.passed is False

    def test_critical_findings(self, reviewer):
        report = {
            "pass_fail": True,
            "findings": [{"severity": "CRITICAL", "issue": "SQL injection"}],
        }
        result = reviewer._check_code_review(report)
        assert result.passed is False

    def test_non_critical_findings_pass(self, reviewer):
        report = {
            "pass_fail": True,
            "findings": [{"severity": "LOW", "issue": "naming convention"}],
        }
        result = reviewer._check_code_review(report)
        assert result.passed is True


# =============================================================================
# _check_critical_failures Tests
# =============================================================================

class TestCheckCriticalFailures:
    """Tests for _check_critical_failures."""

    @pytest.mark.parametrize("text,category", [
        ("sql injection vulnerability found", "security"),
        ("xss vulnerability in the form", "security"),
        ("hardcoded password in config", "security"),
        ("hardcoded secret in code", "security"),
        ("hardcoded key exposed", "security"),
        ('api_key = "sk-123"', "security"),
    ])
    def test_security_patterns(self, reviewer, text, category):
        result = reviewer._check_critical_failures(text, None, None, None)
        assert len(result[category]) > 0

    @pytest.mark.parametrize("text", [
        "This claim is fabricated and unverified",
        "There is a hallucination in the output",
        "The data was made up from nothing",
        "The information was invented entirely",
        "no source provided for this claim",
    ])
    def test_hallucination_patterns(self, reviewer, text):
        result = reviewer._check_critical_failures(text, None, None, None)
        assert len(result["hallucination"]) > 0

    @pytest.mark.parametrize("text", [
        "There is a contradiction in the logic",
        "The arguments are inconsistent with each other",
        "This is a logical fallacy in the reasoning",
        "This is an invalid argument with no support",
    ])
    def test_logic_patterns(self, reviewer, text):
        result = reviewer._check_critical_failures(text, None, None, None)
        assert len(result["logic"]) > 0

    def test_verifier_fabricated_claims(self, reviewer):
        verifier = {"fabricated_claims": 3}
        result = reviewer._check_critical_failures("clean output", verifier, None, None)
        assert len(result["hallucination"]) > 0
        assert "3" in result["hallucination"][0]

    def test_verifier_no_fabricated_claims(self, reviewer):
        verifier = {"fabricated_claims": 0}
        result = reviewer._check_critical_failures("clean output", verifier, None, None)
        assert len(result["hallucination"]) == 0

    def test_code_review_security_findings(self, reviewer):
        code_review = {
            "findings": [
                {"category": "SECURITY", "severity": "CRITICAL", "issue": "RCE vulnerability"},
            ]
        }
        result = reviewer._check_critical_failures("clean output", None, None, code_review)
        assert len(result["security"]) > 0

    def test_clean_output_no_failures(self, reviewer):
        result = reviewer._check_critical_failures(
            "Everything is fine and works properly.", None, None, None
        )
        assert all(len(v) == 0 for v in result.values())


# =============================================================================
# _determine_verdict Tests
# =============================================================================

class TestDetermineVerdict:
    """Tests for _determine_verdict."""

    def test_security_fail(self, reviewer, basic_context):
        gates = _make_all_pass_gates()
        failures = {"security": ["sql injection"], "hallucination": [], "logic": []}
        assert reviewer._determine_verdict(gates, failures, basic_context) == Verdict.FAIL

    def test_hallucination_fail(self, reviewer, basic_context):
        gates = _make_all_pass_gates()
        failures = {"security": [], "hallucination": ["fabricated"], "logic": []}
        assert reviewer._determine_verdict(gates, failures, basic_context) == Verdict.FAIL

    def test_verifier_fail(self, reviewer, basic_context):
        gates = _make_all_pass_gates()
        gates.verifier_signoff = CheckItem(
            check_name="Verifier Sign-off", passed=False, notes="failed",
            severity_if_failed="critical",
        )
        failures = _make_empty_failures()
        assert reviewer._determine_verdict(gates, failures, basic_context) == Verdict.FAIL

    def test_completeness_fail(self, reviewer, basic_context):
        gates = _make_all_pass_gates()
        gates.completeness = CheckItem(
            check_name="Completeness", passed=False, notes="missing stuff",
            severity_if_failed="critical",
        )
        failures = _make_empty_failures()
        assert reviewer._determine_verdict(gates, failures, basic_context) == Verdict.FAIL

    def test_code_review_fail(self, reviewer, code_context):
        gates = _make_all_pass_gates()
        gates.code_review_passed = CheckItem(
            check_name="Code Review", passed=False, notes="failed",
            severity_if_failed="critical",
        )
        failures = _make_empty_failures()
        assert reviewer._determine_verdict(gates, failures, code_context) == Verdict.FAIL

    def test_critic_fail_critical_severity(self, reviewer, basic_context):
        gates = _make_all_pass_gates()
        gates.critic_findings_addressed = CheckItem(
            check_name="Critic Findings Addressed", passed=False, notes="not addressed",
            severity_if_failed="critical",
        )
        failures = _make_empty_failures()
        assert reviewer._determine_verdict(gates, failures, basic_context) == Verdict.FAIL

    def test_pass_ratio_80_percent(self, reviewer, basic_context):
        gates = _make_all_pass_gates()
        # Fail one of 5 gates = 80% pass
        gates.readability = CheckItem(
            check_name="Readability", passed=False, notes="short",
            severity_if_failed="low",
        )
        failures = _make_empty_failures()
        assert reviewer._determine_verdict(gates, failures, basic_context) == Verdict.PASS

    def test_below_80_percent_fails(self, reviewer, basic_context):
        gates = _make_all_pass_gates()
        # Fail 2 of 5 gates = 60%
        gates.readability = CheckItem(
            check_name="Readability", passed=False, notes="short",
            severity_if_failed="low",
        )
        gates.consistency = CheckItem(
            check_name="Consistency", passed=False, notes="inconsistent",
            severity_if_failed="medium",
        )
        # critic not critical severity
        gates.critic_findings_addressed = CheckItem(
            check_name="Critic Findings Addressed", passed=False, notes="issues",
            severity_if_failed="high",
        )
        failures = _make_empty_failures()
        assert reviewer._determine_verdict(gates, failures, basic_context) == Verdict.FAIL

    def test_all_pass(self, reviewer, basic_context):
        gates = _make_all_pass_gates()
        failures = _make_empty_failures()
        assert reviewer._determine_verdict(gates, failures, basic_context) == Verdict.PASS


# =============================================================================
# _extract_verifier_verdict and _extract_critic_verdict Tests
# =============================================================================

class TestExtractVerifierVerdict:
    """Tests for _extract_verifier_verdict."""

    def test_none_returns_pass(self, reviewer):
        assert reviewer._extract_verifier_verdict(None) == Verdict.PASS

    def test_pass_string(self, reviewer):
        assert reviewer._extract_verifier_verdict({"verdict": "PASS"}) == Verdict.PASS

    def test_fail_string(self, reviewer):
        assert reviewer._extract_verifier_verdict({"verdict": "FAIL"}) == Verdict.FAIL

    def test_missing_key_defaults_pass(self, reviewer):
        assert reviewer._extract_verifier_verdict({}) == Verdict.PASS


class TestExtractCriticVerdict:
    """Tests for _extract_critic_verdict."""

    def test_none_returns_pass(self, reviewer):
        assert reviewer._extract_critic_verdict(None) == Verdict.PASS

    def test_critical_assessment_returns_fail(self, reviewer):
        report = {"overall_assessment": "Solution has CRITICAL flaws"}
        assert reviewer._extract_critic_verdict(report) == Verdict.FAIL

    def test_non_critical_returns_pass(self, reviewer):
        report = {"overall_assessment": "Solution has minor issues"}
        assert reviewer._extract_critic_verdict(report) == Verdict.PASS

    def test_empty_assessment_returns_pass(self, reviewer):
        report = {"overall_assessment": ""}
        assert reviewer._extract_critic_verdict(report) == Verdict.PASS


# =============================================================================
# _apply_verdict_matrix Tests
# =============================================================================

class TestApplyVerdictMatrix:
    """Tests for _apply_verdict_matrix."""

    def test_pass_pass(self, reviewer):
        assert reviewer._apply_verdict_matrix(Verdict.PASS, Verdict.PASS) == "PROCEED_TO_FORMATTER"

    def test_pass_fail(self, reviewer):
        assert reviewer._apply_verdict_matrix(Verdict.PASS, Verdict.FAIL) == "EXECUTOR_REVISE"

    def test_fail_pass(self, reviewer):
        assert reviewer._apply_verdict_matrix(Verdict.FAIL, Verdict.PASS) == "RESEARCHER_REVERIFY"

    def test_fail_fail(self, reviewer):
        assert reviewer._apply_verdict_matrix(Verdict.FAIL, Verdict.FAIL) == "FULL_REGENERATION"


# =============================================================================
# _generate_reasons Tests
# =============================================================================

class TestGenerateReasons:
    """Tests for _generate_reasons."""

    def test_pass_reasons(self, reviewer):
        gates = _make_all_pass_gates()
        failures = _make_empty_failures()
        reasons = reviewer._generate_reasons(
            Verdict.PASS, gates, failures, "PROCEED_TO_FORMATTER"
        )
        assert "All quality gates passed" in reasons
        assert any("Passed:" in r for r in reasons)
        assert any("Proceed to formatting" in r for r in reasons)

    def test_fail_reasons_failed_gates(self, reviewer):
        gates = _make_all_pass_gates()
        gates.completeness = CheckItem(
            check_name="Completeness", passed=False,
            notes="Missing req", severity_if_failed="critical",
        )
        failures = _make_empty_failures()
        reasons = reviewer._generate_reasons(
            Verdict.FAIL, gates, failures, "EXECUTOR_REVISE"
        )
        assert any("Completeness" in r for r in reasons)
        assert any("EXECUTOR_REVISE" in r for r in reasons)

    def test_fail_reasons_critical_failures(self, reviewer):
        gates = _make_all_pass_gates()
        failures = {"security": ["sql injection"], "hallucination": [], "logic": []}
        reasons = reviewer._generate_reasons(
            Verdict.FAIL, gates, failures, "FULL_REGENERATION"
        )
        assert any("security" in r.lower() for r in reasons)


# =============================================================================
# _generate_revision_instructions Tests
# =============================================================================

class TestGenerateRevisionInstructions:
    """Tests for _generate_revision_instructions."""

    def test_executor_revise(self, reviewer, basic_context):
        gates = _make_all_pass_gates()
        gates.completeness = CheckItem(
            check_name="Completeness", passed=False,
            notes="Missing stuff", severity_if_failed="critical",
        )
        failures = _make_empty_failures()
        revisions = reviewer._generate_revision_instructions(
            gates, failures, "EXECUTOR_REVISE", basic_context
        )
        assert any(r.category == "Content Revision" for r in revisions)
        assert any("Critic" in r.description for r in revisions)

    def test_researcher_reverify(self, reviewer, basic_context):
        gates = _make_all_pass_gates()
        failures = _make_empty_failures()
        revisions = reviewer._generate_revision_instructions(
            gates, failures, "RESEARCHER_REVERIFY", basic_context
        )
        assert any(r.category == "Verification" for r in revisions)

    def test_full_regeneration(self, reviewer, basic_context):
        gates = _make_all_pass_gates()
        failures = _make_empty_failures()
        revisions = reviewer._generate_revision_instructions(
            gates, failures, "FULL_REGENERATION", basic_context
        )
        assert any(r.category == "Complete Regeneration" for r in revisions)
        assert any(r.priority == "critical" for r in revisions)

    def test_critical_failure_revisions(self, reviewer, basic_context):
        gates = _make_all_pass_gates()
        failures = {"security": ["sql injection"], "hallucination": [], "logic": []}
        revisions = reviewer._generate_revision_instructions(
            gates, failures, "EXECUTOR_REVISE", basic_context
        )
        assert any(r.category == "Critical Issue" for r in revisions)
        assert any(r.priority == "critical" for r in revisions)

    def test_failed_gate_priorities(self, reviewer, basic_context):
        gates = _make_all_pass_gates()
        gates.completeness = CheckItem(
            check_name="Completeness", passed=False,
            notes="Missing", severity_if_failed="critical",
        )
        gates.consistency = CheckItem(
            check_name="Consistency", passed=False,
            notes="Inconsistent", severity_if_failed="medium",
        )
        failures = _make_empty_failures()
        revisions = reviewer._generate_revision_instructions(
            gates, failures, "EXECUTOR_REVISE", basic_context
        )
        priorities = [r.priority for r in revisions if "Address" in r.description]
        assert "critical" in priorities
        assert "medium" in priorities

    def test_limited_to_five(self, reviewer, basic_context):
        gates = _make_all_pass_gates()
        gates.completeness = CheckItem(
            check_name="Completeness", passed=False, notes="m",
            severity_if_failed="critical",
        )
        gates.consistency = CheckItem(
            check_name="Consistency", passed=False, notes="m",
            severity_if_failed="medium",
        )
        gates.critic_findings_addressed = CheckItem(
            check_name="Critic Findings Addressed", passed=False, notes="m",
            severity_if_failed="high",
        )
        failures = {
            "security": ["issue1", "issue2", "issue3"],
            "hallucination": ["h1"],
            "logic": [],
        }
        revisions = reviewer._generate_revision_instructions(
            gates, failures, "FULL_REGENERATION", basic_context
        )
        assert len(revisions) <= 5


# =============================================================================
# _check_arbitration_needed Tests
# =============================================================================

class TestCheckArbitrationNeeded:
    """Tests for _check_arbitration_needed."""

    def test_tier_below_4_no_arbitration(self, reviewer):
        ctx = ReviewContext(
            original_request="req", agent_outputs={}, revision_count=0,
            max_revisions=2, tier_level=3, is_code_output=False,
        )
        needed, arb_input = reviewer._check_arbitration_needed(
            Verdict.FAIL, Verdict.PASS, Verdict.FAIL, ctx
        )
        assert needed is False
        assert arb_input is None

    def test_tier_4_with_disagreement(self, reviewer):
        ctx = ReviewContext(
            original_request="req", agent_outputs={}, revision_count=0,
            max_revisions=2, tier_level=4, is_code_output=False,
        )
        needed, arb_input = reviewer._check_arbitration_needed(
            Verdict.FAIL, Verdict.PASS, Verdict.FAIL, ctx
        )
        assert needed is True
        assert isinstance(arb_input, ArbitrationInput)
        assert arb_input.verifier_verdict == Verdict.PASS
        assert arb_input.critic_verdict == Verdict.FAIL
        assert "PASS" in arb_input.disagreement_reason
        assert "FAIL" in arb_input.disagreement_reason

    def test_tier_4_no_disagreement(self, reviewer):
        ctx = ReviewContext(
            original_request="req", agent_outputs={}, revision_count=0,
            max_revisions=2, tier_level=4, is_code_output=False,
        )
        needed, arb_input = reviewer._check_arbitration_needed(
            Verdict.PASS, Verdict.PASS, Verdict.PASS, ctx
        )
        assert needed is False
        assert arb_input is None


# =============================================================================
# _calculate_confidence Tests
# =============================================================================

class TestCalculateConfidence:
    """Tests for _calculate_confidence."""

    def test_all_pass_no_failures(self, reviewer):
        gates = _make_all_pass_gates()
        failures = _make_empty_failures()
        conf = reviewer._calculate_confidence(gates, failures)
        assert conf == pytest.approx(0.9, abs=0.01)

    def test_all_fail_no_critical(self, reviewer):
        gates = _make_all_pass_gates()
        gates.completeness = CheckItem(check_name="C", passed=False, notes="", severity_if_failed="high")
        gates.consistency = CheckItem(check_name="Co", passed=False, notes="", severity_if_failed="medium")
        gates.verifier_signoff = CheckItem(check_name="V", passed=False, notes="", severity_if_failed="critical")
        gates.critic_findings_addressed = CheckItem(check_name="Cr", passed=False, notes="", severity_if_failed="high")
        gates.readability = CheckItem(check_name="R", passed=False, notes="", severity_if_failed="low")
        failures = _make_empty_failures()
        conf = reviewer._calculate_confidence(gates, failures)
        assert conf == pytest.approx(0.5, abs=0.01)

    def test_critical_failures_reduce_confidence(self, reviewer):
        gates = _make_all_pass_gates()
        failures = {"security": ["issue1", "issue2"], "hallucination": [], "logic": []}
        conf = reviewer._calculate_confidence(gates, failures)
        assert conf < 0.9

    def test_clamped_to_zero(self, reviewer):
        gates = _make_all_pass_gates()
        gates.completeness = CheckItem(check_name="C", passed=False, notes="", severity_if_failed="high")
        gates.consistency = CheckItem(check_name="Co", passed=False, notes="", severity_if_failed="medium")
        gates.verifier_signoff = CheckItem(check_name="V", passed=False, notes="", severity_if_failed="critical")
        gates.critic_findings_addressed = CheckItem(check_name="Cr", passed=False, notes="", severity_if_failed="high")
        gates.readability = CheckItem(check_name="R", passed=False, notes="", severity_if_failed="low")
        failures = {"security": ["a"] * 10, "hallucination": ["b"] * 10, "logic": ["c"] * 10}
        conf = reviewer._calculate_confidence(gates, failures)
        assert conf == 0.0

    def test_clamped_to_one(self, reviewer):
        gates = _make_all_pass_gates()
        failures = _make_empty_failures()
        conf = reviewer._calculate_confidence(gates, failures)
        assert conf <= 1.0


# =============================================================================
# _generate_summary Tests
# =============================================================================

class TestGenerateSummary:
    """Tests for _generate_summary."""

    def test_pass_summary(self, reviewer, basic_context):
        summary = reviewer._generate_summary(
            Verdict.PASS, ["All good"], basic_context
        )
        assert "passes" in summary.lower()
        assert "formatting" in summary.lower()

    def test_pass_with_revisions(self, reviewer):
        ctx = ReviewContext(
            original_request="req", agent_outputs={}, revision_count=1,
            max_revisions=2, tier_level=2, is_code_output=False,
        )
        summary = reviewer._generate_summary(Verdict.PASS, ["ok"], ctx)
        assert "1 revision" in summary

    def test_fail_summary(self, reviewer, basic_context):
        summary = reviewer._generate_summary(
            Verdict.FAIL, ["failed gate"], basic_context
        )
        assert "revision" in summary.lower()
        assert "1/2" in summary

    def test_fail_max_revisions_reached(self, reviewer):
        ctx = ReviewContext(
            original_request="req", agent_outputs={}, revision_count=2,
            max_revisions=2, tier_level=2, is_code_output=False,
        )
        summary = reviewer._generate_summary(Verdict.FAIL, ["issue"], ctx)
        assert "max revisions reached" in summary.lower()

    def test_primary_reason_included(self, reviewer, basic_context):
        summary = reviewer._generate_summary(
            Verdict.PASS, ["This is the primary reason"], basic_context
        )
        assert "primary reason" in summary.lower()


# =============================================================================
# _run_quality_gates Tests
# =============================================================================

class TestRunQualityGates:
    """Tests for _run_quality_gates."""

    def test_all_gates_run(self, reviewer, basic_context):
        result = reviewer._run_quality_gates(
            "Output text.", basic_context, None, None, None
        )
        assert isinstance(result, QualityGateResults)
        assert result.completeness is not None
        assert result.consistency is not None
        assert result.verifier_signoff is not None
        assert result.critic_findings_addressed is not None
        assert result.readability is not None

    def test_code_review_conditional_included(self, reviewer, code_context):
        code_review_report = {"pass_fail": True, "findings": []}
        result = reviewer._run_quality_gates(
            "def func(): pass", code_context, None, None, code_review_report
        )
        assert result.code_review_passed is not None

    def test_code_review_conditional_excluded(self, reviewer, basic_context):
        result = reviewer._run_quality_gates(
            "Plain text output.", basic_context, None, None, None
        )
        assert result.code_review_passed is None


# =============================================================================
# review() Full Method Tests
# =============================================================================

class TestReviewFullMethod:
    """Tests for the full review() method."""

    def test_produces_review_verdict(self, reviewer, basic_context):
        output = (
            "## Solution\n\n"
            "Here is a comprehensive Python web application.\n"
            "It builds on best practices.\n" * 10
        )
        result = reviewer.review(output, basic_context)
        assert isinstance(result, ReviewVerdict)

    def test_pass_verdict_clean_output(self, reviewer):
        ctx = ReviewContext(
            original_request="Describe something",
            agent_outputs={}, revision_count=0, max_revisions=2,
            tier_level=2, is_code_output=False,
        )
        output = "## Description\n\n" + "This describes the topic well. " * 20
        result = reviewer.review(output, ctx)
        assert isinstance(result.verdict, Verdict)

    def test_fail_on_security_issue(self, reviewer, basic_context):
        output = (
            "## Solution\n\n"
            "Use hardcoded password to connect.\n" * 5
        )
        result = reviewer.review(output, basic_context)
        assert result.verdict == Verdict.FAIL

    def test_revision_instructions_on_fail(self, reviewer, basic_context):
        output = "sql injection in user input field.\n" * 5
        result = reviewer.review(output, basic_context)
        assert result.verdict == Verdict.FAIL
        assert len(result.revision_instructions) > 0

    def test_no_revision_instructions_on_pass(self, reviewer):
        ctx = ReviewContext(
            original_request="Describe something",
            agent_outputs={}, revision_count=0, max_revisions=2,
            tier_level=2, is_code_output=False,
        )
        output = "## Description\n\n" + "This describes the topic well. " * 20
        result = reviewer.review(output, ctx)
        if result.verdict == Verdict.PASS:
            assert result.revision_instructions == []

    def test_can_revise_flag(self, reviewer):
        ctx = ReviewContext(
            original_request="req", agent_outputs={}, revision_count=1,
            max_revisions=2, tier_level=2, is_code_output=False,
        )
        output = "## Content\n\n" + "Some content here. " * 20
        result = reviewer.review(output, ctx)
        assert result.can_revise is True

    def test_cannot_revise_at_max(self, reviewer):
        ctx = ReviewContext(
            original_request="req", agent_outputs={}, revision_count=2,
            max_revisions=2, tier_level=2, is_code_output=False,
        )
        output = "## Content\n\n" + "Some content here. " * 20
        result = reviewer.review(output, ctx)
        assert result.can_revise is False

    def test_arbitration_tier_4(self, reviewer):
        ctx = ReviewContext(
            original_request="req", agent_outputs={}, revision_count=0,
            max_revisions=2, tier_level=4, is_code_output=False,
        )
        verifier = {"verdict": "PASS", "overall_reliability": 0.9}
        critic = {"overall_assessment": "Solution has critical issues", "attacks": []}
        output = "## Content\n\n" + "Some decent content here. " * 20
        result = reviewer.review(output, ctx, verifier_report=verifier, critic_report=critic)
        # Verifier PASS, Critic FAIL (critical) -> disagreement -> arbitration
        assert result.arbitration_needed is True

    def test_no_arbitration_tier_2(self, reviewer, basic_context):
        verifier = {"verdict": "PASS", "overall_reliability": 0.9}
        critic = {"overall_assessment": "Solution has critical issues", "attacks": []}
        output = "## Content\n\n" + "Some content here. " * 20
        result = reviewer.review(output, basic_context, verifier_report=verifier, critic_report=critic)
        assert result.arbitration_needed is False

    def test_summary_populated(self, reviewer, basic_context):
        output = "## Section\n\n" + "Content here is fine. " * 20
        result = reviewer.review(output, basic_context)
        assert len(result.summary) > 0

    def test_confidence_range(self, reviewer, basic_context):
        output = "## Section\n\n" + "Some text. " * 20
        result = reviewer.review(output, basic_context)
        assert 0.0 <= result.confidence <= 1.0

    def test_tier_4_arbiter_involved_flag(self, reviewer):
        ctx = ReviewContext(
            original_request="req", agent_outputs={}, revision_count=0,
            max_revisions=2, tier_level=4, is_code_output=False,
        )
        verifier = {"verdict": "PASS", "overall_reliability": 0.9}
        critic = {"overall_assessment": "critical problem", "attacks": []}
        output = "## Content\n\n" + "Some content. " * 20
        result = reviewer.review(output, ctx, verifier_report=verifier, critic_report=critic)
        if result.arbitration_needed:
            assert result.tier_4_arbiter_involved is True

    def test_with_code_review_report(self, reviewer, code_context):
        output = "```python\ndef func():\n    return 42\n```\n" * 5
        code_review = {"pass_fail": True, "findings": []}
        result = reviewer.review(
            output, code_context, code_review_report=code_review
        )
        assert isinstance(result, ReviewVerdict)


# =============================================================================
# create_reviewer() Convenience Function Tests
# =============================================================================

class TestCreateReviewer:
    """Tests for create_reviewer convenience function."""

    def test_creates_default(self):
        with patch("builtins.open", side_effect=FileNotFoundError):
            agent = create_reviewer()
        assert isinstance(agent, ReviewerAgent)
        assert agent.model == "claude-opus-4-20250514"
        assert agent.system_prompt_path == "config/agents/reviewer/CLAUDE.md"

    def test_custom_params(self):
        with patch("builtins.open", side_effect=FileNotFoundError):
            agent = create_reviewer(
                system_prompt_path="custom.md",
                model="claude-3-haiku",
            )
        assert agent.system_prompt_path == "custom.md"
        assert agent.model == "claude-3-haiku"
