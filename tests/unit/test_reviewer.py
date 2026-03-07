"""
Tests for the ReviewerAgent.

Tests quality gate checks, verdict matrix, critical failure detection,
arbitration logic, and revision instruction generation.
"""

import pytest
from unittest.mock import patch, mock_open

from src.agents.reviewer import (
    ReviewerAgent,
    ReviewContext,
    create_reviewer,
)
from src.schemas.reviewer import (
    ReviewVerdict,
    Verdict,
    CheckItem,
)


@pytest.fixture
def reviewer():
    """Create a ReviewerAgent with no system prompt file."""
    return ReviewerAgent(system_prompt_path="nonexistent.md")


@pytest.fixture
def basic_context():
    """Create a basic ReviewContext."""
    return ReviewContext(
        original_request="Write a Python function to sort a list",
        agent_outputs={},
        revision_count=0,
        max_revisions=3,
        tier_level=2,
        is_code_output=False,
    )


@pytest.fixture
def code_context():
    """Create a code output ReviewContext."""
    return ReviewContext(
        original_request="Write a Python function to sort a list",
        agent_outputs={},
        revision_count=0,
        max_revisions=3,
        tier_level=2,
        is_code_output=True,
    )


@pytest.fixture
def tier4_context():
    """Create a Tier 4 ReviewContext."""
    return ReviewContext(
        original_request="Security audit of the system",
        agent_outputs={},
        revision_count=0,
        max_revisions=3,
        tier_level=4,
        is_code_output=False,
    )


GOOD_OUTPUT = """
## Python Sort Function

Here is a function to sort a list using Python:

```python
def sort_list(items):
    return sorted(items)
```

This function uses Python's built-in `sorted()` function for efficient sorting.
"""

SHORT_OUTPUT = "Yes."

SECURITY_OUTPUT = "The password is hardcoded: password = 'admin123'. sql injection is possible."


class TestReviewerInitialization:
    """Tests for ReviewerAgent initialization."""

    def test_default_initialization(self):
        """Test default init parameters."""
        agent = ReviewerAgent(system_prompt_path="nonexistent.md")
        assert agent.model == "claude-3-5-opus-20240507"
        assert agent.max_turns == 30

    def test_verdict_matrix_configured(self):
        """Test verdict matrix is properly configured."""
        agent = ReviewerAgent(system_prompt_path="nonexistent.md")
        assert (Verdict.PASS, Verdict.PASS) in agent.verdict_matrix
        assert agent.verdict_matrix[(Verdict.PASS, Verdict.PASS)] == "PROCEED_TO_FORMATTER"
        assert agent.verdict_matrix[(Verdict.FAIL, Verdict.FAIL)] == "FULL_REGENERATION"

    def test_critical_failure_patterns(self):
        """Test critical failure patterns are configured."""
        agent = ReviewerAgent(system_prompt_path="nonexistent.md")
        assert "security" in agent.critical_failure_patterns
        assert "hallucination" in agent.critical_failure_patterns

    def test_system_prompt_fallback(self):
        """Test fallback prompt."""
        agent = ReviewerAgent(system_prompt_path="nonexistent.md")
        assert "Reviewer" in agent.system_prompt

    def test_system_prompt_from_file(self):
        """Test loading from file."""
        with patch("builtins.open", mock_open(read_data="Reviewer prompt")):
            agent = ReviewerAgent(system_prompt_path="exists.md")
            assert agent.system_prompt == "Reviewer prompt"


class TestReview:
    """Tests for the review method."""

    def test_basic_review(self, reviewer, basic_context):
        """Test basic review produces ReviewVerdict."""
        verdict = reviewer.review(GOOD_OUTPUT, basic_context)
        assert isinstance(verdict, ReviewVerdict)
        assert verdict.verdict in [Verdict.PASS, Verdict.FAIL]

    def test_good_output_passes(self, reviewer, basic_context):
        """Test good output passes review."""
        basic_context.original_request = "sort a list"
        verdict = reviewer.review(GOOD_OUTPUT, basic_context)
        assert verdict.verdict == Verdict.PASS

    def test_security_output_fails(self, reviewer, basic_context):
        """Test security-problematic output fails."""
        verdict = reviewer.review(SECURITY_OUTPUT, basic_context)
        assert verdict.verdict == Verdict.FAIL

    def test_verdict_has_confidence(self, reviewer, basic_context):
        """Test verdict includes confidence score."""
        verdict = reviewer.review(GOOD_OUTPUT, basic_context)
        assert 0.0 <= verdict.confidence <= 1.0

    def test_verdict_has_summary(self, reviewer, basic_context):
        """Test verdict includes summary."""
        verdict = reviewer.review(GOOD_OUTPUT, basic_context)
        assert len(verdict.summary) > 0

    def test_can_revise_flag(self, reviewer, basic_context):
        """Test can_revise flag is set correctly."""
        verdict = reviewer.review(GOOD_OUTPUT, basic_context)
        assert verdict.can_revise is True  # revision_count=0 < max_revisions=3


class TestQualityGates:
    """Tests for quality gate checks."""

    def test_completeness_check(self, reviewer, basic_context):
        """Test completeness check."""
        basic_context.original_request = "sort"
        check = reviewer._check_completeness(GOOD_OUTPUT, basic_context)
        assert isinstance(check, CheckItem)
        assert check.check_name == "Completeness"

    def test_consistency_check(self, reviewer, basic_context):
        """Test consistency check."""
        check = reviewer._check_consistency(GOOD_OUTPUT, basic_context)
        assert isinstance(check, CheckItem)
        assert check.check_name == "Consistency"

    def test_readability_check_good(self, reviewer):
        """Test readability check on good output."""
        check = reviewer._check_readability(GOOD_OUTPUT)
        assert check.passed is True

    def test_readability_check_short(self, reviewer):
        """Test readability check on very short output."""
        check = reviewer._check_readability(SHORT_OUTPUT)
        assert check.passed is False

    def test_verifier_signoff_no_report(self, reviewer):
        """Test verifier signoff without report."""
        check = reviewer._check_verifier_signoff(None)
        assert check.passed is True  # No verifier = assume pass

    def test_verifier_signoff_pass(self, reviewer):
        """Test verifier signoff with passing report."""
        report = {"verdict": "PASS", "overall_reliability": 0.85}
        check = reviewer._check_verifier_signoff(report)
        assert check.passed is True

    def test_verifier_signoff_fail(self, reviewer):
        """Test verifier signoff with failing report."""
        report = {"verdict": "FAIL", "overall_reliability": 0.3}
        check = reviewer._check_verifier_signoff(report)
        assert check.passed is False


class TestVerdictMatrix:
    """Tests for verdict matrix logic."""

    def test_both_pass_proceeds(self, reviewer):
        """Test both PASS -> PROCEED_TO_FORMATTER."""
        action = reviewer._apply_verdict_matrix(Verdict.PASS, Verdict.PASS)
        assert action == "PROCEED_TO_FORMATTER"

    def test_verifier_pass_critic_fail(self, reviewer):
        """Test Verifier PASS, Critic FAIL -> EXECUTOR_REVISE."""
        action = reviewer._apply_verdict_matrix(Verdict.PASS, Verdict.FAIL)
        assert action == "EXECUTOR_REVISE"

    def test_verifier_fail_critic_pass(self, reviewer):
        """Test Verifier FAIL, Critic PASS -> RESEARCHER_REVERIFY."""
        action = reviewer._apply_verdict_matrix(Verdict.FAIL, Verdict.PASS)
        assert action == "RESEARCHER_REVERIFY"

    def test_both_fail_regenerates(self, reviewer):
        """Test both FAIL -> FULL_REGENERATION."""
        action = reviewer._apply_verdict_matrix(Verdict.FAIL, Verdict.FAIL)
        assert action == "FULL_REGENERATION"


class TestCriticalFailures:
    """Tests for critical failure detection."""

    def test_detects_security_failures(self, reviewer):
        """Test security failure detection."""
        failures = reviewer._check_critical_failures(
            SECURITY_OUTPUT, None, None, None
        )
        assert len(failures["security"]) > 0

    def test_detects_hallucination_from_verifier(self, reviewer):
        """Test hallucination detection from verifier report."""
        verifier_report = {"fabricated_claims": 3}
        failures = reviewer._check_critical_failures(
            "Clean output", verifier_report, None, None
        )
        assert len(failures["hallucination"]) > 0

    def test_no_failures_clean_output(self, reviewer):
        """Test no failures for clean output."""
        failures = reviewer._check_critical_failures(
            "This is clean output", None, None, None
        )
        total = sum(len(v) for v in failures.values())
        assert total == 0


class TestArbitration:
    """Tests for arbitration logic."""

    def test_no_arbitration_low_tier(self, reviewer, basic_context):
        """Test no arbitration for Tier 2."""
        needed, input_data = reviewer._check_arbitration_needed(
            Verdict.PASS, Verdict.PASS, Verdict.FAIL, basic_context
        )
        assert needed is False

    def test_arbitration_tier4_disagreement(self, reviewer, tier4_context):
        """Test arbitration needed for Tier 4 with disagreement."""
        needed, input_data = reviewer._check_arbitration_needed(
            Verdict.PASS, Verdict.PASS, Verdict.FAIL, tier4_context
        )
        assert needed is True
        assert input_data is not None

    def test_no_arbitration_tier4_agreement(self, reviewer, tier4_context):
        """Test no arbitration for Tier 4 with agreement."""
        needed, input_data = reviewer._check_arbitration_needed(
            Verdict.PASS, Verdict.PASS, Verdict.PASS, tier4_context
        )
        assert needed is False


class TestRevisionInstructions:
    """Tests for revision instruction generation."""

    def test_fail_generates_revisions(self, reviewer, basic_context):
        """Test FAIL verdict generates revision instructions."""
        verdict = reviewer.review(SECURITY_OUTPUT, basic_context)
        if verdict.verdict == Verdict.FAIL:
            assert len(verdict.revision_instructions) > 0

    def test_max_revisions_reached(self, reviewer, basic_context):
        """Test max revisions flag."""
        basic_context.revision_count = 3
        basic_context.max_revisions = 3
        verdict = reviewer.review(GOOD_OUTPUT, basic_context)
        assert verdict.can_revise is False


class TestConvenienceFunction:
    """Tests for create_reviewer convenience function."""

    def test_create_reviewer(self):
        """Test convenience function creates a ReviewerAgent."""
        agent = create_reviewer(system_prompt_path="nonexistent.md")
        assert isinstance(agent, ReviewerAgent)
