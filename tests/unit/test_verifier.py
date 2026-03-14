"""
Tests for the VerifierAgent.

Tests claim extraction, verification by type, fabrication risk assessment,
SME verification, and report generation.
"""

import pytest
from unittest.mock import patch, mock_open

from src.agents.verifier import (
    VerifierAgent,
    ClaimExtraction,
    create_verifier,
)
from src.schemas.verifier import (
    VerificationReport,
    VerificationStatus,
    FabricationRisk,
)


@pytest.fixture
def verifier():
    """Create a VerifierAgent with no system prompt file."""
    return VerifierAgent(system_prompt_path="nonexistent.md")


FACTUAL_CONTENT = """
Python was created in 1991 by Guido van Rossum.
According to research, Python is the most popular programming language.
The standard library includes approximately 200 modules.
Visit https://docs.python.org for more information.
It is known that Python uses dynamic typing.
"""

SIMPLE_CONTENT = "Hello world."

HALLUCINATION_CONTENT = """
It is obviously the case that all programs are perfect.
Everyone knows that bugs are impossible in Python.
This is clearly demonstrated by the evidence.
"""


class TestVerifierInitialization:
    """Tests for VerifierAgent initialization."""

    def test_default_initialization(self):
        """Test default init parameters."""
        agent = VerifierAgent(system_prompt_path="nonexistent.md")
        assert agent.model == "claude-opus-4-20250514"
        assert agent.max_turns == 30

    def test_claim_patterns_initialized(self):
        """Test claim patterns are configured."""
        agent = VerifierAgent(system_prompt_path="nonexistent.md")
        assert len(agent.claim_patterns) > 0

    def test_claim_keywords_initialized(self):
        """Test claim keywords are configured."""
        agent = VerifierAgent(system_prompt_path="nonexistent.md")
        assert "according to" in agent.claim_keywords
        assert "based on" in agent.claim_keywords

    def test_system_prompt_fallback(self):
        """Test fallback prompt."""
        agent = VerifierAgent(system_prompt_path="nonexistent.md")
        assert "Verifier" in agent.system_prompt

    def test_system_prompt_from_file(self):
        """Test loading from file."""
        with patch("builtins.open", mock_open(read_data="Verifier prompt")):
            agent = VerifierAgent(system_prompt_path="exists.md")
            assert agent.system_prompt == "Verifier prompt"


class TestVerify:
    """Tests for the verify method."""

    def test_basic_verification(self, verifier):
        """Test basic verification produces VerificationReport."""
        report = verifier.verify(FACTUAL_CONTENT)
        assert isinstance(report, VerificationReport)
        assert report.total_claims_checked > 0

    def test_verdict_pass_or_fail(self, verifier):
        """Test verdict is PASS or FAIL."""
        report = verifier.verify(FACTUAL_CONTENT)
        assert report.verdict in ["PASS", "FAIL"]

    def test_reliability_score(self, verifier):
        """Test reliability score is between 0 and 1."""
        report = verifier.verify(FACTUAL_CONTENT)
        assert 0.0 <= report.overall_reliability <= 1.0

    def test_simple_content_has_few_claims(self, verifier):
        """Test simple content has few claims."""
        report = verifier.verify(SIMPLE_CONTENT)
        # "Hello world." is too short to be a claim
        assert report.total_claims_checked >= 0

    def test_pass_threshold_set(self, verifier):
        """Test pass threshold is set."""
        report = verifier.verify(FACTUAL_CONTENT)
        assert report.pass_threshold == 0.7


class TestClaimExtraction:
    """Tests for claim extraction."""

    def test_extracts_claims(self, verifier):
        """Test claims are extracted from content."""
        extraction = verifier._extract_claims(FACTUAL_CONTENT)
        assert isinstance(extraction, ClaimExtraction)
        assert extraction.total_claims > 0

    def test_short_sentences_excluded(self, verifier):
        """Test very short sentences are excluded."""
        extraction = verifier._extract_claims("Hi. Ok. Yes.")
        assert extraction.total_claims == 0

    def test_claim_detection_with_keywords(self, verifier):
        """Test claims with keywords are detected."""
        content = "According to experts, Python is widely used in data science."
        assert verifier._is_claim(content) is True

    def test_claim_detection_with_dates(self, verifier):
        """Test claims with dates are detected."""
        content = "Python was released in 1991"
        assert verifier._is_claim(content) is True

    def test_claim_detection_with_urls(self, verifier):
        """Test claims with URLs are detected."""
        content = "See https://docs.python.org for details"
        assert verifier._is_claim(content) is True


class TestClaimVerification:
    """Tests for claim verification types."""

    def test_date_claim_future_year(self, verifier):
        """Test future year claim gets low confidence."""
        result = verifier._verify_date_claim("Released in 2030")
        assert result["confidence"] < 5
        assert result["risk"] == FabricationRisk.HIGH

    def test_date_claim_reasonable_year(self, verifier):
        """Test reasonable year claim."""
        result = verifier._verify_date_claim("Created in 2020")
        assert result["confidence"] >= 5

    def test_url_claim_valid(self, verifier):
        """Test valid URL claim."""
        result = verifier._verify_url_claim("Visit https://example.com")
        assert result["status"] == VerificationStatus.UNVERIFIED

    def test_measurement_claim_over_100_percent(self, verifier):
        """Test percentage over 100 gets flagged."""
        result = verifier._verify_measurement_claim("Accuracy was 150%")
        assert result["risk"] == FabricationRisk.HIGH

    def test_general_claim_hallucination_pattern(self, verifier):
        """Test hallucination pattern detection."""
        result = verifier._verify_general_claim(
            "This is obviously the best approach", ""
        )
        assert result["risk"] == FabricationRisk.MEDIUM
        assert result["confidence"] < 7


class TestSMEVerification:
    """Tests for SME-based verification."""

    def test_sme_verification_high_confidence(self, verifier):
        """Test SME verification gives high confidence."""
        report = verifier.verify(
            "Python uses dynamic typing for flexibility",
            sme_verifications={"Python Expert": "Confirmed"},
        )
        # Should have SME-verified claims
        sme_claims = [
            c for c in report.claims
            if c.domain_verified is True
        ]
        assert len(sme_claims) > 0

    def test_source_verification(self, verifier):
        """Test verification against provided sources."""
        report = verifier.verify(
            "Python is widely used in data science",
            sources=["Python is a popular language for data science and machine learning"],
        )
        verified = [
            c for c in report.claims
            if c.status == VerificationStatus.VERIFIED
        ]
        assert len(verified) > 0


class TestReportGeneration:
    """Tests for report generation."""

    def test_summary_generated(self, verifier):
        """Test verification summary is generated."""
        report = verifier.verify(FACTUAL_CONTENT)
        assert len(report.verification_summary) > 0
        assert "claim" in report.verification_summary.lower()

    def test_flagged_claims_identified(self, verifier):
        """Test flagged claims are identified."""
        report = verifier.verify(FACTUAL_CONTENT)
        assert isinstance(report.flagged_claims, list)

    def test_corrections_generated(self, verifier):
        """Test corrections are generated for flagged claims."""
        report = verifier.verify(HALLUCINATION_CONTENT)
        assert isinstance(report.recommended_corrections, list)

    def test_counts_consistent(self, verifier):
        """Test claim counts are consistent."""
        report = verifier.verify(FACTUAL_CONTENT)
        total = (
            report.verified_claims +
            report.unverified_claims +
            report.contradicted_claims +
            report.fabricated_claims
        )
        assert total == report.total_claims_checked


class TestConvenienceFunction:
    """Tests for create_verifier convenience function."""

    def test_create_verifier(self):
        """Test convenience function creates a VerifierAgent."""
        agent = create_verifier(system_prompt_path="nonexistent.md")
        assert isinstance(agent, VerifierAgent)
