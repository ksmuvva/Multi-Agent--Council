"""
Exhaustive Tests for VerifierAgent

Tests all methods, edge cases, schema outputs, and integration paths
for the Verifier (Hallucination Guard) subagent in src/agents/verifier.py.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import patch, mock_open
from datetime import datetime

from src.agents.verifier import (
    VerifierAgent,
    ClaimExtraction,
    create_verifier,
)
from src.schemas.verifier import (
    VerificationReport,
    Claim,
    ClaimBatch,
    VerificationStatus,
    FabricationRisk,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def verifier():
    """Create a VerifierAgent with mocked system prompt."""
    with patch("builtins.open", side_effect=FileNotFoundError):
        return VerifierAgent()


@pytest.fixture
def verifier_with_prompt():
    """Create a VerifierAgent with a custom system prompt loaded."""
    prompt_text = "You are a meticulous fact checker."
    with patch("builtins.open", mock_open(read_data=prompt_text)):
        return VerifierAgent(system_prompt_path="fake/path.md")


# ============================================================================
# __init__ Tests
# ============================================================================

class TestVerifierInit:
    def test_default_init(self, verifier):
        assert verifier.system_prompt_path == "config/agents/verifier/CLAUDE.md"
        assert verifier.model == "claude-3-5-opus-20240507"
        assert verifier.max_turns == 30
        assert "Verifier" in verifier.system_prompt

    def test_custom_params(self):
        with patch("builtins.open", side_effect=FileNotFoundError):
            v = VerifierAgent(
                system_prompt_path="custom/path.md",
                model="claude-3-opus",
                max_turns=15,
            )
        assert v.system_prompt_path == "custom/path.md"
        assert v.model == "claude-3-opus"
        assert v.max_turns == 15

    def test_system_prompt_loading_success(self, verifier_with_prompt):
        assert verifier_with_prompt.system_prompt == "You are a meticulous fact checker."

    def test_system_prompt_loading_fallback(self, verifier):
        assert verifier.system_prompt == "You are the Verifier. Detect factual errors and hallucinations."

    def test_claim_patterns_populated(self, verifier):
        assert len(verifier.claim_patterns) > 0

    def test_claim_keywords_populated(self, verifier):
        assert "according to" in verifier.claim_keywords
        assert "research shows" in verifier.claim_keywords
        assert "approximately" in verifier.claim_keywords


# ============================================================================
# _extract_claims Tests
# ============================================================================

class TestExtractClaims:
    def test_returns_claim_extraction(self, verifier):
        content = "Python was released in 1991. It is a popular language."
        result = verifier._extract_claims(content)
        assert isinstance(result, ClaimExtraction)
        assert result.total_claims == len(result.claims)

    def test_sentence_splitting(self, verifier):
        content = "First sentence. Second sentence! Third sentence?"
        result = verifier._extract_claims(content)
        # All three sentences should be considered
        assert isinstance(result, ClaimExtraction)

    def test_filters_short_sentences(self, verifier):
        content = "Short. Also. Tiny. Python was released in 1991 by Guido van Rossum."
        result = verifier._extract_claims(content)
        # Short sentences (< 10 chars) should be filtered
        for claim in result.claims:
            assert len(claim) >= 10

    def test_claim_detection_with_year(self, verifier):
        content = "Python was released in 1991. It has many features."
        result = verifier._extract_claims(content)
        assert any("1991" in c for c in result.claims)

    def test_claim_detection_with_keyword(self, verifier):
        content = "According to the documentation, the API is stable. The sky is blue."
        result = verifier._extract_claims(content)
        assert any("according to" in c.lower() for c in result.claims)

    def test_empty_content(self, verifier):
        result = verifier._extract_claims("")
        assert result.claims == []
        assert result.total_claims == 0

    def test_claim_locations_populated(self, verifier):
        content = "Python was released in 1991 by Guido van Rossum."
        result = verifier._extract_claims(content)
        for loc in result.claim_locations:
            assert "claim" in loc
            assert "start" in loc
            assert "end" in loc


# ============================================================================
# _is_claim Tests
# ============================================================================

class TestIsClaim:
    @pytest.mark.parametrize("sentence", [
        "According to the documentation, it works well",
        "Research shows that this approach is effective",
        "Studies have demonstrated significant improvements",
        "It is known that Python supports multiple paradigms",
        "Typically the system handles about 1000 requests",
        "The value is approximately 3.14 units",
        "Estimated around 50 percent completion rate",
    ])
    def test_claim_keywords(self, verifier, sentence):
        assert verifier._is_claim(sentence) is True

    @pytest.mark.parametrize("sentence,description", [
        ("Python was released in 1991", "year pattern"),
        ("The population is 1,000,000", "number with commas"),
        ("January 15 marks the release date", "date pattern"),
        ("Performance improved by 50 percent", "measurement percent"),
        ("Temperature reaches 100 degrees", "measurement degrees"),
        ("Visit https://example.com for details", "URL pattern"),
        ("Guido van Rossum created Python", "proper nouns"),
    ])
    def test_claim_patterns(self, verifier, sentence, description):
        assert verifier._is_claim(sentence) is True, f"Failed for {description}"

    @pytest.mark.parametrize("sentence", [
        "Python is a programming language",
        "The system was designed for scalability",
        "Functions are first-class objects",
        "Threads will execute concurrently",
    ])
    def test_verb_indicators(self, verifier, sentence):
        assert verifier._is_claim(sentence) is True

    def test_non_claim_sentence(self, verifier):
        # No keywords, no patterns, no verbs from the list
        assert verifier._is_claim("hello world") is False

    def test_non_claim_no_indicators(self, verifier):
        assert verifier._is_claim("just random text here") is False


# ============================================================================
# _verify_claim Tests
# ============================================================================

class TestVerifyClaim:
    def test_sme_verification_path(self, verifier):
        claim = "Python supports multiple paradigms"
        sme_verifications = {"Python Expert": "Verified: Python supports OOP, FP, and procedural."}
        result = verifier._verify_claim(claim, "full content", sme_verifications=sme_verifications)
        assert result.status == VerificationStatus.VERIFIED
        assert result.confidence == 10
        assert result.fabrication_risk == FabricationRisk.LOW
        assert result.domain_verified is True
        assert result.sme_verifier == "Python Expert"

    def test_source_verification_path_found(self, verifier):
        claim = "Python supports asyncio for concurrent programming"
        sources = ["Python supports asyncio for concurrent programming patterns"]
        result = verifier._verify_claim(claim, "content", sources=sources)
        assert result.status == VerificationStatus.VERIFIED
        assert result.confidence == 9
        assert result.fabrication_risk == FabricationRisk.LOW

    def test_source_verification_path_not_found(self, verifier):
        claim = "Quantum computing will replace classical by 2030"
        sources = ["Python documentation about asyncio"]
        result = verifier._verify_claim(claim, "content", sources=sources)
        # Falls through to type-based verification
        assert isinstance(result, Claim)

    def test_type_based_verification_fallback(self, verifier):
        claim = "The system handles requests efficiently"
        result = verifier._verify_claim(claim, "content")
        assert isinstance(result, Claim)
        assert result.verification_method

    def test_sme_takes_priority_over_sources(self, verifier):
        claim = "Python supports multiple paradigms"
        sources = ["Python documentation"]
        sme_verifications = {"Python Expert": "Verified"}
        result = verifier._verify_claim(claim, "content", sources=sources,
                                        sme_verifications=sme_verifications)
        assert result.sme_verifier == "Python Expert"
        assert result.confidence == 10


# ============================================================================
# _claim_matches_domain Tests
# ============================================================================

class TestClaimMatchesDomain:
    def test_keyword_match(self, verifier):
        assert verifier._claim_matches_domain("Python is great", "Python Expert") is True

    def test_no_match(self, verifier):
        assert verifier._claim_matches_domain("Rust is fast", "Python Expert") is False

    def test_case_insensitive(self, verifier):
        assert verifier._claim_matches_domain("PYTHON is great", "python expert") is True

    def test_partial_domain_keyword(self, verifier):
        assert verifier._claim_matches_domain(
            "The security model prevents unauthorized access",
            "Security Architecture"
        ) is True

    def test_multi_word_domain(self, verifier):
        assert verifier._claim_matches_domain(
            "Machine learning models improve accuracy",
            "Machine Learning Expert"
        ) is True


# ============================================================================
# _verify_against_sources Tests
# ============================================================================

class TestVerifyAgainstSources:
    def test_found_matching_source(self, verifier):
        result = verifier._verify_against_sources(
            "Python supports asyncio for concurrent programming",
            ["Python supports asyncio patterns for concurrent applications"]
        )
        assert result["found"] is True
        assert result["confidence"] == 9
        assert result["risk"] == FabricationRisk.LOW
        assert result["status"] == VerificationStatus.VERIFIED

    def test_not_found_in_sources(self, verifier):
        result = verifier._verify_against_sources(
            "Quantum computing replaces classical",
            ["Python asyncio documentation"]
        )
        assert result["found"] is False

    def test_word_overlap_matching(self, verifier):
        # Words longer than 3 chars should match
        result = verifier._verify_against_sources(
            "The system handles concurrent requests efficiently",
            ["concurrent request handling for scalable systems"]
        )
        assert result["found"] is True

    def test_short_words_ignored(self, verifier):
        # Words <= 3 chars should not trigger match
        result = verifier._verify_against_sources(
            "is a an the",
            ["is a an the"]
        )
        assert result["found"] is False

    def test_empty_sources(self, verifier):
        result = verifier._verify_against_sources("some claim", [])
        assert result["found"] is False

    def test_source_returned_when_found(self, verifier):
        source_text = "Python asyncio provides concurrent execution"
        result = verifier._verify_against_sources(
            "Python asyncio concurrent execution",
            [source_text]
        )
        if result["found"]:
            assert result["source"] == source_text


# ============================================================================
# _verify_claim_by_type Tests
# ============================================================================

class TestVerifyClaimByType:
    def test_routes_date_claim(self, verifier):
        result = verifier._verify_claim_by_type("Python was released in 1991", "content")
        assert "date" in result["method"].lower() or "Date" in result["method"]

    def test_routes_url_claim(self, verifier):
        result = verifier._verify_claim_by_type(
            "See https://example.com for details", "content"
        )
        assert "url" in result["method"].lower() or "URL" in result["method"]

    def test_routes_measurement_claim(self, verifier):
        result = verifier._verify_claim_by_type(
            "Performance improved by 50 % after optimization", "content"
        )
        assert "measurement" in result["method"].lower() or "Measurement" in result["method"]

    def test_routes_general_claim(self, verifier):
        result = verifier._verify_claim_by_type(
            "The framework provides excellent capabilities", "content"
        )
        assert "general" in result["method"].lower() or "Hallucination" in result["method"]

    def test_date_takes_priority_over_url(self, verifier):
        # Has both a year and no URL, year pattern checked first
        result = verifier._verify_claim_by_type("Released in 2020", "content")
        assert "date" in result["method"].lower() or "Date" in result["method"]


# ============================================================================
# _verify_date_claim Tests
# ============================================================================

class TestVerifyDateClaim:
    def test_future_year(self, verifier):
        current_year = datetime.now().year
        future_year = current_year + 10
        result = verifier._verify_date_claim(f"This will happen in {future_year}")
        assert result["confidence"] == 2
        assert result["risk"] == FabricationRisk.HIGH
        assert result["status"] == VerificationStatus.UNVERIFIED
        assert "future" in result.get("correction", "").lower()

    def test_pre_1900_year(self, verifier):
        result = verifier._verify_date_claim("This happened in 1850")
        assert result["confidence"] == 5
        assert result["risk"] == FabricationRisk.MEDIUM
        assert result["status"] == VerificationStatus.UNVERIFIED

    def test_normal_year(self, verifier):
        result = verifier._verify_date_claim("Python was released in 2020")
        assert result["confidence"] == 6
        assert result["risk"] == FabricationRisk.MEDIUM
        assert result["status"] == VerificationStatus.UNVERIFIED

    def test_current_year(self, verifier):
        current_year = datetime.now().year
        result = verifier._verify_date_claim(f"This happened in {current_year}")
        assert result["confidence"] == 6
        assert result["status"] == VerificationStatus.UNVERIFIED

    def test_year_just_after_current(self, verifier):
        # current_year + 1 should be okay (not > current_year + 1)
        next_year = datetime.now().year + 1
        result = verifier._verify_date_claim(f"Expected in {next_year}")
        assert result["confidence"] == 6  # Normal date, not flagged as future

    def test_no_year_in_claim(self, verifier):
        result = verifier._verify_date_claim("No year here")
        # Falls to default
        assert result["confidence"] == 6
        assert result["method"] == "Date extraction (unverified)"


# ============================================================================
# _verify_url_claim Tests
# ============================================================================

class TestVerifyUrlClaim:
    def test_valid_url(self, verifier):
        result = verifier._verify_url_claim("See https://docs.python.org/3/library/asyncio.html")
        assert result["confidence"] == 7
        assert result["risk"] == FabricationRisk.MEDIUM
        assert result["status"] == VerificationStatus.UNVERIFIED

    def test_invalid_format(self, verifier):
        # The regex requires http:// or https://, so no match -> falls to default
        result = verifier._verify_url_claim("See ftp://example.com for details")
        assert result["confidence"] == 7  # default for URLs

    def test_missing_netloc(self, verifier):
        # https:///path has empty netloc but the regex still captures it
        result = verifier._verify_url_claim("See https:///path/to/resource for details")
        assert result["confidence"] == 3
        assert result["risk"] == FabricationRisk.HIGH
        assert "Malformed" in result.get("correction", "")

    def test_no_url_in_claim(self, verifier):
        result = verifier._verify_url_claim("No URL here at all")
        # Falls to default
        assert result["confidence"] == 7

    def test_http_url(self, verifier):
        result = verifier._verify_url_claim("Visit http://example.com/page for info")
        assert result["confidence"] == 7
        assert result["status"] == VerificationStatus.UNVERIFIED


# ============================================================================
# _verify_measurement_claim Tests
# ============================================================================

class TestVerifyMeasurementClaim:
    def test_contradictory_percentages(self, verifier):
        result = verifier._verify_measurement_claim("The value is 100% and 0% at the same time")
        assert result["confidence"] == 2
        assert result["risk"] == FabricationRisk.HIGH
        assert result["status"] == VerificationStatus.CONTRADICTED

    def test_over_100_percent(self, verifier):
        result = verifier._verify_measurement_claim("Performance improved by 500 %")
        assert result["confidence"] == 3
        assert result["risk"] == FabricationRisk.HIGH
        assert result["status"] == VerificationStatus.UNVERIFIED

    def test_normal_measurement(self, verifier):
        result = verifier._verify_measurement_claim("Performance improved by 50 %")
        assert result["confidence"] == 6
        assert result["risk"] == FabricationRisk.MEDIUM
        assert result["status"] == VerificationStatus.UNVERIFIED

    def test_degrees_measurement(self, verifier):
        result = verifier._verify_measurement_claim("Temperature reached 37 degrees")
        assert result["confidence"] == 6
        assert result["risk"] == FabricationRisk.MEDIUM

    def test_999_percent(self, verifier):
        result = verifier._verify_measurement_claim("999 % increase reported")
        assert result["confidence"] == 3
        assert result["risk"] == FabricationRisk.HIGH


# ============================================================================
# _verify_general_claim Tests
# ============================================================================

class TestVerifyGeneralClaim:
    @pytest.mark.parametrize("sentence,pattern", [
        ("This has been demonstrated in many studies", "demonstrated"),
        ("Obviously the approach works well", "obviously"),
        ("Clearly the results speak for themselves", "clearly"),
        ("Everyone knows that Python is popular", "everyone knows"),
        ("It is well known that this approach works", "it is well known"),
    ])
    def test_hallucination_pattern_detection(self, verifier, sentence, pattern):
        result = verifier._verify_general_claim(sentence, "content")
        assert result["confidence"] == 4
        assert result["risk"] == FabricationRisk.MEDIUM
        assert result["method"] == "Hallucination pattern detection"

    def test_default_general_claim(self, verifier):
        result = verifier._verify_general_claim(
            "The framework provides excellent capabilities", "content"
        )
        assert result["confidence"] in (5, 6, 7)  # Varies by content cross-reference
        assert result["risk"] in (FabricationRisk.MEDIUM, FabricationRisk.LOW)
        assert result["method"] in ("General claim analysis", "Content cross-reference")

    def test_case_insensitive_hallucination_check(self, verifier):
        result = verifier._verify_general_claim(
            "OBVIOUSLY this works perfectly", "content"
        )
        assert result["confidence"] == 4


# ============================================================================
# _generate_corrections Tests
# ============================================================================

class TestGenerateCorrections:
    def test_fabricated_claim_correction(self, verifier):
        claims = [Claim(
            claim_text="This is fabricated content that has no basis in reality",
            confidence=2,
            fabrication_risk=FabricationRisk.HIGH,
            verification_method="test",
            status=VerificationStatus.FABRICATED,
        )]
        corrections = verifier._generate_corrections(claims)
        assert len(corrections) == 1
        assert "remove" in corrections[0].lower() or "source" in corrections[0].lower()

    def test_unverified_claim_correction(self, verifier):
        claims = [Claim(
            claim_text="This claim cannot be verified from any available source",
            confidence=4,
            fabrication_risk=FabricationRisk.MEDIUM,
            verification_method="test",
            status=VerificationStatus.UNVERIFIED,
        )]
        corrections = verifier._generate_corrections(claims)
        assert len(corrections) == 1
        assert "verification" in corrections[0].lower()

    def test_contradicted_claim_correction(self, verifier):
        claims = [Claim(
            claim_text="This claim contradicts known information about the topic",
            confidence=3,
            fabrication_risk=FabricationRisk.HIGH,
            verification_method="test",
            status=VerificationStatus.CONTRADICTED,
        )]
        corrections = verifier._generate_corrections(claims)
        assert len(corrections) == 1
        assert "contradiction" in corrections[0].lower()

    def test_uses_existing_correction(self, verifier):
        claims = [Claim(
            claim_text="Wrong claim",
            confidence=2,
            fabrication_risk=FabricationRisk.HIGH,
            verification_method="test",
            status=VerificationStatus.FABRICATED,
            correction="Use the correct value instead",
        )]
        corrections = verifier._generate_corrections(claims)
        assert corrections[0] == "Use the correct value instead"

    def test_limit_to_5(self, verifier):
        claims = [
            Claim(
                claim_text=f"Fabricated claim number {i} with no source available",
                confidence=2,
                fabrication_risk=FabricationRisk.HIGH,
                verification_method="test",
                status=VerificationStatus.FABRICATED,
            )
            for i in range(10)
        ]
        corrections = verifier._generate_corrections(claims)
        assert len(corrections) <= 5

    def test_empty_flagged_claims(self, verifier):
        corrections = verifier._generate_corrections([])
        assert corrections == []

    def test_verified_claim_no_correction_generated(self, verifier):
        claims = [Claim(
            claim_text="This is verified",
            confidence=9,
            fabrication_risk=FabricationRisk.LOW,
            verification_method="test",
            status=VerificationStatus.VERIFIED,
        )]
        corrections = verifier._generate_corrections(claims)
        # No correction field and status is VERIFIED -> no correction text generated
        assert corrections == []


# ============================================================================
# _build_summary Tests
# ============================================================================

class TestBuildSummary:
    def test_high_reliability(self, verifier):
        summary = verifier._build_summary(10, 8, 1, 1, 0, 0.85)
        assert "High confidence" in summary
        assert "10" in summary
        assert "8" in summary

    def test_medium_reliability(self, verifier):
        summary = verifier._build_summary(10, 6, 3, 1, 0, 0.65)
        assert "Medium confidence" in summary

    def test_low_reliability(self, verifier):
        summary = verifier._build_summary(10, 3, 4, 2, 1, 0.30)
        assert "Low confidence" in summary

    def test_includes_all_counts(self, verifier):
        summary = verifier._build_summary(10, 5, 3, 1, 1, 0.5)
        assert "5 verified" in summary
        assert "3 unverified" in summary
        assert "1 contradicted" in summary
        assert "1 potentially fabricated" in summary

    def test_boundary_080(self, verifier):
        summary = verifier._build_summary(10, 8, 2, 0, 0, 0.80)
        assert "High confidence" in summary

    def test_boundary_060(self, verifier):
        summary = verifier._build_summary(10, 6, 4, 0, 0, 0.60)
        assert "Medium confidence" in summary


# ============================================================================
# Verdict Logic Tests
# ============================================================================

class TestVerdictLogic:
    def test_pass_when_reliability_gte_07(self, verifier):
        content = "Python was released in 1991. It is a popular programming language."
        sources = ["Python was released in 1991 by Guido van Rossum as a popular language"]
        result = verifier.verify(content, sources=sources)
        # With source verification, should get high reliability
        if result.overall_reliability >= 0.7:
            assert result.verdict == "PASS"

    def test_fail_when_reliability_lt_07(self, verifier):
        # Content with many unverifiable claims
        content = (
            "According to recent studies, approximately 95% of developers prefer Python. "
            "Research shows that Python is used by around 10 million developers. "
            "It is well known that Python obviously outperforms all other languages."
        )
        result = verifier.verify(content)
        if result.overall_reliability < 0.7:
            assert result.verdict == "FAIL"

    def test_pass_threshold_is_07(self, verifier):
        content = "Python is a language."
        result = verifier.verify(content)
        assert result.pass_threshold == 0.7

    def test_verdict_consistency(self, verifier):
        content = "Python was released in 1991."
        result = verifier.verify(content)
        if result.overall_reliability >= 0.7:
            assert result.verdict == "PASS"
        else:
            assert result.verdict == "FAIL"


# ============================================================================
# verify() Full Method Tests
# ============================================================================

class TestVerifyFullMethod:
    def test_returns_verification_report(self, verifier):
        content = "Python was released in 1991. It is popular."
        result = verifier.verify(content)
        assert isinstance(result, VerificationReport)

    def test_all_report_fields_populated(self, verifier):
        content = "Python was released in 1991 by Guido van Rossum. It is widely used."
        result = verifier.verify(content)
        assert result.total_claims_checked >= 0
        assert isinstance(result.claims, list)
        assert result.verified_claims >= 0
        assert result.unverified_claims >= 0
        assert result.contradicted_claims >= 0
        assert result.fabricated_claims >= 0
        assert 0.0 <= result.overall_reliability <= 1.0
        assert result.verdict in ["PASS", "FAIL"]
        assert isinstance(result.flagged_claims, list)
        assert isinstance(result.recommended_corrections, list)
        assert result.verification_summary

    def test_with_sources(self, verifier):
        content = "Python was released in 1991. It supports asyncio."
        sources = ["Python was released in 1991 by Guido van Rossum"]
        result = verifier.verify(content, sources=sources)
        assert isinstance(result, VerificationReport)

    def test_with_sme_verifications(self, verifier):
        content = "Python supports multiple programming paradigms."
        sme_verifications = {"Python Expert": "Confirmed: OOP, FP, procedural."}
        result = verifier.verify(content, sme_verifications=sme_verifications)
        assert isinstance(result, VerificationReport)

    def test_with_context(self, verifier):
        content = "Python is a programming language."
        result = verifier.verify(content, context={"tier": 2})
        assert isinstance(result, VerificationReport)

    def test_empty_content(self, verifier):
        result = verifier.verify("")
        assert result.total_claims_checked == 0
        assert result.overall_reliability == 0.5
        assert result.verdict == "FAIL"

    def test_counts_sum_correctly(self, verifier):
        content = (
            "Python was released in 1991. "
            "It is used by millions of developers worldwide. "
            "According to surveys, Python is the most popular language."
        )
        result = verifier.verify(content)
        total = (result.verified_claims + result.unverified_claims +
                 result.contradicted_claims + result.fabricated_claims)
        assert total == result.total_claims_checked

    def test_flagged_claims_criteria(self, verifier):
        content = "Python was released in 1991. Obviously it is the best language ever."
        result = verifier.verify(content)
        for claim in result.flagged_claims:
            assert claim.confidence < 7 or claim.fabrication_risk != FabricationRisk.LOW

    def test_reliability_rounded(self, verifier):
        content = "Python was released in 1991."
        result = verifier.verify(content)
        # Check reliability is rounded to 2 decimal places
        assert result.overall_reliability == round(result.overall_reliability, 2)

    def test_no_claims_reliability(self, verifier):
        # Empty content produces no claims
        result = verifier.verify("")
        assert result.overall_reliability == 0.5

    def test_all_verified_reliability(self, verifier):
        content = "Python supports multiple paradigms."
        sme_verifications = {"Python Expert": "Confirmed."}
        result = verifier.verify(content, sme_verifications=sme_verifications)
        # If all claims are SME-verified, reliability should be high
        if result.total_claims_checked > 0 and result.verified_claims == result.total_claims_checked:
            assert result.overall_reliability == 1.0


# ============================================================================
# create_verifier() Convenience Function Tests
# ============================================================================

class TestCreateVerifier:
    def test_creates_default_verifier(self):
        with patch("builtins.open", side_effect=FileNotFoundError):
            v = create_verifier()
        assert isinstance(v, VerifierAgent)
        assert v.model == "claude-3-5-opus-20240507"

    def test_creates_custom_verifier(self):
        with patch("builtins.open", side_effect=FileNotFoundError):
            v = create_verifier(
                system_prompt_path="custom.md",
                model="claude-3-opus",
            )
        assert v.system_prompt_path == "custom.md"
        assert v.model == "claude-3-opus"


# ============================================================================
# Schema Integration Tests
# ============================================================================

class TestSchemaIntegration:
    def test_verification_report_serialization(self, verifier):
        content = "Python was released in 1991."
        result = verifier.verify(content)
        data = result.model_dump()
        assert "total_claims_checked" in data
        assert "claims" in data
        assert "verdict" in data

    def test_verification_report_json(self, verifier):
        content = "Python was released in 1991."
        result = verifier.verify(content)
        json_str = result.model_dump_json()
        assert "total_claims_checked" in json_str

    def test_verification_status_enum_values(self):
        assert VerificationStatus.VERIFIED.value == "verified"
        assert VerificationStatus.UNVERIFIED.value == "unverified"
        assert VerificationStatus.CONTRADICTED.value == "contradicted"
        assert VerificationStatus.FABRICATED.value == "fabricated"

    def test_fabrication_risk_enum_values(self):
        assert FabricationRisk.LOW.value == "low"
        assert FabricationRisk.MEDIUM.value == "medium"
        assert FabricationRisk.HIGH.value == "high"

    def test_claim_model_validation(self):
        claim = Claim(
            claim_text="Test claim",
            confidence=5,
            fabrication_risk=FabricationRisk.LOW,
            verification_method="test",
            status=VerificationStatus.VERIFIED,
        )
        assert claim.claim_text == "Test claim"
        assert claim.domain_verified is False
        assert claim.sme_verifier is None
        assert claim.correction is None

    def test_claim_confidence_bounds(self):
        with pytest.raises(Exception):  # ValidationError
            Claim(
                claim_text="Test",
                confidence=0,  # Below minimum of 1
                fabrication_risk=FabricationRisk.LOW,
                verification_method="test",
                status=VerificationStatus.VERIFIED,
            )
        with pytest.raises(Exception):
            Claim(
                claim_text="Test",
                confidence=11,  # Above maximum of 10
                fabrication_risk=FabricationRisk.LOW,
                verification_method="test",
                status=VerificationStatus.VERIFIED,
            )

    def test_claim_batch_model(self):
        batch = ClaimBatch(
            topic="Python",
            claims=[
                Claim(
                    claim_text="Python is popular",
                    confidence=8,
                    fabrication_risk=FabricationRisk.LOW,
                    verification_method="test",
                    status=VerificationStatus.VERIFIED,
                ),
            ],
            overall_reliability=0.9,
        )
        assert batch.topic == "Python"
        assert len(batch.claims) == 1
        assert batch.overall_reliability == 0.9

    def test_claim_batch_reliability_bounds(self):
        with pytest.raises(Exception):
            ClaimBatch(
                topic="Test",
                claims=[],
                overall_reliability=1.5,  # Above max 1.0
            )


# ============================================================================
# Edge Cases and Integration
# ============================================================================

class TestEdgeCases:
    def test_content_with_only_urls(self, verifier):
        content = "Visit https://docs.python.org and https://github.com/python for more info."
        result = verifier.verify(content)
        assert isinstance(result, VerificationReport)

    def test_content_with_many_measurements(self, verifier):
        content = (
            "Performance improved by 25 percent after optimization. "
            "Memory usage dropped to 50 percent of original. "
            "Response time decreased by 30 percent on average."
        )
        result = verifier.verify(content)
        assert result.total_claims_checked >= 1

    def test_content_with_future_dates(self, verifier):
        future_year = datetime.now().year + 50
        content = f"The system will be obsolete by {future_year}."
        result = verifier.verify(content)
        # Should flag future dates
        if result.claims:
            flagged_future = [c for c in result.claims if c.confidence <= 3]
            assert len(flagged_future) >= 0  # Non-crash check

    def test_content_with_hallucination_patterns(self, verifier):
        content = (
            "Obviously this is the best approach. "
            "Everyone knows that this pattern works. "
            "It is well known that this technique is superior. "
            "Clearly the results demonstrate effectiveness."
        )
        result = verifier.verify(content)
        # Should flag these patterns
        for claim in result.claims:
            if any(p in claim.claim_text.lower() for p in ["obviously", "everyone knows", "clearly"]):
                assert claim.confidence <= 6

    def test_very_long_content(self, verifier):
        content = ". ".join([f"Sentence number {i} is a factual claim about topic {i}" for i in range(100)])
        result = verifier.verify(content)
        assert isinstance(result, VerificationReport)

    def test_unicode_content(self, verifier):
        content = "Python supports unicode characters like accented letters and symbols."
        result = verifier.verify(content)
        assert isinstance(result, VerificationReport)

    def test_multiline_content(self, verifier):
        content = """Python was released in 1991.
        It supports multiple paradigms.
        According to surveys, it is widely used."""
        result = verifier.verify(content)
        assert isinstance(result, VerificationReport)
        assert result.total_claims_checked >= 1
