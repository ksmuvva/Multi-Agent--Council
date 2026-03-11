"""
Exhaustive Tests for the Complexity Classification Module

Tests every function, branch, edge case, and boundary condition in
src/core/complexity.py.
"""

import pytest
from pydantic import ValidationError

from src.core.complexity import (
    TierLevel,
    TierClassification,
    TIER_CONFIG,
    TIER_3_KEYWORDS,
    TIER_4_KEYWORDS,
    ESCALATION_KEYWORDS,
    classify_complexity,
    should_escalate,
    get_escalated_tier,
    estimate_agent_count,
    get_active_agents,
    get_council_agents,
)


# =============================================================================
# TierLevel Enum Tests
# =============================================================================


class TestTierLevel:
    """Tests for the TierLevel IntEnum."""

    def test_tier_values(self):
        """Test that tier values are 1-4."""
        assert TierLevel.DIRECT == 1
        assert TierLevel.STANDARD == 2
        assert TierLevel.DEEP == 3
        assert TierLevel.ADVERSARIAL == 4

    def test_tier_is_intenum(self):
        """Test that TierLevel is an IntEnum and supports integer comparison."""
        assert TierLevel.DIRECT < TierLevel.STANDARD
        assert TierLevel.STANDARD < TierLevel.DEEP
        assert TierLevel.DEEP < TierLevel.ADVERSARIAL

    def test_tier_from_int(self):
        """Test creating TierLevel from integer."""
        assert TierLevel(1) == TierLevel.DIRECT
        assert TierLevel(2) == TierLevel.STANDARD
        assert TierLevel(3) == TierLevel.DEEP
        assert TierLevel(4) == TierLevel.ADVERSARIAL

    def test_tier_from_invalid_int(self):
        """Test that invalid integers raise ValueError."""
        with pytest.raises(ValueError):
            TierLevel(0)
        with pytest.raises(ValueError):
            TierLevel(5)
        with pytest.raises(ValueError):
            TierLevel(-1)

    def test_tier_arithmetic(self):
        """Test that IntEnum supports arithmetic."""
        assert TierLevel.DIRECT + 1 == 2
        assert TierLevel.ADVERSARIAL - 1 == 3

    def test_tier_iteration(self):
        """Test iterating over all tier levels."""
        tiers = list(TierLevel)
        assert len(tiers) == 4
        assert tiers == [TierLevel.DIRECT, TierLevel.STANDARD, TierLevel.DEEP, TierLevel.ADVERSARIAL]


# =============================================================================
# TierClassification Model Tests
# =============================================================================


class TestTierClassification:
    """Tests for the TierClassification Pydantic model."""

    def test_valid_construction(self):
        """Test constructing a valid TierClassification."""
        tc = TierClassification(
            tier=TierLevel.DEEP,
            reasoning="Complex task",
            confidence=0.85,
            estimated_agents=12,
            requires_council=True,
            requires_smes=True,
            suggested_sme_count=2,
            escalation_risk=0.2,
            keywords_found=["security", "architecture"],
        )
        assert tc.tier == TierLevel.DEEP
        assert tc.confidence == 0.85
        assert tc.estimated_agents == 12
        assert tc.requires_council is True
        assert tc.requires_smes is True
        assert tc.suggested_sme_count == 2
        assert tc.escalation_risk == 0.2
        assert tc.keywords_found == ["security", "architecture"]

    def test_default_values(self):
        """Test default values for optional fields."""
        tc = TierClassification(
            tier=TierLevel.DIRECT,
            reasoning="Simple",
            confidence=0.5,
            estimated_agents=3,
            requires_council=False,
            requires_smes=False,
        )
        assert tc.suggested_sme_count == 0
        assert tc.escalation_risk == 0.0
        assert tc.keywords_found == []

    def test_confidence_boundary_lower(self):
        """Test confidence at lower boundary (0.0)."""
        tc = TierClassification(
            tier=TierLevel.DIRECT,
            reasoning="test",
            confidence=0.0,
            estimated_agents=3,
            requires_council=False,
            requires_smes=False,
        )
        assert tc.confidence == 0.0

    def test_confidence_boundary_upper(self):
        """Test confidence at upper boundary (1.0)."""
        tc = TierClassification(
            tier=TierLevel.DIRECT,
            reasoning="test",
            confidence=1.0,
            estimated_agents=3,
            requires_council=False,
            requires_smes=False,
        )
        assert tc.confidence == 1.0

    def test_confidence_out_of_range(self):
        """Test that confidence outside [0, 1] raises validation error."""
        with pytest.raises(ValidationError):
            TierClassification(
                tier=TierLevel.DIRECT,
                reasoning="test",
                confidence=1.5,
                estimated_agents=3,
                requires_council=False,
                requires_smes=False,
            )
        with pytest.raises(ValidationError):
            TierClassification(
                tier=TierLevel.DIRECT,
                reasoning="test",
                confidence=-0.1,
                estimated_agents=3,
                requires_council=False,
                requires_smes=False,
            )

    def test_sme_count_boundary(self):
        """Test suggested_sme_count boundaries (0-3)."""
        tc = TierClassification(
            tier=TierLevel.DEEP,
            reasoning="test",
            confidence=0.8,
            estimated_agents=12,
            requires_council=True,
            requires_smes=True,
            suggested_sme_count=3,
        )
        assert tc.suggested_sme_count == 3

    def test_sme_count_out_of_range(self):
        """Test that sme_count > 3 raises validation error."""
        with pytest.raises(ValidationError):
            TierClassification(
                tier=TierLevel.DEEP,
                reasoning="test",
                confidence=0.8,
                estimated_agents=12,
                requires_council=True,
                requires_smes=True,
                suggested_sme_count=4,
            )

    def test_escalation_risk_out_of_range(self):
        """Test that escalation_risk outside [0, 1] raises validation error."""
        with pytest.raises(ValidationError):
            TierClassification(
                tier=TierLevel.DIRECT,
                reasoning="test",
                confidence=0.5,
                estimated_agents=3,
                requires_council=False,
                requires_smes=False,
                escalation_risk=1.5,
            )

    def test_missing_required_fields(self):
        """Test that missing required fields raise validation error."""
        with pytest.raises(ValidationError):
            TierClassification(tier=TierLevel.DIRECT)

    def test_json_serialization(self):
        """Test JSON serialization round-trip."""
        tc = TierClassification(
            tier=TierLevel.DEEP,
            reasoning="Complex task",
            confidence=0.85,
            estimated_agents=12,
            requires_council=True,
            requires_smes=True,
            suggested_sme_count=2,
            escalation_risk=0.2,
            keywords_found=["security"],
        )
        json_str = tc.model_dump_json()
        restored = TierClassification.model_validate_json(json_str)
        assert restored.tier == tc.tier
        assert restored.confidence == tc.confidence
        assert restored.keywords_found == tc.keywords_found


# =============================================================================
# TIER_CONFIG Tests
# =============================================================================


class TestTierConfig:
    """Tests for the TIER_CONFIG dictionary."""

    def test_all_tiers_present(self):
        """Test that all 4 tiers have configurations."""
        for tier in TierLevel:
            assert tier in TIER_CONFIG

    def test_tier_config_has_required_keys(self):
        """Test that each tier config has required keys."""
        required_keys = [
            "name", "description", "active_agents",
            "agent_count", "requires_council", "requires_smes",
            "max_sme_count", "phases",
        ]
        for tier in TierLevel:
            config = TIER_CONFIG[tier]
            for key in required_keys:
                assert key in config, f"Tier {tier.name} missing key: {key}"

    def test_tier1_no_council(self):
        """Test Tier 1 does not require council."""
        config = TIER_CONFIG[TierLevel.DIRECT]
        assert config["requires_council"] is False
        assert config["requires_smes"] is False
        assert config["max_sme_count"] == 0

    def test_tier2_no_council(self):
        """Test Tier 2 does not require council."""
        config = TIER_CONFIG[TierLevel.STANDARD]
        assert config["requires_council"] is False
        assert config["requires_smes"] is False

    def test_tier3_requires_council(self):
        """Test Tier 3 requires council."""
        config = TIER_CONFIG[TierLevel.DEEP]
        assert config["requires_council"] is True
        assert config["requires_smes"] is True
        assert "council_agents" in config

    def test_tier4_requires_full_council(self):
        """Test Tier 4 requires full council."""
        config = TIER_CONFIG[TierLevel.ADVERSARIAL]
        assert config["requires_council"] is True
        assert config["requires_smes"] is True
        assert "council_agents" in config
        assert len(config["council_agents"]) == 3

    def test_agent_count_increases_with_tier(self):
        """Test that agent count generally increases with tier level."""
        counts = [TIER_CONFIG[tier]["agent_count"] for tier in TierLevel]
        assert counts[0] <= counts[1] <= counts[2] <= counts[3]

    def test_tier1_agent_count(self):
        """Test Tier 1 has exactly 3 agents."""
        assert TIER_CONFIG[TierLevel.DIRECT]["agent_count"] == 3

    def test_tier2_agent_count(self):
        """Test Tier 2 has exactly 7 agents."""
        assert TIER_CONFIG[TierLevel.STANDARD]["agent_count"] == 7

    def test_tier3_agent_count(self):
        """Test Tier 3 has exactly 12 agents."""
        assert TIER_CONFIG[TierLevel.DEEP]["agent_count"] == 12

    def test_tier4_agent_count(self):
        """Test Tier 4 has exactly 18 agents."""
        assert TIER_CONFIG[TierLevel.ADVERSARIAL]["agent_count"] == 18


# =============================================================================
# Keyword Lists Tests
# =============================================================================


class TestKeywordLists:
    """Tests for keyword list constants."""

    def test_tier3_keywords_not_empty(self):
        """Test that Tier 3 keywords list is not empty."""
        assert len(TIER_3_KEYWORDS) > 0

    def test_tier4_keywords_not_empty(self):
        """Test that Tier 4 keywords list is not empty."""
        assert len(TIER_4_KEYWORDS) > 0

    def test_escalation_keywords_not_empty(self):
        """Test that escalation keywords list is not empty."""
        assert len(ESCALATION_KEYWORDS) > 0

    def test_tier3_keywords_are_lowercase(self):
        """Test that Tier 3 keywords are lowercase for case-insensitive matching."""
        for kw in TIER_3_KEYWORDS:
            assert kw == kw.lower(), f"Keyword '{kw}' is not lowercase"

    def test_tier4_keywords_are_lowercase(self):
        """Test that Tier 4 keywords are lowercase."""
        for kw in TIER_4_KEYWORDS:
            assert kw == kw.lower(), f"Keyword '{kw}' is not lowercase"

    def test_escalation_keywords_are_lowercase(self):
        """Test that escalation keywords are lowercase."""
        for kw in ESCALATION_KEYWORDS:
            assert kw == kw.lower(), f"Keyword '{kw}' is not lowercase"

    def test_no_duplicate_keywords_within_tier3(self):
        """Test no duplicates in Tier 3 keywords."""
        assert len(TIER_3_KEYWORDS) == len(set(TIER_3_KEYWORDS))

    def test_no_duplicate_keywords_within_tier4(self):
        """Test no duplicates in Tier 4 keywords."""
        assert len(TIER_4_KEYWORDS) == len(set(TIER_4_KEYWORDS))

    def test_tier3_contains_expected_domains(self):
        """Test Tier 3 covers expected domain areas."""
        all_kw = " ".join(TIER_3_KEYWORDS)
        assert "architecture" in all_kw
        assert "machine learning" in all_kw
        assert "security" in all_kw
        assert "migration" in all_kw
        assert "test" in all_kw

    def test_tier4_contains_expected_sensitivity(self):
        """Test Tier 4 covers expected sensitive areas."""
        all_kw = " ".join(TIER_4_KEYWORDS)
        assert "pii" in all_kw
        assert "gdpr" in all_kw
        assert "hipaa" in all_kw
        assert "financial" in all_kw
        assert "medical" in all_kw


# =============================================================================
# classify_complexity Tests
# =============================================================================


class TestClassifyComplexity:
    """Exhaustive tests for the classify_complexity function."""

    # --- Tier 1 (Direct) Classification ---

    def test_empty_prompt_tier1(self):
        """Test empty prompt classifies as Tier 1."""
        result = classify_complexity("")
        assert result.tier == TierLevel.DIRECT

    def test_simple_greeting_tier1(self):
        """Test simple greeting classifies as Tier 1."""
        result = classify_complexity("Hello, how are you?")
        assert result.tier == TierLevel.DIRECT

    def test_simple_math_tier1(self):
        """Test simple math question classifies as Tier 1."""
        result = classify_complexity("What is 2 + 2?")
        assert result.tier == TierLevel.DIRECT

    def test_simple_factual_tier1(self):
        """Test simple factual question classifies as Tier 1."""
        result = classify_complexity("What is the capital of France?")
        assert result.tier == TierLevel.DIRECT

    def test_tier1_no_council(self):
        """Test Tier 1 does not require council."""
        result = classify_complexity("Hello")
        assert result.requires_council is False
        assert result.requires_smes is False

    def test_tier1_agent_count(self):
        """Test Tier 1 has 3 estimated agents."""
        result = classify_complexity("Simple task")
        assert result.estimated_agents == 3

    def test_tier1_reasoning(self):
        """Test Tier 1 has appropriate reasoning."""
        result = classify_complexity("What time is it?")
        assert "simple" in result.reasoning.lower() or "direct" in result.reasoning.lower()

    def test_tier1_confidence(self):
        """Test Tier 1 confidence value."""
        result = classify_complexity("Hello")
        assert result.confidence == 0.7  # tier_score < 3 => 0.7

    # --- Tier 2 (Standard) Classification ---

    def test_escalation_keyword_complex(self):
        """Test 'complex' keyword bumps to Tier 2."""
        result = classify_complexity("This is a complex problem")
        assert result.tier == TierLevel.STANDARD

    def test_escalation_keyword_complicated(self):
        """Test 'complicated' keyword bumps to Tier 2."""
        result = classify_complexity("This is quite complicated")
        assert result.tier == TierLevel.STANDARD

    def test_escalation_keyword_multi_step(self):
        """Test 'multi-step' keyword bumps to Tier 2."""
        result = classify_complexity("This is a multi-step process")
        assert result.tier == TierLevel.STANDARD

    def test_escalation_keyword_uncertain(self):
        """Test 'uncertain' keyword in isolation bumps to Tier 2.

        Known issue: 'ai' is a Tier 3 keyword that matches as a substring
        in 'uncertain' (uncert-ai-n), so 'uncertain' actually triggers
        Tier 3. We test with 'not sure' instead which is a clean
        escalation keyword without false-positive substring matches.
        """
        result = classify_complexity("I am not sure about the outcome")
        assert result.tier == TierLevel.STANDARD

    def test_escalation_keyword_conditional(self):
        """Test 'conditional' keyword bumps to Tier 2."""
        result = classify_complexity("The outcome is conditional on many factors")
        assert result.tier == TierLevel.STANDARD

    def test_tier2_no_council(self):
        """Test Tier 2 does not require council."""
        result = classify_complexity("This is a complex task")
        assert result.requires_council is False
        assert result.requires_smes is False

    def test_tier2_agent_count(self):
        """Test Tier 2 has 7 estimated agents."""
        result = classify_complexity("This is complicated to solve")
        assert result.estimated_agents == 7

    def test_tier2_escalation_risk_includes_base_and_tier2_bonus(self):
        """Test Tier 2 escalation risk includes tier bonus."""
        result = classify_complexity("This is a complex task")
        # base 0.1 + escalation_matches 0.2 + tier2 bonus 0.15 = 0.45
        assert result.escalation_risk == pytest.approx(0.45)

    def test_tier2_confidence(self):
        """Test Tier 2 confidence value."""
        result = classify_complexity("This is complicated")
        assert result.confidence == 0.7  # tier_score < 3 => 0.7

    # --- Tier 3 (Deep) Classification ---

    def test_tier3_keyword_architecture(self):
        """Test 'architecture' triggers Tier 3."""
        result = classify_complexity("Review the architecture of this system")
        assert result.tier == TierLevel.DEEP

    def test_tier3_keyword_machine_learning(self):
        """Test 'machine learning' triggers Tier 3."""
        result = classify_complexity("Build a machine learning model")
        assert result.tier == TierLevel.DEEP

    def test_tier3_keyword_data_pipeline(self):
        """Test 'data pipeline' triggers Tier 3."""
        result = classify_complexity("Design a data pipeline for ETL")
        assert result.tier == TierLevel.DEEP

    def test_tier3_keyword_microservices(self):
        """Test 'microservices' triggers Tier 3."""
        result = classify_complexity("Design a microservices system")
        assert result.tier == TierLevel.DEEP

    def test_tier3_keyword_rag(self):
        """Test 'rag' triggers Tier 3."""
        result = classify_complexity("Implement a rag system")
        assert result.tier == TierLevel.DEEP

    def test_tier3_keyword_test_strategy(self):
        """Test 'test strategy' triggers Tier 3."""
        result = classify_complexity("Define a test strategy for the project")
        assert result.tier == TierLevel.DEEP

    def test_tier3_keyword_gap_analysis(self):
        """Test 'gap analysis' triggers Tier 3."""
        result = classify_complexity("Perform a gap analysis")
        assert result.tier == TierLevel.DEEP

    def test_tier3_requires_council(self):
        """Test Tier 3 requires council."""
        result = classify_complexity("Design the system design for our platform")
        assert result.requires_council is True
        assert result.requires_smes is True

    def test_tier3_agent_count(self):
        """Test Tier 3 has 12 estimated agents."""
        result = classify_complexity("Design a microservices architecture")
        assert result.estimated_agents == 12

    def test_tier3_sme_count(self):
        """Test Tier 3 suggests up to 3 SMEs."""
        result = classify_complexity("Design a microservices architecture")
        assert result.suggested_sme_count == 3

    def test_tier3_confidence(self):
        """Test Tier 3 confidence value."""
        result = classify_complexity("Design an architecture")
        assert result.confidence == 0.8  # tier_score >= 3 => 0.8

    def test_tier3_keywords_found(self):
        """Test keywords are captured in result."""
        result = classify_complexity("Design a microservices architecture")
        assert "microservices" in result.keywords_found or "architecture" in result.keywords_found

    # --- Tier 4 (Adversarial) Classification ---

    def test_tier4_keyword_security_audit(self):
        """Test 'security audit' triggers Tier 4."""
        result = classify_complexity("Perform a security audit")
        assert result.tier == TierLevel.ADVERSARIAL

    def test_tier4_keyword_hipaa(self):
        """Test 'hipaa' triggers Tier 4."""
        result = classify_complexity("Ensure hipaa compliance")
        assert result.tier == TierLevel.ADVERSARIAL

    def test_tier4_keyword_gdpr(self):
        """Test 'gdpr' triggers Tier 4."""
        result = classify_complexity("Implement gdpr data handling")
        assert result.tier == TierLevel.ADVERSARIAL

    def test_tier4_keyword_pii(self):
        """Test 'pii' triggers Tier 4."""
        result = classify_complexity("Process pii data safely")
        assert result.tier == TierLevel.ADVERSARIAL

    def test_tier4_keyword_financial(self):
        """Test 'financial' triggers Tier 4."""
        result = classify_complexity("Build a financial reporting system")
        assert result.tier == TierLevel.ADVERSARIAL

    def test_tier4_keyword_medical(self):
        """Test 'medical' triggers Tier 4."""
        result = classify_complexity("Handle medical records")
        assert result.tier == TierLevel.ADVERSARIAL

    def test_tier4_keyword_vulnerability(self):
        """Test 'vulnerability' triggers Tier 4."""
        result = classify_complexity("Scan for vulnerability in this code")
        assert result.tier == TierLevel.ADVERSARIAL

    def test_tier4_keyword_legal(self):
        """Test 'legal' triggers Tier 4."""
        result = classify_complexity("Review legal requirements for this system")
        assert result.tier == TierLevel.ADVERSARIAL

    def test_tier4_keyword_adversarial(self):
        """Test 'adversarial' triggers Tier 4."""
        result = classify_complexity("Run adversarial testing")
        assert result.tier == TierLevel.ADVERSARIAL

    def test_tier4_keyword_mission_critical(self):
        """Test 'mission critical' triggers Tier 4."""
        result = classify_complexity("Deploy to mission critical infrastructure")
        assert result.tier == TierLevel.ADVERSARIAL

    def test_tier4_keyword_debate(self):
        """Test 'debate' triggers Tier 4."""
        result = classify_complexity("Let's debate the merits of this approach")
        assert result.tier == TierLevel.ADVERSARIAL

    def test_tier4_requires_council(self):
        """Test Tier 4 requires council."""
        result = classify_complexity("Review the security audit findings")
        assert result.requires_council is True
        assert result.requires_smes is True

    def test_tier4_agent_count(self):
        """Test Tier 4 has 18 estimated agents."""
        result = classify_complexity("Run a security audit")
        assert result.estimated_agents == 18

    def test_tier4_confidence(self):
        """Test Tier 4 confidence value."""
        result = classify_complexity("Review pii handling")
        assert result.confidence == 0.8  # tier_score >= 3 => 0.8

    # --- Tier Priority / Overlap Tests ---

    def test_tier4_takes_priority_over_tier3(self):
        """Test Tier 4 keywords take priority when both are present."""
        # "architecture" is Tier 3, "security audit" is Tier 4
        result = classify_complexity(
            "Perform a security audit of the architecture"
        )
        assert result.tier == TierLevel.ADVERSARIAL

    def test_tier4_keywords_found_when_both_tiers_match(self):
        """Test that when both T3 and T4 match, T4 keywords are captured."""
        result = classify_complexity(
            "Perform a security audit of the microservices architecture"
        )
        assert result.tier == TierLevel.ADVERSARIAL
        # Tier 4 keywords should be in keywords_found
        assert "security audit" in result.keywords_found

    def test_tier3_keywords_not_added_when_tier4_active(self):
        """Test that Tier 3 keywords are NOT added when Tier 4 already scored.

        Bug documentation: In classify_complexity, when tier_4_matches sets
        tier_score=4, the condition `tier_score < 3` is False, so tier_3
        keywords are not appended to keywords_found. This tests the current
        behavior.
        """
        result = classify_complexity(
            "Run a security audit on the microservices architecture"
        )
        # "security audit" is T4, "microservices" and "architecture" are T3
        assert result.tier == TierLevel.ADVERSARIAL
        # T3 keywords should NOT be in keywords_found due to the < 3 guard
        assert "microservices" not in result.keywords_found
        assert "architecture" not in result.keywords_found

    def test_escalation_does_not_override_tier3(self):
        """Test escalation keywords don't lower tier from 3 to 2."""
        result = classify_complexity(
            "Design a complex microservices architecture"
        )
        # "microservices" and "architecture" are T3, "complex" is escalation
        assert result.tier == TierLevel.DEEP

    def test_escalation_does_not_override_tier4(self):
        """Test escalation keywords don't lower tier from 4."""
        result = classify_complexity(
            "Complex security audit of the system"
        )
        assert result.tier == TierLevel.ADVERSARIAL

    # --- Analyst Report Influence ---

    def test_analyst_report_upgrades_tier(self):
        """Test analyst report can upgrade tier."""
        result = classify_complexity(
            "Hello world",
            analyst_report={"suggested_tier": 3}
        )
        assert result.tier == TierLevel.DEEP

    def test_analyst_report_does_not_downgrade_tier(self):
        """Test analyst report cannot downgrade from keyword-based tier."""
        result = classify_complexity(
            "Perform a security audit",
            analyst_report={"suggested_tier": 1}
        )
        # T4 keyword wins over analyst suggestion of 1
        assert result.tier == TierLevel.ADVERSARIAL

    def test_analyst_report_escalation_flag(self):
        """Test analyst report escalation_needed flag is read."""
        result = classify_complexity(
            "Hello",
            analyst_report={"suggested_tier": 2, "escalation_needed": True}
        )
        assert result.tier == TierLevel.STANDARD

    def test_analyst_report_empty_dict_is_falsy(self):
        """Test that empty dict analyst_report is treated as no report.

        Known issue: In Python, {} is falsy, so `if analyst_report:` is
        False for empty dicts. The analyst code block is skipped entirely.
        """
        result = classify_complexity("Hello", analyst_report={})
        # Empty dict is falsy, so analyst block is skipped => Tier 1
        assert result.tier == TierLevel.DIRECT

    def test_analyst_report_with_default_tier(self):
        """Test analyst report where suggested_tier key is missing but dict is non-empty."""
        result = classify_complexity(
            "Hello",
            analyst_report={"some_key": "some_value"}
        )
        # Non-empty dict is truthy, suggested_tier defaults to 2
        assert result.tier == TierLevel.STANDARD

    def test_analyst_report_none(self):
        """Test None analyst report is handled."""
        result = classify_complexity("Hello", analyst_report=None)
        assert result.tier == TierLevel.DIRECT

    def test_analyst_report_reasoning_appended(self):
        """Test analyst recommendation appears in reasoning."""
        result = classify_complexity(
            "Hello",
            analyst_report={"suggested_tier": 3}
        )
        assert "Analyst recommended Tier 3" in result.reasoning

    # --- Case Insensitivity ---

    def test_case_insensitive_tier4(self):
        """Test Tier 4 keywords are matched case-insensitively."""
        result = classify_complexity("Perform a SECURITY AUDIT")
        assert result.tier == TierLevel.ADVERSARIAL

    def test_case_insensitive_tier3(self):
        """Test Tier 3 keywords are matched case-insensitively."""
        result = classify_complexity("Design MICROSERVICES Architecture")
        assert result.tier == TierLevel.DEEP

    def test_mixed_case_escalation(self):
        """Test escalation keywords are matched case-insensitively.

        Note: 'UNCERTAIN' contains 'ai' substring which is a Tier 3 keyword.
        Use 'COMPLEX' and 'MULTI-STEP' to avoid false positives.
        """
        result = classify_complexity("This is COMPLEX and MULTI-STEP")
        assert result.tier == TierLevel.STANDARD

    # --- Escalation Risk Calculation ---

    def test_escalation_risk_base(self):
        """Test base escalation risk for Tier 1 (no escalation keywords)."""
        result = classify_complexity("Hello")
        assert result.escalation_risk == pytest.approx(0.1)

    def test_escalation_risk_with_escalation_keywords_tier2(self):
        """Test escalation risk for Tier 2 with escalation keywords."""
        result = classify_complexity("This is complex and multi-step")
        # base 0.1 + escalation 0.2 + tier2 bonus 0.15 = 0.45
        assert result.tier == TierLevel.STANDARD
        assert result.escalation_risk == pytest.approx(0.45)

    def test_escalation_risk_tier3(self):
        """Test escalation risk for Tier 3."""
        result = classify_complexity("Design an architecture")
        # base 0.1, no escalation keywords, not tier 2
        assert result.escalation_risk == pytest.approx(0.1)

    def test_escalation_risk_tier3_with_escalation_kw(self):
        """Test escalation risk for Tier 3 with escalation keywords."""
        result = classify_complexity("Design a complex architecture")
        # base 0.1 + escalation 0.2 = 0.3 (tier is 3, not 2, so no tier2 bonus)
        assert result.escalation_risk == pytest.approx(0.3)

    def test_escalation_risk_capped_at_1(self):
        """Test escalation risk is capped at 1.0."""
        # All factors present: base 0.1 + escalation 0.2 + tier2 0.15 = 0.45
        # Can't exceed 1.0 in normal cases, but test the min() logic
        result = classify_complexity("This is complex")
        assert result.escalation_risk <= 1.0

    # --- Reasoning Tests ---

    def test_reasoning_tier4_mentions_indicators(self):
        """Test Tier 4 reasoning mentions indicators."""
        result = classify_complexity("Review pii and gdpr handling")
        assert "Tier 4 indicators" in result.reasoning

    def test_reasoning_tier3_mentions_keywords(self):
        """Test Tier 3 reasoning mentions domain keywords."""
        result = classify_complexity("Design an architecture")
        assert "domain-specific keywords" in result.reasoning

    def test_reasoning_tier1_mentions_simple(self):
        """Test Tier 1 reasoning mentions simplicity."""
        result = classify_complexity("Hi")
        assert "Simple" in result.reasoning or "simple" in result.reasoning

    def test_reasoning_joined_with_periods(self):
        """Test multiple reasoning parts are joined with periods."""
        result = classify_complexity(
            "Hello",
            analyst_report={"suggested_tier": 3}
        )
        # Should have analyst recommendation in reasoning
        assert ". " in result.reasoning or "Analyst" in result.reasoning

    # --- Edge Cases ---

    def test_very_long_prompt(self):
        """Test handling of very long prompts."""
        long_prompt = "word " * 10000
        result = classify_complexity(long_prompt)
        assert result.tier is not None

    def test_special_characters_in_prompt(self):
        """Test prompt with special characters."""
        result = classify_complexity("!@#$%^&*()_+-=[]{}|;':\",./<>?")
        assert result.tier == TierLevel.DIRECT

    def test_newlines_in_prompt(self):
        """Test prompt with newlines."""
        result = classify_complexity("Line 1\nLine 2\nLine 3")
        assert result.tier == TierLevel.DIRECT

    def test_unicode_in_prompt(self):
        """Test prompt with unicode characters."""
        result = classify_complexity("Design an architecture for 日本語 support")
        assert result.tier == TierLevel.DEEP

    def test_keyword_as_substring(self):
        """Test that keywords match as substrings in the prompt."""
        # "ai" is a Tier 3 keyword - it should match inside words too
        # because the check is `kw in prompt_lower`
        result = classify_complexity("This contains ai somewhere")
        assert "ai" in result.keywords_found

    def test_ai_substring_false_positive_bug(self):
        """Document: 'ai' matches as substring in common words.

        Known bug: The keyword 'ai' (2 chars) matches in words like
        'uncertain', 'maintain', 'explain', 'contain', 'obtain', etc.
        because matching uses `kw in prompt_lower` (substring match).
        """
        # 'uncertain' contains 'ai' as a substring
        result = classify_complexity("I am uncertain")
        assert result.tier == TierLevel.DEEP  # Tier 3 due to 'ai' false positive
        assert "ai" in result.keywords_found

    def test_multiple_tier3_keywords(self):
        """Test multiple Tier 3 keywords all captured."""
        result = classify_complexity(
            "Design a microservices architecture with a data pipeline"
        )
        assert result.tier == TierLevel.DEEP
        assert len(result.keywords_found) >= 2

    def test_multiple_tier4_keywords(self):
        """Test multiple Tier 4 keywords all captured."""
        result = classify_complexity(
            "Review pii handling with gdpr and hipaa compliance"
        )
        assert result.tier == TierLevel.ADVERSARIAL
        assert len(result.keywords_found) >= 3

    def test_tier4_reasoning_limits_to_3_keywords(self):
        """Test Tier 4 reasoning only shows first 3 keywords."""
        result = classify_complexity(
            "Review pii gdpr hipaa financial medical legal compliance"
        )
        # Reasoning should mention at most 3 keywords in the indicator list
        indicator_part = result.reasoning.split("Tier 4 indicators: ")[-1]
        mentioned_keywords = indicator_part.split(", ")
        assert len(mentioned_keywords) <= 3


# =============================================================================
# should_escalate Tests
# =============================================================================


class TestShouldEscalate:
    """Tests for the should_escalate function."""

    def test_explicit_escalation_flag(self):
        """Test escalation when escalation_needed is True."""
        assert should_escalate(
            TierLevel.STANDARD,
            {"escalation_needed": True}
        ) is True

    def test_no_escalation_flag(self):
        """Test no escalation when escalation_needed is False."""
        assert should_escalate(
            TierLevel.STANDARD,
            {"escalation_needed": False, "message": "all good"}
        ) is False

    def test_escalation_indicator_domain_expertise(self):
        """Test escalation on 'domain expertise required'."""
        assert should_escalate(
            TierLevel.STANDARD,
            {"message": "domain expertise required for this task"}
        ) is True

    def test_escalation_indicator_need_specialist(self):
        """Test escalation on 'need specialist'."""
        assert should_escalate(
            TierLevel.STANDARD,
            {"message": "need specialist input"}
        ) is True

    def test_escalation_indicator_outside_scope(self):
        """Test escalation on 'outside scope'."""
        assert should_escalate(
            TierLevel.STANDARD,
            {"message": "This is outside scope of current tier"}
        ) is True

    def test_escalation_indicator_requires_sme(self):
        """Test escalation on 'requires SME'."""
        assert should_escalate(
            TierLevel.STANDARD,
            {"message": "This requires SME input"}
        ) is True

    def test_escalation_indicator_uncertain(self):
        """Test escalation on 'uncertain'."""
        assert should_escalate(
            TierLevel.STANDARD,
            {"message": "I am uncertain about this result"}
        ) is True

    def test_escalation_indicator_need_verification(self):
        """Test escalation on 'need verification'."""
        assert should_escalate(
            TierLevel.STANDARD,
            {"notes": "need verification from expert"}
        ) is True

    def test_no_escalation_clean_feedback(self):
        """Test no escalation with clean feedback."""
        assert should_escalate(
            TierLevel.STANDARD,
            {"message": "Task completed successfully", "status": "done"}
        ) is False

    def test_empty_feedback(self):
        """Test no escalation with empty feedback."""
        assert should_escalate(TierLevel.STANDARD, {}) is False

    def test_nested_dict_feedback(self):
        """Test escalation with nested dict containing indicator."""
        # str() of the dict will include nested values
        assert should_escalate(
            TierLevel.STANDARD,
            {"nested": {"deep": "need specialist for this"}}
        ) is True

    def test_escalation_case_insensitive(self):
        """Test that escalation indicators are case-insensitive (via .lower())."""
        assert should_escalate(
            TierLevel.STANDARD,
            {"message": "DOMAIN EXPERTISE REQUIRED"}
        ) is True

    def test_escalation_from_tier4(self):
        """Test escalation can be triggered even from Tier 4."""
        # The function doesn't check if already at max tier
        assert should_escalate(
            TierLevel.ADVERSARIAL,
            {"escalation_needed": True}
        ) is True


# =============================================================================
# get_escalated_tier Tests
# =============================================================================


class TestGetEscalatedTier:
    """Tests for the get_escalated_tier function."""

    def test_tier1_escalates_to_tier2(self):
        """Test Tier 1 escalates to Tier 2."""
        assert get_escalated_tier(TierLevel.DIRECT) == TierLevel.STANDARD

    def test_tier2_escalates_to_tier3(self):
        """Test Tier 2 escalates to Tier 3."""
        assert get_escalated_tier(TierLevel.STANDARD) == TierLevel.DEEP

    def test_tier3_escalates_to_tier4(self):
        """Test Tier 3 escalates to Tier 4."""
        assert get_escalated_tier(TierLevel.DEEP) == TierLevel.ADVERSARIAL

    def test_tier4_stays_at_tier4(self):
        """Test Tier 4 stays at Tier 4 (capped)."""
        assert get_escalated_tier(TierLevel.ADVERSARIAL) == TierLevel.ADVERSARIAL

    def test_return_type_is_tier_level(self):
        """Test return type is TierLevel."""
        result = get_escalated_tier(TierLevel.DIRECT)
        assert isinstance(result, TierLevel)


# =============================================================================
# estimate_agent_count Tests
# =============================================================================


class TestEstimateAgentCount:
    """Tests for the estimate_agent_count function."""

    def test_tier1_no_smes(self):
        """Test Tier 1 agent count without SMEs."""
        assert estimate_agent_count(TierLevel.DIRECT) == 3

    def test_tier2_no_smes(self):
        """Test Tier 2 agent count without SMEs."""
        assert estimate_agent_count(TierLevel.STANDARD) == 7

    def test_tier3_no_smes(self):
        """Test Tier 3 agent count without SMEs."""
        assert estimate_agent_count(TierLevel.DEEP) == 12

    def test_tier4_no_smes(self):
        """Test Tier 4 agent count without SMEs."""
        assert estimate_agent_count(TierLevel.ADVERSARIAL) == 18

    def test_tier3_with_smes(self):
        """Test Tier 3 agent count with SMEs."""
        assert estimate_agent_count(TierLevel.DEEP, sme_count=2) == 14

    def test_tier4_with_max_smes(self):
        """Test Tier 4 agent count with max SMEs."""
        assert estimate_agent_count(TierLevel.ADVERSARIAL, sme_count=3) == 21

    def test_zero_smes(self):
        """Test explicit zero SMEs."""
        assert estimate_agent_count(TierLevel.STANDARD, sme_count=0) == 7

    def test_default_sme_count(self):
        """Test default sme_count is 0."""
        assert estimate_agent_count(TierLevel.DIRECT) == estimate_agent_count(TierLevel.DIRECT, 0)


# =============================================================================
# get_active_agents Tests
# =============================================================================


class TestGetActiveAgents:
    """Tests for the get_active_agents function."""

    def test_tier1_agents(self):
        """Test Tier 1 returns correct agents."""
        agents = get_active_agents(TierLevel.DIRECT)
        assert "Orchestrator" in agents
        assert "Executor" in agents
        assert "Formatter" in agents
        assert len(agents) == 3

    def test_tier2_agents(self):
        """Test Tier 2 returns correct agents."""
        agents = get_active_agents(TierLevel.STANDARD)
        expected = ["Orchestrator", "Analyst", "Planner", "Clarifier",
                     "Executor", "Verifier", "Reviewer", "Formatter"]
        for agent in expected:
            assert agent in agents

    def test_tier3_agents(self):
        """Test Tier 3 returns correct agents."""
        agents = get_active_agents(TierLevel.DEEP)
        assert "Researcher" in agents
        assert "Critic" in agents
        assert "Code Reviewer" in agents
        assert "Memory Curator" in agents

    def test_tier4_agents(self):
        """Test Tier 4 returns agents list."""
        agents = get_active_agents(TierLevel.ADVERSARIAL)
        assert len(agents) >= 1

    def test_returns_copy(self):
        """Test that returned list is a copy (mutation safe)."""
        agents1 = get_active_agents(TierLevel.DIRECT)
        agents2 = get_active_agents(TierLevel.DIRECT)
        agents1.append("TestAgent")
        assert "TestAgent" not in agents2

    def test_all_tiers_return_list(self):
        """Test all tiers return a list."""
        for tier in TierLevel:
            agents = get_active_agents(tier)
            assert isinstance(agents, list)


# =============================================================================
# get_council_agents Tests
# =============================================================================


class TestGetCouncilAgents:
    """Tests for the get_council_agents function."""

    def test_tier1_no_council(self):
        """Test Tier 1 has no council agents."""
        council = get_council_agents(TierLevel.DIRECT)
        assert council == []

    def test_tier2_no_council(self):
        """Test Tier 2 has no council agents."""
        council = get_council_agents(TierLevel.STANDARD)
        assert council == []

    def test_tier3_has_chair(self):
        """Test Tier 3 has Domain Council Chair."""
        council = get_council_agents(TierLevel.DEEP)
        assert "Domain Council Chair" in council
        assert len(council) == 1

    def test_tier4_full_council(self):
        """Test Tier 4 has full council."""
        council = get_council_agents(TierLevel.ADVERSARIAL)
        assert len(council) == 3
        assert "Domain Council Chair" in council
        assert "Quality Arbiter" in council
        assert "Ethics & Safety Advisor" in council

    def test_all_tiers_return_list(self):
        """Test all tiers return a list."""
        for tier in TierLevel:
            council = get_council_agents(tier)
            assert isinstance(council, list)


# =============================================================================
# Integration / Combined Tests
# =============================================================================


class TestClassifierIntegration:
    """Integration tests combining multiple classifier functions."""

    def test_classify_then_escalate(self):
        """Test classify then escalate workflow."""
        result = classify_complexity("Simple hello world")
        assert result.tier == TierLevel.DIRECT

        escalated = get_escalated_tier(result.tier)
        assert escalated == TierLevel.STANDARD

        escalated_config = TIER_CONFIG[escalated]
        assert escalated_config["agent_count"] > TIER_CONFIG[result.tier]["agent_count"]

    def test_classify_get_agents_and_council(self):
        """Test classify then get agents and council."""
        result = classify_complexity("Design a microservices architecture")
        assert result.tier == TierLevel.DEEP

        agents = get_active_agents(result.tier)
        council = get_council_agents(result.tier)
        total = estimate_agent_count(result.tier, result.suggested_sme_count)

        assert len(agents) > 0
        assert len(council) > 0
        assert total >= result.estimated_agents

    def test_full_tier_escalation_chain(self):
        """Test escalating from Tier 1 through to Tier 4."""
        current = TierLevel.DIRECT
        tiers_visited = [current]

        while current < TierLevel.ADVERSARIAL:
            current = get_escalated_tier(current)
            tiers_visited.append(current)

        assert tiers_visited == [
            TierLevel.DIRECT,
            TierLevel.STANDARD,
            TierLevel.DEEP,
            TierLevel.ADVERSARIAL,
        ]

    def test_classify_config_consistency(self):
        """Test that classify_complexity output matches TIER_CONFIG."""
        for tier in TierLevel:
            config = TIER_CONFIG[tier]
            # Build a prompt that triggers this tier
            if tier == TierLevel.DIRECT:
                prompt = "Hello"
            elif tier == TierLevel.STANDARD:
                prompt = "This is a complex task"
            elif tier == TierLevel.DEEP:
                prompt = "Design an architecture"
            else:
                prompt = "Perform a security audit"

            result = classify_complexity(prompt)
            assert result.tier == tier
            assert result.estimated_agents == config["agent_count"]
            assert result.requires_council == config["requires_council"]
            assert result.requires_smes == config["requires_smes"]

    def test_all_tier3_keywords_trigger_tier3_or_higher(self):
        """Test every Tier 3 keyword triggers at least Tier 3."""
        for keyword in TIER_3_KEYWORDS:
            result = classify_complexity(f"Please do {keyword}")
            assert result.tier.value >= TierLevel.DEEP.value, (
                f"Keyword '{keyword}' only triggered Tier {result.tier.value}"
            )

    def test_all_tier4_keywords_trigger_tier4(self):
        """Test every Tier 4 keyword triggers Tier 4."""
        for keyword in TIER_4_KEYWORDS:
            result = classify_complexity(f"Please handle {keyword}")
            assert result.tier == TierLevel.ADVERSARIAL, (
                f"Keyword '{keyword}' triggered Tier {result.tier.value} instead of 4"
            )

    def test_all_escalation_keywords_trigger_at_least_tier2(self):
        """Test every escalation keyword triggers at least Tier 2."""
        for keyword in ESCALATION_KEYWORDS:
            result = classify_complexity(f"This task is {keyword}")
            assert result.tier.value >= TierLevel.STANDARD.value, (
                f"Escalation keyword '{keyword}' only triggered Tier {result.tier.value}"
            )
