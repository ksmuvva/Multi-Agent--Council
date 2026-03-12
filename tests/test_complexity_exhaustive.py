"""
Exhaustive Tests for Complexity Classification Module

Tests all functions, enums, models, tier configurations,
keyword matching, escalation logic, and edge cases.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

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
    """Tests for TierLevel IntEnum."""

    def test_tier_values(self):
        assert TierLevel.DIRECT == 1
        assert TierLevel.STANDARD == 2
        assert TierLevel.DEEP == 3
        assert TierLevel.ADVERSARIAL == 4

    def test_tier_is_int(self):
        for tier in TierLevel:
            assert isinstance(tier, int)

    def test_tier_ordering(self):
        assert TierLevel.DIRECT < TierLevel.STANDARD
        assert TierLevel.STANDARD < TierLevel.DEEP
        assert TierLevel.DEEP < TierLevel.ADVERSARIAL

    def test_tier_comparison(self):
        assert TierLevel.DIRECT < 2
        assert TierLevel.ADVERSARIAL >= 4
        assert TierLevel.DEEP == 3

    def test_tier_count(self):
        assert len(TierLevel) == 4

    def test_tier_from_value(self):
        assert TierLevel(1) == TierLevel.DIRECT
        assert TierLevel(2) == TierLevel.STANDARD
        assert TierLevel(3) == TierLevel.DEEP
        assert TierLevel(4) == TierLevel.ADVERSARIAL

    def test_tier_invalid_value(self):
        with pytest.raises(ValueError):
            TierLevel(0)
        with pytest.raises(ValueError):
            TierLevel(5)


# =============================================================================
# TierClassification Model Tests
# =============================================================================

class TestTierClassification:
    """Tests for TierClassification Pydantic model."""

    def test_valid_classification(self):
        tc = TierClassification(
            tier=TierLevel.STANDARD,
            reasoning="Standard task",
            confidence=0.7,
            estimated_agents=7,
            requires_council=False,
            requires_smes=False,
        )
        assert tc.tier == TierLevel.STANDARD
        assert tc.confidence == 0.7
        assert tc.estimated_agents == 7
        assert tc.requires_council is False
        assert tc.requires_smes is False

    def test_defaults(self):
        tc = TierClassification(
            tier=TierLevel.DIRECT,
            reasoning="Simple",
            confidence=0.9,
            estimated_agents=3,
            requires_council=False,
            requires_smes=False,
        )
        assert tc.suggested_sme_count == 0
        assert tc.escalation_risk == 0.0
        assert tc.keywords_found == []

    def test_full_classification(self):
        tc = TierClassification(
            tier=TierLevel.ADVERSARIAL,
            reasoning="High stakes",
            confidence=0.85,
            estimated_agents=18,
            requires_council=True,
            requires_smes=True,
            suggested_sme_count=3,
            escalation_risk=0.5,
            keywords_found=["security", "compliance"],
        )
        assert tc.suggested_sme_count == 3
        assert tc.escalation_risk == 0.5
        assert len(tc.keywords_found) == 2

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            TierClassification(
                tier=TierLevel.DIRECT,
                reasoning="Test",
                confidence=-0.1,
                estimated_agents=3,
                requires_council=False,
                requires_smes=False,
            )
        with pytest.raises(ValidationError):
            TierClassification(
                tier=TierLevel.DIRECT,
                reasoning="Test",
                confidence=1.1,
                estimated_agents=3,
                requires_council=False,
                requires_smes=False,
            )

    def test_confidence_edge_values(self):
        tc_zero = TierClassification(
            tier=TierLevel.DIRECT, reasoning="Test", confidence=0.0,
            estimated_agents=3, requires_council=False, requires_smes=False,
        )
        assert tc_zero.confidence == 0.0

        tc_one = TierClassification(
            tier=TierLevel.DIRECT, reasoning="Test", confidence=1.0,
            estimated_agents=3, requires_council=False, requires_smes=False,
        )
        assert tc_one.confidence == 1.0

    def test_sme_count_bounds(self):
        with pytest.raises(ValidationError):
            TierClassification(
                tier=TierLevel.DEEP, reasoning="Test", confidence=0.8,
                estimated_agents=12, requires_council=True, requires_smes=True,
                suggested_sme_count=4,
            )
        with pytest.raises(ValidationError):
            TierClassification(
                tier=TierLevel.DEEP, reasoning="Test", confidence=0.8,
                estimated_agents=12, requires_council=True, requires_smes=True,
                suggested_sme_count=-1,
            )

    def test_escalation_risk_bounds(self):
        with pytest.raises(ValidationError):
            TierClassification(
                tier=TierLevel.DIRECT, reasoning="Test", confidence=0.7,
                estimated_agents=3, requires_council=False, requires_smes=False,
                escalation_risk=1.5,
            )

    def test_json_serialization(self):
        tc = TierClassification(
            tier=TierLevel.DEEP, reasoning="Complex task", confidence=0.85,
            estimated_agents=12, requires_council=True, requires_smes=True,
            suggested_sme_count=2, keywords_found=["security"],
        )
        data = tc.model_dump()
        assert data["tier"] == 3
        assert data["reasoning"] == "Complex task"
        assert isinstance(data["keywords_found"], list)


# =============================================================================
# TIER_CONFIG Tests
# =============================================================================

class TestTierConfig:
    """Tests for TIER_CONFIG dictionary."""

    def test_all_tiers_present(self):
        for tier in TierLevel:
            assert tier in TIER_CONFIG

    def test_tier_1_config(self):
        config = TIER_CONFIG[TierLevel.DIRECT]
        assert config["name"] == "Direct"
        assert config["agent_count"] == 3
        assert config["requires_council"] is False
        assert config["requires_smes"] is False
        assert config["max_sme_count"] == 0
        assert "Executor" in config["active_agents"]
        assert "Formatter" in config["active_agents"]

    def test_tier_2_config(self):
        config = TIER_CONFIG[TierLevel.STANDARD]
        assert config["name"] == "Standard"
        assert config["agent_count"] == 7
        assert config["requires_council"] is False
        assert config["requires_smes"] is False
        assert "Analyst" in config["active_agents"]
        assert "Planner" in config["active_agents"]

    def test_tier_3_config(self):
        config = TIER_CONFIG[TierLevel.DEEP]
        assert config["name"] == "Deep"
        assert config["agent_count"] == 12
        assert config["requires_council"] is True
        assert config["requires_smes"] is True
        assert config["max_sme_count"] == 3
        assert "council_agents" in config
        assert "Domain Council Chair" in config["council_agents"]

    def test_tier_4_config(self):
        config = TIER_CONFIG[TierLevel.ADVERSARIAL]
        assert config["name"] == "Adversarial"
        assert config["agent_count"] == 18
        assert config["requires_council"] is True
        assert config["requires_smes"] is True
        assert config["max_sme_count"] == 3
        assert "Domain Council Chair" in config["council_agents"]
        assert "Quality Arbiter" in config["council_agents"]
        assert "Ethics & Safety Advisor" in config["council_agents"]

    def test_agent_count_increasing(self):
        counts = [TIER_CONFIG[t]["agent_count"] for t in TierLevel]
        for i in range(len(counts) - 1):
            assert counts[i] <= counts[i + 1]

    def test_all_configs_have_required_fields(self):
        required_fields = ["name", "description", "active_agents", "agent_count",
                          "requires_council", "requires_smes", "phases"]
        for tier in TierLevel:
            for field in required_fields:
                assert field in TIER_CONFIG[tier], f"Missing '{field}' in tier {tier}"


# =============================================================================
# Keyword Lists Tests
# =============================================================================

class TestKeywordLists:
    """Tests for keyword lists."""

    def test_tier_3_keywords_not_empty(self):
        assert len(TIER_3_KEYWORDS) > 0

    def test_tier_4_keywords_not_empty(self):
        assert len(TIER_4_KEYWORDS) > 0

    def test_escalation_keywords_not_empty(self):
        assert len(ESCALATION_KEYWORDS) > 0

    def test_tier_3_keywords_are_lowercase(self):
        for kw in TIER_3_KEYWORDS:
            assert kw == kw.lower(), f"Tier 3 keyword '{kw}' is not lowercase"

    def test_tier_4_keywords_are_lowercase(self):
        for kw in TIER_4_KEYWORDS:
            assert kw == kw.lower(), f"Tier 4 keyword '{kw}' is not lowercase"

    def test_escalation_keywords_are_lowercase(self):
        for kw in ESCALATION_KEYWORDS:
            assert kw == kw.lower(), f"Escalation keyword '{kw}' is not lowercase"

    def test_tier_3_contains_expected_domains(self):
        keywords_text = " ".join(TIER_3_KEYWORDS)
        assert "architecture" in keywords_text
        assert "machine learning" in keywords_text
        assert "security" in keywords_text

    def test_tier_4_contains_high_stakes(self):
        keywords_text = " ".join(TIER_4_KEYWORDS)
        assert "pii" in keywords_text
        assert "gdpr" in keywords_text
        assert "financial" in keywords_text


# =============================================================================
# classify_complexity Tests
# =============================================================================

class TestClassifyComplexity:
    """Tests for classify_complexity function."""

    def test_simple_prompt_returns_tier_1_or_2(self):
        result = classify_complexity("Hello world")
        assert result.tier in [TierLevel.DIRECT, TierLevel.STANDARD]

    def test_tier_3_keyword_match(self):
        result = classify_complexity("Design the system architecture for a microservices platform")
        assert result.tier >= TierLevel.DEEP
        assert len(result.keywords_found) > 0

    def test_tier_4_keyword_match(self):
        result = classify_complexity("Perform a security audit of personal data handling with GDPR compliance")
        assert result.tier == TierLevel.ADVERSARIAL
        assert len(result.keywords_found) > 0

    def test_tier_4_overrides_tier_3(self):
        result = classify_complexity("Security review of the architecture design")
        assert result.tier == TierLevel.ADVERSARIAL

    def test_escalation_keyword_raises_to_tier_2(self):
        result = classify_complexity("This is complex and I'm not sure about it")
        assert result.tier >= TierLevel.STANDARD

    def test_case_insensitive_matching(self):
        result = classify_complexity("SECURITY AUDIT of the system")
        assert result.tier == TierLevel.ADVERSARIAL

    def test_with_analyst_report_higher_tier(self):
        analyst_report = {"suggested_tier": 3, "escalation_needed": False}
        result = classify_complexity("Hello world", analyst_report)
        assert result.tier >= TierLevel.DEEP

    def test_with_analyst_report_lower_tier_no_override(self):
        analyst_report = {"suggested_tier": 1}
        result = classify_complexity("Security review of GDPR compliance", analyst_report)
        assert result.tier == TierLevel.ADVERSARIAL

    def test_analyst_report_with_escalation(self):
        analyst_report = {"suggested_tier": 2, "escalation_needed": True}
        result = classify_complexity("Simple request", analyst_report)
        assert result.tier >= TierLevel.STANDARD

    def test_returns_tier_classification_type(self):
        result = classify_complexity("Test prompt")
        assert isinstance(result, TierClassification)

    def test_reasoning_not_empty(self):
        result = classify_complexity("Build a REST API")
        assert len(result.reasoning) > 0

    def test_confidence_is_set(self):
        result = classify_complexity("Build a microservices architecture")
        assert result.confidence > 0

    def test_tier_3_higher_confidence(self):
        result_t3 = classify_complexity("Design a threat model for the application")
        result_simple = classify_complexity("Hello")
        assert result_t3.confidence >= result_simple.confidence

    def test_requires_council_on_tier_3(self):
        result = classify_complexity("Design the system architecture for a cloud migration")
        assert result.requires_council is True

    def test_no_council_on_tier_1(self):
        result = classify_complexity("Hello")
        assert result.requires_council is False

    def test_sme_on_tier_3(self):
        result = classify_complexity("Create a threat model for the IAM architecture")
        assert result.requires_smes is True

    def test_no_sme_on_tier_1(self):
        result = classify_complexity("Hello")
        assert result.requires_smes is False

    def test_estimated_agents_matches_tier(self):
        result = classify_complexity("Hello")
        config = TIER_CONFIG[result.tier]
        assert result.estimated_agents == config["agent_count"]

    def test_escalation_risk_with_keywords(self):
        result = classify_complexity("This is complex and I'm not sure about a multi-step process")
        assert result.escalation_risk > 0.1

    def test_empty_prompt(self):
        result = classify_complexity("")
        assert result.tier >= TierLevel.DIRECT

    def test_very_long_prompt(self):
        long_prompt = "Build a system " * 500
        result = classify_complexity(long_prompt)
        assert isinstance(result, TierClassification)

    def test_multiple_tier_4_keywords(self):
        result = classify_complexity("Security review of personal data with GDPR and HIPAA compliance for healthcare")
        assert result.tier == TierLevel.ADVERSARIAL
        assert len(result.keywords_found) >= 3

    def test_keywords_found_populated(self):
        result = classify_complexity("Perform a pentest on the cloud architecture")
        assert len(result.keywords_found) > 0

    def test_no_analyst_report(self):
        result = classify_complexity("Simple task", None)
        assert isinstance(result, TierClassification)


# =============================================================================
# should_escalate Tests
# =============================================================================

class TestShouldEscalate:
    """Tests for should_escalate function."""

    def test_explicit_escalation_flag(self):
        assert should_escalate(TierLevel.STANDARD, {"escalation_needed": True}) is True

    def test_no_escalation_needed(self):
        assert should_escalate(TierLevel.STANDARD, {"escalation_needed": False}) is False

    def test_escalation_indicator_domain_expertise(self):
        assert should_escalate(TierLevel.STANDARD, {"note": "domain expertise required"}) is True

    def test_escalation_indicator_need_specialist(self):
        assert should_escalate(TierLevel.STANDARD, {"note": "need specialist"}) is True

    def test_escalation_indicator_outside_scope(self):
        assert should_escalate(TierLevel.STANDARD, {"note": "outside scope"}) is True

    def test_escalation_indicator_requires_sme(self):
        assert should_escalate(TierLevel.STANDARD, {"note": "requires sme"}) is True

    def test_escalation_indicator_uncertain(self):
        assert should_escalate(TierLevel.STANDARD, {"note": "I'm uncertain about this"}) is True

    def test_escalation_indicator_need_verification(self):
        assert should_escalate(TierLevel.STANDARD, {"note": "need verification"}) is True

    def test_no_indicators_returns_false(self):
        assert should_escalate(TierLevel.STANDARD, {"note": "Everything is fine"}) is False

    def test_empty_feedback(self):
        assert should_escalate(TierLevel.STANDARD, {}) is False

    def test_already_at_tier_4_still_checks(self):
        result = should_escalate(TierLevel.ADVERSARIAL, {"escalation_needed": True})
        assert result is True

    def test_nested_data_in_feedback(self):
        feedback = {"inner": {"text": "requires sme consultation"}}
        assert should_escalate(TierLevel.STANDARD, feedback) is True


# =============================================================================
# get_escalated_tier Tests
# =============================================================================

class TestGetEscalatedTier:
    """Tests for get_escalated_tier function."""

    def test_tier_1_to_tier_2(self):
        assert get_escalated_tier(TierLevel.DIRECT) == TierLevel.STANDARD

    def test_tier_2_to_tier_3(self):
        assert get_escalated_tier(TierLevel.STANDARD) == TierLevel.DEEP

    def test_tier_3_to_tier_4(self):
        assert get_escalated_tier(TierLevel.DEEP) == TierLevel.ADVERSARIAL

    def test_tier_4_stays_at_tier_4(self):
        assert get_escalated_tier(TierLevel.ADVERSARIAL) == TierLevel.ADVERSARIAL

    def test_returns_tier_level(self):
        for tier in TierLevel:
            result = get_escalated_tier(tier)
            assert isinstance(result, TierLevel)


# =============================================================================
# estimate_agent_count Tests
# =============================================================================

class TestEstimateAgentCount:
    """Tests for estimate_agent_count function."""

    def test_tier_1_base_count(self):
        assert estimate_agent_count(TierLevel.DIRECT) == 3

    def test_tier_2_base_count(self):
        assert estimate_agent_count(TierLevel.STANDARD) == 7

    def test_tier_3_base_count(self):
        assert estimate_agent_count(TierLevel.DEEP) == 12

    def test_tier_4_base_count(self):
        assert estimate_agent_count(TierLevel.ADVERSARIAL) == 18

    def test_with_smes(self):
        assert estimate_agent_count(TierLevel.DEEP, sme_count=2) == 14

    def test_with_zero_smes(self):
        assert estimate_agent_count(TierLevel.STANDARD, sme_count=0) == 7

    def test_with_max_smes(self):
        assert estimate_agent_count(TierLevel.ADVERSARIAL, sme_count=3) == 21


# =============================================================================
# get_active_agents Tests
# =============================================================================

class TestGetActiveAgents:
    """Tests for get_active_agents function."""

    def test_tier_1_agents(self):
        agents = get_active_agents(TierLevel.DIRECT)
        assert "Executor" in agents
        assert "Formatter" in agents

    def test_tier_2_agents(self):
        agents = get_active_agents(TierLevel.STANDARD)
        assert "Analyst" in agents
        assert "Planner" in agents
        assert "Executor" in agents

    def test_returns_copy(self):
        agents1 = get_active_agents(TierLevel.DIRECT)
        agents2 = get_active_agents(TierLevel.DIRECT)
        agents1.append("NewAgent")
        assert "NewAgent" not in agents2

    def test_all_tiers_return_lists(self):
        for tier in TierLevel:
            agents = get_active_agents(tier)
            assert isinstance(agents, list)


# =============================================================================
# get_council_agents Tests
# =============================================================================

class TestGetCouncilAgents:
    """Tests for get_council_agents function."""

    def test_tier_1_no_council(self):
        assert get_council_agents(TierLevel.DIRECT) == []

    def test_tier_2_no_council(self):
        assert get_council_agents(TierLevel.STANDARD) == []

    def test_tier_3_has_chair(self):
        agents = get_council_agents(TierLevel.DEEP)
        assert "Domain Council Chair" in agents

    def test_tier_4_has_full_council(self):
        agents = get_council_agents(TierLevel.ADVERSARIAL)
        assert "Domain Council Chair" in agents
        assert "Quality Arbiter" in agents
        assert "Ethics & Safety Advisor" in agents

    def test_returns_list(self):
        for tier in TierLevel:
            assert isinstance(get_council_agents(tier), list)
