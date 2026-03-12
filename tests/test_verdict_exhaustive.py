"""
Exhaustive Tests for Verdict Matrix Module

Tests all enums, models, matrix logic, phase mapping,
debate triggering, agent requirements, and cost estimates.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from pydantic import ValidationError

from src.core.verdict import (
    Verdict,
    MatrixAction,
    MatrixOutcome,
    VERDICT_MATRIX,
    evaluate_verdict_matrix,
    get_phase_for_action,
    should_trigger_debate,
    DebateConfig,
    get_required_agents_for_phase,
    calculate_phase_cost_estimate,
)


# =============================================================================
# Verdict Enum Tests
# =============================================================================

class TestVerdict:
    """Tests for Verdict enum."""

    def test_pass_value(self):
        assert Verdict.PASS == "PASS"

    def test_fail_value(self):
        assert Verdict.FAIL == "FAIL"

    def test_count(self):
        assert len(Verdict) == 2


# =============================================================================
# MatrixAction Enum Tests
# =============================================================================

class TestMatrixAction:
    """Tests for MatrixAction enum."""

    def test_all_actions(self):
        assert MatrixAction.PROCEED_TO_FORMATTER == "proceed_to_formatter"
        assert MatrixAction.EXECUTOR_REVISE == "executor_revise"
        assert MatrixAction.RESEARCHER_REVERIFY == "researcher_reverify"
        assert MatrixAction.FULL_REGENERATION == "full_regeneration"
        assert MatrixAction.QUALITY_ARBITER == "quality_arbiter"

    def test_count(self):
        assert len(MatrixAction) == 5


# =============================================================================
# VERDICT_MATRIX Tests
# =============================================================================

class TestVerdictMatrix:
    """Tests for the verdict matrix mapping."""

    def test_pass_pass(self):
        assert VERDICT_MATRIX[(Verdict.PASS, Verdict.PASS)] == MatrixAction.PROCEED_TO_FORMATTER

    def test_pass_fail(self):
        assert VERDICT_MATRIX[(Verdict.PASS, Verdict.FAIL)] == MatrixAction.EXECUTOR_REVISE

    def test_fail_pass(self):
        assert VERDICT_MATRIX[(Verdict.FAIL, Verdict.PASS)] == MatrixAction.RESEARCHER_REVERIFY

    def test_fail_fail(self):
        assert VERDICT_MATRIX[(Verdict.FAIL, Verdict.FAIL)] == MatrixAction.FULL_REGENERATION

    def test_all_combinations_covered(self):
        for v1 in Verdict:
            for v2 in Verdict:
                assert (v1, v2) in VERDICT_MATRIX


# =============================================================================
# MatrixOutcome Model Tests
# =============================================================================

class TestMatrixOutcome:
    """Tests for MatrixOutcome Pydantic model."""

    def test_valid_outcome(self):
        outcome = MatrixOutcome(
            verifier_verdict=Verdict.PASS,
            critic_verdict=Verdict.PASS,
            action=MatrixAction.PROCEED_TO_FORMATTER,
            reason="Both passed",
            revision_cycle=0,
            can_retry=True,
        )
        assert outcome.max_revisions == 2  # default

    def test_revision_cycle_min(self):
        with pytest.raises(ValidationError):
            MatrixOutcome(
                verifier_verdict=Verdict.PASS, critic_verdict=Verdict.PASS,
                action=MatrixAction.PROCEED_TO_FORMATTER, reason="Test",
                revision_cycle=-1, can_retry=True,
            )


# =============================================================================
# evaluate_verdict_matrix Tests
# =============================================================================

class TestEvaluateVerdictMatrix:
    """Tests for evaluate_verdict_matrix function."""

    def test_pass_pass_proceeds(self):
        result = evaluate_verdict_matrix(Verdict.PASS, Verdict.PASS)
        assert result.action == MatrixAction.PROCEED_TO_FORMATTER
        assert result.can_retry is True

    def test_pass_fail_revises(self):
        result = evaluate_verdict_matrix(Verdict.PASS, Verdict.FAIL)
        assert result.action == MatrixAction.EXECUTOR_REVISE

    def test_fail_pass_reverifies(self):
        result = evaluate_verdict_matrix(Verdict.FAIL, Verdict.PASS)
        assert result.action == MatrixAction.RESEARCHER_REVERIFY

    def test_fail_fail_regenerates(self):
        result = evaluate_verdict_matrix(Verdict.FAIL, Verdict.FAIL)
        assert result.action == MatrixAction.FULL_REGENERATION

    def test_can_retry_under_max(self):
        result = evaluate_verdict_matrix(Verdict.PASS, Verdict.FAIL, revision_cycle=0, max_revisions=2)
        assert result.can_retry is True

    def test_cannot_retry_at_max(self):
        result = evaluate_verdict_matrix(Verdict.PASS, Verdict.FAIL, revision_cycle=2, max_revisions=2)
        assert result.can_retry is False

    def test_quality_arbiter_on_tier4_at_max(self):
        result = evaluate_verdict_matrix(
            Verdict.FAIL, Verdict.FAIL, revision_cycle=2, max_revisions=2, tier_level=4
        )
        assert result.action == MatrixAction.QUALITY_ARBITER

    def test_no_arbiter_below_tier4(self):
        result = evaluate_verdict_matrix(
            Verdict.FAIL, Verdict.FAIL, revision_cycle=2, max_revisions=2, tier_level=3
        )
        assert result.action == MatrixAction.FULL_REGENERATION

    def test_reasoning_contains_verdicts(self):
        result = evaluate_verdict_matrix(Verdict.PASS, Verdict.FAIL)
        assert "PASS" in result.reason
        assert "FAIL" in result.reason

    def test_revision_cycle_stored(self):
        result = evaluate_verdict_matrix(Verdict.PASS, Verdict.FAIL, revision_cycle=1)
        assert result.revision_cycle == 1

    def test_returns_matrix_outcome(self):
        result = evaluate_verdict_matrix(Verdict.PASS, Verdict.PASS)
        assert isinstance(result, MatrixOutcome)

    def test_default_revision_cycle_zero(self):
        result = evaluate_verdict_matrix(Verdict.PASS, Verdict.PASS)
        assert result.revision_cycle == 0

    def test_default_tier_level_2(self):
        result = evaluate_verdict_matrix(Verdict.FAIL, Verdict.FAIL, revision_cycle=5, max_revisions=2)
        assert result.action != MatrixAction.QUALITY_ARBITER  # tier_level defaults to 2


# =============================================================================
# get_phase_for_action Tests
# =============================================================================

class TestGetPhaseForAction:
    """Tests for get_phase_for_action function."""

    def test_proceed_to_formatter(self):
        assert "Phase 8" in get_phase_for_action(MatrixAction.PROCEED_TO_FORMATTER)

    def test_executor_revise(self):
        assert "Phase 7" in get_phase_for_action(MatrixAction.EXECUTOR_REVISE)

    def test_researcher_reverify(self):
        assert "Phase 4" in get_phase_for_action(MatrixAction.RESEARCHER_REVERIFY)

    def test_full_regeneration(self):
        assert "Phase 5" in get_phase_for_action(MatrixAction.FULL_REGENERATION)

    def test_quality_arbiter(self):
        assert "Arbiter" in get_phase_for_action(MatrixAction.QUALITY_ARBITER)

    def test_all_actions_have_phases(self):
        for action in MatrixAction:
            result = get_phase_for_action(action)
            assert result != "Unknown phase"


# =============================================================================
# should_trigger_debate Tests
# =============================================================================

class TestShouldTriggerDebate:
    """Tests for should_trigger_debate function."""

    def test_tier_4_always_debates(self):
        outcome = MatrixOutcome(
            verifier_verdict=Verdict.PASS, critic_verdict=Verdict.PASS,
            action=MatrixAction.PROCEED_TO_FORMATTER, reason="Both pass",
            revision_cycle=0, can_retry=True,
        )
        assert should_trigger_debate(outcome, tier_level=4) is True

    def test_disagreement_triggers_debate(self):
        outcome = MatrixOutcome(
            verifier_verdict=Verdict.PASS, critic_verdict=Verdict.FAIL,
            action=MatrixAction.EXECUTOR_REVISE, reason="Disagreement",
            revision_cycle=0, can_retry=True,
        )
        assert should_trigger_debate(outcome, tier_level=2) is True

    def test_agreement_no_debate_tier2(self):
        outcome = MatrixOutcome(
            verifier_verdict=Verdict.PASS, critic_verdict=Verdict.PASS,
            action=MatrixAction.PROCEED_TO_FORMATTER, reason="Agreement",
            revision_cycle=0, can_retry=True,
        )
        assert should_trigger_debate(outcome, tier_level=2) is False

    def test_fail_fail_max_revisions_triggers_debate(self):
        outcome = MatrixOutcome(
            verifier_verdict=Verdict.FAIL, critic_verdict=Verdict.FAIL,
            action=MatrixAction.FULL_REGENERATION, reason="Both fail",
            revision_cycle=2, can_retry=False,
        )
        assert should_trigger_debate(outcome, tier_level=3) is True


# =============================================================================
# DebateConfig Model Tests
# =============================================================================

class TestDebateConfig:
    """Tests for DebateConfig Pydantic model."""

    def test_valid_config(self):
        config = DebateConfig(
            participants=["Executor", "Critic", "Verifier"]
        )
        assert config.max_rounds == 2
        assert config.consensus_threshold == 0.8
        assert config.current_round == 0

    def test_max_rounds_bounds(self):
        with pytest.raises(ValidationError):
            DebateConfig(max_rounds=0, participants=["A"])
        with pytest.raises(ValidationError):
            DebateConfig(max_rounds=6, participants=["A"])

    def test_consensus_threshold_bounds(self):
        with pytest.raises(ValidationError):
            DebateConfig(consensus_threshold=1.5, participants=["A"])


# =============================================================================
# get_required_agents_for_phase Tests
# =============================================================================

class TestGetRequiredAgentsForPhase:
    """Tests for get_required_agents_for_phase function."""

    def test_phase_1(self):
        agents = get_required_agents_for_phase("Phase 1", tier=2)
        assert "Task Analyst" in agents

    def test_phase_2_tier_3(self):
        agents = get_required_agents_for_phase("Phase 2", tier=3)
        assert "Domain Council Chair" in agents

    def test_phase_2_tier_2(self):
        agents = get_required_agents_for_phase("Phase 2", tier=2)
        assert agents == []

    def test_phase_5(self):
        agents = get_required_agents_for_phase("Phase 5", tier=2)
        assert "Executor" in agents

    def test_phase_8(self):
        agents = get_required_agents_for_phase("Phase 8", tier=2)
        assert "Reviewer" in agents
        assert "Formatter" in agents

    def test_unknown_phase(self):
        agents = get_required_agents_for_phase("Phase 99", tier=2)
        assert agents == []


# =============================================================================
# calculate_phase_cost_estimate Tests
# =============================================================================

class TestCalculatePhaseCostEstimate:
    """Tests for calculate_phase_cost_estimate function."""

    def test_returns_float(self):
        cost = calculate_phase_cost_estimate(tier=2, phase="Phase 5")
        assert isinstance(cost, float)

    def test_cost_non_negative(self):
        for tier in range(1, 5):
            for phase in ["Phase 1", "Phase 5", "Phase 8"]:
                cost = calculate_phase_cost_estimate(tier=tier, phase=phase)
                assert cost >= 0

    def test_higher_tier_higher_cost(self):
        cost_t1 = calculate_phase_cost_estimate(tier=1, phase="Phase 5")
        cost_t4 = calculate_phase_cost_estimate(tier=4, phase="Phase 5")
        assert cost_t4 >= cost_t1

    def test_empty_phase_returns_zero(self):
        cost = calculate_phase_cost_estimate(tier=2, phase="Phase 99")
        assert cost == 0.0
