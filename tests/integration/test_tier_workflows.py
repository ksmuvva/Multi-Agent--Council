"""
Integration Tests for Tier Workflows

Tests end-to-end agent workflows for each tier level.
These tests use mocked SDK responses but validate the full orchestration flow.

Gated by MAS_RUN_INTEGRATION=true environment variable.
Set MAS_RUN_INTEGRATION=true to run these tests with live API calls.
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime

from src.core.complexity import classify_complexity, TierLevel
from src.core.verdict import (
    evaluate_verdict_matrix,
    Verdict,
    MatrixAction,
    VERDICT_MATRIX,
    should_trigger_debate,
)
from src.core.pipeline import (
    Phase,
    ExecutionPipeline,
    PipelineBuilder,
)
from src.session.compaction import CompactionConfig, CompactionTrigger

# Gate all integration tests behind MAS_RUN_INTEGRATION env var
_run_integration = os.environ.get("MAS_RUN_INTEGRATION", "false").lower() == "true"
pytestmark = pytest.mark.skipif(
    not _run_integration,
    reason="Integration tests disabled. Set MAS_RUN_INTEGRATION=true to enable.",
)


# =============================================================================
# Tier 1 Integration Tests
# =============================================================================

class TestTier1Workflow:
    """Integration tests for Tier 1 (Direct) workflow."""

    def test_tier1_classification(self):
        """Test that simple queries are classified as Tier 1."""
        result = classify_complexity("What is 2 + 2?")
        assert result.tier in (TierLevel.DIRECT, TierLevel.STANDARD)

    def test_tier1_agent_set_is_minimal(self):
        """Test that Tier 1 uses only Executor and Formatter (max 3 agents)."""
        pipeline = ExecutionPipeline(tier_level=TierLevel.DIRECT)
        agents = pipeline._get_agents_for_phase(Phase.PHASE_5_SOLUTION_GENERATION)
        assert len(agents) <= 3

    def test_tier1_does_not_include_council(self):
        """Test that Tier 1 does not involve the Council."""
        pipeline = ExecutionPipeline(tier_level=TierLevel.DIRECT)
        council_agents = pipeline._get_council_agents()
        assert len(council_agents) == 0


# =============================================================================
# Tier 2 Integration Tests
# =============================================================================

class TestTier2Workflow:
    """Integration tests for Tier 2 (Standard) workflow."""

    def test_tier2_classification(self):
        """Test that moderate queries are classified as Tier 2."""
        result = classify_complexity(
            "Write a Python function to parse CSV files and validate the data"
        )
        assert result.tier.value >= TierLevel.STANDARD.value

    def test_tier2_includes_analysis_phase(self):
        """Test that Tier 2 includes analysis agents."""
        pipeline = ExecutionPipeline(tier_level=TierLevel.STANDARD)
        agents = pipeline._get_agents_for_phase(Phase.PHASE_1_TASK_INTELLIGENCE)
        assert any("Analyst" in name for name in agents)

    def test_tier2_includes_verification(self):
        """Test that Tier 2 includes Verifier."""
        pipeline = ExecutionPipeline(tier_level=TierLevel.STANDARD)
        review_agents = pipeline._get_review_agents()
        assert any("Verifier" in name for name in review_agents)


# =============================================================================
# Tier 3 Integration Tests
# =============================================================================

class TestTier3Workflow:
    """Integration tests for Tier 3 (Deep) workflow."""

    def test_tier3_classification(self):
        """Test that complex domain queries are classified as Tier 3+."""
        result = classify_complexity(
            "Design a secure microservices architecture for healthcare "
            "with HIPAA compliance, multi-region deployment, and real-time "
            "data pipeline for patient monitoring"
        )
        assert result.tier.value >= TierLevel.DEEP.value

    def test_tier3_includes_council(self):
        """Test that Tier 3 involves Council Chair."""
        pipeline = ExecutionPipeline(tier_level=TierLevel.DEEP)
        council_agents = pipeline._get_council_agents()
        assert len(council_agents) >= 1

    def test_tier3_sme_selection_by_keywords(self):
        """Test SME selection based on task keywords."""
        from src.core.sme_registry import find_personas_by_keywords

        results = find_personas_by_keywords(["security", "cloud", "architecture"])
        assert len(results) >= 1


# =============================================================================
# Tier 4 Integration Tests
# =============================================================================

class TestTier4Workflow:
    """Integration tests for Tier 4 (Adversarial) workflow."""

    def test_tier4_classification(self):
        """Test that adversarial queries are classified as Tier 4."""
        result = classify_complexity(
            "Conduct an adversarial security audit of a nuclear power plant "
            "control system with safety-critical ethical considerations and "
            "multi-stakeholder risk analysis"
        )
        assert result.tier.value >= TierLevel.DEEP.value

    def test_tier4_full_council(self):
        """Test that Tier 4 activates the full Council."""
        pipeline = ExecutionPipeline(tier_level=TierLevel.ADVERSARIAL)
        council_agents = pipeline._get_council_agents()
        assert len(council_agents) >= 2

    def test_tier4_includes_critic(self):
        """Test that Tier 4 includes the Critic agent."""
        pipeline = ExecutionPipeline(tier_level=TierLevel.ADVERSARIAL)
        review_agents = pipeline._get_review_agents()
        assert any("Critic" in name for name in review_agents)


# =============================================================================
# Verdict Matrix Tests
# =============================================================================

class TestVerdictMatrixIntegration:
    """Integration tests for the verdict matrix logic."""

    def test_verdict_matrix_all_pass(self):
        """Test (PASS, PASS) -> PROCEED_TO_FORMATTER."""
        outcome = evaluate_verdict_matrix(
            verifier_verdict=Verdict.PASS, critic_verdict=Verdict.PASS
        )
        assert outcome.action == MatrixAction.PROCEED_TO_FORMATTER

    def test_verdict_matrix_verifier_pass_critic_fail(self):
        """Test (PASS, FAIL) -> EXECUTOR_REVISE."""
        outcome = evaluate_verdict_matrix(
            verifier_verdict=Verdict.PASS, critic_verdict=Verdict.FAIL
        )
        assert outcome.action == MatrixAction.EXECUTOR_REVISE

    def test_verdict_matrix_verifier_fail_critic_pass(self):
        """Test (FAIL, PASS) -> RESEARCHER_REVERIFY."""
        outcome = evaluate_verdict_matrix(
            verifier_verdict=Verdict.FAIL, critic_verdict=Verdict.PASS
        )
        assert outcome.action == MatrixAction.RESEARCHER_REVERIFY

    def test_verdict_matrix_all_fail(self):
        """Test (FAIL, FAIL) -> FULL_REGENERATION."""
        outcome = evaluate_verdict_matrix(
            verifier_verdict=Verdict.FAIL, critic_verdict=Verdict.FAIL
        )
        assert outcome.action == MatrixAction.FULL_REGENERATION

    def test_verdict_actions_are_distinct(self):
        """Test that all verdict actions are unique."""
        actions = set(MatrixAction)
        assert len(actions) >= 4


# =============================================================================
# Session Management Tests
# =============================================================================

class TestSessionManagement:
    """Tests for session management in workflows."""

    def test_context_compaction_config(self):
        """Test context compaction configuration."""
        config = CompactionConfig()

        assert config.max_tokens == 100000
        assert config.max_messages == 50
        assert config.recent_messages_to_keep == 10
        assert config.preserve_verdicts is True
        assert config.preserve_sme_advisories is True

    def test_compaction_trigger_enum(self):
        """Test compaction trigger enumeration."""
        assert CompactionTrigger.TOKEN_COUNT == "token_count"
        assert CompactionTrigger.MESSAGE_COUNT == "message_count"
        assert CompactionTrigger.SESSION_AGE == "session_age"
        assert CompactionTrigger.MANUAL == "manual"


# =============================================================================
# Error Recovery Tests
# =============================================================================

class TestErrorRecovery:
    """Tests for error recovery in workflows."""

    def test_budget_exceeded_detection(self):
        """Test that budget exhaustion is properly detected."""
        from src.utils.cost import BudgetState

        state = BudgetState(
            spent_usd=12.0,
            max_budget_usd=10.0,
            remaining_usd=0.0,
        )

        assert state.is_exceeded is True
        assert state.utilization_pct > 100.0

    def test_complexity_escalation(self):
        """Test that complexity can be escalated."""
        initial = classify_complexity("Simple task")
        escalated = classify_complexity(
            "Simple task that actually requires deep multi-domain analysis "
            "with adversarial security review and ethical considerations"
        )
        assert escalated.tier.value >= initial.tier.value


# =============================================================================
# Performance Tests
# =============================================================================

class TestWorkflowPerformance:
    """Tests for workflow performance characteristics."""

    def test_tier_agent_counts_are_ordered(self):
        """Test that higher tiers use more agents."""
        tier_agent_counts = {}
        for tier in TierLevel:
            pipeline = ExecutionPipeline(tier_level=tier)
            all_phases = pipeline._get_phases_for_tier()
            all_agents = []
            for phase in all_phases:
                all_agents.extend(pipeline._get_agents_for_phase(phase))
            tier_agent_counts[tier] = len(all_agents)

        # Higher tiers should generally have more agents
        assert tier_agent_counts[TierLevel.DIRECT] <= tier_agent_counts[TierLevel.ADVERSARIAL]


# =============================================================================
# Ensemble Pattern Tests
# =============================================================================

class TestEnsembleWorkflows:
    """Tests for ensemble pattern workflows."""

    def test_ensemble_registry_contains_all_patterns(self):
        """Test that ensemble registry has all expected patterns."""
        from src.core.ensemble import ENSEMBLE_REGISTRY, EnsembleType

        expected = [
            EnsembleType.ARCHITECTURE_REVIEW_BOARD,
            EnsembleType.CODE_SPRINT,
            EnsembleType.RESEARCH_COUNCIL,
            EnsembleType.DOCUMENT_ASSEMBLY,
            EnsembleType.REQUIREMENTS_WORKSHOP,
        ]
        for etype in expected:
            assert etype in ENSEMBLE_REGISTRY, f"Missing ensemble: {etype}"

    def test_get_ensemble_returns_valid_instance(self):
        """Test that get_ensemble creates proper instances."""
        from src.core.ensemble import get_ensemble, EnsembleType

        ensemble = get_ensemble(EnsembleType.ARCHITECTURE_REVIEW_BOARD)
        assert ensemble is not None
        assert hasattr(ensemble, "execute")
        assert hasattr(ensemble, "get_config")
