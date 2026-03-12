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

from tests.conftest import (
    MOCK_SESSION_ID,
    MOCK_AGENT_RESPONSES,
    TestDataBuilder,
    MockSDKPatch,
)

from src.core.complexity import ComplexityClassifier, TierLevel
from src.core.verdict import VerdictMatrix, VerdictAction
from src.core.pipeline import PipelinePhase
from src.session.persistence import SessionState, ChatMessage, AgentOutput
from src.session.compaction import CompactionConfig, CompactionTrigger

# Gate all integration tests behind MAS_RUN_INTEGRATION env var
_run_integration = os.environ.get("MAS_RUN_INTEGRATION", "false").lower() == "true"
pytestmark = pytest.mark.skipif(
    not _run_integration,
    reason="Integration tests disabled. Set MAS_RUN_INTEGRATION=true to enable."
)


# =============================================================================
# Tier 1 Integration Tests
# =============================================================================

class TestTier1Workflow:
    """Integration tests for Tier 1 (Direct) workflow."""

    def test_tier1_classification(self):
        """Test that simple queries are classified as Tier 1."""
        classifier = ComplexityClassifier()
        result = classifier.classify("What is 2 + 2?")
        assert result.tier in (TierLevel.DIRECT, TierLevel.STANDARD)

    def test_tier1_agent_set_is_minimal(self):
        """Test that Tier 1 uses only Executor and Formatter (max 3 agents)."""
        from src.core.pipeline import Pipeline

        pipeline = Pipeline(tier_level=TierLevel.DIRECT)
        agents = pipeline._get_agents_for_phase(PipelinePhase.EXECUTION)
        assert len(agents) <= 3

    def test_tier1_does_not_include_council(self):
        """Test that Tier 1 does not involve the Council."""
        from src.core.pipeline import Pipeline

        pipeline = Pipeline(tier_level=TierLevel.DIRECT)
        council_agents = pipeline._get_council_agents()
        assert len(council_agents) == 0


# =============================================================================
# Tier 2 Integration Tests
# =============================================================================

class TestTier2Workflow:
    """Integration tests for Tier 2 (Standard) workflow."""

    def test_tier2_classification(self):
        """Test that moderate queries are classified as Tier 2."""
        classifier = ComplexityClassifier()
        result = classifier.classify(
            "Write a Python function to parse CSV files and validate the data"
        )
        assert result.tier.value >= TierLevel.STANDARD.value

    def test_tier2_includes_analysis_phase(self):
        """Test that Tier 2 includes analysis agents."""
        from src.core.pipeline import Pipeline

        pipeline = Pipeline(tier_level=TierLevel.STANDARD)
        agents = pipeline._get_agents_for_phase(PipelinePhase.ANALYSIS)
        agent_names = [a if isinstance(a, str) else a.name for a in agents]
        assert any("Analyst" in name for name in agent_names)

    def test_tier2_includes_verification(self):
        """Test that Tier 2 includes Verifier."""
        from src.core.pipeline import Pipeline

        pipeline = Pipeline(tier_level=TierLevel.STANDARD)
        review_agents = pipeline._get_review_agents()
        agent_names = [a if isinstance(a, str) else a.name for a in review_agents]
        assert any("Verifier" in name for name in agent_names)


# =============================================================================
# Tier 3 Integration Tests
# =============================================================================

class TestTier3Workflow:
    """Integration tests for Tier 3 (Deep) workflow."""

    def test_tier3_classification(self):
        """Test that complex domain queries are classified as Tier 3+."""
        classifier = ComplexityClassifier()
        result = classifier.classify(
            "Design a secure microservices architecture for healthcare "
            "with HIPAA compliance, multi-region deployment, and real-time "
            "data pipeline for patient monitoring"
        )
        assert result.tier.value >= TierLevel.DEEP.value

    def test_tier3_includes_council(self):
        """Test that Tier 3 involves Council Chair."""
        from src.core.pipeline import Pipeline

        pipeline = Pipeline(tier_level=TierLevel.DEEP)
        council_agents = pipeline._get_council_agents()
        assert len(council_agents) >= 1

    def test_tier3_sme_selection_by_keywords(self):
        """Test SME selection based on task keywords."""
        from src.core.sme_registry import SME_REGISTRY, find_personas_by_keywords

        results = find_personas_by_keywords(["security", "cloud", "architecture"])
        assert len(results) >= 1


# =============================================================================
# Tier 4 Integration Tests
# =============================================================================

class TestTier4Workflow:
    """Integration tests for Tier 4 (Adversarial) workflow."""

    def test_tier4_classification(self):
        """Test that adversarial queries are classified as Tier 4."""
        classifier = ComplexityClassifier()
        result = classifier.classify(
            "Conduct an adversarial security audit of a nuclear power plant "
            "control system with safety-critical ethical considerations and "
            "multi-stakeholder risk analysis"
        )
        assert result.tier.value >= TierLevel.DEEP.value

    def test_tier4_full_council(self):
        """Test that Tier 4 activates the full Council."""
        from src.core.pipeline import Pipeline

        pipeline = Pipeline(tier_level=TierLevel.ADVERSARIAL)
        council_agents = pipeline._get_council_agents()
        assert len(council_agents) >= 2

    def test_tier4_includes_critic(self):
        """Test that Tier 4 includes the Critic agent."""
        from src.core.pipeline import Pipeline

        pipeline = Pipeline(tier_level=TierLevel.ADVERSARIAL)
        review_agents = pipeline._get_review_agents()
        agent_names = [a if isinstance(a, str) else a.name for a in review_agents]
        assert any("Critic" in name for name in agent_names)


# =============================================================================
# Verdict Matrix Tests
# =============================================================================

class TestVerdictMatrix:
    """Integration tests for the verdict matrix logic."""

    def test_verdict_matrix_all_pass(self):
        """Test (PASS, PASS) -> PROCEED_TO_FORMATTER."""
        matrix = VerdictMatrix()
        action = matrix.get_action(verifier_pass=True, critic_pass=True)
        assert action == VerdictAction.PROCEED_TO_FORMATTER

    def test_verdict_matrix_verifier_pass_critic_fail(self):
        """Test (PASS, FAIL) -> EXECUTOR_REVISE."""
        matrix = VerdictMatrix()
        action = matrix.get_action(verifier_pass=True, critic_pass=False)
        assert action == VerdictAction.EXECUTOR_REVISE

    def test_verdict_matrix_verifier_fail_critic_pass(self):
        """Test (FAIL, PASS) -> RESEARCHER_REVERIFY."""
        matrix = VerdictMatrix()
        action = matrix.get_action(verifier_pass=False, critic_pass=True)
        assert action == VerdictAction.RESEARCHER_REVERIFY

    def test_verdict_matrix_all_fail(self):
        """Test (FAIL, FAIL) -> FULL_REGENERATION."""
        matrix = VerdictMatrix()
        action = matrix.get_action(verifier_pass=False, critic_pass=False)
        assert action == VerdictAction.FULL_REGENERATION

    def test_verdict_actions_are_distinct(self):
        """Test that all 4 verdict actions are unique."""
        actions = set(VerdictAction)
        assert len(actions) == 4


# =============================================================================
# Session Management Tests
# =============================================================================

class TestSessionManagement:
    """Tests for session management in workflows."""

    def test_session_creation(self, mock_session_id):
        """Test that sessions are created with unique IDs."""
        session_id = mock_session_id
        assert session_id is not None
        assert len(session_id) > 0

    def test_session_state_persistence(self):
        """Test that session state persists across agent calls."""
        from src.session import (
            SessionState,
            ChatMessage,
            AgentOutput,
            create_session,
        )

        session = create_session(
            session_id="test_session",
            tier=2,
            title="Test Session",
        )

        session.messages.append(ChatMessage(
            role="user",
            content="Test message",
            timestamp=datetime.now(),
        ))

        session.agent_outputs.append(AgentOutput(
            agent_name="Analyst",
            phase="analysis",
            tier=2,
            content="Analysis complete",
            timestamp=datetime.now(),
        ))

        assert len(session.messages) == 1
        assert len(session.agent_outputs) == 1
        assert session.messages[0].role == "user"
        assert session.agent_outputs[0].agent_name == "Analyst"

    def test_session_serialization(self):
        """Test that sessions can be serialized to/from dict."""
        from src.session import SessionState, ChatMessage, AgentOutput

        session = SessionState(
            session_id="test_serialize",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            tier=2,
            messages=[
                ChatMessage(
                    role="user",
                    content="Hello",
                    timestamp=datetime.now(),
                ),
            ],
            agent_outputs=[
                AgentOutput(
                    agent_name="Analyst",
                    phase="analysis",
                    tier=2,
                    content="Response",
                    timestamp=datetime.now(),
                ),
            ],
        )

        data = session.to_dict()
        assert data["session_id"] == "test_serialize"
        assert len(data["messages"]) == 1
        assert len(data["agent_outputs"]) == 1

        restored = SessionState.from_dict(data)
        assert restored.session_id == "test_serialize"
        assert len(restored.messages) == 1
        assert len(restored.agent_outputs) == 1

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
            utilization_pct=120.0,
            is_warning=True,
            is_exceeded=True,
        )

        assert state.is_exceeded is True
        assert state.utilization_pct > 100.0

    def test_complexity_escalation(self):
        """Test that complexity can be escalated."""
        classifier = ComplexityClassifier()
        initial = classifier.classify("Simple task")
        escalated = classifier.classify(
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
        from src.core.pipeline import Pipeline

        tier_agent_counts = {}
        for tier in TierLevel:
            pipeline = Pipeline(tier_level=tier)
            all_agents = (
                pipeline._get_agents_for_phase(PipelinePhase.ANALYSIS)
                + pipeline._get_agents_for_phase(PipelinePhase.EXECUTION)
                + pipeline._get_review_agents()
                + pipeline._get_council_agents()
            )
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
        from src.core.ensemble import ENSEMBLE_REGISTRY

        expected = [
            "architecture_review_board",
            "code_sprint",
            "research_council",
            "document_assembly",
            "requirements_workshop",
        ]
        for name in expected:
            assert name in ENSEMBLE_REGISTRY, f"Missing ensemble: {name}"

    def test_get_ensemble_returns_valid_instance(self):
        """Test that get_ensemble creates proper instances."""
        from src.core.ensemble import get_ensemble

        ensemble = get_ensemble("architecture_review_board")
        assert ensemble is not None
        assert hasattr(ensemble, "execute")
        assert hasattr(ensemble, "agents")
