"""
Unit Tests for Orchestrator Agent

Tests for the main orchestrator including session integration,
tier classification, and agent coordination.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from src.agents.orchestrator import (
    OrchestratorAgent,
    AgentExecution,
    SessionState,
    create_orchestrator,
)
from src.core.complexity import TierLevel


class TestOrchestratorAgent:
    """Tests for OrchestratorAgent initialization and configuration."""

    def test_orchestrator_initialization(self):
        """Test that orchestrator initializes with correct defaults."""
        orchestrator = OrchestratorAgent()

        assert orchestrator.max_budget_usd == 5.0
        assert orchestrator.max_revisions == 2
        assert orchestrator.max_debate_rounds == 2
        assert orchestrator.verbose is False
        assert orchestrator.enable_persistence is True
        assert orchestrator.enable_auto_compact is True

    def test_orchestrator_custom_config(self):
        """Test orchestrator with custom configuration."""
        orchestrator = OrchestratorAgent(
            max_budget_usd=10.0,
            max_revisions=5,
            max_debate_rounds=3,
            verbose=True,
            enable_persistence=False,
        )

        assert orchestrator.max_budget_usd == 10.0
        assert orchestrator.max_revisions == 5
        assert orchestrator.max_debate_rounds == 3
        assert orchestrator.verbose is True
        assert orchestrator.enable_persistence is False

    def test_orchestrator_model_configuration(self):
        """Test that models are configured correctly."""
        orchestrator = OrchestratorAgent()

        assert orchestrator.orchestrator_model is not None
        assert orchestrator.council_model is not None
        assert orchestrator.operational_model is not None
        assert orchestrator.sme_model is not None

    def test_orchestrator_max_turns(self):
        """Test max turns configuration."""
        orchestrator = OrchestratorAgent()

        assert orchestrator.max_turns_orchestrator > 0
        assert orchestrator.max_turns_subagent > 0
        assert orchestrator.max_turns_executor > 0
        assert orchestrator.max_turns_executor >= orchestrator.max_turns_subagent


class TestSessionState:
    """Tests for internal SessionState dataclass."""

    def test_session_state_creation(self):
        """Test creating a session state."""
        session = SessionState(
            session_id="test_123",
            user_prompt="Test prompt",
        )

        assert session.session_id == "test_123"
        assert session.user_prompt == "Test prompt"
        assert session.current_tier == TierLevel.STANDARD
        assert session.revision_cycle == 0
        assert session.total_cost_usd == 0.0

    def test_session_state_budget_tracking(self):
        """Test budget tracking properties."""
        session = SessionState(
            session_id="budget_test",
            user_prompt="Test",
            max_budget_usd=10.0,
        )

        # Initially under budget
        assert session.budget_utilization == 0.0
        assert session.is_budget_exceeded() is False
        assert session.should_warn_budget() is False

        # Add some cost
        session.total_cost_usd = 5.0
        assert session.budget_utilization == 0.5
        assert session.is_budget_exceeded() is False

        # Exceed budget
        session.total_cost_usd = 12.0
        assert session.is_budget_exceeded() is True

    def test_session_state_budget_warning(self):
        """Test budget warning threshold."""
        session = SessionState(
            session_id="warn_test",
            user_prompt="Test",
            max_budget_usd=10.0,
            budget_warning_threshold=0.8,
        )

        # Below warning threshold
        session.total_cost_usd = 7.0
        assert session.should_warn_budget() is False

        # At warning threshold
        session.total_cost_usd = 8.0
        assert session.should_warn_budget() is True

    def test_session_state_duration(self):
        """Test session duration calculation."""
        import time

        session = SessionState(
            session_id="duration_test",
            user_prompt="Test",
        )

        # Session has just started
        initial_duration = session.duration_seconds
        assert initial_duration >= 0

        # Simulate some time passing
        session.end_time = time.time()
        final_duration = session.duration_seconds
        assert final_duration >= 0


class TestAgentExecution:
    """Tests for AgentExecution tracking."""

    def test_agent_execution_creation(self):
        """Test creating an agent execution record."""
        execution = AgentExecution(
            agent_name="Analyst",
            start_time=1234567890.0,
        )

        assert execution.agent_name == "Analyst"
        assert execution.start_time == 1234567890.0
        assert execution.status == "pending"
        assert execution.end_time is None
        assert execution.output is None
        assert execution.error is None

    def test_agent_execution_completion(self):
        """Test marking an execution as complete."""
        execution = AgentExecution(
            agent_name="Executor",
            start_time=1234567890.0,
        )

        execution.end_time = 1234567900.0
        execution.status = "complete"
        execution.output = {"result": "success"}
        execution.tokens_used = 1000
        execution.cost_usd = 0.01

        assert execution.status == "complete"
        assert execution.end_time == 1234567900.0
        assert execution.output["result"] == "success"
        assert execution.tokens_used == 1000
        assert execution.cost_usd == 0.01


class TestOrchestratorProcessRequest:
    """Tests for the main process_request method."""

    @patch('src.agents.orchestrator.classify_complexity')
    def test_process_request_tier_classification(self, mock_classify):
        """Test that process_request classifies complexity."""
        from src.core.complexity import TierClassification

        mock_classify.return_value = TierClassification(
            tier=TierLevel.STANDARD,
            reasoning="Standard complexity",
            confidence=0.8,
            estimated_agents=7,
            requires_council=False,
            requires_smes=False,
        )

        orchestrator = OrchestratorAgent()
        orchestrator.verbose = False

        # Process request (will use mock classification)
        # Note: Full execution requires pipeline implementation
        result = orchestrator.process_request(
            user_prompt="Test prompt",
            session_id="test_session",
        )

        # Verify response structure
        assert "response" in result
        assert "session_id" in result
        assert "metadata" in result
        assert result["session_id"] == "test_session"

    def test_process_request_with_tier_override(self):
        """Test process_request with tier override."""
        orchestrator = OrchestratorAgent()

        result = orchestrator.process_request(
            user_prompt="Test",
            tier_override=1,  # Force Tier 1
        )

        assert result["metadata"]["tier"] == 1

    @patch('src.agents.orchestrator.classify_complexity')
    def test_process_request_budget_check(self, mock_classify):
        """Test that budget is checked during processing."""
        from src.core.complexity import TierClassification

        mock_classify.return_value = TierClassification(
            tier=TierLevel.STANDARD,
            reasoning="Test",
            confidence=0.8,
            estimated_agents=7,
            requires_council=False,
            requires_smes=False,
        )

        orchestrator = OrchestratorAgent(max_budget_usd=0.001)  # Very low budget

        result = orchestrator.process_request(
            user_prompt="Test",
        )

        # Should get budget exceeded response
        assert "budget" in result["metadata"].get("error", "").lower()


class TestOrchestratorExecute:
    """Tests for the execute method (CLI entry point)."""

    @patch('src.agents.orchestrator.classify_complexity')
    def test_execute_default_format(self, mock_classify):
        """Test execute with default markdown format."""
        from src.core.complexity import TierClassification

        mock_classify.return_value = TierClassification(
            tier=TierLevel.STANDARD,
            reasoning="Test",
            confidence=0.8,
            estimated_agents=7,
            requires_council=False,
            requires_smes=False,
        )

        orchestrator = OrchestratorAgent()

        result = orchestrator.execute(
            user_prompt="Test prompt",
            tier_level=2,
            format="markdown",
        )

        assert "formatted_output" in result
        assert "raw_output" in result
        assert "summary" in result
        assert "duration_seconds" in result
        assert "total_cost_usd" in result

    @patch('src.agents.orchestrator.classify_complexity')
    def test_execute_json_format(self, mock_classify):
        """Test execute with JSON format."""
        from src.core.complexity import TierClassification

        mock_classify.return_value = TierClassification(
            tier=TierLevel.STANDARD,
            reasoning="Test",
            confidence=0.8,
            estimated_agents=7,
            requires_council=False,
            requires_smes=False,
        )

        orchestrator = OrchestratorAgent()

        result = orchestrator.execute(
            user_prompt="Test prompt",
            tier_level=2,
            format="json",
        )

        assert "formatted_output" in result
        # JSON output should start with {
        assert result["formatted_output"].strip().startswith("{")

    @patch('src.agents.orchestrator.classify_complexity')
    def test_execute_text_format(self, mock_classify):
        """Test execute with plain text format."""
        from src.core.complexity import TierClassification

        mock_classify.return_value = TierClassification(
            tier=TierLevel.STANDARD,
            reasoning="Test",
            confidence=0.8,
            estimated_agents=7,
            requires_council=False,
            requires_smes=False,
        )

        orchestrator = OrchestratorAgent()

        result = orchestrator.execute(
            user_prompt="Test prompt",
            tier_level=2,
            format="text",
        )

        assert "formatted_output" in result
        # Text format should be just the response
        assert "Session ID" not in result["formatted_output"]


class TestOrchestratorSessionIntegration:
    """Tests for orchestrator session persistence integration."""

    def test_save_session_state_with_persistence_enabled(self):
        """Test that session state is saved when persistence is enabled."""
        orchestrator = OrchestratorAgent(enable_persistence=True)

        session = SessionState(
            session_id="save_test",
            user_prompt="Test",
        )
        session.agent_executions.append(AgentExecution(
            agent_name="Analyst",
            start_time=1234567890.0,
            status="complete",
            output="Test output",
        ))

        # This should save without error
        # (In real test, we'd mock the persistence layer)
        orchestrator._save_session_state(session, {"response": "Test"})

    def test_save_session_state_with_persistence_disabled(self):
        """Test that nothing happens when persistence is disabled."""
        orchestrator = OrchestratorAgent(enable_persistence=False)

        session = SessionState(
            session_id="no_save_test",
            user_prompt="Test",
        )

        # Should not raise any errors
        orchestrator._save_session_state(session, None)

    def test_load_session_with_persistence_enabled(self):
        """Test loading a session."""
        orchestrator = OrchestratorAgent(enable_persistence=True)

        # Load non-existent session should return None
        # (In real test, we'd mock the persistence layer)
        result = orchestrator.load_session("nonexistent_session")
        # Result might be None or raise error depending on implementation

    def test_load_session_with_persistence_disabled(self):
        """Test that load returns None when persistence is disabled."""
        orchestrator = OrchestratorAgent(enable_persistence=False)

        result = orchestrator.load_session("any_session")
        assert result is None


class TestCreateOrchestrator:
    """Tests for the convenience factory function."""

    def test_create_orchestrator_default(self):
        """Test creating orchestrator with defaults."""
        orchestrator = create_orchestrator()

        assert isinstance(orchestrator, OrchestratorAgent)
        assert orchestrator.max_budget_usd == 5.0
        assert orchestrator.verbose is False

    def test_create_orchestrator_with_config(self):
        """Test creating orchestrator with custom config."""
        orchestrator = create_orchestrator(
            max_budget_usd=15.0,
            verbose=True,
        )

        assert isinstance(orchestrator, OrchestratorAgent)
        assert orchestrator.max_budget_usd == 15.0
        assert orchestrator.verbose is True
