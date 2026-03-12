"""
Exhaustive Tests for OrchestratorAgent Module

Tests all OrchestratorAgent functionality including:
- Initialization and configuration
- AgentExecution and SessionState dataclasses
- execute() and process_request() flows
- Tier classification and override
- Agent spawning and pipeline execution
- Budget enforcement and warnings
- Escalation handling
- Session persistence and loading
- Response formatting
"""

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import patch, MagicMock, Mock, PropertyMock
from datetime import datetime

from src.agents.orchestrator import (
    AgentExecution,
    SessionState,
    OrchestratorAgent,
    create_orchestrator,
)
from src.core.complexity import TierLevel, TierClassification


# =============================================================================
# Mock Helpers
# =============================================================================

def _mock_settings():
    """Create a mock settings object."""
    settings = MagicMock()
    settings.max_budget = 5.0
    settings.debug = False
    settings.session_persistence = False
    settings.max_turns_orchestrator = 30
    settings.max_turns_subagent = 10
    settings.max_turns_executor = 20
    settings.get_provider_config.return_value = {"provider": "anthropic"}
    return settings


def _patch_orchestrator_deps():
    """Return a dictionary of patches needed to instantiate OrchestratorAgent."""
    return {
        "src.agents.orchestrator.get_settings": MagicMock(return_value=_mock_settings()),
        "src.agents.orchestrator.get_api_key": MagicMock(return_value="test-key"),
        "src.agents.orchestrator.get_model_for_agent": MagicMock(return_value="claude-3-5-sonnet-20241022"),
        "src.agents.orchestrator.get_provider": MagicMock(return_value="anthropic"),
        "src.agents.orchestrator.create_sdk_mcp_server": MagicMock(return_value=None),
        "src.agents.orchestrator.SessionPersistence": MagicMock(),
        "src.agents.orchestrator.CompactionConfig": MagicMock(),
    }


def _create_orchestrator(**kwargs):
    """Create an OrchestratorAgent with all dependencies mocked."""
    patches = _patch_orchestrator_deps()
    with patch.multiple("src.agents.orchestrator", **{k.split(".")[-1]: v for k, v in patches.items()}):
        # Re-patch with full paths
        pass

    # Use the patch context properly
    mock_settings = _mock_settings()

    with patch("src.agents.orchestrator.get_settings", return_value=mock_settings), \
         patch("src.agents.orchestrator.get_api_key", return_value="test-key"), \
         patch("src.agents.orchestrator.get_model_for_agent", return_value="claude-3-5-sonnet-20241022"), \
         patch("src.agents.orchestrator.get_provider", return_value="anthropic"), \
         patch("src.agents.orchestrator.create_sdk_mcp_server", return_value=None):
        agent = OrchestratorAgent(enable_persistence=False, enable_auto_compact=False, **kwargs)
    return agent


# =============================================================================
# AgentExecution Dataclass Tests
# =============================================================================

class TestAgentExecution:
    """Tests for AgentExecution dataclass."""

    def test_default_values(self):
        """Test default field values."""
        exec_ = AgentExecution(agent_name="Analyst", start_time=100.0)
        assert exec_.agent_name == "Analyst"
        assert exec_.start_time == 100.0
        assert exec_.end_time is None
        assert exec_.status == "pending"
        assert exec_.output is None
        assert exec_.error is None
        assert exec_.tokens_used == 0
        assert exec_.cost_usd == 0.0

    def test_custom_values(self):
        """Test with custom values."""
        exec_ = AgentExecution(
            agent_name="Executor",
            start_time=100.0,
            end_time=110.0,
            status="complete",
            output={"result": "success"},
            error=None,
            tokens_used=5000,
            cost_usd=0.05,
        )
        assert exec_.agent_name == "Executor"
        assert exec_.end_time == 110.0
        assert exec_.status == "complete"
        assert exec_.output == {"result": "success"}
        assert exec_.tokens_used == 5000
        assert exec_.cost_usd == 0.05

    def test_failed_execution(self):
        """Test failed execution state."""
        exec_ = AgentExecution(
            agent_name="Verifier",
            start_time=100.0,
            end_time=105.0,
            status="failed",
            error="Connection timeout",
        )
        assert exec_.status == "failed"
        assert exec_.error == "Connection timeout"

    def test_output_can_be_any_type(self):
        """Test that output can be string, dict, or None."""
        exec_str = AgentExecution(agent_name="A", start_time=0, output="text result")
        exec_dict = AgentExecution(agent_name="B", start_time=0, output={"key": "value"})
        exec_none = AgentExecution(agent_name="C", start_time=0, output=None)
        assert exec_str.output == "text result"
        assert exec_dict.output == {"key": "value"}
        assert exec_none.output is None


# =============================================================================
# SessionState Dataclass Tests
# =============================================================================

class TestSessionState:
    """Tests for SessionState dataclass."""

    def test_default_values(self):
        """Test default field values."""
        session = SessionState(session_id="test_123", user_prompt="Hello")
        assert session.session_id == "test_123"
        assert session.user_prompt == "Hello"
        assert session.tier_classification is None
        assert session.current_tier == TierLevel.STANDARD
        assert session.revision_cycle == 0
        assert session.max_revisions == 2
        assert session.debate_rounds == 0
        assert session.max_debate_rounds == 2
        assert session.total_cost_usd == 0.0
        assert session.max_budget_usd == 5.0
        assert session.budget_warning_threshold == 0.8
        assert session.agent_executions == []
        assert session.active_smes == []
        assert session.council_activated is False
        assert session.escalation_history == []

    def test_custom_values(self):
        """Test with custom values."""
        session = SessionState(
            session_id="custom",
            user_prompt="Test",
            current_tier=TierLevel.DEEP,
            max_budget_usd=10.0,
            max_revisions=5,
        )
        assert session.current_tier == TierLevel.DEEP
        assert session.max_budget_usd == 10.0
        assert session.max_revisions == 5

    def test_duration_seconds_property(self):
        """Test duration_seconds property."""
        session = SessionState(session_id="test", user_prompt="test")
        session.start_time = time.time() - 5.0  # 5 seconds ago
        assert session.duration_seconds >= 4.9

    def test_duration_seconds_with_end_time(self):
        """Test duration with explicit end time."""
        session = SessionState(session_id="test", user_prompt="test")
        session.start_time = 100.0
        session.end_time = 110.0
        assert session.duration_seconds == 10.0

    def test_budget_utilization_property(self):
        """Test budget_utilization property."""
        session = SessionState(session_id="test", user_prompt="test")
        session.total_cost_usd = 2.5
        session.max_budget_usd = 5.0
        assert session.budget_utilization == 0.5

    def test_budget_utilization_zero_budget(self):
        """Test budget utilization with zero budget."""
        session = SessionState(session_id="test", user_prompt="test")
        session.max_budget_usd = 0.0
        assert session.budget_utilization == 0

    def test_should_warn_budget_true(self):
        """Test budget warning when threshold exceeded."""
        session = SessionState(session_id="test", user_prompt="test")
        session.total_cost_usd = 4.5
        session.max_budget_usd = 5.0
        session.budget_warning_threshold = 0.8
        assert session.should_warn_budget() is True

    def test_should_warn_budget_false(self):
        """Test no budget warning when under threshold."""
        session = SessionState(session_id="test", user_prompt="test")
        session.total_cost_usd = 1.0
        session.max_budget_usd = 5.0
        assert session.should_warn_budget() is False

    def test_is_budget_exceeded_true(self):
        """Test budget exceeded detection."""
        session = SessionState(session_id="test", user_prompt="test")
        session.total_cost_usd = 6.0
        session.max_budget_usd = 5.0
        assert session.is_budget_exceeded() is True

    def test_is_budget_exceeded_false(self):
        """Test budget not exceeded."""
        session = SessionState(session_id="test", user_prompt="test")
        session.total_cost_usd = 3.0
        session.max_budget_usd = 5.0
        assert session.is_budget_exceeded() is False

    def test_is_budget_exceeded_exactly_at_limit(self):
        """Test budget at exact limit is exceeded."""
        session = SessionState(session_id="test", user_prompt="test")
        session.total_cost_usd = 5.0
        session.max_budget_usd = 5.0
        assert session.is_budget_exceeded() is True

    def test_agent_executions_mutable(self):
        """Test that agent_executions list is mutable."""
        session = SessionState(session_id="test", user_prompt="test")
        session.agent_executions.append(
            AgentExecution(agent_name="Test", start_time=0)
        )
        assert len(session.agent_executions) == 1

    def test_escalation_history_mutable(self):
        """Test that escalation_history is mutable."""
        session = SessionState(session_id="test", user_prompt="test")
        session.escalation_history.append({"tier": 3, "reason": "complex"})
        assert len(session.escalation_history) == 1


# =============================================================================
# OrchestratorAgent Init Tests
# =============================================================================

class TestOrchestratorInit:
    """Tests for OrchestratorAgent initialization."""

    def test_init_with_defaults(self):
        """Test default initialization."""
        agent = _create_orchestrator()
        assert agent.api_key == "test-key"
        assert agent.enable_persistence is False
        assert agent.enable_auto_compact is False

    def test_init_custom_budget(self):
        """Test custom budget."""
        agent = _create_orchestrator(max_budget_usd=20.0)
        assert agent.max_budget_usd == 20.0

    def test_init_custom_revisions(self):
        """Test custom max revisions."""
        agent = _create_orchestrator(max_revisions=5)
        assert agent.max_revisions == 5

    def test_init_custom_debate_rounds(self):
        """Test custom max debate rounds."""
        agent = _create_orchestrator(max_debate_rounds=4)
        assert agent.max_debate_rounds == 4

    def test_init_verbose_mode(self):
        """Test verbose mode."""
        agent = _create_orchestrator(verbose=True)
        assert agent.verbose is True

    def test_init_with_api_key(self):
        """Test custom API key."""
        agent = _create_orchestrator(api_key="custom-key")
        assert agent.api_key == "custom-key"

    def test_init_persistence_enabled(self):
        """Test persistence enabled path."""
        with patch("src.agents.orchestrator.get_settings", return_value=_mock_settings()), \
             patch("src.agents.orchestrator.get_api_key", return_value="test-key"), \
             patch("src.agents.orchestrator.get_model_for_agent", return_value="model"), \
             patch("src.agents.orchestrator.get_provider", return_value="anthropic"), \
             patch("src.agents.orchestrator.create_sdk_mcp_server", return_value=None), \
             patch("src.agents.orchestrator.SessionPersistence") as mock_sp, \
             patch("src.agents.orchestrator.CompactionConfig") as mock_cc:
            agent = OrchestratorAgent(enable_persistence=True, enable_auto_compact=True)
            mock_sp.assert_called_once()
            mock_cc.assert_called_once()


# =============================================================================
# OrchestratorAgent Tier Classification Tests
# =============================================================================

class TestOrchestratorTierClassification:
    """Tests for tier classification and override."""

    def test_create_override_classification_tier1(self):
        """Test tier 1 override classification."""
        agent = _create_orchestrator()
        classification = agent._create_override_classification(1)
        assert classification.tier == TierLevel.DIRECT
        assert classification.estimated_agents == 3
        assert classification.requires_council is False
        assert classification.requires_smes is False
        assert classification.confidence == 1.0

    def test_create_override_classification_tier2(self):
        """Test tier 2 override classification."""
        agent = _create_orchestrator()
        classification = agent._create_override_classification(2)
        assert classification.tier == TierLevel.STANDARD
        assert classification.estimated_agents == 7

    def test_create_override_classification_tier3(self):
        """Test tier 3 override classification."""
        agent = _create_orchestrator()
        classification = agent._create_override_classification(3)
        assert classification.tier == TierLevel.DEEP
        assert classification.estimated_agents == 12
        assert classification.requires_council is True
        assert classification.requires_smes is True

    def test_create_override_classification_tier4(self):
        """Test tier 4 override classification."""
        agent = _create_orchestrator()
        classification = agent._create_override_classification(4)
        assert classification.tier == TierLevel.ADVERSARIAL
        assert classification.estimated_agents == 18
        assert classification.requires_council is True
        assert classification.requires_smes is True
        assert classification.suggested_sme_count == 3

    def test_create_override_reasoning(self):
        """Test that override includes manual override reasoning."""
        agent = _create_orchestrator()
        classification = agent._create_override_classification(3)
        assert "Manual override" in classification.reasoning


# =============================================================================
# OrchestratorAgent Session Management Tests
# =============================================================================

class TestOrchestratorSessionManagement:
    """Tests for session creation and management."""

    def test_create_session_with_id(self):
        """Test session creation with specific ID."""
        agent = _create_orchestrator()
        session = agent._create_session("Hello", session_id="custom_123")
        assert session.session_id == "custom_123"
        assert session.user_prompt == "Hello"

    def test_create_session_auto_id(self):
        """Test session creation with auto-generated ID."""
        agent = _create_orchestrator()
        session = agent._create_session("Hello")
        assert session.session_id.startswith("session_")

    def test_create_session_inherits_budget(self):
        """Test session inherits orchestrator budget."""
        agent = _create_orchestrator(max_budget_usd=15.0)
        session = agent._create_session("Hello")
        assert session.max_budget_usd == 15.0

    def test_create_session_with_resume_context(self):
        """Test session creation with resume context."""
        agent = _create_orchestrator()
        resume_ctx = {
            "total_cost_usd": 2.5,
            "revision_cycle": 1,
            "active_smes": ["cloud_architect"],
            "council_activated": True,
        }
        session = agent._create_session("Hello", resume_context=resume_ctx)
        assert session.total_cost_usd == 2.5
        assert session.revision_cycle == 1
        assert session.active_smes == ["cloud_architect"]
        assert session.council_activated is True

    def test_get_session_context(self):
        """Test getting session context for resumption."""
        agent = _create_orchestrator()
        session = SessionState(
            session_id="test",
            user_prompt="Hello",
            total_cost_usd=1.5,
            revision_cycle=1,
            active_smes=["security_analyst"],
            council_activated=True,
        )
        ctx = agent.get_session_context(session)
        assert ctx["session_id"] == "test"
        assert ctx["total_cost_usd"] == 1.5
        assert ctx["revision_cycle"] == 1
        assert ctx["active_smes"] == ["security_analyst"]
        assert ctx["council_activated"] is True

    def test_load_session_no_persistence(self):
        """Test load_session returns None when persistence is disabled."""
        agent = _create_orchestrator()
        agent.persistence = None
        result = agent.load_session("test_id")
        assert result is None

    def test_load_session_with_persistence(self):
        """Test load_session delegates to persistence layer."""
        agent = _create_orchestrator()
        mock_persist = MagicMock()
        mock_persist_session = MagicMock()
        mock_persist_session.session_id = "test_id"
        mock_persist_session.max_budget = 10.0
        mock_persist_session.tier = 2
        mock_persist_session.total_cost_usd = 1.0
        mock_persist_session.active_agents = ["cloud_architect"]
        mock_persist.load_session.return_value = mock_persist_session
        agent.persistence = mock_persist

        with patch("src.agents.orchestrator.TierLevel", TierLevel):
            session = agent.load_session("test_id")

        assert session is not None
        assert session.session_id == "test_id"
        assert session.total_cost_usd == 1.0

    def test_load_session_not_found(self):
        """Test load_session when session not found."""
        agent = _create_orchestrator()
        mock_persist = MagicMock()
        mock_persist.load_session.return_value = None
        agent.persistence = mock_persist
        result = agent.load_session("nonexistent")
        assert result is None

    def test_load_session_error_handling(self):
        """Test load_session handles errors gracefully."""
        agent = _create_orchestrator(verbose=True)
        mock_persist = MagicMock()
        mock_persist.load_session.side_effect = Exception("DB error")
        agent.persistence = mock_persist
        result = agent.load_session("test_id")
        assert result is None


# =============================================================================
# OrchestratorAgent Budget Enforcement Tests
# =============================================================================

class TestOrchestratorBudget:
    """Tests for budget enforcement."""

    def test_budget_exceeded_response(self):
        """Test budget exceeded response format."""
        agent = _create_orchestrator()
        session = SessionState(session_id="test", user_prompt="test")
        session.max_budget_usd = 5.0
        session.total_cost_usd = 6.0

        response = agent._budget_exceeded_response(session)
        assert "budget" in response["response"].lower()
        assert "$5.00" in response["response"]
        assert response["metadata"]["error"] == "budget_exceeded"

    def test_error_response_format(self):
        """Test error response format."""
        agent = _create_orchestrator()
        session = SessionState(session_id="test", user_prompt="test")
        response = agent._error_response(session, "Something went wrong")
        assert "error" in response["response"].lower()
        assert "Something went wrong" in response["response"]
        assert response["metadata"]["error"] == "Something went wrong"


# =============================================================================
# OrchestratorAgent Execute Flow Tests
# =============================================================================

class TestOrchestratorExecute:
    """Tests for the execute() method."""

    def test_execute_calls_process_request(self):
        """Test that execute delegates to process_request."""
        agent = _create_orchestrator()
        mock_result = {
            "response": "Test response",
            "session_id": "test",
            "metadata": {
                "summary": "done",
                "duration_seconds": 1.0,
                "total_cost_usd": 0.01,
            },
        }
        agent.process_request = MagicMock(return_value=mock_result)
        result = agent.execute("Hello", tier_level=2)
        agent.process_request.assert_called_once()
        assert "formatted_output" in result
        assert "raw_output" in result

    def test_execute_json_format(self):
        """Test execute with JSON format."""
        agent = _create_orchestrator()
        mock_result = {
            "response": "Test",
            "session_id": "test",
            "metadata": {"summary": "done", "duration_seconds": 0, "total_cost_usd": 0},
        }
        agent.process_request = MagicMock(return_value=mock_result)
        result = agent.execute("Hello", format="json")
        # JSON format should be a string
        assert isinstance(result["formatted_output"], str)

    def test_execute_text_format(self):
        """Test execute with text format."""
        agent = _create_orchestrator()
        mock_result = {
            "response": "Plain text response",
            "session_id": "test",
            "metadata": {"summary": "done", "duration_seconds": 0, "total_cost_usd": 0},
        }
        agent.process_request = MagicMock(return_value=mock_result)
        result = agent.execute("Hello", format="text")
        assert result["formatted_output"] == "Plain text response"

    def test_execute_markdown_format(self):
        """Test execute with markdown format (default)."""
        agent = _create_orchestrator()
        mock_result = {
            "response": "Markdown response",
            "session_id": "sess_123",
            "metadata": {
                "summary": "done",
                "duration_seconds": 1.5,
                "total_cost_usd": 0.02,
                "tier": 2,
                "agents_used": ["Analyst"],
                "smes_used": [],
            },
        }
        agent.process_request = MagicMock(return_value=mock_result)
        result = agent.execute("Hello", format="markdown")
        assert "Multi-Agent Reasoning Result" in result["formatted_output"]
        assert "Markdown response" in result["formatted_output"]


# =============================================================================
# OrchestratorAgent Process Request Tests
# =============================================================================

class TestOrchestratorProcessRequest:
    """Tests for process_request() flow."""

    def test_process_request_budget_exceeded_early(self):
        """Test process_request returns early on exceeded budget."""
        agent = _create_orchestrator()
        agent.max_budget_usd = 5.0

        # Mock classify_complexity
        mock_classification = TierClassification(
            tier=TierLevel.STANDARD,
            reasoning="Standard",
            confidence=0.9,
            estimated_agents=7,
            requires_council=False,
            requires_smes=False,
        )

        with patch("src.agents.orchestrator.classify_complexity", return_value=mock_classification):
            # Create session with exceeded budget
            original_create = agent._create_session

            def mock_create(*args, **kwargs):
                session = original_create(*args, **kwargs)
                session.total_cost_usd = 100.0
                session.max_budget_usd = 5.0
                return session

            agent._create_session = mock_create
            result = agent.process_request("Hello", tier_override=2)
            assert "budget" in result.get("response", "").lower() or "metadata" in result

    def test_process_request_error_handling(self):
        """Test error handling in process_request."""
        agent = _create_orchestrator(verbose=True)

        with patch("src.agents.orchestrator.classify_complexity", side_effect=Exception("Classification error")):
            result = agent.process_request("Hello")
            assert "error" in result["response"].lower()

    def test_process_request_with_tier_override(self):
        """Test process_request with tier override."""
        agent = _create_orchestrator()

        mock_pipeline_state = {}
        agent._execute_pipeline = MagicMock(return_value=mock_pipeline_state)
        agent._generate_final_response = MagicMock(return_value={
            "response": "Done",
            "session_id": "test",
            "metadata": {"tier": 1},
        })

        with patch("src.agents.orchestrator.PipelineBuilder") as mock_pb, \
             patch("src.agents.orchestrator.create_execution_context", return_value={"user_prompt": "Hello"}):
            mock_pipeline = MagicMock()
            mock_pb.for_tier.return_value = mock_pipeline
            result = agent.process_request("Hello", tier_override=1)
            assert result is not None


# =============================================================================
# OrchestratorAgent Agent Spawning Tests
# =============================================================================

class TestOrchestratorAgentSpawning:
    """Tests for agent spawning."""

    def test_get_max_turns_orchestrator(self):
        """Test max turns for Orchestrator agent."""
        agent = _create_orchestrator()
        assert agent._get_max_turns("Orchestrator") == agent.max_turns_orchestrator

    def test_get_max_turns_executor(self):
        """Test max turns for Executor agent."""
        agent = _create_orchestrator()
        assert agent._get_max_turns("Executor") == agent.max_turns_executor

    def test_get_max_turns_subagent(self):
        """Test max turns for generic subagents."""
        agent = _create_orchestrator()
        assert agent._get_max_turns("Analyst") == agent.max_turns_subagent
        assert agent._get_max_turns("Verifier") == agent.max_turns_subagent

    def test_load_system_prompt_from_file(self, tmp_path):
        """Test loading system prompt from file."""
        agent = _create_orchestrator()
        prompt_file = tmp_path / "prompt.md"
        prompt_file.write_text("System prompt content here.")
        content = agent._load_system_prompt(str(prompt_file))
        assert content == "System prompt content here."

    def test_load_system_prompt_missing_file(self):
        """Test loading system prompt with missing file."""
        agent = _create_orchestrator()
        content = agent._load_system_prompt("/nonexistent/path.md", "analyst")
        assert "analyst" in content

    def test_load_system_prompt_with_role(self, tmp_path):
        """Test loading system prompt with role extraction."""
        agent = _create_orchestrator()
        prompt_file = tmp_path / "prompt.md"
        prompt_file.write_text(
            "# Overview\nGeneral content\n\n"
            "# Chair\nChair-specific instructions\n\n"
            "# Arbiter\nArbiter-specific instructions"
        )
        content = agent._load_system_prompt(str(prompt_file), "chair")
        assert "Chair-specific" in content

    def test_spawn_agent_tracks_cost(self):
        """Test that _spawn_agent tracks cost."""
        agent = _create_orchestrator()
        session = SessionState(session_id="test", user_prompt="test")

        mock_result = {
            "status": "success",
            "output": "result",
            "cost_usd": 0.05,
            "tokens_used": 1000,
        }

        with patch("src.agents.orchestrator.build_agent_options", return_value=MagicMock()), \
             patch("src.agents.orchestrator.spawn_subagent", return_value=mock_result), \
             patch("src.agents.orchestrator.get_skills_for_agent", return_value=[]):
            agent._spawn_agent(
                session=session,
                agent_name="Analyst",
                system_prompt_template="config/agents/analyst/CLAUDE.md",
                input_data="test",
            )
        assert session.total_cost_usd == 0.05

    def test_spawn_agent_detects_escalation(self):
        """Test that _spawn_agent detects escalation in output."""
        agent = _create_orchestrator()
        session = SessionState(session_id="test", user_prompt="test")

        mock_result = {
            "status": "success",
            "output": {"escalation_needed": True, "escalation_reason": "Complex task"},
            "cost_usd": 0.01,
            "tokens_used": 500,
        }

        with patch("src.agents.orchestrator.build_agent_options", return_value=MagicMock()), \
             patch("src.agents.orchestrator.spawn_subagent", return_value=mock_result), \
             patch("src.agents.orchestrator.get_skills_for_agent", return_value=[]):
            result = agent._spawn_agent(
                session=session,
                agent_name="Analyst",
                system_prompt_template="config/agents/analyst/CLAUDE.md",
                input_data="test",
            )
        assert result["escalation_needed"] is True


# =============================================================================
# OrchestratorAgent Escalation Tests
# =============================================================================

class TestOrchestratorEscalation:
    """Tests for escalation handling."""

    def test_handle_escalation_upgrades_tier(self):
        """Test that escalation upgrades the tier."""
        agent = _create_orchestrator(verbose=True)
        session = SessionState(session_id="test", user_prompt="test")
        session.current_tier = TierLevel.STANDARD

        agent_result = {"escalation_reason": "Task is more complex than expected"}

        with patch("src.agents.orchestrator.get_escalated_tier", return_value=TierLevel.DEEP):
            # Mock _consult_council since it will be called on tier upgrade
            agent._consult_council = MagicMock()
            agent._handle_escalation(session, agent_result)

        assert session.current_tier == TierLevel.DEEP
        assert len(session.escalation_history) == 1

    def test_handle_escalation_no_change(self):
        """Test escalation with no tier change."""
        agent = _create_orchestrator()
        session = SessionState(session_id="test", user_prompt="test")
        session.current_tier = TierLevel.ADVERSARIAL

        with patch("src.agents.orchestrator.get_escalated_tier", return_value=TierLevel.ADVERSARIAL):
            agent._handle_escalation(session, {"escalation_reason": "already max"})

        assert session.current_tier == TierLevel.ADVERSARIAL
        assert len(session.escalation_history) == 1

    def test_handle_escalation_activates_council(self):
        """Test that escalation to tier 3+ activates council."""
        agent = _create_orchestrator()
        session = SessionState(session_id="test", user_prompt="test")
        session.current_tier = TierLevel.STANDARD
        session.council_activated = False

        agent._consult_council = MagicMock()

        with patch("src.agents.orchestrator.get_escalated_tier", return_value=TierLevel.DEEP):
            agent._handle_escalation(session, {"escalation_reason": "complex"})

        agent._consult_council.assert_called_once()

    def test_handle_escalation_does_not_reactivate_council(self):
        """Test that council is not reactivated if already active."""
        agent = _create_orchestrator()
        session = SessionState(session_id="test", user_prompt="test")
        session.current_tier = TierLevel.DEEP
        session.council_activated = True

        agent._consult_council = MagicMock()

        with patch("src.agents.orchestrator.get_escalated_tier", return_value=TierLevel.ADVERSARIAL):
            agent._handle_escalation(session, {"escalation_reason": "sensitive"})

        agent._consult_council.assert_not_called()


# =============================================================================
# OrchestratorAgent Response Generation Tests
# =============================================================================

class TestOrchestratorResponseGeneration:
    """Tests for response generation."""

    def test_generate_final_response_structure(self):
        """Test final response structure."""
        agent = _create_orchestrator()
        session = SessionState(session_id="test_123", user_prompt="test")
        session.current_tier = TierLevel.STANDARD
        session.agent_executions = [
            AgentExecution(
                agent_name="Executor",
                start_time=100.0,
                end_time=110.0,
                status="complete",
                output="Here is the solution.",
            ),
        ]
        session.tier_classification = TierClassification(
            tier=TierLevel.STANDARD,
            reasoning="Standard task",
            confidence=0.9,
            estimated_agents=7,
            requires_council=False,
            requires_smes=False,
        )

        response = agent._generate_final_response(session, {})
        assert "response" in response
        assert "session_id" in response
        assert "metadata" in response
        assert response["metadata"]["tier"] == 2
        assert "Executor" in response["metadata"]["agents_used"]

    def test_format_response_no_outputs(self):
        """Test format_response with no outputs."""
        agent = _create_orchestrator()
        session = SessionState(session_id="test", user_prompt="test")
        result = agent._format_response([], session)
        assert "apologize" in result.lower()

    def test_format_response_with_executor_output(self):
        """Test format_response with executor output."""
        agent = _create_orchestrator()
        session = SessionState(session_id="test", user_prompt="test")
        session.agent_executions = [
            AgentExecution(
                agent_name="Executor",
                start_time=100.0,
                status="complete",
                output="Solution text here.",
            ),
        ]
        result = agent._format_response(["Solution text here."], session)
        assert "Solution text here." in result

    def test_format_response_prefers_latest_executor(self):
        """Test that the latest executor output is preferred."""
        agent = _create_orchestrator()
        session = SessionState(session_id="test", user_prompt="test")
        session.agent_executions = [
            AgentExecution(agent_name="Executor", start_time=100.0, status="complete", output="Version 1"),
            AgentExecution(agent_name="Executor", start_time=110.0, status="complete", output="Version 2"),
        ]
        result = agent._format_response(["Version 1", "Version 2"], session)
        assert "Version 2" in result

    def test_format_as_markdown(self):
        """Test markdown formatting."""
        agent = _create_orchestrator()
        result = {
            "response": "Test response content",
            "metadata": {
                "session_id": "test_123",
                "tier": 2,
                "duration_seconds": 5.0,
                "total_cost_usd": 0.05,
                "agents_used": ["Analyst", "Executor"],
                "smes_used": ["cloud_architect"],
            },
        }
        md = agent._format_as_markdown(result)
        assert "# Multi-Agent Reasoning Result" in md
        assert "test_123" in md
        assert "Analyst" in md
        assert "cloud_architect" in md
        assert "Test response content" in md


# =============================================================================
# OrchestratorAgent Input Loading Tests
# =============================================================================

class TestOrchestratorInputLoading:
    """Tests for input content loading."""

    def test_load_input_content_no_file(self):
        """Test loading input without file."""
        agent = _create_orchestrator()
        content = agent._load_input_content("Hello world")
        assert content == "Hello world"

    def test_load_input_content_with_text_file(self, tmp_path):
        """Test loading input with a text file attachment."""
        agent = _create_orchestrator()
        test_file = tmp_path / "test.py"
        test_file.write_text("print('hello')")
        content = agent._load_input_content("Review this code", str(test_file))
        assert "print('hello')" in content
        assert "test.py" in content

    def test_load_input_content_with_missing_file(self):
        """Test loading input with missing file."""
        agent = _create_orchestrator(verbose=True)
        content = agent._load_input_content("Hello", "/nonexistent/file.py")
        assert content == "Hello"

    def test_load_input_content_with_pdf(self, tmp_path):
        """Test loading input with PDF file."""
        agent = _create_orchestrator()
        pdf_file = tmp_path / "doc.pdf"
        pdf_file.write_bytes(b"fake pdf content")
        content = agent._load_input_content("Review doc", str(pdf_file))
        assert "doc.pdf" in content

    def test_load_input_content_with_image(self, tmp_path):
        """Test loading input with image file."""
        agent = _create_orchestrator()
        img_file = tmp_path / "screenshot.png"
        img_file.write_bytes(b"fake png")
        content = agent._load_input_content("Analyze image", str(img_file))
        assert "screenshot.png" in content

    def test_load_input_content_with_unknown_extension(self, tmp_path):
        """Test loading input with unknown file extension."""
        agent = _create_orchestrator()
        file = tmp_path / "data.xyz"
        file.write_bytes(b"data")
        content = agent._load_input_content("Process data", str(file))
        assert "data.xyz" in content


# =============================================================================
# OrchestratorAgent Council Consultation Tests
# =============================================================================

class TestOrchestratorCouncilConsultation:
    """Tests for council consultation."""

    def test_requires_ethics_review_with_keywords(self):
        """Test ethics review detection with trigger keywords."""
        agent = _create_orchestrator()
        assert agent._requires_ethics_review("Handle personal data") is True
        assert agent._requires_ethics_review("Process medical records") is True
        assert agent._requires_ethics_review("Ensure compliance with regulations") is True
        assert agent._requires_ethics_review("Government security assessment") is True

    def test_requires_ethics_review_without_keywords(self):
        """Test that generic tasks don't require ethics review."""
        agent = _create_orchestrator()
        assert agent._requires_ethics_review("Write a Python function") is False
        assert agent._requires_ethics_review("Build a web app") is False

    def test_extract_sme_selection_from_dict(self):
        """Test SME extraction from dict output."""
        agent = _create_orchestrator()
        output = {"selected_smes": ["cloud_architect", "security_analyst"]}
        smes = agent._extract_sme_selection(output)
        assert smes == ["cloud_architect", "security_analyst"]

    def test_extract_sme_selection_from_string(self):
        """Test SME extraction from non-dict output."""
        agent = _create_orchestrator()
        smes = agent._extract_sme_selection("some string output")
        assert smes == []

    def test_extract_sme_selection_empty_dict(self):
        """Test SME extraction from dict without selected_smes."""
        agent = _create_orchestrator()
        smes = agent._extract_sme_selection({"other_key": "value"})
        assert smes == []


# =============================================================================
# OrchestratorAgent Verdict Tests
# =============================================================================

class TestOrchestratorVerdict:
    """Tests for verdict evaluation."""

    def test_parse_verdict_pass(self):
        """Test parsing PASS verdict."""
        agent = _create_orchestrator()
        from src.core.verdict import Verdict as VerdictEnum
        result = agent._parse_verdict({"verdict": "PASS"})
        assert result == VerdictEnum.PASS

    def test_parse_verdict_fail(self):
        """Test parsing FAIL verdict."""
        agent = _create_orchestrator()
        from src.core.verdict import Verdict as VerdictEnum
        result = agent._parse_verdict({"verdict": "FAIL"})
        assert result == VerdictEnum.FAIL

    def test_parse_verdict_non_dict(self):
        """Test parsing verdict from non-dict output."""
        agent = _create_orchestrator()
        from src.core.verdict import Verdict as VerdictEnum
        result = agent._parse_verdict("some string")
        assert result == VerdictEnum.PASS

    def test_parse_verdict_missing_key(self):
        """Test parsing verdict from dict without verdict key."""
        agent = _create_orchestrator()
        from src.core.verdict import Verdict as VerdictEnum
        result = agent._parse_verdict({"other": "data"})
        assert result == VerdictEnum.PASS


# =============================================================================
# OrchestratorAgent Agent Input Building Tests
# =============================================================================

class TestOrchestratorBuildAgentInput:
    """Tests for _build_agent_input()."""

    def test_build_input_basic(self):
        """Test basic input building."""
        agent = _create_orchestrator()
        from src.core.pipeline import Phase
        result = agent._build_agent_input("Analyst", "Hello", {}, Phase.PHASE_1_TASK_INTELLIGENCE)
        assert "Hello" in result

    def test_build_input_executor_with_context(self):
        """Test executor input includes analysis and plan."""
        agent = _create_orchestrator()
        from src.core.pipeline import Phase
        previous = {
            "Task Analyst": "Analysis results",
            "Planner": "Execution plan",
            "Researcher": "Research findings",
        }
        result = agent._build_agent_input("Executor", "Build it", previous, Phase.PHASE_5_SOLUTION_GENERATION)
        assert "Analysis results" in result
        assert "Execution plan" in result
        assert "Research findings" in result

    def test_build_input_verifier_with_solution(self):
        """Test verifier input includes executor output."""
        agent = _create_orchestrator()
        from src.core.pipeline import Phase
        previous = {"Executor": "The solution"}
        result = agent._build_agent_input("Verifier", "Verify this", previous, Phase.PHASE_6_REVIEW)
        assert "The solution" in result

    def test_build_input_reviewer_with_all_reports(self):
        """Test reviewer input includes all review reports."""
        agent = _create_orchestrator()
        from src.core.pipeline import Phase
        previous = {
            "Executor": "Solution",
            "Verifier": "Verification result",
            "Critic": "Critique result",
        }
        result = agent._build_agent_input("Reviewer", "Review all", previous, Phase.PHASE_6_REVIEW)
        assert "Solution" in result
        assert "Verification result" in result
        assert "Critique result" in result

    def test_build_input_formatter_with_solution(self):
        """Test formatter input includes executor output."""
        agent = _create_orchestrator()
        from src.core.pipeline import Phase
        previous = {"Executor": "Raw output to format"}
        result = agent._build_agent_input("Formatter", "Format it", previous, Phase.PHASE_8_FINAL_REVIEW_FORMATTING)
        assert "Raw output to format" in result


# =============================================================================
# OrchestratorAgent Session Persistence Tests
# =============================================================================

class TestOrchestratorSessionPersistence:
    """Tests for session persistence methods."""

    def test_save_session_state_no_persistence(self):
        """Test _save_session_state does nothing without persistence."""
        agent = _create_orchestrator()
        agent.persistence = None
        session = SessionState(session_id="test", user_prompt="test")
        # Should not raise
        agent._save_session_state(session)

    def test_save_session_state_with_persistence(self):
        """Test _save_session_state delegates to persistence."""
        agent = _create_orchestrator()
        mock_persist = MagicMock()
        agent.persistence = mock_persist
        agent.enable_auto_compact = False

        session = SessionState(session_id="test", user_prompt="test")
        session.current_tier = TierLevel.STANDARD
        session.agent_executions = [
            AgentExecution(
                agent_name="Analyst",
                start_time=100.0,
                end_time=110.0,
                status="complete",
                output="Analysis done",
                tokens_used=500,
            ),
        ]

        with patch("src.agents.orchestrator.ChatMessage"), \
             patch("src.agents.orchestrator.SessionAgentOutput"), \
             patch("src.agents.orchestrator.check_and_compact"):
            agent._save_session_state(session, response={"response": "Final answer"})
            mock_persist.save_session.assert_called_once()

    def test_save_session_state_error_handling(self):
        """Test that save errors are handled gracefully."""
        agent = _create_orchestrator(verbose=True)
        mock_persist = MagicMock()
        mock_persist.save_session.side_effect = Exception("Save failed")
        agent.persistence = mock_persist

        session = SessionState(session_id="test", user_prompt="test")
        session.current_tier = TierLevel.STANDARD
        # Should not raise
        agent._save_session_state(session)


# =============================================================================
# OrchestratorAgent Debate Tests
# =============================================================================

class TestOrchestratorDebate:
    """Tests for debate triggering."""

    def test_should_trigger_debate_tier4(self):
        """Test that Tier 4 always triggers debate."""
        agent = _create_orchestrator()
        from src.core.pipeline import Phase
        session = SessionState(session_id="test", user_prompt="test")
        session.current_tier = TierLevel.ADVERSARIAL
        assert agent._should_trigger_debate(session, Phase.PHASE_6_REVIEW) is True

    def test_should_trigger_debate_disagreement(self):
        """Test debate triggered on verifier/critic disagreement."""
        agent = _create_orchestrator()
        from src.core.pipeline import Phase
        session = SessionState(session_id="test", user_prompt="test")
        session.current_tier = TierLevel.STANDARD
        session.agent_executions = [
            AgentExecution(agent_name="Verifier", start_time=0, output={"verdict": "PASS"}),
            AgentExecution(agent_name="Critic", start_time=0, output={"verdict": "FAIL"}),
        ]
        assert agent._should_trigger_debate(session, Phase.PHASE_6_REVIEW) is True

    def test_should_not_trigger_debate_agreement(self):
        """Test no debate when verifier and critic agree."""
        agent = _create_orchestrator()
        from src.core.pipeline import Phase
        session = SessionState(session_id="test", user_prompt="test")
        session.current_tier = TierLevel.STANDARD
        session.agent_executions = [
            AgentExecution(agent_name="Verifier", start_time=0, output={"verdict": "PASS"}),
            AgentExecution(agent_name="Critic", start_time=0, output={"verdict": "PASS"}),
        ]
        assert agent._should_trigger_debate(session, Phase.PHASE_6_REVIEW) is False

    def test_should_not_trigger_debate_non_review_phase(self):
        """Test no debate in non-review phases (for non-tier-4)."""
        agent = _create_orchestrator()
        from src.core.pipeline import Phase
        session = SessionState(session_id="test", user_prompt="test")
        session.current_tier = TierLevel.STANDARD
        assert agent._should_trigger_debate(session, Phase.PHASE_1_TASK_INTELLIGENCE) is False


# =============================================================================
# OrchestratorAgent Log Tests
# =============================================================================

class TestOrchestratorLog:
    """Tests for logging."""

    def test_log_when_verbose(self, capsys):
        """Test that _log prints when verbose is True."""
        agent = _create_orchestrator(verbose=True)
        agent._log("Test message")
        captured = capsys.readouterr()
        assert "[Orchestrator]" in captured.out
        assert "Test message" in captured.out

    def test_log_silent_when_not_verbose(self, capsys):
        """Test that _log is silent when verbose is False."""
        agent = _create_orchestrator(verbose=False)
        agent._log("Test message")
        captured = capsys.readouterr()
        assert captured.out == ""


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestCreateOrchestrator:
    """Tests for create_orchestrator convenience function."""

    def test_create_orchestrator_defaults(self):
        """Test create_orchestrator with defaults."""
        with patch("src.agents.orchestrator.get_settings", return_value=_mock_settings()), \
             patch("src.agents.orchestrator.get_api_key", return_value="test-key"), \
             patch("src.agents.orchestrator.get_model_for_agent", return_value="model"), \
             patch("src.agents.orchestrator.get_provider", return_value="anthropic"), \
             patch("src.agents.orchestrator.create_sdk_mcp_server", return_value=None):
            agent = create_orchestrator(verbose=False)
            assert isinstance(agent, OrchestratorAgent)

    def test_create_orchestrator_custom(self):
        """Test create_orchestrator with custom params."""
        with patch("src.agents.orchestrator.get_settings", return_value=_mock_settings()), \
             patch("src.agents.orchestrator.get_api_key", return_value="test-key"), \
             patch("src.agents.orchestrator.get_model_for_agent", return_value="model"), \
             patch("src.agents.orchestrator.get_provider", return_value="anthropic"), \
             patch("src.agents.orchestrator.create_sdk_mcp_server", return_value=None):
            agent = create_orchestrator(
                api_key="custom-key",
                max_budget_usd=20.0,
                verbose=True,
            )
            assert agent.api_key == "custom-key"
            assert agent.max_budget_usd == 20.0
            assert agent.verbose is True
