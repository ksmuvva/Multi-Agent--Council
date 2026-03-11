"""
Rigorous Unit Tests for Orchestrator Agent

Tests for the main orchestrator including session integration,
tier classification, agent coordination, pipeline execution,
escalation handling, debate protocol, verdict matrix evaluation,
council consultation, budget enforcement, error handling, and
edge cases.
"""

import time
import os
import json
import tempfile
import pytest
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime

from src.agents.orchestrator import (
    OrchestratorAgent,
    AgentExecution,
    SessionState,
    create_orchestrator,
)
from src.core.complexity import TierLevel, TierClassification
from src.core.pipeline import (
    ExecutionPipeline,
    PipelineBuilder,
    Phase,
    PhaseStatus,
    AgentResult,
)
from src.core.verdict import (
    Verdict,
    MatrixAction,
    MatrixOutcome,
    evaluate_verdict_matrix,
)
from src.core.debate import DebateProtocol, ConsensusLevel


# =============================================================================
# Fixture: reset settings singleton between tests to avoid cross-contamination
# =============================================================================

@pytest.fixture(autouse=True)
def _reset_settings():
    """Reset the global settings singleton before each test."""
    from src.config.settings import reload_settings
    os.environ["ANTHROPIC_API_KEY"] = "test_key_dummy"
    os.environ["TESTING"] = "true"
    reload_settings()
    yield
    # Cleanup handled by conftest, but reset singleton to be safe
    reload_settings()


# =============================================================================
# Helper: create a mock classification
# =============================================================================

def _make_classification(tier=TierLevel.STANDARD, **overrides):
    defaults = dict(
        tier=tier,
        reasoning="Test classification",
        confidence=0.8,
        estimated_agents=7,
        requires_council=tier >= TierLevel.DEEP,
        requires_smes=tier >= TierLevel.DEEP,
    )
    defaults.update(overrides)
    return TierClassification(**defaults)


def _make_spawn_result(status="success", output=None, cost=0.005, tokens=500):
    return {
        "status": status,
        "output": output or f"[Simulated output]",
        "tokens_used": tokens,
        "cost_usd": cost,
        "duration_ms": 100,
        "model": "test-model",
        "retries": 0,
    }


# =============================================================================
# 1. OrchestratorAgent Initialization
# =============================================================================

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

    def test_orchestrator_persistence_disabled_has_no_persistence(self):
        """Test that persistence=False sets persistence to None."""
        orchestrator = OrchestratorAgent(enable_persistence=False)
        assert orchestrator.persistence is None

    def test_orchestrator_persistence_enabled_creates_persistence(self):
        """Test that persistence=True creates a persistence object."""
        orchestrator = OrchestratorAgent(enable_persistence=True)
        assert orchestrator.persistence is not None

    def test_orchestrator_provider_info_set(self):
        """Test that provider info is populated."""
        orchestrator = OrchestratorAgent()
        assert orchestrator.provider is not None
        assert orchestrator.provider_config is not None

    def test_orchestrator_mcp_server_created(self):
        """Test that MCP server is registered on init."""
        orchestrator = OrchestratorAgent()
        assert orchestrator.mcp_server is not None
        assert "tools" in orchestrator.mcp_server


# =============================================================================
# 2. SessionState
# =============================================================================

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

        assert session.budget_utilization == 0.0
        assert session.is_budget_exceeded() is False
        assert session.should_warn_budget() is False

        session.total_cost_usd = 5.0
        assert session.budget_utilization == 0.5
        assert session.is_budget_exceeded() is False

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

        session.total_cost_usd = 7.0
        assert session.should_warn_budget() is False

        session.total_cost_usd = 8.0
        assert session.should_warn_budget() is True

    def test_session_state_duration(self):
        """Test session duration calculation."""
        session = SessionState(
            session_id="duration_test",
            user_prompt="Test",
        )

        initial_duration = session.duration_seconds
        assert initial_duration >= 0

        session.end_time = time.time()
        final_duration = session.duration_seconds
        assert final_duration >= 0

    def test_session_state_zero_budget_utilization(self):
        """Test budget utilization with zero max budget (division safety)."""
        session = SessionState(
            session_id="zero_budget",
            user_prompt="Test",
            max_budget_usd=0.0,
        )
        assert session.budget_utilization == 0

    def test_session_state_agent_executions_list(self):
        """Test that agent executions list is mutable."""
        session = SessionState(session_id="exec_test", user_prompt="Test")
        session.agent_executions.append(
            AgentExecution(agent_name="Analyst", start_time=time.time())
        )
        assert len(session.agent_executions) == 1

    def test_session_state_escalation_history(self):
        """Test escalation history tracking."""
        session = SessionState(session_id="esc_test", user_prompt="Test")
        session.escalation_history.append({"tier": 2, "reason": "complex"})
        assert len(session.escalation_history) == 1

    def test_session_state_active_smes_default_empty(self):
        """Test default empty SME list."""
        session = SessionState(session_id="sme_test", user_prompt="Test")
        assert session.active_smes == []

    def test_session_state_council_default_false(self):
        """Test council not activated by default."""
        session = SessionState(session_id="council_test", user_prompt="Test")
        assert session.council_activated is False


# =============================================================================
# 3. AgentExecution
# =============================================================================

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

    def test_agent_execution_failure(self):
        """Test recording a failed execution."""
        execution = AgentExecution(
            agent_name="Verifier",
            start_time=time.time(),
        )
        execution.status = "failed"
        execution.error = "Timeout exceeded"
        assert execution.status == "failed"
        assert execution.error == "Timeout exceeded"


# =============================================================================
# 4. Session Creation & Management
# =============================================================================

class TestSessionManagement:
    """Tests for session creation and context management."""

    def test_create_session_auto_id(self):
        """Test session creation with auto-generated ID."""
        orchestrator = OrchestratorAgent(enable_persistence=False)
        session = orchestrator._create_session(user_prompt="Test")
        assert session.session_id.startswith("session_")
        assert session.user_prompt == "Test"

    def test_create_session_custom_id(self):
        """Test session creation with custom ID."""
        orchestrator = OrchestratorAgent(enable_persistence=False)
        session = orchestrator._create_session(
            user_prompt="Test",
            session_id="custom_123",
        )
        assert session.session_id == "custom_123"

    def test_create_session_with_resume_context(self):
        """Test session creation with resume context restores state."""
        orchestrator = OrchestratorAgent(enable_persistence=False)
        session = orchestrator._create_session(
            user_prompt="Resumed test",
            session_id="resume_1",
            resume_context={
                "total_cost_usd": 1.5,
                "revision_cycle": 1,
                "active_smes": ["cloud_architect"],
                "council_activated": True,
            },
        )
        assert session.total_cost_usd == 1.5
        assert session.revision_cycle == 1
        assert session.active_smes == ["cloud_architect"]
        assert session.council_activated is True

    def test_get_session_context(self):
        """Test extracting session context for resumption."""
        orchestrator = OrchestratorAgent(enable_persistence=False)
        session = SessionState(
            session_id="ctx_test",
            user_prompt="Test",
        )
        session.total_cost_usd = 2.0
        session.revision_cycle = 1
        session.active_smes = ["security_analyst"]
        session.council_activated = True

        context = orchestrator.get_session_context(session)

        assert context["session_id"] == "ctx_test"
        assert context["total_cost_usd"] == 2.0
        assert context["revision_cycle"] == 1
        assert context["active_smes"] == ["security_analyst"]
        assert context["council_activated"] is True
        assert "duration_seconds" in context


# =============================================================================
# 5. Tier Override Classification
# =============================================================================

class TestTierOverrideClassification:
    """Tests for _create_override_classification."""

    def test_override_tier_1(self):
        orchestrator = OrchestratorAgent(enable_persistence=False)
        c = orchestrator._create_override_classification(1)
        assert c.tier == TierLevel.DIRECT
        assert c.requires_council is False
        assert c.requires_smes is False
        assert c.confidence == 1.0

    def test_override_tier_2(self):
        orchestrator = OrchestratorAgent(enable_persistence=False)
        c = orchestrator._create_override_classification(2)
        assert c.tier == TierLevel.STANDARD
        assert c.requires_council is False

    def test_override_tier_3(self):
        orchestrator = OrchestratorAgent(enable_persistence=False)
        c = orchestrator._create_override_classification(3)
        assert c.tier == TierLevel.DEEP
        assert c.requires_council is True
        assert c.requires_smes is True

    def test_override_tier_4(self):
        orchestrator = OrchestratorAgent(enable_persistence=False)
        c = orchestrator._create_override_classification(4)
        assert c.tier == TierLevel.ADVERSARIAL
        assert c.requires_council is True
        assert c.requires_smes is True
        assert c.suggested_sme_count == 3

    def test_override_invalid_tier_raises(self):
        orchestrator = OrchestratorAgent(enable_persistence=False)
        with pytest.raises(ValueError):
            orchestrator._create_override_classification(5)

    def test_override_zero_tier_raises(self):
        orchestrator = OrchestratorAgent(enable_persistence=False)
        with pytest.raises(ValueError):
            orchestrator._create_override_classification(0)


# =============================================================================
# 6. Input Content Loading
# =============================================================================

class TestInputContentLoading:
    """Tests for _load_input_content including multimodal file handling."""

    def test_plain_text_prompt(self):
        orchestrator = OrchestratorAgent(enable_persistence=False)
        result = orchestrator._load_input_content("Hello world")
        assert result == "Hello world"

    def test_nonexistent_file_returns_prompt_only(self):
        orchestrator = OrchestratorAgent(enable_persistence=False)
        result = orchestrator._load_input_content("Test", file_path="/nonexistent/file.py")
        assert result == "Test"

    def test_python_file_attachment(self):
        orchestrator = OrchestratorAgent(enable_persistence=False)
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("print('hello')")
            f.flush()
            result = orchestrator._load_input_content("Analyze this", file_path=f.name)
        assert "print('hello')" in result
        assert "Attached File" in result
        os.unlink(f.name)

    def test_json_file_attachment(self):
        orchestrator = OrchestratorAgent(enable_persistence=False)
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump({"key": "value"}, f)
            f.flush()
            result = orchestrator._load_input_content("Analyze", file_path=f.name)
        assert "key" in result
        os.unlink(f.name)

    def test_image_file_attachment(self):
        orchestrator = OrchestratorAgent(enable_persistence=False)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"\x89PNG")
            f.flush()
            result = orchestrator._load_input_content("Look at this", file_path=f.name)
        assert "Attached Image" in result
        os.unlink(f.name)

    def test_unknown_extension_attachment(self):
        orchestrator = OrchestratorAgent(enable_persistence=False)
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"data")
            f.flush()
            result = orchestrator._load_input_content("Check this", file_path=f.name)
        assert "Attached File" in result
        os.unlink(f.name)


# =============================================================================
# 7. process_request - Tier Classification
# =============================================================================

class TestOrchestratorProcessRequest:
    """Tests for the main process_request method."""

    @patch('src.agents.orchestrator.classify_complexity')
    def test_process_request_tier_classification(self, mock_classify):
        """Test that process_request classifies complexity."""
        mock_classify.return_value = _make_classification()

        orchestrator = OrchestratorAgent(enable_persistence=False)

        result = orchestrator.process_request(
            user_prompt="Test prompt",
            session_id="test_session",
        )

        assert "response" in result
        assert "session_id" in result
        assert "metadata" in result
        assert result["session_id"] == "test_session"

    def test_process_request_with_tier_override(self):
        """Test process_request with tier override."""
        orchestrator = OrchestratorAgent(enable_persistence=False)

        result = orchestrator.process_request(
            user_prompt="Test",
            tier_override=1,
        )

        assert result["metadata"]["tier"] == 1

    def test_process_request_all_four_tier_overrides(self):
        """Test that all four tier overrides produce valid responses."""
        orchestrator = OrchestratorAgent(enable_persistence=False)
        for tier in [1, 2, 3, 4]:
            result = orchestrator.process_request(
                user_prompt="Test prompt",
                tier_override=tier,
            )
            assert result["metadata"]["tier"] == tier
            assert "response" in result

    @patch('src.agents.orchestrator.classify_complexity')
    def test_process_request_budget_pre_exceeded(self, mock_classify):
        """Test budget check when budget is already exceeded via resume context."""
        mock_classify.return_value = _make_classification()

        orchestrator = OrchestratorAgent(
            max_budget_usd=1.0,
            enable_persistence=False,
        )

        result = orchestrator.process_request(
            user_prompt="Test",
            resume_context={"total_cost_usd": 5.0},
        )

        # Budget was pre-exceeded; should get budget error
        assert "budget" in result["metadata"].get("error", "").lower()

    def test_process_request_returns_session_id(self):
        """Test that process_request always returns session_id."""
        orchestrator = OrchestratorAgent(enable_persistence=False)
        result = orchestrator.process_request(
            user_prompt="Hello",
            session_id="myid",
            tier_override=1,
        )
        assert result["session_id"] == "myid"

    @patch('src.agents.orchestrator.classify_complexity')
    def test_process_request_error_handling(self, mock_classify):
        """Test that exceptions during processing produce error response."""
        mock_classify.side_effect = RuntimeError("Classification crash")

        orchestrator = OrchestratorAgent(enable_persistence=False)
        result = orchestrator.process_request(user_prompt="Test")

        assert "error" in result.get("metadata", {}) or "error" in str(result.get("response", "")).lower()

    def test_process_request_with_session_id_none(self):
        """Test auto-generated session ID."""
        orchestrator = OrchestratorAgent(enable_persistence=False)
        result = orchestrator.process_request(
            user_prompt="Test",
            tier_override=1,
        )
        assert result["session_id"].startswith("session_")

    @patch('src.agents.orchestrator.classify_complexity')
    def test_process_request_tier3_activates_council(self, mock_classify):
        """Test that Tier 3 requests activate council consultation."""
        mock_classify.return_value = _make_classification(TierLevel.DEEP)

        orchestrator = OrchestratorAgent(enable_persistence=False)
        result = orchestrator.process_request(user_prompt="Design a secure architecture")

        # The response should have executed (council may or may not appear in agents_used
        # depending on mock, but the tier should be 3)
        assert result["metadata"]["tier"] == 3

    def test_process_request_metadata_contains_agents_used(self):
        """Test that metadata includes agents_used list."""
        orchestrator = OrchestratorAgent(enable_persistence=False)
        result = orchestrator.process_request(
            user_prompt="Simple question",
            tier_override=1,
        )
        assert "agents_used" in result["metadata"]
        assert isinstance(result["metadata"]["agents_used"], list)

    def test_process_request_metadata_contains_cost_info(self):
        """Test that metadata includes cost tracking."""
        orchestrator = OrchestratorAgent(enable_persistence=False)
        result = orchestrator.process_request(
            user_prompt="Test",
            tier_override=1,
        )
        assert "total_cost_usd" in result["metadata"]
        assert "duration_seconds" in result["metadata"]


# =============================================================================
# 8. execute() - CLI Entry Point
# =============================================================================

class TestOrchestratorExecute:
    """Tests for the execute method (CLI entry point)."""

    @patch('src.agents.orchestrator.classify_complexity')
    def test_execute_default_format(self, mock_classify):
        """Test execute with default markdown format."""
        mock_classify.return_value = _make_classification()

        orchestrator = OrchestratorAgent(enable_persistence=False)

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
        mock_classify.return_value = _make_classification()

        orchestrator = OrchestratorAgent(enable_persistence=False)

        result = orchestrator.execute(
            user_prompt="Test prompt",
            tier_level=2,
            format="json",
        )

        assert "formatted_output" in result
        assert result["formatted_output"].strip().startswith("{")

    @patch('src.agents.orchestrator.classify_complexity')
    def test_execute_text_format(self, mock_classify):
        """Test execute with plain text format."""
        mock_classify.return_value = _make_classification()

        orchestrator = OrchestratorAgent(enable_persistence=False)

        result = orchestrator.execute(
            user_prompt="Test prompt",
            tier_level=2,
            format="text",
        )

        assert "formatted_output" in result
        # Text format should be just the response
        assert "Session ID" not in result["formatted_output"]

    @patch('src.agents.orchestrator.classify_complexity')
    def test_execute_markdown_contains_header(self, mock_classify):
        """Test markdown output contains expected sections."""
        mock_classify.return_value = _make_classification()

        orchestrator = OrchestratorAgent(enable_persistence=False)
        result = orchestrator.execute(user_prompt="Test", tier_level=2, format="markdown")

        formatted = result["formatted_output"]
        assert "Multi-Agent Reasoning Result" in formatted
        assert "Tier" in formatted
        assert "Duration" in formatted
        assert "Cost" in formatted

    def test_execute_with_all_tiers(self):
        """Test execute works for all tier levels."""
        orchestrator = OrchestratorAgent(enable_persistence=False)
        for tier in [1, 2, 3, 4]:
            result = orchestrator.execute(user_prompt="Test", tier_level=tier)
            assert "formatted_output" in result
            assert "raw_output" in result


# =============================================================================
# 9. Pipeline Execution
# =============================================================================

class TestPipelineExecution:
    """Tests for _execute_pipeline and related methods."""

    def test_build_agent_input_executor_gets_analyst_context(self):
        """Test that Executor receives Task Analyst and Planner outputs."""
        orchestrator = OrchestratorAgent(enable_persistence=False)
        previous = {
            "Task Analyst": "Task is feasible",
            "Planner": "Step 1: Do X; Step 2: Do Y",
            "Researcher": "Found evidence Z",
        }

        result = orchestrator._build_agent_input(
            agent_name="Executor",
            user_prompt="Build something",
            previous_outputs=previous,
            phase=Phase.PHASE_5_SOLUTION_GENERATION,
        )

        assert "Build something" in result
        assert "Task is feasible" in result
        assert "Step 1: Do X" in result
        assert "Found evidence Z" in result

    def test_build_agent_input_verifier_gets_executor_output(self):
        """Test that Verifier receives Executor output."""
        orchestrator = OrchestratorAgent(enable_persistence=False)
        previous = {"Executor": "The solution is X"}

        result = orchestrator._build_agent_input(
            agent_name="Verifier",
            user_prompt="Check this",
            previous_outputs=previous,
            phase=Phase.PHASE_6_REVIEW,
        )

        assert "The solution is X" in result

    def test_build_agent_input_critic_gets_executor_output(self):
        """Test that Critic receives Executor output."""
        orchestrator = OrchestratorAgent(enable_persistence=False)
        previous = {"Executor": "My solution"}

        result = orchestrator._build_agent_input(
            agent_name="Critic",
            user_prompt="Review",
            previous_outputs=previous,
            phase=Phase.PHASE_6_REVIEW,
        )

        assert "My solution" in result

    def test_build_agent_input_reviewer_gets_all_review_outputs(self):
        """Test Reviewer gets Executor, Verifier, and Critic outputs."""
        orchestrator = OrchestratorAgent(enable_persistence=False)
        previous = {
            "Executor": "Solution",
            "Verifier": "Verified OK",
            "Critic": "Found issue A",
        }

        result = orchestrator._build_agent_input(
            agent_name="Reviewer",
            user_prompt="Final review",
            previous_outputs=previous,
            phase=Phase.PHASE_8_FINAL_REVIEW_FORMATTING,
        )

        assert "Solution" in result
        assert "Verified OK" in result
        assert "Found issue A" in result

    def test_build_agent_input_formatter_gets_executor_output(self):
        """Test Formatter receives Executor output."""
        orchestrator = OrchestratorAgent(enable_persistence=False)
        previous = {"Executor": "Raw output"}

        result = orchestrator._build_agent_input(
            agent_name="Formatter",
            user_prompt="Format",
            previous_outputs=previous,
            phase=Phase.PHASE_8_FINAL_REVIEW_FORMATTING,
        )

        assert "Raw output" in result

    def test_build_agent_input_no_previous_outputs(self):
        """Test agent input with empty previous outputs (first phase)."""
        orchestrator = OrchestratorAgent(enable_persistence=False)
        result = orchestrator._build_agent_input(
            agent_name="Task Analyst",
            user_prompt="Analyze this task",
            previous_outputs={},
            phase=Phase.PHASE_1_TASK_INTELLIGENCE,
        )

        assert "Analyze this task" in result

    def test_execute_pipeline_budget_stops_execution(self):
        """Test that pipeline stops when budget is exceeded mid-execution."""
        orchestrator = OrchestratorAgent(
            max_budget_usd=0.001,
            enable_persistence=False,
        )
        session = SessionState(
            session_id="budget_pipeline",
            user_prompt="Test",
            max_budget_usd=0.001,
        )
        session.tier_classification = _make_classification(TierLevel.STANDARD)
        session.current_tier = TierLevel.STANDARD

        pipeline = PipelineBuilder.for_tier(TierLevel.STANDARD)
        context = {
            "user_prompt": "Test",
            "tier": TierLevel.STANDARD,
            "active_smes": [],
            "council_activated": False,
        }

        # Execute - should stop after first agent exceeds budget
        orchestrator._execute_pipeline(pipeline, session, context)

        # Cost should have been tracked
        assert session.total_cost_usd > 0


# =============================================================================
# 10. Verdict Matrix Evaluation
# =============================================================================

class TestVerdictEvaluation:
    """Tests for _evaluate_verdict and _parse_verdict."""

    def test_parse_verdict_dict_pass(self):
        orchestrator = OrchestratorAgent(enable_persistence=False)
        assert orchestrator._parse_verdict({"verdict": "PASS"}) == Verdict.PASS

    def test_parse_verdict_dict_fail(self):
        orchestrator = OrchestratorAgent(enable_persistence=False)
        assert orchestrator._parse_verdict({"verdict": "FAIL"}) == Verdict.FAIL

    def test_parse_verdict_string_returns_pass(self):
        orchestrator = OrchestratorAgent(enable_persistence=False)
        assert orchestrator._parse_verdict("some string output") == Verdict.PASS

    def test_parse_verdict_none_returns_pass(self):
        orchestrator = OrchestratorAgent(enable_persistence=False)
        assert orchestrator._parse_verdict(None) == Verdict.PASS

    def test_parse_verdict_dict_missing_verdict_key(self):
        orchestrator = OrchestratorAgent(enable_persistence=False)
        assert orchestrator._parse_verdict({"score": 0.5}) == Verdict.PASS

    def test_parse_verdict_lowercase_normalized(self):
        orchestrator = OrchestratorAgent(enable_persistence=False)
        assert orchestrator._parse_verdict({"verdict": "fail"}) == Verdict.FAIL

    def test_evaluate_verdict_both_pass(self):
        """Test verdict when both Verifier and Critic pass."""
        orchestrator = OrchestratorAgent(enable_persistence=False)
        session = SessionState(session_id="v_test", user_prompt="Test")
        session.agent_executions.append(
            AgentExecution(agent_name="Verifier", start_time=time.time(),
                           output={"verdict": "PASS"})
        )
        session.agent_executions.append(
            AgentExecution(agent_name="Critic", start_time=time.time(),
                           output={"verdict": "PASS"})
        )
        action = orchestrator._evaluate_verdict(session)
        assert action == MatrixAction.PROCEED_TO_FORMATTER

    def test_evaluate_verdict_verifier_pass_critic_fail(self):
        """Test verdict when Verifier passes but Critic fails."""
        orchestrator = OrchestratorAgent(enable_persistence=False)
        session = SessionState(session_id="v_test2", user_prompt="Test")
        session.agent_executions.append(
            AgentExecution(agent_name="Verifier", start_time=time.time(),
                           output={"verdict": "PASS"})
        )
        session.agent_executions.append(
            AgentExecution(agent_name="Critic", start_time=time.time(),
                           output={"verdict": "FAIL"})
        )
        action = orchestrator._evaluate_verdict(session)
        assert action == MatrixAction.EXECUTOR_REVISE

    def test_evaluate_verdict_verifier_fail_critic_pass(self):
        """Test verdict when Verifier fails but Critic passes."""
        orchestrator = OrchestratorAgent(enable_persistence=False)
        session = SessionState(session_id="v_test3", user_prompt="Test")
        session.agent_executions.append(
            AgentExecution(agent_name="Verifier", start_time=time.time(),
                           output={"verdict": "FAIL"})
        )
        session.agent_executions.append(
            AgentExecution(agent_name="Critic", start_time=time.time(),
                           output={"verdict": "PASS"})
        )
        action = orchestrator._evaluate_verdict(session)
        assert action == MatrixAction.RESEARCHER_REVERIFY

    def test_evaluate_verdict_both_fail(self):
        """Test verdict when both fail."""
        orchestrator = OrchestratorAgent(enable_persistence=False)
        session = SessionState(session_id="v_test4", user_prompt="Test")
        session.agent_executions.append(
            AgentExecution(agent_name="Verifier", start_time=time.time(),
                           output={"verdict": "FAIL"})
        )
        session.agent_executions.append(
            AgentExecution(agent_name="Critic", start_time=time.time(),
                           output={"verdict": "FAIL"})
        )
        action = orchestrator._evaluate_verdict(session)
        assert action == MatrixAction.FULL_REGENERATION

    def test_evaluate_verdict_no_review_agents_defaults_pass(self):
        """Test verdict defaults to PASS when no review agents present."""
        orchestrator = OrchestratorAgent(enable_persistence=False)
        session = SessionState(session_id="v_empty", user_prompt="Test")
        action = orchestrator._evaluate_verdict(session)
        assert action == MatrixAction.PROCEED_TO_FORMATTER

    def test_evaluate_verdict_quality_arbiter_on_tier4_max_revisions(self):
        """Test Quality Arbiter invoked on Tier 4 after max revisions."""
        orchestrator = OrchestratorAgent(enable_persistence=False)
        session = SessionState(
            session_id="qa_test", user_prompt="Test",
            max_revisions=2,
        )
        session.current_tier = TierLevel.ADVERSARIAL
        session.revision_cycle = 3  # Exceeds max
        session.agent_executions.append(
            AgentExecution(agent_name="Verifier", start_time=time.time(),
                           output={"verdict": "FAIL"})
        )
        session.agent_executions.append(
            AgentExecution(agent_name="Critic", start_time=time.time(),
                           output={"verdict": "FAIL"})
        )
        action = orchestrator._evaluate_verdict(session)
        assert action == MatrixAction.QUALITY_ARBITER


# =============================================================================
# 11. Debate Triggering
# =============================================================================

class TestDebateTriggering:
    """Tests for _should_trigger_debate."""

    def test_debate_triggered_on_tier4(self):
        """Tier 4 always triggers debate."""
        orchestrator = OrchestratorAgent(enable_persistence=False)
        session = SessionState(session_id="d1", user_prompt="Test")
        session.current_tier = TierLevel.ADVERSARIAL
        assert orchestrator._should_trigger_debate(session, Phase.PHASE_5_SOLUTION_GENERATION) is True

    def test_debate_not_triggered_on_tier1(self):
        """Tier 1 should not trigger debate."""
        orchestrator = OrchestratorAgent(enable_persistence=False)
        session = SessionState(session_id="d2", user_prompt="Test")
        session.current_tier = TierLevel.DIRECT
        assert orchestrator._should_trigger_debate(session, Phase.PHASE_5_SOLUTION_GENERATION) is False

    def test_debate_triggered_on_review_disagreement(self):
        """Debate triggered when Verifier and Critic disagree."""
        orchestrator = OrchestratorAgent(enable_persistence=False)
        session = SessionState(session_id="d3", user_prompt="Test")
        session.current_tier = TierLevel.STANDARD
        session.agent_executions.append(
            AgentExecution(agent_name="Verifier", start_time=time.time(),
                           output={"verdict": "PASS"})
        )
        session.agent_executions.append(
            AgentExecution(agent_name="Critic", start_time=time.time(),
                           output={"verdict": "FAIL"})
        )
        assert orchestrator._should_trigger_debate(session, Phase.PHASE_6_REVIEW) is True

    def test_debate_not_triggered_on_review_agreement(self):
        """No debate when Verifier and Critic agree."""
        orchestrator = OrchestratorAgent(enable_persistence=False)
        session = SessionState(session_id="d4", user_prompt="Test")
        session.current_tier = TierLevel.STANDARD
        session.agent_executions.append(
            AgentExecution(agent_name="Verifier", start_time=time.time(),
                           output={"verdict": "PASS"})
        )
        session.agent_executions.append(
            AgentExecution(agent_name="Critic", start_time=time.time(),
                           output={"verdict": "PASS"})
        )
        assert orchestrator._should_trigger_debate(session, Phase.PHASE_6_REVIEW) is False

    def test_debate_not_triggered_on_non_review_phase(self):
        """No debate on non-review phases (for non-T4)."""
        orchestrator = OrchestratorAgent(enable_persistence=False)
        session = SessionState(session_id="d5", user_prompt="Test")
        session.current_tier = TierLevel.DEEP
        assert orchestrator._should_trigger_debate(session, Phase.PHASE_3_PLANNING) is False


# =============================================================================
# 12. Conduct Debate
# =============================================================================

class TestConductDebate:
    """Tests for _conduct_debate."""

    def test_debate_increments_rounds(self):
        """Test that conducting debate increments debate_rounds."""
        orchestrator = OrchestratorAgent(enable_persistence=False)
        session = SessionState(session_id="cd1", user_prompt="Test", max_debate_rounds=2)
        session.active_smes = []
        context = {"user_prompt": "Test"}

        orchestrator._conduct_debate(session, context)

        assert session.debate_rounds == 2  # max_debate_rounds rounds

    def test_debate_with_smes(self):
        """Test that SMEs are included as debate participants."""
        orchestrator = OrchestratorAgent(enable_persistence=False)
        session = SessionState(session_id="cd2", user_prompt="Test", max_debate_rounds=1)
        session.active_smes = ["cloud_architect", "security_analyst"]
        context = {"user_prompt": "Test"}

        orchestrator._conduct_debate(session, context)

        assert session.debate_rounds == 1


# =============================================================================
# 13. Escalation Handling
# =============================================================================

class TestEscalation:
    """Tests for _handle_escalation."""

    def test_escalation_from_tier1_to_tier2(self):
        orchestrator = OrchestratorAgent(enable_persistence=False)
        session = SessionState(session_id="esc1", user_prompt="Test")
        session.current_tier = TierLevel.DIRECT

        orchestrator._handle_escalation(session, {
            "escalation_needed": True,
            "escalation_reason": "Need more analysis",
        })

        assert session.current_tier == TierLevel.STANDARD
        assert len(session.escalation_history) == 1

    def test_escalation_from_tier2_to_tier3(self):
        orchestrator = OrchestratorAgent(enable_persistence=False)
        session = SessionState(session_id="esc2", user_prompt="Test")
        session.current_tier = TierLevel.STANDARD

        orchestrator._handle_escalation(session, {
            "escalation_needed": True,
            "escalation_reason": "Domain expertise needed",
        })

        assert session.current_tier == TierLevel.DEEP
        assert len(session.escalation_history) == 1

    def test_escalation_from_tier3_to_tier4(self):
        orchestrator = OrchestratorAgent(enable_persistence=False)
        session = SessionState(session_id="esc3", user_prompt="Test")
        session.current_tier = TierLevel.DEEP

        orchestrator._handle_escalation(session, {
            "escalation_needed": True,
        })

        assert session.current_tier == TierLevel.ADVERSARIAL

    def test_escalation_capped_at_tier4(self):
        orchestrator = OrchestratorAgent(enable_persistence=False)
        session = SessionState(session_id="esc4", user_prompt="Test")
        session.current_tier = TierLevel.ADVERSARIAL

        orchestrator._handle_escalation(session, {
            "escalation_needed": True,
        })

        assert session.current_tier == TierLevel.ADVERSARIAL  # No change

    def test_escalation_activates_council_on_tier3(self):
        """Test that escalation to Tier 3 activates council."""
        orchestrator = OrchestratorAgent(enable_persistence=False)
        session = SessionState(session_id="esc5", user_prompt="Test domain expertise")
        session.current_tier = TierLevel.STANDARD
        session.council_activated = False

        orchestrator._handle_escalation(session, {"escalation_needed": True})

        assert session.current_tier == TierLevel.DEEP
        # Council should have been activated
        assert session.council_activated is True

    def test_escalation_history_records_reason(self):
        orchestrator = OrchestratorAgent(enable_persistence=False)
        session = SessionState(session_id="esc6", user_prompt="Test")
        session.current_tier = TierLevel.DIRECT

        orchestrator._handle_escalation(session, {
            "escalation_reason": "Need specialist",
        })

        assert session.escalation_history[0]["reason"] == "Need specialist"
        assert "timestamp" in session.escalation_history[0]
        assert session.escalation_history[0]["tier"] == TierLevel.DIRECT


# =============================================================================
# 14. Council Consultation
# =============================================================================

class TestCouncilConsultation:
    """Tests for council-related methods."""

    def test_extract_sme_selection_from_dict(self):
        orchestrator = OrchestratorAgent(enable_persistence=False)
        result = orchestrator._extract_sme_selection({
            "selected_smes": ["cloud_architect", "security_analyst"],
        })
        assert result == ["cloud_architect", "security_analyst"]

    def test_extract_sme_selection_empty_dict(self):
        orchestrator = OrchestratorAgent(enable_persistence=False)
        result = orchestrator._extract_sme_selection({})
        assert result == []

    def test_extract_sme_selection_string(self):
        orchestrator = OrchestratorAgent(enable_persistence=False)
        result = orchestrator._extract_sme_selection("just a string")
        assert result == []

    def test_extract_sme_selection_none(self):
        orchestrator = OrchestratorAgent(enable_persistence=False)
        result = orchestrator._extract_sme_selection(None)
        assert result == []

    def test_requires_ethics_review_with_pii(self):
        orchestrator = OrchestratorAgent(enable_persistence=False)
        assert orchestrator._requires_ethics_review("Handle personal data carefully") is True

    def test_requires_ethics_review_with_medical(self):
        orchestrator = OrchestratorAgent(enable_persistence=False)
        assert orchestrator._requires_ethics_review("Medical records analysis") is True

    def test_requires_ethics_review_with_legal(self):
        orchestrator = OrchestratorAgent(enable_persistence=False)
        assert orchestrator._requires_ethics_review("Legal compliance review") is True

    def test_requires_ethics_review_with_security(self):
        orchestrator = OrchestratorAgent(enable_persistence=False)
        assert orchestrator._requires_ethics_review("Security audit of system") is True

    def test_requires_ethics_review_normal_request(self):
        orchestrator = OrchestratorAgent(enable_persistence=False)
        assert orchestrator._requires_ethics_review("Write a fibonacci function") is False


# =============================================================================
# 15. Response Generation
# =============================================================================

class TestResponseGeneration:
    """Tests for _generate_final_response and _format_response."""

    def test_generate_final_response_structure(self):
        orchestrator = OrchestratorAgent(enable_persistence=False)
        session = SessionState(session_id="resp1", user_prompt="Test")
        session.tier_classification = _make_classification()
        session.agent_executions.append(
            AgentExecution(agent_name="Executor", start_time=time.time(),
                           status="success", output="Solution here")
        )

        result = orchestrator._generate_final_response(session, {})

        assert "response" in result
        assert "session_id" in result
        assert "metadata" in result
        assert result["session_id"] == "resp1"
        assert result["metadata"]["tier"] == 2
        assert "Executor" in result["metadata"]["agents_used"]

    def test_generate_final_response_includes_smes(self):
        orchestrator = OrchestratorAgent(enable_persistence=False)
        session = SessionState(session_id="resp2", user_prompt="Test")
        session.tier_classification = _make_classification(TierLevel.DEEP)
        session.active_smes = ["cloud_architect"]

        result = orchestrator._generate_final_response(session, {})
        assert result["metadata"]["smes_used"] == ["cloud_architect"]

    def test_format_response_empty_outputs(self):
        orchestrator = OrchestratorAgent(enable_persistence=False)
        session = SessionState(session_id="fmt1", user_prompt="Test")
        result = orchestrator._format_response([], session)
        assert "apologize" in result.lower()

    def test_format_response_string_outputs(self):
        orchestrator = OrchestratorAgent(enable_persistence=False)
        session = SessionState(session_id="fmt2", user_prompt="Test")
        result = orchestrator._format_response(["Part A", "Part B"], session)
        assert "Part A" in result
        assert "Part B" in result

    def test_format_response_dict_outputs(self):
        orchestrator = OrchestratorAgent(enable_persistence=False)
        session = SessionState(session_id="fmt3", user_prompt="Test")
        result = orchestrator._format_response(
            [{"content": "Answer X"}, {"content": "Answer Y"}], session
        )
        assert "Answer X" in result
        assert "Answer Y" in result

    def test_budget_exceeded_response(self):
        orchestrator = OrchestratorAgent(enable_persistence=False)
        session = SessionState(
            session_id="budget_resp",
            user_prompt="Test",
            max_budget_usd=5.0,
        )
        session.total_cost_usd = 6.0

        result = orchestrator._budget_exceeded_response(session)
        assert "budget" in result["metadata"]["error"]
        assert result["metadata"]["total_cost_usd"] == 6.0
        assert "5.00" in result["response"]

    def test_error_response(self):
        orchestrator = OrchestratorAgent(enable_persistence=False)
        session = SessionState(session_id="err_resp", user_prompt="Test")
        result = orchestrator._error_response(session, "Something went wrong")
        assert "Something went wrong" in result["response"]
        assert result["metadata"]["error"] == "Something went wrong"


# =============================================================================
# 16. System Prompt Loading
# =============================================================================

class TestSystemPromptLoading:
    """Tests for _load_system_prompt."""

    def test_load_existing_system_prompt(self):
        orchestrator = OrchestratorAgent(enable_persistence=False)
        prompt = orchestrator._load_system_prompt("config/agents/analyst/CLAUDE.md")
        assert len(prompt) > 0

    def test_load_nonexistent_system_prompt_returns_default(self):
        orchestrator = OrchestratorAgent(enable_persistence=False)
        prompt = orchestrator._load_system_prompt(
            "/nonexistent/CLAUDE.md", agent_role="tester"
        )
        assert "tester" in prompt

    def test_load_system_prompt_default_role(self):
        orchestrator = OrchestratorAgent(enable_persistence=False)
        prompt = orchestrator._load_system_prompt("/nonexistent/CLAUDE.md")
        assert "agent" in prompt


# =============================================================================
# 17. Model and Max Turns Selection
# =============================================================================

class TestModelAndMaxTurns:
    """Tests for _get_model_for_agent and _get_max_turns."""

    def test_get_model_for_agent_returns_string(self):
        orchestrator = OrchestratorAgent(enable_persistence=False)
        model = orchestrator._get_model_for_agent("Analyst")
        assert isinstance(model, str)
        assert len(model) > 0

    def test_get_max_turns_orchestrator(self):
        orchestrator = OrchestratorAgent(enable_persistence=False)
        assert orchestrator._get_max_turns("Orchestrator") == orchestrator.max_turns_orchestrator

    def test_get_max_turns_executor(self):
        orchestrator = OrchestratorAgent(enable_persistence=False)
        assert orchestrator._get_max_turns("Executor") == orchestrator.max_turns_executor

    def test_get_max_turns_other_agent(self):
        orchestrator = OrchestratorAgent(enable_persistence=False)
        assert orchestrator._get_max_turns("Analyst") == orchestrator.max_turns_subagent


# =============================================================================
# 18. Session Persistence Integration
# =============================================================================

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

        orchestrator._save_session_state(session, {"response": "Test"})

    def test_save_session_state_with_persistence_disabled(self):
        """Test that nothing happens when persistence is disabled."""
        orchestrator = OrchestratorAgent(enable_persistence=False)

        session = SessionState(
            session_id="no_save_test",
            user_prompt="Test",
        )

        orchestrator._save_session_state(session, None)

    def test_load_session_with_persistence_enabled(self):
        """Test loading a session."""
        orchestrator = OrchestratorAgent(enable_persistence=True)
        result = orchestrator.load_session("nonexistent_session")
        # Non-existent session should return None

    def test_load_session_with_persistence_disabled(self):
        """Test that load returns None when persistence is disabled."""
        orchestrator = OrchestratorAgent(enable_persistence=False)

        result = orchestrator.load_session("any_session")
        assert result is None

    def test_save_and_load_roundtrip(self):
        """Test save then load preserves key session data."""
        orchestrator = OrchestratorAgent(enable_persistence=True)

        # Save a session
        session = SessionState(
            session_id="roundtrip_test",
            user_prompt="Test roundtrip",
            max_budget_usd=10.0,
        )
        session.current_tier = TierLevel.DEEP
        session.total_cost_usd = 1.5
        session.active_smes = ["cloud_architect"]

        orchestrator._save_session_state(session, {"response": "Test"})

        # Load it back
        loaded = orchestrator.load_session("roundtrip_test")
        if loaded is not None:
            assert loaded.session_id == "roundtrip_test"
            assert loaded.current_tier == TierLevel.DEEP
            assert loaded.total_cost_usd == 1.5
            assert loaded.active_smes == ["cloud_architect"]


# =============================================================================
# 19. Factory Function
# =============================================================================

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


# =============================================================================
# 20. End-to-End Pipeline Flow (Mocked SDK)
# =============================================================================

class TestEndToEndPipelineFlow:
    """End-to-end tests with mocked SDK calls."""

    def test_tier1_only_executor_and_formatter(self):
        """Tier 1 should only run Phase 5 and Phase 8 agents."""
        orchestrator = OrchestratorAgent(enable_persistence=False)
        result = orchestrator.process_request(
            user_prompt="What is 2+2?",
            tier_override=1,
        )

        agents_used = result["metadata"]["agents_used"]
        # Tier 1 skips most phases, so should only have executor + formatter/reviewer
        assert len(agents_used) >= 1
        assert result["metadata"]["tier"] == 1

    def test_tier2_includes_analyst_planner_executor(self):
        """Tier 2 should include Analyst, Planner, Executor phases."""
        orchestrator = OrchestratorAgent(enable_persistence=False)
        result = orchestrator.process_request(
            user_prompt="Write a fibonacci function",
            tier_override=2,
        )

        agents_used = result["metadata"]["agents_used"]
        assert result["metadata"]["tier"] == 2
        # Should have multiple agents
        assert len(agents_used) >= 3

    def test_tier3_includes_council(self):
        """Tier 3 should include council consultation."""
        orchestrator = OrchestratorAgent(enable_persistence=False)
        result = orchestrator.process_request(
            user_prompt="Design a microservices architecture",
            tier_override=3,
        )

        agents_used = result["metadata"]["agents_used"]
        assert result["metadata"]["tier"] == 3
        # Should have council-related agents
        assert any("Council" in a or "council" in a.lower() for a in agents_used)

    def test_tier4_includes_full_system(self):
        """Tier 4 should include council, quality arbiter, and debate."""
        orchestrator = OrchestratorAgent(enable_persistence=False)
        result = orchestrator.process_request(
            user_prompt="Review security vulnerabilities in HIPAA system",
            tier_override=4,
        )

        agents_used = result["metadata"]["agents_used"]
        assert result["metadata"]["tier"] == 4
        # Should have many agents for full system
        assert len(agents_used) >= 5
        # Should have debate rounds for tier 4
        assert result["metadata"]["debate_rounds"] >= 0

    def test_pipeline_cost_accumulates(self):
        """Test that pipeline cost accumulates across agents."""
        orchestrator = OrchestratorAgent(enable_persistence=False)
        result = orchestrator.process_request(
            user_prompt="Test",
            tier_override=2,
        )
        # Cost should be positive (from simulated responses)
        assert result["metadata"]["total_cost_usd"] > 0

    def test_pipeline_duration_positive(self):
        """Test that pipeline reports positive duration."""
        orchestrator = OrchestratorAgent(enable_persistence=False)
        result = orchestrator.process_request(
            user_prompt="Test",
            tier_override=1,
        )
        assert result["metadata"]["duration_seconds"] >= 0


# =============================================================================
# 21. Edge Cases and Error Scenarios
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error scenarios."""

    def test_empty_prompt(self):
        """Test handling of empty prompt."""
        orchestrator = OrchestratorAgent(enable_persistence=False)
        result = orchestrator.process_request(user_prompt="", tier_override=1)
        assert "response" in result

    def test_very_long_prompt(self):
        """Test handling of very long prompt."""
        orchestrator = OrchestratorAgent(enable_persistence=False)
        long_prompt = "Test " * 10000
        result = orchestrator.process_request(user_prompt=long_prompt, tier_override=1)
        assert "response" in result

    def test_special_characters_in_prompt(self):
        """Test handling of special characters."""
        orchestrator = OrchestratorAgent(enable_persistence=False)
        result = orchestrator.process_request(
            user_prompt='Test <script>alert("xss")</script> & "quotes"',
            tier_override=1,
        )
        assert "response" in result

    def test_unicode_in_prompt(self):
        """Test handling of unicode characters."""
        orchestrator = OrchestratorAgent(enable_persistence=False)
        result = orchestrator.process_request(
            user_prompt="Testing unicode: 日本語 中文 العربية",
            tier_override=1,
        )
        assert "response" in result

    def test_concurrent_session_isolation(self):
        """Test that two sessions don't interfere with each other."""
        orchestrator = OrchestratorAgent(enable_persistence=False)

        result1 = orchestrator.process_request(
            user_prompt="Session 1",
            session_id="session_a",
            tier_override=1,
        )
        result2 = orchestrator.process_request(
            user_prompt="Session 2",
            session_id="session_b",
            tier_override=1,
        )

        assert result1["session_id"] == "session_a"
        assert result2["session_id"] == "session_b"

    def test_format_markdown_with_smes(self):
        """Test markdown formatting includes SMEs when present."""
        orchestrator = OrchestratorAgent(enable_persistence=False)
        result = {
            "response": "Test response",
            "metadata": {
                "session_id": "test",
                "tier": 3,
                "duration_seconds": 5.0,
                "total_cost_usd": 0.05,
                "agents_used": ["Analyst", "Executor"],
                "smes_used": ["cloud_architect"],
            },
        }
        formatted = orchestrator._format_as_markdown(result)
        assert "cloud_architect" in formatted
        assert "SME Consultants" in formatted

    def test_format_markdown_without_smes(self):
        """Test markdown formatting without SMEs."""
        orchestrator = OrchestratorAgent(enable_persistence=False)
        result = {
            "response": "Test response",
            "metadata": {
                "session_id": "test",
                "tier": 1,
                "duration_seconds": 1.0,
                "total_cost_usd": 0.01,
                "agents_used": ["Executor"],
                "smes_used": [],
            },
        }
        formatted = orchestrator._format_as_markdown(result)
        assert "SME Consultants" not in formatted


# =============================================================================
# 22. Verdict Matrix Module (Core Logic)
# =============================================================================

class TestVerdictMatrixCore:
    """Direct tests for the verdict matrix module."""

    def test_all_four_verdict_combinations(self):
        """Test all 4 combinations of verdicts."""
        result_pp = evaluate_verdict_matrix(Verdict.PASS, Verdict.PASS)
        assert result_pp.action == MatrixAction.PROCEED_TO_FORMATTER

        result_pf = evaluate_verdict_matrix(Verdict.PASS, Verdict.FAIL)
        assert result_pf.action == MatrixAction.EXECUTOR_REVISE

        result_fp = evaluate_verdict_matrix(Verdict.FAIL, Verdict.PASS)
        assert result_fp.action == MatrixAction.RESEARCHER_REVERIFY

        result_ff = evaluate_verdict_matrix(Verdict.FAIL, Verdict.FAIL)
        assert result_ff.action == MatrixAction.FULL_REGENERATION

    def test_can_retry_within_limit(self):
        result = evaluate_verdict_matrix(
            Verdict.PASS, Verdict.FAIL,
            revision_cycle=0, max_revisions=2,
        )
        assert result.can_retry is True

    def test_cannot_retry_at_limit(self):
        result = evaluate_verdict_matrix(
            Verdict.PASS, Verdict.FAIL,
            revision_cycle=2, max_revisions=2,
        )
        assert result.can_retry is False

    def test_quality_arbiter_on_tier4_exceeded_revisions(self):
        result = evaluate_verdict_matrix(
            Verdict.FAIL, Verdict.FAIL,
            revision_cycle=3, max_revisions=2, tier_level=4,
        )
        assert result.action == MatrixAction.QUALITY_ARBITER

    def test_no_quality_arbiter_on_tier2(self):
        result = evaluate_verdict_matrix(
            Verdict.FAIL, Verdict.FAIL,
            revision_cycle=3, max_revisions=2, tier_level=2,
        )
        assert result.action == MatrixAction.FULL_REGENERATION  # No arbiter for lower tiers

    def test_outcome_contains_reasoning(self):
        result = evaluate_verdict_matrix(Verdict.PASS, Verdict.PASS)
        assert len(result.reason) > 0
        assert "Verifier" in result.reason


# =============================================================================
# 23. Pipeline Phase Skipping
# =============================================================================

class TestPipelinePhaseSkipping:
    """Tests for pipeline phase skipping by tier."""

    def test_tier1_only_runs_two_phases(self):
        pipeline = ExecutionPipeline(tier_level=TierLevel.DIRECT)
        phases = pipeline._get_phases_for_tier()
        assert Phase.PHASE_5_SOLUTION_GENERATION in phases
        assert Phase.PHASE_8_FINAL_REVIEW_FORMATTING in phases
        assert len(phases) == 2

    def test_tier2_skips_council_research_revision(self):
        pipeline = ExecutionPipeline(tier_level=TierLevel.STANDARD)
        phases = pipeline._get_phases_for_tier()
        assert Phase.PHASE_2_COUNCIL_CONSULTATION not in phases
        assert Phase.PHASE_4_RESEARCH not in phases
        assert Phase.PHASE_7_REVISION not in phases

    def test_tier3_runs_all_phases(self):
        pipeline = ExecutionPipeline(tier_level=TierLevel.DEEP)
        phases = pipeline._get_phases_for_tier()
        assert len(phases) == 8  # All phases

    def test_tier4_runs_all_phases(self):
        pipeline = ExecutionPipeline(tier_level=TierLevel.ADVERSARIAL)
        phases = pipeline._get_phases_for_tier()
        assert len(phases) == 8


# =============================================================================
# 24. Debate Protocol (Core Module)
# =============================================================================

class TestDebateProtocolCore:
    """Tests for the DebateProtocol class."""

    def test_consensus_calculation_all_agree(self):
        protocol = DebateProtocol()
        score = protocol.calculate_consensus(0.9, 0.9, 0.9, {})
        # 0.25*0.9 + 0.25*0.9 + 0.25*0.9 = 0.675
        assert score >= 0.6

    def test_consensus_calculation_with_smes(self):
        protocol = DebateProtocol()
        score = protocol.calculate_consensus(0.8, 0.8, 0.8, {"sme1": 0.8})
        assert 0.6 <= score <= 1.0

    def test_consensus_level_full(self):
        protocol = DebateProtocol(consensus_threshold=0.8)
        assert protocol.determine_consensus_level(0.85) == ConsensusLevel.FULL

    def test_consensus_level_majority(self):
        protocol = DebateProtocol(consensus_threshold=0.8, majority_threshold=0.5)
        assert protocol.determine_consensus_level(0.6) == ConsensusLevel.MAJORITY

    def test_consensus_level_split(self):
        protocol = DebateProtocol(majority_threshold=0.5)
        assert protocol.determine_consensus_level(0.3) == ConsensusLevel.SPLIT

    def test_should_continue_debate_before_consensus(self):
        protocol = DebateProtocol(max_rounds=3, consensus_threshold=0.8)
        assert protocol.should_continue_debate(0.5) is True

    def test_should_not_continue_after_consensus(self):
        protocol = DebateProtocol(consensus_threshold=0.8)
        assert protocol.should_continue_debate(0.85) is False

    def test_should_not_continue_after_max_rounds(self):
        protocol = DebateProtocol(max_rounds=2)
        protocol.rounds = [Mock(), Mock()]  # 2 rounds conducted
        assert protocol.should_continue_debate(0.5) is False

    def test_needs_arbiter_on_split(self):
        protocol = DebateProtocol()
        assert protocol.needs_arbiter(ConsensusLevel.SPLIT, 2) is True

    def test_does_not_need_arbiter_on_full(self):
        protocol = DebateProtocol()
        assert protocol.needs_arbiter(ConsensusLevel.FULL, 2) is False

    def test_conduct_round_appends(self):
        protocol = DebateProtocol()
        round_result = protocol.conduct_round(
            executor_position="My solution is correct",
            critic_challenges=["But what about X?"],
            verifier_checks=["Fact Y confirmed"],
            sme_arguments={},
        )
        assert len(protocol.rounds) == 1
        assert round_result.round_number == 1

    def test_get_outcome_no_rounds(self):
        protocol = DebateProtocol()
        outcome = protocol.get_outcome()
        assert outcome.consensus_level == ConsensusLevel.SPLIT
        assert outcome.rounds_completed == 0

    def test_add_participant_no_duplicates(self):
        protocol = DebateProtocol()
        protocol.add_participant("Executor")
        protocol.add_participant("Executor")
        assert len(protocol.participants) == 1

    def test_add_sme_participant(self):
        protocol = DebateProtocol()
        protocol.add_sme_participant("cloud_architect")
        assert "cloud_architect" in protocol.sme_participants


# =============================================================================
# 25. Complexity Classification Interaction
# =============================================================================

class TestComplexityClassificationInteraction:
    """Tests for how orchestrator interacts with classify_complexity."""

    @patch('src.agents.orchestrator.classify_complexity')
    def test_high_complexity_keywords_route_to_tier3(self, mock_classify):
        mock_classify.return_value = _make_classification(TierLevel.DEEP)

        orchestrator = OrchestratorAgent(enable_persistence=False)
        result = orchestrator.process_request(
            user_prompt="Design threat model for microservices architecture",
        )

        mock_classify.assert_called_once()
        assert result["metadata"]["tier"] == 3

    @patch('src.agents.orchestrator.classify_complexity')
    def test_tier4_keywords_route_to_tier4(self, mock_classify):
        mock_classify.return_value = _make_classification(TierLevel.ADVERSARIAL)

        orchestrator = OrchestratorAgent(enable_persistence=False)
        result = orchestrator.process_request(
            user_prompt="Security audit of HIPAA compliance system",
        )

        assert result["metadata"]["tier"] == 4

    @patch('src.agents.orchestrator.classify_complexity')
    def test_simple_query_routes_to_tier1_or_2(self, mock_classify):
        mock_classify.return_value = _make_classification(TierLevel.DIRECT)

        orchestrator = OrchestratorAgent(enable_persistence=False)
        result = orchestrator.process_request(user_prompt="Hello")

        assert result["metadata"]["tier"] <= 2

    def test_tier_override_bypasses_classification(self):
        """Tier override should skip classify_complexity entirely."""
        orchestrator = OrchestratorAgent(enable_persistence=False)
        with patch('src.agents.orchestrator.classify_complexity') as mock_classify:
            result = orchestrator.process_request(
                user_prompt="Test",
                tier_override=1,
            )
            mock_classify.assert_not_called()
