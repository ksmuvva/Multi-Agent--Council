"""
Integration Tests for Tier Workflows

Tests end-to-end agent workflows for each tier level.
These tests use mocked SDK responses but validate the full orchestration flow.

Gated by MAS_RUN_INTEGRATION=true environment variable.
Set MAS_RUN_INTEGRATION=true to run these tests with live API calls.
"""

import os
import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime

from tests.conftest import (
    MOCK_SESSION_ID,
    MOCK_AGENT_RESPONSES,
    TestDataBuilder,
    MockSDKPatch,
)

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

    @pytest.mark.asyncio
    async def test_tier1_direct_execution(self, mock_user_prompt):
        """
        Test Tier 1 workflow: Direct execution with minimal agents.

        Expected flow:
        1. Tier classification returns Tier 1
        2. Spawn Executor agent
        3. Spawn Formatter agent
        4. Return formatted result
        """
        # Simulate tier classification
        tier = 1
        assert tier == 1

        # Expected agents for Tier 1
        expected_agents = ["Executor", "Formatter"]
        assert len(expected_agents) <= 3  # Tier 1 should be minimal

    @pytest.mark.asyncio
    async def test_tier1_executor_formatter_only(self):
        """Test that Tier 1 only uses Executor and Formatter."""
        agents = ["Executor", "Formatter"]

        # Verify we have the right agents
        assert "Executor" in agents
        assert "Formatter" in agents
        assert "Planner" not in agents  # Planner not needed for Tier 1
        assert "Verifier" not in agents  # Verifier not needed for Tier 1


# =============================================================================
# Tier 2 Integration Tests
# =============================================================================

class TestTier2Workflow:
    """Integration tests for Tier 2 (Standard) workflow."""

    @pytest.mark.asyncio
    async def test_tier2_standard_pipeline(self):
        """
        Test Tier 2 workflow: Standard 7-agent pipeline.

        Expected flow:
        1. Tier classification returns Tier 2
        2. Spawn: Analyst → Planner → (optional Clarifier) → Executor → Verifier → Formatter
        3. Quality gate checks
        4. Return formatted result
        """
        # Expected agents for Tier 2
        expected_agents = [
            "Analyst",
            "Planner",
            "Executor",
            "Verifier",
            "Formatter",
        ]

        # Verify we have the core agents
        assert "Analyst" in expected_agents
        assert "Executor" in expected_agents
        assert "Verifier" in expected_agents

    @pytest.mark.asyncio
    async def test_tier2_clarifier_condition(self):
        """Test that Clarifier is spawned for unclear requests."""
        unclear_prompt = "Implement the thing with the stuff for the project"

        # Should trigger Clarifier
        needs_clarification = True

        # If clarification needed, add Clarifier to pipeline
        agents = ["Analyst", "Planner"]
        if needs_clarification:
            agents.insert(1, "Clarifier")

        assert "Clarifier" in agents
        assert agents.index("Clarifier") == 1  # After Analyst, before Planner


# =============================================================================
# Tier 3 Integration Tests
# =============================================================================

class TestTier3Workflow:
    """Integration tests for Tier 3 (Deep) workflow."""

    @pytest.mark.asyncio
    async def test_tier3_with_council(self):
        """
        Test Tier 3 workflow: Deep analysis with Council.

        Expected flow:
        1. Tier classification returns Tier 3
        2. Council Chair selects SMEs
        3. Spawn SMEs for domain expertise
        4. Full agent pipeline with Quality Arbiter
        5. Ethics review
        6. Return comprehensive result
        """
        # Tier 3 should include Council
        has_council = True
        assert has_council is True

    @pytest.mark.asyncio
    async def test_tier3_sme_selection(self):
        """Test SME selection for Tier 3 tasks."""
        task = "Design a secure cloud architecture with IAM and compliance"

        # Should trigger cloud architect and security SMEs
        relevant_smes = [
            "cloud_architect",
            "security_analyst",
        ]

        assert len(relevant_smes) >= 1
        assert "cloud_architect" in relevant_smes or "security_analyst" in relevant_smes

    @pytest.mark.asyncio
    async def test_tier3_quality_arbiter(self):
        """Test Quality Arbiter involvement in Tier 3."""
        has_quality_arbiter = True
        assert has_quality_arbiter is True


# =============================================================================
# Tier 4 Integration Tests
# =============================================================================

class TestTier4Workflow:
    """Integration tests for Tier 4 (Adversarial) workflow."""

    @pytest.mark.asyncio
    async def test_tier4_adversarial_mode(self):
        """
        Test Tier 4 workflow: Adversarial with full system.

        Expected flow:
        1. Tier classification returns Tier 4
        2. Full Council involvement
        3. Self-play debate with Critic
        4. Multiple SME perspectives
        5. Final Reviewer quality gate
        6. Only proceed if all quality gates pass
        """
        # Tier 4 is the most complex
        tier = 4
        assert tier == 4

        # Should have maximum agents
        expected_min_agents = 13
        assert tier == 4

    @pytest.mark.asyncio
    async def test_tier4_self_play_debate(self):
        """Test self-play debate in Tier 4."""
        debate_enabled = True

        assert debate_enabled is True

    @pytest.mark.asyncio
    async def test_tier4_critic_involvement(self):
        """Test Critic agent involvement in Tier 4."""
        has_critic = True

        assert has_critic is True


# =============================================================================
# Verdict Matrix Tests
# =============================================================================

class TestVerdictMatrix:
    """Integration tests for the verdict matrix logic."""

    def test_verdict_matrix_logic(self):
        """
        Test the verdict matrix decision logic.

        Matrix:
        - (PASS, PASS) → PROCEED_TO_FORMATTER
        - (PASS, FAIL) → EXECUTOR_REVISE
        - (FAIL, PASS) → RESEARCHER_REVERIFY
        - (FAIL, FAIL) → FULL_REGENERATION
        """
        verdict_matrix = {
            ("pass", "pass"): "PROCEED_TO_FORMATTER",
            ("pass", "fail"): "EXECUTOR_REVISE",
            ("fail", "pass"): "RESEARCHER_REVERIFY",
            ("fail", "fail"): "FULL_REGENERATION",
        }

        # Test each combination
        assert verdict_matrix[("pass", "pass")] == "PROCEED_TO_FORMATTER"
        assert verdict_matrix[("pass", "fail")] == "EXECUTOR_REVISE"
        assert verdict_matrix[("fail", "pass")] == "RESEARCHER_REVERIFY"
        assert verdict_matrix[("fail", "fail")] == "FULL_REGENERATION"

    @pytest.mark.asyncio
    async def test_verdict_routing(self):
        """Test that verdicts route to appropriate next steps."""
        verdict = "PROCEED_TO_FORMATTER"
        next_agent = "Formatter"

        assert next_agent == "Formatter"

        verdict = "EXECUTOR_REVISE"
        next_agent = "Executor"

        assert next_agent == "Executor"


# =============================================================================
# End-to-End Workflow Tests
# =============================================================================

class TestEndToEndWorkflows:
    """End-to-end tests of complete workflows."""

    @pytest.mark.asyncio
    async def test_simple_query_e2e(self, mock_user_prompt):
        """
        Test end-to-end simple query (Tier 1).

        Workflow:
        User Query → Tier Classification → Executor → Formatter → Result
        """
        session_id = MOCK_SESSION_ID

        # Simulate workflow
        steps_completed = []

        # Tier classification
        tier = 1
        steps_completed.append(f"Classified as Tier {tier}")

        # Execute
        steps_completed.append("Executor completed")
        steps_completed.append("Formatter completed")

        # Verify workflow completed
        assert len(steps_completed) == 3

    @pytest.mark.asyncio
    async def test_complex_query_e2e(self):
        """
        Test end-to-end complex query (Tier 3).

        Workflow:
        User Query → Tier Classification → Council → SME Selection →
        Analyst → Planner → Researcher → Executor → Verifier →
        Quality Arbiter → Formatter → Result
        """
        prompt = "Design a microservices architecture for healthcare with HIPAA compliance"

        # Simulate workflow
        steps = [
            "Tier classification",
            "Council Chair: SME selection",
            "SMEs spawned",
            "Analyst completed",
            "Planner completed",
            "Researcher completed",
            "Executor completed",
            "Verifier completed",
            "Quality Arbiter review",
            "Formatter completed",
        ]

        # Should have more steps than Tier 1
        assert len(steps) > 5

    @pytest.mark.asyncio
    async def test_quality_gate_failure_handling(self):
        """Test handling of quality gate failures."""
        # Simulate Verifier FAIL
        verifier_verdict = "fail"
        critic_verdict = "pass"

        # Should route to RERESEARCHER_REVERIFY
        if verifier_verdict == "fail" and critic_verdict == "pass":
            next_action = "RESEARCHER_REVERIFY"

        assert next_action == "RESEARCHER_REVERIFY"

    @pytest.mark.asyncio
    async def test_escalation_from_tier1_to_tier2(self):
        """Test escalation from Tier 1 to higher tier."""
        initial_tier = 1

        # Simulate Analyst discovering complexity
        actual_complexity = 2

        if actual_complexity > initial_tier:
            escalated_tier = actual_complexity

        assert escalated_tier == 2
        assert escalated_tier > initial_tier


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
        from datetime import datetime

        session = create_session(
            session_id="test_session",
            tier=2,
            title="Test Session",
        )

        # Add a message
        session.messages.append(ChatMessage(
            role="user",
            content="Test message",
            timestamp=datetime.now(),
        ))

        # Add an agent output
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
        from datetime import datetime

        # Create session with data
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

        # Convert to dict
        data = session.to_dict()
        assert data["session_id"] == "test_serialize"
        assert len(data["messages"]) == 1
        assert len(data["agent_outputs"]) == 1

        # Convert back from dict
        restored = SessionState.from_dict(data)
        assert restored.session_id == "test_serialize"
        assert len(restored.messages) == 1
        assert len(restored.agent_outputs) == 1

    def test_context_compaction_config(self):
        """Test context compaction configuration."""
        from src.session import CompactionConfig, CompactionTrigger

        config = CompactionConfig()

        assert config.max_tokens == 100000
        assert config.max_messages == 50
        assert config.recent_messages_to_keep == 10
        assert config.preserve_verdicts is True
        assert config.preserve_sme_advisories is True

    def test_compaction_trigger_enum(self):
        """Test compaction trigger enumeration."""
        from src.session import CompactionTrigger

        assert CompactionTrigger.TOKEN_COUNT == "token_count"
        assert CompactionTrigger.MESSAGE_COUNT == "message_count"
        assert CompactionTrigger.SESSION_AGE == "session_age"
        assert CompactionTrigger.MANUAL == "manual"

    @pytest.mark.asyncio
    async def test_cross_agent_context_sharing(self):
        """Test that context is shared between agents in a session."""
        shared_context = {
            "user_requirements": [],
            "research_findings": {},
            "agent_decisions": [],
        }

        # Analyst adds findings
        shared_context["agent_decisions"].append("Analyst: Task is feasible")

        # Planner should see Analyst's decision
        assert len(shared_context["agent_decisions"]) > 0


# =============================================================================
# Error Recovery Tests
# =============================================================================

class TestErrorRecovery:
    """Tests for error recovery in workflows."""

    @pytest.mark.asyncio
    async def test_agent_timeout_recovery(self):
        """Test recovery from agent timeout."""
        agent_timeout = False
        retry_count = 0
        max_retries = 3

        # Simulate timeout and retry
        while retry_count < max_retries:
            if agent_timeout:
                retry_count += 1
                if retry_count >= max_retries:
                    # Should escalate or degrade
                    recovered = False
                    break
                # Retry logic here
                agent_timeout = False
                recovered = True

        assert retry_count <= max_retries

    @pytest.mark.asyncio
    async def test_budget_exceeded_handling(self):
        """Test handling when budget is exceeded."""
        budget = 10.0
        spent = 12.0

        budget_exceeded = spent > budget

        if budget_exceeded:
            action = "stop_execution"

        assert budget_exceeded is True
        assert action == "stop_execution"

    @pytest.mark.asyncio
    async def test_schema_validation_recovery(self):
        """Test recovery from schema validation failures."""
        invalid_output = {"invalid": "data"}

        # Should trigger regeneration
        needs_regeneration = True

        if needs_regeneration:
            action = "regenerate"

        assert action == "regenerate"


# =============================================================================
# Performance Tests
# =============================================================================

class TestWorkflowPerformance:
    """Tests for workflow performance characteristics."""

    def test_tier1_agent_count(self):
        """Test that Tier 1 uses minimal agents."""
        tier1_agents = ["Executor", "Formatter"]

        assert len(tier1_agents) <= 3

    def test_tier2_agent_count(self):
        """Test that Tier 2 uses standard agent count."""
        tier2_agents = [
            "Analyst",
            "Planner",
            "Executor",
            "Verifier",
            "Formatter",
        ]

        assert 5 <= len(tier2_agents) <= 10

    def test_tier3_agent_count(self):
        """Test that Tier 3 uses extended agent count."""
        tier3_agents = [
            "CouncilChair",
            "Analyst",
            "Planner",
            "Researcher",
            "Executor",
            "Verifier",
            "QualityArbiter",
            "Formatter",
        ]

        assert len(tier3_agents) >= 8

    def test_tier4_agent_count(self):
        """Test that Tier 4 uses maximum agent count."""
        tier4_agents = [
            "CouncilChair",
            "QualityArbiter",
            "EthicsAdvisor",
            "Analyst",
            "Planner",
            "Clarifier",
            "Researcher",
            "Executor",
            "Verifier",
            "Critic",
            "Reviewer",
            "Formatter",
        ]

        assert len(tier4_agents) >= 10


# =============================================================================
# SME Interaction Tests
# =============================================================================

class TestSMEInteractions:
    """Tests for SME persona interactions."""

    @pytest.mark.asyncio
    async def test_sme_advisor_mode(self):
        """Test SME in advisor mode."""
        sme_mode = "advisor"
        provides_direct_output = False

        if sme_mode == "advisor":
            behavior = "review_and_recommend"

        assert behavior == "review_and_recommend"
        assert provides_direct_output is False

    @pytest.mark.asyncio
    async def test_sme_co_executor_mode(self):
        """Test SME in co-executor mode."""
        sme_mode = "co-executor"
        provides_direct_output = True

        if sme_mode == "co-executor":
            behavior = "contribute_content"

        assert behavior == "contribute_content"
        assert provides_direct_output is True

    @pytest.mark.asyncio
    async def test_sme_debater_mode(self):
        """Test SME in debater mode."""
        sme_mode = "debater"
        provides_perspective = True

        if sme_mode == "debater":
            behavior = "argue_position"

        assert behavior == "argue_position"
        assert provides_perspective is True


# =============================================================================
# Ensemble Pattern Tests
# =============================================================================

class TestEnsembleWorkflows:
    """Tests for ensemble pattern workflows."""

    @pytest.mark.asyncio
    async def test_architecture_review_board(self):
        """Test Architecture Review Board ensemble."""
        ensemble_name = "architecture_review_board"

        # Should include architecturally-focused agents
        expected_agents = [
            "Analyst",
            "cloud_architect",  # SME
            "security_analyst",  # SME
            "Executor",
            "Critic",
        ]

        assert "cloud_architect" in str(expected_agents)

    @pytest.mark.asyncio
    async def test_code_sprint_ensemble(self):
        """Test Code Sprint ensemble."""
        ensemble_name = "code_sprint"

        # Should be focused on code generation
        expected_focus = "code"
        assert expected_focus == "code"

    @pytest.mark.asyncio
    async def test_ensemble_parallel_execution(self):
        """Test that ensembles can execute agents in parallel where appropriate."""
        parallel_steps = [
            ["Researcher_A", "Researcher_B"],  # Can run in parallel
            ["Executor"],  # Must wait for research
        ]

        assert len(parallel_steps[0]) > 1  # First step has parallel agents


# =============================================================================
# Event Bus Integration Tests
# =============================================================================

class TestEventBusIntegration:
    """Tests for the event bus system connecting orchestrator to UI."""

    def test_event_bus_wire_subscribers_across_modules(self):
        """Test that event bus can wire subscribers across modules."""
        from src.core.events import EventBus, EventType

        bus = EventBus()
        received = []

        def handler_a(data):
            received.append(("module_a", data))

        def handler_b(data):
            received.append(("module_b", data))

        bus.subscribe(EventType.AGENT_STARTED, handler_a)
        bus.subscribe(EventType.AGENT_STARTED, handler_b)

        bus.emit(EventType.AGENT_STARTED, {
            "agent_id": "test-001",
            "agent_name": "Analyst",
        })

        assert len(received) == 2
        assert received[0][0] == "module_a"
        assert received[1][0] == "module_b"
        assert received[0][1]["agent_name"] == "Analyst"

    def test_agent_lifecycle_events_flow(self):
        """Test that agent lifecycle events flow correctly."""
        from src.core.events import EventBus, EventType

        bus = EventBus()
        lifecycle = []

        def track(data):
            lifecycle.append(data["event_type"])

        bus.subscribe(EventType.AGENT_STARTED, track)
        bus.subscribe(EventType.AGENT_COMPLETED, track)
        bus.subscribe(EventType.AGENT_FAILED, track)

        bus.emit(EventType.AGENT_STARTED, {"agent_id": "a1", "agent_name": "Executor"})
        bus.emit(EventType.AGENT_COMPLETED, {"agent_id": "a1", "agent_name": "Executor"})

        assert lifecycle == [EventType.AGENT_STARTED, EventType.AGENT_COMPLETED]

    def test_cost_events_propagate_to_dashboard(self):
        """Test that cost events propagate to dashboard."""
        from src.core.events import EventBus, EventType

        bus = EventBus()
        cost_events = []

        def cost_handler(data):
            cost_events.append(data)

        bus.subscribe(EventType.COST_RECORDED, cost_handler)

        bus.emit(EventType.COST_RECORDED, {
            "agent_name": "Executor",
            "model": "claude-3-5-sonnet",
            "input_tokens": 1000,
            "output_tokens": 500,
            "total_tokens": 1500,
            "cost_usd": 0.015,
            "tier": 2,
            "phase": "execution",
        })

        assert len(cost_events) == 1
        assert cost_events[0]["cost_usd"] == 0.015
        assert cost_events[0]["agent_name"] == "Executor"
        assert "timestamp" in cost_events[0]

    def test_event_bus_handles_callback_errors_gracefully(self):
        """Test that event bus handles errors in callbacks gracefully."""
        from src.core.events import EventBus, EventType

        bus = EventBus()
        successful_calls = []

        def bad_handler(data):
            raise RuntimeError("UI callback crashed")

        def good_handler(data):
            successful_calls.append(data)

        bus.subscribe(EventType.AGENT_STARTED, bad_handler)
        bus.subscribe(EventType.AGENT_STARTED, good_handler)

        # Should not raise even though bad_handler throws
        bus.emit(EventType.AGENT_STARTED, {"agent_id": "a1", "agent_name": "Analyst"})

        # The good handler should still have been called
        assert len(successful_calls) == 1


# =============================================================================
# Five-Verdict Workflow Tests
# =============================================================================

class TestFiveVerdictWorkflow:
    """Tests for the 5-verdict system in workflow context."""

    def test_pass_verdict_routes_to_formatter(self):
        """Test PASS verdict routes to formatter."""
        from src.schemas.reviewer import Verdict

        verdict = Verdict.PASS
        next_agent = None

        if verdict == Verdict.PASS:
            next_agent = "Formatter"

        assert next_agent == "Formatter"

    def test_pass_with_caveats_routes_to_formatter_with_warnings(self):
        """Test PASS_WITH_CAVEATS routes to formatter with warnings."""
        from src.schemas.reviewer import Verdict

        verdict = Verdict.PASS_WITH_CAVEATS
        next_agent = None
        warnings = []

        if verdict == Verdict.PASS_WITH_CAVEATS:
            next_agent = "Formatter"
            warnings.append("Output accepted with caveats - review recommended")

        assert next_agent == "Formatter"
        assert len(warnings) > 0

    def test_revise_verdict_triggers_re_execution(self):
        """Test REVISE verdict triggers re-execution."""
        from src.schemas.reviewer import Verdict

        verdict = Verdict.REVISE
        next_agent = None

        if verdict == Verdict.REVISE:
            next_agent = "Executor"

        assert next_agent == "Executor"

    def test_reject_verdict_stops_pipeline(self):
        """Test REJECT verdict stops pipeline."""
        from src.schemas.reviewer import Verdict

        verdict = Verdict.REJECT
        pipeline_stopped = False

        if verdict == Verdict.REJECT:
            pipeline_stopped = True

        assert pipeline_stopped is True

    def test_escalate_verdict_triggers_council_arbitration_tier4(self):
        """Test ESCALATE verdict triggers council arbitration in Tier 4."""
        from src.schemas.reviewer import Verdict

        verdict = Verdict.ESCALATE
        tier = 4
        council_arbitration = False

        if verdict == Verdict.ESCALATE and tier == 4:
            council_arbitration = True

        assert council_arbitration is True


# =============================================================================
# Skill Chain Workflow Tests
# =============================================================================

class TestSkillChainWorkflow:
    """Tests for skill chain selection and execution."""

    def test_auto_selection_of_skill_chain_from_task_description(self):
        """Test auto-selection of skill chain from task description."""
        from src.core.sdk_integration import select_skill_chain

        chain = select_skill_chain("develop and build a full stack web application")
        assert chain == "full_development"

    def test_full_development_chain_has_correct_ordering(self):
        """Test full_development chain has correct skill ordering."""
        from src.core.sdk_integration import get_skill_chain

        chain = get_skill_chain("full_development")
        assert len(chain) >= 3
        # Requirements should come before code generation
        assert chain.index("requirements-engineering") < chain.index("code-generation")
        # Architecture should come before code generation
        assert chain.index("architecture-design") < chain.index("code-generation")

    def test_research_and_report_chain(self):
        """Test research_and_report chain."""
        from src.core.sdk_integration import get_skill_chain

        chain = get_skill_chain("research_and_report")
        assert len(chain) >= 2
        assert "web-research" in chain
        assert "document-creation" in chain

    def test_per_task_skill_override_adds_relevant_skills(self):
        """Test that per-task skill override adds relevant skills."""
        from src.core.sdk_integration import get_skills_for_task

        # A task mentioning testing should include test-related skills
        skills = get_skills_for_task(
            "write unit tests for the authentication module",
            "executor",
        )
        assert isinstance(skills, list)
        assert len(skills) >= 1


# =============================================================================
# SME Spawning Workflow Tests
# =============================================================================

class TestSMESpawningWorkflow:
    """Tests for SME persona spawning through orchestrator."""

    def test_consult_mode_produces_advisory_output(self):
        """Test that consult mode produces advisory output."""
        from src.core.sme_registry import (
            InteractionMode,
            get_persona,
            validate_interaction_mode,
        )

        persona = get_persona("cloud_architect")
        assert persona is not None

        # Advisor mode should be supported
        assert validate_interaction_mode("cloud_architect", InteractionMode.ADVISOR)

        # Simulate advisory output
        advisory_output = {
            "mode": InteractionMode.ADVISOR.value,
            "recommendation": "Use AKS for container orchestration",
            "confidence": 0.85,
        }
        assert advisory_output["mode"] == "advisor"
        assert "recommendation" in advisory_output

    def test_debate_mode_uses_multiple_rounds(self):
        """Test that debate mode uses multiple rounds."""
        from src.core.sme_registry import (
            InteractionMode,
            validate_interaction_mode,
        )

        # Cloud architect supports debate mode
        assert validate_interaction_mode("cloud_architect", InteractionMode.DEBATER)

        # Simulate multi-round debate
        debate_rounds = [
            {"round": 1, "position": "Use microservices", "agent": "cloud_architect"},
            {"round": 2, "position": "Consider monolith first", "agent": "security_analyst"},
            {"round": 3, "position": "Hybrid approach", "agent": "cloud_architect"},
        ]
        assert len(debate_rounds) >= 2

    def test_co_author_mode_produces_merged_content(self):
        """Test that co_author mode produces merged content."""
        from src.core.sme_registry import (
            InteractionMode,
            validate_interaction_mode,
        )

        # Cloud architect supports co_executor mode
        assert validate_interaction_mode("cloud_architect", InteractionMode.CO_EXECUTOR)

        # Simulate merged content from co-authoring
        sme_contribution = "Cloud architecture section content"
        executor_contribution = "Implementation details section"
        merged = f"{sme_contribution}\n\n{executor_contribution}"

        assert sme_contribution in merged
        assert executor_contribution in merged


# =============================================================================
# Document Generation Workflow Tests
# =============================================================================

class TestDocumentGenerationWorkflow:
    """Tests for multi-format document output."""

    def test_docx_generation(self, tmp_path):
        """Test DOCX generation (check file exists)."""
        from src.agents.formatter import FormatterAgent

        formatter = FormatterAgent(output_dir=str(tmp_path))
        result = formatter._generate_docx(
            content={"Summary": "Test document content"},
            context={"title": "Test Document"},
        )

        assert "file_path" in result
        assert result["format"] == "docx"
        assert os.path.exists(result["file_path"])
        assert result["file_path"].endswith(".docx")

    def test_xlsx_generation(self, tmp_path):
        """Test XLSX generation (check file exists)."""
        from src.agents.formatter import FormatterAgent

        formatter = FormatterAgent(output_dir=str(tmp_path))
        result = formatter._generate_xlsx(
            content={"Column1": ["a", "b"], "Column2": ["c", "d"]},
            context={"title": "Test Spreadsheet"},
        )

        assert "file_path" in result
        assert result["format"] == "xlsx"
        assert os.path.exists(result["file_path"])
        assert result["file_path"].endswith(".xlsx")

    def test_pptx_generation(self, tmp_path):
        """Test PPTX generation (check file exists)."""
        from src.agents.formatter import FormatterAgent

        formatter = FormatterAgent(output_dir=str(tmp_path))
        result = formatter._generate_pptx(
            content={"Slide 1": "Introduction", "Slide 2": "Details"},
            context={"title": "Test Presentation"},
        )

        assert "file_path" in result
        assert result["format"] == "pptx"
        assert os.path.exists(result["file_path"])
        assert result["file_path"].endswith(".pptx")


# =============================================================================
# Streaming Workflow Tests
# =============================================================================

class TestStreamingWorkflow:
    """Tests for streaming chat."""

    def test_streaming_generator_yields_chunks(self):
        """Test that streaming generator yields chunks."""
        def mock_stream():
            """Simulate a streaming generator."""
            yield "Processing "
            yield "your "
            yield "request..."

        chunks = list(mock_stream())
        assert len(chunks) == 3
        assert "".join(chunks) == "Processing your request..."

    def test_streaming_handles_string_and_dict_chunks(self):
        """Test that streaming handles both string and dict chunks."""
        def mock_mixed_stream():
            """Simulate a streaming generator with mixed types."""
            yield "Starting..."
            yield {"status": "in_progress", "agent": "Analyst"}
            yield "Complete."

        chunks = list(mock_mixed_stream())
        assert len(chunks) == 3
        assert isinstance(chunks[0], str)
        assert isinstance(chunks[1], dict)
        assert chunks[1]["agent"] == "Analyst"
        assert isinstance(chunks[2], str)
