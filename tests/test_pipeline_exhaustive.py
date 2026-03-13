"""
Exhaustive Tests for Pipeline Orchestration Module

Tests pipeline phases, state management, tier-based routing,
verdict matrix integration, debate protocol, and builder.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from pydantic import ValidationError

from src.core.pipeline import (
    Phase,
    PhaseStatus,
    AgentResult,
    PhaseResult,
    PipelineState,
    ExecutionPipeline,
    PipelineBuilder,
    create_execution_context,
    estimate_pipeline_duration,
)
from src.core.complexity import TierLevel, TierClassification
from src.core.verdict import MatrixAction, Verdict
from src.core.debate import DebateProtocol


# =============================================================================
# Phase Enum Tests
# =============================================================================

class TestPhase:
    def test_all_phases(self):
        assert len(Phase) == 8

    def test_phase_values(self):
        assert "Phase 1" in Phase.PHASE_1_TASK_INTELLIGENCE.value
        assert "Phase 2" in Phase.PHASE_2_COUNCIL_CONSULTATION.value
        assert "Phase 8" in Phase.PHASE_8_FINAL_REVIEW_FORMATTING.value


class TestPhaseStatus:
    def test_all_statuses(self):
        assert PhaseStatus.PENDING == "pending"
        assert PhaseStatus.IN_PROGRESS == "in_progress"
        assert PhaseStatus.COMPLETE == "complete"
        assert PhaseStatus.SKIPPED == "skipped"
        assert PhaseStatus.FAILED == "failed"

    def test_count(self):
        assert len(PhaseStatus) == 5


# =============================================================================
# AgentResult Tests
# =============================================================================

class TestAgentResult:
    def test_create(self):
        ar = AgentResult(
            agent_name="Executor", status="success",
            output={"result": "done"}, duration_ms=1000,
        )
        assert ar.agent_name == "Executor"
        assert ar.error is None
        assert ar.tokens_used == 0

    def test_with_error(self):
        ar = AgentResult(
            agent_name="Verifier", status="error",
            output=None, duration_ms=500, error="Timeout",
        )
        assert ar.error == "Timeout"

    def test_with_tokens(self):
        ar = AgentResult(
            agent_name="Analyst", status="success",
            output={}, duration_ms=200, tokens_used=1500,
        )
        assert ar.tokens_used == 1500


# =============================================================================
# PhaseResult Tests
# =============================================================================

class TestPhaseResult:
    def test_create(self):
        pr = PhaseResult(
            phase=Phase.PHASE_5_SOLUTION_GENERATION,
            status=PhaseStatus.COMPLETE,
            agent_results=[],
            duration_ms=5000,
        )
        assert pr.output is None
        assert pr.error is None

    def test_with_output(self):
        pr = PhaseResult(
            phase=Phase.PHASE_1_TASK_INTELLIGENCE,
            status=PhaseStatus.COMPLETE,
            agent_results=[AgentResult("Analyst", "success", {"data": 1}, 100)],
            duration_ms=100,
            output={"data": 1},
        )
        assert pr.output == {"data": 1}


# =============================================================================
# PipelineState Tests
# =============================================================================

class TestPipelineState:
    def test_defaults(self):
        state = PipelineState()
        assert state.current_phase is None
        assert state.completed_phases == []
        assert state.tier_level == TierLevel.STANDARD
        assert state.revision_cycle == 0
        assert state.debate_rounds == 0
        assert state.total_cost_usd == 0.0
        assert state.total_tokens == 0

    def test_with_values(self):
        state = PipelineState(
            tier_level=TierLevel.DEEP,
            revision_cycle=1,
            total_cost_usd=0.5,
            total_tokens=5000,
        )
        assert state.tier_level == TierLevel.DEEP
        assert state.revision_cycle == 1

    def test_revision_cycle_non_negative(self):
        with pytest.raises(ValidationError):
            PipelineState(revision_cycle=-1)

    def test_cost_non_negative(self):
        with pytest.raises(ValidationError):
            PipelineState(total_cost_usd=-0.1)


# =============================================================================
# ExecutionPipeline Tests
# =============================================================================

class TestExecutionPipeline:
    def test_default_init(self):
        pipeline = ExecutionPipeline()
        assert pipeline.tier_level == TierLevel.STANDARD
        assert pipeline.max_revisions == 2
        assert pipeline.max_debate_rounds == 2

    def test_custom_init(self):
        pipeline = ExecutionPipeline(
            tier_level=TierLevel.DEEP, max_revisions=3, max_debate_rounds=3
        )
        assert pipeline.tier_level == TierLevel.DEEP
        assert pipeline.max_revisions == 3

    # --------- Phase Skipping ---------

    def test_tier_1_skips_most_phases(self):
        pipeline = ExecutionPipeline(tier_level=TierLevel.DIRECT)
        assert pipeline._should_skip_phase(Phase.PHASE_1_TASK_INTELLIGENCE) is True
        assert pipeline._should_skip_phase(Phase.PHASE_2_COUNCIL_CONSULTATION) is True
        assert pipeline._should_skip_phase(Phase.PHASE_5_SOLUTION_GENERATION) is False
        assert pipeline._should_skip_phase(Phase.PHASE_8_FINAL_REVIEW_FORMATTING) is False

    def test_tier_2_skips_council_research_revision(self):
        pipeline = ExecutionPipeline(tier_level=TierLevel.STANDARD)
        assert pipeline._should_skip_phase(Phase.PHASE_2_COUNCIL_CONSULTATION) is True
        assert pipeline._should_skip_phase(Phase.PHASE_4_RESEARCH) is True
        assert pipeline._should_skip_phase(Phase.PHASE_7_REVISION) is True
        assert pipeline._should_skip_phase(Phase.PHASE_1_TASK_INTELLIGENCE) is False

    def test_tier_3_no_skip(self):
        pipeline = ExecutionPipeline(tier_level=TierLevel.DEEP)
        for phase in Phase:
            assert pipeline._should_skip_phase(phase) is False

    def test_tier_4_no_skip(self):
        pipeline = ExecutionPipeline(tier_level=TierLevel.ADVERSARIAL)
        for phase in Phase:
            assert pipeline._should_skip_phase(phase) is False

    # --------- Phases for Tier ---------

    def test_tier_1_phases(self):
        pipeline = ExecutionPipeline(tier_level=TierLevel.DIRECT)
        phases = pipeline._get_phases_for_tier()
        assert Phase.PHASE_5_SOLUTION_GENERATION in phases
        assert Phase.PHASE_8_FINAL_REVIEW_FORMATTING in phases
        assert len(phases) == 2

    def test_tier_2_phases(self):
        pipeline = ExecutionPipeline(tier_level=TierLevel.STANDARD)
        phases = pipeline._get_phases_for_tier()
        assert Phase.PHASE_1_TASK_INTELLIGENCE in phases
        assert Phase.PHASE_3_PLANNING in phases
        assert Phase.PHASE_5_SOLUTION_GENERATION in phases
        assert Phase.PHASE_2_COUNCIL_CONSULTATION not in phases

    def test_tier_3_all_phases(self):
        pipeline = ExecutionPipeline(tier_level=TierLevel.DEEP)
        phases = pipeline._get_phases_for_tier()
        assert len(phases) == 8

    # --------- Agents for Phase ---------

    def test_phase_1_agents(self):
        pipeline = ExecutionPipeline()
        agents = pipeline._get_agents_for_phase(Phase.PHASE_1_TASK_INTELLIGENCE)
        assert "Task Analyst" in agents

    def test_phase_5_agents(self):
        pipeline = ExecutionPipeline()
        agents = pipeline._get_agents_for_phase(Phase.PHASE_5_SOLUTION_GENERATION)
        assert "Executor" in agents

    def test_phase_6_agents(self):
        pipeline = ExecutionPipeline()
        agents = pipeline._get_agents_for_phase(Phase.PHASE_6_REVIEW)
        assert "Verifier" in agents
        assert "Critic" in agents

    def test_phase_8_agents(self):
        pipeline = ExecutionPipeline()
        agents = pipeline._get_agents_for_phase(Phase.PHASE_8_FINAL_REVIEW_FORMATTING)
        assert "Reviewer" in agents
        assert "Formatter" in agents

    # --------- Council Agents ---------

    def test_council_tier_1(self):
        pipeline = ExecutionPipeline(tier_level=TierLevel.DIRECT)
        assert pipeline._get_council_agents() == []

    def test_council_tier_3(self):
        pipeline = ExecutionPipeline(tier_level=TierLevel.DEEP)
        agents = pipeline._get_council_agents()
        assert "Domain Council Chair" in agents

    # --------- Parse Verdict ---------

    def test_parse_verdict_pass(self):
        pipeline = ExecutionPipeline()
        assert pipeline._parse_verdict({"verdict": "PASS"}) == Verdict.PASS

    def test_parse_verdict_fail(self):
        pipeline = ExecutionPipeline()
        assert pipeline._parse_verdict({"verdict": "FAIL"}) == Verdict.FAIL

    def test_parse_verdict_default(self):
        pipeline = ExecutionPipeline()
        assert pipeline._parse_verdict("some string") == Verdict.FAIL

    def test_parse_verdict_missing_key(self):
        pipeline = ExecutionPipeline()
        assert pipeline._parse_verdict({}) == Verdict.PASS

    # --------- Determine Phase Status ---------

    def test_determine_status_success(self):
        pipeline = ExecutionPipeline()
        results = [AgentResult("Executor", "success", {}, 100)]
        assert pipeline._determine_phase_status(results) == PhaseStatus.COMPLETE

    def test_determine_status_critical_error(self):
        pipeline = ExecutionPipeline()
        results = [AgentResult("Verifier", "error", None, 100, error="Timeout")]
        assert pipeline._determine_phase_status(results) == PhaseStatus.FAILED

    def test_determine_status_non_critical_error(self):
        pipeline = ExecutionPipeline()
        results = [
            AgentResult("Executor", "error", None, 100),
            AgentResult("Formatter", "success", {}, 100),
        ]
        assert pipeline._determine_phase_status(results) == PhaseStatus.COMPLETE

    def test_determine_status_all_error(self):
        pipeline = ExecutionPipeline()
        results = [AgentResult("Executor", "error", None, 100)]
        assert pipeline._determine_phase_status(results) == PhaseStatus.FAILED

    # --------- Extract Phase Output ---------

    def test_extract_output(self):
        pipeline = ExecutionPipeline()
        results = [
            AgentResult("A", "success", {"data": "result"}, 100),
            AgentResult("B", "success", {"data": "result2"}, 100),
        ]
        assert pipeline._extract_phase_output(results) == {"data": "result"}

    def test_extract_output_empty(self):
        pipeline = ExecutionPipeline()
        results = [AgentResult("A", "error", None, 100)]
        assert pipeline._extract_phase_output(results) is None

    # --------- Handle Phase Failure ---------

    def test_handle_failure_research_continues(self):
        pipeline = ExecutionPipeline()
        result = PhaseResult(Phase.PHASE_4_RESEARCH, PhaseStatus.FAILED, [], 0)
        assert pipeline._handle_phase_failure(Phase.PHASE_4_RESEARCH, result) is True

    def test_handle_failure_other_stops(self):
        pipeline = ExecutionPipeline()
        result = PhaseResult(Phase.PHASE_5_SOLUTION_GENERATION, PhaseStatus.FAILED, [], 0)
        assert pipeline._handle_phase_failure(Phase.PHASE_5_SOLUTION_GENERATION, result) is False

    # --------- Execute Phase ---------

    def test_execute_phase_skipped(self):
        pipeline = ExecutionPipeline(tier_level=TierLevel.DIRECT)

        def mock_executor(**kwargs):
            return AgentResult("Agent", "success", {}, 100)

        result = pipeline.execute_phase(Phase.PHASE_1_TASK_INTELLIGENCE, mock_executor, {})
        assert result.status == PhaseStatus.SKIPPED

    def test_execute_phase_success(self):
        pipeline = ExecutionPipeline(tier_level=TierLevel.DEEP)

        def mock_executor(**kwargs):
            return AgentResult(kwargs["agent_name"], "success", {"data": "ok"}, 100)

        result = pipeline.execute_phase(Phase.PHASE_5_SOLUTION_GENERATION, mock_executor, {})
        assert result.status == PhaseStatus.COMPLETE
        assert len(result.agent_results) > 0

    # --------- Initiate Debate ---------

    def test_initiate_debate(self):
        pipeline = ExecutionPipeline()
        context = {"active_smes": ["cloud_architect"]}
        dp = pipeline.initiate_debate(context)
        assert isinstance(dp, DebateProtocol)
        assert "Executor" in dp.participants
        assert "cloud_architect" in dp.sme_participants

    def test_initiate_debate_no_smes(self):
        pipeline = ExecutionPipeline()
        dp = pipeline.initiate_debate({})
        assert len(dp.sme_participants) == 0

    # --------- Invoke Quality Arbiter ---------

    def test_invoke_quality_arbiter(self):
        pipeline = ExecutionPipeline()
        context = {}
        pipeline._invoke_quality_arbiter(context)
        assert context.get("require_arbiter") is True


# =============================================================================
# PipelineBuilder Tests
# =============================================================================

class TestPipelineBuilder:
    def test_for_tier(self):
        pipeline = PipelineBuilder.for_tier(TierLevel.DEEP)
        assert pipeline.tier_level == TierLevel.DEEP

    def test_from_classification(self):
        classification = TierClassification(
            tier=TierLevel.ADVERSARIAL, reasoning="Test",
            confidence=0.9, estimated_agents=18,
            requires_council=True, requires_smes=True,
        )
        pipeline = PipelineBuilder.from_classification(classification)
        assert pipeline.tier_level == TierLevel.ADVERSARIAL


# =============================================================================
# create_execution_context Tests
# =============================================================================

class TestCreateExecutionContext:
    def test_basic_context(self):
        tc = TierClassification(
            tier=TierLevel.STANDARD, reasoning="Test", confidence=0.7,
            estimated_agents=7, requires_council=False, requires_smes=False,
        )
        ctx = create_execution_context("Hello", tc)
        assert ctx["user_prompt"] == "Hello"
        assert ctx["tier"] == TierLevel.STANDARD
        assert ctx["requires_council"] is False

    def test_with_session_id(self):
        tc = TierClassification(
            tier=TierLevel.STANDARD, reasoning="Test", confidence=0.7,
            estimated_agents=7, requires_council=False, requires_smes=False,
        )
        ctx = create_execution_context("Hello", tc, session_id="sess123")
        assert ctx["session_id"] == "sess123"

    def test_with_additional_context(self):
        tc = TierClassification(
            tier=TierLevel.STANDARD, reasoning="Test", confidence=0.7,
            estimated_agents=7, requires_council=False, requires_smes=False,
        )
        ctx = create_execution_context("Hello", tc, additional_context={"extra": "data"})
        assert ctx["extra"] == "data"


# =============================================================================
# estimate_pipeline_duration Tests
# =============================================================================

class TestEstimatePipelineDuration:
    def test_tier_1_fastest(self):
        est = estimate_pipeline_duration(TierLevel.DIRECT)
        assert est["min"] < est["max"]
        assert est["estimated"] >= est["min"]
        assert est["estimated"] <= est["max"]

    def test_tier_4_slowest(self):
        est = estimate_pipeline_duration(TierLevel.ADVERSARIAL)
        est_t1 = estimate_pipeline_duration(TierLevel.DIRECT)
        assert est["estimated"] > est_t1["estimated"]

    def test_all_tiers_have_estimates(self):
        for tier in TierLevel:
            est = estimate_pipeline_duration(tier)
            assert "min" in est
            assert "max" in est
            assert "estimated" in est

    def test_increasing_duration(self):
        durations = [estimate_pipeline_duration(t)["estimated"] for t in TierLevel]
        for i in range(len(durations) - 1):
            assert durations[i] <= durations[i + 1]
