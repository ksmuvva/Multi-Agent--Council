"""
Tests for the PlannerAgent.

Tests execution plan creation, step generation, agent assignment,
parallel group identification, and critical path calculation.
"""

import pytest
from unittest.mock import patch, mock_open

from src.agents.planner import PlannerAgent, create_planner
from src.schemas.planner import ExecutionPlan, ExecutionStep, ParallelGroup, StepStatus
from src.schemas.analyst import (
    TaskIntelligenceReport,
    SubTask,
    MissingInfo,
    SeverityLevel,
    ModalityType,
)


@pytest.fixture
def planner():
    """Create a PlannerAgent with no system prompt file."""
    return PlannerAgent(system_prompt_path="nonexistent.md")


@pytest.fixture
def basic_report():
    """Create a basic TaskIntelligenceReport for testing."""
    return TaskIntelligenceReport(
        literal_request="Write a Python function",
        inferred_intent="User wants to create code",
        sub_tasks=[
            SubTask(description="Understand requirements", dependencies=[], estimated_complexity="low"),
            SubTask(description="Implement solution", dependencies=["Understand requirements"], estimated_complexity="high"),
        ],
        missing_info=[],
        assumptions=["Using Python"],
        modality=ModalityType.CODE,
        recommended_approach="Standard approach",
        escalation_needed=False,
        suggested_tier=2,
        confidence=0.9,
    )


@pytest.fixture
def report_with_missing_info():
    """Create a report with critical missing info."""
    return TaskIntelligenceReport(
        literal_request="Build an API",
        inferred_intent="Create API endpoints",
        sub_tasks=[
            SubTask(description="Design API endpoints", dependencies=[], estimated_complexity="medium"),
        ],
        missing_info=[
            MissingInfo(
                requirement="Authentication method",
                severity=SeverityLevel.CRITICAL,
                impact="Security depends on auth choice",
                default_assumption="JWT",
            ),
        ],
        assumptions=["JWT auth"],
        modality=ModalityType.CODE,
        recommended_approach="Clarify first",
        escalation_needed=False,
        suggested_tier=2,
        confidence=0.7,
    )


class TestPlannerInitialization:
    """Tests for PlannerAgent initialization."""

    def test_default_initialization(self):
        """Test default init parameters."""
        agent = PlannerAgent(system_prompt_path="nonexistent.md")
        assert agent.model == "claude-sonnet-4-20250514"
        assert agent.max_turns == 30

    def test_custom_initialization(self):
        """Test custom init parameters."""
        agent = PlannerAgent(
            system_prompt_path="custom.md",
            model="claude-3-opus",
            max_turns=50,
        )
        assert agent.model == "claude-3-opus"
        assert agent.max_turns == 50

    def test_agent_capabilities_initialized(self):
        """Test agent capabilities mapping exists."""
        agent = PlannerAgent(system_prompt_path="nonexistent.md")
        assert "Executor" in agent.agent_capabilities
        assert "Verifier" in agent.agent_capabilities

    def test_system_prompt_fallback(self):
        """Test fallback prompt when file not found."""
        agent = PlannerAgent(system_prompt_path="nonexistent.md")
        assert "Planner" in agent.system_prompt

    def test_system_prompt_from_file(self):
        """Test loading system prompt from file."""
        with patch("builtins.open", mock_open(read_data="Custom prompt")):
            agent = PlannerAgent(system_prompt_path="exists.md")
            assert agent.system_prompt == "Custom prompt"


class TestPlanCreation:
    """Tests for execution plan creation."""

    def test_creates_execution_plan(self, planner, basic_report):
        """Test that create_plan produces a valid ExecutionPlan."""
        plan = planner.create_plan(basic_report)
        assert isinstance(plan, ExecutionPlan)
        assert plan.total_steps > 0
        assert len(plan.steps) == plan.total_steps

    def test_plan_includes_review_steps(self, planner, basic_report):
        """Test that review steps are added (Verifier, Critic, Reviewer, Formatter)."""
        plan = planner.create_plan(basic_report)
        agent_names = []
        for step in plan.steps:
            for assignment in step.agent_assignments:
                agent_names.append(assignment.agent_name)
        assert "Verifier" in agent_names
        assert "Reviewer" in agent_names
        assert "Formatter" in agent_names

    def test_plan_with_clarification_step(self, planner, report_with_missing_info):
        """Test that clarification step is added for critical missing info."""
        plan = planner.create_plan(report_with_missing_info)
        descriptions = [s.description for s in plan.steps]
        assert any("clarif" in d.lower() for d in descriptions)

    def test_plan_has_critical_path(self, planner, basic_report):
        """Test critical path is calculated."""
        plan = planner.create_plan(basic_report)
        assert len(plan.critical_path) > 0

    def test_plan_has_duration_estimate(self, planner, basic_report):
        """Test duration estimate is provided."""
        plan = planner.create_plan(basic_report)
        assert plan.estimated_duration_minutes is not None
        assert plan.estimated_duration_minutes > 0

    def test_plan_has_task_summary(self, planner, basic_report):
        """Test plan summary is generated."""
        plan = planner.create_plan(basic_report)
        assert len(plan.task_summary) > 0


class TestStepGeneration:
    """Tests for step generation and agent assignment."""

    def test_code_review_for_code_modality(self, planner, basic_report):
        """Test Code Reviewer step is added for code modality."""
        plan = planner.create_plan(basic_report)
        agent_names = []
        for step in plan.steps:
            for a in step.agent_assignments:
                agent_names.append(a.agent_name)
        assert "Code Reviewer" in agent_names

    def test_executor_assigned_to_implementation(self, planner, basic_report):
        """Test Executor is assigned for implementation tasks."""
        plan = planner.create_plan(basic_report)
        executor_steps = [
            s for s in plan.steps
            if any(a.agent_name == "Executor" for a in s.agent_assignments)
        ]
        assert len(executor_steps) > 0

    def test_steps_have_dependencies(self, planner, basic_report):
        """Test that later steps have dependencies on earlier ones."""
        plan = planner.create_plan(basic_report)
        # At least some steps should have dependencies
        steps_with_deps = [s for s in plan.steps if s.dependencies]
        assert len(steps_with_deps) > 0

    def test_all_steps_have_status_pending(self, planner, basic_report):
        """Test all steps start with PENDING status."""
        plan = planner.create_plan(basic_report)
        for step in plan.steps:
            assert step.status == StepStatus.PENDING

    def test_steps_numbered_sequentially(self, planner, basic_report):
        """Test steps are numbered starting from 1."""
        plan = planner.create_plan(basic_report)
        for i, step in enumerate(plan.steps, 1):
            assert step.step_number == i


class TestRiskIdentification:
    """Tests for risk identification and contingency plans."""

    def test_risk_from_critical_missing_info(self, planner, report_with_missing_info):
        """Test that critical missing info is flagged as a risk."""
        plan = planner.create_plan(report_with_missing_info)
        assert len(plan.risk_factors) > 0
        assert any("missing" in r.lower() for r in plan.risk_factors)

    def test_contingency_plans_generated(self, planner, report_with_missing_info):
        """Test contingency plans are generated for risks."""
        plan = planner.create_plan(report_with_missing_info)
        assert len(plan.contingency_plans) > 0

    def test_no_risk_for_simple_task(self, planner, basic_report):
        """Test simple tasks may have no specific risks."""
        plan = planner.create_plan(basic_report)
        # Contingency plans always have at least a default entry
        assert len(plan.contingency_plans) >= 1


class TestSMEDetermination:
    """Tests for SME determination."""

    def test_sme_selection_from_report(self, planner):
        """Test SME detection from task content."""
        report = TaskIntelligenceReport(
            literal_request="Deploy to AWS cloud with kubernetes",
            inferred_intent="Deploy application",
            sub_tasks=[SubTask(description="Deploy", dependencies=[], estimated_complexity="high")],
            missing_info=[],
            assumptions=[],
            modality=ModalityType.TEXT,
            recommended_approach="Deploy",
            escalation_needed=False,
        )
        plan = planner.create_plan(report)
        assert any("Cloud" in sme or "DevOps" in sme for sme in plan.required_sme_personas)

    def test_explicit_sme_selections(self, planner, basic_report):
        """Test explicit SME selections are used."""
        plan = planner.create_plan(basic_report, sme_selections=["Security Analyst"])
        assert "Security Analyst" in plan.required_sme_personas


class TestConvenienceFunction:
    """Tests for the create_planner convenience function."""

    def test_create_planner(self):
        """Test convenience function creates a PlannerAgent."""
        agent = create_planner(system_prompt_path="nonexistent.md")
        assert isinstance(agent, PlannerAgent)
