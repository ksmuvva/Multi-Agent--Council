"""
Exhaustive Tests for PlannerAgent

Tests all methods, edge cases, and branch paths for the Planner subagent.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import patch, mock_open, MagicMock

from src.agents.planner import PlannerAgent, create_planner
from src.schemas.planner import (
    ExecutionPlan,
    ExecutionStep,
    AgentAssignment,
    ParallelGroup,
    StepStatus,
)
from src.schemas.analyst import (
    TaskIntelligenceReport,
    SubTask,
    MissingInfo,
    SeverityLevel,
    ModalityType,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_system_prompt():
    """Provide a mock system prompt for tests."""
    return "You are the Planner. Create sequenced execution plans."


@pytest.fixture
def planner(mock_system_prompt):
    """Create a PlannerAgent with mocked file I/O."""
    with patch("builtins.open", mock_open(read_data=mock_system_prompt)):
        return PlannerAgent()


@pytest.fixture
def simple_report():
    """A simple TaskIntelligenceReport with no missing info."""
    return TaskIntelligenceReport(
        literal_request="Create a hello world script",
        inferred_intent="Generate a simple Python script",
        sub_tasks=[
            SubTask(description="Implement the script", dependencies=[], estimated_complexity="low"),
        ],
        missing_info=[],
        assumptions=["Using Python"],
        modality=ModalityType.CODE,
        recommended_approach="Direct implementation",
        escalation_needed=False,
        suggested_tier=1,
        confidence=0.95,
    )


@pytest.fixture
def complex_report():
    """A complex TaskIntelligenceReport with missing info and multiple sub-tasks."""
    return TaskIntelligenceReport(
        literal_request="Build a cloud-deployed REST API with authentication",
        inferred_intent="Create a production REST API",
        sub_tasks=[
            SubTask(description="Design the API architecture", dependencies=[], estimated_complexity="high"),
            SubTask(description="Implement endpoints", dependencies=["Design the API architecture"], estimated_complexity="high"),
            SubTask(description="Create tests for the API", dependencies=["Implement endpoints"], estimated_complexity="medium"),
            SubTask(description="Analyze performance requirements", dependencies=[], estimated_complexity="medium"),
        ],
        missing_info=[
            MissingInfo(
                requirement="authentication method",
                severity=SeverityLevel.CRITICAL,
                impact="Cannot design security layer",
                default_assumption="JWT",
            ),
            MissingInfo(
                requirement="database technology",
                severity=SeverityLevel.IMPORTANT,
                impact="Affects schema design",
                default_assumption="PostgreSQL",
            ),
        ],
        assumptions=["Python/FastAPI", "Docker deployment"],
        modality=ModalityType.CODE,
        recommended_approach="Start with design, then implement",
        escalation_needed=True,
        suggested_tier=3,
        confidence=0.7,
    )


@pytest.fixture
def text_report():
    """A text-modality TaskIntelligenceReport."""
    return TaskIntelligenceReport(
        literal_request="Write a technical document about microservices",
        inferred_intent="Create technical documentation",
        sub_tasks=[
            SubTask(description="Generate the document outline", dependencies=[], estimated_complexity="low"),
        ],
        missing_info=[],
        assumptions=["Markdown format"],
        modality=ModalityType.TEXT,
        recommended_approach="Direct writing",
        escalation_needed=False,
        suggested_tier=1,
        confidence=0.9,
    )


# ============================================================================
# __init__ Tests
# ============================================================================

class TestPlannerInit:
    """Tests for PlannerAgent.__init__."""

    def test_default_params(self):
        """Test initialization with default parameters."""
        with patch("builtins.open", mock_open(read_data="prompt content")):
            agent = PlannerAgent()
        assert agent.system_prompt_path == "config/agents/planner/CLAUDE.md"
        assert agent.model == "claude-sonnet-4-20250514"
        assert agent.max_turns == 30
        assert agent.system_prompt == "prompt content"

    def test_custom_params(self):
        """Test initialization with custom parameters."""
        with patch("builtins.open", mock_open(read_data="custom prompt")):
            agent = PlannerAgent(
                system_prompt_path="custom/path.md",
                model="claude-3-opus-20240229",
                max_turns=10,
            )
        assert agent.system_prompt_path == "custom/path.md"
        assert agent.model == "claude-3-opus-20240229"
        assert agent.max_turns == 10
        assert agent.system_prompt == "custom prompt"

    def test_system_prompt_file_not_found(self):
        """Test fallback when system prompt file does not exist."""
        with patch("builtins.open", side_effect=FileNotFoundError):
            agent = PlannerAgent()
        assert "Planner" in agent.system_prompt
        assert "sequenced execution plans" in agent.system_prompt

    def test_agent_capabilities_populated(self):
        """Test that agent_capabilities dict is populated."""
        with patch("builtins.open", mock_open(read_data="")):
            agent = PlannerAgent()
        assert "Task Analyst" in agent.agent_capabilities
        assert "Executor" in agent.agent_capabilities
        assert "Code Reviewer" in agent.agent_capabilities
        assert "Formatter" in agent.agent_capabilities
        assert "Verifier" in agent.agent_capabilities
        assert "Critic" in agent.agent_capabilities
        assert "Reviewer" in agent.agent_capabilities
        assert "Planner" in agent.agent_capabilities
        assert "Clarifier" in agent.agent_capabilities
        assert "Researcher" in agent.agent_capabilities
        assert len(agent.agent_capabilities) == 10


# ============================================================================
# create_plan() Tests
# ============================================================================

class TestCreatePlan:
    """Tests for PlannerAgent.create_plan."""

    def test_simple_plan(self, planner, simple_report):
        """Test creating a plan from a simple report."""
        plan = planner.create_plan(simple_report)
        assert isinstance(plan, ExecutionPlan)
        assert plan.total_steps == len(plan.steps)
        assert plan.total_steps > 0
        assert plan.task_summary
        assert plan.critical_path

    def test_complex_plan(self, planner, complex_report):
        """Test creating a plan from a complex report with missing info."""
        plan = planner.create_plan(complex_report)
        assert isinstance(plan, ExecutionPlan)
        # Should have clarification step + sub-task steps + review steps
        assert plan.total_steps > 4
        assert len(plan.risk_factors) > 0
        assert len(plan.contingency_plans) > 0

    def test_plan_with_sme_selections(self, planner, simple_report):
        """Test creating a plan with explicit SME selections."""
        smes = ["Cloud Architect", "Security Analyst"]
        plan = planner.create_plan(simple_report, sme_selections=smes)
        assert plan.required_sme_personas == smes

    def test_plan_with_context(self, planner, simple_report):
        """Test creating a plan with additional context."""
        context = {"tier": 2, "previous_plans": []}
        plan = planner.create_plan(simple_report, context=context)
        assert isinstance(plan, ExecutionPlan)

    def test_plan_estimated_duration(self, planner, simple_report):
        """Test that estimated_duration_minutes is set."""
        plan = planner.create_plan(simple_report)
        assert plan.estimated_duration_minutes is not None
        assert plan.estimated_duration_minutes >= 1

    def test_plan_without_sme_selections_auto_detects(self, planner, complex_report):
        """Test auto-detection of SMEs when none provided."""
        plan = planner.create_plan(complex_report)
        # complex_report literal_request contains "cloud" and "authentication"
        assert isinstance(plan.required_sme_personas, list)

    def test_text_modality_plan(self, planner, text_report):
        """Test plan for text modality (no code review step)."""
        plan = planner.create_plan(text_report)
        code_review_steps = [
            s for s in plan.steps
            if "Code Review" in s.description
        ]
        assert len(code_review_steps) == 0


# ============================================================================
# _generate_steps() Tests
# ============================================================================

class TestGenerateSteps:
    """Tests for PlannerAgent._generate_steps."""

    def test_no_missing_info_no_clarification_step(self, planner, simple_report):
        """Test that no clarification step is added without missing info."""
        steps = planner._generate_steps(simple_report)
        clarification_steps = [s for s in steps if "Clarifier" in str(s.agent_assignments)]
        assert len(clarification_steps) == 0

    def test_critical_missing_info_adds_clarification(self, planner, complex_report):
        """Test that critical missing info triggers a clarification step."""
        steps = planner._generate_steps(complex_report)
        first_step = steps[0]
        assert "Clarify" in first_step.description
        assert first_step.agent_assignments[0].agent_name == "Clarifier"
        assert first_step.can_parallelize is False
        assert first_step.estimated_complexity == "low"

    def test_non_critical_missing_info_no_clarification(self, planner):
        """Test that only important/nice_to_have missing info skips clarification."""
        report = TaskIntelligenceReport(
            literal_request="Build something",
            inferred_intent="Build it",
            sub_tasks=[SubTask(description="Do the thing", dependencies=[], estimated_complexity="low")],
            missing_info=[
                MissingInfo(
                    requirement="color scheme",
                    severity=SeverityLevel.NICE_TO_HAVE,
                    impact="Aesthetic only",
                    default_assumption="Blue",
                ),
            ],
            assumptions=[],
            modality=ModalityType.TEXT,
            recommended_approach="Direct",
            escalation_needed=False,
            suggested_tier=1,
            confidence=0.9,
        )
        steps = planner._generate_steps(report)
        clarification_steps = [s for s in steps if "Clarify" in s.description]
        assert len(clarification_steps) == 0

    def test_sub_tasks_become_steps(self, planner, complex_report):
        """Test that each sub-task becomes an execution step."""
        steps = planner._generate_steps(complex_report)
        # Filter out clarification and review steps
        sub_task_descriptions = {st.description for st in complex_report.sub_tasks}
        matching = [s for s in steps if s.description in sub_task_descriptions]
        assert len(matching) == len(complex_report.sub_tasks)

    def test_review_steps_appended(self, planner, simple_report):
        """Test that review steps are always appended."""
        steps = planner._generate_steps(simple_report)
        # Should have at least: 1 sub-task + verification + critic + reviewer + formatter
        assert len(steps) >= 5

    def test_step_numbers_sequential(self, planner, complex_report):
        """Test that step numbers are sequential starting from 1."""
        steps = planner._generate_steps(complex_report)
        for i, step in enumerate(steps):
            assert step.step_number == i + 1

    def test_dependencies_populated(self, planner, complex_report):
        """Test that dependencies are populated for sub-task steps."""
        steps = planner._generate_steps(complex_report)
        # After the first step, subsequent steps should have dependencies
        for step in steps[1:]:
            assert len(step.dependencies) > 0 or step.can_parallelize

    def test_all_steps_pending_status(self, planner, simple_report):
        """Test that all generated steps have PENDING status."""
        steps = planner._generate_steps(simple_report)
        for step in steps:
            assert step.status == StepStatus.PENDING


# ============================================================================
# _assign_agents_to_task() Tests
# ============================================================================

class TestAssignAgentsToTask:
    """Tests for PlannerAgent._assign_agents_to_task."""

    @pytest.mark.parametrize("description,expected_agent", [
        ("Implement the REST API", "Executor"),
        ("Generate test data", "Executor"),
        ("Create the deployment script", "Executor"),
    ])
    def test_implement_generate_create_assigns_executor(self, planner, description, expected_agent):
        """Test that implement/generate/create keywords assign Executor."""
        sub_task = SubTask(description=description, dependencies=[], estimated_complexity="medium")
        assignments = planner._assign_agents_to_task(sub_task)
        assert len(assignments) == 1
        assert assignments[0].agent_name == expected_agent
        assert assignments[0].role == "Generate solution"

    @pytest.mark.parametrize("description", [
        "Analyze the requirements",
        "Understand the existing codebase",
    ])
    def test_analyze_understand_assigns_analyst(self, planner, description):
        """Test that analyze/understand keywords assign Task Analyst."""
        sub_task = SubTask(description=description, dependencies=[], estimated_complexity="medium")
        assignments = planner._assign_agents_to_task(sub_task)
        assert len(assignments) == 1
        assert assignments[0].agent_name == "Task Analyst"
        assert assignments[0].role == "Deep analysis"

    def test_design_without_sme(self, planner):
        """Test that design keyword assigns Planner when no architect SME."""
        sub_task = SubTask(description="Design the system", dependencies=[], estimated_complexity="medium")
        assignments = planner._assign_agents_to_task(sub_task)
        assert len(assignments) == 1
        assert assignments[0].agent_name == "Planner"
        assert assignments[0].role == "Create design"

    def test_design_with_architect_sme(self, planner):
        """Test that design keyword adds architect SME when available."""
        sub_task = SubTask(description="Design the system", dependencies=[], estimated_complexity="medium")
        smes = ["Cloud Architect"]
        assignments = planner._assign_agents_to_task(sub_task, sme_selections=smes)
        assert len(assignments) == 2
        assert assignments[0].agent_name == "Cloud Architect SME"
        assert assignments[0].role == "Domain-specific design"
        assert assignments[1].agent_name == "Planner"

    def test_design_with_non_architect_sme(self, planner):
        """Test that design keyword does not add non-architect SME."""
        sub_task = SubTask(description="Design the system", dependencies=[], estimated_complexity="medium")
        smes = ["Security Analyst"]
        assignments = planner._assign_agents_to_task(sub_task, sme_selections=smes)
        assert len(assignments) == 1
        assert assignments[0].agent_name == "Planner"

    def test_test_keyword_assigns_test_engineer_and_executor(self, planner):
        """Test that 'test' keyword assigns both Test Engineer SME and Executor."""
        sub_task = SubTask(description="Test the API endpoints", dependencies=[], estimated_complexity="medium")
        assignments = planner._assign_agents_to_task(sub_task)
        assert len(assignments) == 2
        assert assignments[0].agent_name == "Test Engineer SME"
        assert assignments[0].role == "Test strategy"
        assert assignments[1].agent_name == "Executor"
        assert assignments[1].role == "Generate tests"

    def test_default_assigns_executor(self, planner):
        """Test that unrecognized description defaults to Executor."""
        sub_task = SubTask(description="Do something unusual", dependencies=[], estimated_complexity="medium")
        assignments = planner._assign_agents_to_task(sub_task)
        assert len(assignments) == 1
        assert assignments[0].agent_name == "Executor"
        assert assignments[0].role == "Execute task"
        assert assignments[0].reason == "Default execution agent"

    def test_case_insensitive_matching(self, planner):
        """Test that keyword matching is case-insensitive."""
        sub_task = SubTask(description="IMPLEMENT the solution", dependencies=[], estimated_complexity="medium")
        assignments = planner._assign_agents_to_task(sub_task)
        assert assignments[0].agent_name == "Executor"


# ============================================================================
# _can_parallelize() Tests
# ============================================================================

class TestCanParallelize:
    """Tests for PlannerAgent._can_parallelize."""

    def test_no_dependencies_no_existing_independent(self, planner):
        """Test task with no deps and no existing independent steps cannot parallelize."""
        sub_task = SubTask(description="First task", dependencies=[], estimated_complexity="low")
        existing = []
        assert planner._can_parallelize(sub_task, existing) is False

    def test_no_dependencies_with_existing_independent(self, planner):
        """Test task with no deps and existing independent steps can parallelize."""
        sub_task = SubTask(description="Second task", dependencies=[], estimated_complexity="low")
        existing = [
            ExecutionStep(
                step_number=1,
                description="First task",
                agent_assignments=[AgentAssignment(agent_name="Executor", role="r", reason="r")],
                dependencies=[],
                can_parallelize=False,
                estimated_complexity="low",
                expected_outputs=["out"],
                status=StepStatus.PENDING,
            )
        ]
        assert planner._can_parallelize(sub_task, existing) is True

    def test_has_dependencies_cannot_parallelize(self, planner):
        """Test task with dependencies cannot parallelize."""
        sub_task = SubTask(description="Dependent task", dependencies=["other"], estimated_complexity="low")
        existing = [
            ExecutionStep(
                step_number=1,
                description="First",
                agent_assignments=[AgentAssignment(agent_name="Executor", role="r", reason="r")],
                dependencies=[],
                can_parallelize=False,
                estimated_complexity="low",
                expected_outputs=["out"],
                status=StepStatus.PENDING,
            )
        ]
        assert planner._can_parallelize(sub_task, existing) is False

    def test_no_deps_with_multiple_existing_independent(self, planner):
        """Test task with no deps and multiple existing independent steps."""
        sub_task = SubTask(description="Third task", dependencies=[], estimated_complexity="low")
        existing = [
            ExecutionStep(
                step_number=1, description="First",
                agent_assignments=[AgentAssignment(agent_name="Executor", role="r", reason="r")],
                dependencies=[], can_parallelize=False,
                estimated_complexity="low", expected_outputs=["out"], status=StepStatus.PENDING,
            ),
            ExecutionStep(
                step_number=2, description="Second",
                agent_assignments=[AgentAssignment(agent_name="Executor", role="r", reason="r")],
                dependencies=[], can_parallelize=False,
                estimated_complexity="low", expected_outputs=["out"], status=StepStatus.PENDING,
            ),
        ]
        assert planner._can_parallelize(sub_task, existing) is True


# ============================================================================
# _add_review_steps() Tests
# ============================================================================

class TestAddReviewSteps:
    """Tests for PlannerAgent._add_review_steps."""

    def test_code_modality_includes_code_review(self, planner, simple_report):
        """Test that code modality adds a Code Review step."""
        steps = planner._add_review_steps(simple_report, start_step=2)
        code_review = [s for s in steps if "Code Review" in s.description]
        assert len(code_review) == 1
        assert code_review[0].agent_assignments[0].agent_name == "Code Reviewer"

    def test_text_modality_no_code_review(self, planner, text_report):
        """Test that text modality does not add a Code Review step."""
        steps = planner._add_review_steps(text_report, start_step=2)
        code_review = [s for s in steps if "Code Review" in s.description]
        assert len(code_review) == 0

    def test_always_includes_verifier(self, planner, text_report):
        """Test that Verifier step is always included."""
        steps = planner._add_review_steps(text_report, start_step=2)
        verifier = [s for s in steps if "Verifier" in str(s.agent_assignments)]
        assert len(verifier) == 1

    def test_always_includes_critic(self, planner, text_report):
        """Test that Critic step is always included."""
        steps = planner._add_review_steps(text_report, start_step=2)
        critic = [s for s in steps if "Critic" in str(s.agent_assignments)]
        assert len(critic) == 1

    def test_always_includes_reviewer(self, planner, text_report):
        """Test that Reviewer step is always included."""
        steps = planner._add_review_steps(text_report, start_step=2)
        reviewer = [s for s in steps if s.agent_assignments[0].agent_name == "Reviewer"]
        assert len(reviewer) == 1

    def test_always_includes_formatter(self, planner, text_report):
        """Test that Formatter step is always last."""
        steps = planner._add_review_steps(text_report, start_step=2)
        last_step = steps[-1]
        assert last_step.agent_assignments[0].agent_name == "Formatter"

    def test_code_review_steps_sequential_numbering(self, planner, simple_report):
        """Test sequential numbering for code review steps."""
        steps = planner._add_review_steps(simple_report, start_step=5)
        for i, step in enumerate(steps):
            assert step.step_number == 5 + i

    def test_critic_parallel_group_id(self, planner, text_report):
        """Test that Critic step has a parallel_group_id."""
        steps = planner._add_review_steps(text_report, start_step=2)
        critic = [s for s in steps if "Critic" in str(s.agent_assignments)]
        assert critic[0].parallel_group_id == "review_group"
        assert critic[0].can_parallelize is True

    def test_non_code_has_4_review_steps(self, planner, text_report):
        """Non-code modality has 4 review steps: Verifier, Critic, Reviewer, Formatter."""
        steps = planner._add_review_steps(text_report, start_step=1)
        assert len(steps) == 4

    def test_code_has_5_review_steps(self, planner, simple_report):
        """Code modality has 5 review steps: Code Review, Verifier, Critic, Reviewer, Formatter."""
        steps = planner._add_review_steps(simple_report, start_step=1)
        assert len(steps) == 5

    def test_start_step_1_no_dependencies(self, planner, text_report):
        """Test dependencies when start_step is 1."""
        steps = planner._add_review_steps(text_report, start_step=1)
        # First step at step_number=1 should have dependencies=[] since step_number > 1 is False
        first = steps[0]
        assert first.dependencies == []


# ============================================================================
# _identify_parallel_groups() Tests
# ============================================================================

class TestIdentifyParallelGroups:
    """Tests for PlannerAgent._identify_parallel_groups."""

    def test_no_parallel_steps(self, planner):
        """Test with no parallel steps returns empty list."""
        steps = [
            ExecutionStep(
                step_number=1, description="Step 1",
                agent_assignments=[AgentAssignment(agent_name="Executor", role="r", reason="r")],
                dependencies=[], can_parallelize=False,
                estimated_complexity="low", expected_outputs=["out"], status=StepStatus.PENDING,
            )
        ]
        groups = planner._identify_parallel_groups(steps)
        assert groups == []

    def test_parallel_steps_with_group_id(self, planner):
        """Test with parallel steps having group IDs."""
        steps = [
            ExecutionStep(
                step_number=1, description="Verify",
                agent_assignments=[AgentAssignment(agent_name="Verifier", role="r", reason="r")],
                dependencies=[], can_parallelize=True, parallel_group_id="review_group",
                estimated_complexity="medium", expected_outputs=["out"], status=StepStatus.PENDING,
            ),
            ExecutionStep(
                step_number=2, description="Critique",
                agent_assignments=[AgentAssignment(agent_name="Critic", role="r", reason="r")],
                dependencies=[], can_parallelize=True, parallel_group_id="review_group",
                estimated_complexity="medium", expected_outputs=["out"], status=StepStatus.PENDING,
            ),
        ]
        groups = planner._identify_parallel_groups(steps)
        assert len(groups) == 1
        assert groups[0].group_id == "review_group"
        assert groups[0].steps == [1, 2]

    def test_parallel_without_group_id_ignored(self, planner):
        """Test that parallel steps without group_id are ignored."""
        steps = [
            ExecutionStep(
                step_number=1, description="Step 1",
                agent_assignments=[AgentAssignment(agent_name="Executor", role="r", reason="r")],
                dependencies=[], can_parallelize=True, parallel_group_id=None,
                estimated_complexity="low", expected_outputs=["out"], status=StepStatus.PENDING,
            ),
        ]
        groups = planner._identify_parallel_groups(steps)
        assert groups == []

    def test_multiple_groups(self, planner):
        """Test with multiple parallel groups."""
        steps = [
            ExecutionStep(
                step_number=1, description="A",
                agent_assignments=[AgentAssignment(agent_name="Executor", role="r", reason="r")],
                dependencies=[], can_parallelize=True, parallel_group_id="group_a",
                estimated_complexity="low", expected_outputs=["out"], status=StepStatus.PENDING,
            ),
            ExecutionStep(
                step_number=2, description="B",
                agent_assignments=[AgentAssignment(agent_name="Executor", role="r", reason="r")],
                dependencies=[], can_parallelize=True, parallel_group_id="group_a",
                estimated_complexity="low", expected_outputs=["out"], status=StepStatus.PENDING,
            ),
            ExecutionStep(
                step_number=3, description="C",
                agent_assignments=[AgentAssignment(agent_name="Executor", role="r", reason="r")],
                dependencies=[], can_parallelize=True, parallel_group_id="group_b",
                estimated_complexity="low", expected_outputs=["out"], status=StepStatus.PENDING,
            ),
        ]
        groups = planner._identify_parallel_groups(steps)
        assert len(groups) == 2
        group_ids = {g.group_id for g in groups}
        assert group_ids == {"group_a", "group_b"}


# ============================================================================
# _calculate_critical_path() Tests
# ============================================================================

class TestCalculateCriticalPath:
    """Tests for PlannerAgent._calculate_critical_path."""

    def test_empty_steps(self, planner):
        """Test with empty steps returns empty list."""
        assert planner._calculate_critical_path([]) == []

    def test_single_step(self, planner):
        """Test with a single step returns that step."""
        steps = [
            ExecutionStep(
                step_number=1, description="Only step",
                agent_assignments=[AgentAssignment(agent_name="Executor", role="r", reason="r")],
                dependencies=[], estimated_complexity="low",
                expected_outputs=["out"], status=StepStatus.PENDING,
            ),
        ]
        result = planner._calculate_critical_path(steps)
        assert result == [1]

    def test_linear_chain(self, planner):
        """Test with linear chain of dependencies."""
        steps = [
            ExecutionStep(
                step_number=1, description="Step 1",
                agent_assignments=[AgentAssignment(agent_name="E", role="r", reason="r")],
                dependencies=[], estimated_complexity="low",
                expected_outputs=["out"], status=StepStatus.PENDING,
            ),
            ExecutionStep(
                step_number=2, description="Step 2",
                agent_assignments=[AgentAssignment(agent_name="E", role="r", reason="r")],
                dependencies=[1], estimated_complexity="low",
                expected_outputs=["out"], status=StepStatus.PENDING,
            ),
            ExecutionStep(
                step_number=3, description="Step 3",
                agent_assignments=[AgentAssignment(agent_name="E", role="r", reason="r")],
                dependencies=[2], estimated_complexity="low",
                expected_outputs=["out"], status=StepStatus.PENDING,
            ),
        ]
        result = planner._calculate_critical_path(steps)
        # Step 3 has longest path (3), so it should be on critical path
        assert 3 in result

    def test_parallel_paths(self, planner):
        """Test with parallel paths of different lengths."""
        steps = [
            ExecutionStep(
                step_number=1, description="Start",
                agent_assignments=[AgentAssignment(agent_name="E", role="r", reason="r")],
                dependencies=[], estimated_complexity="low",
                expected_outputs=["out"], status=StepStatus.PENDING,
            ),
            ExecutionStep(
                step_number=2, description="Short path",
                agent_assignments=[AgentAssignment(agent_name="E", role="r", reason="r")],
                dependencies=[1], estimated_complexity="low",
                expected_outputs=["out"], status=StepStatus.PENDING,
            ),
            ExecutionStep(
                step_number=3, description="Long path A",
                agent_assignments=[AgentAssignment(agent_name="E", role="r", reason="r")],
                dependencies=[1], estimated_complexity="low",
                expected_outputs=["out"], status=StepStatus.PENDING,
            ),
            ExecutionStep(
                step_number=4, description="Long path B",
                agent_assignments=[AgentAssignment(agent_name="E", role="r", reason="r")],
                dependencies=[3], estimated_complexity="low",
                expected_outputs=["out"], status=StepStatus.PENDING,
            ),
        ]
        result = planner._calculate_critical_path(steps)
        # Step 4 has path length 3, step 2 has path length 2
        assert 4 in result


# ============================================================================
# _visit_step() Tests
# ============================================================================

class TestVisitStep:
    """Tests for PlannerAgent._visit_step."""

    def test_already_visited(self, planner):
        """Test that already-visited steps return cached length."""
        step_map = {
            1: ExecutionStep(
                step_number=1, description="Step 1",
                agent_assignments=[AgentAssignment(agent_name="E", role="r", reason="r")],
                dependencies=[], estimated_complexity="low",
                expected_outputs=["out"], status=StepStatus.PENDING,
            ),
        }
        visited = {1}
        path_lengths = {1: 5}
        result = planner._visit_step(1, step_map, visited, path_lengths)
        assert result == 5

    def test_step_not_in_map(self, planner):
        """Test that missing step returns 0."""
        result = planner._visit_step(99, {}, set(), {})
        assert result == 0

    def test_step_with_dependencies(self, planner):
        """Test recursive path calculation with dependencies."""
        step_map = {
            1: ExecutionStep(
                step_number=1, description="Step 1",
                agent_assignments=[AgentAssignment(agent_name="E", role="r", reason="r")],
                dependencies=[], estimated_complexity="low",
                expected_outputs=["out"], status=StepStatus.PENDING,
            ),
            2: ExecutionStep(
                step_number=2, description="Step 2",
                agent_assignments=[AgentAssignment(agent_name="E", role="r", reason="r")],
                dependencies=[1], estimated_complexity="low",
                expected_outputs=["out"], status=StepStatus.PENDING,
            ),
        }
        visited = set()
        path_lengths = {1: 0, 2: 0}
        result = planner._visit_step(2, step_map, visited, path_lengths)
        assert result == 2
        assert path_lengths[1] == 1
        assert path_lengths[2] == 2

    def test_step_with_multiple_dependencies_takes_max(self, planner):
        """Test that the max dependency path is chosen."""
        step_map = {
            1: ExecutionStep(
                step_number=1, description="Short",
                agent_assignments=[AgentAssignment(agent_name="E", role="r", reason="r")],
                dependencies=[], estimated_complexity="low",
                expected_outputs=["out"], status=StepStatus.PENDING,
            ),
            2: ExecutionStep(
                step_number=2, description="Long chain",
                agent_assignments=[AgentAssignment(agent_name="E", role="r", reason="r")],
                dependencies=[1], estimated_complexity="low",
                expected_outputs=["out"], status=StepStatus.PENDING,
            ),
            3: ExecutionStep(
                step_number=3, description="Merges",
                agent_assignments=[AgentAssignment(agent_name="E", role="r", reason="r")],
                dependencies=[1, 2], estimated_complexity="low",
                expected_outputs=["out"], status=StepStatus.PENDING,
            ),
        }
        visited = set()
        path_lengths = {1: 0, 2: 0, 3: 0}
        result = planner._visit_step(3, step_map, visited, path_lengths)
        # Step 1: 1, Step 2: 2, Step 3: max(1,2)+1 = 3
        assert result == 3


# ============================================================================
# _estimate_duration() Tests
# ============================================================================

class TestEstimateDuration:
    """Tests for PlannerAgent._estimate_duration."""

    def test_single_low_complexity(self, planner):
        """Test duration for a single low-complexity step."""
        steps = [
            ExecutionStep(
                step_number=1, description="Step",
                agent_assignments=[AgentAssignment(agent_name="E", role="r", reason="r")],
                dependencies=[], estimated_complexity="low",
                expected_outputs=["out"], status=StepStatus.PENDING,
            ),
        ]
        duration = planner._estimate_duration(steps, [])
        assert duration == 2

    def test_mixed_complexity(self, planner):
        """Test duration for steps with mixed complexity."""
        steps = [
            ExecutionStep(
                step_number=1, description="Low",
                agent_assignments=[AgentAssignment(agent_name="E", role="r", reason="r")],
                dependencies=[], estimated_complexity="low",
                expected_outputs=["out"], status=StepStatus.PENDING,
            ),
            ExecutionStep(
                step_number=2, description="Medium",
                agent_assignments=[AgentAssignment(agent_name="E", role="r", reason="r")],
                dependencies=[], estimated_complexity="medium",
                expected_outputs=["out"], status=StepStatus.PENDING,
            ),
            ExecutionStep(
                step_number=3, description="High",
                agent_assignments=[AgentAssignment(agent_name="E", role="r", reason="r")],
                dependencies=[], estimated_complexity="high",
                expected_outputs=["out"], status=StepStatus.PENDING,
            ),
        ]
        duration = planner._estimate_duration(steps, [])
        assert duration == 2 + 5 + 10  # 17

    def test_unknown_complexity_defaults_to_5(self, planner):
        """Test that unknown complexity defaults to 5 minutes."""
        steps = [
            ExecutionStep(
                step_number=1, description="Step",
                agent_assignments=[AgentAssignment(agent_name="E", role="r", reason="r")],
                dependencies=[], estimated_complexity="extreme",
                expected_outputs=["out"], status=StepStatus.PENDING,
            ),
        ]
        duration = planner._estimate_duration(steps, [])
        assert duration == 5

    def test_parallel_savings(self, planner):
        """Test that parallel groups reduce estimated duration."""
        steps = [
            ExecutionStep(
                step_number=1, description="A",
                agent_assignments=[AgentAssignment(agent_name="E", role="r", reason="r")],
                dependencies=[], estimated_complexity="medium",
                expected_outputs=["out"], status=StepStatus.PENDING,
            ),
            ExecutionStep(
                step_number=2, description="B",
                agent_assignments=[AgentAssignment(agent_name="E", role="r", reason="r")],
                dependencies=[], estimated_complexity="medium",
                expected_outputs=["out"], status=StepStatus.PENDING,
            ),
        ]
        groups = [ParallelGroup(group_id="g1", steps=[1, 2], description="Parallel")]
        duration_without = planner._estimate_duration(steps, [])
        duration_with = planner._estimate_duration(steps, groups)
        assert duration_with < duration_without

    def test_minimum_duration_is_1(self, planner):
        """Test that minimum duration is capped at 1."""
        # Even with aggressive savings, minimum is 1
        steps = [
            ExecutionStep(
                step_number=1, description="A",
                agent_assignments=[AgentAssignment(agent_name="E", role="r", reason="r")],
                dependencies=[], estimated_complexity="low",
                expected_outputs=["out"], status=StepStatus.PENDING,
            ),
        ]
        duration = planner._estimate_duration(steps, [])
        assert duration >= 1

    def test_single_step_parallel_group_no_savings(self, planner):
        """Test that single-step parallel groups provide no savings."""
        steps = [
            ExecutionStep(
                step_number=1, description="A",
                agent_assignments=[AgentAssignment(agent_name="E", role="r", reason="r")],
                dependencies=[], estimated_complexity="medium",
                expected_outputs=["out"], status=StepStatus.PENDING,
            ),
        ]
        groups = [ParallelGroup(group_id="g1", steps=[1], description="Solo")]
        duration = planner._estimate_duration(steps, groups)
        assert duration == 5  # No savings for single-step group


# ============================================================================
# _determine_required_smes() Tests
# ============================================================================

class TestDetermineRequiredSmes:
    """Tests for PlannerAgent._determine_required_smes."""

    @pytest.mark.parametrize("keyword,expected_sme", [
        ("sailpoint integration", "IAM Architect"),
        ("cyberark vault setup", "IAM Architect"),
        ("identity management", "IAM Architect"),
        ("rbac system", "IAM Architect"),
        ("azure deployment", "Cloud Architect"),
        ("aws lambda function", "Cloud Architect"),
        ("cloud infrastructure", "Cloud Architect"),
        ("kubernetes cluster", "Cloud Architect"),
        ("security audit", "Security Analyst"),
        ("threat modeling", "Security Analyst"),
        ("vulnerability scan", "Security Analyst"),
        ("database schema design", "Data Engineer"),
        ("etl pipeline", "Data Engineer"),
        ("data pipeline", "Data Engineer"),
        ("ml model training", "AI/ML Engineer"),
        ("ai chatbot", "AI/ML Engineer"),
        ("rag pipeline", "AI/ML Engineer"),
        ("llm integration", "AI/ML Engineer"),
        ("test automation", "Test Engineer"),
        ("testing strategy", "Test Engineer"),
        ("qa plan", "Test Engineer"),
        ("requirements gathering", "Business Analyst"),
        ("process improvement", "Business Analyst"),
        ("workflow design", "Business Analyst"),
        ("document the API", "Technical Writer"),
        ("write docs", "Technical Writer"),
        ("readme file", "Technical Writer"),
        ("deploy to production", "DevOps Engineer"),
        ("ci/cd pipeline", "DevOps Engineer"),
        ("docker compose", "DevOps Engineer"),
        ("ui component", "Frontend Developer"),
        ("frontend dashboard", "Frontend Developer"),
        ("streamlit app", "Frontend Developer"),
    ])
    def test_keyword_triggers_sme(self, planner, keyword, expected_sme):
        """Test that specific keywords trigger the correct SME."""
        report = TaskIntelligenceReport(
            literal_request=keyword,
            inferred_intent="Do the thing",
            sub_tasks=[],
            missing_info=[],
            assumptions=[],
            modality=ModalityType.TEXT,
            recommended_approach="Direct",
            escalation_needed=False,
            suggested_tier=1,
            confidence=0.9,
        )
        smes = planner._determine_required_smes(report)
        assert expected_sme in smes

    def test_max_3_smes(self, planner):
        """Test that at most 3 SMEs are returned."""
        # Request that triggers many SMEs
        report = TaskIntelligenceReport(
            literal_request="Deploy security-tested cloud AI data pipeline with docs",
            inferred_intent="Everything",
            sub_tasks=[],
            missing_info=[],
            assumptions=[],
            modality=ModalityType.TEXT,
            recommended_approach="Direct",
            escalation_needed=False,
            suggested_tier=1,
            confidence=0.9,
        )
        smes = planner._determine_required_smes(report)
        assert len(smes) <= 3

    def test_no_smes_for_unrelated_request(self, planner):
        """Test that no SMEs are returned for requests without trigger keywords."""
        report = TaskIntelligenceReport(
            literal_request="Say hello world",
            inferred_intent="Simple greeting",
            sub_tasks=[],
            missing_info=[],
            assumptions=[],
            modality=ModalityType.TEXT,
            recommended_approach="Direct",
            escalation_needed=False,
            suggested_tier=1,
            confidence=0.9,
        )
        smes = planner._determine_required_smes(report)
        assert len(smes) == 0

    def test_case_insensitive_matching(self, planner):
        """Test that keyword matching is case-insensitive."""
        report = TaskIntelligenceReport(
            literal_request="AZURE CLOUD DEPLOYMENT",
            inferred_intent="Deploy",
            sub_tasks=[],
            missing_info=[],
            assumptions=[],
            modality=ModalityType.TEXT,
            recommended_approach="Direct",
            escalation_needed=False,
            suggested_tier=1,
            confidence=0.9,
        )
        smes = planner._determine_required_smes(report)
        assert "Cloud Architect" in smes


# ============================================================================
# _identify_risks() Tests
# ============================================================================

class TestIdentifyRisks:
    """Tests for PlannerAgent._identify_risks."""

    def test_critical_missing_info_risk(self, planner, complex_report):
        """Test that critical missing info adds a risk."""
        steps = planner._generate_steps(complex_report)
        risks = planner._identify_risks(complex_report, steps)
        assert any("Critical requirements missing" in r for r in risks)

    def test_no_critical_missing_no_risk(self, planner, simple_report):
        """Test that no critical missing info adds no missing info risk."""
        steps = planner._generate_steps(simple_report)
        risks = planner._identify_risks(simple_report, steps)
        assert not any("Critical requirements missing" in r for r in risks)

    def test_many_complex_steps_risk(self, planner):
        """Test that >3 high-complexity steps adds a risk."""
        report = TaskIntelligenceReport(
            literal_request="Complex task",
            inferred_intent="Do it",
            sub_tasks=[],
            missing_info=[],
            assumptions=[],
            modality=ModalityType.TEXT,
            recommended_approach="Direct",
            escalation_needed=False,
            suggested_tier=1,
            confidence=0.9,
        )
        steps = [
            ExecutionStep(
                step_number=i, description=f"Step {i}",
                agent_assignments=[AgentAssignment(agent_name="E", role="r", reason="r")],
                dependencies=[], estimated_complexity="high",
                expected_outputs=["out"], status=StepStatus.PENDING,
            )
            for i in range(1, 6)
        ]
        risks = planner._identify_risks(report, steps)
        assert any("high-complexity" in r for r in risks)

    def test_few_complex_steps_no_risk(self, planner, simple_report):
        """Test that <=3 high-complexity steps do not add complexity risk."""
        steps = [
            ExecutionStep(
                step_number=i, description=f"Step {i}",
                agent_assignments=[AgentAssignment(agent_name="E", role="r", reason="r")],
                dependencies=[], estimated_complexity="high",
                expected_outputs=["out"], status=StepStatus.PENDING,
            )
            for i in range(1, 4)
        ]
        risks = planner._identify_risks(simple_report, steps)
        assert not any("high-complexity" in r for r in risks)

    def test_escalation_needed_risk(self, planner, complex_report):
        """Test that escalation_needed adds a risk."""
        steps = planner._generate_steps(complex_report)
        risks = planner._identify_risks(complex_report, steps)
        assert any("escalation" in r for r in risks)

    def test_no_escalation_no_risk(self, planner, simple_report):
        """Test that no escalation does not add escalation risk."""
        steps = planner._generate_steps(simple_report)
        risks = planner._identify_risks(simple_report, steps)
        assert not any("escalation" in r for r in risks)


# ============================================================================
# _create_contingency_plans() Tests
# ============================================================================

class TestCreateContingencyPlans:
    """Tests for PlannerAgent._create_contingency_plans."""

    def test_missing_risk_contingency(self, planner):
        """Test contingency for missing requirements risk."""
        risks = ["Critical requirements missing: auth method"]
        plans = planner._create_contingency_plans(risks)
        assert any("assumptions" in p.lower() for p in plans)

    def test_complexity_risk_contingency(self, planner):
        """Test contingency for complexity risk."""
        risks = ["Multiple high-complexity steps may extend execution time"]
        plans = planner._create_contingency_plans(risks)
        assert any("splitting" in p.lower() or "complex" in p.lower() for p in plans)

    def test_escalation_risk_contingency(self, planner):
        """Test contingency for escalation risk."""
        risks = ["May require escalation to higher tier during execution"]
        plans = planner._create_contingency_plans(risks)
        assert any("Council" in p or "SME" in p for p in plans)

    def test_no_risks_default_contingency(self, planner):
        """Test default contingency when no risks exist."""
        plans = planner._create_contingency_plans([])
        assert len(plans) == 1
        assert "Standard execution" in plans[0]

    def test_multiple_risks_multiple_plans(self, planner):
        """Test multiple risks generate multiple contingency plans."""
        risks = [
            "Critical requirements missing: X",
            "Multiple high-complexity steps",
            "May require escalation",
        ]
        plans = planner._create_contingency_plans(risks)
        assert len(plans) == 3

    def test_unrecognized_risk_gets_default(self, planner):
        """Test that unrecognized risk patterns produce the default contingency."""
        risks = ["Something entirely unknown"]
        plans = planner._create_contingency_plans(risks)
        # No matching patterns in the loop, but the risk iterates without adding,
        # leaving plans empty, so the default plan is appended
        assert len(plans) == 1
        assert "Standard execution" in plans[0]


# ============================================================================
# _build_summary() Tests
# ============================================================================

class TestBuildSummary:
    """Tests for PlannerAgent._build_summary."""

    def test_summary_contains_intent(self, planner, simple_report):
        """Test that summary contains the inferred intent."""
        summary = planner._build_summary(simple_report, 5)
        assert simple_report.inferred_intent in summary

    def test_summary_contains_step_count(self, planner, simple_report):
        """Test that summary contains the step count."""
        summary = planner._build_summary(simple_report, 7)
        assert "7 steps" in summary

    def test_summary_contains_modality(self, planner, simple_report):
        """Test that summary contains the modality."""
        summary = planner._build_summary(simple_report, 5)
        assert simple_report.modality.value in summary

    def test_summary_contains_subtask_count(self, planner, simple_report):
        """Test that summary contains sub-task count."""
        summary = planner._build_summary(simple_report, 5)
        assert str(len(simple_report.sub_tasks)) in summary


# ============================================================================
# create_planner() Convenience Function Tests
# ============================================================================

class TestCreatePlannerFunction:
    """Tests for the create_planner() convenience function."""

    def test_default_params(self):
        """Test create_planner with default parameters."""
        with patch("builtins.open", mock_open(read_data="prompt")):
            agent = create_planner()
        assert isinstance(agent, PlannerAgent)
        assert agent.system_prompt_path == "config/agents/planner/CLAUDE.md"
        assert agent.model == "claude-sonnet-4-20250514"

    def test_custom_params(self):
        """Test create_planner with custom parameters."""
        with patch("builtins.open", mock_open(read_data="custom")):
            agent = create_planner(
                system_prompt_path="custom/path.md",
                model="custom-model",
            )
        assert agent.system_prompt_path == "custom/path.md"
        assert agent.model == "custom-model"

    def test_returns_planner_agent(self):
        """Test that create_planner returns a PlannerAgent instance."""
        with patch("builtins.open", mock_open(read_data="")):
            agent = create_planner()
        assert isinstance(agent, PlannerAgent)


# ============================================================================
# Schema Validation Tests
# ============================================================================

class TestSchemaIntegration:
    """Test that PlannerAgent output conforms to schemas."""

    def test_execution_plan_schema_valid(self, planner, simple_report):
        """Test that create_plan output is valid ExecutionPlan."""
        plan = planner.create_plan(simple_report)
        assert isinstance(plan, ExecutionPlan)
        # Verify it can be serialized
        data = plan.model_dump()
        assert "task_summary" in data
        assert "total_steps" in data
        assert "steps" in data

    def test_execution_step_schema_valid(self, planner, simple_report):
        """Test that generated steps are valid ExecutionStep instances."""
        steps = planner._generate_steps(simple_report)
        for step in steps:
            assert isinstance(step, ExecutionStep)
            assert step.step_number >= 1
            assert len(step.agent_assignments) > 0

    def test_agent_assignment_schema_valid(self, planner):
        """Test AgentAssignment schema validation."""
        sub_task = SubTask(description="Implement something", dependencies=[], estimated_complexity="low")
        assignments = planner._assign_agents_to_task(sub_task)
        for a in assignments:
            assert isinstance(a, AgentAssignment)
            assert a.agent_name
            assert a.role
            assert a.reason

    def test_plan_json_roundtrip(self, planner, simple_report):
        """Test that plan can be serialized and deserialized."""
        plan = planner.create_plan(simple_report)
        json_str = plan.model_dump_json()
        restored = ExecutionPlan.model_validate_json(json_str)
        assert restored.total_steps == plan.total_steps
        assert len(restored.steps) == len(plan.steps)


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_report_with_no_sub_tasks(self, planner):
        """Test plan with no sub-tasks (only review steps)."""
        report = TaskIntelligenceReport(
            literal_request="Simple request",
            inferred_intent="Do it",
            sub_tasks=[],
            missing_info=[],
            assumptions=[],
            modality=ModalityType.TEXT,
            recommended_approach="Direct",
            escalation_needed=False,
            suggested_tier=1,
            confidence=0.9,
        )
        plan = planner.create_plan(report)
        # Should still have review steps
        assert plan.total_steps > 0

    def test_report_with_many_missing_info_only_critical_counted(self, planner):
        """Test that only critical missing info triggers clarification step."""
        report = TaskIntelligenceReport(
            literal_request="Build something",
            inferred_intent="Build it",
            sub_tasks=[SubTask(description="Do work", dependencies=[], estimated_complexity="low")],
            missing_info=[
                MissingInfo(requirement="color", severity=SeverityLevel.NICE_TO_HAVE, impact="minor"),
                MissingInfo(requirement="size", severity=SeverityLevel.IMPORTANT, impact="moderate"),
                MissingInfo(requirement="API key", severity=SeverityLevel.CRITICAL, impact="cannot proceed"),
            ],
            assumptions=[],
            modality=ModalityType.TEXT,
            recommended_approach="Direct",
            escalation_needed=False,
            suggested_tier=1,
            confidence=0.9,
        )
        steps = planner._generate_steps(report)
        clarify_steps = [s for s in steps if "Clarify" in s.description]
        assert len(clarify_steps) == 1
        assert "1 critical" in clarify_steps[0].description

    def test_all_step_statuses_valid(self, planner, simple_report):
        """Test that all StepStatus enum values are valid."""
        assert StepStatus.PENDING == "pending"
        assert StepStatus.IN_PROGRESS == "in_progress"
        assert StepStatus.COMPLETE == "complete"
        assert StepStatus.SKIPPED == "skipped"

    def test_multiple_critical_missing_info(self, planner):
        """Test clarification step description with multiple critical items."""
        report = TaskIntelligenceReport(
            literal_request="Build API",
            inferred_intent="Build it",
            sub_tasks=[],
            missing_info=[
                MissingInfo(requirement="auth", severity=SeverityLevel.CRITICAL, impact="critical"),
                MissingInfo(requirement="db", severity=SeverityLevel.CRITICAL, impact="critical"),
                MissingInfo(requirement="deploy", severity=SeverityLevel.CRITICAL, impact="critical"),
            ],
            assumptions=[],
            modality=ModalityType.CODE,
            recommended_approach="Direct",
            escalation_needed=False,
            suggested_tier=1,
            confidence=0.9,
        )
        steps = planner._generate_steps(report)
        clarify = [s for s in steps if "Clarify" in s.description]
        assert len(clarify) == 1
        assert "3 critical" in clarify[0].description
