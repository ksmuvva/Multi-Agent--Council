"""
Tests for the ExecutorAgent.

Tests Tree of Thoughts approach generation, scoring, selection,
and task execution across different modalities.
"""

import pytest
from unittest.mock import patch, mock_open

from src.agents.executor import (
    ExecutorAgent,
    Approach,
    ExecutionResult,
    ThoughtBranch,
    create_executor,
)
from src.schemas.analyst import (
    TaskIntelligenceReport,
    SubTask,
    ModalityType,
)


@pytest.fixture
def executor():
    """Create an ExecutorAgent with no system prompt file."""
    return ExecutorAgent(system_prompt_path="nonexistent.md")


@pytest.fixture
def code_report():
    """Create a code-modality report."""
    return TaskIntelligenceReport(
        literal_request="Write a Python function to sort a list",
        inferred_intent="Create sorting function",
        sub_tasks=[
            SubTask(description="Implement sort function", dependencies=[]),
        ],
        missing_info=[],
        assumptions=["Python"],
        modality=ModalityType.CODE,
        recommended_approach="Direct implementation",
        escalation_needed=False,
    )


class TestExecutorInitialization:
    """Tests for ExecutorAgent initialization."""

    def test_default_initialization(self):
        """Test default init parameters."""
        agent = ExecutorAgent(system_prompt_path="nonexistent.md")
        assert agent.model == "claude-3-5-sonnet-20241022"
        assert agent.max_turns == 50

    def test_custom_initialization(self):
        """Test custom init parameters."""
        agent = ExecutorAgent(
            system_prompt_path="custom.md",
            model="claude-3-opus",
            max_turns=100,
        )
        assert agent.model == "claude-3-opus"
        assert agent.max_turns == 100

    def test_scoring_weights_initialized(self):
        """Test scoring weights are set."""
        agent = ExecutorAgent(system_prompt_path="nonexistent.md")
        assert "completeness" in agent.scoring_weights
        assert "quality" in agent.scoring_weights
        assert sum(agent.scoring_weights.values()) == pytest.approx(1.0)

    def test_system_prompt_fallback(self):
        """Test fallback prompt."""
        agent = ExecutorAgent(system_prompt_path="nonexistent.md")
        assert "Executor" in agent.system_prompt

    def test_system_prompt_from_file(self):
        """Test loading from file."""
        with patch("builtins.open", mock_open(read_data="Executor prompt")):
            agent = ExecutorAgent(system_prompt_path="exists.md")
            assert agent.system_prompt == "Executor prompt"


class TestExecute:
    """Tests for the execute method."""

    def test_basic_execution(self, executor):
        """Test basic task execution returns ExecutionResult."""
        result = executor.execute("Write a simple function")
        assert isinstance(result, ExecutionResult)
        assert result.status in ["success", "partial", "failed"]

    def test_code_task_execution(self, executor, code_report):
        """Test code task execution with analyst report."""
        result = executor.execute(
            "Write a Python function to sort a list",
            analyst_report=code_report,
        )
        assert result.status == "success"
        assert result.output is not None

    def test_document_task_execution(self, executor):
        """Test document task execution."""
        result = executor.execute("Write documentation for the API")
        assert result.status == "success"
        assert "documentation" in result.output.lower() or "document" in result.output.lower()

    def test_analysis_task_execution(self, executor):
        """Test analysis task execution."""
        result = executor.execute("Analyze the performance bottleneck")
        assert result.status == "success"
        assert "analysis" in result.output.lower() or "analyz" in result.output.lower()

    def test_general_task_execution(self, executor):
        """Test general task execution."""
        result = executor.execute("Do something random")
        assert result.status == "success"
        assert result.output is not None

    def test_execution_time_recorded(self, executor):
        """Test execution time is recorded."""
        result = executor.execute("Quick task")
        assert result.execution_time >= 0

    def test_quality_score_set(self, executor):
        """Test quality score is set."""
        result = executor.execute("Write a function")
        assert 0.0 <= result.quality_score <= 1.0


class TestApproachGeneration:
    """Tests for approach generation."""

    def test_generates_multiple_approaches(self, executor, code_report):
        """Test multiple approaches are generated."""
        sub_problems = executor._decompose_problem("Write code", code_report)
        approaches = executor._generate_approaches(sub_problems, code_report)
        assert len(approaches) >= 2

    def test_code_modality_extra_approach(self, executor, code_report):
        """Test code modality generates optimized approach."""
        sub_problems = ["Implement code"]
        approaches = executor._generate_approaches(sub_problems, code_report)
        names = [a.name for a in approaches]
        assert "Optimized Implementation" in names

    def test_approach_has_required_fields(self, executor):
        """Test approaches have all required fields."""
        sub_problems = ["Do something"]
        approaches = executor._generate_approaches(sub_problems)
        for approach in approaches:
            assert approach.name
            assert approach.description
            assert len(approach.steps) > 0
            assert len(approach.pros) > 0
            assert len(approach.cons) > 0


class TestApproachScoring:
    """Tests for approach scoring."""

    def test_scoring_produces_valid_scores(self, executor):
        """Test scoring produces valid scores."""
        approaches = [
            Approach(
                name="Direct", description="Direct", steps=["step1"],
                pros=["fast"], cons=["simple"],
                estimated_time="low", complexity="low",
            ),
            Approach(
                name="Comprehensive", description="Comprehensive",
                steps=["s1", "s2", "s3", "s4"],
                pros=["thorough"], cons=["slow"],
                estimated_time="high", complexity="high",
            ),
        ]
        scored = executor._score_approaches(approaches, "Write code")
        for approach in scored:
            assert 0.0 <= approach.score <= 1.0

    def test_approaches_sorted_by_score(self, executor):
        """Test approaches are sorted descending by score."""
        approaches = [
            Approach(name="A", description="A", steps=["s"], pros=["p"], cons=["c"],
                     estimated_time="high", complexity="high"),
            Approach(name="B", description="B", steps=["s"], pros=["p"], cons=["c"],
                     estimated_time="low", complexity="low"),
        ]
        scored = executor._score_approaches(approaches, "Write code")
        assert scored[0].score >= scored[-1].score

    def test_select_best_approach(self, executor):
        """Test best approach selection."""
        approach = executor._select_best_approach([])
        assert approach.name == "Standard Approach"

    def test_select_best_approach_returns_highest(self, executor):
        """Test best approach is the highest scored."""
        a1 = Approach(name="A", description="A", steps=["s"], pros=[], cons=[],
                      estimated_time="low", complexity="low", score=0.9)
        a2 = Approach(name="B", description="B", steps=["s"], pros=[], cons=[],
                      estimated_time="high", complexity="high", score=0.5)
        best = executor._select_best_approach([a1, a2])
        assert best.name == "A"


class TestSMEAdaptation:
    """Tests for SME advisory adaptation."""

    def test_adapt_to_sme_advice(self, executor):
        """Test approach adapts to SME advice."""
        approach = Approach(
            name="Direct", description="Direct",
            steps=["step1"], pros=["fast"], cons=["simple"],
            estimated_time="low", complexity="low",
        )
        adapted = executor._adapt_to_sme_advice(
            approach,
            {"Security": "Add input validation"},
        )
        assert any("SME" in p for p in adapted.pros)
        assert adapted.complexity == "medium"  # Bumped from low

    def test_no_sme_no_change(self, executor):
        """Test no change without SME advice."""
        approach = Approach(
            name="Direct", description="Direct",
            steps=["step1"], pros=["fast"], cons=["simple"],
            estimated_time="low", complexity="low",
        )
        # Without SME, complexity stays the same
        original_complexity = approach.complexity
        adapted = executor._adapt_to_sme_advice(approach, {})
        assert adapted.complexity == original_complexity


class TestProblemDecomposition:
    """Tests for problem decomposition."""

    def test_api_decomposition(self, executor):
        """Test API task decomposition."""
        sub_problems = executor._decompose_problem("Build an API endpoint", None)
        assert len(sub_problems) > 0
        assert any("api" in p.lower() or "endpoint" in p.lower() for p in sub_problems)

    def test_decomposition_from_report(self, executor, code_report):
        """Test decomposition uses analyst report subtasks."""
        sub_problems = executor._decompose_problem("Write code", code_report)
        assert any("Implement" in p for p in sub_problems)

    @pytest.mark.parametrize("task,expected_keyword", [
        ("Write unit tests", "test"),
        ("Create documentation", "document"),
        ("Build an API endpoint", "api"),
    ])
    def test_task_specific_decomposition(self, executor, task, expected_keyword):
        """Test task-specific decomposition keywords."""
        sub_problems = executor._decompose_problem(task, None)
        assert any(expected_keyword in p.lower() for p in sub_problems)


class TestFilePath:
    """Tests for file path determination."""

    @pytest.mark.parametrize("task,expected_ext", [
        ("Write Python code", ".py"),
        ("Create a JavaScript module", ".js"),
        ("Build a TypeScript service", ".ts"),
        ("Write a Java class", ".java"),
    ])
    def test_file_path_detection(self, executor, task, expected_ext):
        """Test file path detection from task description."""
        path = executor._determine_file_path(task)
        assert path is not None
        assert path.endswith(expected_ext)

    def test_no_file_path_for_generic(self, executor):
        """Test no file path for generic tasks."""
        path = executor._determine_file_path("Do something random")
        assert path is None


class TestConvenienceFunction:
    """Tests for create_executor convenience function."""

    def test_create_executor(self):
        """Test convenience function creates an ExecutorAgent."""
        agent = create_executor(system_prompt_path="nonexistent.md")
        assert isinstance(agent, ExecutorAgent)


# =============================================================================
# SDK Execution Tests
# =============================================================================

class TestSDKExecution:
    """Tests for _execute_via_sdk method."""

    def test_execute_via_sdk_success(self, executor):
        """Test _execute_via_sdk returns ExecutionResult when SDK succeeds."""
        mock_result = {
            "status": "success",
            "output": "Generated solution code here",
        }
        approach = Approach(
            name="Direct", description="Direct",
            steps=["step1"], pros=["fast"], cons=["simple"],
            estimated_time="low", complexity="low",
        )

        with patch("src.core.sdk_integration.spawn_subagent", return_value=mock_result), \
             patch("src.core.sdk_integration.build_agent_options"):
            result = executor._execute_via_sdk(approach, "Write a sort function")
            assert result is not None
            assert isinstance(result, ExecutionResult)
            assert result.status == "success"
            assert result.output == "Generated solution code here"
            # Quality score is derived from output length, not hardcoded
            assert 0.0 < result.quality_score <= 1.0

    def test_execute_via_sdk_failure_returns_none(self, executor):
        """Test _execute_via_sdk returns None when SDK fails."""
        mock_result = {"status": "error", "output": None, "error": "API error"}
        approach = Approach(
            name="Direct", description="Direct",
            steps=["step1"], pros=["fast"], cons=["simple"],
            estimated_time="low", complexity="low",
        )

        with patch("src.core.sdk_integration.spawn_subagent", return_value=mock_result), \
             patch("src.core.sdk_integration.build_agent_options"):
            result = executor._execute_via_sdk(approach, "Write a function")
            assert result is None

    def test_execute_via_sdk_import_error_returns_simulated(self, executor):
        """Test _execute_via_sdk falls back to simulated result when SDK is not available."""
        approach = Approach(
            name="Direct", description="Direct",
            steps=["step1"], pros=["fast"], cons=["simple"],
            estimated_time="low", complexity="low",
        )
        # Without patching, SDK import may fail but executor falls back to simulation
        result = executor._execute_via_sdk(approach, "Write a function")
        # Should return a simulated result (graceful degradation per FR-054)
        assert result is not None
        assert isinstance(result, ExecutionResult)

    def test_execute_via_sdk_with_context(self, executor):
        """Test _execute_via_sdk passes context to the prompt."""
        mock_result = {
            "status": "success",
            "output": "Solution with context",
        }
        approach = Approach(
            name="Comprehensive", description="Full solution",
            steps=["step1", "step2"], pros=["thorough"], cons=["slow"],
            estimated_time="high", complexity="high",
        )
        context = {
            "analyst_report": "Task analysis details",
            "plan": "Execution plan details",
        }

        with patch("src.core.sdk_integration.spawn_subagent", return_value=mock_result) as mock_spawn, \
             patch("src.core.sdk_integration.build_agent_options"):
            result = executor._execute_via_sdk(approach, "Build API", context)
            assert result is not None
            assert result.status == "success"
            # Verify spawn_subagent was called with context in the prompt
            call_args = mock_spawn.call_args
            input_data = call_args.kwargs.get("input_data", call_args[1].get("input_data", ""))
            assert "Analyst_Report" in input_data or "analyst_report" in input_data.lower() or "Analyst" in input_data

    def test_execute_falls_back_to_local_when_sdk_fails(self, executor):
        """Test that execute() falls back to local execution when SDK returns None."""
        result = executor.execute("Write a Python function")
        # Should still succeed via local execution fallback
        assert isinstance(result, ExecutionResult)
        assert result.status == "success"
