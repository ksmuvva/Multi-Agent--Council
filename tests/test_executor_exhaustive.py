"""
Exhaustive tests for ExecutorAgent (src/agents/executor.py).

Covers:
- __init__ with defaults, custom params, system prompt loading
- ThoughtBranch enum values
- Approach dataclass fields and scoring
- ThoughtNode dataclass (parent, children, depth, explored, selected)
- ExecutionResult dataclass (status values)
- execute() full method
- _decompose_problem() - with and without analyst report
- _generate_approaches() - multiple approach generation
- _score_approaches() / _score_completeness / _score_quality / _score_efficiency /
  _score_maintainability / _score_feasibility
- _select_best_approach() - selection criteria, empty list fallback
- _adapt_to_sme_advice() - step injection, complexity escalation
- _execute_approach() - dispatch to task-type handlers
- _execute_code_task, _execute_document_task, _execute_analysis_task, _execute_general_task
- _validate_output() - None, empty, short output handling
- _load_system_prompt() - file and fallback
- Edge cases: empty request, no approaches
"""

import os
import time
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from dataclasses import dataclass, field

from src.agents.executor import (
    ExecutorAgent,
    ThoughtBranch,
    Approach,
    ThoughtNode,
    ExecutionResult,
    create_executor,
)
from src.schemas.analyst import ModalityType, TaskIntelligenceReport, SubTask


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def executor(tmp_path):
    """Create an ExecutorAgent with fallback system prompt."""
    with patch("builtins.open", side_effect=FileNotFoundError):
        agent = ExecutorAgent(system_prompt_path="nonexistent.md")
    return agent


@pytest.fixture
def executor_with_prompt(tmp_path):
    """Create an ExecutorAgent with a real system prompt file."""
    prompt_file = tmp_path / "executor_prompt.md"
    prompt_file.write_text("You are the Executor agent.")
    return ExecutorAgent(system_prompt_path=str(prompt_file))


@pytest.fixture
def sample_approach():
    return Approach(
        name="Direct Implementation",
        description="Straightforward approach",
        steps=["Understand", "Implement", "Test"],
        pros=["Simple", "Fast"],
        cons=["May miss edge cases"],
        estimated_time="low",
        complexity="low",
    )


@pytest.fixture
def sample_approaches():
    return [
        Approach(
            name="Direct Implementation",
            description="Standard approach",
            steps=["Step 1", "Step 2"],
            pros=["Simple"], cons=["Basic"],
            estimated_time="low", complexity="low",
        ),
        Approach(
            name="Comprehensive Solution",
            description="Robust approach",
            steps=["Analyze", "Design", "Implement", "Test", "Document"],
            pros=["Thorough"], cons=["Slow"],
            estimated_time="high", complexity="high",
        ),
    ]


@pytest.fixture
def mock_analyst_report():
    return TaskIntelligenceReport(
        literal_request="Build an API",
        inferred_intent="Create a REST API",
        sub_tasks=[
            SubTask(description="Design data models"),
            SubTask(description="Implement endpoints"),
            SubTask(description="Add authentication"),
        ],
        missing_info=[],
        assumptions=["Python will be used"],
        modality=ModalityType.CODE,
        recommended_approach="Direct implementation",
        escalation_needed=False,
    )


# ============================================================================
# ThoughtBranch Enum
# ============================================================================

class TestThoughtBranch:
    def test_values(self):
        assert ThoughtBranch.SEQUENTIAL == "sequential"
        assert ThoughtBranch.PARALLEL == "parallel"
        assert ThoughtBranch.DECOMPOSE == "decompose"
        assert ThoughtBranch.SIMPLIFY == "simplify"

    def test_is_string(self):
        assert isinstance(ThoughtBranch.SEQUENTIAL, str)

    def test_count(self):
        assert len(ThoughtBranch) == 4


# ============================================================================
# Approach Dataclass
# ============================================================================

class TestApproach:
    def test_creation(self, sample_approach):
        assert sample_approach.name == "Direct Implementation"
        assert sample_approach.estimated_time == "low"
        assert sample_approach.complexity == "low"
        assert sample_approach.score == 0.0

    def test_default_score(self):
        a = Approach(
            name="A", description="B", steps=[], pros=[], cons=[],
            estimated_time="low", complexity="low",
        )
        assert a.score == 0.0

    def test_score_settable(self):
        a = Approach(
            name="A", description="B", steps=[], pros=[], cons=[],
            estimated_time="low", complexity="low", score=0.95,
        )
        assert a.score == 0.95

    @pytest.mark.parametrize("time_val", ["low", "medium", "high"])
    def test_estimated_time_values(self, time_val):
        a = Approach(
            name="A", description="B", steps=[], pros=[], cons=[],
            estimated_time=time_val, complexity="low",
        )
        assert a.estimated_time == time_val

    @pytest.mark.parametrize("complexity_val", ["low", "medium", "high"])
    def test_complexity_values(self, complexity_val):
        a = Approach(
            name="A", description="B", steps=[], pros=[], cons=[],
            estimated_time="low", complexity=complexity_val,
        )
        assert a.complexity == complexity_val


# ============================================================================
# ThoughtNode Dataclass
# ============================================================================

class TestThoughtNode:
    def test_creation(self, sample_approach):
        node = ThoughtNode(approach=sample_approach)
        assert node.parent is None
        assert node.children == []
        assert node.depth == 0
        assert node.explored is False
        assert node.selected is False

    def test_parent_child_relationship(self, sample_approach):
        parent = ThoughtNode(approach=sample_approach, depth=0)
        child = ThoughtNode(approach=sample_approach, parent=parent, depth=1)
        parent.children.append(child)
        assert child.parent is parent
        assert len(parent.children) == 1
        assert parent.children[0] is child

    def test_depth(self, sample_approach):
        node = ThoughtNode(approach=sample_approach, depth=3)
        assert node.depth == 3

    def test_explored_flag(self, sample_approach):
        node = ThoughtNode(approach=sample_approach, explored=True)
        assert node.explored is True

    def test_selected_flag(self, sample_approach):
        node = ThoughtNode(approach=sample_approach, selected=True)
        assert node.selected is True


# ============================================================================
# ExecutionResult Dataclass
# ============================================================================

class TestExecutionResult:
    def test_creation(self):
        result = ExecutionResult(
            approach_name="Direct",
            status="success",
        )
        assert result.approach_name == "Direct"
        assert result.status == "success"
        assert result.output is None
        assert result.files_created == []
        assert result.execution_time == 0.0
        assert result.error is None
        assert result.quality_score == 0.0

    @pytest.mark.parametrize("status", ["success", "partial", "failed"])
    def test_status_values(self, status):
        result = ExecutionResult(approach_name="Test", status=status)
        assert result.status == status

    def test_with_output(self):
        result = ExecutionResult(
            approach_name="Test",
            status="success",
            output="Generated code here",
            files_created=["output/main.py"],
            execution_time=1.5,
            quality_score=0.9,
        )
        assert result.output == "Generated code here"
        assert "output/main.py" in result.files_created
        assert result.execution_time == 1.5
        assert result.quality_score == 0.9

    def test_with_error(self):
        result = ExecutionResult(
            approach_name="Test",
            status="failed",
            error="Something went wrong",
        )
        assert result.error == "Something went wrong"


# ============================================================================
# __init__
# ============================================================================

class TestInit:
    def test_defaults(self, executor):
        assert executor.model == "claude-sonnet-4-20250514"
        assert executor.max_turns == 50
        assert executor.system_prompt_path == "nonexistent.md"

    def test_custom_params(self, tmp_path):
        prompt = tmp_path / "p.md"
        prompt.write_text("custom prompt")
        agent = ExecutorAgent(
            system_prompt_path=str(prompt),
            model="claude-3-opus",
            max_turns=20,
        )
        assert agent.model == "claude-3-opus"
        assert agent.max_turns == 20
        assert agent.system_prompt == "custom prompt"

    def test_system_prompt_fallback(self, executor):
        assert "Executor" in executor.system_prompt
        assert "Tree of Thoughts" in executor.system_prompt

    def test_system_prompt_loaded(self, executor_with_prompt):
        assert executor_with_prompt.system_prompt == "You are the Executor agent."

    def test_scoring_weights(self, executor):
        weights = executor.scoring_weights
        assert "completeness" in weights
        assert "quality" in weights
        assert "efficiency" in weights
        assert "maintainability" in weights
        assert "feasibility" in weights
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_output_dir(self, executor):
        assert executor.output_dir == "output"


# ============================================================================
# _decompose_problem
# ============================================================================

class TestDecomposeProblem:
    def test_with_analyst_report(self, executor, mock_analyst_report):
        subs = executor._decompose_problem("Build API", mock_analyst_report)
        assert len(subs) == 3
        assert "Design data models" in subs

    def test_api_task(self, executor):
        subs = executor._decompose_problem("Create an API endpoint", None)
        assert len(subs) >= 3
        assert any("api" in s.lower() or "endpoint" in s.lower() for s in subs)

    def test_test_task(self, executor):
        subs = executor._decompose_problem("Write test cases", None)
        assert len(subs) >= 2
        assert any("test" in s.lower() for s in subs)

    def test_document_task(self, executor):
        subs = executor._decompose_problem("Create documentation", None)
        assert len(subs) >= 2
        assert any("document" in s.lower() for s in subs)

    def test_general_task(self, executor):
        subs = executor._decompose_problem("Do something generic", None)
        assert len(subs) >= 2


# ============================================================================
# _generate_approaches
# ============================================================================

class TestGenerateApproaches:
    def test_always_generates_at_least_two(self, executor):
        subs = ["sub1", "sub2"]
        approaches = executor._generate_approaches(subs)
        assert len(approaches) >= 2

    def test_direct_approach_present(self, executor):
        approaches = executor._generate_approaches(["s1", "s2"])
        names = [a.name for a in approaches]
        assert "Direct Implementation" in names

    def test_comprehensive_approach_present(self, executor):
        approaches = executor._generate_approaches(["s1", "s2"])
        names = [a.name for a in approaches]
        assert "Comprehensive Solution" in names

    def test_code_modality_adds_optimized(self, executor, mock_analyst_report):
        approaches = executor._generate_approaches(["s1"], mock_analyst_report)
        names = [a.name for a in approaches]
        assert any("Optimized" in n or "Performance" in n for n in names)

    def test_approaches_have_steps(self, executor):
        approaches = executor._generate_approaches(["s1", "s2"])
        for a in approaches:
            assert len(a.steps) >= 1

    def test_approaches_have_pros_cons(self, executor):
        approaches = executor._generate_approaches(["s1"])
        for a in approaches:
            assert len(a.pros) >= 1
            assert len(a.cons) >= 1


# ============================================================================
# Scoring Methods
# ============================================================================

class TestScoring:
    def test_score_approaches_returns_sorted(self, executor, sample_approaches):
        scored = executor._score_approaches(sample_approaches, "Build an API")
        assert len(scored) == 2
        assert scored[0].score >= scored[1].score

    def test_all_approaches_scored(self, executor, sample_approaches):
        scored = executor._score_approaches(sample_approaches, "task")
        for a in scored:
            assert a.score > 0

    def test_score_completeness_base(self, executor):
        a = Approach(
            name="A", description="B", steps=["s1"],
            pros=[], cons=[], estimated_time="low", complexity="low",
        )
        score = executor._score_completeness(a, "task", None)
        assert score >= 0.7

    def test_score_completeness_many_steps(self, executor):
        a = Approach(
            name="Comprehensive", description="B",
            steps=["s1", "s2", "s3", "s4"],
            pros=[], cons=[], estimated_time="low", complexity="low",
        )
        score = executor._score_completeness(a, "task", None)
        assert score >= 0.9

    def test_score_completeness_comprehensive_bonus(self, executor):
        a = Approach(
            name="Comprehensive Solution", description="B",
            steps=["s1", "s2", "s3", "s4"],
            pros=[], cons=[], estimated_time="low", complexity="low",
        )
        score = executor._score_completeness(a, "task", None)
        assert score == pytest.approx(1.0)

    def test_score_completeness_capped_at_1(self, executor):
        a = Approach(
            name="Comprehensive Robust Solution", description="B",
            steps=["s1", "s2", "s3", "s4", "s5"],
            pros=[], cons=[], estimated_time="low", complexity="low",
        )
        score = executor._score_completeness(a, "task", None)
        assert score <= 1.0

    @pytest.mark.parametrize("complexity,expected", [
        ("high", 0.9),
        ("medium", 0.7),
        ("low", 0.5),
    ])
    def test_score_quality(self, executor, complexity, expected):
        a = Approach(
            name="A", description="B", steps=[], pros=[], cons=[],
            estimated_time="low", complexity=complexity,
        )
        assert executor._score_quality(a) == expected

    @pytest.mark.parametrize("time_val,expected", [
        ("low", 1.0),
        ("medium", 0.7),
        ("high", 0.4),
    ])
    def test_score_efficiency(self, executor, time_val, expected):
        a = Approach(
            name="A", description="B", steps=[], pros=[], cons=[],
            estimated_time=time_val, complexity="low",
        )
        assert executor._score_efficiency(a) == expected

    def test_score_maintainability_direct(self, executor):
        a = Approach(
            name="Direct Implementation", description="B",
            steps=[], pros=[], cons=[],
            estimated_time="low", complexity="low",
        )
        assert executor._score_maintainability(a) == 0.8

    def test_score_maintainability_comprehensive(self, executor):
        a = Approach(
            name="Comprehensive Solution", description="B",
            steps=[], pros=[], cons=[],
            estimated_time="low", complexity="low",
        )
        assert executor._score_maintainability(a) == 0.9

    def test_score_maintainability_optimized(self, executor):
        a = Approach(
            name="Optimized Approach", description="B",
            steps=[], pros=[], cons=[],
            estimated_time="low", complexity="low",
        )
        assert executor._score_maintainability(a) == 0.6

    def test_score_maintainability_other(self, executor):
        a = Approach(
            name="Custom Approach", description="B",
            steps=[], pros=[], cons=[],
            estimated_time="low", complexity="low",
        )
        assert executor._score_maintainability(a) == 0.7

    @pytest.mark.parametrize("complexity,expected", [
        ("low", 1.0),
        ("medium", 0.8),
        ("high", 0.6),
    ])
    def test_score_feasibility(self, executor, complexity, expected):
        a = Approach(
            name="A", description="B", steps=[], pros=[], cons=[],
            estimated_time="low", complexity=complexity,
        )
        assert executor._score_feasibility(a) == expected


# ============================================================================
# _select_best_approach
# ============================================================================

class TestSelectBestApproach:
    def test_returns_highest_scored(self, executor):
        a1 = Approach(
            name="A", description="", steps=[], pros=[], cons=[],
            estimated_time="low", complexity="low", score=0.5,
        )
        a2 = Approach(
            name="B", description="", steps=[], pros=[], cons=[],
            estimated_time="low", complexity="low", score=0.9,
        )
        # _select_best_approach expects sorted list (first is best)
        selected = executor._select_best_approach([a2, a1])
        assert selected.name == "B"

    def test_empty_list_returns_default(self, executor):
        selected = executor._select_best_approach([])
        assert selected.name == "Standard Approach"
        assert selected.complexity == "medium"
        assert len(selected.steps) >= 2

    def test_single_approach(self, executor, sample_approach):
        selected = executor._select_best_approach([sample_approach])
        assert selected.name == "Direct Implementation"


# ============================================================================
# _adapt_to_sme_advice
# ============================================================================

class TestAdaptToSmeAdvice:
    def test_adds_sme_steps(self, executor, sample_approach):
        sme_advice = {
            "SecuritySME": "Implement rate limiting. Use encryption for all data."
        }
        adapted = executor._adapt_to_sme_advice(sample_approach, sme_advice)
        sme_steps = [s for s in adapted.steps if "[SME:" in s]
        assert len(sme_steps) >= 1

    def test_adds_sme_endorsement_to_pros(self, executor, sample_approach):
        sme_advice = {"CloudSME": "Use containerization."}
        adapted = executor._adapt_to_sme_advice(sample_approach, sme_advice)
        assert any("SME" in p and "CloudSME" in p for p in adapted.pros)

    def test_escalates_complexity_on_security(self, executor):
        a = Approach(
            name="A", description="B", steps=["s1", "s2"],
            pros=[], cons=[], estimated_time="low", complexity="low",
        )
        sme_advice = {"SecSME": "Add security measures and encryption."}
        adapted = executor._adapt_to_sme_advice(a, sme_advice)
        assert adapted.complexity in ("medium", "high")

    def test_escalates_complexity_medium_to_high(self, executor):
        a = Approach(
            name="A", description="B", steps=["s1", "s2"],
            pros=[], cons=[], estimated_time="low", complexity="medium",
        )
        sme_advice = {"SME": "Ensure security compliance."}
        adapted = executor._adapt_to_sme_advice(a, sme_advice)
        assert adapted.complexity == "high"

    def test_no_complexity_change_without_keywords(self, executor):
        a = Approach(
            name="A", description="B", steps=["s1", "s2"],
            pros=[], cons=[], estimated_time="low", complexity="low",
        )
        sme_advice = {"SME": "Make it pretty."}
        adapted = executor._adapt_to_sme_advice(a, sme_advice)
        assert adapted.complexity == "low"

    def test_description_updated_for_architecture(self, executor, sample_approach):
        sme_advice = {"ArchSME": "Review the architecture design."}
        adapted = executor._adapt_to_sme_advice(sample_approach, sme_advice)
        assert "architecture" in adapted.description.lower() or "guided" in adapted.description.lower()

    def test_description_updated_for_security(self, executor, sample_approach):
        sme_advice = {"SecSME": "Implement security hardening."}
        adapted = executor._adapt_to_sme_advice(sample_approach, sme_advice)
        assert "security" in adapted.description.lower()

    def test_empty_sme_advice(self, executor, sample_approach):
        adapted = executor._adapt_to_sme_advice(sample_approach, {})
        assert adapted.name == sample_approach.name

    def test_should_must_recommend_steps(self, executor):
        a = Approach(
            name="A", description="B", steps=["s1", "s2", "final"],
            pros=[], cons=[], estimated_time="low", complexity="low",
        )
        sme_advice = {"SME": "You should validate inputs. You must handle errors."}
        adapted = executor._adapt_to_sme_advice(a, sme_advice)
        sme_steps = [s for s in adapted.steps if "[SME:" in s]
        assert len(sme_steps) >= 1


# ============================================================================
# _execute_approach (dispatch)
# ============================================================================

class TestExecuteApproach:
    def test_code_task(self, executor, sample_approach):
        result = executor._execute_approach(
            sample_approach, "Implement a Python function for caching", None
        )
        assert isinstance(result, ExecutionResult)
        assert result.status == "success"

    def test_document_task(self, executor, sample_approach):
        result = executor._execute_approach(
            sample_approach, "Write documentation for the API", None
        )
        assert isinstance(result, ExecutionResult)
        assert result.status == "success"

    def test_analysis_task(self, executor, sample_approach):
        result = executor._execute_approach(
            sample_approach, "Analyze the code quality", None
        )
        assert isinstance(result, ExecutionResult)
        assert result.status == "success"

    def test_general_task(self, executor, sample_approach):
        result = executor._execute_approach(
            sample_approach, "Do something unique and special", None
        )
        assert isinstance(result, ExecutionResult)
        assert result.status == "success"

    def test_approach_metadata_appended(self, executor, sample_approach):
        result = executor._execute_approach(
            sample_approach, "Do something unique", None
        )
        if isinstance(result.output, str):
            assert sample_approach.name in result.output


# ============================================================================
# _execute_code_task
# ============================================================================

class TestExecuteCodeTask:
    def test_returns_success(self, executor, sample_approach):
        result = executor._execute_code_task(
            sample_approach, "Implement a Python function", None
        )
        assert result.status == "success"
        assert result.output is not None

    def test_quality_score(self, executor, sample_approach):
        result = executor._execute_code_task(
            sample_approach, "Implement a Python class", None
        )
        assert 0.0 <= result.quality_score <= 1.0


# ============================================================================
# _execute_document_task
# ============================================================================

class TestExecuteDocumentTask:
    def test_returns_success(self, executor, sample_approach):
        result = executor._execute_document_task(
            sample_approach, "Write a README document", None
        )
        assert result.status == "success"
        assert result.output is not None

    def test_readme_file_created(self, executor, sample_approach):
        result = executor._execute_document_task(
            sample_approach, "Write a README", None
        )
        assert any("README" in f for f in result.files_created)

    def test_api_doc_file(self, executor, sample_approach):
        result = executor._execute_document_task(
            sample_approach, "Write API documentation", None
        )
        assert any("api" in f.lower() for f in result.files_created)

    def test_guide_file(self, executor, sample_approach):
        result = executor._execute_document_task(
            sample_approach, "Write a guide for users", None
        )
        assert any("guide" in f.lower() for f in result.files_created)

    def test_report_file(self, executor, sample_approach):
        result = executor._execute_document_task(
            sample_approach, "Write a report on performance", None
        )
        assert any("report" in f.lower() for f in result.files_created)


# ============================================================================
# _execute_analysis_task
# ============================================================================

class TestExecuteAnalysisTask:
    def test_returns_success(self, executor, sample_approach):
        result = executor._execute_analysis_task(
            sample_approach, "Analyze the codebase", None
        )
        assert result.status == "success"
        assert result.quality_score == 0.88

    def test_no_files_created(self, executor, sample_approach):
        result = executor._execute_analysis_task(
            sample_approach, "Analyze the system", None
        )
        assert result.files_created == []


# ============================================================================
# _execute_general_task
# ============================================================================

class TestExecuteGeneralTask:
    def test_returns_success(self, executor, sample_approach):
        result = executor._execute_general_task(
            sample_approach, "Do something general", None
        )
        assert result.status == "success"
        assert result.output is not None

    def test_output_contains_approach_info(self, executor, sample_approach):
        result = executor._execute_general_task(
            sample_approach, "General task", None
        )
        assert sample_approach.name in result.output

    def test_output_contains_steps(self, executor, sample_approach):
        result = executor._execute_general_task(
            sample_approach, "General task", None
        )
        for step in sample_approach.steps:
            assert step in result.output

    def test_quality_increases_with_steps(self, executor):
        a_few = Approach(
            name="A", description="B", steps=["s1", "s2", "s3"],
            pros=["p"], cons=["c"], estimated_time="low", complexity="low",
        )
        a_less = Approach(
            name="A", description="B", steps=["s1"],
            pros=["p"], cons=["c"], estimated_time="low", complexity="low",
        )
        r_few = executor._execute_general_task(a_few, "task", None)
        r_less = executor._execute_general_task(a_less, "task", None)
        assert r_few.quality_score >= r_less.quality_score


# ============================================================================
# _validate_output
# ============================================================================

class TestValidateOutput:
    def test_none_output(self, executor):
        result = ExecutionResult(approach_name="T", status="success")
        result.output = None
        validated = executor._validate_output(result)
        assert validated.status == "failed"
        assert validated.error is not None

    def test_empty_string_output(self, executor):
        result = ExecutionResult(approach_name="T", status="success", output="")
        validated = executor._validate_output(result)
        assert validated.status == "failed"

    def test_short_output(self, executor):
        result = ExecutionResult(
            approach_name="T", status="success",
            output="short", quality_score=0.8,
        )
        validated = executor._validate_output(result)
        assert validated.quality_score < 0.8

    def test_empty_dict_output(self, executor):
        result = ExecutionResult(
            approach_name="T", status="success", output={},
        )
        validated = executor._validate_output(result)
        assert validated.status == "partial"

    def test_valid_output_gets_bonus(self, executor):
        result = ExecutionResult(
            approach_name="T", status="success",
            output="This is a properly long output string that passes validation.",
            quality_score=0.85,
        )
        validated = executor._validate_output(result)
        assert validated.status == "success"
        # Validation adds +0.1 bonus when no issues found
        assert validated.quality_score == pytest.approx(0.95)


# ============================================================================
# execute() full flow
# ============================================================================

class TestExecuteFullFlow:
    def test_basic_execution(self, executor):
        result = executor.execute("Implement a Python function for sorting")
        assert isinstance(result, ExecutionResult)
        assert result.status in ("success", "partial", "failed")
        assert result.execution_time > 0

    def test_with_analyst_report(self, executor, mock_analyst_report):
        result = executor.execute(
            "Build an API",
            analyst_report=mock_analyst_report,
        )
        assert isinstance(result, ExecutionResult)

    def test_with_sme_advisory(self, executor):
        result = executor.execute(
            "Implement a secure API",
            sme_advisory={"SecuritySME": "Use rate limiting."},
        )
        assert isinstance(result, ExecutionResult)

    def test_with_context(self, executor):
        result = executor.execute(
            "Build a module",
            context={"output_dir": "/tmp/test"},
        )
        assert isinstance(result, ExecutionResult)

    def test_code_task_execution(self, executor):
        result = executor.execute("Implement a Python class for data processing")
        assert result.status in ("success", "partial", "failed")
        if result.status == "success":
            assert result.output is not None

    def test_document_task_execution(self, executor):
        result = executor.execute("Write documentation for the system")
        assert result.status in ("success", "partial", "failed")

    def test_analysis_task_execution(self, executor):
        result = executor.execute("Analyze the performance of the system")
        assert result.status in ("success", "partial", "failed")

    def test_empty_task(self, executor):
        result = executor.execute("")
        assert isinstance(result, ExecutionResult)

    def test_timing(self, executor):
        result = executor.execute("Do something simple")
        assert result.execution_time >= 0

    def test_all_context_combined(self, executor, mock_analyst_report):
        result = executor.execute(
            "Implement a secure API endpoint",
            analyst_report=mock_analyst_report,
            sme_advisory={
                "SecuritySME": "Add authentication.",
                "CloudSME": "Use containerization.",
            },
            context={"output_dir": "/tmp"},
        )
        assert isinstance(result, ExecutionResult)
        assert result.execution_time > 0


# ============================================================================
# _load_system_prompt
# ============================================================================

class TestLoadSystemPrompt:
    def test_file_exists(self, tmp_path):
        prompt_file = tmp_path / "prompt.md"
        prompt_file.write_text("Custom Executor prompt")
        agent = ExecutorAgent(system_prompt_path=str(prompt_file))
        assert agent.system_prompt == "Custom Executor prompt"

    def test_file_not_found_fallback(self):
        with patch("builtins.open", side_effect=FileNotFoundError):
            agent = ExecutorAgent(system_prompt_path="missing.md")
        assert "Executor" in agent.system_prompt
        assert "Tree of Thoughts" in agent.system_prompt


# ============================================================================
# Convenience function
# ============================================================================

class TestCreateExecutor:
    def test_creates_instance(self, tmp_path):
        prompt = tmp_path / "p.md"
        prompt.write_text("prompt")
        agent = create_executor(system_prompt_path=str(prompt))
        assert isinstance(agent, ExecutorAgent)

    def test_with_custom_model(self, tmp_path):
        prompt = tmp_path / "p.md"
        prompt.write_text("prompt")
        agent = create_executor(
            system_prompt_path=str(prompt),
            model="claude-3-opus",
        )
        assert agent.model == "claude-3-opus"


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    def test_very_long_task(self, executor):
        long_task = "implement " * 500
        result = executor.execute(long_task)
        assert isinstance(result, ExecutionResult)

    def test_special_characters_in_task(self, executor):
        result = executor.execute("Build API with @special #chars & symbols!")
        assert isinstance(result, ExecutionResult)

    def test_unicode_task(self, executor):
        result = executor.execute("Build a system for handling unicode: \u00e9\u00e8\u00ea\u00eb")
        assert isinstance(result, ExecutionResult)

    def test_multiple_task_types_combined(self, executor):
        # Task that matches multiple categories
        result = executor.execute(
            "Implement a function and document it, then analyze the code"
        )
        assert isinstance(result, ExecutionResult)

    def test_no_analyst_no_sme(self, executor):
        result = executor.execute("Simple task")
        assert isinstance(result, ExecutionResult)
        assert result.execution_time >= 0
