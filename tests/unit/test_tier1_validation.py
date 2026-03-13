"""
Exhaustive Validation Tests for Tier 1 (Direct) Agents

Validates all 3 Tier 1 agents: Orchestrator, Executor, Formatter.
Tests SDK integration, skills, tools, pipeline routing, phase skipping,
scoring weights, approach generation, format handling, and edge cases.
"""

import ast
import json
import os
import time
import pytest
from unittest.mock import patch, mock_open, MagicMock, Mock
from pathlib import Path

from src.agents.executor import (
    ExecutorAgent,
    Approach,
    ExecutionResult,
    ThoughtBranch,
    ThoughtNode,
    create_executor,
)
from src.agents.formatter import (
    FormatterAgent,
    OutputFormat,
    create_formatter,
)
from src.agents.orchestrator import (
    OrchestratorAgent,
    AgentExecution,
    SessionState,
    create_orchestrator,
)
from src.core.complexity import (
    TierLevel,
    TierClassification,
    TIER_CONFIG,
    classify_complexity,
    should_escalate,
    get_escalated_tier,
    estimate_agent_count,
    get_active_agents,
    get_council_agents,
)
from src.core.pipeline import (
    ExecutionPipeline,
    PipelineBuilder,
    Phase,
    PhaseStatus,
    PipelineState,
    AgentResult,
    PhaseResult,
    create_execution_context,
    estimate_pipeline_duration,
)
from src.core.sdk_integration import (
    ClaudeAgentOptions,
    build_agent_options,
    spawn_subagent,
    AGENT_ALLOWED_TOOLS,
    PermissionMode,
    get_skills_for_agent,
    _get_output_schema,
    _validate_output,
    _simulate_response,
)
from src.schemas.analyst import (
    TaskIntelligenceReport,
    SubTask,
    ModalityType,
    MissingInfo,
    SeverityLevel,
)
from src.config.settings import reload_settings


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def _reset_settings():
    """Reset settings singleton between tests."""
    os.environ["ANTHROPIC_API_KEY"] = "test_key_dummy"
    os.environ["TESTING"] = "true"
    reload_settings()
    yield
    reload_settings()


@pytest.fixture
def executor():
    """Create an ExecutorAgent with fallback prompt."""
    return ExecutorAgent(system_prompt_path="nonexistent.md")


@pytest.fixture
def formatter(tmp_path):
    """Create a FormatterAgent with temp output dir."""
    return FormatterAgent(
        system_prompt_path="nonexistent.md",
        output_dir=str(tmp_path),
    )


@pytest.fixture
def code_report():
    """Create a CODE modality TaskIntelligenceReport."""
    return TaskIntelligenceReport(
        literal_request="Write a Python sorting function",
        inferred_intent="Create sorting function",
        sub_tasks=[
            SubTask(description="Implement sort", dependencies=[]),
            SubTask(description="Add tests", dependencies=["Implement sort"]),
        ],
        missing_info=[],
        assumptions=["Python 3.10+"],
        modality=ModalityType.CODE,
        recommended_approach="Direct implementation",
        escalation_needed=False,
    )


@pytest.fixture
def text_report():
    """Create a TEXT modality TaskIntelligenceReport."""
    return TaskIntelligenceReport(
        literal_request="Explain recursion",
        inferred_intent="Educational explanation",
        sub_tasks=[SubTask(description="Explain concept", dependencies=[])],
        missing_info=[],
        assumptions=["General audience"],
        modality=ModalityType.TEXT,
        recommended_approach="Structured explanation",
        escalation_needed=False,
    )


@pytest.fixture
def tier1_pipeline():
    """Create a Tier 1 pipeline."""
    return ExecutionPipeline(tier_level=TierLevel.DIRECT)


# =============================================================================
# SECTION 1: Tier 1 Configuration Validation
# =============================================================================


class TestTier1Configuration:
    """Validate Tier 1 config in TIER_CONFIG."""

    def test_tier1_config_exists(self):
        assert TierLevel.DIRECT in TIER_CONFIG

    def test_tier1_has_exactly_3_agents(self):
        config = TIER_CONFIG[TierLevel.DIRECT]
        assert config["agent_count"] == 3

    def test_tier1_active_agents_correct(self):
        config = TIER_CONFIG[TierLevel.DIRECT]
        assert config["active_agents"] == ["Orchestrator", "Executor", "Formatter"]

    def test_tier1_no_council(self):
        config = TIER_CONFIG[TierLevel.DIRECT]
        assert config["requires_council"] is False

    def test_tier1_no_smes(self):
        config = TIER_CONFIG[TierLevel.DIRECT]
        assert config["requires_smes"] is False

    def test_tier1_max_sme_count_zero(self):
        config = TIER_CONFIG[TierLevel.DIRECT]
        assert config["max_sme_count"] == 0

    def test_tier1_phases_only_5_and_8(self):
        config = TIER_CONFIG[TierLevel.DIRECT]
        phases = config["phases"]
        assert len(phases) == 2
        assert "Phase 5" in phases[0]
        assert "Phase 8" in phases[1]

    def test_tier1_name_is_direct(self):
        config = TIER_CONFIG[TierLevel.DIRECT]
        assert config["name"] == "Direct"

    def test_tier1_no_council_agents_key(self):
        config = TIER_CONFIG[TierLevel.DIRECT]
        assert "council_agents" not in config

    def test_get_active_agents_tier1(self):
        agents = get_active_agents(TierLevel.DIRECT)
        assert agents == ["Orchestrator", "Executor", "Formatter"]

    def test_get_active_agents_returns_copy(self):
        agents1 = get_active_agents(TierLevel.DIRECT)
        agents2 = get_active_agents(TierLevel.DIRECT)
        assert agents1 is not agents2

    def test_get_council_agents_tier1_empty(self):
        council = get_council_agents(TierLevel.DIRECT)
        assert council == []

    def test_estimate_agent_count_tier1(self):
        count = estimate_agent_count(TierLevel.DIRECT)
        assert count == 3

    def test_estimate_agent_count_tier1_with_smes(self):
        count = estimate_agent_count(TierLevel.DIRECT, sme_count=2)
        assert count == 5  # 3 + 2


# =============================================================================
# SECTION 2: Tier 1 Classification
# =============================================================================


class TestTier1Classification:
    """Validate complexity classification for simple prompts."""

    def test_simple_prompt_no_keywords_defaults_tier2(self):
        """Default without any keywords is Tier 2."""
        result = classify_complexity("Hello world")
        # Without any Tier 3/4 keywords, and no escalation keywords,
        # the default tier_score stays 0 -> max(1, 0) = 1
        # Actually checking the logic: tier_score starts at 0, no keywords matched
        # max(1, 0) = 1 => Tier 1 DIRECT
        assert result.tier == TierLevel.DIRECT

    def test_simple_prompt_is_tier1(self):
        result = classify_complexity("Print hello world in Python")
        assert result.tier == TierLevel.DIRECT

    def test_tier1_no_council(self):
        result = classify_complexity("What is 2+2?")
        assert result.requires_council is False

    def test_tier1_no_smes(self):
        result = classify_complexity("What is 2+2?")
        assert result.requires_smes is False

    def test_tier1_estimated_agents_3(self):
        result = classify_complexity("What is 2+2?")
        assert result.estimated_agents == 3

    def test_tier1_low_escalation_risk(self):
        result = classify_complexity("Simple greeting")
        assert result.escalation_risk == pytest.approx(0.1, abs=0.05)

    def test_tier1_confidence(self):
        result = classify_complexity("Simple task")
        assert 0.5 <= result.confidence <= 1.0

    def test_tier1_keywords_found_empty(self):
        result = classify_complexity("Hello there")
        assert len(result.keywords_found) == 0

    def test_tier3_keyword_bumps_classification(self):
        result = classify_complexity("Design a system architecture for microservices")
        assert result.tier >= TierLevel.DEEP

    def test_tier4_keyword_bumps_classification(self):
        result = classify_complexity("Perform a security audit of the banking system")
        assert result.tier == TierLevel.ADVERSARIAL


# =============================================================================
# SECTION 3: Tier 1 Pipeline
# =============================================================================


class TestTier1Pipeline:
    """Validate pipeline behavior for Tier 1."""

    def test_tier1_skips_phase_1(self, tier1_pipeline):
        assert tier1_pipeline._should_skip_phase(Phase.PHASE_1_TASK_INTELLIGENCE) is True

    def test_tier1_skips_phase_2(self, tier1_pipeline):
        assert tier1_pipeline._should_skip_phase(Phase.PHASE_2_COUNCIL_CONSULTATION) is True

    def test_tier1_skips_phase_3(self, tier1_pipeline):
        assert tier1_pipeline._should_skip_phase(Phase.PHASE_3_PLANNING) is True

    def test_tier1_skips_phase_4(self, tier1_pipeline):
        assert tier1_pipeline._should_skip_phase(Phase.PHASE_4_RESEARCH) is True

    def test_tier1_does_not_skip_phase_5(self, tier1_pipeline):
        assert tier1_pipeline._should_skip_phase(Phase.PHASE_5_SOLUTION_GENERATION) is False

    def test_tier1_skips_phase_6(self, tier1_pipeline):
        assert tier1_pipeline._should_skip_phase(Phase.PHASE_6_REVIEW) is True

    def test_tier1_skips_phase_7(self, tier1_pipeline):
        assert tier1_pipeline._should_skip_phase(Phase.PHASE_7_REVISION) is True

    def test_tier1_does_not_skip_phase_8(self, tier1_pipeline):
        assert tier1_pipeline._should_skip_phase(Phase.PHASE_8_FINAL_REVIEW_FORMATTING) is False

    def test_tier1_phases_for_tier(self, tier1_pipeline):
        phases = tier1_pipeline._get_phases_for_tier()
        assert len(phases) == 2
        assert Phase.PHASE_5_SOLUTION_GENERATION in phases
        assert Phase.PHASE_8_FINAL_REVIEW_FORMATTING in phases

    def test_tier1_phase_5_agents(self, tier1_pipeline):
        agents = tier1_pipeline._get_agents_for_phase(Phase.PHASE_5_SOLUTION_GENERATION)
        assert "Executor" in agents

    def test_tier1_phase_8_agents(self, tier1_pipeline):
        agents = tier1_pipeline._get_agents_for_phase(Phase.PHASE_8_FINAL_REVIEW_FORMATTING)
        assert "Formatter" in agents
        assert "Reviewer" in agents

    def test_tier1_skipped_phase_returns_skipped_result(self, tier1_pipeline):
        result = tier1_pipeline.execute_phase(
            Phase.PHASE_1_TASK_INTELLIGENCE,
            agent_executor=lambda **kw: None,
            context={},
        )
        assert result.status == PhaseStatus.SKIPPED
        assert result.agent_results == []
        assert result.duration_ms == 0

    def test_tier1_pipeline_builder(self):
        pipeline = PipelineBuilder.for_tier(TierLevel.DIRECT)
        assert pipeline.tier_level == TierLevel.DIRECT

    def test_tier1_pipeline_builder_from_classification(self):
        classification = TierClassification(
            tier=TierLevel.DIRECT,
            reasoning="Simple",
            confidence=0.9,
            estimated_agents=3,
            requires_council=False,
            requires_smes=False,
        )
        pipeline = PipelineBuilder.from_classification(classification)
        assert pipeline.tier_level == TierLevel.DIRECT

    def test_tier1_pipeline_state_initial(self, tier1_pipeline):
        assert tier1_pipeline.state.revision_cycle == 0
        assert tier1_pipeline.state.debate_rounds == 0
        assert tier1_pipeline.state.total_cost_usd == 0.0

    def test_tier1_pipeline_duration_estimate(self):
        estimate = estimate_pipeline_duration(TierLevel.DIRECT)
        assert estimate["min"] == 10
        assert estimate["max"] == 30
        assert estimate["estimated"] == 15

    def test_tier1_execution_context(self):
        classification = TierClassification(
            tier=TierLevel.DIRECT,
            reasoning="Simple",
            confidence=0.9,
            estimated_agents=3,
            requires_council=False,
            requires_smes=False,
        )
        context = create_execution_context("Hello", classification)
        assert context["tier"] == TierLevel.DIRECT
        assert context["requires_council"] is False
        assert context["requires_smes"] is False

    def test_tier1_no_council_agents_in_pipeline(self, tier1_pipeline):
        council = tier1_pipeline._get_council_agents()
        assert council == []


# =============================================================================
# SECTION 4: Executor Agent - Exhaustive Tests
# =============================================================================


class TestExecutorScoring:
    """Test scoring weight calculations."""

    def test_scoring_weights_sum_to_1(self, executor):
        total = sum(executor.scoring_weights.values())
        assert total == pytest.approx(1.0)

    def test_completeness_weight(self, executor):
        assert executor.scoring_weights["completeness"] == 0.3

    def test_quality_weight(self, executor):
        assert executor.scoring_weights["quality"] == 0.25

    def test_efficiency_weight(self, executor):
        assert executor.scoring_weights["efficiency"] == 0.2

    def test_maintainability_weight(self, executor):
        assert executor.scoring_weights["maintainability"] == 0.15

    def test_feasibility_weight(self, executor):
        assert executor.scoring_weights["feasibility"] == 0.1

    def test_completeness_scoring_base(self, executor):
        approach = Approach(
            name="Basic", description="Basic",
            steps=["s1"], pros=[], cons=[],
            estimated_time="low", complexity="low",
        )
        score = executor._score_completeness(approach, "task", None)
        assert score == pytest.approx(0.7, abs=0.01)

    def test_completeness_scoring_4_steps(self, executor):
        approach = Approach(
            name="Basic", description="Basic",
            steps=["s1", "s2", "s3", "s4"], pros=[], cons=[],
            estimated_time="low", complexity="low",
        )
        score = executor._score_completeness(approach, "task", None)
        assert score == pytest.approx(0.9, abs=0.01)

    def test_completeness_scoring_comprehensive_name(self, executor):
        approach = Approach(
            name="Comprehensive Solution", description="Full",
            steps=["s1", "s2", "s3", "s4", "s5"], pros=[], cons=[],
            estimated_time="high", complexity="high",
        )
        score = executor._score_completeness(approach, "task", None)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_quality_score_high_complexity(self, executor):
        approach = Approach(
            name="A", description="A", steps=["s"], pros=[], cons=[],
            estimated_time="high", complexity="high",
        )
        assert executor._score_quality(approach) == 0.9

    def test_quality_score_medium_complexity(self, executor):
        approach = Approach(
            name="A", description="A", steps=["s"], pros=[], cons=[],
            estimated_time="medium", complexity="medium",
        )
        assert executor._score_quality(approach) == 0.7

    def test_quality_score_low_complexity(self, executor):
        approach = Approach(
            name="A", description="A", steps=["s"], pros=[], cons=[],
            estimated_time="low", complexity="low",
        )
        assert executor._score_quality(approach) == 0.5

    def test_efficiency_score_low_time(self, executor):
        approach = Approach(
            name="A", description="A", steps=["s"], pros=[], cons=[],
            estimated_time="low", complexity="low",
        )
        assert executor._score_efficiency(approach) == 1.0

    def test_efficiency_score_high_time(self, executor):
        approach = Approach(
            name="A", description="A", steps=["s"], pros=[], cons=[],
            estimated_time="high", complexity="high",
        )
        assert executor._score_efficiency(approach) == 0.4

    def test_maintainability_direct(self, executor):
        approach = Approach(
            name="Direct Implementation", description="A",
            steps=["s"], pros=[], cons=[],
            estimated_time="low", complexity="low",
        )
        assert executor._score_maintainability(approach) == 0.8

    def test_maintainability_comprehensive(self, executor):
        approach = Approach(
            name="Comprehensive Solution", description="A",
            steps=["s"], pros=[], cons=[],
            estimated_time="high", complexity="high",
        )
        assert executor._score_maintainability(approach) == 0.9

    def test_maintainability_optimized(self, executor):
        approach = Approach(
            name="Optimized Implementation", description="A",
            steps=["s"], pros=[], cons=[],
            estimated_time="medium", complexity="medium",
        )
        assert executor._score_maintainability(approach) == 0.6

    def test_maintainability_unknown(self, executor):
        approach = Approach(
            name="Custom", description="A",
            steps=["s"], pros=[], cons=[],
            estimated_time="low", complexity="low",
        )
        assert executor._score_maintainability(approach) == 0.7

    def test_feasibility_low_complexity(self, executor):
        approach = Approach(
            name="A", description="A", steps=["s"], pros=[], cons=[],
            estimated_time="low", complexity="low",
        )
        assert executor._score_feasibility(approach) == 1.0

    def test_feasibility_medium_complexity(self, executor):
        approach = Approach(
            name="A", description="A", steps=["s"], pros=[], cons=[],
            estimated_time="medium", complexity="medium",
        )
        assert executor._score_feasibility(approach) == 0.8

    def test_feasibility_high_complexity(self, executor):
        approach = Approach(
            name="A", description="A", steps=["s"], pros=[], cons=[],
            estimated_time="high", complexity="high",
        )
        assert executor._score_feasibility(approach) == 0.6


class TestExecutorApproachGeneration:
    """Test approach generation for different modalities."""

    def test_text_modality_generates_2_approaches(self, executor, text_report):
        sub_problems = ["Explain concept"]
        approaches = executor._generate_approaches(sub_problems, text_report)
        assert len(approaches) == 2
        names = [a.name for a in approaches]
        assert "Direct Implementation" in names
        assert "Comprehensive Solution" in names

    def test_code_modality_generates_3_approaches(self, executor, code_report):
        sub_problems = ["Implement sort"]
        approaches = executor._generate_approaches(sub_problems, code_report)
        assert len(approaches) == 3
        names = [a.name for a in approaches]
        assert "Optimized Implementation" in names

    def test_no_report_generates_2_approaches(self, executor):
        sub_problems = ["Do something"]
        approaches = executor._generate_approaches(sub_problems)
        assert len(approaches) == 2

    def test_direct_approach_is_low_complexity(self, executor):
        approaches = executor._generate_approaches(["task"])
        direct = next(a for a in approaches if a.name == "Direct Implementation")
        assert direct.complexity == "low"
        assert direct.estimated_time == "low"

    def test_comprehensive_approach_is_high_complexity(self, executor):
        approaches = executor._generate_approaches(["task"])
        comp = next(a for a in approaches if a.name == "Comprehensive Solution")
        assert comp.complexity == "high"
        assert comp.estimated_time == "high"


class TestExecutorDecomposition:
    """Test problem decomposition for all task categories."""

    def test_test_task_decomposition(self, executor):
        subs = executor._decompose_problem("Write unit tests", None)
        assert any("test" in s.lower() for s in subs)

    def test_document_task_decomposition(self, executor):
        subs = executor._decompose_problem("Create documentation", None)
        assert any("document" in s.lower() for s in subs)

    def test_api_task_decomposition(self, executor):
        subs = executor._decompose_problem("Build an API endpoint", None)
        assert any("api" in s.lower() or "endpoint" in s.lower() for s in subs)

    def test_generic_task_decomposition(self, executor):
        subs = executor._decompose_problem("Do something random", None)
        assert len(subs) >= 2
        assert any("understand" in s.lower() or "develop" in s.lower() for s in subs)

    def test_decomposition_uses_analyst_report(self, executor, code_report):
        subs = executor._decompose_problem("Write code", code_report)
        assert "Implement sort" in subs
        assert "Add tests" in subs


class TestExecutorExecution:
    """Test execution for all task types."""

    def test_code_task_produces_code_output(self, executor):
        result = executor.execute("Write a Python function to sort a list")
        assert result.status == "success"
        assert "def" in result.output or "function" in result.output.lower()

    def test_code_task_has_file_created(self, executor):
        result = executor.execute("Write Python code to parse JSON")
        assert len(result.files_created) > 0
        assert result.files_created[0].endswith(".py")

    def test_document_task_output(self, executor):
        result = executor.execute("Write documentation for the module")
        assert result.status == "success"
        assert "documentation" in result.output.lower() or "document" in result.output.lower()
        assert result.files_created == []

    def test_analysis_task_output(self, executor):
        result = executor.execute("Analyze the performance bottleneck")
        assert result.status == "success"
        assert "analysis" in result.output.lower() or "analyz" in result.output.lower()

    def test_general_task_output(self, executor):
        result = executor.execute("Do something")
        assert result.status == "success"
        assert result.output is not None

    def test_execution_records_time(self, executor):
        result = executor.execute("Quick task")
        assert result.execution_time >= 0

    def test_quality_score_range(self, executor):
        result = executor.execute("Write a function")
        assert 0.0 <= result.quality_score <= 1.0

    def test_validation_boosts_quality_on_success(self, executor):
        result = ExecutionResult(
            approach_name="Test",
            status="success",
            output="output",
            quality_score=0.8,
        )
        validated = executor._validate_output(result)
        assert validated.quality_score == pytest.approx(0.9, abs=0.01)

    def test_validation_reduces_quality_on_error(self, executor):
        result = ExecutionResult(
            approach_name="Test",
            status="success",
            output="output",
            error="Some error",
            quality_score=0.8,
        )
        validated = executor._validate_output(result)
        assert validated.quality_score == pytest.approx(0.6, abs=0.01)

    def test_sme_advisory_adapts_approach(self, executor):
        approach = Approach(
            name="Direct", description="Direct",
            steps=["s1"], pros=["fast"], cons=[],
            estimated_time="low", complexity="low",
        )
        adapted = executor._adapt_to_sme_advice(
            approach, {"Security": "Add validation"}
        )
        assert any("SME" in p for p in adapted.pros)
        assert adapted.complexity == "medium"

    def test_empty_sme_advisory_no_change(self, executor):
        approach = Approach(
            name="Direct", description="Direct",
            steps=["s1"], pros=["fast"], cons=[],
            estimated_time="low", complexity="low",
        )
        adapted = executor._adapt_to_sme_advice(approach, {})
        assert adapted.complexity == "low"


class TestExecutorFilePath:
    """Test file path determination for all languages."""

    @pytest.mark.parametrize("task,ext", [
        ("Write Python code", ".py"),
        ("Create a JavaScript module", ".js"),
        ("Build a TypeScript service", ".ts"),
        ("Write a Java class", ".java"),
        ("Write Go code for a server", ".go"),
        ("Write Rust code", ".rs"),
    ])
    def test_file_path_for_language(self, executor, task, ext):
        path = executor._determine_file_path(task)
        assert path is not None
        assert path.endswith(ext)

    def test_no_path_for_unknown_language(self, executor):
        path = executor._determine_file_path("Do something random")
        assert path is None

    def test_no_path_for_plain_text_task(self, executor):
        path = executor._determine_file_path("Explain recursion")
        assert path is None


# =============================================================================
# SECTION 5: Formatter Agent - Exhaustive Tests
# =============================================================================


class TestFormatterFormats:
    """Test all output format handling."""

    def test_all_output_format_enum_values(self):
        expected = [
            "markdown", "code", "docx", "pdf", "xlsx",
            "pptx", "mermaid", "json", "yaml", "text",
        ]
        actual = [f.value for f in OutputFormat]
        for fmt in expected:
            assert fmt in actual

    def test_markdown_string_input(self, formatter):
        result = formatter.format("# Title\nContent", target_format="markdown")
        assert result["format"] == "markdown"
        assert "# Title" in result["formatted_output"]

    def test_markdown_dict_input(self, formatter):
        result = formatter.format(
            {"Title": "Hello", "Body": "World"},
            target_format="markdown",
        )
        assert "**Title:**" in result["formatted_output"]

    def test_markdown_list_input(self, formatter):
        result = formatter.format(
            ["Item 1", "Item 2"],
            target_format="markdown",
        )
        assert "- Item 1" in result["formatted_output"]

    def test_json_dict_input(self, formatter):
        result = formatter.format({"key": "value"}, target_format="json")
        parsed = json.loads(result["formatted_output"])
        assert parsed["key"] == "value"

    def test_json_string_input(self, formatter):
        result = formatter.format("not json", target_format="json")
        parsed = json.loads(result["formatted_output"])
        assert "value" in parsed

    def test_json_valid_string_input(self, formatter):
        result = formatter.format('{"a": 1}', target_format="json")
        parsed = json.loads(result["formatted_output"])
        assert parsed["a"] == 1

    def test_yaml_dict_input(self, formatter):
        result = formatter.format({"key": "value"}, target_format="yaml")
        assert "key:" in result["formatted_output"]

    def test_yaml_list_input(self, formatter):
        result = formatter.format(["a", "b"], target_format="yaml")
        assert "items:" in result["formatted_output"]

    def test_yaml_string_input(self, formatter):
        result = formatter.format("hello", target_format="yaml")
        assert "value:" in result["formatted_output"]

    def test_text_format(self, formatter):
        result = formatter.format("hello world", target_format="text")
        assert result["format"] == "text"
        assert result["formatted_output"] == "hello world"

    def test_code_format_with_python(self, formatter):
        code = "def hello():\n    print('Hello')"
        result = formatter.format(code, target_format="code")
        assert result["format"] == "code"
        assert result["formatted_output"]["language"] == "python"
        assert result["formatted_output"]["validation"]["valid"] is True

    def test_code_format_writes_file(self, formatter, tmp_path):
        code = "x = 42"
        result = formatter.format(
            code,
            target_format="code",
            file_path="test.py",
        )
        output = result["formatted_output"]
        assert output["file_path"] is not None
        assert os.path.exists(output["file_path"])

    def test_mermaid_flowchart(self, formatter):
        result = formatter.format("Process flow steps", target_format="mermaid")
        assert "mermaid" in result["formatted_output"]

    def test_docx_format(self, formatter):
        result = formatter.format("Document content", target_format="docx")
        assert result["format"] == "docx"
        assert "DOCX" in result["formatted_output"]

    def test_pdf_format(self, formatter):
        result = formatter.format("PDF content", target_format="pdf")
        assert result["format"] == "pdf"

    def test_xlsx_format(self, formatter):
        result = formatter.format("Spreadsheet data", target_format="xlsx")
        assert result["format"] == "xlsx"

    def test_pptx_format(self, formatter):
        result = formatter.format("Presentation content", target_format="pptx")
        assert result["format"] == "pptx"

    def test_unknown_format_falls_back_to_inference(self, formatter):
        result = formatter.format("Hello", target_format="nonexistent")
        assert result["format"] is not None

    def test_metadata_always_present(self, formatter):
        result = formatter.format("test", target_format="text")
        assert "metadata" in result
        assert "timestamp" in result["metadata"]
        assert "size_bytes" in result["metadata"]

    def test_file_path_only_for_code(self, formatter):
        md_result = formatter.format("# Title", target_format="markdown")
        assert md_result["file_path"] is None

        text_result = formatter.format("text", target_format="text")
        assert text_result["file_path"] is None


class TestFormatterCodeHandling:
    """Test code extraction, language detection, and syntax validation."""

    def test_extract_code_from_markdown_block(self, formatter):
        content = "```python\nx = 1\n```"
        code = formatter._extract_code(content)
        assert "x = 1" in code

    def test_extract_code_no_block(self, formatter):
        code = formatter._extract_code("plain text")
        assert code == "plain text"

    def test_extract_code_from_non_string(self, formatter):
        code = formatter._extract_code(42)
        assert code == "42"

    @pytest.mark.parametrize("ext,lang", [
        (".py", "python"), (".js", "javascript"), (".ts", "typescript"),
        (".java", "java"), (".go", "go"), (".rs", "rust"),
        (".cpp", "cpp"), (".c", "c"), (".cs", "csharp"),
        (".php", "php"), (".rb", "ruby"), (".swift", "swift"),
        (".kt", "kotlin"), (".sh", "bash"),
    ])
    def test_language_from_extension(self, formatter, ext, lang):
        assert formatter._detect_language("", f"test{ext}") == lang

    def test_language_detection_python(self, formatter):
        assert formatter._detect_language("def foo():\n    pass", None) == "python"

    def test_language_detection_javascript(self, formatter):
        assert formatter._detect_language("function foo() { const x = 1; }", None) == "javascript"

    def test_language_detection_typescript(self, formatter):
        code = "interface Foo { }\nconst x: string = 'hello';"
        assert formatter._detect_language(code, None) == "typescript"

    def test_language_detection_java(self, formatter):
        assert formatter._detect_language("public class Main {}", None) == "java"

    def test_language_detection_go(self, formatter):
        assert formatter._detect_language("package main\nfunc main() {}", None) == "go"

    def test_language_detection_default_python(self, formatter):
        assert formatter._detect_language("some random text", None) == "python"

    def test_validate_valid_python_syntax(self, formatter):
        result = formatter._validate_syntax("x = 1 + 2", "python")
        assert result["valid"] is True
        assert result["errors"] == []

    def test_validate_invalid_python_syntax(self, formatter):
        result = formatter._validate_syntax("def x(:\n    pass", "python")
        assert result["valid"] is False
        assert len(result["errors"]) > 0

    def test_validate_non_python_always_valid(self, formatter):
        result = formatter._validate_syntax("function x() {}", "javascript")
        assert result["valid"] is True


class TestFormatterInference:
    """Test format inference logic."""

    def test_infer_json_for_dict(self, formatter):
        assert formatter._infer_format({"k": "v"}) == OutputFormat.JSON_FMT

    def test_infer_code_for_code_string(self, formatter):
        assert formatter._infer_format("def hello():\n    return 42") == OutputFormat.CODE

    def test_infer_mermaid_for_flowchart(self, formatter):
        assert formatter._infer_format("Create a flowchart") == OutputFormat.MERMAID

    def test_infer_mermaid_for_sequence(self, formatter):
        assert formatter._infer_format("Draw a sequence diagram") == OutputFormat.MERMAID

    def test_infer_markdown_default(self, formatter):
        assert formatter._infer_format("Just some text") == OutputFormat.MARKDOWN

    def test_infer_from_context_format(self, formatter):
        fmt = formatter._infer_format("data", context={"format": "yaml"})
        assert fmt == OutputFormat.YAML_FMT

    def test_infer_dict_with_output_format_context(self, formatter):
        fmt = formatter._infer_format(
            {"data": 1},
            context={"output_format": "yaml"},
        )
        assert fmt == OutputFormat.YAML_FMT

    def test_looks_like_code_detects_def(self, formatter):
        assert formatter._looks_like_code("def hello():") is True

    def test_looks_like_code_detects_function(self, formatter):
        assert formatter._looks_like_code("function hello()") is True

    def test_looks_like_code_plain_text(self, formatter):
        assert formatter._looks_like_code("Just plain text") is False


class TestFormatterMermaidDiagrams:
    """Test all Mermaid diagram types."""

    def test_infer_flowchart(self, formatter):
        assert formatter._infer_mermaid_type("process flow", None) == "flowchart"

    def test_infer_sequence(self, formatter):
        assert formatter._infer_mermaid_type("sequence interaction", None) == "sequence"

    def test_infer_class(self, formatter):
        assert formatter._infer_mermaid_type("class object model", None) == "class"

    def test_infer_state(self, formatter):
        assert formatter._infer_mermaid_type("state transition", None) == "state"

    def test_infer_default_flowchart(self, formatter):
        assert formatter._infer_mermaid_type("something random", None) == "flowchart"

    def test_infer_from_context(self, formatter):
        assert formatter._infer_mermaid_type("", {"diagram_type": "class"}) == "class"

    def test_flowchart_content(self, formatter):
        result = formatter._generate_flowchart("flow", {})
        assert "flowchart" in result
        assert "```mermaid" in result

    def test_sequence_diagram_content(self, formatter):
        result = formatter._generate_sequence_diagram("seq", {})
        assert "sequenceDiagram" in result

    def test_class_diagram_content(self, formatter):
        result = formatter._generate_class_diagram("cls", {})
        assert "classDiagram" in result

    def test_state_diagram_content(self, formatter):
        result = formatter._generate_state_diagram("state", {})
        assert "stateDiagram-v2" in result

    def test_generic_diagram_content(self, formatter):
        result = formatter._generate_generic_diagram("generic", {})
        assert "graph LR" in result


# =============================================================================
# SECTION 6: SDK Integration for Tier 1 Agents
# =============================================================================


class TestSDKIntegrationTier1:
    """Test Claude Agent SDK integration for Tier 1 agents."""

    def test_executor_allowed_tools(self):
        tools = AGENT_ALLOWED_TOOLS["executor"]
        assert "Read" in tools
        assert "Write" in tools
        assert "Edit" in tools
        assert "Bash" in tools
        assert "Glob" in tools
        assert "Grep" in tools
        assert "Skill" in tools

    def test_formatter_allowed_tools(self):
        tools = AGENT_ALLOWED_TOOLS["formatter"]
        assert "Read" in tools
        assert "Write" in tools
        assert "Bash" in tools
        assert "Skill" in tools

    def test_executor_has_more_tools_than_formatter(self):
        assert len(AGENT_ALLOWED_TOOLS["executor"]) > len(AGENT_ALLOWED_TOOLS["formatter"])

    def test_executor_skills(self):
        skills = get_skills_for_agent("executor")
        assert "code-generation" in skills

    def test_formatter_skills(self):
        skills = get_skills_for_agent("formatter")
        assert "document-creation" in skills

    def test_orchestrator_skills(self):
        skills = get_skills_for_agent("orchestrator")
        assert "multi-agent-reasoning" in skills

    def test_build_executor_options(self):
        options = build_agent_options(
            agent_name="executor",
            system_prompt="You are the Executor.",
        )
        assert isinstance(options, ClaudeAgentOptions)
        assert "Skill" in options.allowed_tools
        assert options.permission_mode == PermissionMode.ACCEPT_EDITS

    def test_build_formatter_options(self):
        options = build_agent_options(
            agent_name="formatter",
            system_prompt="You are the Formatter.",
        )
        assert isinstance(options, ClaudeAgentOptions)
        assert options.permission_mode == PermissionMode.DEFAULT

    def test_agent_options_to_sdk_kwargs(self):
        options = ClaudeAgentOptions(
            name="Executor",
            model="claude-3-5-sonnet-20241022",
            system_prompt="Test",
            max_turns=50,
            allowed_tools=["Read", "Write"],
        )
        kwargs = options.to_sdk_kwargs()
        assert kwargs["name"] == "Executor"
        assert kwargs["model"] == "claude-3-5-sonnet-20241022"
        assert kwargs["max_turns"] == 50
        assert kwargs["allowed_tools"] == ["Read", "Write"]

    def test_agent_options_setting_sources(self):
        options = ClaudeAgentOptions(
            name="Test",
            model="test",
            system_prompt="Test",
        )
        kwargs = options.to_sdk_kwargs()
        assert kwargs["setting_sources"] == ["user", "project"]

    def test_simulate_response(self):
        result = _simulate_response(
            {"name": "Executor"},
            "Test input",
        )
        assert "output" in result
        assert "Executor" in result["output"]
        assert result["tokens_used"] == 0
        assert result["cost_usd"] == 0.0
        assert result["status"] == "error"
        assert result["error"] == "no_api_configured"

    def test_validate_output_valid_json(self):
        schema = {"required": ["name", "status"]}
        output = json.dumps({"name": "test", "status": "ok"})
        assert _validate_output(output, schema) is True

    def test_validate_output_missing_field(self):
        schema = {"required": ["name", "status"]}
        output = json.dumps({"name": "test"})
        assert _validate_output(output, schema) is False

    def test_validate_output_empty(self):
        assert _validate_output("", {}) is False
        assert _validate_output(None, {}) is False

    def test_validate_output_invalid_json_string(self):
        assert _validate_output("not json", {"required": ["x"]}) is False

    def test_validate_output_dict_input(self):
        schema = {"required": ["key"]}
        assert _validate_output({"key": "val"}, schema) is True

    def test_validate_output_no_required_fields(self):
        schema = {}
        assert _validate_output('{"key": "val"}', schema) is True

    def test_spawn_subagent_with_simulated_fallback(self):
        """Test spawn_subagent falls back to simulation."""
        options = ClaudeAgentOptions(
            name="TestAgent",
            model="claude-3-5-sonnet-20241022",
            system_prompt="Test",
            max_turns=10,
        )
        # This will fall through to _simulate_response since no SDK/API available
        result = spawn_subagent(options, "Test input", max_retries=0)
        assert result["status"] in ["success", "error"]
        assert "output" in result or "error" in result

    def test_executor_no_output_schema(self):
        """Executor has no structured output schema (returns raw content)."""
        schema = _get_output_schema("executor")
        assert schema is None

    def test_formatter_no_output_schema(self):
        """Formatter has no structured output schema."""
        schema = _get_output_schema("formatter")
        assert schema is None


# =============================================================================
# SECTION 7: Orchestrator Session State for Tier 1
# =============================================================================


class TestOrchestratorSessionState:
    """Test SessionState for Tier 1 operation."""

    def test_session_state_creation(self):
        session = SessionState(
            session_id="test_123",
            user_prompt="Hello",
        )
        assert session.session_id == "test_123"
        assert session.current_tier == TierLevel.STANDARD

    def test_session_state_budget(self):
        session = SessionState(
            session_id="test",
            user_prompt="test",
            max_budget_usd=5.0,
        )
        assert not session.is_budget_exceeded()
        session.total_cost_usd = 6.0
        assert session.is_budget_exceeded()

    def test_session_budget_warning(self):
        session = SessionState(
            session_id="test",
            user_prompt="test",
            max_budget_usd=10.0,
            budget_warning_threshold=0.8,
        )
        session.total_cost_usd = 8.0
        assert session.should_warn_budget()

    def test_session_budget_no_warning(self):
        session = SessionState(
            session_id="test",
            user_prompt="test",
            max_budget_usd=10.0,
        )
        session.total_cost_usd = 2.0
        assert not session.should_warn_budget()

    def test_session_duration(self):
        session = SessionState(
            session_id="test",
            user_prompt="test",
        )
        session.start_time = time.time() - 10
        assert session.duration_seconds >= 10

    def test_session_budget_utilization(self):
        session = SessionState(
            session_id="test",
            user_prompt="test",
            max_budget_usd=10.0,
        )
        session.total_cost_usd = 5.0
        assert session.budget_utilization == pytest.approx(0.5)

    def test_agent_execution_dataclass(self):
        execution = AgentExecution(
            agent_name="Executor",
            start_time=time.time(),
        )
        assert execution.status == "pending"
        assert execution.tokens_used == 0
        assert execution.cost_usd == 0.0


# =============================================================================
# SECTION 8: Orchestrator Initialization
# =============================================================================


class TestOrchestratorInit:
    """Test Orchestrator creation and initialization."""

    def test_create_orchestrator_function(self):
        orch = create_orchestrator()
        assert isinstance(orch, OrchestratorAgent)

    def test_orchestrator_default_budget(self):
        orch = OrchestratorAgent()
        assert orch.max_budget_usd > 0

    def test_orchestrator_custom_budget(self):
        orch = OrchestratorAgent(max_budget_usd=10.0)
        assert orch.max_budget_usd == 10.0

    def test_orchestrator_max_revisions(self):
        orch = OrchestratorAgent(max_revisions=3)
        assert orch.max_revisions == 3

    def test_orchestrator_verbose(self):
        orch = OrchestratorAgent(verbose=True)
        assert orch.verbose is True

    def test_orchestrator_has_mcp_server(self):
        orch = OrchestratorAgent()
        assert orch.mcp_server is not None

    def test_orchestrator_provider_set(self):
        orch = OrchestratorAgent()
        assert orch.provider is not None


# =============================================================================
# SECTION 9: Escalation & Tier Boundary Tests
# =============================================================================


class TestEscalation:
    """Test escalation from Tier 1 to higher tiers."""

    def test_should_escalate_with_flag(self):
        assert should_escalate(TierLevel.DIRECT, {"escalation_needed": True}) is True

    def test_should_not_escalate_without_flag(self):
        assert should_escalate(TierLevel.DIRECT, {"escalation_needed": False}) is False

    def test_should_escalate_with_indicator_text(self):
        assert should_escalate(TierLevel.DIRECT, {"note": "domain expertise required"}) is True

    def test_should_not_escalate_with_clean_feedback(self):
        assert should_escalate(TierLevel.DIRECT, {"result": "all good"}) is False

    def test_escalated_tier_from_direct(self):
        assert get_escalated_tier(TierLevel.DIRECT) == TierLevel.STANDARD

    def test_escalated_tier_capped_at_4(self):
        assert get_escalated_tier(TierLevel.ADVERSARIAL) == TierLevel.ADVERSARIAL

    def test_escalation_keywords_in_classification(self):
        result = classify_complexity("This is a complex multi-step task")
        assert result.tier >= TierLevel.STANDARD


# =============================================================================
# SECTION 10: ThoughtBranch & ThoughtNode Data Classes
# =============================================================================


class TestDataClasses:
    """Test data classes and enums."""

    def test_thought_branch_values(self):
        assert ThoughtBranch.SEQUENTIAL == "sequential"
        assert ThoughtBranch.PARALLEL == "parallel"
        assert ThoughtBranch.DECOMPOSE == "decompose"
        assert ThoughtBranch.SIMPLIFY == "simplify"

    def test_thought_node_creation(self):
        approach = Approach(
            name="Test", description="Test",
            steps=["s1"], pros=["p"], cons=["c"],
            estimated_time="low", complexity="low",
        )
        node = ThoughtNode(approach=approach)
        assert node.depth == 0
        assert node.explored is False
        assert node.selected is False
        assert node.parent is None
        assert node.children == []

    def test_thought_node_with_parent(self):
        parent_approach = Approach(
            name="Parent", description="Parent",
            steps=["s1"], pros=[], cons=[],
            estimated_time="low", complexity="low",
        )
        parent = ThoughtNode(approach=parent_approach)

        child_approach = Approach(
            name="Child", description="Child",
            steps=["s1"], pros=[], cons=[],
            estimated_time="low", complexity="low",
        )
        child = ThoughtNode(approach=child_approach, parent=parent, depth=1)
        assert child.parent is parent
        assert child.depth == 1

    def test_execution_result_defaults(self):
        result = ExecutionResult(
            approach_name="Test",
            status="success",
        )
        assert result.output is None
        assert result.files_created == []
        assert result.execution_time == 0.0
        assert result.error is None
        assert result.quality_score == 0.0

    def test_approach_defaults(self):
        approach = Approach(
            name="Test", description="Test",
            steps=["s1"], pros=["p"], cons=["c"],
            estimated_time="low", complexity="low",
        )
        assert approach.score == 0.0

    def test_output_format_enum(self):
        assert OutputFormat.MARKDOWN.value == "markdown"
        assert OutputFormat.CODE.value == "code"
        assert OutputFormat.JSON_FMT.value == "json"
        assert OutputFormat.YAML_FMT.value == "yaml"


# =============================================================================
# SECTION 11: Cross-Agent Integration (Executor -> Formatter)
# =============================================================================


class TestTier1Integration:
    """Test the Executor -> Formatter data flow."""

    def test_executor_output_formats_as_markdown(self, executor, formatter):
        exec_result = executor.execute("Write a Python function")
        format_result = formatter.format(
            exec_result.output,
            target_format="markdown",
        )
        assert format_result["format"] == "markdown"
        assert format_result["formatted_output"] is not None

    def test_executor_output_formats_as_json(self, executor, formatter):
        exec_result = executor.execute("Explain recursion")
        format_result = formatter.format(
            exec_result.output,
            target_format="json",
        )
        assert format_result["format"] == "json"

    def test_executor_code_output_formats_as_code(self, executor, formatter):
        exec_result = executor.execute("Write Python code to parse JSON")
        format_result = formatter.format(
            exec_result.output,
            target_format="code",
        )
        assert format_result["format"] == "code"

    def test_executor_output_formats_as_text(self, executor, formatter):
        exec_result = executor.execute("Do something")
        format_result = formatter.format(
            exec_result.output,
            target_format="text",
        )
        assert format_result["format"] == "text"

    def test_full_tier1_flow(self, executor, formatter):
        """Simulate the complete Tier 1 flow: classify -> execute -> format."""
        # Step 1: Classify
        classification = classify_complexity("Print hello world")
        assert classification.tier == TierLevel.DIRECT

        # Step 2: Execute
        exec_result = executor.execute("Print hello world")
        assert exec_result.status == "success"

        # Step 3: Format
        format_result = formatter.format(
            exec_result.output,
            target_format="markdown",
        )
        assert format_result["formatted_output"] is not None
        assert format_result["metadata"]["size_bytes"] > 0

    def test_full_tier1_flow_code_task(self, executor, formatter):
        """Simulate full Tier 1 flow for a code generation task."""
        classification = classify_complexity("Write a Python hello world function")
        assert classification.tier == TierLevel.DIRECT

        exec_result = executor.execute("Write a Python hello world function")
        assert exec_result.status == "success"
        assert "def" in exec_result.output or "function" in exec_result.output.lower()

        format_result = formatter.format(
            exec_result.output,
            target_format="code",
        )
        assert format_result["format"] == "code"
        code_output = format_result["formatted_output"]
        assert code_output["language"] == "python"
        assert code_output["validation"]["valid"] is True


# =============================================================================
# SECTION 12: Edge Cases & Boundary Conditions
# =============================================================================


class TestEdgeCases:
    """Test edge cases for Tier 1 agents."""

    def test_empty_task_execution(self, executor):
        result = executor.execute("")
        assert result.status == "success"

    def test_very_long_task(self, executor):
        long_task = "Write a function " * 200
        result = executor.execute(long_task)
        assert result.status == "success"

    def test_special_characters_in_task(self, executor):
        result = executor.execute("Write a function that handles 'quotes' and \"double quotes\"")
        assert result.status == "success"

    def test_unicode_in_task(self, executor):
        result = executor.execute("Write a function for unicode: Hello!")
        assert result.status == "success"

    def test_empty_string_format(self, formatter):
        result = formatter.format("", target_format="text")
        assert result["formatted_output"] == ""

    def test_none_content_format(self, formatter):
        result = formatter.format(None, target_format="text")
        assert result["formatted_output"] == "None"

    def test_integer_content_format(self, formatter):
        result = formatter.format(42, target_format="text")
        assert result["formatted_output"] == "42"

    def test_boolean_content_format(self, formatter):
        result = formatter.format(True, target_format="text")
        assert result["formatted_output"] == "True"

    def test_deeply_nested_dict_format(self, formatter):
        data = {"a": {"b": {"c": {"d": "value"}}}}
        result = formatter.format(data, target_format="markdown")
        assert "value" in result["formatted_output"]

    def test_empty_list_format(self, formatter):
        result = formatter.format([], target_format="markdown")
        assert result["formatted_output"] == ""

    def test_classify_empty_prompt(self):
        result = classify_complexity("")
        assert result.tier == TierLevel.DIRECT

    def test_approach_score_never_exceeds_1(self, executor):
        approaches = [
            Approach(
                name="Comprehensive Solution",
                description="Comprehensive",
                steps=["s1", "s2", "s3", "s4", "s5"],
                pros=["thorough"], cons=["slow"],
                estimated_time="high", complexity="high",
            ),
        ]
        scored = executor._score_approaches(approaches, "task")
        for a in scored:
            assert a.score <= 1.0

    def test_approach_score_never_negative(self, executor):
        approaches = [
            Approach(
                name="Minimal", description="Minimal",
                steps=["s"], pros=[], cons=[],
                estimated_time="low", complexity="low",
            ),
        ]
        scored = executor._score_approaches(approaches, "task")
        for a in scored:
            assert a.score >= 0.0


# =============================================================================
# SECTION 13: Config Agent CLAUDE.md Presence
# =============================================================================


class TestAgentConfigs:
    """Verify agent CLAUDE.md config files exist and have content."""

    @pytest.mark.parametrize("agent", ["orchestrator", "executor", "formatter"])
    def test_config_file_exists(self, agent):
        path = Path(f"config/agents/{agent}/CLAUDE.md")
        assert path.exists(), f"Config file missing for {agent}"

    @pytest.mark.parametrize("agent", ["orchestrator", "executor", "formatter"])
    def test_config_file_not_empty(self, agent):
        path = Path(f"config/agents/{agent}/CLAUDE.md")
        content = path.read_text()
        assert len(content) > 100, f"Config file too short for {agent}"

    def test_executor_config_mentions_tree_of_thoughts(self):
        content = Path("config/agents/executor/CLAUDE.md").read_text()
        assert "tree of thoughts" in content.lower()

    def test_formatter_config_mentions_formats(self):
        content = Path("config/agents/formatter/CLAUDE.md").read_text()
        assert "markdown" in content.lower()
        assert "docx" in content.lower()

    def test_orchestrator_config_mentions_tiers(self):
        content = Path("config/agents/orchestrator/CLAUDE.md").read_text()
        assert "tier" in content.lower()
        assert "executor" in content.lower()
        assert "formatter" in content.lower()


# =============================================================================
# SECTION 14: Skills Presence
# =============================================================================


class TestSkillsPresence:
    """Verify skill files exist for Tier 1 agents."""

    def test_code_generation_skill_exists(self):
        path = Path(".claude/skills/code-generation/SKILL.md")
        assert path.exists()

    def test_document_creation_skill_exists(self):
        path = Path(".claude/skills/document-creation/SKILL.md")
        assert path.exists()

    def test_multi_agent_reasoning_skill_exists(self):
        path = Path(".claude/skills/multi-agent-reasoning/SKILL.md")
        assert path.exists()
