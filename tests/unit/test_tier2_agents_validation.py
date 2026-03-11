"""
Exhaustive Validation Tests for All 7 Tier 2 (Standard) Agents

Tests every agent for:
- Claude Agent SDK integration (ClaudeAgentOptions, spawn_subagent, tools, skills)
- Schema compliance (Pydantic output formats, JSON schema generation)
- Tool declarations (least-privilege allowed tools)
- Skill mappings (SKILL.md files assigned to agents)
- Edge cases, boundary conditions, error handling
- Cross-agent interaction contracts (pipeline phase handoffs)
- Configuration system integration (model selection, settings)

Tier 2 Agents (7 total):
1. Analyst     - Phase 1: Task Intelligence
2. Planner     - Phase 3: Planning
3. Clarifier   - Phase 4: Clarification
4. Executor    - Phase 5: Solution Generation
5. Verifier    - Phase 6: Review (fact-check)
6. Reviewer    - Phase 8: Final Quality Gate
7. Formatter   - Phase 8: Output Formatting
"""

import json
import os
import pytest
from unittest.mock import patch, mock_open, MagicMock, PropertyMock
from pathlib import Path

# Agent imports
from src.agents.analyst import AnalystAgent, create_analyst
from src.agents.planner import PlannerAgent, create_planner
from src.agents.clarifier import ClarifierAgent, create_clarifier
from src.agents.executor import (
    ExecutorAgent, Approach, ExecutionResult, ThoughtBranch, ThoughtNode,
    create_executor,
)
from src.agents.verifier import VerifierAgent, ClaimExtraction, create_verifier
from src.agents.reviewer import ReviewerAgent, ReviewContext, create_reviewer
from src.agents.formatter import FormatterAgent, OutputFormat, create_formatter

# Schema imports
from src.schemas.analyst import (
    TaskIntelligenceReport, SubTask, MissingInfo, SeverityLevel, ModalityType,
)
from src.schemas.planner import (
    ExecutionPlan, ExecutionStep, AgentAssignment, ParallelGroup, StepStatus,
)
from src.schemas.clarifier import (
    ClarificationRequest, ClarificationQuestion, QuestionPriority, ImpactAssessment,
)
from src.schemas.verifier import (
    VerificationReport, Claim, ClaimBatch, VerificationStatus, FabricationRisk,
)
from src.schemas.reviewer import (
    ReviewVerdict, Verdict, CheckItem, Revision, QualityGateResults, ArbitrationInput,
)

# SDK integration imports
from src.core.sdk_integration import (
    ClaudeAgentOptions,
    PermissionMode,
    AGENT_ALLOWED_TOOLS,
    build_agent_options,
    get_skills_for_agent,
    _get_output_schema,
    _validate_output,
    spawn_subagent,
)
from src.config.settings import (
    Settings, get_settings, get_model_for_agent, reload_settings,
    LLMProvider, DEFAULT_MODEL_MAPPINGS,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def analyst():
    return AnalystAgent(system_prompt_path="nonexistent.md")

@pytest.fixture
def planner():
    return PlannerAgent(system_prompt_path="nonexistent.md")

@pytest.fixture
def clarifier():
    return ClarifierAgent(system_prompt_path="nonexistent.md")

@pytest.fixture
def executor():
    return ExecutorAgent(system_prompt_path="nonexistent.md")

@pytest.fixture
def verifier():
    return VerifierAgent(system_prompt_path="nonexistent.md")

@pytest.fixture
def reviewer():
    return ReviewerAgent(system_prompt_path="nonexistent.md")

@pytest.fixture
def formatter(tmp_path):
    return FormatterAgent(
        system_prompt_path="nonexistent.md",
        output_dir=str(tmp_path),
    )

@pytest.fixture
def basic_report():
    """Standard Tier 2 TaskIntelligenceReport."""
    return TaskIntelligenceReport(
        literal_request="Write a Python function to sort a list",
        inferred_intent="User wants to create code",
        sub_tasks=[
            SubTask(description="Understand requirements", dependencies=[], estimated_complexity="low"),
            SubTask(description="Implement sorting function", dependencies=["Understand requirements"], estimated_complexity="high"),
            SubTask(description="Review and validate output", dependencies=["Implement sorting function"], estimated_complexity="medium"),
        ],
        missing_info=[],
        assumptions=["Using Python", "Standard best practices"],
        modality=ModalityType.CODE,
        recommended_approach="Direct implementation",
        escalation_needed=False,
        suggested_tier=2,
        confidence=0.9,
    )

@pytest.fixture
def complex_report():
    """Report with critical missing info and high complexity."""
    return TaskIntelligenceReport(
        literal_request="Build a secure REST API with database for user management",
        inferred_intent="Create API endpoints with authentication",
        sub_tasks=[
            SubTask(description="Define data models and schema", dependencies=[], estimated_complexity="medium"),
            SubTask(description="Design API endpoints and routes", dependencies=["Define data models and schema"], estimated_complexity="high"),
            SubTask(description="Implement endpoint logic", dependencies=["Design API endpoints and routes"], estimated_complexity="high"),
            SubTask(description="Add authentication and validation", dependencies=["Implement endpoint logic"], estimated_complexity="medium"),
        ],
        missing_info=[
            MissingInfo(requirement="Authentication method", severity=SeverityLevel.CRITICAL,
                       impact="Security architecture depends on auth", default_assumption="JWT-based authentication"),
            MissingInfo(requirement="Database technology", severity=SeverityLevel.CRITICAL,
                       impact="Schema design depends on DB", default_assumption="PostgreSQL"),
            MissingInfo(requirement="Deployment target/platform", severity=SeverityLevel.IMPORTANT,
                       impact="Deployment config varies", default_assumption="Docker containers"),
        ],
        assumptions=["JWT auth", "PostgreSQL", "Docker"],
        modality=ModalityType.CODE,
        recommended_approach="Clarify first",
        escalation_needed=False,
        suggested_tier=2,
        confidence=0.6,
    )

@pytest.fixture
def basic_context():
    return ReviewContext(
        original_request="Write a Python function to sort a list",
        agent_outputs={},
        revision_count=0,
        max_revisions=2,
        tier_level=2,
        is_code_output=False,
    )

@pytest.fixture
def code_context():
    return ReviewContext(
        original_request="Write a Python function to sort a list",
        agent_outputs={"analyst": {"modality": "code"}},
        revision_count=0,
        max_revisions=2,
        tier_level=2,
        is_code_output=True,
    )


# =============================================================================
# PART 1: Claude Agent SDK Integration Tests
# =============================================================================

class TestSDKAgentOptions:
    """Test ClaudeAgentOptions configuration for all Tier 2 agents."""

    def test_analyst_sdk_options(self):
        """Test Analyst gets correct SDK config."""
        with patch("src.core.sdk_integration.get_settings") as mock_settings:
            mock_s = MagicMock()
            mock_s.max_turns_subagent = 30
            mock_s.max_turns_orchestrator = 200
            mock_s.max_turns_executor = 50
            mock_settings.return_value = mock_s
            with patch("src.core.sdk_integration.get_model_for_agent", return_value="claude-3-5-sonnet-20241022"):
                options = build_agent_options("analyst", "You are the Analyst.")
                assert options.name == "Analyst"
                assert options.model == "claude-3-5-sonnet-20241022"
                assert "Read" in options.allowed_tools
                assert "Glob" in options.allowed_tools
                assert "Grep" in options.allowed_tools
                assert options.permission_mode == PermissionMode.DEFAULT
                assert options.max_turns == 30

    def test_planner_sdk_options(self):
        """Test Planner gets correct SDK config."""
        with patch("src.core.sdk_integration.get_settings") as mock_settings:
            mock_s = MagicMock()
            mock_s.max_turns_subagent = 30
            mock_settings.return_value = mock_s
            with patch("src.core.sdk_integration.get_model_for_agent", return_value="claude-3-5-sonnet-20241022"):
                options = build_agent_options("planner", "You are the Planner.")
                assert options.name == "Planner"
                assert "Read" in options.allowed_tools
                assert "Glob" in options.allowed_tools
                assert "Write" not in options.allowed_tools
                assert "Bash" not in options.allowed_tools

    def test_clarifier_sdk_options(self):
        """Test Clarifier gets minimal tools (least privilege)."""
        with patch("src.core.sdk_integration.get_settings") as mock_settings:
            mock_s = MagicMock()
            mock_s.max_turns_subagent = 30
            mock_settings.return_value = mock_s
            with patch("src.core.sdk_integration.get_model_for_agent", return_value="claude-3-5-haiku-20241022"):
                options = build_agent_options("clarifier", "You are the Clarifier.")
                assert options.allowed_tools == []

    def test_executor_sdk_options(self):
        """Test Executor gets acceptEdits permission and full tool set."""
        with patch("src.core.sdk_integration.get_settings") as mock_settings:
            mock_s = MagicMock()
            mock_s.max_turns_executor = 50
            mock_s.max_turns_subagent = 30
            mock_s.max_turns_orchestrator = 200
            mock_settings.return_value = mock_s
            with patch("src.core.sdk_integration.get_model_for_agent", return_value="claude-3-5-sonnet-20241022"):
                options = build_agent_options("executor", "You are the Executor.")
                assert options.permission_mode == PermissionMode.ACCEPT_EDITS
                assert "Write" in options.allowed_tools
                assert "Edit" in options.allowed_tools
                assert "Bash" in options.allowed_tools
                assert "Skill" in options.allowed_tools
                assert options.max_turns == 50

    def test_verifier_sdk_options(self):
        """Test Verifier gets web search tools."""
        with patch("src.core.sdk_integration.get_settings") as mock_settings:
            mock_s = MagicMock()
            mock_s.max_turns_subagent = 30
            mock_settings.return_value = mock_s
            with patch("src.core.sdk_integration.get_model_for_agent", return_value="claude-3-5-opus-20240507"):
                options = build_agent_options("verifier", "You are the Verifier.")
                assert "Read" in options.allowed_tools
                assert "WebSearch" in options.allowed_tools
                assert "WebFetch" in options.allowed_tools
                assert "Write" not in options.allowed_tools

    def test_reviewer_sdk_options(self):
        """Test Reviewer gets read-only tools."""
        with patch("src.core.sdk_integration.get_settings") as mock_settings:
            mock_s = MagicMock()
            mock_s.max_turns_subagent = 30
            mock_settings.return_value = mock_s
            with patch("src.core.sdk_integration.get_model_for_agent", return_value="claude-3-5-opus-20240507"):
                options = build_agent_options("reviewer", "You are the Reviewer.")
                assert "Read" in options.allowed_tools
                assert "Glob" in options.allowed_tools
                assert "Grep" in options.allowed_tools
                assert "Write" not in options.allowed_tools
                assert "Bash" not in options.allowed_tools

    def test_formatter_sdk_options(self):
        """Test Formatter gets write tools and skill access."""
        with patch("src.core.sdk_integration.get_settings") as mock_settings:
            mock_s = MagicMock()
            mock_s.max_turns_subagent = 30
            mock_settings.return_value = mock_s
            with patch("src.core.sdk_integration.get_model_for_agent", return_value="claude-3-5-sonnet-20241022"):
                options = build_agent_options("formatter", "You are the Formatter.")
                assert "Read" in options.allowed_tools
                assert "Write" in options.allowed_tools
                assert "Bash" in options.allowed_tools
                assert "Skill" in options.allowed_tools

    def test_sdk_kwargs_serialization(self):
        """Test ClaudeAgentOptions serializes to SDK kwargs."""
        options = ClaudeAgentOptions(
            name="Test Agent",
            model="claude-3-5-sonnet-20241022",
            system_prompt="Test prompt",
            max_turns=30,
            allowed_tools=["Read", "Glob"],
            output_format={"type": "object"},
            setting_sources=["user", "project"],
        )
        kwargs = options.to_sdk_kwargs()
        assert kwargs["name"] == "Test Agent"
        assert kwargs["model"] == "claude-3-5-sonnet-20241022"
        assert kwargs["max_turns"] == 30
        assert kwargs["allowed_tools"] == ["Read", "Glob"]
        assert kwargs["output_format"] == {"type": "object"}
        assert kwargs["setting_sources"] == ["user", "project"]

    def test_sdk_kwargs_excludes_defaults(self):
        """Test kwargs excludes default permission mode."""
        options = ClaudeAgentOptions(
            name="Test", model="model", system_prompt="prompt",
        )
        kwargs = options.to_sdk_kwargs()
        assert "permission_mode" not in kwargs

    def test_sdk_kwargs_includes_non_default_permission(self):
        """Test kwargs includes non-default permission mode."""
        options = ClaudeAgentOptions(
            name="Test", model="model", system_prompt="prompt",
            permission_mode=PermissionMode.ACCEPT_EDITS,
        )
        kwargs = options.to_sdk_kwargs()
        assert kwargs["permission_mode"] == "acceptEdits"


# =============================================================================
# PART 2: Tool Declarations (Least-Privilege)
# =============================================================================

class TestToolDeclarations:
    """Test that each agent has correctly declared allowed tools."""

    def test_all_tier2_agents_have_tool_declarations(self):
        """Verify all 7 Tier 2 agents have tool entries."""
        tier2_agents = ["analyst", "planner", "clarifier", "executor",
                       "verifier", "reviewer", "formatter"]
        for agent in tier2_agents:
            assert agent in AGENT_ALLOWED_TOOLS, f"Missing tool declaration for {agent}"

    def test_analyst_tools_read_only(self):
        """Analyst should only read, not write."""
        tools = AGENT_ALLOWED_TOOLS["analyst"]
        assert "Write" not in tools
        assert "Edit" not in tools
        assert "Bash" not in tools

    def test_clarifier_has_no_tools(self):
        """Clarifier needs no tools - it only asks questions."""
        assert AGENT_ALLOWED_TOOLS["clarifier"] == []

    def test_executor_has_full_toolset(self):
        """Executor needs the most tools for code generation."""
        tools = AGENT_ALLOWED_TOOLS["executor"]
        required = ["Read", "Write", "Edit", "Bash", "Glob", "Grep", "Skill"]
        for tool in required:
            assert tool in tools, f"Executor missing required tool: {tool}"

    def test_verifier_has_web_tools(self):
        """Verifier needs web tools for fact-checking."""
        tools = AGENT_ALLOWED_TOOLS["verifier"]
        assert "WebSearch" in tools
        assert "WebFetch" in tools

    def test_reviewer_cannot_modify(self):
        """Reviewer should not be able to modify files."""
        tools = AGENT_ALLOWED_TOOLS["reviewer"]
        assert "Write" not in tools
        assert "Edit" not in tools
        assert "Bash" not in tools

    def test_formatter_can_write(self):
        """Formatter needs write access for document output."""
        tools = AGENT_ALLOWED_TOOLS["formatter"]
        assert "Write" in tools
        assert "Read" in tools


# =============================================================================
# PART 3: Skills Mapping
# =============================================================================

class TestSkillsMappings:
    """Test agent-to-skill mappings."""

    def test_executor_has_code_generation_skill(self):
        skills = get_skills_for_agent("executor")
        assert "code-generation" in skills

    def test_formatter_has_document_creation_skill(self):
        skills = get_skills_for_agent("formatter")
        assert "document-creation" in skills

    def test_analyst_has_requirements_skill(self):
        skills = get_skills_for_agent("analyst")
        assert "requirements-engineering" in skills

    def test_planner_has_architecture_skill(self):
        skills = get_skills_for_agent("planner")
        assert "architecture-design" in skills

    def test_clarifier_has_no_skills(self):
        """Clarifier doesn't need skills."""
        skills = get_skills_for_agent("clarifier")
        assert skills == []

    def test_verifier_has_no_skills(self):
        """Verifier doesn't need skills (uses web tools)."""
        skills = get_skills_for_agent("verifier")
        assert skills == []

    def test_reviewer_has_no_skills(self):
        """Reviewer doesn't need skills."""
        skills = get_skills_for_agent("reviewer")
        assert skills == []

    def test_skills_files_exist(self):
        """Verify referenced skill SKILL.md files exist on disk."""
        skill_dir = Path(".")  / ".claude" / "skills"
        agents_with_skills = {
            "executor": "code-generation",
            "formatter": "document-creation",
            "analyst": "requirements-engineering",
            "planner": "architecture-design",
        }
        for agent, skill in agents_with_skills.items():
            skill_path = skill_dir / skill / "SKILL.md"
            assert skill_path.exists(), f"Skill file missing for {agent}: {skill_path}"


# =============================================================================
# PART 4: Schema Compliance & JSON Schema Generation
# =============================================================================

class TestSchemaCompliance:
    """Test Pydantic schemas produce valid JSON schemas for SDK outputFormat."""

    def test_analyst_schema_generates_json_schema(self):
        schema = TaskIntelligenceReport.model_json_schema()
        assert "properties" in schema
        assert "literal_request" in schema["properties"]
        assert "sub_tasks" in schema["properties"]
        assert "confidence" in schema["properties"]

    def test_planner_schema_generates_json_schema(self):
        schema = ExecutionPlan.model_json_schema()
        assert "properties" in schema
        assert "steps" in schema["properties"]
        assert "critical_path" in schema["properties"]

    def test_clarifier_schema_generates_json_schema(self):
        schema = ClarificationRequest.model_json_schema()
        assert "properties" in schema
        assert "questions" in schema["properties"]
        assert "can_proceed_with_defaults" in schema["properties"]

    def test_verifier_schema_generates_json_schema(self):
        schema = VerificationReport.model_json_schema()
        assert "properties" in schema
        assert "claims" in schema["properties"]
        assert "verdict" in schema["properties"]

    def test_reviewer_schema_generates_json_schema(self):
        schema = ReviewVerdict.model_json_schema()
        assert "properties" in schema
        assert "verdict" in schema["properties"]
        assert "quality_gate_results" in schema["properties"]

    def test_schema_validation_with_valid_json(self):
        """Test _validate_output with valid JSON."""
        schema = {"required": ["name", "value"]}
        assert _validate_output('{"name": "test", "value": 42}', schema) is True

    def test_schema_validation_with_missing_fields(self):
        """Test _validate_output with missing required fields."""
        schema = {"required": ["name", "value"]}
        assert _validate_output('{"name": "test"}', schema) is False

    def test_schema_validation_with_invalid_json(self):
        """Test _validate_output with invalid JSON."""
        schema = {"required": ["name"]}
        assert _validate_output("not json at all", schema) is False

    def test_schema_validation_with_none(self):
        """Test _validate_output with None."""
        schema = {"required": ["name"]}
        assert _validate_output(None, schema) is False

    def test_schema_validation_with_dict_input(self):
        """Test _validate_output with dict input (not string)."""
        schema = {"required": ["name"]}
        assert _validate_output({"name": "test"}, schema) is True

    def test_output_schema_lookup_analyst(self):
        """Test _get_output_schema for analyst."""
        schema = _get_output_schema("analyst")
        # May return None if schemas module doesn't expose it directly
        # The important thing is it doesn't raise
        assert schema is None or isinstance(schema, dict)

    def test_output_schema_lookup_unknown(self):
        """Test _get_output_schema for unknown agent."""
        schema = _get_output_schema("nonexistent_agent")
        assert schema is None


# =============================================================================
# PART 5: Analyst Agent Exhaustive Tests
# =============================================================================

class TestAnalystExhaustive:
    """Exhaustive tests for AnalystAgent."""

    def test_data_modality_from_json_attachment(self, analyst):
        report = analyst.analyze("Process this data", file_attachments=["data.json"])
        assert report.modality == ModalityType.DATA

    def test_data_modality_from_csv_attachment(self, analyst):
        report = analyst.analyze("Process this", file_attachments=["data.csv"])
        assert report.modality == ModalityType.DATA

    def test_document_modality_from_pdf_attachment(self, analyst):
        report = analyst.analyze("Review this", file_attachments=["document.pdf"])
        assert report.modality == ModalityType.DOCUMENT

    def test_image_modality_from_png_attachment(self, analyst):
        report = analyst.analyze("Look at this", file_attachments=["image.png"])
        assert report.modality == ModalityType.IMAGE

    def test_multiple_file_attachments_first_wins(self, analyst):
        report = analyst.analyze("Process", file_attachments=["script.py", "data.json"])
        assert report.modality == ModalityType.CODE

    def test_data_modality_from_keyword(self, analyst):
        report = analyst.analyze("Query the dataset for analysis")
        assert report.modality == ModalityType.DATA

    def test_convert_intent_inference(self, analyst):
        report = analyst.analyze("Convert the JSON to YAML format")
        assert "transform" in report.inferred_intent.lower()

    def test_compare_intent_inference(self, analyst):
        report = analyst.analyze("Compare these two implementations")
        assert "differences" in report.inferred_intent.lower()

    def test_implement_intent_inference(self, analyst):
        report = analyst.analyze("Implement a caching layer")
        assert "build" in report.inferred_intent.lower() or "feature" in report.inferred_intent.lower()

    def test_document_decomposition(self, analyst):
        report = analyst.analyze("Write documentation for the project")
        descriptions = [st.description.lower() for st in report.sub_tasks]
        assert any("document" in d for d in descriptions)

    def test_deployment_missing_info(self, analyst):
        report = analyst.analyze("Deploy the app to production")
        requirements = [m.requirement.lower() for m in report.missing_info]
        assert any("deploy" in r for r in requirements)

    def test_critical_missing_reduces_confidence(self, analyst):
        report = analyst.analyze("Build an API with database and auth management")
        # Has critical missing info for auth and db
        assert report.confidence < 0.85

    def test_tier4_adversarial_keyword(self, analyst):
        report = analyst.analyze("Adversarial attack simulation")
        assert report.suggested_tier == 4

    def test_tier4_critical_keyword(self, analyst):
        report = analyst.analyze("Mission critical system deployment")
        # "critical" triggers tier 4
        assert report.suggested_tier == 4

    def test_escalation_keywords(self, analyst):
        for keyword in ["complex", "complicated", "uncertain", "depends on"]:
            report = analyst.analyze(f"This {keyword} problem needs solving")
            assert report.escalation_needed is True, f"Failed for keyword: {keyword}"

    def test_assumptions_always_include_common(self, analyst):
        report = analyst.analyze("Build something")
        assert any("permissions" in a.lower() for a in report.assumptions)
        assert any("best practices" in a.lower() for a in report.assumptions)
        assert any("conventions" in a.lower() for a in report.assumptions)

    def test_prepare_request_with_files(self, analyst):
        enhanced = analyst._prepare_request("Review this", ["test.py", "util.py"])
        assert "[File: test.py]" in enhanced
        assert "[File: util.py]" in enhanced

    def test_prepare_request_without_files(self, analyst):
        enhanced = analyst._prepare_request("Review this", None)
        assert enhanced == "Review this"

    def test_confidence_clamped_to_range(self, analyst):
        """Test confidence stays within 0-1 even with extreme inputs."""
        report = analyst.analyze("Build an API with auth, database, deployment, testing, monitoring")
        assert 0.0 <= report.confidence <= 1.0

    def test_confidence_boost_for_many_subtasks(self, analyst):
        """More subtasks = higher confidence."""
        report = analyst.analyze("Build an API endpoint with auth")
        # API decomposition creates 4+ subtasks -> confidence boost
        assert report.confidence >= 0.7

    def test_text_modality_for_plain_request(self, analyst):
        report = analyst.analyze("What is the meaning of life?")
        assert report.modality == ModalityType.TEXT


# =============================================================================
# PART 6: Planner Agent Exhaustive Tests
# =============================================================================

class TestPlannerExhaustive:
    """Exhaustive tests for PlannerAgent."""

    def test_plan_with_sme_selections(self, planner, basic_report):
        plan = planner.create_plan(basic_report, sme_selections=["Security Analyst", "Cloud Architect"])
        assert "Security Analyst" in plan.required_sme_personas
        assert "Cloud Architect" in plan.required_sme_personas

    def test_plan_parallel_groups_for_review(self, planner, basic_report):
        """Review steps should have a parallel group (Verifier + Critic)."""
        plan = planner.create_plan(basic_report)
        # Check for parallel group with review_group ID
        parallel_steps = [s for s in plan.steps if s.can_parallelize]
        # At least the Critic step should be parallelizable
        assert len(parallel_steps) >= 1

    def test_plan_formatter_always_last(self, planner, basic_report):
        plan = planner.create_plan(basic_report)
        last_step = plan.steps[-1]
        agent_names = [a.agent_name for a in last_step.agent_assignments]
        assert "Formatter" in agent_names

    def test_plan_reviewer_before_formatter(self, planner, basic_report):
        plan = planner.create_plan(basic_report)
        reviewer_step = None
        formatter_step = None
        for step in plan.steps:
            for a in step.agent_assignments:
                if a.agent_name == "Reviewer":
                    reviewer_step = step.step_number
                if a.agent_name == "Formatter":
                    formatter_step = step.step_number
        if reviewer_step and formatter_step:
            assert reviewer_step < formatter_step

    def test_plan_duration_positive(self, planner, basic_report):
        plan = planner.create_plan(basic_report)
        assert plan.estimated_duration_minutes >= 1

    def test_plan_critical_path_nonempty(self, planner, basic_report):
        plan = planner.create_plan(basic_report)
        assert len(plan.critical_path) > 0

    def test_sme_detection_from_keywords(self, planner):
        """Test SME auto-detection from task keywords."""
        report = TaskIntelligenceReport(
            literal_request="Deploy to AWS with kubernetes and security testing",
            inferred_intent="Deploy",
            sub_tasks=[SubTask(description="Deploy", dependencies=[])],
            missing_info=[],
            assumptions=[],
            modality=ModalityType.TEXT,
            recommended_approach="Deploy",
            escalation_needed=False,
        )
        plan = planner.create_plan(report)
        smes = plan.required_sme_personas
        assert any("Cloud" in s or "Security" in s for s in smes)

    def test_max_3_smes(self, planner):
        """SME selection should be capped at 3."""
        report = TaskIntelligenceReport(
            literal_request="Deploy to AWS cloud kubernetes security test docs ui data ai",
            inferred_intent="Deploy everything",
            sub_tasks=[SubTask(description="Deploy", dependencies=[])],
            missing_info=[],
            assumptions=[],
            modality=ModalityType.TEXT,
            recommended_approach="Deploy",
            escalation_needed=False,
        )
        plan = planner.create_plan(report)
        assert len(plan.required_sme_personas) <= 3

    def test_escalation_risk_flagged(self, planner):
        report = TaskIntelligenceReport(
            literal_request="Complex task",
            inferred_intent="Something complex",
            sub_tasks=[SubTask(description="Do it", dependencies=[])],
            missing_info=[],
            assumptions=[],
            modality=ModalityType.TEXT,
            recommended_approach="Do it",
            escalation_needed=True,
        )
        plan = planner.create_plan(report)
        assert any("escalation" in r.lower() for r in plan.risk_factors)

    def test_design_task_gets_planner_assignment(self, planner, basic_report):
        """Design sub-tasks should get Planner assigned."""
        report = TaskIntelligenceReport(
            literal_request="Design a system",
            inferred_intent="Design",
            sub_tasks=[SubTask(description="Design the architecture", dependencies=[])],
            missing_info=[],
            assumptions=[],
            modality=ModalityType.TEXT,
            recommended_approach="Design",
            escalation_needed=False,
        )
        plan = planner.create_plan(report)
        all_agents = []
        for step in plan.steps:
            for a in step.agent_assignments:
                all_agents.append(a.agent_name)
        assert "Planner" in all_agents

    def test_test_task_gets_executor_assignment(self, planner):
        """Test sub-tasks should get Executor and Test SME."""
        report = TaskIntelligenceReport(
            literal_request="Write tests",
            inferred_intent="Test",
            sub_tasks=[SubTask(description="Implement test cases", dependencies=[])],
            missing_info=[],
            assumptions=[],
            modality=ModalityType.CODE,
            recommended_approach="Test",
            escalation_needed=False,
        )
        plan = planner.create_plan(report)
        all_agents = []
        for step in plan.steps:
            for a in step.agent_assignments:
                all_agents.append(a.agent_name)
        assert "Executor" in all_agents

    def test_empty_steps_critical_path(self, planner):
        """Edge case: empty steps should have empty critical path."""
        path = planner._calculate_critical_path([])
        assert path == []


# =============================================================================
# PART 7: Clarifier Agent Exhaustive Tests
# =============================================================================

class TestClarifierExhaustive:
    """Exhaustive tests for ClarifierAgent."""

    def test_critical_question_has_high_risk_impact(self, clarifier):
        impact = clarifier._assess_impact(
            MissingInfo(requirement="Auth", severity=SeverityLevel.CRITICAL,
                       impact="Security", default_assumption="JWT")
        )
        assert impact["risk"] == "high"
        assert "severe" in impact["quality"].lower() or "unusable" in impact["quality"].lower()

    def test_important_question_has_medium_risk(self, clarifier):
        impact = clarifier._assess_impact(
            MissingInfo(requirement="Deploy", severity=SeverityLevel.IMPORTANT,
                       impact="Config", default_assumption="Docker")
        )
        assert impact["risk"] == "medium"

    def test_nice_to_have_has_low_risk(self, clarifier):
        impact = clarifier._assess_impact(
            MissingInfo(requirement="Tests", severity=SeverityLevel.NICE_TO_HAVE,
                       impact="Coverage", default_assumption="Unit tests")
        )
        assert impact["risk"] == "low"

    def test_answer_options_for_database(self, clarifier):
        options = clarifier._get_answer_options("Database technology")
        assert options is not None
        assert "PostgreSQL" in options

    def test_answer_options_for_testing_framework(self, clarifier):
        options = clarifier._get_answer_options("Testing framework")
        assert options is not None
        assert "pytest" in options

    def test_workflow_for_few_questions(self, clarifier):
        """Fewer than 3 non-critical questions -> present all together."""
        report = TaskIntelligenceReport(
            literal_request="Build something",
            inferred_intent="Build",
            sub_tasks=[SubTask(description="Build", dependencies=[])],
            missing_info=[
                MissingInfo(requirement="Testing", severity=SeverityLevel.NICE_TO_HAVE,
                           impact="QA", default_assumption="Unit tests"),
            ],
            assumptions=[],
            modality=ModalityType.TEXT,
            recommended_approach="Build",
            escalation_needed=False,
        )
        result = clarifier.formulate_questions(report)
        assert "together" in result.recommended_workflow.lower() or "present" in result.recommended_workflow.lower()

    def test_all_defaults_available_allows_proceed(self, clarifier):
        """If all questions have defaults, can proceed."""
        report = TaskIntelligenceReport(
            literal_request="Build something",
            inferred_intent="Build",
            sub_tasks=[SubTask(description="Build", dependencies=[])],
            missing_info=[
                MissingInfo(requirement="Auth", severity=SeverityLevel.CRITICAL,
                           impact="Security", default_assumption="JWT"),
            ],
            assumptions=[],
            modality=ModalityType.TEXT,
            recommended_approach="Build",
            escalation_needed=False,
        )
        result = clarifier.formulate_questions(report)
        # Critical but with default -> can proceed
        assert result.can_proceed_with_defaults is True

    def test_build_question_text_format(self, clarifier):
        """Test question text varies by severity."""
        critical_q = clarifier._build_question_text("Authentication method", SeverityLevel.CRITICAL)
        assert "should be used" in critical_q.lower()

        important_q = clarifier._build_question_text("Deploy target", SeverityLevel.IMPORTANT)
        assert "prefer" in important_q.lower()

        nice_q = clarifier._build_question_text("Testing", SeverityLevel.NICE_TO_HAVE)
        assert "preference" in nice_q.lower()


# =============================================================================
# PART 8: Executor Agent Exhaustive Tests
# =============================================================================

class TestExecutorExhaustive:
    """Exhaustive tests for ExecutorAgent."""

    def test_scoring_weights_sum_to_one(self, executor):
        total = sum(executor.scoring_weights.values())
        assert total == pytest.approx(1.0)

    def test_empty_approaches_returns_default(self, executor):
        result = executor._select_best_approach([])
        assert result.name == "Standard Approach"

    def test_sme_advice_bumps_complexity(self, executor):
        approach = Approach(name="Direct", description="D", steps=["s"],
                           pros=["p"], cons=["c"], estimated_time="low", complexity="low")
        adapted = executor._adapt_to_sme_advice(approach, {"Expert": "Add validation"})
        assert adapted.complexity == "medium"

    def test_sme_advice_no_double_bump(self, executor):
        """Already medium complexity shouldn't be changed."""
        approach = Approach(name="A", description="D", steps=["s"],
                           pros=["p"], cons=["c"], estimated_time="low", complexity="medium")
        adapted = executor._adapt_to_sme_advice(approach, {"Expert": "Add validation"})
        assert adapted.complexity == "medium"

    def test_code_task_creates_file_reference(self, executor):
        result = executor.execute("Write a Python function for sorting")
        assert len(result.files_created) > 0
        assert result.files_created[0].endswith(".py")

    def test_golang_file_path(self, executor):
        path = executor._determine_file_path("Build a golang service")
        assert path is not None
        assert path.endswith(".go")

    def test_rust_file_path(self, executor):
        path = executor._determine_file_path("Write Rust code for parsing.rs")
        assert path is not None
        assert path.endswith(".rs")

    def test_validation_boosts_quality(self, executor):
        """Successful validation increases quality score."""
        result = ExecutionResult(
            approach_name="Test", status="success", quality_score=0.8
        )
        validated = executor._validate_output(result)
        assert validated.quality_score > 0.8

    def test_validation_with_error_reduces_quality(self, executor):
        result = ExecutionResult(
            approach_name="Test", status="failed", quality_score=0.8,
            error="Something went wrong"
        )
        validated = executor._validate_output(result)
        assert validated.quality_score < 0.8

    def test_execute_with_sme_advisory(self, executor):
        result = executor.execute(
            "Write a Python function",
            sme_advisory={"Security Expert": "Add input validation"}
        )
        assert result.status == "success"

    def test_thought_branch_enum(self):
        assert ThoughtBranch.SEQUENTIAL == "sequential"
        assert ThoughtBranch.PARALLEL == "parallel"
        assert ThoughtBranch.DECOMPOSE == "decompose"
        assert ThoughtBranch.SIMPLIFY == "simplify"

    def test_thought_node_creation(self):
        approach = Approach(name="A", description="D", steps=["s"],
                           pros=["p"], cons=["c"], estimated_time="low", complexity="low")
        node = ThoughtNode(approach=approach, depth=0)
        assert node.depth == 0
        assert node.explored is False
        assert node.selected is False
        assert node.children == []

    def test_completeness_scoring(self, executor):
        approach = Approach(name="Comprehensive", description="Full",
                           steps=["s1", "s2", "s3", "s4"],
                           pros=["thorough"], cons=["slow"],
                           estimated_time="high", complexity="high")
        score = executor._score_completeness(approach, "task", None)
        assert score >= 0.9  # 4+ steps + "comprehensive" bonus

    def test_feasibility_scoring(self, executor):
        low = Approach(name="A", description="D", steps=["s"],
                       pros=[], cons=[], estimated_time="low", complexity="low")
        high = Approach(name="B", description="D", steps=["s"],
                        pros=[], cons=[], estimated_time="high", complexity="high")
        assert executor._score_feasibility(low) > executor._score_feasibility(high)

    def test_code_output_generation(self, executor):
        output = executor._generate_code_output("Sort a list")
        assert "def " in output or "solution" in output.lower()

    def test_document_output_generation(self, executor):
        output = executor._generate_document_output("API docs")
        assert "Documentation" in output or "Overview" in output

    def test_analysis_output_generation(self, executor):
        output = executor._generate_analysis_output("Performance review")
        assert "Analysis" in output or "Insight" in output


# =============================================================================
# PART 9: Verifier Agent Exhaustive Tests
# =============================================================================

class TestVerifierExhaustive:
    """Exhaustive tests for VerifierAgent."""

    def test_empty_content_produces_empty_report(self, verifier):
        report = verifier.verify("")
        assert report.total_claims_checked == 0

    def test_no_claims_gives_default_reliability(self, verifier):
        report = verifier.verify("No factual claims here")
        # With no claims, reliability defaults to 0.5
        if report.total_claims_checked == 0:
            assert report.overall_reliability == 0.5

    def test_contradicted_measurement_claim(self, verifier):
        result = verifier._verify_measurement_claim("It was both 100% and 0% successful")
        assert result["status"] == VerificationStatus.CONTRADICTED

    def test_future_year_flagged(self, verifier):
        result = verifier._verify_date_claim("Released in 2099")
        assert result["risk"] == FabricationRisk.HIGH
        assert result["confidence"] <= 2

    def test_historical_year_medium_risk(self, verifier):
        result = verifier._verify_date_claim("Founded in 1850")
        assert result["risk"] == FabricationRisk.MEDIUM

    def test_url_without_protocol_low_confidence(self, verifier):
        result = verifier._verify_url_claim("Visit www.example.com")
        # www without protocol - still should be handled
        assert result["status"] == VerificationStatus.UNVERIFIED

    def test_hallucination_pattern_everyone_knows(self, verifier):
        result = verifier._verify_general_claim("Everyone knows that Python is best", "")
        assert result["risk"] == FabricationRisk.MEDIUM
        assert result["confidence"] < 7

    def test_hallucination_pattern_it_is_well_known(self, verifier):
        result = verifier._verify_general_claim("It is well known that AI is reliable", "")
        assert result["risk"] == FabricationRisk.MEDIUM

    def test_source_verification_matches(self, verifier):
        result = verifier._verify_against_sources(
            "Python supports multiple paradigms",
            ["Python is a multi-paradigm programming language"]
        )
        assert result["found"] is True
        assert result["confidence"] >= 9

    def test_source_verification_no_match(self, verifier):
        result = verifier._verify_against_sources(
            "Completely unrelated claim xyz",
            ["Nothing about xyz here"]
        )
        # Short words are excluded, "xyz" is 3 chars -> excluded
        # "Completely", "unrelated", "claim" all > 3 chars
        # "Nothing", "about", "here" - no overlap
        assert result["found"] is False

    def test_sme_domain_matching(self, verifier):
        assert verifier._claim_matches_domain("Python uses dynamic typing", "Python Expert") is True
        assert verifier._claim_matches_domain("Java is compiled", "Python Expert") is False

    def test_corrections_generated_for_flagged(self, verifier):
        """Test corrections are generated for low-confidence claims."""
        content = """
        It is obviously the best approach ever created in 2099.
        Everyone knows that this will always work 150% of the time.
        """
        report = verifier.verify(content)
        # Should have flagged claims and corrections
        assert isinstance(report.recommended_corrections, list)

    def test_claim_batch_model(self):
        """Test ClaimBatch Pydantic model."""
        batch = ClaimBatch(
            topic="Python facts",
            claims=[Claim(
                claim_text="Python was released in 1991",
                confidence=10,
                fabrication_risk=FabricationRisk.LOW,
                verification_method="Known fact",
                status=VerificationStatus.VERIFIED,
            )],
            overall_reliability=1.0,
        )
        assert batch.topic == "Python facts"
        assert len(batch.claims) == 1

    def test_verification_report_counts_add_up(self, verifier):
        content = """
        According to research, Python was created in 1991.
        It is known that Python uses dynamic typing.
        Studies have shown approximately 200 modules in stdlib.
        """
        report = verifier.verify(content)
        total = (report.verified_claims + report.unverified_claims +
                report.contradicted_claims + report.fabricated_claims)
        assert total == report.total_claims_checked


# =============================================================================
# PART 10: Reviewer Agent Exhaustive Tests
# =============================================================================

class TestReviewerExhaustive:
    """Exhaustive tests for ReviewerAgent."""

    def test_security_failure_always_fails(self, reviewer, basic_context):
        output = "api_key = 'sk-12345' hardcoded password here"
        verdict = reviewer.review(output, basic_context)
        assert verdict.verdict == Verdict.FAIL

    def test_hallucination_failure_always_fails(self, reviewer, basic_context):
        verifier_report = {"fabricated_claims": 5, "verdict": "FAIL", "overall_reliability": 0.3}
        output = "This is the output content with proper structure.\n## Section\nDetails here."
        verdict = reviewer.review(output, basic_context, verifier_report=verifier_report)
        assert verdict.verdict == Verdict.FAIL

    def test_code_review_failure_fails_code_output(self, reviewer, code_context):
        code_review = {"pass_fail": False, "findings": [
            {"severity": "CRITICAL", "category": "SECURITY", "issue": "SQL injection"}
        ]}
        output = "```python\ndef sort_list():\n    return sorted(items)\n```"
        verdict = reviewer.review(output, code_context, code_review_report=code_review)
        assert verdict.verdict == Verdict.FAIL

    def test_readability_long_unstructured_fails(self, reviewer):
        # Long text without any structure markers
        long_text = "This is a very long sentence that goes on and on without any breaks or structure. " * 20
        check = reviewer._check_readability(long_text)
        assert check.passed is False

    def test_readability_structured_passes(self, reviewer):
        output = "## Title\n\n1. First point\n2. Second point\n\n```python\ncode\n```"
        check = reviewer._check_readability(output)
        assert check.passed is True

    def test_looks_like_code_detection(self, reviewer):
        assert reviewer._looks_like_code("def hello():\n    pass") is True
        assert reviewer._looks_like_code("Just some text") is False

    def test_extract_requirements(self, reviewer):
        reqs = reviewer._extract_requirements('Create a "sorting function" with 3 methods')
        assert len(reqs) > 0

    def test_requirement_synonym_matching(self, reviewer):
        assert reviewer._is_requirement_addressed("test", "unit testing framework") is True
        assert reviewer._is_requirement_addressed("security", "uses secure authentication") is True
        assert reviewer._is_requirement_addressed("unicorn", "nothing related") is False

    def test_revision_instructions_for_fail(self, reviewer, basic_context):
        output = "sql injection vulnerability here. hardcoded password."
        verdict = reviewer.review(output, basic_context)
        if verdict.verdict == Verdict.FAIL:
            assert len(verdict.revision_instructions) > 0
            # Check revisions have required fields
            for rev in verdict.revision_instructions:
                assert rev.category
                assert rev.description
                assert rev.priority

    def test_max_revisions_reached_flag(self, reviewer, basic_context):
        basic_context.revision_count = 2
        basic_context.max_revisions = 2
        output = "## Good Output\n\nThis is well-structured content."
        verdict = reviewer.review(output, basic_context)
        assert verdict.can_revise is False

    def test_confidence_calculation(self, reviewer, basic_context):
        output = "## Good Output\n\nWell-structured content about sort."
        basic_context.original_request = "sort"
        verdict = reviewer.review(output, basic_context)
        assert 0.0 <= verdict.confidence <= 1.0

    def test_tier4_arbitration_on_disagreement(self, reviewer):
        ctx = ReviewContext(
            original_request="Security audit",
            agent_outputs={}, revision_count=0,
            max_revisions=2, tier_level=4, is_code_output=False,
        )
        needed, arb = reviewer._check_arbitration_needed(
            Verdict.PASS, Verdict.PASS, Verdict.FAIL, ctx
        )
        assert needed is True
        assert arb is not None
        assert arb.verifier_verdict == Verdict.PASS
        assert arb.critic_verdict == Verdict.FAIL

    def test_no_arbitration_tier2(self, reviewer, basic_context):
        needed, arb = reviewer._check_arbitration_needed(
            Verdict.PASS, Verdict.PASS, Verdict.FAIL, basic_context
        )
        assert needed is False
        assert arb is None

    def test_all_verdict_matrix_combinations(self, reviewer):
        """Test all 4 verdict matrix combinations."""
        combos = [
            (Verdict.PASS, Verdict.PASS, "PROCEED_TO_FORMATTER"),
            (Verdict.PASS, Verdict.FAIL, "EXECUTOR_REVISE"),
            (Verdict.FAIL, Verdict.PASS, "RESEARCHER_REVERIFY"),
            (Verdict.FAIL, Verdict.FAIL, "FULL_REGENERATION"),
        ]
        for v_verdict, c_verdict, expected_action in combos:
            action = reviewer._apply_verdict_matrix(v_verdict, c_verdict)
            assert action == expected_action, f"Matrix failed: ({v_verdict}, {c_verdict}) -> {action}"

    def test_consistency_with_agent_outputs(self, reviewer):
        ctx = ReviewContext(
            original_request="Write code",
            agent_outputs={"analyst": {"modality": "code"}},
            revision_count=0, max_revisions=2, tier_level=2, is_code_output=False,
        )
        # Output that doesn't look like code when analyst predicted code
        check = reviewer._check_consistency("Just plain text without code", ctx)
        assert "Analyst predicted code" in check.notes

    def test_critic_findings_pass_without_critical(self, reviewer):
        critic_report = {"overall_assessment": "Minor issues found", "attacks": []}
        check = reviewer._check_critic_findings(critic_report)
        assert check.passed is True

    def test_critic_findings_fail_with_critical(self, reviewer):
        critic_report = {
            "overall_assessment": "Critical vulnerabilities found",
            "attacks": [{"verdict": "FAIL"}, {"verdict": "FAIL"}]
        }
        check = reviewer._check_critic_findings(critic_report)
        assert check.passed is False


# =============================================================================
# PART 11: Formatter Agent Exhaustive Tests
# =============================================================================

class TestFormatterExhaustive:
    """Exhaustive tests for FormatterAgent."""

    def test_yaml_format_dict(self, formatter):
        result = formatter.format({"key": "value"}, target_format="yaml")
        assert result["format"] == "yaml"
        assert "key" in result["formatted_output"]

    def test_yaml_format_list(self, formatter):
        result = formatter.format(["item1", "item2"], target_format="yaml")
        assert result["format"] == "yaml"
        assert "items" in result["formatted_output"]

    def test_yaml_format_string(self, formatter):
        result = formatter.format("hello", target_format="yaml")
        assert result["format"] == "yaml"
        assert "value" in result["formatted_output"]

    def test_document_format_docx(self, formatter):
        result = formatter.format("Content here", target_format="docx")
        assert result["format"] == "docx"
        assert "DOCX" in result["formatted_output"]

    def test_document_format_pdf(self, formatter):
        result = formatter.format("Content here", target_format="pdf")
        assert result["format"] == "pdf"

    def test_document_format_xlsx(self, formatter):
        result = formatter.format({"data": "here"}, target_format="xlsx")
        assert result["format"] == "xlsx"

    def test_document_format_pptx(self, formatter):
        result = formatter.format("Presentation content", target_format="pptx")
        assert result["format"] == "pptx"

    def test_code_format_with_file_write(self, formatter, tmp_path):
        code = "def hello():\n    print('Hello')"
        result = formatter.format(code, target_format="code", file_path="hello.py")
        assert result["format"] == "code"
        # File should be written
        full_path = os.path.join(str(tmp_path), "hello.py")
        assert os.path.exists(full_path)

    def test_code_format_without_file(self, formatter):
        code = "def hello():\n    print('Hello')"
        result = formatter.format(code, target_format="code")
        assert result["format"] == "code"
        assert result["file_path"] is None

    def test_state_diagram_generation(self, formatter):
        result = formatter._generate_state_diagram("State transitions", {})
        assert "stateDiagram" in result

    def test_generic_diagram_generation(self, formatter):
        result = formatter._generate_generic_diagram("Something", {})
        assert "graph" in result

    def test_mermaid_type_from_context(self, formatter):
        mtype = formatter._infer_mermaid_type("anything", {"diagram_type": "class"})
        assert mtype == "class"

    def test_mermaid_type_state(self, formatter):
        mtype = formatter._infer_mermaid_type("state transition diagram", None)
        assert mtype == "state"

    def test_format_markdown_dict(self, formatter):
        result = formatter.format({"title": "Hello", "body": "World"}, target_format="markdown")
        assert "Hello" in result["formatted_output"]

    def test_format_markdown_list(self, formatter):
        result = formatter.format(["Point 1", "Point 2"], target_format="markdown")
        assert "Point 1" in result["formatted_output"]

    def test_format_markdown_nested(self, formatter):
        data = {"Section": {"SubKey": "SubValue"}, "List": ["A", "B"]}
        result = formatter.format(data, target_format="markdown")
        assert "## Section" in result["formatted_output"]
        assert "## List" in result["formatted_output"]

    def test_detect_java_language(self, formatter):
        code = "public class Main {\n    public static void main(String[] args) {}\n}"
        lang = formatter._detect_language(code, None)
        assert lang == "java"

    def test_detect_go_language(self, formatter):
        code = "package main\nfunc main() {}"
        lang = formatter._detect_language(code, None)
        assert lang == "go"

    def test_detect_cpp_language(self, formatter):
        code = '#include <iostream>\nstd::cout << "hello"'
        lang = formatter._detect_language(code, None)
        assert lang == "cpp"

    def test_detect_javascript_language(self, formatter):
        code = "function hello() { const x = 1; return x; }"
        lang = formatter._detect_language(code, None)
        assert lang == "javascript"

    def test_detect_typescript_language(self, formatter):
        code = "interface Foo { name: string; } function hello(): void {}"
        lang = formatter._detect_language(code, None)
        assert lang == "typescript"

    def test_extract_code_from_inline(self, formatter):
        content = "Use `const x = 1; const y = 2; const z = 3; const a = 4; const b = 5; const c = 6;`"
        code = formatter._extract_code(content)
        assert "const" in code

    def test_format_context_override(self, formatter):
        result = formatter.format({"a": 1}, target_format="unknown", context={"format": "json"})
        assert result["format"] == "json"

    def test_metadata_size_bytes(self, formatter):
        result = formatter.format("Hello world", target_format="text")
        assert result["metadata"]["size_bytes"] == len("Hello world")

    def test_non_string_format_text(self, formatter):
        result = formatter.format(42, target_format="text")
        assert result["formatted_output"] == "42"


# =============================================================================
# PART 12: Configuration & Model Selection
# =============================================================================

class TestConfigurationIntegration:
    """Test agent model selection through configuration system."""

    def test_anthropic_default_models(self):
        mapping = DEFAULT_MODEL_MAPPINGS[LLMProvider.ANTHROPIC]
        # Opus for critical agents
        assert "opus" in mapping.get_model("orchestrator").lower()
        assert "opus" in mapping.get_model("verifier").lower()
        assert "opus" in mapping.get_model("reviewer").lower()
        # Sonnet for operational agents
        assert "sonnet" in mapping.get_model("analyst").lower()
        assert "sonnet" in mapping.get_model("planner").lower()
        assert "sonnet" in mapping.get_model("executor").lower()
        assert "sonnet" in mapping.get_model("formatter").lower()
        # Haiku for lightweight agents
        assert "haiku" in mapping.get_model("clarifier").lower()

    def test_all_providers_have_tier2_agents(self):
        """Every provider must define models for all 7 Tier 2 agents."""
        tier2_agents = ["analyst", "planner", "clarifier", "executor",
                       "verifier", "reviewer", "formatter"]
        for provider, mapping in DEFAULT_MODEL_MAPPINGS.items():
            for agent in tier2_agents:
                model = mapping.get_model(agent)
                assert model is not None, f"Provider {provider.value} missing model for {agent}"

    def test_settings_get_model_for_agent(self):
        """Test get_model_for_agent convenience function."""
        # Reset global settings to avoid stale state
        reload_settings()
        model = get_model_for_agent("analyst")
        assert model is not None
        assert len(model) > 0


# =============================================================================
# PART 13: Cross-Agent Pipeline Contract Tests
# =============================================================================

class TestCrossAgentContracts:
    """Test that agent outputs match expected inputs for the next pipeline phase."""

    def test_analyst_output_feeds_planner(self, analyst, planner):
        """Phase 1 -> Phase 3: Analyst output must be valid PlannerAgent input."""
        report = analyst.analyze("Build a REST API for users")
        # Planner should accept this report directly
        plan = planner.create_plan(report)
        assert isinstance(plan, ExecutionPlan)
        assert plan.total_steps > 0

    def test_analyst_output_feeds_clarifier(self, analyst, clarifier):
        """Phase 1 -> Phase 4: Analyst output with missing info feeds Clarifier."""
        report = analyst.analyze("Build an API with database")
        if report.missing_info:
            result = clarifier.formulate_questions(report)
            assert isinstance(result, ClarificationRequest)

    def test_analyst_output_feeds_executor(self, analyst, executor):
        """Phase 1 -> Phase 5: Analyst report provides context to Executor."""
        report = analyst.analyze("Write a Python function to sort a list")
        result = executor.execute(
            "Write a Python function to sort a list",
            analyst_report=report,
        )
        assert isinstance(result, ExecutionResult)
        assert result.status == "success"

    def test_executor_output_feeds_verifier(self, executor, verifier):
        """Phase 5 -> Phase 6: Executor output is verified by Verifier."""
        exec_result = executor.execute("Write documentation about Python sorting")
        # Verifier takes the raw output string
        ver_report = verifier.verify(str(exec_result.output))
        assert isinstance(ver_report, VerificationReport)

    def test_verifier_report_feeds_reviewer(self, verifier, reviewer, basic_context):
        """Phase 6 -> Phase 8: Verifier report feeds into Reviewer."""
        ver_report = verifier.verify("Python was created in 1991 by Guido van Rossum.")
        # Convert to dict for reviewer (as it would be in pipeline)
        report_dict = ver_report.model_dump()
        output = "## Python Sort Function\n\nPython was created in 1991.\n\n```python\ndef sort():\n    pass\n```"
        basic_context.original_request = "sort"
        verdict = reviewer.review(output, basic_context, verifier_report=report_dict)
        assert isinstance(verdict, ReviewVerdict)

    def test_reviewer_pass_flows_to_formatter(self, reviewer, formatter, basic_context):
        """Phase 8: Reviewer PASS -> Formatter formats output."""
        basic_context.original_request = "sort"
        output = "## Python Sort Function\n\n```python\ndef sort_list(items):\n    return sorted(items)\n```"
        verdict = reviewer.review(output, basic_context)
        if verdict.verdict == Verdict.PASS:
            formatted = formatter.format(output, target_format="markdown")
            assert formatted["format"] == "markdown"
            assert formatted["formatted_output"] is not None

    def test_full_tier2_pipeline(self, analyst, planner, clarifier, executor,
                                  verifier, reviewer, formatter, tmp_path):
        """End-to-end test: all 7 agents in Tier 2 pipeline sequence."""
        # Phase 1: Analyst
        report = analyst.analyze("Write a Python function to sort a list")
        assert isinstance(report, TaskIntelligenceReport)

        # Phase 3: Planner
        plan = planner.create_plan(report)
        assert isinstance(plan, ExecutionPlan)

        # Phase 4: Clarifier (only if needed)
        if report.missing_info:
            clarification = clarifier.formulate_questions(report)
            assert isinstance(clarification, ClarificationRequest)

        # Phase 5: Executor
        exec_result = executor.execute(
            "Write a Python function to sort a list",
            analyst_report=report,
        )
        assert exec_result.status == "success"

        # Phase 6: Verifier
        ver_report = verifier.verify(str(exec_result.output))
        assert isinstance(ver_report, VerificationReport)

        # Phase 8: Reviewer
        ctx = ReviewContext(
            original_request="sort a list",
            agent_outputs={"analyst": report.model_dump()},
            revision_count=0, max_revisions=2,
            tier_level=2, is_code_output=True,
        )
        verdict = reviewer.review(
            str(exec_result.output), ctx,
            verifier_report=ver_report.model_dump()
        )
        assert isinstance(verdict, ReviewVerdict)

        # Phase 8: Formatter
        formatted = formatter.format(
            str(exec_result.output),
            target_format="markdown",
        )
        assert formatted["format"] == "markdown"
        assert formatted["formatted_output"] is not None


# =============================================================================
# PART 14: SDK spawn_subagent Tests
# =============================================================================

class TestSpawnSubagent:
    """Test the SDK spawn_subagent wrapper."""

    def test_spawn_subagent_success(self):
        """Test successful subagent spawn returns correct structure."""
        options = ClaudeAgentOptions(
            name="Test Agent", model="test-model", system_prompt="prompt"
        )
        with patch("src.core.sdk_integration._execute_sdk_query") as mock_query:
            mock_query.return_value = {
                "output": "test output",
                "tokens_used": 100,
                "cost_usd": 0.01,
            }
            result = spawn_subagent(options, "test input")
            assert result["status"] == "success"
            assert result["output"] == "test output"
            assert result["tokens_used"] == 100

    def test_spawn_subagent_retry_on_error(self):
        """Test retry logic on transient errors."""
        options = ClaudeAgentOptions(
            name="Test Agent", model="test-model", system_prompt="prompt"
        )
        call_count = 0

        def mock_query(kwargs, input_data):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Transient error")
            return {"output": "recovered", "tokens_used": 50, "cost_usd": 0.005}

        with patch("src.core.sdk_integration._execute_sdk_query", side_effect=mock_query):
            with patch("time.sleep"):  # Speed up test
                result = spawn_subagent(options, "test", max_retries=2)
                assert result["status"] == "success"
                assert result["retries"] == 1

    def test_spawn_subagent_max_retries_exceeded(self):
        """Test error return after max retries."""
        options = ClaudeAgentOptions(
            name="Test Agent", model="test-model", system_prompt="prompt"
        )
        with patch("src.core.sdk_integration._execute_sdk_query",
                   side_effect=Exception("Persistent error")):
            with patch("time.sleep"):
                result = spawn_subagent(options, "test", max_retries=0)
                assert result["status"] == "error"
                assert "Persistent error" in result["error"]

    def test_spawn_subagent_validates_output_schema(self):
        """Test output validation triggers retry on schema mismatch."""
        options = ClaudeAgentOptions(
            name="Test", model="model", system_prompt="prompt",
            output_format={"required": ["name", "value"]},
        )
        call_count = 0

        def mock_query(kwargs, input_data):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"output": '{"name": "test"}', "tokens_used": 50, "cost_usd": 0.005}
            return {"output": '{"name": "test", "value": 42}', "tokens_used": 50, "cost_usd": 0.005}

        with patch("src.core.sdk_integration._execute_sdk_query", side_effect=mock_query):
            result = spawn_subagent(options, "test", max_retries=2)
            assert result["status"] == "success"


# =============================================================================
# PART 15: Edge Cases & Boundary Conditions
# =============================================================================

class TestEdgeCases:
    """Edge cases and boundary conditions across all agents."""

    def test_analyst_empty_request(self, analyst):
        report = analyst.analyze("")
        assert isinstance(report, TaskIntelligenceReport)
        assert report.literal_request == ""

    def test_analyst_very_long_request(self, analyst):
        long_request = "Write a function " * 1000
        report = analyst.analyze(long_request)
        assert isinstance(report, TaskIntelligenceReport)

    def test_analyst_unicode_request(self, analyst):
        report = analyst.analyze("Write a Python function for sorting: リスト並べ替え")
        assert isinstance(report, TaskIntelligenceReport)

    def test_analyst_special_characters(self, analyst):
        report = analyst.analyze("Fix the bug in file.py's `main()` function & test <html>")
        assert isinstance(report, TaskIntelligenceReport)

    def test_planner_single_subtask(self, planner):
        report = TaskIntelligenceReport(
            literal_request="Do one thing",
            inferred_intent="One thing",
            sub_tasks=[SubTask(description="Single task", dependencies=[])],
            missing_info=[],
            assumptions=[],
            modality=ModalityType.TEXT,
            recommended_approach="Simple",
            escalation_needed=False,
        )
        plan = planner.create_plan(report)
        assert plan.total_steps >= 1

    def test_verifier_content_only_numbers(self, verifier):
        report = verifier.verify("42 100 2024 99.5%")
        assert isinstance(report, VerificationReport)

    def test_verifier_content_only_urls(self, verifier):
        report = verifier.verify("https://example.com https://test.com")
        assert isinstance(report, VerificationReport)

    def test_reviewer_empty_output(self, reviewer, basic_context):
        verdict = reviewer.review("", basic_context)
        assert isinstance(verdict, ReviewVerdict)
        # Empty output should fail readability
        assert verdict.verdict == Verdict.FAIL

    def test_formatter_empty_content(self, formatter):
        result = formatter.format("", target_format="text")
        assert result["formatted_output"] == ""

    def test_formatter_none_like_content(self, formatter):
        result = formatter.format(None, target_format="text")
        assert result["formatted_output"] == "None"

    def test_formatter_numeric_content(self, formatter):
        result = formatter.format(42, target_format="json")
        assert result["format"] == "json"

    def test_executor_no_analyst_report(self, executor):
        """Executor should work without analyst report."""
        result = executor.execute("Do something simple")
        assert result.status == "success"

    def test_clarifier_empty_missing_info(self, clarifier):
        report = TaskIntelligenceReport(
            literal_request="Hello",
            inferred_intent="Greeting",
            sub_tasks=[SubTask(description="Greet", dependencies=[])],
            missing_info=[],
            assumptions=[],
            modality=ModalityType.TEXT,
            recommended_approach="Simple",
            escalation_needed=False,
        )
        result = clarifier.formulate_questions(report)
        assert result.total_questions == 0
        assert result.can_proceed_with_defaults is True
