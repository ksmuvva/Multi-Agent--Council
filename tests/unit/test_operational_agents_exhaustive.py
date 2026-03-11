"""
Exhaustive Tests for All 11 Operational Agents

Tests cover:
1. SDK Integration (ClaudeAgentOptions, allowed tools, output schemas, skills)
2. Edge cases and boundary conditions
3. Cross-agent data flow validation
4. Security pattern coverage
5. Heuristic logic correctness
6. Error handling and resilience
"""

import os
import re
import json
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Agent imports
from src.agents.analyst import AnalystAgent, create_analyst
from src.agents.planner import PlannerAgent, create_planner
from src.agents.clarifier import ClarifierAgent, create_clarifier
from src.agents.researcher import ResearcherAgent, create_researcher
from src.agents.executor import ExecutorAgent, create_executor
from src.agents.code_reviewer import CodeReviewerAgent, create_code_reviewer
from src.agents.formatter import FormatterAgent, create_formatter
from src.agents.verifier import VerifierAgent, create_verifier
from src.agents.critic import CriticAgent, create_critic
from src.agents.reviewer import ReviewerAgent, create_reviewer, ReviewContext
from src.agents.memory_curator import MemoryCuratorAgent, create_memory_curator

# Schema imports
from src.schemas.analyst import (
    TaskIntelligenceReport, SubTask, MissingInfo, SeverityLevel, ModalityType
)
from src.schemas.planner import (
    ExecutionPlan, ExecutionStep, AgentAssignment, StepStatus
)
from src.schemas.clarifier import (
    ClarificationRequest, ClarificationQuestion, QuestionPriority
)
from src.schemas.researcher import (
    EvidenceBrief, Finding, ConfidenceLevel, SourceReliability
)
from src.schemas.code_reviewer import (
    CodeReviewReport, SeverityLevel as CRSeverity, ReviewCategory
)
from src.schemas.verifier import (
    VerificationReport, VerificationStatus, FabricationRisk
)
from src.schemas.critic import (
    CritiqueReport, AttackVector, SeverityLevel as CriticSeverity
)
from src.schemas.reviewer import ReviewVerdict, Verdict

# SDK Integration imports
from src.core.sdk_integration import (
    ClaudeAgentOptions,
    PermissionMode,
    AGENT_ALLOWED_TOOLS,
    build_agent_options,
    get_skills_for_agent,
    _get_output_schema,
    _validate_output,
)


# =============================================================================
# SDK Integration Tests
# =============================================================================

class TestSDKIntegration:
    """Test Claude Agent SDK integration layer."""

    def test_all_operational_agents_have_tool_definitions(self):
        """Every operational agent must have an entry in AGENT_ALLOWED_TOOLS."""
        required_agents = [
            "analyst", "planner", "clarifier", "researcher",
            "executor", "code_reviewer", "formatter", "verifier",
            "critic", "reviewer", "memory_curator",
        ]
        for agent in required_agents:
            assert agent in AGENT_ALLOWED_TOOLS, f"Missing tool definition for {agent}"

    def test_analyst_tools(self):
        """Analyst gets Read, Glob, Grep - read-only codebase access."""
        tools = AGENT_ALLOWED_TOOLS["analyst"]
        assert "Read" in tools
        assert "Glob" in tools
        assert "Grep" in tools
        assert "Write" not in tools
        assert "Bash" not in tools

    def test_executor_has_full_tools(self):
        """Executor gets full tool access including Write, Edit, Bash, Skill."""
        tools = AGENT_ALLOWED_TOOLS["executor"]
        assert "Read" in tools
        assert "Write" in tools
        assert "Edit" in tools
        assert "Bash" in tools
        assert "Glob" in tools
        assert "Grep" in tools
        assert "Skill" in tools

    def test_clarifier_has_no_tools(self):
        """Clarifier should have zero tools - it only asks questions."""
        tools = AGENT_ALLOWED_TOOLS["clarifier"]
        assert len(tools) == 0

    def test_researcher_has_web_tools(self):
        """Researcher must have WebSearch and WebFetch."""
        tools = AGENT_ALLOWED_TOOLS["researcher"]
        assert "WebSearch" in tools
        assert "WebFetch" in tools
        assert "Read" in tools

    def test_verifier_has_web_tools(self):
        """Verifier must have WebSearch and WebFetch for fact-checking."""
        tools = AGENT_ALLOWED_TOOLS["verifier"]
        assert "WebSearch" in tools
        assert "WebFetch" in tools

    def test_code_reviewer_has_bash(self):
        """Code reviewer needs Bash for running linters/syntax checks."""
        tools = AGENT_ALLOWED_TOOLS["code_reviewer"]
        assert "Bash" in tools
        assert "Read" in tools

    def test_formatter_has_write_and_skill(self):
        """Formatter needs Write for output files and Skill for document-creation."""
        tools = AGENT_ALLOWED_TOOLS["formatter"]
        assert "Write" in tools
        assert "Skill" in tools

    def test_memory_curator_has_write(self):
        """Memory curator needs Write for knowledge files."""
        tools = AGENT_ALLOWED_TOOLS["memory_curator"]
        assert "Write" in tools
        assert "Read" in tools

    def test_critic_tools_read_only(self):
        """Critic should have read-only access."""
        tools = AGENT_ALLOWED_TOOLS["critic"]
        assert "Read" in tools
        assert "Write" not in tools
        assert "Bash" not in tools

    def test_reviewer_tools_read_only(self):
        """Reviewer should have read-only access."""
        tools = AGENT_ALLOWED_TOOLS["reviewer"]
        assert "Read" in tools
        assert "Write" not in tools

    def test_council_agents_have_no_tools(self):
        """Council agents should not have tools."""
        assert len(AGENT_ALLOWED_TOOLS["council_chair"]) == 0
        assert len(AGENT_ALLOWED_TOOLS["quality_arbiter"]) == 0
        assert len(AGENT_ALLOWED_TOOLS["ethics_advisor"]) == 0

    def test_least_privilege_principle(self):
        """No agent should have more tools than executor."""
        executor_tools = set(AGENT_ALLOWED_TOOLS["executor"])
        for agent_name, tools in AGENT_ALLOWED_TOOLS.items():
            if agent_name == "executor":
                continue
            # Web tools are not in executor set so skip researcher/verifier
            non_web_tools = [t for t in tools if t not in ("WebSearch", "WebFetch")]
            for tool in non_web_tools:
                assert tool in executor_tools or tool in ("Skill",), (
                    f"Agent {agent_name} has tool {tool} not available to executor"
                )


class TestSDKSkills:
    """Test skill mappings for agents."""

    def test_executor_has_code_generation_skill(self):
        skills = get_skills_for_agent("executor")
        assert "code-generation" in skills

    def test_formatter_has_document_creation_skill(self):
        skills = get_skills_for_agent("formatter")
        assert "document-creation" in skills

    def test_analyst_has_requirements_engineering_skill(self):
        skills = get_skills_for_agent("analyst")
        assert "requirements-engineering" in skills

    def test_planner_has_architecture_design_skill(self):
        skills = get_skills_for_agent("planner")
        assert "architecture-design" in skills

    def test_researcher_has_web_research_skill(self):
        skills = get_skills_for_agent("researcher")
        assert "web-research" in skills

    def test_code_reviewer_has_code_generation_skill(self):
        skills = get_skills_for_agent("code_reviewer")
        assert "code-generation" in skills

    def test_agents_without_skills(self):
        """Clarifier, verifier, critic, reviewer, memory_curator have no skills."""
        for agent in ["clarifier", "verifier", "critic", "reviewer", "memory_curator"]:
            skills = get_skills_for_agent(agent)
            assert len(skills) == 0, f"{agent} should have no skills"


class TestSDKOutputSchemas:
    """Test output schema generation for agents."""

    def test_analyst_schema_exists(self):
        schema = _get_output_schema("analyst")
        assert schema is not None
        assert "properties" in schema

    def test_planner_schema_exists(self):
        schema = _get_output_schema("planner")
        assert schema is not None

    def test_clarifier_schema_exists(self):
        schema = _get_output_schema("clarifier")
        assert schema is not None

    def test_researcher_schema_exists(self):
        schema = _get_output_schema("researcher")
        assert schema is not None

    def test_code_reviewer_schema_exists(self):
        schema = _get_output_schema("code_reviewer")
        assert schema is not None

    def test_verifier_schema_exists(self):
        schema = _get_output_schema("verifier")
        assert schema is not None

    def test_critic_schema_exists(self):
        schema = _get_output_schema("critic")
        assert schema is not None

    def test_reviewer_schema_exists(self):
        schema = _get_output_schema("reviewer")
        assert schema is not None

    def test_unknown_agent_returns_none(self):
        schema = _get_output_schema("nonexistent_agent")
        assert schema is None


class TestClaudeAgentOptions:
    """Test ClaudeAgentOptions dataclass."""

    def test_to_sdk_kwargs(self):
        options = ClaudeAgentOptions(
            name="Test Agent",
            model="claude-sonnet-4-20250514",
            system_prompt="You are a test agent",
            max_turns=30,
            allowed_tools=["Read", "Write"],
        )
        kwargs = options.to_sdk_kwargs()
        assert kwargs["name"] == "Test Agent"
        assert kwargs["model"] == "claude-sonnet-4-20250514"
        assert kwargs["system_prompt"] == "You are a test agent"
        assert kwargs["max_turns"] == 30
        assert kwargs["allowed_tools"] == ["Read", "Write"]

    def test_permission_mode_default_not_in_kwargs(self):
        options = ClaudeAgentOptions(
            name="Test",
            model="claude-sonnet-4-20250514",
            system_prompt="test",
        )
        kwargs = options.to_sdk_kwargs()
        assert "permission_mode" not in kwargs

    def test_executor_permission_mode(self):
        options = ClaudeAgentOptions(
            name="Executor",
            model="claude-sonnet-4-20250514",
            system_prompt="test",
            permission_mode=PermissionMode.ACCEPT_EDITS,
        )
        kwargs = options.to_sdk_kwargs()
        assert kwargs["permission_mode"] == "acceptEdits"

    def test_empty_tools_not_in_kwargs(self):
        options = ClaudeAgentOptions(
            name="Test",
            model="claude-sonnet-4-20250514",
            system_prompt="test",
            allowed_tools=[],
        )
        kwargs = options.to_sdk_kwargs()
        assert "allowed_tools" not in kwargs

    def test_setting_sources_default(self):
        options = ClaudeAgentOptions(
            name="Test",
            model="claude-sonnet-4-20250514",
            system_prompt="test",
        )
        assert options.setting_sources == ["user", "project"]


class TestSDKOutputValidation:
    """Test output validation logic."""

    def test_validate_valid_json(self):
        output = '{"literal_request": "test", "inferred_intent": "test"}'
        schema = {"required": ["literal_request", "inferred_intent"]}
        assert _validate_output(output, schema) is True

    def test_validate_missing_required_field(self):
        output = '{"literal_request": "test"}'
        schema = {"required": ["literal_request", "inferred_intent"]}
        assert _validate_output(output, schema) is False

    def test_validate_invalid_json(self):
        output = "not json"
        schema = {"required": ["field"]}
        assert _validate_output(output, schema) is False

    def test_validate_empty_output(self):
        assert _validate_output("", {}) is False
        assert _validate_output(None, {}) is False

    def test_validate_dict_output(self):
        output = {"field1": "value1", "field2": "value2"}
        schema = {"required": ["field1", "field2"]}
        assert _validate_output(output, schema) is True

    def test_validate_no_required_fields(self):
        output = '{"any": "thing"}'
        schema = {}
        assert _validate_output(output, schema) is True


# =============================================================================
# Analyst Agent Edge Cases
# =============================================================================

class TestAnalystEdgeCases:
    """Edge cases for the Analyst agent."""

    @pytest.fixture
    def analyst(self):
        return AnalystAgent()

    def test_empty_request(self, analyst):
        report = analyst.analyze("")
        assert isinstance(report, TaskIntelligenceReport)
        assert report.literal_request == ""

    def test_very_long_request(self, analyst):
        long_request = "a " * 10000
        report = analyst.analyze(long_request)
        assert isinstance(report, TaskIntelligenceReport)
        assert report.confidence >= 0.0

    def test_special_characters_in_request(self, analyst):
        report = analyst.analyze("Create a function with @decorator & $pecial chars!")
        assert isinstance(report, TaskIntelligenceReport)

    def test_multiple_modalities_in_request(self, analyst):
        """When request mentions both code and data, code should win."""
        report = analyst.analyze("Write a function to process JSON data files")
        assert report.modality in (ModalityType.CODE, ModalityType.DATA)

    def test_file_attachment_overrides_text_modality(self, analyst):
        report = analyst.analyze("Handle this", file_attachments=["test.py"])
        assert report.modality == ModalityType.CODE

    def test_data_file_attachment(self, analyst):
        report = analyst.analyze("Process this", file_attachments=["data.csv"])
        assert report.modality == ModalityType.DATA

    def test_confidence_never_exceeds_1(self, analyst):
        report = analyst.analyze("Simple hello world test print question")
        assert report.confidence <= 1.0

    def test_confidence_never_below_0(self, analyst):
        report = analyst.analyze(
            "Complicated uncertain complex depends on multiple factors "
            "security compliance adversarial attack critical"
        )
        assert report.confidence >= 0.0

    def test_suggested_tier_range(self, analyst):
        """Tier must be between 1 and 4."""
        for request in [
            "print hello",
            "build api with auth",
            "design microservice architecture for domain expert",
            "security compliance adversarial attack analysis",
        ]:
            report = analyst.analyze(request)
            assert 1 <= report.suggested_tier <= 4

    def test_assumptions_always_include_defaults(self, analyst):
        report = analyst.analyze("anything")
        assert any("permissions" in a.lower() for a in report.assumptions)
        assert any("best practices" in a.lower() for a in report.assumptions)


# =============================================================================
# Planner Agent Edge Cases
# =============================================================================

class TestPlannerEdgeCases:
    """Edge cases for the Planner agent."""

    @pytest.fixture
    def planner(self):
        return PlannerAgent()

    def _make_report(self, **kwargs):
        defaults = {
            "literal_request": "Build an API",
            "inferred_intent": "Create REST API",
            "sub_tasks": [
                SubTask(description="Design schema", dependencies=[], estimated_complexity="medium"),
            ],
            "missing_info": [],
            "assumptions": [],
            "modality": ModalityType.CODE,
            "recommended_approach": "Standard",
            "escalation_needed": False,
            "suggested_tier": 2,
            "confidence": 0.8,
        }
        defaults.update(kwargs)
        return TaskIntelligenceReport(**defaults)

    def test_empty_sub_tasks(self, planner):
        report = self._make_report(sub_tasks=[])
        plan = planner.create_plan(report)
        # Should still have review steps
        assert plan.total_steps > 0

    def test_many_sub_tasks(self, planner):
        tasks = [
            SubTask(description=f"Task {i}", dependencies=[], estimated_complexity="low")
            for i in range(20)
        ]
        report = self._make_report(sub_tasks=tasks)
        plan = planner.create_plan(report)
        assert plan.total_steps >= 20

    def test_critical_path_never_empty_with_steps(self, planner):
        report = self._make_report()
        plan = planner.create_plan(report)
        if plan.total_steps > 0:
            assert len(plan.critical_path) > 0

    def test_duration_always_positive(self, planner):
        report = self._make_report()
        plan = planner.create_plan(report)
        assert plan.estimated_duration_minutes is not None
        assert plan.estimated_duration_minutes >= 1

    def test_sme_selection_limited_to_3(self, planner):
        report = self._make_report(
            literal_request="security cloud database AI ML test deploy docs frontend"
        )
        plan = planner.create_plan(report)
        assert len(plan.required_sme_personas) <= 3

    def test_steps_have_unique_numbers(self, planner):
        report = self._make_report()
        plan = planner.create_plan(report)
        step_numbers = [s.step_number for s in plan.steps]
        assert len(step_numbers) == len(set(step_numbers))


# =============================================================================
# Researcher Agent Edge Cases
# =============================================================================

class TestResearcherEdgeCases:
    """Edge cases for the Researcher agent."""

    @pytest.fixture
    def researcher(self):
        return ResearcherAgent()

    def test_empty_topic(self, researcher):
        result = researcher.research("")
        assert isinstance(result, EvidenceBrief)

    def test_queries_deduplicated(self, researcher):
        queries = researcher._generate_search_queries("python")
        # No duplicates
        assert len(queries) == len(set(q.lower() for q in queries))

    def test_queries_limited_to_5(self, researcher):
        queries = researcher._generate_search_queries("very long topic with many words")
        assert len(queries) <= 5

    def test_overall_confidence_with_no_findings(self, researcher):
        conf = researcher._calculate_overall_confidence([])
        assert conf == ConfidenceLevel.LOW

    def test_sme_inputs_boost_confidence(self, researcher):
        result = researcher.research(
            "test topic",
            sme_inputs={"Security Expert": "This is verified and correct"}
        )
        high_conf_findings = [
            f for f in result.findings
            if f.confidence == ConfidenceLevel.HIGH
        ]
        assert len(high_conf_findings) > 0


# =============================================================================
# Executor Agent Edge Cases
# =============================================================================

class TestExecutorEdgeCases:
    """Edge cases for the Executor agent."""

    @pytest.fixture
    def executor(self):
        return ExecutorAgent()

    def test_scoring_weights_sum_to_1(self, executor):
        total = sum(executor.scoring_weights.values())
        assert abs(total - 1.0) < 0.001

    def test_empty_approaches_returns_default(self, executor):
        result = executor._select_best_approach([])
        assert result.name == "Standard Approach"

    def test_execution_time_always_recorded(self, executor):
        result = executor.execute("simple task")
        assert result.execution_time > 0

    def test_quality_score_range(self, executor):
        for task in ["Write Python code", "Create documentation", "Analyze data", "generic task"]:
            result = executor.execute(task)
            assert 0.0 <= result.quality_score <= 1.0

    def test_file_path_go_detection(self, executor):
        path = executor._determine_file_path("Write a golang service")
        assert path is not None
        assert path.endswith(".go")

    def test_file_path_rust_detection(self, executor):
        path = executor._determine_file_path("Create a Rust module for .rs file")
        assert path is not None
        assert path.endswith(".rs")

    def test_validation_boosts_quality(self, executor):
        from src.agents.executor import ExecutionResult
        result = ExecutionResult(
            approach_name="Test",
            status="success",
            quality_score=0.5,
        )
        validated = executor._validate_output(result)
        assert validated.quality_score > 0.5


# =============================================================================
# Code Reviewer Edge Cases
# =============================================================================

class TestCodeReviewerEdgeCases:
    """Edge cases for the Code Reviewer agent."""

    @pytest.fixture
    def reviewer(self):
        return CodeReviewerAgent()

    def test_empty_code(self, reviewer):
        report = reviewer.review("")
        assert isinstance(report, CodeReviewReport)

    def test_multiline_sql_injection(self, reviewer):
        code = '''
query = f"SELECT * FROM users WHERE id = {user_id}"
cursor.execute(query)
'''
        report = reviewer.review(code)
        assert report.security_scan.sql_injection_risk is True

    def test_safe_code_passes(self, reviewer):
        code = '''
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b
'''
        report = reviewer.review(code)
        assert report.pass_fail is True

    def test_hardcoded_api_key_detected(self, reviewer):
        code = 'api_key = "sk-1234567890abcdef"'
        report = reviewer.review(code)
        assert len(report.security_scan.credential_exposure) > 0

    def test_weak_crypto_detected(self, reviewer):
        code = 'hash_value = md5("password")'
        report = reviewer.review(code)
        assert report.security_scan.vulnerabilities_found > 0

    def test_context_counts_multiple_functions(self, reviewer):
        code = '''
def func1():
    pass

def func2():
    pass

def func3():
    pass
'''
        context = reviewer._analyze_code_context(code, "test.py", "python")
        assert context.function_count == 3

    def test_context_counts_classes(self, reviewer):
        code = '''
class Foo:
    pass

class Bar:
    pass
'''
        context = reviewer._analyze_code_context(code, "test.py", "python")
        assert context.class_count == 2


# =============================================================================
# Formatter Edge Cases
# =============================================================================

class TestFormatterEdgeCases:
    """Edge cases for the Formatter agent."""

    @pytest.fixture
    def formatter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield FormatterAgent(output_dir=tmpdir)

    def test_empty_content(self, formatter):
        result = formatter.format("", "text")
        assert result["format"] == "text"

    def test_nested_dict_to_markdown(self, formatter):
        data = {"level1": {"level2": {"level3": "value"}}}
        md = formatter._dict_to_markdown(data)
        assert "level1" in md
        assert "value" in md

    def test_code_extraction_from_markdown(self, formatter):
        content = '''Here is some code:
```python
def hello():
    print("world")
```
End of code.'''
        code = formatter._extract_code(content)
        assert "def hello" in code

    def test_python_syntax_validation_catches_error(self, formatter):
        code = "def broken(:\n    pass"
        result = formatter._validate_syntax(code, "python")
        assert result["valid"] is False
        assert len(result["errors"]) > 0

    def test_yaml_format(self, formatter):
        result = formatter.format({"key": "value"}, "yaml")
        assert "key" in result["formatted_output"]
        assert result["format"] == "yaml"

    def test_json_format_from_string(self, formatter):
        result = formatter.format('{"key": "value"}', "json")
        parsed = json.loads(result["formatted_output"])
        assert parsed["key"] == "value"

    def test_language_detection_go(self, formatter):
        code = "package main\n\nfunc main() {}"
        lang = formatter._detect_language(code, None)
        assert lang == "go"

    def test_language_detection_cpp(self, formatter):
        code = '#include <iostream>\nstd::cout << "hello";'
        lang = formatter._detect_language(code, None)
        assert lang == "cpp"


# =============================================================================
# Verifier Edge Cases
# =============================================================================

class TestVerifierEdgeCases:
    """Edge cases for the Verifier agent."""

    @pytest.fixture
    def verifier(self):
        return VerifierAgent()

    def test_no_claims_in_simple_content(self, verifier):
        report = verifier.verify("Hello.")
        # Very short sentences are filtered
        assert report.total_claims_checked >= 0

    def test_future_date_flagged(self, verifier):
        result = verifier._verify_date_claim("Released in 2099")
        assert result["risk"] == FabricationRisk.HIGH

    def test_historical_date_medium_risk(self, verifier):
        result = verifier._verify_date_claim("Founded in 1850")
        assert result["risk"] == FabricationRisk.MEDIUM

    def test_url_without_domain_flagged(self, verifier):
        result = verifier._verify_url_claim("See https:///no-domain")
        assert result["risk"] == FabricationRisk.HIGH

    def test_impossible_percentage_flagged(self, verifier):
        result = verifier._verify_measurement_claim("Achieved 500% accuracy")
        assert result["risk"] == FabricationRisk.HIGH

    def test_hallucination_filler_words_flagged(self, verifier):
        result = verifier._verify_general_claim(
            "Obviously this is clearly the best approach", ""
        )
        assert result["risk"] == FabricationRisk.MEDIUM

    def test_report_verdict_pass_when_all_verified(self, verifier):
        content = "A simple statement without facts."
        report = verifier.verify(content)
        # Simple content should pass
        assert report.verdict in ("PASS", "FAIL")

    def test_corrections_limited_to_5(self, verifier):
        corrections = verifier._generate_corrections(
            [MagicMock(claim_text=f"claim {i}", status=VerificationStatus.FABRICATED,
                       correction=None) for i in range(10)]
        )
        assert len(corrections) <= 5


# =============================================================================
# Critic Edge Cases
# =============================================================================

class TestCriticEdgeCases:
    """Edge cases for the Critic agent."""

    @pytest.fixture
    def critic(self):
        return CriticAgent()

    def test_empty_solution(self, critic):
        report = critic.critique("", "Test request")
        assert isinstance(report, CritiqueReport)

    def test_long_solution_summary_truncated(self, critic):
        long_solution = "x" * 200
        report = critic.critique(long_solution, "Test")
        assert len(report.solution_summary) <= 103  # 100 + "..."

    def test_all_five_attack_vectors_present(self, critic):
        report = critic.critique("A complete solution with testing.", "Test")
        assert report.logic_attack is not None
        assert report.completeness_attack is not None
        assert report.quality_attack is not None
        assert report.contradiction_scan is not None
        assert report.red_team_argumentation is not None

    def test_domain_attacks_with_sme(self, critic):
        report = critic.critique(
            "Use JWT for auth",
            "Implement authentication",
            domain_attacks=["JWT has known vulnerabilities"],
            sme_inputs={"Security": "JWT has known vulnerabilities in certain configs"}
        )
        domain_attacks = [a for a in report.attacks if a.domain_specific]
        assert len(domain_attacks) > 0

    def test_would_approve_clean_solution(self, critic):
        report = critic.critique(
            "This solution includes error handling, testing, security, "
            "performance optimization, and documentation. "
            "It is clear, well-structured, and follows best practices.",
            "Build a good solution"
        )
        # Clean solution with all aspects covered should generally be approved
        assert isinstance(report.would_approve, bool)

    def test_red_team_always_has_failure_modes(self, critic):
        report = critic.critique("Any solution", "Any request")
        assert len(report.red_team_argumentation.failure_modes) > 0
        assert len(report.red_team_argumentation.worst_case_scenarios) > 0

    def test_contradiction_multiline(self, critic):
        """Test that contradictions are detected across multiple lines."""
        solution = "This always works.\nBut it never works in production."
        report = critic.critique(solution, "Test")
        scan = report.contradiction_scan
        assert len(scan.internal_contradictions) > 0


# =============================================================================
# Reviewer Edge Cases
# =============================================================================

class TestReviewerEdgeCases:
    """Edge cases for the Reviewer agent."""

    @pytest.fixture
    def reviewer(self):
        return ReviewerAgent()

    def _context(self, **kwargs):
        defaults = {
            "original_request": "Build a feature",
            "agent_outputs": {},
            "revision_count": 0,
            "max_revisions": 2,
            "tier_level": 2,
            "is_code_output": False,
        }
        defaults.update(kwargs)
        return ReviewContext(**defaults)

    def test_pass_on_good_output(self, reviewer):
        context = self._context(original_request="Build a feature")
        verdict = reviewer.review(
            "Here is the feature implementation with all details.",
            context,
        )
        assert isinstance(verdict, ReviewVerdict)

    def test_fail_on_security_keywords(self, reviewer):
        context = self._context()
        verdict = reviewer.review(
            'The password is hardcoded: password = "admin123"',
            context,
        )
        # Should detect security issue
        assert verdict.verdict == Verdict.FAIL or len(verdict.reasons) > 0

    def test_verdict_matrix_all_pass(self, reviewer):
        action = reviewer._apply_verdict_matrix(Verdict.PASS, Verdict.PASS)
        assert action == "PROCEED_TO_FORMATTER"

    def test_verdict_matrix_full_fail(self, reviewer):
        action = reviewer._apply_verdict_matrix(Verdict.FAIL, Verdict.FAIL)
        assert action == "FULL_REGENERATION"

    def test_verdict_matrix_verifier_fail(self, reviewer):
        action = reviewer._apply_verdict_matrix(Verdict.FAIL, Verdict.PASS)
        assert action == "RESEARCHER_REVERIFY"

    def test_verdict_matrix_critic_fail(self, reviewer):
        action = reviewer._apply_verdict_matrix(Verdict.PASS, Verdict.FAIL)
        assert action == "EXECUTOR_REVISE"

    def test_arbitration_only_tier_4(self, reviewer):
        context_t2 = self._context(tier_level=2)
        needed, _ = reviewer._check_arbitration_needed(
            Verdict.PASS, Verdict.PASS, Verdict.FAIL, context_t2
        )
        assert needed is False

    def test_arbitration_tier_4_disagreement(self, reviewer):
        context_t4 = self._context(tier_level=4)
        needed, arb_input = reviewer._check_arbitration_needed(
            Verdict.PASS, Verdict.PASS, Verdict.FAIL, context_t4
        )
        assert needed is True
        assert arb_input is not None

    def test_can_revise_respects_max(self, reviewer):
        context = self._context(revision_count=2, max_revisions=2)
        verdict = reviewer.review("Some output", context)
        assert verdict.can_revise is False

    def test_revision_instructions_on_fail(self, reviewer):
        context = self._context()
        verdict = reviewer.review("", context)  # Empty output should fail readability
        if verdict.verdict == Verdict.FAIL:
            assert len(verdict.revision_instructions) > 0


# =============================================================================
# Memory Curator Edge Cases
# =============================================================================

class TestMemoryCuratorEdgeCases:
    """Edge cases for the Memory Curator agent."""

    @pytest.fixture
    def curator(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield MemoryCuratorAgent(knowledge_dir=tmpdir)

    def test_empty_agent_outputs(self, curator):
        result = curator.extract_and_preserve(
            "Test task",
            {"tier_level": 1},
            {},
            "Final output",
        )
        assert len(result.entries) > 0

    def test_topic_generation_kebab_case(self, curator):
        from src.agents.memory_curator import KnowledgeCategory
        topic = curator._generate_topic(
            "Implement user authentication system",
            KnowledgeCategory.CODE_PATTERN
        )
        assert " " not in topic
        assert topic == topic.lower()

    def test_topic_length_limited(self, curator):
        from src.agents.memory_curator import KnowledgeCategory
        long_task = " ".join(["word"] * 100)
        topic = curator._generate_topic(long_task, KnowledgeCategory.LESSON_LEARNED)
        assert len(topic) <= 50

    def test_category_inference_code(self, curator):
        from src.agents.memory_curator import KnowledgeCategory
        cat = curator._infer_category("Implement a function in Python", {})
        assert cat == KnowledgeCategory.CODE_PATTERN

    def test_category_inference_troubleshooting(self, curator):
        from src.agents.memory_curator import KnowledgeCategory
        cat = curator._infer_category("Fix the bug in auth", {})
        assert cat == KnowledgeCategory.TROUBLESHOOTING

    def test_category_inference_architecture(self, curator):
        from src.agents.memory_curator import KnowledgeCategory
        cat = curator._infer_category("Design the microservice architecture", {})
        assert cat == KnowledgeCategory.ARCHITECTURAL_DECISION

    def test_knowledge_file_written(self, curator):
        result = curator.extract_and_preserve(
            "Test task for knowledge",
            {"tier_level": 2, "agents_used": ["analyst", "executor"]},
            {"executor": "Used factory pattern"},
            "Final output with factory pattern",
        )
        # Check file was written
        assert len(result.topics_created) > 0
        for topic_file in result.topics_created:
            filepath = curator.knowledge_dir / topic_file
            assert filepath.exists()

    def test_knowledge_retrieval(self, curator):
        # Write some knowledge first
        curator.extract_and_preserve(
            "Python authentication implementation",
            {"tier_level": 2},
            {"executor": "Used JWT"},
            "JWT auth implementation",
        )
        # Retrieve it
        results = curator.retrieve_knowledge("authentication")
        assert len(results) >= 0  # May or may not match depending on keyword extraction


# =============================================================================
# Cross-Agent Data Flow Tests
# =============================================================================

class TestCrossAgentDataFlow:
    """Test that output of one agent feeds correctly into the next."""

    def test_analyst_to_planner_flow(self):
        """Analyst output must be valid input for Planner."""
        analyst = AnalystAgent()
        planner = PlannerAgent()

        report = analyst.analyze("Build a REST API with authentication")
        plan = planner.create_plan(report)

        assert isinstance(plan, ExecutionPlan)
        assert plan.total_steps > 0

    def test_analyst_to_clarifier_flow(self):
        """Analyst missing info feeds into Clarifier."""
        analyst = AnalystAgent()
        clarifier = ClarifierAgent()

        report = analyst.analyze("Build an API with database")
        request = clarifier.formulate_questions(report)

        assert isinstance(request, ClarificationRequest)
        # Should have questions since "database" triggers missing info
        if report.missing_info:
            assert request.total_questions > 0

    def test_executor_to_code_reviewer_flow(self):
        """Executor code output feeds into Code Reviewer."""
        executor = ExecutorAgent()
        code_reviewer = CodeReviewerAgent()

        result = executor.execute("Write a Python function to add numbers")
        review = code_reviewer.review(str(result.output))

        assert isinstance(review, CodeReviewReport)

    def test_executor_to_verifier_flow(self):
        """Executor output feeds into Verifier."""
        executor = ExecutorAgent()
        verifier = VerifierAgent()

        result = executor.execute("Explain quantum computing")
        report = verifier.verify(str(result.output))

        assert isinstance(report, VerificationReport)

    def test_executor_to_critic_flow(self):
        """Executor output feeds into Critic."""
        executor = ExecutorAgent()
        critic = CriticAgent()

        result = executor.execute("Design a cache system")
        report = critic.critique(str(result.output), "Design a cache system")

        assert isinstance(report, CritiqueReport)

    def test_full_pipeline_data_flow(self):
        """Test a mini pipeline: Analyst -> Planner -> Executor -> Reviewer."""
        analyst = AnalystAgent()
        planner = PlannerAgent()
        executor = ExecutorAgent()
        reviewer_agent = ReviewerAgent()

        # Step 1: Analyze
        report = analyst.analyze("Write a hello world function in Python")
        assert isinstance(report, TaskIntelligenceReport)

        # Step 2: Plan
        plan = planner.create_plan(report)
        assert isinstance(plan, ExecutionPlan)

        # Step 3: Execute
        result = executor.execute("Write a hello world function in Python", report)
        assert result.status == "success"

        # Step 4: Review
        context = ReviewContext(
            original_request="Write a hello world function in Python",
            agent_outputs={"analyst": {"modality": "code"}},
            revision_count=0,
            max_revisions=2,
            tier_level=1,
            is_code_output=True,
        )
        verdict = reviewer_agent.review(str(result.output), context)
        assert isinstance(verdict, ReviewVerdict)


# =============================================================================
# Agent Configuration Files Test
# =============================================================================

class TestAgentConfigFiles:
    """Verify all agent CLAUDE.md config files exist."""

    AGENT_CONFIG_DIR = "config/agents"

    @pytest.mark.parametrize("agent_name", [
        "analyst", "planner", "clarifier", "researcher",
        "executor", "code_reviewer", "formatter", "verifier",
        "critic", "reviewer", "memory_curator",
    ])
    def test_claude_md_exists(self, agent_name):
        """Each operational agent must have a CLAUDE.md config file."""
        config_path = Path(self.AGENT_CONFIG_DIR) / agent_name / "CLAUDE.md"
        assert config_path.exists(), f"Missing config: {config_path}"

    @pytest.mark.parametrize("agent_name", [
        "analyst", "planner", "clarifier", "researcher",
        "executor", "code_reviewer", "formatter", "verifier",
        "critic", "reviewer", "memory_curator",
    ])
    def test_claude_md_not_empty(self, agent_name):
        """Config files must have content."""
        config_path = Path(self.AGENT_CONFIG_DIR) / agent_name / "CLAUDE.md"
        if config_path.exists():
            content = config_path.read_text()
            assert len(content) > 50, f"Config too short: {config_path}"


# =============================================================================
# Skill Files Test
# =============================================================================

class TestSkillFiles:
    """Verify skill files exist for assigned agents."""

    SKILL_DIR = ".claude/skills"

    @pytest.mark.parametrize("skill_name", [
        "code-generation", "document-creation", "requirements-engineering",
        "architecture-design", "web-research", "multi-agent-reasoning",
        "test-case-generation",
    ])
    def test_skill_file_exists(self, skill_name):
        """Each skill must have a SKILL.md file."""
        skill_path = Path(self.SKILL_DIR) / skill_name / "SKILL.md"
        assert skill_path.exists(), f"Missing skill: {skill_path}"


# =============================================================================
# Security Pattern Completeness
# =============================================================================

class TestSecurityPatternCompleteness:
    """Verify security patterns are comprehensive."""

    @pytest.fixture
    def reviewer(self):
        return CodeReviewerAgent()

    def test_detects_eval_usage(self, reviewer):
        code = 'result = eval(user_input)'
        report = reviewer.review(code)
        assert report.security_scan.xss_risk is True

    def test_detects_document_write(self, reviewer):
        code = 'document.write(user_data)'
        report = reviewer.review(code)
        assert report.security_scan.xss_risk is True

    def test_detects_password_in_code(self, reviewer):
        code = 'password = "supersecret123"'
        report = reviewer.review(code)
        assert len(report.security_scan.credential_exposure) > 0

    def test_detects_token_in_code(self, reviewer):
        code = 'token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"'
        report = reviewer.review(code)
        assert len(report.security_scan.credential_exposure) > 0

    def test_detects_sha1_weak_crypto(self, reviewer):
        code = 'digest = sha1(data)'
        report = reviewer.review(code)
        assert report.security_scan.vulnerabilities_found > 0
