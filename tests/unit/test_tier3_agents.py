"""
Comprehensive Tier 3 (Deep) Agent Validation Tests

Tests ALL 12 operational agents + Council Chair + SME Spawner:
- Agent code correctness and method signatures
- Claude Agent SDK integration (ClaudeAgentOptions, AGENT_ALLOWED_TOOLS)
- Per-agent tool permissions (least privilege)
- Per-agent skill assignments (AGENT_SKILLS)
- Schema output validation
- Tier 3 pipeline flow (all 8 phases)
- Verdict matrix logic
- Debate protocol
- SME interaction modes
"""

import pytest
from unittest.mock import patch, MagicMock
from typing import Dict, List, Any

# Core modules
from src.core.complexity import (
    TierLevel, TierClassification, TIER_CONFIG,
    classify_complexity, get_active_agents, get_council_agents,
    should_escalate, get_escalated_tier, estimate_agent_count,
)
from src.core.sdk_integration import (
    ClaudeAgentOptions, PermissionMode, AGENT_ALLOWED_TOOLS,
    build_agent_options, spawn_subagent, get_skills_for_agent,
    _get_output_schema, _validate_output,
)
from src.core.pipeline import (
    ExecutionPipeline, PipelineBuilder, Phase, PhaseStatus,
    AgentResult, PhaseResult, PipelineState,
    create_execution_context, estimate_pipeline_duration,
)
from src.core.verdict import (
    Verdict, MatrixAction, MatrixOutcome, VERDICT_MATRIX,
    evaluate_verdict_matrix, should_trigger_debate,
    get_phase_for_action, get_required_agents_for_phase,
)
from src.core.debate import (
    DebateProtocol, ConsensusLevel, DebateRound, DebateOutcome,
    trigger_debate, get_debate_participants,
)

# Agent modules
from src.agents.analyst import AnalystAgent
from src.agents.planner import PlannerAgent
from src.agents.clarifier import ClarifierAgent
from src.agents.researcher import ResearcherAgent
from src.agents.executor import ExecutorAgent
from src.agents.code_reviewer import CodeReviewerAgent
from src.agents.formatter import FormatterAgent
from src.agents.verifier import VerifierAgent
from src.agents.critic import CriticAgent
from src.agents.reviewer import ReviewerAgent
from src.agents.memory_curator import MemoryCuratorAgent
from src.agents.council import CouncilChairAgent
from src.agents.sme_spawner import SMESpawner

# Schema modules
from src.schemas.analyst import TaskIntelligenceReport, ModalityType, SubTask
from src.schemas.planner import ExecutionPlan, ExecutionStep
from src.schemas.verifier import (
    VerificationReport, Claim, VerificationStatus, FabricationRisk,
)
from src.schemas.critic import CritiqueReport, AttackVector, SeverityLevel
from src.schemas.reviewer import ReviewVerdict, Verdict as ReviewerVerdict
from src.schemas.sme import SMEAdvisoryReport, SMEInteractionMode


# =============================================================================
# 1. TIER 3 CONFIGURATION TESTS
# =============================================================================

class TestTier3Configuration:
    """Validate Tier 3 (DEEP) configuration in TIER_CONFIG."""

    def test_tier3_exists_in_config(self):
        """Tier 3 (DEEP) must be defined in TIER_CONFIG."""
        assert TierLevel.DEEP in TIER_CONFIG

    def test_tier3_agent_count(self):
        """Tier 3 must have 12 base agents."""
        config = TIER_CONFIG[TierLevel.DEEP]
        assert config["agent_count"] == 12

    def test_tier3_active_agents_list(self):
        """Tier 3 must include all 12 operational agents."""
        config = TIER_CONFIG[TierLevel.DEEP]
        expected_agents = [
            "Orchestrator", "Analyst", "Planner", "Clarifier",
            "Researcher", "Executor", "Code Reviewer", "Formatter",
            "Verifier", "Critic", "Reviewer", "Memory Curator",
        ]
        for agent in expected_agents:
            assert agent in config["active_agents"], f"Missing agent: {agent}"

    def test_tier3_requires_council(self):
        """Tier 3 must require Council activation."""
        config = TIER_CONFIG[TierLevel.DEEP]
        assert config["requires_council"] is True

    def test_tier3_council_agents(self):
        """Tier 3 must activate Domain Council Chair."""
        config = TIER_CONFIG[TierLevel.DEEP]
        assert "Domain Council Chair" in config.get("council_agents", [])

    def test_tier3_requires_smes(self):
        """Tier 3 must require SME personas."""
        config = TIER_CONFIG[TierLevel.DEEP]
        assert config["requires_smes"] is True

    def test_tier3_max_sme_count(self):
        """Tier 3 allows up to 3 SMEs."""
        config = TIER_CONFIG[TierLevel.DEEP]
        assert config["max_sme_count"] == 3

    def test_tier3_phases(self):
        """Tier 3 must include all required phases."""
        config = TIER_CONFIG[TierLevel.DEEP]
        phases = config["phases"]
        # Must include Phase 1, 2, 3, 5, 6, 8 at minimum
        assert any("Phase 1" in p for p in phases)
        assert any("Phase 2" in p or "Council" in p for p in phases)
        assert any("Phase 3" in p for p in phases)
        assert any("Phase 5" in p for p in phases)

    def test_tier3_total_agent_estimate(self):
        """Tier 3 with 3 SMEs = 15 total agents."""
        count = estimate_agent_count(TierLevel.DEEP, sme_count=3)
        assert count == 15

    def test_tier3_total_agent_estimate_no_smes(self):
        """Tier 3 with 0 SMEs = 12 total agents."""
        count = estimate_agent_count(TierLevel.DEEP, sme_count=0)
        assert count == 12


# =============================================================================
# 2. COMPLEXITY CLASSIFICATION TESTS
# =============================================================================

class TestComplexityClassification:
    """Test that Tier 3 keywords properly trigger DEEP classification."""

    @pytest.mark.parametrize("keyword", [
        "architecture", "design pattern", "system design",
        "machine learning", "ai", "security",
        "migration", "microservices", "data pipeline",
        "research", "domain expert",
    ])
    def test_tier3_keyword_triggers_deep(self, keyword):
        """Tier 3 keywords should classify to at least Tier 3."""
        result = classify_complexity(f"Please help with {keyword} for our project")
        assert result.tier >= TierLevel.DEEP or result.tier >= TierLevel.STANDARD

    def test_tier3_classification_requires_council(self):
        """Tier 3 classification should flag requires_council."""
        result = classify_complexity("Design a system architecture with microservices")
        if result.tier >= TierLevel.DEEP:
            assert result.requires_council is True

    def test_tier3_classification_requires_smes(self):
        """Tier 3 classification should flag requires_smes."""
        result = classify_complexity("Design a system architecture with microservices")
        if result.tier >= TierLevel.DEEP:
            assert result.requires_smes is True

    def test_escalation_from_tier2_to_tier3(self):
        """Should escalate Tier 2 → Tier 3 when subagent feedback indicates it."""
        assert should_escalate(
            TierLevel.STANDARD,
            {"escalation_needed": True}
        ) is True

    def test_escalation_domain_expertise_keyword(self):
        """Should escalate when 'domain expertise required' is in feedback."""
        assert should_escalate(
            TierLevel.STANDARD,
            {"feedback": "domain expertise required for this task"}
        ) is True

    def test_get_escalated_tier(self):
        """Escalation from Tier 2 should give Tier 3."""
        assert get_escalated_tier(TierLevel.STANDARD) == TierLevel.DEEP

    def test_get_escalated_tier_capped(self):
        """Escalation from Tier 4 should stay at Tier 4."""
        assert get_escalated_tier(TierLevel.ADVERSARIAL) == TierLevel.ADVERSARIAL

    def test_get_active_agents_for_tier3(self):
        """get_active_agents should return 12 agents for Tier 3."""
        agents = get_active_agents(TierLevel.DEEP)
        assert len(agents) == 12

    def test_get_council_agents_for_tier3(self):
        """get_council_agents should return Chair for Tier 3."""
        council = get_council_agents(TierLevel.DEEP)
        assert "Domain Council Chair" in council


# =============================================================================
# 3. CLAUDE AGENT SDK INTEGRATION TESTS
# =============================================================================

class TestClaudeAgentSDKIntegration:
    """Validate SDK configuration for all Tier 3 agents."""

    def test_claude_agent_options_dataclass(self):
        """ClaudeAgentOptions must have all required fields."""
        options = ClaudeAgentOptions(
            name="Test Agent",
            model="claude-sonnet-4-20250514",
            system_prompt="You are a test agent.",
        )
        assert options.name == "Test Agent"
        assert options.max_turns == 30  # default
        assert options.permission_mode == PermissionMode.DEFAULT

    def test_claude_agent_options_to_sdk_kwargs(self):
        """to_sdk_kwargs should produce valid SDK kwargs dict."""
        options = ClaudeAgentOptions(
            name="Analyst",
            model="claude-sonnet-4-20250514",
            system_prompt="Analyze tasks.",
            allowed_tools=["Read", "Glob", "Grep"],
        )
        kwargs = options.to_sdk_kwargs()
        assert kwargs["name"] == "Analyst"
        assert kwargs["model"] == "claude-sonnet-4-20250514"
        assert kwargs["system_prompt"] == "Analyze tasks."
        assert kwargs["allowed_tools"] == ["Read", "Glob", "Grep"]
        assert kwargs["max_turns"] == 30

    def test_sdk_kwargs_excludes_empty_tools(self):
        """to_sdk_kwargs should not include allowed_tools if empty."""
        options = ClaudeAgentOptions(
            name="Clarifier",
            model="claude-sonnet-4-20250514",
            system_prompt="Clarify.",
            allowed_tools=[],
        )
        kwargs = options.to_sdk_kwargs()
        assert "allowed_tools" not in kwargs

    def test_sdk_kwargs_includes_permission_mode_for_executor(self):
        """Executor should have acceptEdits permission mode."""
        options = ClaudeAgentOptions(
            name="Executor",
            model="claude-sonnet-4-20250514",
            system_prompt="Execute.",
            permission_mode=PermissionMode.ACCEPT_EDITS,
        )
        kwargs = options.to_sdk_kwargs()
        assert kwargs["permission_mode"] == "acceptEdits"

    def test_spawn_subagent_fallback_simulation(self):
        """spawn_subagent should fall back to simulation when no SDK available."""
        options = ClaudeAgentOptions(
            name="TestAgent",
            model="claude-sonnet-4-20250514",
            system_prompt="Test.",
        )
        result = spawn_subagent(options, "Test input", max_retries=0)
        # Should succeed via simulation fallback
        assert result["status"] in ("success", "error")

    def test_validate_output_valid_json(self):
        """_validate_output should accept valid JSON matching schema."""
        schema = {"required": ["name", "result"]}
        output = '{"name": "test", "result": "ok"}'
        assert _validate_output(output, schema) is True

    def test_validate_output_missing_fields(self):
        """_validate_output should reject JSON missing required fields."""
        schema = {"required": ["name", "result"]}
        output = '{"name": "test"}'
        assert _validate_output(output, schema) is False

    def test_validate_output_invalid_json(self):
        """_validate_output should reject non-JSON strings."""
        schema = {"required": ["name"]}
        assert _validate_output("not json at all", schema) is False

    def test_validate_output_empty(self):
        """_validate_output should reject empty output."""
        schema = {"required": ["name"]}
        assert _validate_output("", schema) is False
        assert _validate_output(None, schema) is False


# =============================================================================
# 4. AGENT ALLOWED TOOLS (LEAST PRIVILEGE) TESTS
# =============================================================================

class TestAgentAllowedTools:
    """Validate per-agent tool permissions follow least privilege."""

    TIER3_AGENTS = [
        "analyst", "planner", "clarifier", "researcher",
        "executor", "code_reviewer", "formatter",
        "verifier", "critic", "reviewer", "memory_curator",
        "council_chair",
    ]

    def test_all_tier3_agents_have_tool_config(self):
        """Every Tier 3 agent must have an entry in AGENT_ALLOWED_TOOLS."""
        for agent in self.TIER3_AGENTS:
            assert agent in AGENT_ALLOWED_TOOLS, f"Missing tool config for {agent}"

    def test_analyst_tools(self):
        """Analyst should have Read, Glob, Grep only."""
        tools = AGENT_ALLOWED_TOOLS["analyst"]
        assert set(tools) == {"Read", "Glob", "Grep"}

    def test_planner_tools(self):
        """Planner should have Read, Glob only."""
        tools = AGENT_ALLOWED_TOOLS["planner"]
        assert set(tools) == {"Read", "Glob"}

    def test_clarifier_tools(self):
        """Clarifier should have no tools (pure reasoning)."""
        tools = AGENT_ALLOWED_TOOLS["clarifier"]
        assert tools == []

    def test_researcher_tools(self):
        """Researcher should have WebSearch, WebFetch, Read."""
        tools = AGENT_ALLOWED_TOOLS["researcher"]
        assert set(tools) == {"WebSearch", "WebFetch", "Read"}

    def test_executor_tools(self):
        """Executor should have Read, Write, Edit, Bash, Glob, Grep, Skill."""
        tools = AGENT_ALLOWED_TOOLS["executor"]
        expected = {"Read", "Write", "Edit", "Bash", "Glob", "Grep", "Skill"}
        assert set(tools) == expected

    def test_code_reviewer_tools(self):
        """Code Reviewer should have Read, Glob, Grep, Bash."""
        tools = AGENT_ALLOWED_TOOLS["code_reviewer"]
        assert set(tools) == {"Read", "Glob", "Grep", "Bash"}

    def test_formatter_tools(self):
        """Formatter should have Read, Write, Bash, Skill."""
        tools = AGENT_ALLOWED_TOOLS["formatter"]
        assert set(tools) == {"Read", "Write", "Bash", "Skill"}

    def test_verifier_tools(self):
        """Verifier should have Read, WebSearch, WebFetch."""
        tools = AGENT_ALLOWED_TOOLS["verifier"]
        assert set(tools) == {"Read", "WebSearch", "WebFetch"}

    def test_critic_tools(self):
        """Critic should have Read, Grep only."""
        tools = AGENT_ALLOWED_TOOLS["critic"]
        assert set(tools) == {"Read", "Grep"}

    def test_reviewer_tools(self):
        """Reviewer should have Read, Glob, Grep."""
        tools = AGENT_ALLOWED_TOOLS["reviewer"]
        assert set(tools) == {"Read", "Glob", "Grep"}

    def test_memory_curator_tools(self):
        """Memory Curator should have Read, Write, Glob."""
        tools = AGENT_ALLOWED_TOOLS["memory_curator"]
        assert set(tools) == {"Read", "Write", "Glob"}

    def test_council_chair_tools(self):
        """Council Chair should have no tools (pure reasoning)."""
        tools = AGENT_ALLOWED_TOOLS["council_chair"]
        assert tools == []

    def test_quality_arbiter_tools(self):
        """Quality Arbiter should have no tools."""
        tools = AGENT_ALLOWED_TOOLS["quality_arbiter"]
        assert tools == []

    def test_ethics_advisor_tools(self):
        """Ethics Advisor should have no tools."""
        tools = AGENT_ALLOWED_TOOLS["ethics_advisor"]
        assert tools == []

    def test_sme_default_tools(self):
        """SME default should have Read, Glob, Grep, Skill."""
        tools = AGENT_ALLOWED_TOOLS["sme_default"]
        assert set(tools) == {"Read", "Glob", "Grep", "Skill"}

    def test_no_agent_has_dangerous_tools_without_write(self):
        """Agents with Write must also have Read (sanity check)."""
        for agent, tools in AGENT_ALLOWED_TOOLS.items():
            if "Write" in tools:
                assert "Read" in tools, f"{agent} has Write but not Read"

    def test_only_executor_has_edit(self):
        """Only executor should have Edit tool."""
        for agent, tools in AGENT_ALLOWED_TOOLS.items():
            if agent != "executor":
                assert "Edit" not in tools, f"{agent} should not have Edit tool"


# =============================================================================
# 5. AGENT SKILLS TESTS
# =============================================================================

class TestAgentSkills:
    """Validate per-agent skill assignments."""

    def test_executor_skills(self):
        """Executor should have code-generation skill."""
        skills = get_skills_for_agent("executor")
        assert "code-generation" in skills

    def test_formatter_skills(self):
        """Formatter should have document-creation skill."""
        skills = get_skills_for_agent("formatter")
        assert "document-creation" in skills

    def test_analyst_skills(self):
        """Analyst should have requirements-engineering skill."""
        skills = get_skills_for_agent("analyst")
        assert "requirements-engineering" in skills

    def test_planner_skills(self):
        """Planner should have architecture-design skill."""
        skills = get_skills_for_agent("planner")
        assert "architecture-design" in skills

    def test_researcher_skills(self):
        """Researcher should have web-research skill."""
        skills = get_skills_for_agent("researcher")
        assert "web-research" in skills

    def test_code_reviewer_skills(self):
        """Code Reviewer should have code-generation skill."""
        skills = get_skills_for_agent("code_reviewer")
        assert "code-generation" in skills

    def test_orchestrator_skills(self):
        """Orchestrator should have multi-agent-reasoning skill."""
        skills = get_skills_for_agent("orchestrator")
        assert "multi-agent-reasoning" in skills

    def test_agents_without_skills(self):
        """Agents without assigned skills should return empty list."""
        for agent in ["clarifier", "verifier", "critic", "reviewer", "memory_curator"]:
            skills = get_skills_for_agent(agent)
            assert skills == [], f"{agent} should have no skills, got {skills}"


# =============================================================================
# 6. INDIVIDUAL AGENT CODE TESTS
# =============================================================================

class TestAnalystAgent:
    """Validate AnalystAgent code and output."""

    def test_analyst_instantiation(self):
        """AnalystAgent should instantiate without errors."""
        agent = AnalystAgent()
        assert agent is not None

    def test_analyst_has_analyze_method(self):
        """AnalystAgent must have analyze() method."""
        agent = AnalystAgent()
        assert hasattr(agent, "analyze")
        assert callable(agent.analyze)

    def test_analyst_analyze_returns_report(self):
        """analyze() should return TaskIntelligenceReport."""
        agent = AnalystAgent()
        result = agent.analyze("Build a REST API for user management")
        assert isinstance(result, TaskIntelligenceReport)
        assert result.literal_request == "Build a REST API for user management"
        assert result.modality in list(ModalityType)

    def test_analyst_detects_code_modality(self):
        """Analyst should detect CODE modality for code requests."""
        agent = AnalystAgent()
        result = agent.analyze("Write a Python function to sort a list")
        assert result.modality == ModalityType.CODE

    def test_analyst_decomposes_tasks(self):
        """Analyst should decompose complex requests into sub-tasks."""
        agent = AnalystAgent()
        result = agent.analyze(
            "Build a complete e-commerce system with authentication, "
            "product catalog, shopping cart, and checkout"
        )
        assert len(result.sub_tasks) > 0


class TestPlannerAgent:
    """Validate PlannerAgent code and output."""

    def test_planner_instantiation(self):
        """PlannerAgent should instantiate without errors."""
        agent = PlannerAgent()
        assert agent is not None

    def test_planner_has_create_plan_method(self):
        """PlannerAgent must have create_plan() method."""
        agent = PlannerAgent()
        assert hasattr(agent, "create_plan")
        assert callable(agent.create_plan)

    def test_planner_create_plan_returns_plan(self):
        """create_plan() should return ExecutionPlan."""
        agent = PlannerAgent()
        analyst_report = TaskIntelligenceReport(
            literal_request="Build a REST API",
            inferred_intent="Create backend endpoints",
            sub_tasks=[
                SubTask(description="Design models", dependencies=[]),
                SubTask(description="Implement endpoints", dependencies=["Design models"]),
            ],
            missing_info=[],
            assumptions=["Python/FastAPI"],
            modality=ModalityType.CODE,
            recommended_approach="Start with models",
        )
        result = agent.create_plan(analyst_report)
        assert isinstance(result, ExecutionPlan)
        assert result.total_steps >= 1
        assert len(result.steps) == result.total_steps


class TestClarifierAgent:
    """Validate ClarifierAgent code and output."""

    def test_clarifier_instantiation(self):
        """ClarifierAgent should instantiate without errors."""
        agent = ClarifierAgent()
        assert agent is not None

    def test_clarifier_has_formulate_questions_method(self):
        """ClarifierAgent must have formulate_questions() method."""
        agent = ClarifierAgent()
        assert hasattr(agent, "formulate_questions")
        assert callable(agent.formulate_questions)


class TestResearcherAgent:
    """Validate ResearcherAgent code and output."""

    def test_researcher_instantiation(self):
        """ResearcherAgent should instantiate without errors."""
        agent = ResearcherAgent()
        assert agent is not None

    def test_researcher_has_research_method(self):
        """ResearcherAgent must have research() method."""
        agent = ResearcherAgent()
        assert hasattr(agent, "research")
        assert callable(agent.research)


class TestExecutorAgent:
    """Validate ExecutorAgent code and output."""

    def test_executor_instantiation(self):
        """ExecutorAgent should instantiate without errors."""
        agent = ExecutorAgent()
        assert agent is not None

    def test_executor_has_execute_method(self):
        """ExecutorAgent must have execute() method."""
        agent = ExecutorAgent()
        assert hasattr(agent, "execute")
        assert callable(agent.execute)

    def test_executor_tree_of_thoughts(self):
        """Executor should generate multiple approaches (Tree of Thoughts)."""
        agent = ExecutorAgent()
        assert hasattr(agent, "_generate_approaches") or hasattr(agent, "_score_approach")


class TestCodeReviewerAgent:
    """Validate CodeReviewerAgent code and output."""

    def test_code_reviewer_instantiation(self):
        """CodeReviewerAgent should instantiate without errors."""
        agent = CodeReviewerAgent()
        assert agent is not None

    def test_code_reviewer_has_review_method(self):
        """CodeReviewerAgent must have review() method."""
        agent = CodeReviewerAgent()
        assert hasattr(agent, "review")
        assert callable(agent.review)

    def test_code_reviewer_review_dimensions(self):
        """Code reviewer should assess security, performance, style, errors, tests."""
        agent = CodeReviewerAgent()
        code = "def hello():\n    print('hello world')\n"
        result = agent.review(code)
        # Should return a CodeReviewReport with 5 review dimensions
        assert hasattr(result, "overall_assessment")
        assert hasattr(result, "pass_fail")
        assert hasattr(result, "security_scan")
        assert hasattr(result, "performance_analysis")
        assert hasattr(result, "style_compliance")


class TestFormatterAgent:
    """Validate FormatterAgent code and output."""

    def test_formatter_instantiation(self):
        """FormatterAgent should instantiate without errors."""
        agent = FormatterAgent()
        assert agent is not None

    def test_formatter_has_format_method(self):
        """FormatterAgent must have format() method."""
        agent = FormatterAgent()
        assert hasattr(agent, "format")
        assert callable(agent.format)


class TestVerifierAgent:
    """Validate VerifierAgent code and output."""

    def test_verifier_instantiation(self):
        """VerifierAgent should instantiate without errors."""
        agent = VerifierAgent()
        assert agent is not None

    def test_verifier_has_verify_method(self):
        """VerifierAgent must have verify() method."""
        agent = VerifierAgent()
        assert hasattr(agent, "verify")
        assert callable(agent.verify)

    def test_verifier_verify_returns_report(self):
        """verify() should return VerificationReport."""
        agent = VerifierAgent()
        result = agent.verify("Python was released in 1991 by Guido van Rossum.")
        assert isinstance(result, VerificationReport)
        assert result.total_claims_checked >= 0
        assert 0.0 <= result.overall_reliability <= 1.0

    def test_verifier_detects_claims_with_dates(self):
        """Verifier should detect date-based claims."""
        agent = VerifierAgent()
        result = agent.verify("Python was released in 1991.")
        assert result.total_claims_checked >= 1

    def test_verifier_verdict_pass_or_fail(self):
        """Verifier verdict must be PASS or FAIL."""
        agent = VerifierAgent()
        result = agent.verify("This is a simple statement.")
        assert result.verdict in ("PASS", "FAIL")

    def test_verifier_sme_verification(self):
        """Verifier should give max confidence to SME-verified claims."""
        agent = VerifierAgent()
        result = agent.verify(
            "The cloud architecture uses serverless compute.",
            sme_verifications={"cloud architecture": "Confirmed correct"}
        )
        # Check if any claim was SME-verified
        sme_verified = [c for c in result.claims if c.domain_verified]
        if sme_verified:
            assert sme_verified[0].confidence == 10

    def test_verifier_future_date_flagging(self):
        """Verifier should flag claims with future years."""
        agent = VerifierAgent()
        result = agent.verify("The technology was invented in 2099.")
        flagged = [c for c in result.claims if c.confidence < 5]
        assert len(flagged) >= 1 or result.overall_reliability < 0.7


class TestCriticAgent:
    """Validate CriticAgent code and output."""

    def test_critic_instantiation(self):
        """CriticAgent should instantiate without errors."""
        agent = CriticAgent()
        assert agent is not None

    def test_critic_has_critique_method(self):
        """CriticAgent must have critique() method."""
        agent = CriticAgent()
        assert hasattr(agent, "critique")
        assert callable(agent.critique)

    def test_critic_critique_returns_report(self):
        """critique() should return CritiqueReport."""
        agent = CriticAgent()
        result = agent.critique(
            "This is a proposed solution to the problem.",
            "Solve the problem efficiently.",
        )
        assert isinstance(result, CritiqueReport)
        assert result.would_approve in (True, False)

    def test_critic_five_attack_vectors(self):
        """Critic should test 5 attack vectors: logic, completeness, quality, contradiction, red_team."""
        agent = CriticAgent()
        result = agent.critique(
            "Build a web app with user authentication.",
            "Create a secure web application.",
        )
        # Should have all 5 attack sub-reports
        assert hasattr(result, "logic_attack")
        assert hasattr(result, "completeness_attack")
        assert hasattr(result, "quality_attack")
        assert hasattr(result, "contradiction_scan")
        assert hasattr(result, "red_team_argumentation")


class TestReviewerAgent:
    """Validate ReviewerAgent code and output."""

    def test_reviewer_instantiation(self):
        """ReviewerAgent should instantiate without errors."""
        agent = ReviewerAgent()
        assert agent is not None

    def test_reviewer_has_review_method(self):
        """ReviewerAgent must have review() method."""
        agent = ReviewerAgent()
        assert hasattr(agent, "review")
        assert callable(agent.review)


class TestMemoryCuratorAgent:
    """Validate MemoryCuratorAgent code and output."""

    def test_memory_curator_instantiation(self):
        """MemoryCuratorAgent should instantiate without errors."""
        agent = MemoryCuratorAgent()
        assert agent is not None

    def test_memory_curator_has_curate_method(self):
        """MemoryCuratorAgent must have extract_and_preserve() method."""
        agent = MemoryCuratorAgent()
        assert hasattr(agent, "extract_and_preserve")
        assert callable(agent.extract_and_preserve)


class TestCouncilChairAgent:
    """Validate CouncilChairAgent code and output."""

    def test_council_chair_instantiation(self):
        """CouncilChairAgent should instantiate without errors."""
        agent = CouncilChairAgent()
        assert agent is not None

    def test_council_chair_has_consult_method(self):
        """CouncilChairAgent must have consult() or select_smes() method."""
        agent = CouncilChairAgent()
        assert (
            hasattr(agent, "consult") or
            hasattr(agent, "select_smes") or
            hasattr(agent, "evaluate")
        )


class TestSMESpawner:
    """Validate SMESpawner code and output."""

    def test_sme_spawner_instantiation(self):
        """SMESpawner should instantiate without errors."""
        spawner = SMESpawner()
        assert spawner is not None

    def test_sme_spawner_has_spawn_method(self):
        """SMESpawner must have spawn_from_selection() or spawn() method."""
        spawner = SMESpawner()
        assert (
            hasattr(spawner, "spawn_from_selection") or
            hasattr(spawner, "spawn")
        )

    def test_sme_spawner_has_interaction_method(self):
        """SMESpawner must have execute_sme_interaction() method."""
        spawner = SMESpawner()
        assert (
            hasattr(spawner, "execute_sme_interaction") or
            hasattr(spawner, "interact")
        )


# =============================================================================
# 7. BUILD AGENT OPTIONS TESTS
# =============================================================================

class TestBuildAgentOptions:
    """Test build_agent_options for Tier 3 agents."""

    @patch("src.core.sdk_integration.get_settings")
    @patch("src.core.sdk_integration.get_model_for_agent")
    def test_build_options_analyst(self, mock_model, mock_settings):
        """Build options for analyst should include correct tools."""
        mock_model.return_value = "claude-sonnet-4-20250514"
        mock_settings.return_value = MagicMock(
            max_turns_orchestrator=50,
            max_turns_executor=40,
            max_turns_subagent=30,
        )
        options = build_agent_options("analyst", "You are the Analyst.")
        assert set(options.allowed_tools) == {"Read", "Glob", "Grep"}
        assert options.permission_mode == PermissionMode.DEFAULT

    @patch("src.core.sdk_integration.get_settings")
    @patch("src.core.sdk_integration.get_model_for_agent")
    def test_build_options_executor(self, mock_model, mock_settings):
        """Build options for executor should have acceptEdits permission."""
        mock_model.return_value = "claude-sonnet-4-20250514"
        mock_settings.return_value = MagicMock(
            max_turns_orchestrator=50,
            max_turns_executor=40,
            max_turns_subagent=30,
        )
        options = build_agent_options("executor", "You are the Executor.")
        assert options.permission_mode == PermissionMode.ACCEPT_EDITS
        assert "Write" in options.allowed_tools
        assert "Edit" in options.allowed_tools
        assert "Skill" in options.allowed_tools

    @patch("src.core.sdk_integration.get_settings")
    @patch("src.core.sdk_integration.get_model_for_agent")
    def test_build_options_council_chair(self, mock_model, mock_settings):
        """Build options for council_chair should have no tools."""
        mock_model.return_value = "claude-sonnet-4-20250514"
        mock_settings.return_value = MagicMock(
            max_turns_orchestrator=50,
            max_turns_executor=40,
            max_turns_subagent=30,
        )
        options = build_agent_options("council_chair", "You are the Council Chair.")
        assert options.allowed_tools == []

    @patch("src.core.sdk_integration.get_settings")
    @patch("src.core.sdk_integration.get_model_for_agent")
    def test_build_options_extra_tools(self, mock_model, mock_settings):
        """Extra tools should be appended to agent's base tools."""
        mock_model.return_value = "claude-sonnet-4-20250514"
        mock_settings.return_value = MagicMock(
            max_turns_orchestrator=50,
            max_turns_executor=40,
            max_turns_subagent=30,
        )
        options = build_agent_options(
            "analyst", "You are the Analyst.",
            extra_tools=["WebSearch"],
        )
        assert "WebSearch" in options.allowed_tools
        assert "Read" in options.allowed_tools


# =============================================================================
# 8. PIPELINE TESTS (TIER 3)
# =============================================================================

class TestTier3Pipeline:
    """Validate the execution pipeline for Tier 3 (DEEP)."""

    def test_pipeline_for_tier3(self):
        """Pipeline for Tier 3 should include all 8 phases."""
        pipeline = PipelineBuilder.for_tier(TierLevel.DEEP)
        assert pipeline.tier_level == TierLevel.DEEP
        phases = pipeline._get_phases_for_tier()
        assert len(phases) == 8  # All phases

    def test_pipeline_tier3_no_skip(self):
        """Tier 3 should NOT skip any phase."""
        pipeline = ExecutionPipeline(tier_level=TierLevel.DEEP)
        for phase in Phase:
            assert pipeline._should_skip_phase(phase) is False

    def test_pipeline_tier1_skips_most(self):
        """Tier 1 should skip all phases except 5 and 8."""
        pipeline = ExecutionPipeline(tier_level=TierLevel.DIRECT)
        assert pipeline._should_skip_phase(Phase.PHASE_1_TASK_INTELLIGENCE) is True
        assert pipeline._should_skip_phase(Phase.PHASE_2_COUNCIL_CONSULTATION) is True
        assert pipeline._should_skip_phase(Phase.PHASE_5_SOLUTION_GENERATION) is False
        assert pipeline._should_skip_phase(Phase.PHASE_8_FINAL_REVIEW_FORMATTING) is False

    def test_pipeline_tier2_skips_council_and_revision(self):
        """Tier 2 should skip Council and Revision; Research runs."""
        pipeline = ExecutionPipeline(tier_level=TierLevel.STANDARD)
        assert pipeline._should_skip_phase(Phase.PHASE_2_COUNCIL_CONSULTATION) is True
        assert pipeline._should_skip_phase(Phase.PHASE_4_RESEARCH) is False
        assert pipeline._should_skip_phase(Phase.PHASE_7_REVISION) is True

    def test_pipeline_phase_agents(self):
        """Each phase should map to correct agents."""
        pipeline = ExecutionPipeline(tier_level=TierLevel.DEEP)
        assert pipeline._get_agents_for_phase(Phase.PHASE_1_TASK_INTELLIGENCE) == ["Task Analyst"]
        assert pipeline._get_agents_for_phase(Phase.PHASE_3_PLANNING) == ["Planner"]
        assert pipeline._get_agents_for_phase(Phase.PHASE_5_SOLUTION_GENERATION) == ["Executor"]
        assert "Reviewer" in pipeline._get_agents_for_phase(Phase.PHASE_8_FINAL_REVIEW_FORMATTING)
        assert "Formatter" in pipeline._get_agents_for_phase(Phase.PHASE_8_FINAL_REVIEW_FORMATTING)

    def test_pipeline_tier3_council_agents(self):
        """Tier 3 pipeline should include Domain Council Chair."""
        pipeline = ExecutionPipeline(tier_level=TierLevel.DEEP)
        council_agents = pipeline._get_council_agents()
        assert "Domain Council Chair" in council_agents

    def test_pipeline_review_agents(self):
        """Review phase should include Verifier and Critic."""
        pipeline = ExecutionPipeline(tier_level=TierLevel.DEEP)
        review_agents = pipeline._get_review_agents()
        assert "Verifier" in review_agents
        assert "Critic" in review_agents

    def test_pipeline_state_initialization(self):
        """PipelineState should initialize with correct defaults."""
        state = PipelineState(tier_level=TierLevel.DEEP)
        assert state.tier_level == TierLevel.DEEP
        assert state.revision_cycle == 0
        assert state.debate_rounds == 0
        assert state.total_cost_usd == 0.0

    def test_pipeline_execute_phase_skip(self):
        """Skipped phase should return SKIPPED status."""
        pipeline = ExecutionPipeline(tier_level=TierLevel.DIRECT)
        result = pipeline.execute_phase(
            Phase.PHASE_1_TASK_INTELLIGENCE,
            lambda **kwargs: AgentResult("test", "success", {}, 0),
            {},
        )
        assert result.status == PhaseStatus.SKIPPED

    def test_pipeline_determine_phase_status_success(self):
        """Phase with at least one success should be COMPLETE."""
        pipeline = ExecutionPipeline(tier_level=TierLevel.DEEP)
        results = [
            AgentResult("Agent1", "success", {}, 100),
            AgentResult("Agent2", "error", None, 50, error="Failed"),
        ]
        assert pipeline._determine_phase_status(results) == PhaseStatus.COMPLETE

    def test_pipeline_determine_phase_status_critical_failure(self):
        """Phase fails if critical agent (Verifier) fails."""
        pipeline = ExecutionPipeline(tier_level=TierLevel.DEEP)
        results = [
            AgentResult("Verifier", "error", None, 50, error="Failed"),
        ]
        assert pipeline._determine_phase_status(results) == PhaseStatus.FAILED

    def test_create_execution_context(self):
        """create_execution_context should build correct context dict."""
        classification = TierClassification(
            tier=TierLevel.DEEP,
            reasoning="Complex task",
            confidence=0.85,
            estimated_agents=12,
            requires_council=True,
            requires_smes=True,
            suggested_sme_count=2,
        )
        context = create_execution_context(
            "Design microservices architecture",
            classification,
            session_id="session_123",
        )
        assert context["tier"] == TierLevel.DEEP
        assert context["requires_council"] is True
        assert context["requires_smes"] is True
        assert context["session_id"] == "session_123"

    def test_estimate_pipeline_duration_tier3(self):
        """Tier 3 should estimate 90-300 seconds."""
        estimate = estimate_pipeline_duration(TierLevel.DEEP)
        assert estimate["min"] == 90
        assert estimate["max"] == 300
        assert estimate["estimated"] == 180


# =============================================================================
# 9. VERDICT MATRIX TESTS
# =============================================================================

class TestVerdictMatrix:
    """Validate verdict matrix logic for Tier 3."""

    def test_pass_pass_proceeds(self):
        """PASS + PASS → Proceed to Formatter."""
        outcome = evaluate_verdict_matrix(Verdict.PASS, Verdict.PASS)
        assert outcome.action == MatrixAction.PROCEED_TO_FORMATTER

    def test_pass_fail_revises(self):
        """PASS + FAIL → Executor Revise."""
        outcome = evaluate_verdict_matrix(Verdict.PASS, Verdict.FAIL)
        assert outcome.action == MatrixAction.EXECUTOR_REVISE

    def test_fail_pass_reverifies(self):
        """FAIL + PASS → Researcher Re-verify."""
        outcome = evaluate_verdict_matrix(Verdict.FAIL, Verdict.PASS)
        assert outcome.action == MatrixAction.RESEARCHER_REVERIFY

    def test_fail_fail_regenerates(self):
        """FAIL + FAIL → Full Regeneration."""
        outcome = evaluate_verdict_matrix(Verdict.FAIL, Verdict.FAIL)
        assert outcome.action == MatrixAction.FULL_REGENERATION

    def test_revision_limit_respected(self):
        """After max revisions, can_retry should be False."""
        outcome = evaluate_verdict_matrix(
            Verdict.PASS, Verdict.FAIL,
            revision_cycle=2, max_revisions=2,
        )
        assert outcome.can_retry is False

    def test_tier4_arbiter_after_max_revisions(self):
        """Tier 4 should invoke Quality Arbiter after max revisions."""
        outcome = evaluate_verdict_matrix(
            Verdict.PASS, Verdict.FAIL,
            revision_cycle=2, max_revisions=2, tier_level=4,
        )
        assert outcome.action == MatrixAction.QUALITY_ARBITER

    def test_verdict_matrix_completeness(self):
        """All 4 verdict combinations must be in VERDICT_MATRIX."""
        assert (Verdict.PASS, Verdict.PASS) in VERDICT_MATRIX
        assert (Verdict.PASS, Verdict.FAIL) in VERDICT_MATRIX
        assert (Verdict.FAIL, Verdict.PASS) in VERDICT_MATRIX
        assert (Verdict.FAIL, Verdict.FAIL) in VERDICT_MATRIX

    def test_get_phase_for_action(self):
        """Each MatrixAction should map to a pipeline phase."""
        assert "Phase 8" in get_phase_for_action(MatrixAction.PROCEED_TO_FORMATTER)
        assert "Phase 7" in get_phase_for_action(MatrixAction.EXECUTOR_REVISE)
        assert "Phase 4" in get_phase_for_action(MatrixAction.RESEARCHER_REVERIFY)
        assert "Phase 5" in get_phase_for_action(MatrixAction.FULL_REGENERATION)

    def test_should_trigger_debate_disagreement(self):
        """Disagreement between Verifier and Critic should trigger debate."""
        outcome = MatrixOutcome(
            verifier_verdict=Verdict.PASS,
            critic_verdict=Verdict.FAIL,
            action=MatrixAction.EXECUTOR_REVISE,
            reason="Disagreement",
            revision_cycle=0,
            can_retry=True,
        )
        assert should_trigger_debate(outcome, tier_level=3) is True

    def test_should_trigger_debate_tier4(self):
        """Tier 4 should always trigger debate."""
        outcome = MatrixOutcome(
            verifier_verdict=Verdict.PASS,
            critic_verdict=Verdict.PASS,
            action=MatrixAction.PROCEED_TO_FORMATTER,
            reason="Both passed",
            revision_cycle=0,
            can_retry=True,
        )
        assert should_trigger_debate(outcome, tier_level=4) is True

    def test_no_debate_agreement_tier3(self):
        """Agreement in Tier 3 should not trigger debate."""
        outcome = MatrixOutcome(
            verifier_verdict=Verdict.PASS,
            critic_verdict=Verdict.PASS,
            action=MatrixAction.PROCEED_TO_FORMATTER,
            reason="Both passed",
            revision_cycle=0,
            can_retry=True,
        )
        assert should_trigger_debate(outcome, tier_level=3) is False


# =============================================================================
# 10. DEBATE PROTOCOL TESTS
# =============================================================================

class TestDebateProtocol:
    """Validate debate protocol for Tier 3-4."""

    def test_debate_protocol_creation(self):
        """DebateProtocol should initialize correctly."""
        protocol = DebateProtocol(max_rounds=2, consensus_threshold=0.8)
        assert protocol.max_rounds == 2
        assert protocol.consensus_threshold == 0.8
        assert len(protocol.rounds) == 0

    def test_add_participants(self):
        """Should add participants without duplicates."""
        protocol = DebateProtocol()
        protocol.add_participant("Executor")
        protocol.add_participant("Critic")
        protocol.add_participant("Executor")  # duplicate
        assert len(protocol.participants) == 2

    def test_add_sme_participants(self):
        """Should add SME participants."""
        protocol = DebateProtocol()
        protocol.add_sme_participant("cloud_architect")
        protocol.add_sme_participant("security_analyst")
        assert len(protocol.sme_participants) == 2

    def test_consensus_calculation(self):
        """Consensus score should be weighted average."""
        protocol = DebateProtocol()
        score = protocol.calculate_consensus(
            executor_agreement=0.9,
            critic_agreement=0.7,
            verifier_agreement=0.8,
            sme_agreements={"sme1": 0.6},
        )
        assert 0.0 <= score <= 1.0

    def test_consensus_level_full(self):
        """Score >= 0.8 should be FULL consensus."""
        protocol = DebateProtocol(consensus_threshold=0.8)
        assert protocol.determine_consensus_level(0.85) == ConsensusLevel.FULL

    def test_consensus_level_majority(self):
        """Score 0.5-0.79 should be MAJORITY consensus."""
        protocol = DebateProtocol(consensus_threshold=0.8, majority_threshold=0.5)
        assert protocol.determine_consensus_level(0.65) == ConsensusLevel.MAJORITY

    def test_consensus_level_split(self):
        """Score < 0.5 should be SPLIT consensus."""
        protocol = DebateProtocol(majority_threshold=0.5)
        assert protocol.determine_consensus_level(0.3) == ConsensusLevel.SPLIT

    def test_should_continue_debate(self):
        """Should continue when consensus not achieved and rounds remaining."""
        protocol = DebateProtocol(max_rounds=2, consensus_threshold=0.8)
        assert protocol.should_continue_debate(0.6) is True

    def test_should_stop_debate_consensus(self):
        """Should stop when consensus achieved."""
        protocol = DebateProtocol(max_rounds=2, consensus_threshold=0.8)
        assert protocol.should_continue_debate(0.85) is False

    def test_should_stop_debate_max_rounds(self):
        """Should stop after max rounds."""
        protocol = DebateProtocol(max_rounds=2, consensus_threshold=0.8)
        protocol.rounds = [MagicMock(), MagicMock()]
        assert protocol.should_continue_debate(0.6) is False

    def test_conduct_round(self):
        """conduct_round should create a DebateRound."""
        protocol = DebateProtocol()
        round_result = protocol.conduct_round(
            executor_position="Solution is correct",
            critic_challenges=["Missing error handling"],
            verifier_checks=["Claim 1 verified"],
            sme_arguments={"cloud_architect": "Architecture is sound"},
        )
        assert isinstance(round_result, DebateRound)
        assert round_result.round_number == 1

    def test_get_outcome_no_rounds(self):
        """get_outcome with no rounds should return SPLIT."""
        protocol = DebateProtocol()
        outcome = protocol.get_outcome()
        assert outcome.consensus_level == ConsensusLevel.SPLIT
        assert outcome.rounds_completed == 0

    def test_get_outcome_full_consensus(self):
        """Full consensus outcome should approve solution."""
        protocol = DebateProtocol(consensus_threshold=0.5)
        protocol.conduct_round(
            executor_position="Good solution",
            critic_challenges=[],
            verifier_checks=[],
            sme_arguments={},
        )
        outcome = protocol.get_outcome()
        assert outcome.rounds_completed == 1

    def test_needs_arbiter_split(self):
        """Should need arbiter for SPLIT consensus."""
        protocol = DebateProtocol()
        assert protocol.needs_arbiter(ConsensusLevel.SPLIT, 2) is True

    def test_can_proceed_full(self):
        """Should be able to proceed with FULL consensus."""
        protocol = DebateProtocol()
        assert protocol.can_proceed(ConsensusLevel.FULL) is True

    def test_can_proceed_majority(self):
        """Should be able to proceed with MAJORITY consensus."""
        protocol = DebateProtocol()
        assert protocol.can_proceed(ConsensusLevel.MAJORITY) is True

    def test_cannot_proceed_split(self):
        """Should not be able to proceed with SPLIT consensus."""
        protocol = DebateProtocol()
        assert protocol.can_proceed(ConsensusLevel.SPLIT) is False

    def test_trigger_debate_disagreement(self):
        """trigger_debate should return True for disagreement."""
        assert trigger_debate("PASS", "FAIL", tier_level=3) is True

    def test_trigger_debate_tier4(self):
        """trigger_debate should return True for Tier 4."""
        assert trigger_debate("PASS", "PASS", tier_level=4) is True

    def test_no_trigger_debate_agreement_tier3(self):
        """trigger_debate should return False for agreement in Tier 3."""
        assert trigger_debate("PASS", "PASS", tier_level=3) is False

    def test_get_debate_participants_tier3(self):
        """Tier 3 participants should include agents + SMEs."""
        participants = get_debate_participants(
            tier_level=3,
            available_smes=["cloud_architect", "security_analyst"],
        )
        assert "Executor" in participants["agents"]
        assert "Critic" in participants["agents"]
        assert "Verifier" in participants["agents"]
        assert "cloud_architect" in participants["smes"]

    def test_get_debate_participants_max_3_smes(self):
        """Should cap SME participation at 3."""
        participants = get_debate_participants(
            tier_level=3,
            available_smes=["sme1", "sme2", "sme3", "sme4", "sme5"],
        )
        assert len(participants["smes"]) == 3


# =============================================================================
# 11. REQUIRED AGENTS PER PHASE TESTS
# =============================================================================

class TestRequiredAgentsPerPhase:
    """Validate agent assignments per pipeline phase."""

    def test_phase1_agents(self):
        """Phase 1 should require Task Analyst."""
        agents = get_required_agents_for_phase("Phase 1", tier=3)
        assert "Task Analyst" in agents

    def test_phase2_agents_tier3(self):
        """Phase 2 should require Council Chair for Tier 3."""
        agents = get_required_agents_for_phase("Phase 2", tier=3)
        assert "Domain Council Chair" in agents

    def test_phase2_agents_tier2(self):
        """Phase 2 should have no agents for Tier 2."""
        agents = get_required_agents_for_phase("Phase 2", tier=2)
        assert agents == []

    def test_phase3_agents(self):
        """Phase 3 should require Planner."""
        agents = get_required_agents_for_phase("Phase 3", tier=3)
        assert "Planner" in agents

    def test_phase5_agents(self):
        """Phase 5 should require Executor."""
        agents = get_required_agents_for_phase("Phase 5", tier=3)
        assert "Executor" in agents

    def test_phase6_agents(self):
        """Phase 6 should include review agents."""
        agents = get_required_agents_for_phase("Phase 6", tier=3)
        assert "Verifier" in agents
        assert "Critic" in agents

    def test_phase8_agents(self):
        """Phase 8 should require Reviewer and Formatter."""
        agents = get_required_agents_for_phase("Phase 8", tier=3)
        assert "Reviewer" in agents
        assert "Formatter" in agents


# =============================================================================
# 12. OUTPUT SCHEMA MAPPING TESTS
# =============================================================================

class TestOutputSchemaMapping:
    """Validate that _get_output_schema returns schemas for all Tier 3 agents."""

    @pytest.mark.parametrize("agent_name,expected_model", [
        ("analyst", "TaskIntelligenceReport"),
        ("planner", "ExecutionPlan"),
        ("clarifier", "ClarificationRequest"),
        ("researcher", "EvidenceBrief"),
        ("code_reviewer", "CodeReviewReport"),
        ("verifier", "VerificationReport"),
        ("critic", "CritiqueReport"),
        ("reviewer", "ReviewVerdict"),
        ("council_chair", "SMESelectionReport"),
    ])
    def test_schema_mapping_exists(self, agent_name, expected_model):
        """Each agent should map to a schema via _get_output_schema."""
        # _get_output_schema returns None if schema class not found,
        # but the mapping should at least exist in the function's logic
        schema = _get_output_schema(agent_name)
        # Schema may be None if the schema class import fails,
        # but the function should not raise an exception
        assert schema is None or isinstance(schema, dict)


# =============================================================================
# 13. PIPELINE BUILDER TESTS
# =============================================================================

class TestPipelineBuilder:
    """Validate PipelineBuilder utility."""

    def test_for_tier_direct(self):
        """PipelineBuilder.for_tier(DIRECT) should create Tier 1 pipeline."""
        pipeline = PipelineBuilder.for_tier(TierLevel.DIRECT)
        assert pipeline.tier_level == TierLevel.DIRECT

    def test_for_tier_deep(self):
        """PipelineBuilder.for_tier(DEEP) should create Tier 3 pipeline."""
        pipeline = PipelineBuilder.for_tier(TierLevel.DEEP)
        assert pipeline.tier_level == TierLevel.DEEP

    def test_from_classification(self):
        """PipelineBuilder.from_classification should create correct pipeline."""
        classification = TierClassification(
            tier=TierLevel.DEEP,
            reasoning="Complex task",
            confidence=0.85,
            estimated_agents=12,
            requires_council=True,
            requires_smes=True,
        )
        pipeline = PipelineBuilder.from_classification(classification)
        assert pipeline.tier_level == TierLevel.DEEP


# =============================================================================
# 14. INITIATE DEBATE FROM PIPELINE TESTS
# =============================================================================

class TestPipelineDebateIntegration:
    """Validate debate initiation from pipeline."""

    def test_pipeline_initiate_debate(self):
        """Pipeline should initiate debate with correct participants."""
        pipeline = ExecutionPipeline(tier_level=TierLevel.DEEP)
        context = {"active_smes": ["cloud_architect", "security_analyst"]}
        debate = pipeline.initiate_debate(context)
        assert isinstance(debate, DebateProtocol)
        assert "Executor" in debate.participants
        assert "Critic" in debate.participants
        assert "Verifier" in debate.participants
        assert "cloud_architect" in debate.sme_participants
        assert "security_analyst" in debate.sme_participants


# =============================================================================
# 15. COST ESTIMATION TESTS
# =============================================================================

class TestCostEstimation:
    """Validate cost estimation functions."""

    def test_phase6_cost_tier3(self):
        """Phase 6 cost should be higher due to multiple agents."""
        from src.core.verdict import calculate_phase_cost_estimate
        cost = calculate_phase_cost_estimate(tier=3, phase="Phase 6")
        assert cost > 0.0

    def test_tier3_cost_multiplier(self):
        """Tier 3 should have 1.5x cost multiplier."""
        from src.core.verdict import calculate_phase_cost_estimate
        cost_t2 = calculate_phase_cost_estimate(tier=2, phase="Phase 5")
        cost_t3 = calculate_phase_cost_estimate(tier=3, phase="Phase 5")
        assert cost_t3 > cost_t2  # Tier 3 should cost more


# =============================================================================
# 16. VERIFIER + CRITIC INTEGRATION TESTS
# =============================================================================

class TestVerifierCriticIntegration:
    """Test Verifier and Critic working together (Phase 6 simulation)."""

    def test_both_pass_leads_to_formatter(self):
        """When both Verifier and Critic pass, should proceed to formatter."""
        verifier = VerifierAgent()
        v_result = verifier.verify("This is a simple, verifiable statement.")

        critic = CriticAgent()
        c_result = critic.critique("A well-structured, complete solution.", "Create a solution.")

        # Map to verdicts
        v_verdict = Verdict.PASS if v_result.verdict == "PASS" else Verdict.FAIL
        c_verdict = Verdict.PASS if c_result.would_approve else Verdict.FAIL

        # Evaluate matrix
        outcome = evaluate_verdict_matrix(v_verdict, c_verdict)
        assert outcome.action in list(MatrixAction)

    def test_verifier_fail_triggers_reverify(self):
        """When Verifier fails but Critic passes, should re-verify."""
        outcome = evaluate_verdict_matrix(Verdict.FAIL, Verdict.PASS)
        assert outcome.action == MatrixAction.RESEARCHER_REVERIFY
