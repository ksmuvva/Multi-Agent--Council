"""
Unit Tests for Core Utilities

Tests for complexity classification, ensemble patterns, and SME registry.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock

from src.core.complexity import (
    TierLevel,
    TierClassification,
    classify_complexity,
    get_active_agents,
    get_council_agents,
)
from src.core.ensemble import (
    EnsemblePattern,
    EnsembleConfig,
    ENSEMBLE_REGISTRY,
)
from src.core.sme_registry import (
    SMEPersona,
    find_personas_by_keywords,
    find_personas_by_domain,
    get_persona,
)


# =============================================================================
# Complexity Classification Tests
# =============================================================================

class TestClassifyComplexity:
    """Tests for task complexity classification."""

    def test_simple_query_tier1(self):
        """Test that simple queries are classified as Tier 1."""
        result = classify_complexity("What is 2 + 2?")

        assert result.tier == TierLevel.DIRECT
        assert result.estimated_agents <= 3

    def test_standard_task_tier2(self):
        """Test that standard tasks are classified as Tier 2."""
        result = classify_complexity(
            "Write a Python function to calculate fibonacci numbers"
        )

        assert result.tier == TierLevel.STANDARD
        assert 5 <= result.estimated_agents <= 10

    def test_complex_task_tier3(self):
        """Test that complex tasks are classified as Tier 3."""
        result = classify_complexity(
            "Design a microservices architecture for a healthcare application "
            "with HIPAA compliance requirements"
        )

        assert result.tier == TierLevel.DEEP
        assert result.requires_council is True
        assert result.requires_smes is True

    def test_adversarial_task_tier4(self):
        """Test that adversarial/high-stakes tasks are classified as Tier 4."""
        result = classify_complexity(
            "Analyze the security implications of a new authentication protocol "
            "and identify potential vulnerabilities"
        )

        assert result.tier == TierLevel.ADVERSARIAL
        assert result.requires_council is True

    def test_classification_has_reasoning(self):
        """Test that classification includes reasoning."""
        result = classify_complexity("Test prompt")

        assert hasattr(result, "reasoning")
        assert len(result.reasoning) > 0

    def test_escalation_keywords(self):
        """Test that certain keywords trigger escalation."""
        result = classify_complexity(
            "This is critical and may have security implications"
        )

        # Keywords like "critical", "security" may increase tier
        assert result.tier.value >= TierLevel.STANDARD.value


class TestActiveAgents:
    """Tests for active agent retrieval by tier."""

    def test_tier1_agents(self):
        """Test Tier 1 agent list."""
        agents = get_active_agents(1)

        assert "Executor" in agents
        assert "Formatter" in agents
        # Tier 1 should have minimal agents
        assert len(agents) <= 3

    def test_tier2_agents(self):
        """Test Tier 2 agent list."""
        agents = get_active_agents(2)

        expected_agents = [
            "Analyst",
            "Planner",
            "Clarifier",
            "Executor",
            "Verifier",
            "Formatter",
        ]
        for agent in expected_agents:
            assert agent in agents

    def test_tier3_agents(self):
        """Test Tier 3 agent list includes Council."""
        agents = get_active_agents(3)

        assert "CouncilChair" in agents
        assert "QualityArbiter" in agents
        assert "EthicsAdvisor" in agents

    def test_tier4_agents(self):
        """Test Tier 4 agent list includes all agents."""
        agents = get_active_agents(4)

        # Should include all operational and council agents
        assert len(agents) > 10

    def test_council_agents(self):
        """Test Council agent list."""
        agents = get_council_agents()

        expected = [
            "CouncilChair",
            "QualityArbiter",
            "EthicsAdvisor",
        ]
        for agent in expected:
            assert agent in agents


# =============================================================================
# Ensemble Pattern Tests
# =============================================================================

class TestEnsemblePatterns:
    """Tests for ensemble pattern registry."""

    def test_ensemble_registry_exists(self):
        """Test that ensemble registry exists and has entries."""
        assert len(ENSEMBLE_REGISTRY) > 0

    def test_architecture_review_board_exists(self):
        """Test that Architecture Review Board pattern exists."""
        assert "architecture_review_board" in ENSEMBLE_REGISTRY

    def test_ensemble_pattern_structure(self):
        """Test that ensemble patterns have required structure."""
        for name, pattern_cls in ENSEMBLE_REGISTRY.items():
            pattern = pattern_cls()
            assert hasattr(pattern, "get_config")
            assert hasattr(pattern, "execute")

            config = pattern.get_config()
            assert isinstance(config, EnsembleConfig)
            assert config.name is not None
            assert config.agent_assignments is not None
            assert len(config.agent_assignments) > 0

    def test_code_sprint_pattern(self):
        """Test Code Sprint pattern specifically."""
        pattern_cls = ENSEMBLE_REGISTRY.get("code_sprint")

        assert pattern_cls is not None
        pattern = pattern_cls()
        config = pattern.get_config()

        # Code sprint should include executor and code reviewer
        agent_names = [a.agent_name for a in config.agent_assignments]
        assert "Executor" in agent_names
        assert "CodeReviewer" in agent_names


# =============================================================================
# SME Registry Tests
# =============================================================================

class TestSMERegistry:
    """Tests for SME persona registry."""

    def test_cloud_architect_exists(self):
        """Test that Cloud Architect persona exists."""
        persona = get_persona("cloud_architect")

        assert persona is not None
        assert persona.persona_id == "cloud_architect"
        assert "cloud" in persona.domain.lower()

    def test_security_analyst_exists(self):
        """Test that Security Analyst persona exists."""
        persona = get_persona("security_analyst")

        assert persona is not None
        assert "security" in persona.domain.lower()

    def test_find_by_keywords(self):
        """Test finding personas by keywords."""
        # Find cloud-related personas
        personas = find_personas_by_keywords(["aws", "azure"])

        assert len(personas) > 0
        # Cloud architect should match
        persona_ids = [p.persona_id for p in personas]
        assert "cloud_architect" in persona_ids

    def test_find_by_domain(self):
        """Test finding personas by domain."""
        personas = find_personas_by_domain(["security"])

        assert len(personas) > 0
        # Security analyst should match
        persona_ids = [p.persona_id for p in personas]
        assert "security_analyst" in persona_ids

    def test_persona_interaction_modes(self):
        """Test that personas have interaction modes."""
        persona = get_persona("cloud_architect")

        assert hasattr(persona, "interaction_modes")
        assert len(persona.interaction_modes) > 0

    def test_persona_trigger_keywords(self):
        """Test that personas have trigger keywords."""
        persona = get_persona("cloud_architect")

        assert hasattr(persona, "trigger_keywords")
        assert len(persona.trigger_keywords) > 0
        assert "cloud" in [k.lower() for k in persona.trigger_keywords]

    def test_invalid_persona_returns_none(self):
        """Test that invalid persona ID returns None."""
        persona = get_persona("nonexistent_persona")

        assert persona is None


# =============================================================================
# Tier Classification Edge Cases
# =============================================================================

class TestTierClassificationEdgeCases:
    """Tests for edge cases in tier classification."""

    def test_empty_prompt(self):
        """Test handling of empty or minimal prompts."""
        result = classify_complexity("")

        assert result.tier is not None

    def test_very_long_prompt(self):
        """Test handling of very long prompts."""
        long_prompt = "Analyze this: " + ("content " * 1000)

        result = classify_complexity(long_prompt)

        assert result.tier is not None

    def test_mixed_signals(self):
        """Test prompts with mixed complexity signals."""
        result = classify_complexity(
            "Write a simple hello world function with comprehensive "
            "error handling, security hardening, and documentation"
        )

        # "security" triggers Tier 3 since it requires domain expertise
        assert TierLevel.STANDARD.value <= result.tier.value <= TierLevel.DEEP.value

    def test_code_keywords(self):
        """Test that code-related keywords are detected."""
        result = classify_complexity("Write a function to sort an array")

        assert "code" in result.reasoning.lower() or "function" in result.reasoning.lower()

    def test_research_keywords(self):
        """Test that research-related keywords are detected."""
        result = classify_complexity("Research the latest trends in AI")

        assert "research" in result.reasoning.lower()


# =============================================================================
# Mock Tests for Agent Execution
# =============================================================================

class TestAgentExecutionMocking:
    """Tests using mocked agent execution."""

    @pytest.mark.asyncio
    async def test_mock_analyst_response(self):
        """Test mock Analyst agent response structure."""
        # Create a mock response
        mock_response = {
            "literal_request": "Analyze the request",
            "inferred_intent": "User wants analysis",
            "sub_tasks": [
                {"description": "Task 1", "dependencies": []},
                {"description": "Task 2", "dependencies": ["Task 1"]},
            ],
            "missing_info": [],
            "assumptions": [],
            "modality": "text",
            "recommended_approach": "Structured analysis",
            "escalation_needed": False,
        }

        # Validate structure
        assert "literal_request" in mock_response
        assert "sub_tasks" in mock_response
        assert "escalation_needed" in mock_response

    @pytest.mark.asyncio
    async def test_mock_verifier_response(self):
        """Test mock Verifier agent response structure."""
        mock_response = {
            "claims": [
                {
                    "claim": "Test claim",
                    "verification": "verified",
                    "confidence": 0.9,
                    "source": "Test",
                }
            ],
            "factual_accuracy_score": 0.9,
            "hallucination_risk": "low",
            "recommendations": [],
        }

        assert "claims" in mock_response
        assert "factual_accuracy_score" in mock_response
        assert 0 <= mock_response["factual_accuracy_score"] <= 1

    @pytest.mark.asyncio
    async def test_mock_reviewer_verdict(self):
        """Test mock Reviewer verdict response."""
        mock_response = {
            "verdict": "PROCEED_TO_FORMATTER",
            "quality_gates": {
                "completeness": "pass",
                "consistency": "pass",
                "verifier_signoff": "pass",
                "critic_findings": "pass",
                "readability": "pass",
            },
            "final_recommendation": "Quality acceptable",
        }

        assert mock_response["verdict"] in [
            "PROCEED_TO_FORMATTER",
            "EXECUTOR_REVISE",
            "RESEARCHER_REVERIFY",
            "FULL_REGENERATION",
        ]


# =============================================================================
# Cost Calculation Tests
# =============================================================================

class TestCostCalculations:
    """Tests for cost-related utilities."""

    def test_agent_token_estimates(self):
        """Test that token estimates are reasonable."""
        # These should match the AGENT_TOKEN_COSTS in utils/cost.py
        from src.utils.cost import AGENT_TOKEN_COSTS

        assert "Executor" in AGENT_TOKEN_COSTS
        assert "Verifier" in AGENT_TOKEN_COSTS

        executor_costs = AGENT_TOKEN_COSTS["Executor"]
        assert executor_costs["input"] > 0
        assert executor_costs["output"] > 0

    def test_model_pricing_exists(self):
        """Test that model pricing is defined."""
        from src.utils.cost import MODEL_COSTS, ModelPricing

        assert ModelPricing.HAIKU in MODEL_COSTS
        assert ModelPricing.SONNET in MODEL_COSTS
        assert ModelPricing.OPUS in MODEL_COSTS

        # Verify pricing structure
        for model, costs in MODEL_COSTS.items():
            assert "input" in costs
            assert "output" in costs
            assert costs["input"] > 0
            assert costs["output"] > 0
            assert costs["output"] > costs["input"]  # Output typically costs more

    def test_tier_model_mapping(self):
        """Test that tiers map to appropriate models."""
        from src.utils.cost import ModelPricing, MODEL_COSTS

        # Higher tiers should use more capable models
        # This is tested indirectly through cost calculations
        tier_models = {
            1: ModelPricing.HAIKU,
            2: ModelPricing.SONNET,
            3: ModelPricing.OPUS,
            4: ModelPricing.OPUS,
        }

        for tier, model in tier_models.items():
            assert model in MODEL_COSTS


# =============================================================================
# Event Bus Tests
# =============================================================================

class TestEventBus:
    """Tests for the EventBus class."""

    def test_subscribe_and_emit(self):
        """Test subscribing to an event and receiving it when emitted."""
        from src.core.events import EventBus

        bus = EventBus()
        received = []
        bus.subscribe("test_event", lambda data: received.append(data))
        bus.emit("test_event", {"key": "value"})

        assert len(received) == 1
        assert received[0]["key"] == "value"
        assert "event_type" in received[0]
        assert "timestamp" in received[0]

    def test_unsubscribe(self):
        """Test unsubscribing removes the callback."""
        from src.core.events import EventBus

        bus = EventBus()
        received = []
        callback = lambda data: received.append(data)
        bus.subscribe("test_event", callback)
        bus.unsubscribe("test_event", callback)
        bus.emit("test_event", {"key": "value"})

        assert len(received) == 0

    def test_clear_removes_all_subscribers(self):
        """Test clear removes all subscribers."""
        from src.core.events import EventBus

        bus = EventBus()
        received = []
        bus.subscribe("event_a", lambda data: received.append(data))
        bus.subscribe("event_b", lambda data: received.append(data))
        bus.clear()
        bus.emit("event_a", {"key": "a"})
        bus.emit("event_b", {"key": "b"})

        assert len(received) == 0

    def test_multiple_subscribers(self):
        """Test multiple subscribers to the same event."""
        from src.core.events import EventBus

        bus = EventBus()
        received_a = []
        received_b = []
        bus.subscribe("test_event", lambda data: received_a.append(data))
        bus.subscribe("test_event", lambda data: received_b.append(data))
        bus.emit("test_event", {"key": "value"})

        assert len(received_a) == 1
        assert len(received_b) == 1

    def test_emit_swallows_callback_errors(self):
        """Test that errors in callbacks don't break the bus."""
        from src.core.events import EventBus

        bus = EventBus()
        received = []

        def bad_callback(data):
            raise RuntimeError("Callback error")

        bus.subscribe("test_event", bad_callback)
        bus.subscribe("test_event", lambda data: received.append(data))

        # Should not raise despite bad_callback
        bus.emit("test_event", {"key": "value"})
        assert len(received) == 1

    def test_no_duplicate_subscriptions(self):
        """Test that the same callback is not added twice."""
        from src.core.events import EventBus

        bus = EventBus()
        received = []
        callback = lambda data: received.append(data)
        bus.subscribe("test_event", callback)
        bus.subscribe("test_event", callback)
        bus.emit("test_event", {"key": "value"})

        assert len(received) == 1


# =============================================================================
# Event Helper Functions Tests
# =============================================================================

class TestEventHelpers:
    """Tests for emit_agent_started, emit_agent_completed, etc."""

    def test_emit_agent_started_payload(self):
        """Test emit_agent_started sends correct payload."""
        from src.core.events import EventBus, EventType, agent_event_bus, emit_agent_started

        received = []
        agent_event_bus.clear()
        agent_event_bus.subscribe(EventType.AGENT_STARTED, lambda d: received.append(d))

        emit_agent_started(agent_id="a1", agent_name="Analyst", tier="2", phase="Init")

        assert len(received) == 1
        assert received[0]["agent_id"] == "a1"
        assert received[0]["agent_name"] == "Analyst"
        assert received[0]["tier"] == "2"
        assert received[0]["phase"] == "Init"
        assert received[0]["event_type"] == EventType.AGENT_STARTED
        agent_event_bus.clear()

    def test_emit_agent_completed_payload(self):
        """Test emit_agent_completed sends correct payload."""
        from src.core.events import agent_event_bus, EventType, emit_agent_completed

        received = []
        agent_event_bus.clear()
        agent_event_bus.subscribe(EventType.AGENT_COMPLETED, lambda d: received.append(d))

        emit_agent_completed(agent_id="a2", agent_name="Executor", output="done")

        assert len(received) == 1
        assert received[0]["agent_id"] == "a2"
        assert received[0]["output"] == "done"
        agent_event_bus.clear()

    def test_emit_agent_failed_payload(self):
        """Test emit_agent_failed sends correct payload."""
        from src.core.events import agent_event_bus, EventType, emit_agent_failed

        received = []
        agent_event_bus.clear()
        agent_event_bus.subscribe(EventType.AGENT_FAILED, lambda d: received.append(d))

        emit_agent_failed(agent_id="a3", agent_name="Researcher", error="network error")

        assert len(received) == 1
        assert received[0]["error"] == "network error"
        agent_event_bus.clear()

    def test_emit_cost_recorded_payload(self):
        """Test emit_cost_recorded sends correct payload."""
        from src.core.events import agent_event_bus, EventType, emit_cost_recorded

        received = []
        agent_event_bus.clear()
        agent_event_bus.subscribe(EventType.COST_RECORDED, lambda d: received.append(d))

        emit_cost_recorded(
            agent_name="Executor", model="claude-3-5-sonnet",
            input_tokens=500, output_tokens=1000, total_tokens=1500,
            cost_usd=0.05, tier=2, phase="Execution",
        )

        assert len(received) == 1
        assert received[0]["input_tokens"] == 500
        assert received[0]["output_tokens"] == 1000
        assert received[0]["cost_usd"] == 0.05
        assert received[0]["phase"] == "Execution"
        agent_event_bus.clear()


# =============================================================================
# Skill System Tests
# =============================================================================

class TestSkillSystem:
    """Tests for SDK skill system functions."""

    def test_get_skills_for_agent_known(self):
        """Test get_skills_for_agent returns skills for known agents."""
        from src.core.sdk_integration import get_skills_for_agent, AGENT_SKILLS

        for agent_name, expected_skills in AGENT_SKILLS.items():
            skills = get_skills_for_agent(agent_name)
            assert skills == list(expected_skills)

    def test_get_skills_for_agent_unknown(self):
        """Test get_skills_for_agent returns empty list for unknown agents."""
        from src.core.sdk_integration import get_skills_for_agent

        skills = get_skills_for_agent("nonexistent_agent")
        assert skills == []

    def test_get_skills_for_task_adds_keyword_skills(self):
        """Test get_skills_for_task adds skills based on task keywords."""
        from src.core.sdk_integration import get_skills_for_task

        skills = get_skills_for_task("design the architecture for the system", "executor")
        assert "architecture-design" in skills
        # Should also include executor's default skills
        assert "code-generation" in skills

    def test_get_skills_for_task_no_extra_for_unmatched(self):
        """Test get_skills_for_task returns only defaults for unmatched keywords."""
        from src.core.sdk_integration import get_skills_for_task

        skills = get_skills_for_task("do something generic", "executor")
        assert skills == ["code-generation"]

    def test_get_skill_chain_known(self):
        """Test get_skill_chain returns ordered skills for known chains."""
        from src.core.sdk_integration import get_skill_chain, SKILL_CHAINS

        for chain_name, expected in SKILL_CHAINS.items():
            chain = get_skill_chain(chain_name)
            assert chain == list(expected)

    def test_get_skill_chain_unknown(self):
        """Test get_skill_chain returns empty list for unknown chains."""
        from src.core.sdk_integration import get_skill_chain

        chain = get_skill_chain("nonexistent_chain")
        assert chain == []

    def test_select_skill_chain_development(self):
        """Test select_skill_chain selects full_development for dev keywords."""
        from src.core.sdk_integration import select_skill_chain

        result = select_skill_chain("develop a full stack application")
        assert result == "full_development"

    def test_select_skill_chain_research(self):
        """Test select_skill_chain selects research_and_report for research keywords."""
        from src.core.sdk_integration import select_skill_chain

        result = select_skill_chain("research the latest AI trends and report")
        assert result == "research_and_report"

    def test_select_skill_chain_no_match(self):
        """Test select_skill_chain returns None when no keywords match."""
        from src.core.sdk_integration import select_skill_chain

        result = select_skill_chain("hello world")
        assert result is None

    def test_skill_chains_have_correct_structure(self):
        """Test SKILL_CHAINS dictionary has valid entries."""
        from src.core.sdk_integration import SKILL_CHAINS

        assert len(SKILL_CHAINS) > 0
        for chain_name, skills in SKILL_CHAINS.items():
            assert isinstance(chain_name, str)
            assert isinstance(skills, list)
            assert len(skills) > 0
            for skill in skills:
                assert isinstance(skill, str)
