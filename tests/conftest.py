"""
Test Configuration and Fixtures

Shared test configuration, fixtures, and utilities for the multi-agent system tests.
"""

import sys
import os
from pathlib import Path
from typing import Any, Dict, Optional, List
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
import json

import pytest
from pydantic import BaseModel


# =============================================================================
# Path Setup
# =============================================================================

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Mock Data Constants
# =============================================================================

MOCK_SESSION_ID = "test_session_123"
MOCK_REQUEST_ID = "test_request_456"

MOCK_AGENT_RESPONSES = {
    "Analyst": {
        "literal_request": "Analyze the user request",
        "inferred_intent": "The user wants an analysis",
        "sub_tasks": [{"description": "Task 1"}, {"description": "Task 2"}],
        "missing_info": [],
        "assumptions": [],
        "modality": "text",
        "recommended_approach": "Step-by-step analysis",
        "escalation_needed": False,
    },
    "Planner": {
        "steps": [
            {"step": 1, "description": "Gather information", "agent": "Researcher"},
            {"step": 2, "description": "Generate solution", "agent": "Executor"},
        ],
        "estimated_duration": "5-10 minutes",
        "dependencies": [],
        "parallel_opportunities": [],
    },
    "Executor": {
        "solution": "Here is the solution to your request.",
        "approach_description": "Step-by-step approach was used",
        "alternatives_considered": ["Alternative A", "Alternative B"],
        "implementation_notes": "Some notes here",
    },
    "Verifier": {
        "claims": [{"claim": "Fact 1", "verification": "verified", "confidence": 0.95}],
        "factual_accuracy_score": 0.95,
        "hallucination_risk": "low",
        "recommendations": [],
    },
    "Critic": {
        "attack_vectors": ["logic", "completeness"],
        "findings": [{"category": "logic", "severity": "low", "description": "Minor issue"}],
        "overall_score": 0.85,
        "recommendations": [],
    },
    "Reviewer": {
        "verdict": "PROCEED_TO_FORMATTER",
        "quality_gates": {
            "completeness": "pass",
            "consistency": "pass",
            "verifier_signoff": "pass",
            "critic_findings": "pass",
            "readability": "pass",
        },
        "final_recommendation": "Quality is acceptable",
    },
}


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_session_id():
    """Provide a mock session ID."""
    return MOCK_SESSION_ID


@pytest.fixture
def mock_timestamp():
    """Provide a mock timestamp."""
    return datetime(2025, 1, 1, 12, 0, 0)


@pytest.fixture
def mock_user_prompt():
    """Provide a mock user prompt."""
    return "Write a Python function to calculate fibonacci numbers."


@pytest.fixture
def mock_agent_config():
    """Provide a mock agent configuration."""
    return {
        "name": "Analyst",
        "model": "claude-sonnet-4-20250514",
        "max_turns": 10,
        "temperature": 0.7,
    }


@pytest.fixture
def mock_sme_registry():
    """Provide a mock SME registry."""
    return {
        "cloud_architect": {
            "persona_id": "cloud_architect",
            "name": "Cloud Architect",
            "domain": "Cloud Infrastructure",
            "trigger_keywords": ["aws", "azure", "cloud"],
        },
        "security_analyst": {
            "persona_id": "security_analyst",
            "name": "Security Analyst",
            "domain": "Security",
            "trigger_keywords": ["security", "vulnerability"],
        },
    }


@pytest.fixture
def mock_tier_classification():
    """Provide a mock tier classification result."""
    return {
        "tier": 2,
        "reasoning": "Standard complexity task requiring research and planning",
        "estimated_agents": 7,
        "requires_council": False,
        "requires_smes": False,
    }


# =============================================================================
# Mock SDK Functions
# =============================================================================

class MockClaudeResponse:
    """Mock Claude SDK response."""

    def __init__(self, content: str, model: str = "claude-sonnet-4-20250514"):
        self.content = content
        self.model = model
        self.stop_reason = "end_turn"
        self.usage = Mock(
            input_tokens=1000,
            output_tokens=500,
        )


def create_mock_query_response(
    agent_name: str,
    content: str = None,
    structured_data: Dict = None,
) -> MockClaudeResponse:
    """
    Create a mock query response for an agent.

    Args:
        agent_name: Name of the agent
        content: Raw content (optional)
        structured_data: Structured data (optional)

    Returns:
        MockClaudeResponse instance
    """
    if content is None:
        content = f"Mock response from {agent_name}"

    if structured_data:
        content = json.dumps(structured_data)

    return MockClaudeResponse(content)


@pytest.fixture
def mock_query():
    """Mock the claude_agent_sdk.query function."""
    async def _mock_query(
        prompt: str,
        model: str = None,
        max_turns: int = None,
        **kwargs
    ) -> MockClaudeResponse:
        # Return a generic response
        return MockClaudeResponse("Mock agent response")

    return _mock_query


@pytest.fixture
def mock_query_with_agent_response():
    """Mock query that returns agent-specific responses."""
    async def _mock_query(
        prompt: str,
        model: str = None,
        max_turns: int = None,
        agent_name: str = None,
        **kwargs
    ) -> MockClaudeResponse:
        if agent_name and agent_name in MOCK_AGENT_RESPONSES:
            return create_mock_query_response(
                agent_name,
                structured_data=MOCK_AGENT_RESPONSES[agent_name]
            )
        return MockClaudeResponse("Mock response")

    return _mock_query


@pytest.fixture
def mock_task_tool():
    """Mock the Task tool for spawning subagents."""
    async def _mock_task(
        agent_name: str,
        prompt: str,
        **kwargs
    ) -> MockClaudeResponse:
        return create_mock_query_response(
            agent_name,
            structured_data=MOCK_AGENT_RESPONSES.get(agent_name, {})
        )

    return _mock_task


# =============================================================================
# Patch Helpers
# =============================================================================

class MockSDKPatch:
    """Context manager for mocking SDK calls."""

    def __init__(self, query_response=None, task_response=None):
        self.query_response = query_response or MockClaudeResponse("Mock response")
        self.task_response = task_response
        self._patches = []

    def _create_query_mock(self):
        """Create a mock for the query function."""
        async def mock_query_func(*args, **kwargs):
            return self.query_response

        mock_query = AsyncMock(side_effect=mock_query_func)
        return mock_query

    def _create_task_mock(self):
        """Create a mock for the Task tool."""
        async def mock_task_func(*args, **kwargs):
            return self.task_response or self.query_response

        mock_task = AsyncMock(side_effect=mock_task_func)
        return mock_task

    def __enter__(self):
        """Enter the context and apply patches."""
        # Patch the SDK query function
        query_patch = patch("claude_agent_sdk.query", self._create_query_mock())
        query_patch.start()
        self._patches.append(query_patch)

        # Patch the Task tool
        task_patch = patch("src.orchestrator.Task", self._create_task_mock())
        task_patch.start()
        self._patches.append(task_patch)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context and remove patches."""
        for patch in self._patches:
            patch.stop()


@pytest.fixture
def mock_sdk():
    """Fixture providing a mock SDK context manager."""
    return MockSDKPatch()


# =============================================================================
# Test Data Builders
# =============================================================================

class TestDataBuilder:
    """Builder for creating test data."""

    @staticmethod
    def create_task_intelligence_report(**overrides):
        """Create a mock TaskIntelligenceReport."""
        default = {
            "literal_request": "Test request",
            "inferred_intent": "Test intent",
            "sub_tasks": [],
            "missing_info": [],
            "assumptions": [],
            "modality": "text",
            "recommended_approach": "Test approach",
            "escalation_needed": False,
        }
        default.update(overrides)
        return default

    @staticmethod
    def create_execution_plan(**overrides):
        """Create a mock ExecutionPlan."""
        default = {
            "steps": [],
            "estimated_duration": "5 minutes",
            "dependencies": [],
            "parallel_opportunities": [],
            "risk_factors": [],
        }
        default.update(overrides)
        return default

    @staticmethod
    def create_verification_report(**overrides):
        """Create a mock VerificationReport."""
        default = {
            "claims": [],
            "factual_accuracy_score": 1.0,
            "hallucination_risk": "low",
            "recommendations": [],
        }
        default.update(overrides)
        return default

    @staticmethod
    def create_critique_report(**overrides):
        """Create a mock CritiqueReport."""
        default = {
            "attack_vectors_tested": [],
            "findings": [],
            "overall_score": 1.0,
            "recommendations": [],
        }
        default.update(overrides)
        return default

    @staticmethod
    def create_review_verdict(**overrides):
        """Create a mock ReviewVerdict."""
        default = {
            "verdict": "PROCEED",
            "quality_gates": {},
            "final_recommendation": "Approved",
        }
        default.update(overrides)
        return default


@pytest.fixture
def test_data_builder():
    """Fixture providing the TestDataBuilder."""
    return TestDataBuilder()


# =============================================================================
# Assertion Helpers
# =============================================================================

class CustomAssertions:
    """Custom assertion helpers for testing."""

    @staticmethod
    def assert_valid_schema(obj: BaseModel, schema_class: type):
        """Assert that an object is valid for a schema class."""
        assert isinstance(obj, schema_class), f"Expected {schema_class}, got {type(obj)}"

    @staticmethod
    def assert_agent_response(response: Dict[str, Any], agent_name: str):
        """Assert that a response contains expected agent fields."""
        assert "content" in response or "data" in response, f"{agent_name} response missing content/data"

    @staticmethod
    def assert_quality_gate_verdict(verdict: str, expected_verdict: str = None):
        """Assert that a verdict is valid."""
        valid_verdicts = [
            "PROCEED_TO_FORMATTER",
            "EXECUTOR_REVISE",
            "RESEARCHER_REVERIFY",
            "FULL_REGENERATION",
        ]
        assert verdict in valid_verdicts, f"Invalid verdict: {verdict}"

        if expected_verdict:
            assert verdict == expected_verdict, f"Expected {expected_verdict}, got {verdict}"

    @staticmethod
    def assert_token_cost(tokens: int, cost: float, model: str):
        """Assert that cost calculation is reasonable for the model and tokens."""
        if "haiku" in model.lower():
            expected_max = (tokens / 1_000_000) * 1.5  # Haiku max cost
        elif "opus" in model.lower():
            expected_max = (tokens / 1_000_000) * 80  # Opus max cost
        else:  # Sonnet
            expected_max = (tokens / 1_000_000) * 20  # Sonnet max cost

        assert cost <= expected_max, f"Cost ${cost:.4f} exceeds expected max ${expected_max:.4f}"
        assert cost > 0, f"Cost ${cost:.4f} should be positive"


@pytest.fixture
def custom_assertions():
    """Fixture providing custom assertions."""
    return CustomAssertions()


# =============================================================================
# Environment Setup
# =============================================================================

@pytest.fixture(autouse=True)
def set_test_environment():
    """Set up test environment variables."""
    # Set test environment variables
    os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test-000000000000000000000000000000000000000000000000"
    os.environ["TESTING"] = "true"

    yield

    # Cleanup
    os.environ.pop("ANTHROPIC_API_KEY", None)
    os.environ.pop("TESTING", None)


# =============================================================================
# Test Utilities
# =============================================================================

def async_test(coro):
    """
    Decorator to run async tests in pytest.

    Usage:
        @async_test
        async def test_something():
            await async_function()
    """
    def wrapper(*args, **kwargs):
        import asyncio
        return asyncio.run(coro(*args, **kwargs))

    return wrapper


def skip_if_no_api_key():
    """Decorator to skip tests if API key is not available."""
    return pytest.mark.skipif(
        not os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY").startswith("sk-ant-test"),
        reason="No real API key available"
    )


def requires_integration():
    """Decorator for tests requiring integration environment."""
    return pytest.mark.skipif(
        os.getenv("RUN_INTEGRATION_TESTS") != "true",
        reason="Integration tests not enabled"
    )
