"""
Exhaustive Tests for SDK Integration Module

Tests PermissionMode enum, ClaudeAgentOptions dataclass, AGENT_ALLOWED_TOOLS,
build_agent_options(), spawn_subagent(), _execute_sdk_query(), _execute_anthropic_api(),
_validate_output(), get_skills_for_agent(), create_sdk_mcp_server(), and edge cases.
"""

import sys
import os
import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock
from dataclasses import fields

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from src.core.sdk_integration import (
    PermissionMode,
    ClaudeAgentOptions,
    AGENT_ALLOWED_TOOLS,
    build_agent_options,
    spawn_subagent,
    _execute_sdk_query,
    _execute_anthropic_api,
    _validate_output,
    _get_output_schema,
    get_skills_for_agent,
    get_skills_for_sme,
    create_sdk_mcp_server,
    _simulate_response,
)

import src.config.settings as settings_module


@pytest.fixture(autouse=True)
def reset_global_settings():
    """Reset global settings singleton between tests."""
    settings_module._global_settings = None
    yield
    settings_module._global_settings = None


# =============================================================================
# PermissionMode Enum Tests
# =============================================================================

class TestPermissionMode:
    """Tests for the PermissionMode enum."""

    def test_default_value(self):
        assert PermissionMode.DEFAULT.value == "default"

    def test_accept_edits_value(self):
        assert PermissionMode.ACCEPT_EDITS.value == "acceptEdits"

    def test_member_count(self):
        assert len(PermissionMode) == 2

    def test_is_str_enum(self):
        assert isinstance(PermissionMode.DEFAULT, str)
        assert isinstance(PermissionMode.ACCEPT_EDITS, str)

    def test_from_value(self):
        assert PermissionMode("default") == PermissionMode.DEFAULT
        assert PermissionMode("acceptEdits") == PermissionMode.ACCEPT_EDITS

    def test_invalid_value(self):
        with pytest.raises(ValueError):
            PermissionMode("invalid")

    def test_string_comparison(self):
        assert PermissionMode.DEFAULT == "default"
        assert PermissionMode.ACCEPT_EDITS == "acceptEdits"


# =============================================================================
# ClaudeAgentOptions Dataclass Tests
# =============================================================================

class TestClaudeAgentOptions:
    """Tests for the ClaudeAgentOptions dataclass."""

    def test_required_fields(self):
        """name, model, system_prompt are required."""
        opts = ClaudeAgentOptions(
            name="TestAgent",
            model="claude-3-5-sonnet-20241022",
            system_prompt="You are a test agent.",
        )
        assert opts.name == "TestAgent"
        assert opts.model == "claude-3-5-sonnet-20241022"
        assert opts.system_prompt == "You are a test agent."

    def test_default_values(self):
        opts = ClaudeAgentOptions(
            name="Agent", model="model", system_prompt="prompt"
        )
        assert opts.max_turns == 30
        assert opts.allowed_tools == []
        assert opts.output_format is None
        assert opts.setting_sources == ["user", "project"]
        assert opts.permission_mode == PermissionMode.DEFAULT
        assert opts.append_system_prompt is None

    def test_custom_values(self):
        opts = ClaudeAgentOptions(
            name="Executor",
            model="claude-3-5-opus-20240507",
            system_prompt="Execute tasks.",
            max_turns=50,
            allowed_tools=["Read", "Write", "Bash"],
            output_format={"type": "json_schema"},
            setting_sources=["project"],
            permission_mode=PermissionMode.ACCEPT_EDITS,
            append_system_prompt="Extra instructions.",
        )
        assert opts.max_turns == 50
        assert opts.allowed_tools == ["Read", "Write", "Bash"]
        assert opts.output_format == {"type": "json_schema"}
        assert opts.setting_sources == ["project"]
        assert opts.permission_mode == PermissionMode.ACCEPT_EDITS
        assert opts.append_system_prompt == "Extra instructions."

    def test_field_count(self):
        """Verify number of fields in the dataclass."""
        assert len(fields(ClaudeAgentOptions)) == 9

    def test_allowed_tools_default_factory(self):
        """Each instance should get its own list."""
        opts1 = ClaudeAgentOptions(name="A", model="m", system_prompt="p")
        opts2 = ClaudeAgentOptions(name="B", model="m", system_prompt="p")
        opts1.allowed_tools.append("Read")
        assert "Read" not in opts2.allowed_tools

    def test_setting_sources_default_factory(self):
        opts1 = ClaudeAgentOptions(name="A", model="m", system_prompt="p")
        opts2 = ClaudeAgentOptions(name="B", model="m", system_prompt="p")
        opts1.setting_sources.append("extra")
        assert "extra" not in opts2.setting_sources

    # -------------------------------------------------------------------------
    # to_sdk_kwargs() tests
    # -------------------------------------------------------------------------

    def test_to_sdk_kwargs_minimal(self):
        opts = ClaudeAgentOptions(
            name="Agent", model="model-id", system_prompt="sys prompt"
        )
        kwargs = opts.to_sdk_kwargs()
        assert kwargs["name"] == "Agent"
        assert kwargs["model"] == "model-id"
        assert kwargs["system_prompt"] == "sys prompt"
        assert kwargs["max_turns"] == 30
        # DEFAULT permission mode should NOT be included
        assert "permission_mode" not in kwargs

    def test_to_sdk_kwargs_with_allowed_tools(self):
        opts = ClaudeAgentOptions(
            name="A", model="m", system_prompt="p",
            allowed_tools=["Read", "Write"],
        )
        kwargs = opts.to_sdk_kwargs()
        assert kwargs["allowed_tools"] == ["Read", "Write"]

    def test_to_sdk_kwargs_empty_allowed_tools_excluded(self):
        opts = ClaudeAgentOptions(
            name="A", model="m", system_prompt="p",
            allowed_tools=[],
        )
        kwargs = opts.to_sdk_kwargs()
        assert "allowed_tools" not in kwargs

    def test_to_sdk_kwargs_with_output_format(self):
        schema = {"type": "object", "required": ["result"]}
        opts = ClaudeAgentOptions(
            name="A", model="m", system_prompt="p",
            output_format=schema,
        )
        kwargs = opts.to_sdk_kwargs()
        assert kwargs["output_format"] == schema

    def test_to_sdk_kwargs_output_format_none_excluded(self):
        opts = ClaudeAgentOptions(name="A", model="m", system_prompt="p")
        kwargs = opts.to_sdk_kwargs()
        assert "output_format" not in kwargs

    def test_to_sdk_kwargs_with_setting_sources(self):
        opts = ClaudeAgentOptions(
            name="A", model="m", system_prompt="p",
            setting_sources=["user", "project"],
        )
        kwargs = opts.to_sdk_kwargs()
        assert kwargs["setting_sources"] == ["user", "project"]

    def test_to_sdk_kwargs_empty_setting_sources_excluded(self):
        opts = ClaudeAgentOptions(
            name="A", model="m", system_prompt="p",
            setting_sources=[],
        )
        kwargs = opts.to_sdk_kwargs()
        assert "setting_sources" not in kwargs

    def test_to_sdk_kwargs_with_accept_edits(self):
        opts = ClaudeAgentOptions(
            name="A", model="m", system_prompt="p",
            permission_mode=PermissionMode.ACCEPT_EDITS,
        )
        kwargs = opts.to_sdk_kwargs()
        assert kwargs["permission_mode"] == "acceptEdits"

    def test_to_sdk_kwargs_default_permission_excluded(self):
        opts = ClaudeAgentOptions(
            name="A", model="m", system_prompt="p",
            permission_mode=PermissionMode.DEFAULT,
        )
        kwargs = opts.to_sdk_kwargs()
        assert "permission_mode" not in kwargs

    def test_to_sdk_kwargs_with_append_system_prompt(self):
        opts = ClaudeAgentOptions(
            name="A", model="m", system_prompt="p",
            append_system_prompt="Extra info.",
        )
        kwargs = opts.to_sdk_kwargs()
        assert kwargs["append_system_prompt"] == "Extra info."

    def test_to_sdk_kwargs_append_system_prompt_none_excluded(self):
        opts = ClaudeAgentOptions(name="A", model="m", system_prompt="p")
        kwargs = opts.to_sdk_kwargs()
        assert "append_system_prompt" not in kwargs

    def test_to_sdk_kwargs_full(self):
        opts = ClaudeAgentOptions(
            name="Executor",
            model="claude-3-5-sonnet-20241022",
            system_prompt="Execute.",
            max_turns=50,
            allowed_tools=["Read", "Write", "Bash"],
            output_format={"type": "object"},
            setting_sources=["user"],
            permission_mode=PermissionMode.ACCEPT_EDITS,
            append_system_prompt="Be careful.",
        )
        kwargs = opts.to_sdk_kwargs()
        assert len(kwargs) == 9  # all keys present


# =============================================================================
# AGENT_ALLOWED_TOOLS Tests
# =============================================================================

class TestAgentAllowedTools:
    """Tests for the AGENT_ALLOWED_TOOLS dictionary."""

    EXPECTED_AGENTS = [
        "analyst", "planner", "clarifier", "researcher", "executor",
        "code_reviewer", "formatter", "verifier", "critic", "reviewer",
        "memory_curator", "council_chair", "quality_arbiter",
        "ethics_advisor", "sme_default",
    ]

    def test_all_expected_agents_present(self):
        for agent in self.EXPECTED_AGENTS:
            assert agent in AGENT_ALLOWED_TOOLS, f"Missing agent: {agent}"

    def test_agent_count(self):
        assert len(AGENT_ALLOWED_TOOLS) == len(self.EXPECTED_AGENTS)

    # Tool sets per agent

    def test_analyst_tools(self):
        assert AGENT_ALLOWED_TOOLS["analyst"] == ["Read", "Glob", "Grep"]

    def test_planner_tools(self):
        assert AGENT_ALLOWED_TOOLS["planner"] == ["Read", "Glob"]

    def test_clarifier_tools(self):
        assert AGENT_ALLOWED_TOOLS["clarifier"] == []

    def test_researcher_tools(self):
        assert AGENT_ALLOWED_TOOLS["researcher"] == ["WebSearch", "WebFetch", "Read"]

    def test_executor_tools(self):
        expected = ["Read", "Write", "Edit", "Bash", "Glob", "Grep", "Skill"]
        assert AGENT_ALLOWED_TOOLS["executor"] == expected

    def test_code_reviewer_tools(self):
        assert AGENT_ALLOWED_TOOLS["code_reviewer"] == ["Read", "Glob", "Grep", "Bash"]

    def test_formatter_tools(self):
        assert AGENT_ALLOWED_TOOLS["formatter"] == ["Read", "Write", "Bash", "Skill"]

    def test_verifier_tools(self):
        assert AGENT_ALLOWED_TOOLS["verifier"] == ["Read", "WebSearch", "WebFetch"]

    def test_critic_tools(self):
        assert AGENT_ALLOWED_TOOLS["critic"] == ["Read", "Grep"]

    def test_reviewer_tools(self):
        assert AGENT_ALLOWED_TOOLS["reviewer"] == ["Read", "Glob", "Grep"]

    def test_memory_curator_tools(self):
        assert AGENT_ALLOWED_TOOLS["memory_curator"] == ["Read", "Write", "Glob"]

    def test_council_chair_tools(self):
        assert AGENT_ALLOWED_TOOLS["council_chair"] == []

    def test_quality_arbiter_tools(self):
        assert AGENT_ALLOWED_TOOLS["quality_arbiter"] == []

    def test_ethics_advisor_tools(self):
        assert AGENT_ALLOWED_TOOLS["ethics_advisor"] == []

    def test_sme_default_tools(self):
        assert AGENT_ALLOWED_TOOLS["sme_default"] == ["Read", "Glob", "Grep", "Skill"]

    # Tool access verification

    @pytest.mark.parametrize("tool", ["Read"])
    def test_read_access(self, tool):
        """Most agents should have Read access."""
        agents_with_read = [
            a for a, tools in AGENT_ALLOWED_TOOLS.items() if tool in tools
        ]
        assert len(agents_with_read) >= 10

    @pytest.mark.parametrize("tool", ["Write"])
    def test_write_access_restricted(self, tool):
        """Only executor, formatter, and memory_curator should have Write."""
        agents_with_write = [
            a for a, tools in AGENT_ALLOWED_TOOLS.items() if tool in tools
        ]
        assert set(agents_with_write) == {"executor", "formatter", "memory_curator"}

    def test_bash_access_restricted(self):
        agents_with_bash = [
            a for a, tools in AGENT_ALLOWED_TOOLS.items() if "Bash" in tools
        ]
        assert set(agents_with_bash) == {"executor", "code_reviewer", "formatter"}

    def test_websearch_access(self):
        agents_with_web = [
            a for a, tools in AGENT_ALLOWED_TOOLS.items() if "WebSearch" in tools
        ]
        assert set(agents_with_web) == {"researcher", "verifier"}

    def test_webfetch_access(self):
        agents_with_fetch = [
            a for a, tools in AGENT_ALLOWED_TOOLS.items() if "WebFetch" in tools
        ]
        assert set(agents_with_fetch) == {"researcher", "verifier"}

    def test_skill_access(self):
        agents_with_skill = [
            a for a, tools in AGENT_ALLOWED_TOOLS.items() if "Skill" in tools
        ]
        assert set(agents_with_skill) == {"executor", "formatter", "sme_default"}

    def test_edit_access(self):
        agents_with_edit = [
            a for a, tools in AGENT_ALLOWED_TOOLS.items() if "Edit" in tools
        ]
        assert set(agents_with_edit) == {"executor"}

    def test_glob_access(self):
        agents_with_glob = [
            a for a, tools in AGENT_ALLOWED_TOOLS.items() if "Glob" in tools
        ]
        assert len(agents_with_glob) >= 6

    def test_grep_access(self):
        agents_with_grep = [
            a for a, tools in AGENT_ALLOWED_TOOLS.items() if "Grep" in tools
        ]
        assert len(agents_with_grep) >= 5

    def test_all_tools_are_strings(self):
        for agent, tools in AGENT_ALLOWED_TOOLS.items():
            assert isinstance(tools, list), f"{agent} tools is not a list"
            for tool in tools:
                assert isinstance(tool, str), f"{agent} has non-string tool: {tool}"


# =============================================================================
# _get_output_schema() Tests
# =============================================================================

class TestGetOutputSchema:
    """Tests for the _get_output_schema() function."""

    @pytest.mark.parametrize("agent_name", [
        "analyst", "planner", "clarifier", "researcher",
        "code_reviewer", "verifier", "critic", "reviewer",
        "council_chair", "quality_arbiter", "ethics_advisor",
    ])
    def test_known_agents(self, agent_name):
        """Known agents should return a schema or None without error."""
        # This may return None if schemas aren't importable, but should not raise
        result = _get_output_schema(agent_name)
        assert result is None or isinstance(result, dict)

    def test_unknown_agent_returns_none(self):
        assert _get_output_schema("nonexistent_agent") is None

    def test_executor_has_no_schema(self):
        """Executor is not in the schema_map."""
        assert _get_output_schema("executor") is None

    def test_formatter_has_no_schema(self):
        assert _get_output_schema("formatter") is None


# =============================================================================
# build_agent_options() Tests
# =============================================================================

class TestBuildAgentOptions:
    """Tests for the build_agent_options() function."""

    @patch("src.core.sdk_integration.get_settings")
    @patch("src.core.sdk_integration.get_model_for_agent")
    def test_basic_build(self, mock_model, mock_settings):
        mock_model.return_value = "claude-3-5-sonnet-20241022"
        mock_settings.return_value = MagicMock(
            max_turns_orchestrator=200,
            max_turns_subagent=30,
            max_turns_executor=50,
        )
        opts = build_agent_options("analyst", "You are an analyst.")
        assert opts.name == "Analyst"
        assert opts.model == "claude-3-5-sonnet-20241022"
        assert opts.system_prompt == "You are an analyst."
        assert opts.max_turns == 30
        assert opts.allowed_tools == ["Read", "Glob", "Grep"]
        assert opts.permission_mode == PermissionMode.DEFAULT

    @patch("src.core.sdk_integration.get_settings")
    @patch("src.core.sdk_integration.get_model_for_agent")
    def test_executor_permission_mode(self, mock_model, mock_settings):
        mock_model.return_value = "claude-3-5-sonnet-20241022"
        mock_settings.return_value = MagicMock(
            max_turns_orchestrator=200,
            max_turns_subagent=30,
            max_turns_executor=50,
        )
        opts = build_agent_options("executor", "Execute tasks.")
        assert opts.permission_mode == PermissionMode.ACCEPT_EDITS

    @patch("src.core.sdk_integration.get_settings")
    @patch("src.core.sdk_integration.get_model_for_agent")
    def test_executor_max_turns(self, mock_model, mock_settings):
        mock_model.return_value = "model"
        mock_settings.return_value = MagicMock(
            max_turns_orchestrator=200,
            max_turns_subagent=30,
            max_turns_executor=50,
        )
        opts = build_agent_options("executor", "prompt")
        assert opts.max_turns == 50

    @patch("src.core.sdk_integration.get_settings")
    @patch("src.core.sdk_integration.get_model_for_agent")
    def test_orchestrator_max_turns(self, mock_model, mock_settings):
        mock_model.return_value = "model"
        mock_settings.return_value = MagicMock(
            max_turns_orchestrator=200,
            max_turns_subagent=30,
            max_turns_executor=50,
        )
        opts = build_agent_options("orchestrator", "prompt")
        assert opts.max_turns == 200

    @patch("src.core.sdk_integration.get_settings")
    @patch("src.core.sdk_integration.get_model_for_agent")
    def test_subagent_max_turns(self, mock_model, mock_settings):
        mock_model.return_value = "model"
        mock_settings.return_value = MagicMock(
            max_turns_orchestrator=200,
            max_turns_subagent=30,
            max_turns_executor=50,
        )
        opts = build_agent_options("analyst", "prompt")
        assert opts.max_turns == 30

    @patch("src.core.sdk_integration.get_settings")
    @patch("src.core.sdk_integration.get_model_for_agent")
    def test_max_turns_override(self, mock_model, mock_settings):
        mock_model.return_value = "model"
        mock_settings.return_value = MagicMock(
            max_turns_orchestrator=200,
            max_turns_subagent=30,
            max_turns_executor=50,
        )
        opts = build_agent_options("analyst", "prompt", max_turns_override=99)
        assert opts.max_turns == 99

    @patch("src.core.sdk_integration.get_settings")
    @patch("src.core.sdk_integration.get_model_for_agent")
    def test_model_override(self, mock_model, mock_settings):
        mock_model.return_value = "default-model"
        mock_settings.return_value = MagicMock(
            max_turns_orchestrator=200,
            max_turns_subagent=30,
            max_turns_executor=50,
        )
        opts = build_agent_options(
            "analyst", "prompt", model_override="custom-model"
        )
        assert opts.model == "custom-model"

    @patch("src.core.sdk_integration.get_settings")
    @patch("src.core.sdk_integration.get_model_for_agent")
    def test_extra_tools(self, mock_model, mock_settings):
        mock_model.return_value = "model"
        mock_settings.return_value = MagicMock(
            max_turns_orchestrator=200,
            max_turns_subagent=30,
            max_turns_executor=50,
        )
        opts = build_agent_options(
            "analyst", "prompt", extra_tools=["CustomTool"]
        )
        assert "CustomTool" in opts.allowed_tools
        assert "Read" in opts.allowed_tools  # original tools preserved

    @patch("src.core.sdk_integration.get_settings")
    @patch("src.core.sdk_integration.get_model_for_agent")
    def test_extra_system_prompt(self, mock_model, mock_settings):
        mock_model.return_value = "model"
        mock_settings.return_value = MagicMock(
            max_turns_orchestrator=200,
            max_turns_subagent=30,
            max_turns_executor=50,
        )
        opts = build_agent_options(
            "analyst", "prompt", extra_system_prompt="Be thorough."
        )
        assert opts.append_system_prompt == "Be thorough."

    @patch("src.core.sdk_integration.get_settings")
    @patch("src.core.sdk_integration.get_model_for_agent")
    def test_agent_role_display_name(self, mock_model, mock_settings):
        mock_model.return_value = "model"
        mock_settings.return_value = MagicMock(
            max_turns_orchestrator=200,
            max_turns_subagent=30,
            max_turns_executor=50,
        )
        opts = build_agent_options(
            "council", "prompt", agent_role="chair"
        )
        assert opts.name == "Chair"

    @patch("src.core.sdk_integration.get_settings")
    @patch("src.core.sdk_integration.get_model_for_agent")
    def test_agent_role_tool_resolution(self, mock_model, mock_settings):
        """When agent_role creates a valid key like council_chair, use its tools."""
        mock_model.return_value = "model"
        mock_settings.return_value = MagicMock(
            max_turns_orchestrator=200,
            max_turns_subagent=30,
            max_turns_executor=50,
        )
        opts = build_agent_options(
            "council", "prompt", agent_role="chair"
        )
        # council_chair is in AGENT_ALLOWED_TOOLS with empty tools
        assert opts.allowed_tools == []

    @patch("src.core.sdk_integration.get_settings")
    @patch("src.core.sdk_integration.get_model_for_agent")
    def test_unknown_agent_empty_tools(self, mock_model, mock_settings):
        mock_model.return_value = "model"
        mock_settings.return_value = MagicMock(
            max_turns_orchestrator=200,
            max_turns_subagent=30,
            max_turns_executor=50,
        )
        opts = build_agent_options("nonexistent_agent", "prompt")
        assert opts.allowed_tools == []

    @patch("src.core.sdk_integration.get_settings")
    @patch("src.core.sdk_integration.get_model_for_agent")
    def test_setting_sources_always_set(self, mock_model, mock_settings):
        mock_model.return_value = "model"
        mock_settings.return_value = MagicMock(
            max_turns_orchestrator=200,
            max_turns_subagent=30,
            max_turns_executor=50,
        )
        opts = build_agent_options("analyst", "prompt")
        assert opts.setting_sources == ["user", "project"]

    @patch("src.core.sdk_integration.get_settings")
    @patch("src.core.sdk_integration.get_model_for_agent")
    def test_non_executor_permission_default(self, mock_model, mock_settings):
        mock_model.return_value = "model"
        mock_settings.return_value = MagicMock(
            max_turns_orchestrator=200,
            max_turns_subagent=30,
            max_turns_executor=50,
        )
        for agent in ["analyst", "planner", "critic", "reviewer", "verifier"]:
            opts = build_agent_options(agent, "prompt")
            assert opts.permission_mode == PermissionMode.DEFAULT, f"{agent} should have DEFAULT"

    @pytest.mark.parametrize("agent_name,expected_display", [
        ("analyst", "Analyst"),
        ("code_reviewer", "Code Reviewer"),
        ("memory_curator", "Memory Curator"),
        ("council_chair", "Council Chair"),
    ])
    @patch("src.core.sdk_integration.get_settings")
    @patch("src.core.sdk_integration.get_model_for_agent")
    def test_display_name_formatting(self, mock_model, mock_settings, agent_name, expected_display):
        mock_model.return_value = "model"
        mock_settings.return_value = MagicMock(
            max_turns_orchestrator=200,
            max_turns_subagent=30,
            max_turns_executor=50,
        )
        opts = build_agent_options(agent_name, "prompt")
        assert opts.name == expected_display


# =============================================================================
# spawn_subagent() Tests
# =============================================================================

class TestSpawnSubagent:
    """Tests for the spawn_subagent() function."""

    def _make_options(self, **kwargs):
        defaults = {
            "name": "TestAgent",
            "model": "test-model",
            "system_prompt": "Test prompt.",
        }
        defaults.update(kwargs)
        return ClaudeAgentOptions(**defaults)

    @patch("src.core.sdk_integration._execute_sdk_query")
    def test_success_path(self, mock_exec):
        mock_exec.return_value = {
            "output": "result text",
            "tokens_used": 100,
            "cost_usd": 0.01,
        }
        opts = self._make_options()
        result = spawn_subagent(opts, "test input")

        assert result["status"] == "success"
        assert result["output"] == "result text"
        assert result["tokens_used"] == 100
        assert result["cost_usd"] == 0.01
        assert result["model"] == "test-model"
        assert result["retries"] == 0
        assert "duration_ms" in result

    @patch("src.core.sdk_integration._execute_sdk_query")
    def test_duration_is_positive(self, mock_exec):
        mock_exec.return_value = {"output": "ok", "tokens_used": 0, "cost_usd": 0.0}
        result = spawn_subagent(self._make_options(), "input")
        assert result["duration_ms"] >= 0

    @patch("src.core.sdk_integration._execute_sdk_query")
    def test_retry_on_exception(self, mock_exec):
        """On exception, retry up to max_retries."""
        mock_exec.side_effect = [
            Exception("transient error"),
            {"output": "success", "tokens_used": 50, "cost_usd": 0.005},
        ]
        opts = self._make_options()
        with patch("time.sleep"):  # avoid actual sleep
            result = spawn_subagent(opts, "input", max_retries=2)
        assert result["status"] == "success"
        assert result["retries"] == 1

    @patch("src.core.sdk_integration._execute_sdk_query")
    def test_max_retries_exceeded(self, mock_exec):
        mock_exec.side_effect = Exception("persistent error")
        opts = self._make_options()
        with patch("time.sleep"):
            result = spawn_subagent(opts, "input", max_retries=1)
        assert result["status"] == "error"
        assert "persistent error" in result["error"]
        assert result["output"] is None
        assert result["retries"] == 1

    @patch("src.core.sdk_integration._execute_sdk_query")
    def test_zero_max_retries(self, mock_exec):
        mock_exec.side_effect = Exception("fail")
        opts = self._make_options()
        result = spawn_subagent(opts, "input", max_retries=0)
        assert result["status"] == "error"
        assert result["retries"] == 0

    @patch("src.core.sdk_integration._execute_sdk_query")
    @patch("src.core.sdk_integration._validate_output")
    def test_retry_on_validation_failure(self, mock_validate, mock_exec):
        mock_exec.return_value = {
            "output": '{"bad": "data"}',
            "tokens_used": 50,
            "cost_usd": 0.005,
        }
        # First call fails validation, second succeeds
        mock_validate.side_effect = [False, True]
        opts = self._make_options(output_format={"required": ["result"]})
        result = spawn_subagent(opts, "input", max_retries=2)
        assert result["retries"] == 1

    @patch("src.core.sdk_integration._execute_sdk_query")
    def test_no_validation_without_output_format(self, mock_exec):
        mock_exec.return_value = {
            "output": "plain text",
            "tokens_used": 50,
            "cost_usd": 0.005,
        }
        opts = self._make_options()  # No output_format
        result = spawn_subagent(opts, "input")
        assert result["status"] == "success"

    @patch("src.core.sdk_integration._execute_sdk_query")
    def test_error_result_structure(self, mock_exec):
        mock_exec.side_effect = Exception("boom")
        opts = self._make_options()
        result = spawn_subagent(opts, "input", max_retries=0)
        assert "status" in result
        assert "output" in result
        assert "error" in result
        assert "tokens_used" in result
        assert "cost_usd" in result
        assert "duration_ms" in result
        assert "model" in result
        assert "retries" in result

    @patch("src.core.sdk_integration._execute_sdk_query")
    def test_empty_prompt(self, mock_exec):
        mock_exec.return_value = {"output": "", "tokens_used": 0, "cost_usd": 0.0}
        result = spawn_subagent(self._make_options(), "")
        assert result["status"] == "success"


# =============================================================================
# _execute_sdk_query() Tests
# =============================================================================

class TestExecuteSdkQuery:
    """Tests for the _execute_sdk_query() function."""

    def test_sdk_import_path(self):
        """When claude_agent_sdk is not available, falls back to anthropic API."""
        with patch(
            "src.core.sdk_integration._execute_anthropic_api",
            return_value={"output": "fallback", "tokens_used": 10, "cost_usd": 0.001},
        ) as mock_api:
            result = _execute_sdk_query(
                {"system_prompt": "sys", "model": "m", "max_turns": 5},
                "hello",
            )
            assert result["output"] == "fallback"
            mock_api.assert_called_once()

    @patch.dict("sys.modules", {"claude_agent_sdk": MagicMock()})
    def test_sdk_available_path(self):
        """When SDK is available, use it."""
        mock_sdk = sys.modules["claude_agent_sdk"]
        mock_sdk.query.return_value = {
            "response": "sdk result",
            "usage": {"total_tokens": 200},
            "cost": 0.02,
        }
        result = _execute_sdk_query(
            {"system_prompt": "sys", "model": "m", "max_turns": 5},
            "hello",
        )
        assert result["output"] == "sdk result"
        assert result["tokens_used"] == 200
        assert result["cost_usd"] == 0.02

    @patch.dict("sys.modules", {"claude_agent_sdk": MagicMock()})
    def test_sdk_output_key_fallback(self):
        """SDK may return 'output' instead of 'response'."""
        mock_sdk = sys.modules["claude_agent_sdk"]
        mock_sdk.query.return_value = {
            "output": "alt output",
            "usage": {"total_tokens": 100},
            "cost": 0.01,
        }
        result = _execute_sdk_query(
            {"system_prompt": "sys", "model": "m"},
            "hello",
        )
        assert result["output"] == "alt output"


# =============================================================================
# _execute_anthropic_api() Tests
# =============================================================================

class TestExecuteAnthropicApi:
    """Tests for the _execute_anthropic_api() function."""

    def test_anthropic_not_available_falls_to_simulation(self):
        """When anthropic is not importable, fall back to _simulate_response."""
        with patch(
            "src.core.sdk_integration._simulate_response",
            return_value={"output": "simulated", "tokens_used": 500, "cost_usd": 0.005},
        ) as mock_sim:
            # Force ImportError for anthropic
            with patch.dict("sys.modules", {"anthropic": None}):
                result = _execute_anthropic_api({"name": "Agent"}, "test input")
            assert result["output"] == "simulated"

    def test_anthropic_api_call(self):
        """Test the direct API call path with mocked Anthropic client."""
        mock_block = MagicMock()
        mock_block.text = "API response text"

        mock_response = MagicMock()
        mock_response.content = [mock_block]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        mock_anthropic_module = MagicMock()
        mock_anthropic_module.Anthropic.return_value = mock_client

        with patch.dict("sys.modules", {"anthropic": mock_anthropic_module}):
            result = _execute_anthropic_api(
                {"model": "claude-3-5-sonnet-20241022", "system_prompt": "sys"},
                "test input",
            )
            assert isinstance(result, dict)
            assert result["output"] == "API response text"
            assert result["tokens_used"] == 150
            assert result["cost_usd"] > 0


# =============================================================================
# _simulate_response() Tests
# =============================================================================

class TestSimulateResponse:
    """Tests for the _simulate_response() function."""

    def test_basic_simulation(self):
        result = _simulate_response({"name": "TestAgent"}, "hello world")
        assert "TestAgent" in result["output"]
        assert "hello world" in result["output"]
        assert result["tokens_used"] == 500
        assert result["cost_usd"] == 0.005

    def test_truncation_of_long_input(self):
        long_input = "x" * 500
        result = _simulate_response({"name": "A"}, long_input)
        # Input is truncated to 200 chars in the output
        assert "..." in result["output"]

    def test_missing_name(self):
        result = _simulate_response({}, "input")
        assert "Agent" in result["output"]


# =============================================================================
# _validate_output() Tests
# =============================================================================

class TestValidateOutput:
    """Tests for the _validate_output() function."""

    def test_valid_json_string(self):
        schema = {"required": ["name", "value"]}
        output = json.dumps({"name": "test", "value": 42})
        assert _validate_output(output, schema) is True

    def test_missing_required_field(self):
        schema = {"required": ["name", "value"]}
        output = json.dumps({"name": "test"})
        assert _validate_output(output, schema) is False

    def test_dict_input(self):
        schema = {"required": ["a"]}
        assert _validate_output({"a": 1}, schema) is True

    def test_empty_output(self):
        assert _validate_output("", {"required": []}) is False
        assert _validate_output(None, {"required": []}) is False

    def test_no_required_fields(self):
        schema = {}
        output = json.dumps({"anything": "goes"})
        assert _validate_output(output, schema) is True

    def test_invalid_json_string(self):
        schema = {"required": ["a"]}
        assert _validate_output("not json", schema) is False

    def test_non_dict_non_str_input(self):
        schema = {"required": []}
        assert _validate_output(42, schema) is False
        assert _validate_output(["list"], schema) is False

    def test_empty_required_list(self):
        schema = {"required": []}
        output = json.dumps({"x": 1})
        assert _validate_output(output, schema) is True

    @pytest.mark.parametrize("output,expected", [
        ('{"a":1,"b":2}', True),
        ('{"a":1}', False),
        ('{}', False),
    ])
    def test_parametrized_required_fields(self, output, expected):
        schema = {"required": ["a", "b"]}
        assert _validate_output(output, schema) is expected


# =============================================================================
# get_skills_for_agent() Tests
# =============================================================================

class TestGetSkillsForAgent:
    """Tests for the get_skills_for_agent() function."""

    @pytest.mark.parametrize("agent_name,expected_skills", [
        ("executor", ["code-generation"]),
        ("formatter", ["document-creation"]),
        ("analyst", ["requirements-engineering"]),
        ("planner", ["architecture-design"]),
        ("researcher", ["web-research"]),
        ("code_reviewer", ["code-generation"]),
        ("orchestrator", ["multi-agent-reasoning"]),
    ])
    def test_known_agents(self, agent_name, expected_skills):
        assert get_skills_for_agent(agent_name) == expected_skills

    def test_unknown_agent_returns_empty(self):
        assert get_skills_for_agent("nonexistent") == []

    def test_clarifier_no_skills(self):
        assert get_skills_for_agent("clarifier") == []

    def test_council_chair_no_skills(self):
        assert get_skills_for_agent("council_chair") == []

    def test_verifier_no_skills(self):
        assert get_skills_for_agent("verifier") == []

    def test_critic_no_skills(self):
        assert get_skills_for_agent("critic") == []

    def test_reviewer_no_skills(self):
        assert get_skills_for_agent("reviewer") == []

    def test_memory_curator_no_skills(self):
        assert get_skills_for_agent("memory_curator") == []

    def test_returns_list_type(self):
        for agent in ["executor", "analyst", "unknown"]:
            result = get_skills_for_agent(agent)
            assert isinstance(result, list)


# =============================================================================
# get_skills_for_sme() Tests
# =============================================================================

class TestGetSkillsForSme:
    """Tests for the get_skills_for_sme() function."""

    @patch("src.core.sme_registry.get_persona")
    def test_known_persona(self, mock_get_persona):
        mock_persona = MagicMock()
        mock_persona.skill_files = ["skill-a", "skill-b"]
        mock_get_persona.return_value = mock_persona
        assert get_skills_for_sme("cloud_architect") == ["skill-a", "skill-b"]

    @patch("src.core.sme_registry.get_persona")
    def test_unknown_persona_returns_empty(self, mock_get_persona):
        mock_get_persona.return_value = None
        assert get_skills_for_sme("nonexistent") == []


# =============================================================================
# create_sdk_mcp_server() Tests
# =============================================================================

class TestCreateSdkMcpServer:
    """Tests for the create_sdk_mcp_server() function."""

    @patch("src.tools.custom_tools.get_all_tools")
    def test_server_config_structure(self, mock_tools):
        mock_meta = MagicMock()
        mock_meta.description = "A test tool"
        mock_meta.parameters = {"param1": "desc1", "param2": "desc2"}
        mock_tools.return_value = {"test_tool": mock_meta}

        config = create_sdk_mcp_server()

        assert config["name"] == "multi-agent-reasoning"
        assert config["version"] == "1.0.0"
        assert config["tool_count"] == 1
        assert len(config["tools"]) == 1

    @patch("src.tools.custom_tools.get_all_tools")
    def test_tool_definition_format(self, mock_tools):
        mock_meta = MagicMock()
        mock_meta.description = "Does something"
        mock_meta.parameters = {"input": "The input value"}
        mock_tools.return_value = {"my_tool": mock_meta}

        config = create_sdk_mcp_server()
        tool_def = config["tools"][0]

        assert tool_def["name"] == "my_tool"
        assert tool_def["description"] == "Does something"
        assert "parameters" in tool_def
        assert tool_def["parameters"]["type"] == "object"
        assert "input" in tool_def["parameters"]["properties"]

    @patch("src.tools.custom_tools.get_all_tools")
    def test_empty_tools(self, mock_tools):
        mock_tools.return_value = {}
        config = create_sdk_mcp_server()
        assert config["tools"] == []
        assert config["tool_count"] == 0

    @patch("src.tools.custom_tools.get_all_tools")
    def test_multiple_tools(self, mock_tools):
        tools = {}
        for i in range(5):
            meta = MagicMock()
            meta.description = f"Tool {i}"
            meta.parameters = {"p": "d"}
            tools[f"tool_{i}"] = meta
        mock_tools.return_value = tools

        config = create_sdk_mcp_server()
        assert config["tool_count"] == 5
        assert len(config["tools"]) == 5


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Edge case tests for SDK integration."""

    @patch("src.core.sdk_integration.get_settings")
    @patch("src.core.sdk_integration.get_model_for_agent")
    def test_build_agent_options_empty_agent_name(self, mock_model, mock_settings):
        mock_model.return_value = "model"
        mock_settings.return_value = MagicMock(
            max_turns_orchestrator=200,
            max_turns_subagent=30,
            max_turns_executor=50,
        )
        opts = build_agent_options("", "prompt")
        assert opts.name == ""  # Title of empty string is empty string
        assert opts.allowed_tools == []  # Not in AGENT_ALLOWED_TOOLS

    @patch("src.core.sdk_integration._execute_sdk_query")
    def test_spawn_with_none_output(self, mock_exec):
        mock_exec.return_value = {"output": None, "tokens_used": 0, "cost_usd": 0.0}
        opts = ClaudeAgentOptions(name="A", model="m", system_prompt="p")
        result = spawn_subagent(opts, "input")
        assert result["status"] == "success"
        assert result["output"] is None

    @patch("src.core.sdk_integration._execute_sdk_query")
    def test_spawn_with_large_output(self, mock_exec):
        large_output = "x" * 100000
        mock_exec.return_value = {
            "output": large_output,
            "tokens_used": 50000,
            "cost_usd": 0.5,
        }
        opts = ClaudeAgentOptions(name="A", model="m", system_prompt="p")
        result = spawn_subagent(opts, "input")
        assert result["status"] == "success"
        assert len(result["output"]) == 100000

    def test_validate_output_deeply_nested_json(self):
        schema = {"required": ["level1"]}
        output = json.dumps({"level1": {"level2": {"level3": "deep"}}})
        assert _validate_output(output, schema) is True

    def test_validate_output_unicode(self):
        schema = {"required": ["name"]}
        output = json.dumps({"name": "unicode test"})
        assert _validate_output(output, schema) is True

    def test_permission_mode_in_kwargs_only_for_accept_edits(self):
        """Verify DEFAULT mode does not pollute sdk kwargs."""
        opts = ClaudeAgentOptions(
            name="A", model="m", system_prompt="p",
            permission_mode=PermissionMode.DEFAULT,
        )
        assert "permission_mode" not in opts.to_sdk_kwargs()

        opts2 = ClaudeAgentOptions(
            name="A", model="m", system_prompt="p",
            permission_mode=PermissionMode.ACCEPT_EDITS,
        )
        assert opts2.to_sdk_kwargs()["permission_mode"] == "acceptEdits"
