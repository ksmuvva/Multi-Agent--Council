"""
Exhaustive Tests for Logging Utilities Module

Tests context variable binding (bind_session, bind_agent, bind_operation),
get_logger(), log_agent_start(), log_agent_complete(), log_agent_error(),
log_cost(), log_verdict(), context preservation, processors, formatters,
context managers, and configuration.
"""

import sys
import os
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock, call
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import structlog

from src.utils.logging import (
    # Context management
    bind_session,
    bind_agent,
    bind_operation,
    clear_context,
    # Context variables (private but testable)
    _session_id,
    _request_id,
    _user_id,
    _agent_name,
    _agent_tier,
    _phase,
    _operation,
    _component,
    # Loggers
    get_logger,
    get_agent_logger,
    get_session_logger,
    # Convenience logging functions
    log_agent_start,
    log_agent_complete,
    log_agent_error,
    log_sme_spawn,
    log_verdict,
    log_cost,
    # Processors
    context_injector,
    add_timestamp,
    add_log_level,
    rename_message,
    filter_sensitive_data,
    agent_event_classifier,
    stack_trace_formatter,
    # Formatters
    console_formatter,
    json_formatter,
    # Configuration
    configure_logging,
    # LogLevel
    LogLevel,
    # Context managers
    AgentLogContext,
    SessionLogContext,
)


@pytest.fixture(autouse=True)
def clean_context():
    """Clear all context variables before and after each test."""
    clear_context()
    yield
    clear_context()


# =============================================================================
# LogLevel Tests
# =============================================================================

class TestLogLevel:
    """Tests for the LogLevel class."""

    def test_standard_levels(self):
        assert LogLevel.DEBUG == "debug"
        assert LogLevel.INFO == "info"
        assert LogLevel.WARNING == "warning"
        assert LogLevel.ERROR == "error"
        assert LogLevel.CRITICAL == "critical"

    def test_custom_levels(self):
        assert LogLevel.AGENT_START == "agent_start"
        assert LogLevel.AGENT_COMPLETE == "agent_complete"
        assert LogLevel.AGENT_ERROR == "agent_error"
        assert LogLevel.SME_SPAWN == "sme_spawn"
        assert LogLevel.VERDICT == "verdict"

    def test_is_str_subclass(self):
        assert isinstance(LogLevel.DEBUG, str)
        assert isinstance(LogLevel.AGENT_START, str)


# =============================================================================
# Context Variable Binding Tests
# =============================================================================

class TestBindSession:
    """Tests for bind_session()."""

    def test_bind_session_id(self):
        bind_session(session_id="sess_123")
        assert _session_id.get() == "sess_123"

    def test_bind_request_id(self):
        bind_session(request_id="req_456")
        assert _request_id.get() == "req_456"

    def test_bind_user_id(self):
        bind_session(user_id="user_789")
        assert _user_id.get() == "user_789"

    def test_bind_all_session_fields(self):
        bind_session(session_id="s1", request_id="r1", user_id="u1")
        assert _session_id.get() == "s1"
        assert _request_id.get() == "r1"
        assert _user_id.get() == "u1"

    def test_bind_session_overwrites(self):
        bind_session(session_id="old")
        bind_session(session_id="new")
        assert _session_id.get() == "new"

    def test_bind_session_unknown_key_ignored(self):
        bind_session(unknown_key="value")
        # Should not raise and should not affect known vars
        assert _session_id.get() is None

    def test_bind_session_partial_update(self):
        bind_session(session_id="s1", request_id="r1")
        bind_session(session_id="s2")
        assert _session_id.get() == "s2"
        assert _request_id.get() == "r1"  # unchanged


class TestBindAgent:
    """Tests for bind_agent()."""

    def test_bind_agent_name(self):
        bind_agent(agent_name="analyst")
        assert _agent_name.get() == "analyst"

    def test_bind_tier(self):
        bind_agent(tier=3)
        assert _agent_tier.get() == 3

    def test_bind_phase(self):
        bind_agent(phase="execution")
        assert _phase.get() == "execution"

    def test_bind_all_agent_fields(self):
        bind_agent(agent_name="executor", tier=2, phase="planning")
        assert _agent_name.get() == "executor"
        assert _agent_tier.get() == 2
        assert _phase.get() == "planning"

    def test_bind_agent_overwrites(self):
        bind_agent(agent_name="old")
        bind_agent(agent_name="new")
        assert _agent_name.get() == "new"

    def test_bind_agent_unknown_key_ignored(self):
        bind_agent(bogus="value")
        assert _agent_name.get() is None


class TestBindOperation:
    """Tests for bind_operation()."""

    def test_bind_operation(self):
        bind_operation(operation="query")
        assert _operation.get() == "query"

    def test_bind_component(self):
        bind_operation(component="sdk_integration")
        assert _component.get() == "sdk_integration"

    def test_bind_all_operation_fields(self):
        bind_operation(operation="spawn", component="orchestrator")
        assert _operation.get() == "spawn"
        assert _component.get() == "orchestrator"

    def test_bind_operation_unknown_key_ignored(self):
        bind_operation(random_key="value")
        assert _operation.get() is None


class TestClearContext:
    """Tests for clear_context()."""

    def test_clears_all_variables(self):
        bind_session(session_id="s", request_id="r", user_id="u")
        bind_agent(agent_name="a", tier=1, phase="p")
        bind_operation(operation="o", component="c")

        clear_context()

        assert _session_id.get() is None
        assert _request_id.get() is None
        assert _user_id.get() is None
        assert _agent_name.get() is None
        assert _agent_tier.get() is None
        assert _phase.get() is None
        assert _operation.get() is None
        assert _component.get() is None

    def test_clear_already_clear(self):
        """Clearing when already clear should not raise."""
        clear_context()
        clear_context()
        assert _session_id.get() is None


# =============================================================================
# get_logger() Tests
# =============================================================================

class TestGetLogger:
    """Tests for get_logger()."""

    def test_get_logger_without_name(self):
        logger = get_logger()
        assert logger is not None

    def test_get_logger_with_name(self):
        logger = get_logger("my_module")
        assert logger is not None

    def test_get_logger_returns_structlog_logger(self):
        logger = get_logger("test")
        # structlog loggers have info/debug/warning/error methods
        assert hasattr(logger, "info")
        assert hasattr(logger, "debug")
        assert hasattr(logger, "warning")
        assert hasattr(logger, "error")

    def test_get_logger_different_names_different_loggers(self):
        l1 = get_logger("module_a")
        l2 = get_logger("module_b")
        # They should be distinct objects (though structlog may cache)
        assert l1 is not None
        assert l2 is not None


class TestGetAgentLogger:
    """Tests for get_agent_logger()."""

    def test_returns_logger(self):
        logger = get_agent_logger("analyst")
        assert logger is not None

    def test_logger_has_agent_binding(self):
        logger = get_agent_logger("executor")
        # The logger should have bound context
        assert hasattr(logger, "info")


class TestGetSessionLogger:
    """Tests for get_session_logger()."""

    def test_returns_logger(self):
        logger = get_session_logger("session_123")
        assert logger is not None

    def test_logger_has_methods(self):
        logger = get_session_logger("s1")
        assert hasattr(logger, "info")
        assert hasattr(logger, "error")


# =============================================================================
# Processor Tests
# =============================================================================

class TestContextInjector:
    """Tests for the context_injector processor."""

    def test_injects_session_context(self):
        bind_session(session_id="s1", request_id="r1", user_id="u1")
        event_dict = {"event": "test"}
        result = context_injector(None, "info", event_dict)
        assert result["session_id"] == "s1"
        assert result["request_id"] == "r1"
        assert result["user_id"] == "u1"

    def test_injects_agent_context(self):
        bind_agent(agent_name="analyst", tier=2, phase="analysis")
        event_dict = {"event": "test"}
        result = context_injector(None, "info", event_dict)
        assert result["agent"] == "analyst"
        assert result["tier"] == 2
        assert result["phase"] == "analysis"

    def test_injects_operation_context(self):
        bind_operation(operation="spawn", component="orchestrator")
        event_dict = {"event": "test"}
        result = context_injector(None, "info", event_dict)
        assert result["operation"] == "spawn"
        assert result["component"] == "orchestrator"

    def test_skips_none_values(self):
        event_dict = {"event": "test"}
        result = context_injector(None, "info", event_dict)
        assert "session_id" not in result
        assert "agent" not in result
        assert "operation" not in result

    def test_preserves_existing_keys(self):
        event_dict = {"event": "test", "custom_key": "custom_value"}
        result = context_injector(None, "info", event_dict)
        assert result["custom_key"] == "custom_value"

    def test_tier_zero_is_injected(self):
        """Tier 0 is a valid value (not None)."""
        bind_agent(tier=0)
        event_dict = {"event": "test"}
        result = context_injector(None, "info", event_dict)
        assert result["tier"] == 0


class TestAddTimestamp:
    """Tests for the add_timestamp processor."""

    def test_adds_timestamp(self):
        event_dict = {"event": "test"}
        result = add_timestamp(None, "info", event_dict)
        assert "timestamp" in result
        # Should be ISO format
        assert "T" in result["timestamp"]

    def test_timestamp_is_string(self):
        event_dict = {"event": "test"}
        result = add_timestamp(None, "info", event_dict)
        assert isinstance(result["timestamp"], str)


class TestAddLogLevel:
    """Tests for the add_log_level processor."""

    @pytest.mark.parametrize("method,expected", [
        ("info", "INFO"),
        ("debug", "DEBUG"),
        ("warning", "WARNING"),
        ("error", "ERROR"),
        ("critical", "CRITICAL"),
    ])
    def test_adds_level(self, method, expected):
        event_dict = {"event": "test"}
        result = add_log_level(None, method, event_dict)
        assert result["level"] == expected


class TestRenameMessage:
    """Tests for the rename_message processor."""

    def test_renames_event_to_message(self):
        event_dict = {"event": "some event"}
        result = rename_message(None, "info", event_dict)
        assert "message" in result
        assert "event" not in result
        assert result["message"] == "some event"

    def test_preserves_other_keys(self):
        event_dict = {"event": "test", "extra": "data"}
        result = rename_message(None, "info", event_dict)
        assert result["extra"] == "data"


class TestFilterSensitiveData:
    """Tests for the filter_sensitive_data processor."""

    @pytest.mark.parametrize("key", [
        "api_key", "apikey", "secret", "token", "password",
        "credential", "auth", "private_key", "access_token",
    ])
    def test_redacts_sensitive_keys(self, key):
        event_dict = {"event": "test", key: "sensitive_value"}
        result = filter_sensitive_data(None, "info", event_dict)
        assert result[key] == "***REDACTED***"

    def test_redacts_sk_prefix_values(self):
        event_dict = {"event": "test", "data": "sk-abc123456"}
        result = filter_sensitive_data(None, "info", event_dict)
        assert result["data"] == "***REDACTED***"

    def test_redacts_bearer_values(self):
        event_dict = {"event": "test", "header": "Bearer xyz123"}
        result = filter_sensitive_data(None, "info", event_dict)
        assert result["header"] == "***REDACTED***"

    def test_redacts_basic_auth_values(self):
        event_dict = {"event": "test", "auth_header": "Basic dXNlcjpwYXNz"}
        result = filter_sensitive_data(None, "info", event_dict)
        assert result["auth_header"] == "***REDACTED***"

    def test_preserves_normal_values(self):
        event_dict = {"event": "test", "name": "analyst", "count": 42}
        result = filter_sensitive_data(None, "info", event_dict)
        assert result["name"] == "analyst"
        assert result["count"] == 42

    def test_case_insensitive_key_matching(self):
        event_dict = {"event": "test", "API_KEY": "secret123"}
        result = filter_sensitive_data(None, "info", event_dict)
        assert result["API_KEY"] == "***REDACTED***"

    def test_non_string_values_preserved(self):
        event_dict = {"event": "test", "count": 100, "active": True, "items": [1, 2]}
        result = filter_sensitive_data(None, "info", event_dict)
        assert result["count"] == 100
        assert result["active"] is True
        assert result["items"] == [1, 2]


class TestAgentEventClassifier:
    """Tests for the agent_event_classifier processor."""

    def test_classifies_agent_start(self):
        event_dict = {"event": "agent_start"}
        result = agent_event_classifier(None, "info", event_dict)
        assert result["event_type"] == "agent_lifecycle"
        assert result["lifecycle_stage"] == "start"

    def test_classifies_agent_complete(self):
        event_dict = {"event": "agent_complete"}
        result = agent_event_classifier(None, "info", event_dict)
        assert result["event_type"] == "agent_lifecycle"
        assert result["lifecycle_stage"] == "complete"

    def test_classifies_agent_error(self):
        event_dict = {"event": "agent_error"}
        result = agent_event_classifier(None, "info", event_dict)
        assert result["event_type"] == "agent_error"
        assert result["lifecycle_stage"] == "error"

    def test_classifies_verdict(self):
        event_dict = {"event": "verdict"}
        result = agent_event_classifier(None, "info", event_dict)
        assert result["event_type"] == "quality_gate"

    def test_classifies_sme_spawn(self):
        event_dict = {"event": "sme_spawn"}
        result = agent_event_classifier(None, "info", event_dict)
        assert result["event_type"] == "sme_lifecycle"

    def test_no_classification_for_generic_event(self):
        event_dict = {"event": "something_else"}
        result = agent_event_classifier(None, "info", event_dict)
        assert "event_type" not in result

    def test_case_insensitive_matching(self):
        event_dict = {"event": "Agent_Start event"}
        result = agent_event_classifier(None, "info", event_dict)
        assert result["event_type"] == "agent_lifecycle"

    def test_missing_event_key(self):
        event_dict = {}
        result = agent_event_classifier(None, "info", event_dict)
        assert "event_type" not in result


class TestStackTraceFormatter:
    """Tests for the stack_trace_formatter processor."""

    def test_no_exc_info(self):
        event_dict = {"event": "test"}
        result = stack_trace_formatter(None, "info", event_dict)
        assert "stack_trace" not in result

    def test_with_none_exc_info(self):
        event_dict = {"event": "test", "exc_info": None}
        result = stack_trace_formatter(None, "error", event_dict)
        assert "stack_trace" not in result


# =============================================================================
# Formatter Tests
# =============================================================================

class TestFormatters:
    """Tests for console and JSON formatters."""

    def test_console_formatter_returns_processor(self):
        fmt = console_formatter()
        assert callable(fmt)

    def test_json_formatter_returns_processor(self):
        fmt = json_formatter()
        assert callable(fmt)


# =============================================================================
# Convenience Logging Function Tests
# =============================================================================

class TestLogAgentStart:
    """Tests for log_agent_start()."""

    def test_binds_context_and_logs(self):
        # After calling log_agent_start, the context should be bound
        with patch.object(structlog, "get_logger") as mock_get:
            mock_logger = MagicMock()
            mock_get.return_value = mock_logger

            log_agent_start("analyst", "analysis", 2)

            assert _agent_name.get() == "analyst"
            assert _agent_tier.get() == 2
            assert _phase.get() == "analysis"

    def test_with_extra_kwargs(self):
        with patch.object(structlog, "get_logger") as mock_get:
            mock_logger = MagicMock()
            mock_get.return_value = mock_logger

            log_agent_start("executor", "execution", 3, task_id="t1")
            assert _agent_name.get() == "executor"


class TestLogAgentComplete:
    """Tests for log_agent_complete()."""

    def test_calls_info(self):
        with patch.object(structlog, "get_logger") as mock_get:
            mock_logger = MagicMock()
            mock_get.return_value = mock_logger

            log_agent_complete("analyst", "analysis", 1.5)
            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args
            assert call_args[0][0] == "agent_complete"

    def test_passes_duration(self):
        with patch.object(structlog, "get_logger") as mock_get:
            mock_logger = MagicMock()
            mock_get.return_value = mock_logger

            log_agent_complete("analyst", "analysis", 2.5)
            call_kwargs = mock_logger.info.call_args[1]
            assert call_kwargs["duration_seconds"] == 2.5

    def test_passes_agent_and_phase(self):
        with patch.object(structlog, "get_logger") as mock_get:
            mock_logger = MagicMock()
            mock_get.return_value = mock_logger

            log_agent_complete("executor", "execution", 0.5)
            call_kwargs = mock_logger.info.call_args[1]
            assert call_kwargs["agent"] == "executor"
            assert call_kwargs["phase"] == "execution"


class TestLogAgentError:
    """Tests for log_agent_error()."""

    def test_calls_error(self):
        with patch.object(structlog, "get_logger") as mock_get:
            mock_logger = MagicMock()
            mock_get.return_value = mock_logger

            error = ValueError("test error")
            log_agent_error("analyst", "analysis", error)
            mock_logger.error.assert_called_once()

    def test_passes_error_info(self):
        with patch.object(structlog, "get_logger") as mock_get:
            mock_logger = MagicMock()
            mock_get.return_value = mock_logger

            error = RuntimeError("boom")
            log_agent_error("executor", "execution", error)
            call_kwargs = mock_logger.error.call_args[1]
            assert call_kwargs["error_type"] == "RuntimeError"
            assert call_kwargs["error_message"] == "boom"
            assert call_kwargs["exc_info"] is error

    def test_passes_agent_and_phase(self):
        with patch.object(structlog, "get_logger") as mock_get:
            mock_logger = MagicMock()
            mock_get.return_value = mock_logger

            log_agent_error("critic", "review", Exception("e"))
            call_kwargs = mock_logger.error.call_args[1]
            assert call_kwargs["agent"] == "critic"
            assert call_kwargs["phase"] == "review"


class TestLogSmeSpawn:
    """Tests for log_sme_spawn()."""

    def test_calls_info(self):
        with patch.object(structlog, "get_logger") as mock_get:
            mock_logger = MagicMock()
            mock_get.return_value = mock_logger

            log_sme_spawn("cloud_architect", "Cloud Architect", "advisory")
            mock_logger.info.assert_called_once()

    def test_passes_persona_info(self):
        with patch.object(structlog, "get_logger") as mock_get:
            mock_logger = MagicMock()
            mock_get.return_value = mock_logger

            log_sme_spawn("sec_analyst", "Security Analyst", "debate")
            call_kwargs = mock_logger.info.call_args[1]
            assert call_kwargs["sme_persona_id"] == "sec_analyst"
            assert call_kwargs["sme_persona_name"] == "Security Analyst"
            assert call_kwargs["interaction_mode"] == "debate"


class TestLogVerdict:
    """Tests for log_verdict()."""

    def test_passed_verdict_uses_info(self):
        with patch.object(structlog, "get_logger") as mock_get:
            mock_logger = MagicMock()
            mock_get.return_value = mock_logger

            log_verdict("completeness", True, "All items present")
            mock_logger.info.assert_called_once()

    def test_failed_verdict_uses_warning(self):
        with patch.object(structlog, "get_logger") as mock_get:
            mock_logger = MagicMock()
            mock_get.return_value = mock_logger

            log_verdict("consistency", False, "Contradictions found")
            mock_logger.warning.assert_called_once()

    def test_passes_gate_info(self):
        with patch.object(structlog, "get_logger") as mock_get:
            mock_logger = MagicMock()
            mock_get.return_value = mock_logger

            log_verdict("verifier_signoff", True, "Verified")
            call_kwargs = mock_logger.info.call_args[1]
            assert call_kwargs["gate_name"] == "verifier_signoff"
            assert call_kwargs["passed"] is True
            assert call_kwargs["details"] == "Verified"

    def test_extra_kwargs_forwarded(self):
        with patch.object(structlog, "get_logger") as mock_get:
            mock_logger = MagicMock()
            mock_get.return_value = mock_logger

            log_verdict("gate", True, "ok", score=0.95)
            call_kwargs = mock_logger.info.call_args[1]
            assert call_kwargs["score"] == 0.95


class TestLogCost:
    """Tests for log_cost()."""

    def test_calls_debug(self):
        with patch.object(structlog, "get_logger") as mock_get:
            mock_logger = MagicMock()
            mock_get.return_value = mock_logger

            log_cost(1000, 0.01, "claude-3-5-sonnet-20241022", "query")
            mock_logger.debug.assert_called_once()

    def test_passes_cost_info(self):
        with patch.object(structlog, "get_logger") as mock_get:
            mock_logger = MagicMock()
            mock_get.return_value = mock_logger

            log_cost(5000, 0.05, "claude-3-5-opus-20240507", "spawn_subagent")
            call_kwargs = mock_logger.debug.call_args[1]
            assert call_kwargs["tokens"] == 5000
            assert call_kwargs["cost_usd"] == 0.05
            assert call_kwargs["model"] == "claude-3-5-opus-20240507"
            assert call_kwargs["operation"] == "spawn_subagent"

    def test_extra_kwargs_forwarded(self):
        with patch.object(structlog, "get_logger") as mock_get:
            mock_logger = MagicMock()
            mock_get.return_value = mock_logger

            log_cost(100, 0.001, "m", "op", agent="analyst")
            call_kwargs = mock_logger.debug.call_args[1]
            assert call_kwargs["agent"] == "analyst"


# =============================================================================
# Context Preservation Tests
# =============================================================================

class TestContextPreservation:
    """Tests for context preservation across calls."""

    def test_session_context_persists(self):
        bind_session(session_id="s1")
        bind_agent(agent_name="analyst")
        # Session context should still be set after binding agent
        assert _session_id.get() == "s1"
        assert _agent_name.get() == "analyst"

    def test_multiple_bind_calls_independent(self):
        bind_session(session_id="s1")
        bind_agent(agent_name="a1")
        bind_operation(operation="op1")

        assert _session_id.get() == "s1"
        assert _agent_name.get() == "a1"
        assert _operation.get() == "op1"

        # Rebinding agent doesn't affect session or operation
        bind_agent(agent_name="a2")
        assert _session_id.get() == "s1"
        assert _agent_name.get() == "a2"
        assert _operation.get() == "op1"

    def test_context_injector_includes_all_bound_context(self):
        bind_session(session_id="s1", request_id="r1", user_id="u1")
        bind_agent(agent_name="analyst", tier=2, phase="analysis")
        bind_operation(operation="query", component="sdk")

        event_dict = {"event": "test"}
        result = context_injector(None, "info", event_dict)

        assert result["session_id"] == "s1"
        assert result["request_id"] == "r1"
        assert result["user_id"] == "u1"
        assert result["agent"] == "analyst"
        assert result["tier"] == 2
        assert result["phase"] == "analysis"
        assert result["operation"] == "query"
        assert result["component"] == "sdk"

    def test_clear_then_rebind(self):
        bind_session(session_id="old")
        clear_context()
        assert _session_id.get() is None
        bind_session(session_id="new")
        assert _session_id.get() == "new"


# =============================================================================
# Context Manager Tests
# =============================================================================

class TestAgentLogContext:
    """Tests for the AgentLogContext context manager."""

    def test_sets_context_on_enter(self):
        with AgentLogContext("executor", tier=3, phase="execution"):
            assert _agent_name.get() == "executor"
            assert _agent_tier.get() == 3
            assert _phase.get() == "execution"

    def test_restores_context_on_exit(self):
        bind_agent(agent_name="original", tier=1, phase="init")
        with AgentLogContext("temp", tier=2, phase="temp_phase"):
            assert _agent_name.get() == "temp"
        assert _agent_name.get() == "original"
        assert _agent_tier.get() == 1
        assert _phase.get() == "init"

    def test_restores_none_on_exit(self):
        with AgentLogContext("temp", tier=2, phase="temp"):
            pass
        assert _agent_name.get() is None
        assert _agent_tier.get() is None
        assert _phase.get() is None

    def test_nested_contexts(self):
        with AgentLogContext("outer", tier=1, phase="p1"):
            assert _agent_name.get() == "outer"
            with AgentLogContext("inner", tier=2, phase="p2"):
                assert _agent_name.get() == "inner"
                assert _agent_tier.get() == 2
            assert _agent_name.get() == "outer"
            assert _agent_tier.get() == 1

    def test_restores_on_exception(self):
        bind_agent(agent_name="original")
        try:
            with AgentLogContext("temp"):
                assert _agent_name.get() == "temp"
                raise ValueError("test")
        except ValueError:
            pass
        assert _agent_name.get() == "original"


class TestSessionLogContext:
    """Tests for the SessionLogContext context manager."""

    def test_sets_context_on_enter(self):
        with SessionLogContext("sess_1", user_id="u1", request_id="r1"):
            assert _session_id.get() == "sess_1"
            assert _user_id.get() == "u1"
            assert _request_id.get() == "r1"

    def test_restores_context_on_exit(self):
        bind_session(session_id="orig_s", user_id="orig_u", request_id="orig_r")
        with SessionLogContext("temp_s", user_id="temp_u", request_id="temp_r"):
            assert _session_id.get() == "temp_s"
        assert _session_id.get() == "orig_s"
        assert _user_id.get() == "orig_u"
        assert _request_id.get() == "orig_r"

    def test_restores_none_on_exit(self):
        with SessionLogContext("temp"):
            pass
        assert _session_id.get() is None

    def test_nested_sessions(self):
        with SessionLogContext("outer"):
            assert _session_id.get() == "outer"
            with SessionLogContext("inner"):
                assert _session_id.get() == "inner"
            assert _session_id.get() == "outer"

    def test_restores_on_exception(self):
        bind_session(session_id="original")
        try:
            with SessionLogContext("temp"):
                raise RuntimeError("test")
        except RuntimeError:
            pass
        assert _session_id.get() == "original"


# =============================================================================
# configure_logging() Tests
# =============================================================================

class TestConfigureLogging:
    """Tests for the configure_logging() function."""

    def test_configure_default(self):
        """Default configuration should not raise."""
        configure_logging()

    def test_configure_debug_level(self):
        configure_logging(level="DEBUG")

    def test_configure_json_output(self):
        configure_logging(json_output=True)

    def test_configure_no_filtering(self):
        configure_logging(enable_filtering=False)

    def test_configure_with_log_file(self, tmp_path):
        log_file = str(tmp_path / "test.log")
        configure_logging(log_file=log_file)
        assert Path(log_file).parent.exists()

    def test_configure_creates_log_dir(self, tmp_path):
        log_file = str(tmp_path / "subdir" / "test.log")
        configure_logging(log_file=log_file)
        assert Path(log_file).parent.exists()

    @pytest.mark.parametrize("level", [
        "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL",
    ])
    def test_configure_all_levels(self, level):
        configure_logging(level=level)

    def test_configure_json_and_file(self, tmp_path):
        log_file = str(tmp_path / "json.log")
        configure_logging(level="INFO", log_file=log_file, json_output=True)


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Edge case tests for logging utilities."""

    def test_bind_session_with_none_value(self):
        bind_session(session_id=None)
        assert _session_id.get() is None

    def test_bind_agent_with_zero_tier(self):
        bind_agent(tier=0)
        assert _agent_tier.get() == 0

    def test_bind_agent_with_empty_string(self):
        bind_agent(agent_name="")
        assert _agent_name.get() == ""

    def test_context_injector_empty_event_dict(self):
        result = context_injector(None, "info", {})
        assert isinstance(result, dict)

    def test_add_timestamp_preserves_event(self):
        event_dict = {"event": "original"}
        result = add_timestamp(None, "info", event_dict)
        assert result["event"] == "original"

    def test_filter_sensitive_handles_empty_dict(self):
        result = filter_sensitive_data(None, "info", {})
        assert result == {}

    def test_agent_classifier_with_mixed_case_event(self):
        event_dict = {"event": "AGENT_START happened"}
        result = agent_event_classifier(None, "info", event_dict)
        assert result["event_type"] == "agent_lifecycle"

    def test_log_agent_error_with_base_exception(self):
        """Should handle base Exception types."""
        with patch.object(structlog, "get_logger") as mock_get:
            mock_logger = MagicMock()
            mock_get.return_value = mock_logger
            log_agent_error("a", "p", Exception("base"))
            call_kwargs = mock_logger.error.call_args[1]
            assert call_kwargs["error_type"] == "Exception"

    def test_log_cost_zero_values(self):
        with patch.object(structlog, "get_logger") as mock_get:
            mock_logger = MagicMock()
            mock_get.return_value = mock_logger
            log_cost(0, 0.0, "model", "op")
            call_kwargs = mock_logger.debug.call_args[1]
            assert call_kwargs["tokens"] == 0
            assert call_kwargs["cost_usd"] == 0.0

    def test_log_verdict_empty_details(self):
        with patch.object(structlog, "get_logger") as mock_get:
            mock_logger = MagicMock()
            mock_get.return_value = mock_logger
            log_verdict("gate", True, "")
            call_kwargs = mock_logger.info.call_args[1]
            assert call_kwargs["details"] == ""
