"""
Exhaustive Tests for Error Handling Module

Tests error types, retry strategies, circuit breaker,
degradation manager, and error recovery handlers.
"""

import sys
import time
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from src.utils.errors import (
    AgentError,
    AgentTimeoutError,
    AgentRateLimitError,
    AgentTokenLimitError,
    AgentValidationError,
    AgentContextError,
    AgentDegradedError,
    RetryStrategy,
    RetryConfig,
    DegradationLevel,
    DegradationAction,
    CircuitBreakerState,
    CircuitBreakerConfig,
    CircuitBreaker,
    DegradationManager,
    retry_on_failure,
    with_retry,
    with_circuit_breaker,
    get_circuit_breaker,
    get_degradation_manager,
    handle_rate_limit_error,
    handle_token_limit_error,
    handle_validation_error,
)


# =============================================================================
# Error Type Tests
# =============================================================================

class TestAgentError:
    def test_basic_error(self):
        err = AgentError("Something went wrong")
        assert str(err) == "Something went wrong"
        assert err.agent_name is None
        assert err.phase is None
        assert err.cause is None

    def test_error_with_details(self):
        cause = ValueError("inner")
        err = AgentError("Failed", agent_name="Executor", phase="Phase 5", cause=cause)
        assert err.agent_name == "Executor"
        assert err.phase == "Phase 5"
        assert err.cause is cause

    def test_is_exception(self):
        assert issubclass(AgentError, Exception)


class TestAgentTimeoutError:
    def test_inherits_agent_error(self):
        assert issubclass(AgentTimeoutError, AgentError)

    def test_create(self):
        err = AgentTimeoutError("Timeout", agent_name="Verifier")
        assert err.agent_name == "Verifier"


class TestAgentRateLimitError:
    def test_retry_after(self):
        err = AgentRateLimitError("Rate limited", retry_after=5.0)
        assert err.retry_after == 5.0

    def test_default_retry_after(self):
        err = AgentRateLimitError("Rate limited")
        assert err.retry_after is None


class TestAgentTokenLimitError:
    def test_token_info(self):
        err = AgentTokenLimitError("Tokens exceeded", tokens_used=150000, tokens_limit=100000)
        assert err.tokens_used == 150000
        assert err.tokens_limit == 100000

    def test_defaults(self):
        err = AgentTokenLimitError("Tokens exceeded")
        assert err.tokens_used is None
        assert err.tokens_limit is None


class TestAgentValidationError:
    def test_with_errors(self):
        err = AgentValidationError("Validation failed", errors=["err1", "err2"])
        assert len(err.errors) == 2

    def test_default_errors(self):
        err = AgentValidationError("Validation failed")
        assert err.errors == []


class TestAgentContextError:
    def test_inherits(self):
        assert issubclass(AgentContextError, AgentError)


class TestAgentDegradedError:
    def test_inherits(self):
        assert issubclass(AgentDegradedError, AgentError)


# =============================================================================
# RetryStrategy Tests
# =============================================================================

class TestRetryStrategy:
    def test_all_strategies(self):
        assert RetryStrategy.NONE == "none"
        assert RetryStrategy.IMMEDIATE == "immediate"
        assert RetryStrategy.LINEAR_BACKOFF == "linear_backoff"
        assert RetryStrategy.EXPONENTIAL_BACKOFF == "exponential_backoff"
        assert RetryStrategy.FIBONACCI_BACKOFF == "fibonacci_backoff"
        assert RetryStrategy.JITTER == "jitter"

    def test_count(self):
        assert len(RetryStrategy) == 6


# =============================================================================
# RetryConfig Tests
# =============================================================================

class TestRetryConfig:
    def test_defaults(self):
        config = RetryConfig()
        assert config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF
        assert config.max_attempts == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0

    def test_get_delay_attempt_1(self):
        config = RetryConfig()
        assert config.get_delay(1) == 0

    def test_linear_backoff(self):
        config = RetryConfig(strategy=RetryStrategy.LINEAR_BACKOFF, base_delay=2.0)
        assert config.get_delay(2) == 2.0
        assert config.get_delay(3) == 4.0
        assert config.get_delay(4) == 6.0

    def test_exponential_backoff(self):
        config = RetryConfig(strategy=RetryStrategy.EXPONENTIAL_BACKOFF, base_delay=1.0)
        assert config.get_delay(2) == 1.0
        assert config.get_delay(3) == 2.0
        assert config.get_delay(4) == 4.0

    def test_fibonacci_backoff(self):
        config = RetryConfig(strategy=RetryStrategy.FIBONACCI_BACKOFF, base_delay=1.0)
        d2 = config.get_delay(2)
        d3 = config.get_delay(3)
        assert d2 >= 0
        assert d3 >= 0

    def test_jitter(self):
        config = RetryConfig(strategy=RetryStrategy.JITTER, base_delay=1.0, jitter_factor=0.1)
        delay = config.get_delay(2)
        assert 0.9 <= delay <= 1.1

    def test_max_delay_cap(self):
        config = RetryConfig(
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF, base_delay=10.0, max_delay=20.0
        )
        delay = config.get_delay(10)
        assert delay <= 20.0

    def test_immediate_strategy(self):
        config = RetryConfig(strategy=RetryStrategy.IMMEDIATE)
        assert config.get_delay(2) == 1.0  # base_delay

    def test_none_strategy(self):
        config = RetryConfig(strategy=RetryStrategy.NONE)
        assert config.get_delay(2) == 1.0  # fallback to base_delay


# =============================================================================
# DegradationLevel Tests
# =============================================================================

class TestDegradationLevel:
    def test_values(self):
        assert DegradationLevel.NONE == "none"
        assert DegradationLevel.MILD == "mild"
        assert DegradationLevel.MODERATE == "moderate"
        assert DegradationLevel.SEVERE == "severe"
        assert DegradationLevel.CRITICAL == "critical"

    def test_count(self):
        assert len(DegradationLevel) == 5


# =============================================================================
# CircuitBreaker Tests
# =============================================================================

class TestCircuitBreakerState:
    def test_values(self):
        assert CircuitBreakerState.CLOSED == "closed"
        assert CircuitBreakerState.OPEN == "open"
        assert CircuitBreakerState.HALF_OPEN == "half_open"


class TestCircuitBreakerConfig:
    def test_defaults(self):
        config = CircuitBreakerConfig()
        assert config.failure_threshold == 5
        assert config.success_threshold == 2
        assert config.timeout == 60.0


class TestCircuitBreaker:
    def test_initial_state(self):
        cb = CircuitBreaker("test")
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.failure_count == 0
        assert cb.is_open() is False

    def test_record_success_closed(self):
        cb = CircuitBreaker("test")
        cb.failure_count = 2
        cb.record_success()
        assert cb.failure_count == 1  # decremented

    def test_record_failure_opens_circuit(self):
        cb = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=3))
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN

    def test_is_open_after_threshold(self):
        cb = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=2))
        cb.record_failure()
        cb.record_failure()
        assert cb.is_open() is True

    def test_half_open_after_timeout(self):
        cb = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=2, timeout=0.01))
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN
        cb.opened_at = datetime.now() - timedelta(seconds=1)
        assert cb.is_open() is False
        assert cb.state == CircuitBreakerState.HALF_OPEN

    def test_success_in_half_open_closes(self):
        cb = CircuitBreaker("test", CircuitBreakerConfig(
            failure_threshold=2, success_threshold=1, timeout=0.01
        ))
        cb.record_failure()
        cb.record_failure()
        cb.opened_at = datetime.now() - timedelta(seconds=1)
        cb.is_open()  # transitions to HALF_OPEN
        cb.record_success()
        assert cb.state == CircuitBreakerState.CLOSED

    def test_as_decorator(self):
        cb = CircuitBreaker("test_dec")

        @cb
        def my_func():
            return "success"

        assert my_func() == "success"

    def test_decorator_records_failure(self):
        cb = CircuitBreaker("test_dec_fail", CircuitBreakerConfig(failure_threshold=3))

        @cb
        def failing_func():
            raise ValueError("fail")

        with pytest.raises(ValueError):
            failing_func()
        assert cb.failure_count == 1

    def test_decorator_blocks_when_open(self):
        cb = CircuitBreaker("test_block", CircuitBreakerConfig(failure_threshold=1))
        cb.record_failure()

        @cb
        def blocked_func():
            return "should not run"

        with pytest.raises(AgentDegradedError):
            blocked_func()


# =============================================================================
# DegradationManager Tests
# =============================================================================

class TestDegradationManager:
    def test_initial_level(self):
        dm = DegradationManager()
        assert dm.level == DegradationLevel.NONE
        assert dm.is_degraded() is False

    def test_set_level(self):
        dm = DegradationManager()
        dm.set_level(DegradationLevel.MILD, "test")
        assert dm.level == DegradationLevel.MILD
        assert dm.is_degraded() is True

    def test_set_same_level_no_change(self):
        dm = DegradationManager()
        dm.set_level(DegradationLevel.MILD, "first")
        dm.set_level(DegradationLevel.MILD, "second")  # no-op
        assert dm.level == DegradationLevel.MILD

    def test_register_action(self):
        dm = DegradationManager()
        handler = MagicMock()
        action = DegradationAction(
            level=DegradationLevel.SEVERE,
            description="Test action",
            handler=handler,
        )
        dm.register_action(action)
        dm.set_level(DegradationLevel.SEVERE, "test")
        handler.assert_called()

    def test_action_exception_handled(self):
        dm = DegradationManager()
        handler = MagicMock(side_effect=Exception("handler error"))
        action = DegradationAction(
            level=DegradationLevel.CRITICAL,
            description="Failing action",
            handler=handler,
        )
        dm.register_action(action)
        dm.set_level(DegradationLevel.CRITICAL, "test")
        # Should not raise


# =============================================================================
# retry_on_failure Decorator Tests
# =============================================================================

class TestRetryOnFailure:
    def test_success_no_retry(self):
        call_count = 0

        @retry_on_failure(RetryConfig(max_attempts=3, base_delay=0.01))
        def success_func():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = success_func()
        assert result == "ok"
        assert call_count == 1

    def test_retry_then_success(self):
        call_count = 0

        @retry_on_failure(RetryConfig(max_attempts=3, base_delay=0.01))
        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("fail")
            return "ok"

        result = flaky_func()
        assert result == "ok"
        assert call_count == 3

    def test_all_retries_fail(self):
        @retry_on_failure(RetryConfig(max_attempts=2, base_delay=0.01))
        def always_fail():
            raise ValueError("always fails")

        with pytest.raises(ValueError):
            always_fail()

    def test_fallback_function(self):
        def fallback(*args, **kwargs):
            return "fallback_value"

        @retry_on_failure(RetryConfig(max_attempts=1, base_delay=0.01), fallback=fallback)
        def failing_func():
            raise Exception("fail")

        result = failing_func()
        assert result == "fallback_value"

    def test_on_failure_callback(self):
        captured = {}

        def on_fail(exc):
            captured["error"] = str(exc)
            return "handled"

        @retry_on_failure(RetryConfig(max_attempts=1, base_delay=0.01), on_failure=on_fail)
        def failing_func():
            raise ValueError("test error")

        result = failing_func()
        assert result == "handled"
        assert "test error" in captured["error"]


# =============================================================================
# Error Recovery Handler Tests
# =============================================================================

class TestHandleRateLimitError:
    def test_with_retry_after(self):
        err = AgentRateLimitError("limited", retry_after=10.0)
        result = handle_rate_limit_error(err)
        assert result["action"] == "retry_after"
        assert result["delay"] == 10.0

    def test_default_retry_after(self):
        err = AgentRateLimitError("limited")
        result = handle_rate_limit_error(err)
        assert result["delay"] == 5.0


class TestHandleTokenLimitError:
    def test_with_token_info(self):
        err = AgentTokenLimitError("exceeded", tokens_used=150000, tokens_limit=100000)
        result = handle_token_limit_error(err)
        assert result["action"] == "reduce_context"
        assert result["reduction_factor"] > 0

    def test_without_token_info(self):
        err = AgentTokenLimitError("exceeded")
        result = handle_token_limit_error(err)
        assert result["reduction_factor"] == 0.5


class TestHandleValidationError:
    def test_with_errors(self):
        err = AgentValidationError("failed", errors=["e1", "e2"])
        result = handle_validation_error(err)
        assert result["action"] == "regenerate"
        assert len(result["errors"]) == 2
        assert "2 errors" in result["message"]

    def test_no_errors(self):
        err = AgentValidationError("failed")
        result = handle_validation_error(err)
        assert result["errors"] == []


# =============================================================================
# Global Instance Tests
# =============================================================================

class TestGlobalInstances:
    def test_get_circuit_breaker(self):
        cb = get_circuit_breaker("test_global")
        assert isinstance(cb, CircuitBreaker)
        cb2 = get_circuit_breaker("test_global")
        assert cb is cb2

    def test_get_degradation_manager(self):
        dm = get_degradation_manager()
        assert isinstance(dm, DegradationManager)
