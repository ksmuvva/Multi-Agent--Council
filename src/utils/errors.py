"""
Error Handling - Retry Logic and Error Recovery

Provides retry strategies, degradation handlers, and error recovery
mechanisms for robust agent operations.
"""

import time
import functools
from typing import (
    Type,
    Optional,
    Callable,
    Any,
    TypeVar,
    ParamSpec,
    Dict,
    List,
)
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta

from .logging import get_logger, log_agent_error


# =============================================================================
# Error Types
# =============================================================================

class AgentError(Exception):
    """Base exception for agent-related errors."""

    def __init__(
        self,
        message: str,
        agent_name: str = None,
        phase: str = None,
        cause: Exception = None,
    ):
        self.agent_name = agent_name
        self.phase = phase
        self.cause = cause
        super().__init__(message)


class AgentTimeoutError(AgentError):
    """Raised when an agent operation times out."""
    pass


class AgentRateLimitError(AgentError):
    """Raised when API rate limit is hit."""
    def __init__(self, message: str, retry_after: float = None, **kwargs):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class AgentTokenLimitError(AgentError):
    """Raised when token limit is exceeded."""
    def __init__(self, message: str, tokens_used: int = None, tokens_limit: int = None, **kwargs):
        super().__init__(message, **kwargs)
        self.tokens_used = tokens_used
        self.tokens_limit = tokens_limit


class AgentValidationError(AgentError):
    """Raised when agent output validation fails."""
    def __init__(self, message: str, errors: List[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.errors = errors or []


class AgentContextError(AgentError):
    """Raised when agent context is invalid."""
    pass


class AgentDegradedError(AgentError):
    """Raised when agent is operating in degraded mode."""
    pass


# =============================================================================
# Retry Strategy
# =============================================================================

class RetryStrategy(str, Enum):
    """Retry strategies for handling failures."""
    NONE = "none"  # No retry
    IMMEDIATE = "immediate"  # Retry immediately
    LINEAR_BACKOFF = "linear_backoff"  # Linear backoff (1s, 2s, 3s, ...)
    EXPONENTIAL_BACKOFF = "exponential_backoff"  # Exponential backoff (1s, 2s, 4s, 8s, ...)
    FIBONACCI_BACKOFF = "fibonacci_backoff"  # Fibonacci backoff (1s, 1s, 2s, 3s, 5s, ...)
    JITTER = "jitter"  # Random jitter between retries


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    max_attempts: int = 3
    base_delay: float = 1.0  # Base delay in seconds
    max_delay: float = 60.0  # Maximum delay between retries
    jitter_factor: float = 0.1  # Random jitter factor (0.1 = ±10%)
    retry_on: tuple = (Exception,)  # Exceptions to retry on
    stop_on: tuple = ()  # Exceptions that should stop retries

    def get_delay(self, attempt: int) -> float:
        """
        Calculate delay for a given retry attempt.

        Args:
            attempt: Attempt number (1-indexed)

        Returns:
            Delay in seconds
        """
        if attempt <= 1:
            return 0

        delay = self.base_delay

        if self.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.base_delay * (attempt - 1)

        elif self.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.base_delay * (2 ** (attempt - 2))

        elif self.strategy == RetryStrategy.FIBONACCI_BACKOFF:
            fib = [1, 1]
            for _ in range(2, attempt):
                fib.append(fib[-1] + fib[-2])
            delay = self.base_delay * (fib[-1] if len(fib) > 0 else 1)

        elif self.strategy == RetryStrategy.JITTER:
            import random
            delay = self.base_delay * random.uniform(1 - self.jitter_factor, 1 + self.jitter_factor)

        # Cap at max_delay
        return min(delay, self.max_delay)


# =============================================================================
# Degradation Strategies
# =============================================================================

class DegradationLevel(str, Enum):
    """Levels of system degradation."""
    NONE = "none"  # Full functionality
    MILD = "mild"  # Some features reduced
    MODERATE = "moderate"  # Significant reduction
    SEVERE = "severe"  # Minimal functionality
    CRITICAL = "critical"  # Emergency mode only


@dataclass
class DegradationAction:
    """An action to take during degradation."""
    level: DegradationLevel
    description: str
    handler: Callable[[], Any]
    enabled: bool = True


# =============================================================================
# Retry Decorator
# =============================================================================

P = ParamSpec("P")
T = TypeVar("T")


def retry_on_failure(
    config: Optional[RetryConfig] = None,
    fallback: Optional[Callable] = None,
    on_failure: Optional[Callable[[Exception], Any]] = None,
) -> Callable:
    """
    Decorator to retry function on failure.

    Args:
        config: Retry configuration
        fallback: Fallback function to call if all retries fail
        on_failure: Callback called on final failure

    Returns:
        Decorated function
    """
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        logger = get_logger(func.__name__)

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_exception = None

            for attempt in range(1, config.max_attempts + 1):
                try:
                    if attempt > 1:
                        delay = config.get_delay(attempt)
                        if delay > 0:
                            logger.debug(
                                "retry_attempt",
                                function=func.__name__,
                                attempt=attempt,
                                delay=delay,
                            )
                            time.sleep(delay)

                    return func(*args, **kwargs)

                except config.stop_on as e:
                    # Don't retry on these exceptions
                    logger.error(
                        "error_stop_retry",
                        function=func.__name__,
                        error_type=type(e).__name__,
                        error_message=str(e),
                    )
                    raise

                except config.retry_on as e:
                    last_exception = e

                    if attempt < config.max_attempts:
                        logger.warning(
                            "error_will_retry",
                            function=func.__name__,
                            attempt=attempt,
                            max_attempts=config.max_attempts,
                            error_type=type(e).__name__,
                            error_message=str(e),
                        )
                    else:
                        logger.error(
                            "error_all_retries_failed",
                            function=func.__name__,
                            attempts=attempt,
                            error_type=type(e).__name__,
                            error_message=str(e),
                        )

            # All retries exhausted
            if on_failure:
                return on_failure(last_exception)

            if fallback:
                logger.info("using_fallback", function=func.__name__)
                return fallback(*args, **kwargs)

            # Re-raise the last exception
            raise last_exception

        return wrapper
    return decorator


# =============================================================================
# Circuit Breaker
# =============================================================================

class CircuitBreakerState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Circuit is open, requests fail fast
    HALF_OPEN = "half_open"  # Testing if service has recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 2  # Successes to close circuit
    timeout: float = 60.0  # Seconds to wait before trying again


class CircuitBreaker:
    """
    Circuit breaker pattern implementation.

    Prevents cascading failures by stopping requests to a failing service.
    """

    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig = None,
    ):
        """
        Initialize circuit breaker.

        Args:
            name: Name of the circuit breaker
            config: Configuration
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.opened_at = None
        self._logger = get_logger("circuit_breaker")

    def is_open(self) -> bool:
        """Check if circuit is open."""
        if self.state == CircuitBreakerState.OPEN:
            # Check if timeout has elapsed
            if self.opened_at and (datetime.now() - self.opened_at).total_seconds() > self.config.timeout:
                self._logger.info("circuit_half_open", breaker=self.name)
                self.state = CircuitBreakerState.HALF_OPEN
                self.success_count = 0
                return False
            return True
        return False

    def record_success(self) -> None:
        """Record a successful operation."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1

            if self.success_count >= self.config.success_threshold:
                self._logger.info("circuit_closed", breaker=self.name)
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0

        elif self.state == CircuitBreakerState.CLOSED:
            # Reset failure count on success in closed state
            self.failure_count = max(0, self.failure_count - 1)

    def record_failure(self) -> None:
        """Record a failed operation."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.failure_count >= self.config.failure_threshold:
            if self.state != CircuitBreakerState.OPEN:
                self._logger.warning(
                    "circuit_opened",
                    breaker=self.name,
                    failure_count=self.failure_count,
                    threshold=self.config.failure_threshold,
                )
                self.state = CircuitBreakerState.OPEN
                self.opened_at = datetime.now()

    def __call__(self, func: Callable[P, T]) -> Callable[P, T]:
        """Use as a decorator."""

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            if self.is_open():
                raise AgentDegradedError(
                    f"Circuit breaker '{self.name}' is open",
                    agent_name=self.name,
                )

            try:
                result = func(*args, **kwargs)
                self.record_success()
                return result

            except Exception as e:
                self.record_failure()
                raise

        return wrapper


# =============================================================================
# Degradation Manager
# =============================================================================

class DegradationManager:
    """
    Manages system degradation levels and actions.

    Coordinates responses to system stress or failures.
    """

    def __init__(self):
        self._level = DegradationLevel.NONE
        self._actions: List[DegradationAction] = []
        self._logger = get_logger("degradation_manager")

    @property
    def level(self) -> DegradationLevel:
        """Current degradation level."""
        return self._level

    def set_level(self, level: DegradationLevel, reason: str = "") -> None:
        """
        Set the degradation level.

        Args:
            level: New degradation level
            reason: Reason for the change
        """
        if level != self._level:
            self._logger.warning(
                "degradation_level_changed",
                from_level=self._level,
                to_level=level,
                reason=reason,
            )
            self._level = level
            self._execute_actions(level)

    def register_action(self, action: DegradationAction) -> None:
        """Register a degradation action."""
        self._actions.append(action)

    def _execute_actions(self, level: DegradationLevel) -> None:
        """Execute actions for a given level."""
        level_order = [
            DegradationLevel.CRITICAL,
            DegradationLevel.SEVERE,
            DegradationLevel.MODERATE,
            DegradationLevel.MILD,
        ]

        # Execute actions at or above the current level
        for action_level in level_order:
            if level_order.index(action_level) >= level_order.index(level):
                for action in self._actions:
                    if action.level == action_level and action.enabled:
                        try:
                            action.handler()
                        except Exception as e:
                            self._logger.error(
                                "degradation_action_failed",
                                action=action.description,
                                error=str(e),
                            )

    def is_degraded(self) -> bool:
        """Check if system is currently degraded."""
        return self._level != DegradationLevel.NONE


# =============================================================================
# Global Instances
# =============================================================================

# Global circuit breakers for common services
_circuit_breakers: Dict[str, CircuitBreaker] = {}


def get_circuit_breaker(
    name: str,
    config: CircuitBreakerConfig = None,
) -> CircuitBreaker:
    """
    Get or create a circuit breaker.

    Args:
        name: Name of the circuit breaker
        config: Optional configuration

    Returns:
        CircuitBreaker instance
    """
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(name, config)
    return _circuit_breakers[name]


# Global degradation manager
_degradation_manager = DegradationManager()


def get_degradation_manager() -> DegradationManager:
    """Get the global degradation manager."""
    return _degradation_manager


# =============================================================================
# Convenience Functions
# =============================================================================

def with_retry(
    max_attempts: int = 3,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF,
    base_delay: float = 1.0,
):
    """
    Convenience decorator for retry logic.

    Args:
        max_attempts: Maximum number of attempts
        strategy: Retry strategy
        base_delay: Base delay between retries
    """
    config = RetryConfig(
        strategy=strategy,
        max_attempts=max_attempts,
        base_delay=base_delay,
    )
    return retry_on_failure(config)


def with_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    timeout: float = 60.0,
):
    """
    Convenience decorator for circuit breaker.

    Args:
        name: Name of the circuit breaker
        failure_threshold: Failures before opening
        timeout: Seconds to wait before retry
    """
    config = CircuitBreakerConfig(
        failure_threshold=failure_threshold,
        timeout=timeout,
    )
    breaker = get_circuit_breaker(name, config)
    return breaker


# =============================================================================
# Error Recovery Handlers
# =============================================================================

def handle_rate_limit_error(
    error: AgentRateLimitError,
) -> Dict[str, Any]:
    """
    Handle rate limit errors with appropriate backoff.

    Args:
        error: The rate limit error

    Returns:
        Recovery information
    """
    retry_after = error.retry_after or 5.0

    return {
        "action": "retry_after",
        "delay": retry_after,
        "message": f"Rate limited. Retrying after {retry_after} seconds.",
    }


def handle_token_limit_error(
    error: AgentTokenLimitError,
) -> Dict[str, Any]:
    """
    Handle token limit errors with context reduction.

    Args:
        error: The token limit error

    Returns:
        Recovery information
    """
    # Calculate how much to reduce context
    if error.tokens_used and error.tokens_limit:
        reduction = (error.tokens_used - error.tokens_limit) / error.tokens_used
    else:
        reduction = 0.5  # Default to 50% reduction

    return {
        "action": "reduce_context",
        "reduction_factor": reduction,
        "message": f"Token limit exceeded. Reducing context by {int(reduction * 100)}%.",
    }


def handle_validation_error(
    error: AgentValidationError,
) -> Dict[str, Any]:
    """
    Handle validation errors with detailed feedback.

    Args:
        error: The validation error

    Returns:
        Recovery information
    """
    return {
        "action": "regenerate",
        "errors": error.errors,
        "message": f"Validation failed with {len(error.errors)} errors.",
    }
