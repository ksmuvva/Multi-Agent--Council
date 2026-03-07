"""
Utility Functions and Helpers

This package contains utility modules for the multi-agent system.
"""

from .events import (
    EventType,
    Event,
    EventSubscription,
    EventEmitter,
    EventStreamer,
    # Global instances
    get_event_emitter,
    get_event_streamer,
    # Convenience functions
    emit_task_started,
    emit_task_progress,
    emit_task_completed,
    emit_agent_started,
    emit_agent_completed,
    emit_finding,
    emit_error,
    emit_quality_gate,
    emit_system_message,
    format_sse_event,
)

# Logging
from .logging import (
    # Context variables
    bind_session,
    bind_agent,
    bind_operation,
    clear_context,
    # Loggers
    get_logger,
    get_agent_logger,
    get_session_logger,
    # Configuration
    configure_logging,
    # Convenience functions
    log_agent_start,
    log_agent_complete,
    log_agent_error,
    log_sme_spawn,
    log_verdict,
    log_cost,
    # Context managers
    AgentLogContext,
    SessionLogContext,
)

# Cost tracking
from .cost import (
    # Pricing
    ModelPricing,
    MODEL_COSTS,
    AGENT_TOKEN_COSTS,
    # Data structures
    TokenUsage,
    OperationCost,
    BudgetState,
    SessionCosts,
    # Exceptions
    BudgetExceededError,
    BudgetWarning,
    # Cost tracker
    CostTracker,
    get_cost_tracker,
    # Context managers
    CostLimit,
    # Utilities
    calculate_tokens_from_text,
    calculate_max_turns_for_budget,
)

# Error handling
from .errors import (
    # Exceptions
    AgentError,
    AgentTimeoutError,
    AgentRateLimitError,
    AgentTokenLimitError,
    AgentValidationError,
    AgentContextError,
    AgentDegradedError,
    # Retry
    RetryStrategy,
    RetryConfig,
    retry_on_failure,
    with_retry,
    # Circuit breaker
    CircuitBreakerState,
    CircuitBreakerConfig,
    CircuitBreaker,
    get_circuit_breaker,
    with_circuit_breaker,
    # Degradation
    DegradationLevel,
    DegradationAction,
    DegradationManager,
    get_degradation_manager,
    # Error handlers
    handle_rate_limit_error,
    handle_token_limit_error,
    handle_validation_error,
)

__all__ = [
    # Event classes
    "EventType",
    "Event",
    "EventSubscription",
    "EventEmitter",
    "EventStreamer",
    # Global instances (events)
    "get_event_emitter",
    "get_event_streamer",
    # Emit functions
    "emit_task_started",
    "emit_task_progress",
    "emit_task_completed",
    "emit_agent_started",
    "emit_agent_completed",
    "emit_finding",
    "emit_error",
    "emit_quality_gate",
    "emit_system_message",
    # Utility (events)
    "format_sse_event",

    # Logging
    "bind_session",
    "bind_agent",
    "bind_operation",
    "clear_context",
    "get_logger",
    "get_agent_logger",
    "get_session_logger",
    "configure_logging",
    "log_agent_start",
    "log_agent_complete",
    "log_agent_error",
    "log_sme_spawn",
    "log_verdict",
    "log_cost",
    "AgentLogContext",
    "SessionLogContext",

    # Cost tracking
    "ModelPricing",
    "MODEL_COSTS",
    "AGENT_TOKEN_COSTS",
    "TokenUsage",
    "OperationCost",
    "BudgetState",
    "SessionCosts",
    "BudgetExceededError",
    "BudgetWarning",
    "CostTracker",
    "get_cost_tracker",
    "CostLimit",
    "calculate_tokens_from_text",
    "calculate_max_turns_for_budget",

    # Error handling
    "AgentError",
    "AgentTimeoutError",
    "AgentRateLimitError",
    "AgentTokenLimitError",
    "AgentValidationError",
    "AgentContextError",
    "AgentDegradedError",
    "RetryStrategy",
    "RetryConfig",
    "retry_on_failure",
    "with_retry",
    "CircuitBreakerState",
    "CircuitBreakerConfig",
    "CircuitBreaker",
    "get_circuit_breaker",
    "with_circuit_breaker",
    "DegradationLevel",
    "DegradationAction",
    "DegradationManager",
    "get_degradation_manager",
    "handle_rate_limit_error",
    "handle_token_limit_error",
    "handle_validation_error",
]
