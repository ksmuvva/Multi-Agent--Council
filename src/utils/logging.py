"""
Logging Infrastructure - Structured Logging with Context

Provides structured logging using structlog with context injection,
custom formatters, and handlers for the multi-agent system.
"""

import sys
import logging
import threading
from typing import Any, Dict, Optional, Callable
from datetime import datetime
from pathlib import Path
from contextvars import ContextVar

import structlog
from structlog.types import Processor


# =============================================================================
# Context Variables for Thread-Safe Context
# =============================================================================

# Session context
_session_id: ContextVar[Optional[str]] = ContextVar("session_id", default=None)
_request_id: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
_user_id: ContextVar[Optional[str]] = ContextVar("user_id", default=None)

# Agent context
_agent_name: ContextVar[Optional[str]] = ContextVar("agent_name", default=None)
_agent_tier: ContextVar[Optional[int]] = ContextVar("agent_tier", default=None)
_phase: ContextVar[Optional[str]] = ContextVar("phase", default=None)

# Operation context
_operation: ContextVar[Optional[str]] = ContextVar("operation", default=None)
_component: ContextVar[Optional[str]] = ContextVar("component", default=None)


# =============================================================================
# Log Levels
# =============================================================================

class LogLevel(str):
    """Custom log levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    # Custom levels
    AGENT_START = "agent_start"
    AGENT_COMPLETE = "agent_complete"
    AGENT_ERROR = "agent_error"
    SME_SPAWN = "sme_spawn"
    VERDICT = "verdict"


# =============================================================================
# Context Management
# =============================================================================

def bind_session(**kwargs) -> None:
    """Bind session-level context to all future logs."""
    for key, value in kwargs.items():
        if key == "session_id":
            _session_id.set(value)
        elif key == "request_id":
            _request_id.set(value)
        elif key == "user_id":
            _user_id.set(value)


def bind_agent(**kwargs) -> None:
    """Bind agent context to all future logs."""
    for key, value in kwargs.items():
        if key == "agent_name":
            _agent_name.set(value)
        elif key == "tier":
            _agent_tier.set(value)
        elif key == "phase":
            _phase.set(value)


def bind_operation(**kwargs) -> None:
    """Bind operation context to all future logs."""
    for key, value in kwargs.items():
        if key == "operation":
            _operation.set(value)
        elif key == "component":
            _component.set(value)


def clear_context() -> None:
    """Clear all context variables."""
    _session_id.set(None)
    _request_id.set(None)
    _user_id.set(None)
    _agent_name.set(None)
    _agent_tier.set(None)
    _phase.set(None)
    _operation.set(None)
    _component.set(None)


# =============================================================================
# Custom Processors
# =============================================================================

def context_injector(logger: Any, method_name: str, event_dict: Dict) -> Dict:
    """
    Inject context variables into the log event.

    This processor automatically adds thread-safe context to all log events.
    """
    # Session context
    session_id = _session_id.get()
    if session_id:
        event_dict["session_id"] = session_id

    request_id = _request_id.get()
    if request_id:
        event_dict["request_id"] = request_id

    user_id = _user_id.get()
    if user_id:
        event_dict["user_id"] = user_id

    # Agent context
    agent_name = _agent_name.get()
    if agent_name:
        event_dict["agent"] = agent_name

    agent_tier = _agent_tier.get()
    if agent_tier is not None:
        event_dict["tier"] = agent_tier

    phase = _phase.get()
    if phase:
        event_dict["phase"] = phase

    # Operation context
    operation = _operation.get()
    if operation:
        event_dict["operation"] = operation

    component = _component.get()
    if component:
        event_dict["component"] = component

    return event_dict


def add_timestamp(logger: Any, method_name: str, event_dict: Dict) -> Dict:
    """Add ISO 8601 timestamp to log events."""
    event_dict["timestamp"] = datetime.utcnow().isoformat()
    return event_dict


def add_log_level(logger: Any, method_name: str, event_dict: Dict) -> Dict:
    """Add the log level to events."""
    event_dict["level"] = method_name.upper()
    return event_dict


def rename_message(logger: Any, method_name: str, event_dict: Dict) -> Dict:
    """Rename 'event' key to 'message' for consistency."""
    event_dict["message"] = event_dict.pop("event")
    return event_dict


def stack_trace_formatter(logger: Any, method_name: str, event_dict: Dict) -> Dict:
    """
    Format stack traces nicely when an exception is present.
    """
    if "exc_info" in event_dict:
        exc_info = event_dict["exc_info"]

        if exc_info:
            # Format the exception using Python's built-in traceback
            import traceback as tb
            import io
            import sys

            output = io.StringIO()
            # exc_info can be True (boolean) or a 3-tuple
            # If it's True, get the current exception info
            if exc_info is True:
                exc_info = sys.exc_info()
            # Now format the exception
            if exc_info and exc_info != (None, None, None):
                tb.print_exception(exc_info[0], exc_info[1], exc_info[2], file=output)
                event_dict["stack_trace"] = output.getvalue()

    return event_dict


def filter_sensitive_data(logger: Any, method_name: str, event_dict: Dict) -> Dict:
    """
    Filter out sensitive data from logs.
    """
    sensitive_keys = [
        "api_key", "apikey", "secret", "token", "password",
        "credential", "auth", "private_key", "access_token",
    ]

    filtered_event = {}

    for key, value in event_dict.items():
        # Check if this is a sensitive key
        if any(sensitive in key.lower() for sensitive in sensitive_keys):
            # Mask the value
            filtered_event[key] = "***REDACTED***"
        elif isinstance(value, str):
            # Check if value looks like sensitive data
            value_lower = value.lower()
            if any(sensitive in value_lower for sensitive in ["sk-", "bearer ", "basic "]):
                filtered_event[key] = "***REDACTED***"
            else:
                filtered_event[key] = value
        else:
            filtered_event[key] = value

    return filtered_event


def agent_event_classifier(logger: Any, method_name: str, event_dict: Dict) -> Dict:
    """
    Classify agent-related events with appropriate markers.
    """
    # Check for agent lifecycle events
    if "agent_start" in event_dict.get("event", "").lower():
        event_dict["event_type"] = "agent_lifecycle"
        event_dict["lifecycle_stage"] = "start"

    elif "agent_complete" in event_dict.get("event", "").lower():
        event_dict["event_type"] = "agent_lifecycle"
        event_dict["lifecycle_stage"] = "complete"

    elif "agent_error" in event_dict.get("event", "").lower():
        event_dict["event_type"] = "agent_error"
        event_dict["lifecycle_stage"] = "error"

    # Check for verdict events
    elif "verdict" in event_dict.get("event", "").lower():
        event_dict["event_type"] = "quality_gate"

    # Check for SME events
    elif "sme_spawn" in event_dict.get("event", "").lower():
        event_dict["event_type"] = "sme_lifecycle"

    return event_dict


# =============================================================================
# Formatters
# =============================================================================

def console_formatter() -> Processor:
    """
    Get the console log formatter.

    Returns:
        Processor for console formatting
    """
    # Disable colors on Windows to avoid colorama recursion bug
    import sys
    use_colors = sys.platform != 'win32'

    return structlog.dev.ConsoleRenderer(
        colors=use_colors,
        exception_formatter=structlog.dev.plain_traceback,
    )


def json_formatter() -> Processor:
    """
    Get the JSON log formatter for file output.

    Returns:
        Processor for JSON formatting
    """
    return structlog.processors.JSONRenderer()


# =============================================================================
# Configuration
# =============================================================================

def configure_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    json_output: bool = False,
    enable_filtering: bool = True,
) -> None:
    """
    Configure structlog for the multi-agent system.

    Args:
        level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for file logging
        json_output: Whether to output JSON format (for production)
        enable_filtering: Whether to enable sensitive data filtering
    """
    # Base processors
    processors = [
        context_injector,
        add_timestamp,
        add_log_level,
        agent_event_classifier,
    ]

    if enable_filtering:
        processors.append(filter_sensitive_data)

    if json_output:
        # Production: JSON output
        processors.extend([
            rename_message,
            stack_trace_formatter,
            json_formatter(),
        ])
    else:
        # Development: pretty console output
        processors.extend([
            rename_message,
            stack_trace_formatter,
            console_formatter(),
        ])

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Also configure standard logging for libraries
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper(), logging.INFO),
    )

    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper(), logging.INFO))

        # Use structlog's file renderer
        file_processor = structlog.processors.JSONRenderer()
        file_handler.setFormatter(
            logging.Formatter("%(message)s")
        )

        # Add to root logger
        logging.root.addHandler(file_handler)


# =============================================================================
# Logger Getters
# =============================================================================

def get_logger(name: str = None) -> Any:
    """
    Get a structlog logger with context.

    Args:
        name: Optional logger name

    Returns:
        Structlog logger instance
    """
    if name:
        return structlog.get_logger(name)
    return structlog.get_logger()


def get_agent_logger(agent_name: str) -> Any:
    """
    Get a logger specifically for an agent.

    Args:
        agent_name: Name of the agent

    Returns:
        Structlog logger with agent context pre-bound
    """
    logger = structlog.get_logger(agent_name)
    logger = logger.bind(agent=agent_name)
    return logger


def get_session_logger(session_id: str) -> Any:
    """
    Get a logger specifically for a session.

    Args:
        session_id: Session identifier

    Returns:
        Structlog logger with session context pre-bound
    """
    logger = structlog.get_logger()
    logger = logger.bind(session_id=session_id)
    return logger


# =============================================================================
# Convenience Logging Functions
# =============================================================================

def log_agent_start(agent_name: str, phase: str, tier: int, **kwargs) -> None:
    """Log an agent starting."""
    logger = get_logger()
    bind_agent(agent_name=agent_name, tier=tier, phase=phase)
    logger.info("agent_start", agent=agent_name, phase=phase, tier=tier, **kwargs)


def log_agent_complete(agent_name: str, phase: str, duration: float, **kwargs) -> None:
    """Log an agent completing."""
    logger = get_logger()
    logger.info(
        "agent_complete",
        agent=agent_name,
        phase=phase,
        duration_seconds=duration,
        **kwargs
    )


def log_agent_error(agent_name: str, phase: str, error: Exception, **kwargs) -> None:
    """Log an agent error."""
    logger = get_logger()
    logger.error(
        "agent_error",
        agent=agent_name,
        phase=phase,
        error_type=type(error).__name__,
        error_message=str(error),
        exc_info=error,
        **kwargs
    )


def log_sme_spawn(persona_id: str, persona_name: str, interaction_mode: str, **kwargs) -> None:
    """Log an SME persona being spawned."""
    logger = get_logger()
    logger.info(
        "sme_spawn",
        sme_persona_id=persona_id,
        sme_persona_name=persona_name,
        interaction_mode=interaction_mode,
        **kwargs
    )


def log_verdict(gate_name: str, passed: bool, details: str, **kwargs) -> None:
    """Log a quality gate verdict."""
    logger = get_logger()
    level = "info" if passed else "warning"
    getattr(logger, level)(
        "verdict",
        gate_name=gate_name,
        passed=passed,
        details=details,
        **kwargs
    )


def log_cost(tokens: int, cost_usd: float, model: str, operation: str, **kwargs) -> None:
    """Log cost information."""
    logger = get_logger()
    logger.debug(
        "cost",
        tokens=tokens,
        cost_usd=cost_usd,
        model=model,
        operation=operation,
        **kwargs
    )


# =============================================================================
# Context Managers
# =============================================================================

class AgentLogContext:
    """Context manager for agent logging context."""

    def __init__(self, agent_name: str, tier: int = None, phase: str = None):
        self.agent_name = agent_name
        self.tier = tier
        self.phase = phase
        self.previous_agent = _agent_name.get(None)
        self.previous_tier = _agent_tier.get(None)
        self.previous_phase = _phase.get(None)

    def __enter__(self):
        bind_agent(agent_name=self.agent_name, tier=self.tier, phase=self.phase)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore previous context
        _agent_name.set(self.previous_agent)
        _agent_tier.set(self.previous_tier)
        _phase.set(self.previous_phase)


class SessionLogContext:
    """Context manager for session logging context."""

    def __init__(self, session_id: str, user_id: str = None, request_id: str = None):
        self.session_id = session_id
        self.user_id = user_id
        self.request_id = request_id
        self.previous_session = _session_id.get(None)
        self.previous_user = _user_id.get(None)
        self.previous_request = _request_id.get(None)

    def __enter__(self):
        bind_session(
            session_id=self.session_id,
            user_id=self.user_id,
            request_id=self.request_id,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore previous context
        _session_id.set(self.previous_session)
        _user_id.set(self.previous_user)
        _request_id.set(self.previous_request)


# =============================================================================
# Initialization
# =============================================================================

# Default configuration (can be overridden)
configure_logging(
    level="INFO",
    json_output=False,
    enable_filtering=True,
)
