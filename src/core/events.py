"""
Event Bus - Lightweight publish/subscribe system for agent lifecycle events.

Connects the orchestrator's agent spawning to the Streamlit UI components
(Agent Activity Panel and Cost Dashboard).
"""

from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List


# =============================================================================
# Event Types
# =============================================================================

class EventType(str, Enum):
    """Agent lifecycle event types."""
    AGENT_STARTED = "agent_started"
    AGENT_PROGRESS = "agent_progress"
    AGENT_COMPLETED = "agent_completed"
    AGENT_FAILED = "agent_failed"
    COST_RECORDED = "cost_recorded"


# =============================================================================
# Event Bus
# =============================================================================

class EventBus:
    """Simple synchronous publish/subscribe event bus."""

    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}

    def subscribe(self, event_type: str, callback: Callable) -> None:
        """
        Register a callback for an event type.

        Args:
            event_type: The event type to listen for (use EventType values).
            callback: A callable that accepts a single dict argument.
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        if callback not in self._subscribers[event_type]:
            self._subscribers[event_type].append(callback)

    def unsubscribe(self, event_type: str, callback: Callable) -> None:
        """Remove a callback for an event type."""
        if event_type in self._subscribers:
            self._subscribers[event_type] = [
                cb for cb in self._subscribers[event_type] if cb != callback
            ]

    def emit(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Emit an event to all subscribers.

        Args:
            event_type: The event type being emitted.
            data: Event payload dictionary (not mutated).
        """
        # Copy to avoid mutating the caller's dict
        data = {
            **data,
            "event_type": event_type,
            "timestamp": data.get("timestamp", datetime.now().isoformat()),
        }
        for callback in self._subscribers.get(event_type, []):
            try:
                callback(data)
            except Exception as exc:
                # Log but swallow UI callback errors so they never break the orchestrator
                import logging
                logging.getLogger(__name__).debug(
                    "Event callback %s raised: %s", callback.__name__, exc,
                )

    def clear(self) -> None:
        """Remove all subscribers."""
        self._subscribers.clear()


# =============================================================================
# Global singleton
# =============================================================================

agent_event_bus = EventBus()


# =============================================================================
# Helper functions
# =============================================================================

def emit_agent_started(
    agent_id: str,
    agent_name: str,
    tier: str,
    phase: str = "Initializing",
) -> None:
    """Emit an AGENT_STARTED event."""
    agent_event_bus.emit(EventType.AGENT_STARTED, {
        "agent_id": agent_id,
        "agent_name": agent_name,
        "tier": tier,
        "phase": phase,
    })


def emit_agent_completed(
    agent_id: str,
    agent_name: str,
    output: str = "",
) -> None:
    """Emit an AGENT_COMPLETED event."""
    agent_event_bus.emit(EventType.AGENT_COMPLETED, {
        "agent_id": agent_id,
        "agent_name": agent_name,
        "output": output,
    })


def emit_agent_failed(
    agent_id: str,
    agent_name: str,
    error: str = "",
) -> None:
    """Emit an AGENT_FAILED event."""
    agent_event_bus.emit(EventType.AGENT_FAILED, {
        "agent_id": agent_id,
        "agent_name": agent_name,
        "error": error,
    })


def emit_cost_recorded(
    agent_name: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    total_tokens: int,
    cost_usd: float,
    tier: int,
    phase: str,
) -> None:
    """Emit a COST_RECORDED event."""
    agent_event_bus.emit(EventType.COST_RECORDED, {
        "agent_name": agent_name,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cost_usd": cost_usd,
        "tier": tier,
        "phase": phase,
    })
