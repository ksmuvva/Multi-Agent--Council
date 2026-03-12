"""
Event System - Real-time Updates for UI Components

Implements publish-subscribe pattern for agent events.
UI components can subscribe to events for real-time updates.
"""

import json
import time
import threading
from typing import Callable, Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from collections import defaultdict
from queue import Queue, Empty


# =============================================================================
# Event Types
# =============================================================================

class EventType(str, Enum):
    """Types of events that can be emitted."""
    # Task lifecycle
    TASK_STARTED = "task_started"
    TASK_PROGRESS = "task_progress"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    TASK_CANCELLED = "task_cancelled"

    # Agent lifecycle
    AGENT_STARTED = "agent_started"
    AGENT_PROGRESS = "agent_progress"
    AGENT_COMPLETED = "agent_completed"
    AGENT_FAILED = "agent_failed"

    # Phase lifecycle
    PHASE_STARTED = "phase_started"
    PHASE_COMPLETED = "phase_completed"
    PHASE_FAILED = "phase_failed"

    # Quality gates
    VERDICT_PASSED = "verdict_passed"
    VERDICT_FAILED = "verdict_failed"
    QUALITY_GATE = "quality_gate"

    # Findings
    FINDING_REPORTED = "finding_reported"
    ISSUE_FLAGGED = "issue_flagged"
    CORRECTION_SUGGESTED = "correction_suggested"

    # System
    SYSTEM_MESSAGE = "system_message"
    WARNING = "warning"
    ERROR = "error"

    # SME events
    SME_SPAWNED = "sme_spawned"
    SME_ADVISORY = "sme_advisory"

    # Memory
    KNOWLEDGE_SAVED = "knowledge_saved"


# =============================================================================
# Event Data Structures
# =============================================================================

@dataclass
class Event:
    """An event emitted by the system."""
    event_type: EventType
    timestamp: str
    source: str  # Agent or component that emitted
    data: Dict[str, Any]
    session_id: Optional[str] = None
    event_id: Optional[str] = None
    correlation_id: Optional[str] = None  # Links related events


@dataclass
class EventSubscription:
    """A subscription to events."""
    subscriber_id: str
    event_types: Set[EventType]
    filter_func: Optional[Callable[[Event], bool]]
    callback: Callable[[Event], None]
    session_id: Optional[str] = None


# =============================================================================
# Event Emitter
# =============================================================================

class EventEmitter:
    """
    Event emitter for publishing events.

    Components emit events through this emitter.
    Subscribers receive matching events.
    """

    def __init__(self):
        self._subscriptions: Dict[str, EventSubscription] = {}
        self._subscription_counter = 0
        self._event_queue: Queue = Queue()
        self._event_history: List[Event] = []
        self._max_history = 1000
        self._lock = threading.Lock()

    def subscribe(
        self,
        event_types: List[EventType],
        callback: Callable[[Event], None],
        subscriber_id: Optional[str] = None,
        filter_func: Optional[Callable[[Event], bool]] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """
        Subscribe to events.

        Args:
            event_types: List of event types to subscribe to
            callback: Function to call when event occurs
            subscriber_id: Optional subscriber identifier
            filter_func: Optional filter function for events
            session_id: Optional session ID for filtering

        Returns:
            Subscription ID for unsubscribing
        """
        if subscriber_id is None:
            subscriber_id = f"sub_{self._subscription_counter}"
            self._subscription_counter += 1

        subscription = EventSubscription(
            subscriber_id=subscriber_id,
            event_types=set(event_types),
            filter_func=filter_func,
            callback=callback,
            session_id=session_id,
        )

        with self._lock:
            self._subscriptions[subscriber_id] = subscription

        return subscriber_id

    def unsubscribe(self, subscription_id: str) -> bool:
        """
        Unsubscribe from events.

        Args:
            subscription_id: ID from subscribe()

        Returns:
            True if unsubscribed, False if not found
        """
        with self._lock:
            if subscription_id in self._subscriptions:
                del self._subscriptions[subscription_id]
                return True
        return False

    def emit(
        self,
        event_type: EventType,
        source: str,
        data: Dict[str, Any],
        session_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> str:
        """
        Emit an event to all subscribers.

        Args:
            event_type: Type of event
            source: Source component
            data: Event payload
            session_id: Optional session ID
            correlation_id: Optional correlation ID for related events

        Returns:
            Event ID
        """
        # Create event
        event_id = f"evt_{int(time.time() * 1000000)}_{source}"
        event = Event(
            event_type=event_type,
            timestamp=datetime.now(timezone.utc).isoformat(),
            source=source,
            data=data,
            session_id=session_id,
            event_id=event_id,
            correlation_id=correlation_id,
        )

        # Add to history
        with self._lock:
            self._event_history.append(event)
            if len(self._event_history) > self._max_history:
                self._event_history.pop(0)

        # Queue for processing
        self._event_queue.put(event)

        # Notify subscribers
        self._notify_subscribers(event)

        return event_id

    def _notify_subscribers(self, event: Event) -> None:
        """Notify all matching subscribers."""
        with self._lock:
            subscriptions = list(self._subscriptions.values())

        for subscription in subscriptions:
            # Check if subscription matches
            if event.event_type not in subscription.event_types:
                continue

            # Check session filter
            if subscription.session_id and subscription.session_id != event.session_id:
                continue

            # Check custom filter
            if subscription.filter_func and not subscription.filter_func(event):
                continue

            # Call callback (in a try/except to avoid breaking other subscribers)
            try:
                subscription.callback(event)
            except Exception as e:
                # Log error but don't stop other subscribers
                self._logger.error(f"Error in subscriber callback: {e}")

    def get_event_history(
        self,
        event_type: Optional[EventType] = None,
        session_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Event]:
        """
        Get event history.

        Args:
            event_type: Optional event type filter
            session_id: Optional session ID filter
            limit: Maximum events to return

        Returns:
            List of matching events
        """
        with self._lock:
            history = self._event_history.copy()

        # Apply filters
        if event_type:
            history = [e for e in history if e.event_type == event_type]
        if session_id:
            history = [e for e in history if e.session_id == session_id]

        # Return most recent first
        history.reverse()
        return history[:limit]

    def get_event_count(self) -> int:
        """Get total number of events in history."""
        with self._lock:
            return len(self._event_history)

    def clear_history(self, session_id: Optional[str] = None) -> int:
        """
        Clear event history.

        Args:
            session_id: Optional session ID to clear for (None = all)

        Returns:
            Number of events cleared
        """
        with self._lock:
            if session_id is None:
                count = len(self._event_history)
                self._event_history.clear()
            else:
                old_count = len(self._event_history)
                self._event_history = [
                    e for e in self._event_history
                    if e.session_id != session_id
                ]
                count = old_count - len(self._event_history)

        return count

    def wait_for_event(
        self,
        event_type: EventType,
        timeout: float = 30.0,
        filter_func: Optional[Callable[[Event], bool]] = None,
    ) -> Optional[Event]:
        """
        Wait for a specific event type.

        Args:
            event_type: Event type to wait for
            timeout: Maximum time to wait (seconds)
            filter_func: Optional filter function

        Returns:
            Matching event, or None if timeout
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            # Check queue
            try:
                event = self._event_queue.get(timeout=0.1)
                if event.event_type == event_type:
                    if filter_func is None or filter_func(event):
                        return event
            except Empty:
                continue

        return None


# =============================================================================
# Global Event Emitter
# =============================================================================

_global_emitter: Optional[EventEmitter] = None


def get_event_emitter() -> EventEmitter:
    """Get the global event emitter instance."""
    global _global_emitter
    if _global_emitter is None:
        _global_emitter = EventEmitter()
    return _global_emitter


# =============================================================================
# Convenience Functions for Common Events
# =============================================================================

def emit_task_started(
    source: str,
    task_description: str,
    tier: int,
    session_id: Optional[str] = None,
) -> str:
    """Emit a task started event."""
    return get_event_emitter().emit(
        EventType.TASK_STARTED,
        source=source,
        data={
            "task_description": task_description,
            "tier": tier,
            "status": "started",
        },
        session_id=session_id,
    )


def emit_task_progress(
    source: str,
    progress: float,  # 0.0 to 1.0
    message: str,
    session_id: Optional[str] = None,
) -> str:
    """Emit a task progress event."""
    return get_event_emitter().emit(
        EventType.TASK_PROGRESS,
        source=source,
        data={
            "progress": progress,
            "message": message,
            "percentage": round(progress * 100, 1),
        },
        session_id=session_id,
    )


def emit_task_completed(
    source: str,
    result_summary: str,
    duration_seconds: float,
    session_id: Optional[str] = None,
) -> str:
    """Emit a task completed event."""
    return get_event_emitter().emit(
        EventType.TASK_COMPLETED,
        source=source,
        data={
            "result_summary": result_summary,
            "duration_seconds": duration_seconds,
            "status": "completed",
        },
        session_id=session_id,
    )


def emit_agent_started(
    agent_name: str,
    phase: str,
    session_id: Optional[str] = None,
) -> str:
    """Emit an agent started event."""
    return get_event_emitter().emit(
        EventType.AGENT_STARTED,
        source=agent_name,
        data={
            "agent": agent_name,
            "phase": phase,
            "status": "started",
        },
        session_id=session_id,
    )


def emit_agent_completed(
    agent_name: str,
    output_summary: str,
    session_id: Optional[str] = None,
) -> str:
    """Emit an agent completed event."""
    return get_event_emitter().emit(
        EventType.AGENT_COMPLETED,
        source=agent_name,
        data={
            "agent": agent_name,
            "output_summary": output_summary,
            "status": "completed",
        },
        session_id=session_id,
    )


def emit_finding(
    source: str,
    finding_type: str,
    severity: str,
    description: str,
    recommendation: Optional[str] = None,
    session_id: Optional[str] = None,
) -> str:
    """Emit a finding event."""
    return get_event_emitter().emit(
        EventType.FINDING_REPORTED,
        source=source,
        data={
            "finding_type": finding_type,
            "severity": severity,
            "description": description,
            "recommendation": recommendation,
        },
        session_id=session_id,
    )


def emit_error(
    source: str,
    error_message: str,
    error_type: str = "error",
    session_id: Optional[str] = None,
) -> str:
    """Emit an error event."""
    return get_event_emitter().emit(
        EventType.ERROR,
        source=source,
        data={
            "error_type": error_type,
            "error_message": error_message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        session_id=session_id,
    )


def emit_quality_gate(
    source: str,
    gate_name: str,
    passed: bool,
    details: Optional[str] = None,
    session_id: Optional[str] = None,
) -> str:
    """Emit a quality gate event."""
    event_type = EventType.VERDICT_PASSED if passed else EventType.VERDICT_FAILED
    return get_event_emitter().emit(
        event_type,
        source=source,
        data={
            "gate_name": gate_name,
            "passed": passed,
            "details": details,
        },
        session_id=session_id,
    )


def emit_system_message(
    message: str,
    level: str = "info",
    session_id: Optional[str] = None,
) -> str:
    """Emit a system message event."""
    return get_event_emitter().emit(
        EventType.SYSTEM_MESSAGE,
        source="system",
        data={
            "message": message,
            "level": level,
        },
        session_id=session_id,
    )


# =============================================================================
# Event Streamer (for UI)
# =============================================================================

class EventStreamer:
    """
    Streams events to UI components.

    Provides a simple interface for UIs to consume events
    via Server-Sent Events or WebSocket.
    """

    def __init__(self, emitter: EventEmitter):
        self.emitter = emitter
        self._streams: Dict[str, List[Event]] = {}
        self._stream_subscribers: Dict[str, Set[str]] = {}

    def create_stream(
        self,
        stream_id: str,
        event_types: List[EventType],
        session_id: Optional[str] = None,
    ) -> None:
        """Create a new event stream."""
        if stream_id not in self._streams:
            self._streams[stream_id] = []

        # Subscribe to events
        def event_handler(event: Event):
            self._streams[stream_id].append(event)

        self.emitter.subscribe(
            event_types=event_types,
            callback=event_handler,
            subscriber_id=f"stream_{stream_id}",
            session_id=session_id,
        )

    def get_stream_events(
        self,
        stream_id: str,
        since_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Event]:
        """Get events from a stream."""
        if stream_id not in self._streams:
            return []

        events = self._streams[stream_id]

        if since_id:
            # Find events after the specified ID
            try:
                index = next(
                    i for i, e in enumerate(events)
                    if e.event_id == since_id
                )
                events = events[index + 1:]
            except StopIteration:
                return []

        # Return most recent first
        events = list(reversed(events))
        return events[:limit]

    def clear_stream(self, stream_id: str) -> int:
        """Clear events for a stream."""
        if stream_id in self._streams:
            count = len(self._streams[stream_id])
            del self._streams[stream_id]
            return count
        return 0


# =============================================================================
# Global Event Streamer
# =============================================================================

_global_streamer: Optional[EventStreamer] = None


def get_event_streamer() -> EventStreamer:
    """Get the global event streamer instance."""
    global _global_streamer
    if _global_streamer is None:
        _global_streamer = EventStreamer(get_event_emitter())
    return _global_streamer


# =============================================================================
# SSE Formatter
# =============================================================================

def format_sse_event(event: Event) -> str:
    """
    Format an event as a Server-Sent Event.

    Args:
        event: The event to format

    Returns:
        SSE formatted string
    """
    lines = [
        f"id: {event.event_id}",
        f"event: {event.event_type.value}",
        f"data: {json.dumps(event.to_dict() if hasattr(event, 'to_dict') else event.__dict__)}",
        "",  # Empty line marks end of SSE event
    ]
    return "\n".join(lines)


# Extend Event with to_dict for serialization
def _add_to_dict_to_event():
    """Add to_dict method to Event class."""
    original_init = Event.__init__

    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "source": self.source,
            "data": self.data,
            "session_id": self.session_id,
            "event_id": self.event_id,
            "correlation_id": self.correlation_id,
        }

    Event.to_dict = to_dict


_add_to_dict_to_event()
