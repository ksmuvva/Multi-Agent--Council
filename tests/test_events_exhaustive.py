"""
Exhaustive Tests for Events Module

Tests event types, emitter, subscriptions, history,
streaming, SSE formatting, and convenience functions.
"""

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from src.utils.events import (
    EventType,
    Event,
    EventSubscription,
    EventEmitter,
    EventStreamer,
    get_event_emitter,
    get_event_streamer,
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

# Reset global emitter before each test
import src.utils.events as events_module


@pytest.fixture(autouse=True)
def reset_global_emitter():
    events_module._global_emitter = None
    events_module._global_streamer = None
    yield
    events_module._global_emitter = None
    events_module._global_streamer = None


# =============================================================================
# EventType Tests
# =============================================================================

class TestEventType:
    def test_task_events(self):
        assert EventType.TASK_STARTED == "task_started"
        assert EventType.TASK_PROGRESS == "task_progress"
        assert EventType.TASK_COMPLETED == "task_completed"
        assert EventType.TASK_FAILED == "task_failed"
        assert EventType.TASK_CANCELLED == "task_cancelled"

    def test_agent_events(self):
        assert EventType.AGENT_STARTED == "agent_started"
        assert EventType.AGENT_PROGRESS == "agent_progress"
        assert EventType.AGENT_COMPLETED == "agent_completed"
        assert EventType.AGENT_FAILED == "agent_failed"

    def test_phase_events(self):
        assert EventType.PHASE_STARTED == "phase_started"
        assert EventType.PHASE_COMPLETED == "phase_completed"
        assert EventType.PHASE_FAILED == "phase_failed"

    def test_quality_events(self):
        assert EventType.VERDICT_PASSED == "verdict_passed"
        assert EventType.VERDICT_FAILED == "verdict_failed"
        assert EventType.QUALITY_GATE == "quality_gate"

    def test_system_events(self):
        assert EventType.SYSTEM_MESSAGE == "system_message"
        assert EventType.ERROR == "error"
        assert EventType.WARNING == "warning"

    def test_sme_events(self):
        assert EventType.SME_SPAWNED == "sme_spawned"
        assert EventType.SME_ADVISORY == "sme_advisory"

    def test_count(self):
        assert len(EventType) >= 20


# =============================================================================
# EventEmitter Tests
# =============================================================================

class TestEventEmitter:
    def test_init(self):
        emitter = EventEmitter()
        assert emitter.get_event_count() == 0

    def test_emit_event(self):
        emitter = EventEmitter()
        event_id = emitter.emit(EventType.TASK_STARTED, "test", {"data": "value"})
        assert event_id is not None
        assert emitter.get_event_count() == 1

    def test_subscribe_and_receive(self):
        emitter = EventEmitter()
        received = []
        emitter.subscribe([EventType.TASK_STARTED], lambda e: received.append(e))
        emitter.emit(EventType.TASK_STARTED, "test", {})
        assert len(received) == 1

    def test_subscribe_filter_by_type(self):
        emitter = EventEmitter()
        received = []
        emitter.subscribe([EventType.TASK_STARTED], lambda e: received.append(e))
        emitter.emit(EventType.TASK_COMPLETED, "test", {})
        assert len(received) == 0

    def test_subscribe_multiple_types(self):
        emitter = EventEmitter()
        received = []
        emitter.subscribe(
            [EventType.TASK_STARTED, EventType.TASK_COMPLETED],
            lambda e: received.append(e)
        )
        emitter.emit(EventType.TASK_STARTED, "test", {})
        emitter.emit(EventType.TASK_COMPLETED, "test", {})
        assert len(received) == 2

    def test_unsubscribe(self):
        emitter = EventEmitter()
        received = []
        sub_id = emitter.subscribe([EventType.TASK_STARTED], lambda e: received.append(e))
        assert emitter.unsubscribe(sub_id) is True
        emitter.emit(EventType.TASK_STARTED, "test", {})
        assert len(received) == 0

    def test_unsubscribe_nonexistent(self):
        emitter = EventEmitter()
        assert emitter.unsubscribe("nonexistent") is False

    def test_session_filter(self):
        emitter = EventEmitter()
        received = []
        emitter.subscribe(
            [EventType.TASK_STARTED], lambda e: received.append(e),
            session_id="session1"
        )
        emitter.emit(EventType.TASK_STARTED, "test", {}, session_id="session1")
        emitter.emit(EventType.TASK_STARTED, "test", {}, session_id="session2")
        assert len(received) == 1

    def test_custom_filter(self):
        emitter = EventEmitter()
        received = []
        emitter.subscribe(
            [EventType.TASK_PROGRESS],
            lambda e: received.append(e),
            filter_func=lambda e: e.data.get("progress", 0) > 0.5,
        )
        emitter.emit(EventType.TASK_PROGRESS, "test", {"progress": 0.3})
        emitter.emit(EventType.TASK_PROGRESS, "test", {"progress": 0.7})
        assert len(received) == 1

    def test_subscriber_id(self):
        emitter = EventEmitter()
        sub_id = emitter.subscribe(
            [EventType.TASK_STARTED],
            lambda e: None,
            subscriber_id="my_sub",
        )
        assert sub_id == "my_sub"

    def test_callback_exception_handled(self):
        emitter = EventEmitter()
        def bad_callback(e):
            raise RuntimeError("callback error")
        emitter.subscribe([EventType.TASK_STARTED], bad_callback)
        # Should not raise
        emitter.emit(EventType.TASK_STARTED, "test", {})

    def test_event_history(self):
        emitter = EventEmitter()
        emitter.emit(EventType.TASK_STARTED, "test", {})
        emitter.emit(EventType.TASK_COMPLETED, "test", {})
        history = emitter.get_event_history()
        assert len(history) == 2

    def test_event_history_by_type(self):
        emitter = EventEmitter()
        emitter.emit(EventType.TASK_STARTED, "test", {})
        emitter.emit(EventType.TASK_COMPLETED, "test", {})
        history = emitter.get_event_history(event_type=EventType.TASK_STARTED)
        assert len(history) == 1

    def test_event_history_by_session(self):
        emitter = EventEmitter()
        emitter.emit(EventType.TASK_STARTED, "test", {}, session_id="s1")
        emitter.emit(EventType.TASK_STARTED, "test", {}, session_id="s2")
        history = emitter.get_event_history(session_id="s1")
        assert len(history) == 1

    def test_event_history_limit(self):
        emitter = EventEmitter()
        for i in range(10):
            emitter.emit(EventType.TASK_PROGRESS, "test", {"i": i})
        history = emitter.get_event_history(limit=5)
        assert len(history) == 5

    def test_max_history(self):
        emitter = EventEmitter()
        emitter._max_history = 5
        for i in range(10):
            emitter.emit(EventType.TASK_PROGRESS, "test", {"i": i})
        assert emitter.get_event_count() == 5

    def test_clear_history_all(self):
        emitter = EventEmitter()
        emitter.emit(EventType.TASK_STARTED, "test", {})
        count = emitter.clear_history()
        assert count == 1
        assert emitter.get_event_count() == 0

    def test_clear_history_by_session(self):
        emitter = EventEmitter()
        emitter.emit(EventType.TASK_STARTED, "test", {}, session_id="s1")
        emitter.emit(EventType.TASK_STARTED, "test", {}, session_id="s2")
        count = emitter.clear_history(session_id="s1")
        assert count == 1
        assert emitter.get_event_count() == 1

    def test_correlation_id(self):
        emitter = EventEmitter()
        emitter.emit(EventType.TASK_STARTED, "test", {}, correlation_id="corr1")
        history = emitter.get_event_history()
        assert history[0].correlation_id == "corr1"


# =============================================================================
# EventStreamer Tests
# =============================================================================

class TestEventStreamer:
    def test_create_stream(self):
        emitter = EventEmitter()
        streamer = EventStreamer(emitter)
        streamer.create_stream("stream1", [EventType.TASK_STARTED])
        assert "stream1" in streamer._streams

    def test_stream_receives_events(self):
        emitter = EventEmitter()
        streamer = EventStreamer(emitter)
        streamer.create_stream("stream1", [EventType.TASK_STARTED])
        emitter.emit(EventType.TASK_STARTED, "test", {"data": "value"})
        events = streamer.get_stream_events("stream1")
        assert len(events) == 1

    def test_stream_nonexistent(self):
        emitter = EventEmitter()
        streamer = EventStreamer(emitter)
        events = streamer.get_stream_events("nonexistent")
        assert events == []

    def test_clear_stream(self):
        emitter = EventEmitter()
        streamer = EventStreamer(emitter)
        streamer.create_stream("stream1", [EventType.TASK_STARTED])
        emitter.emit(EventType.TASK_STARTED, "test", {})
        count = streamer.clear_stream("stream1")
        assert count == 1

    def test_clear_nonexistent_stream(self):
        emitter = EventEmitter()
        streamer = EventStreamer(emitter)
        assert streamer.clear_stream("nonexistent") == 0


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestConvenienceFunctions:
    def test_emit_task_started(self):
        event_id = emit_task_started("orchestrator", "Build API", tier=2)
        assert event_id is not None

    def test_emit_task_progress(self):
        event_id = emit_task_progress("orchestrator", 0.5, "Halfway done")
        assert event_id is not None

    def test_emit_task_completed(self):
        event_id = emit_task_completed("orchestrator", "Done", 10.0)
        assert event_id is not None

    def test_emit_agent_started(self):
        event_id = emit_agent_started("Executor", "Phase 5")
        assert event_id is not None

    def test_emit_agent_completed(self):
        event_id = emit_agent_completed("Executor", "Code generated")
        assert event_id is not None

    def test_emit_finding(self):
        event_id = emit_finding("Critic", "security", "high", "SQL injection found")
        assert event_id is not None

    def test_emit_error(self):
        event_id = emit_error("Verifier", "Timeout occurred")
        assert event_id is not None

    def test_emit_quality_gate_pass(self):
        event_id = emit_quality_gate("Reviewer", "completeness", True)
        assert event_id is not None

    def test_emit_quality_gate_fail(self):
        event_id = emit_quality_gate("Reviewer", "security", False)
        assert event_id is not None

    def test_emit_system_message(self):
        event_id = emit_system_message("System ready")
        assert event_id is not None

    def test_with_session_id(self):
        event_id = emit_task_started("test", "task", tier=1, session_id="sess1")
        assert event_id is not None


# =============================================================================
# SSE Formatting Tests
# =============================================================================

class TestFormatSSEEvent:
    def test_format(self):
        event = Event(
            event_type=EventType.TASK_STARTED,
            timestamp="2025-01-01T00:00:00Z",
            source="test",
            data={"key": "value"},
            event_id="evt_123",
        )
        sse = format_sse_event(event)
        assert "id: evt_123" in sse
        assert "event: task_started" in sse
        assert "data:" in sse

    def test_to_dict(self):
        event = Event(
            event_type=EventType.TASK_STARTED,
            timestamp="2025-01-01T00:00:00Z",
            source="test",
            data={"key": "value"},
            event_id="evt_123",
        )
        d = event.to_dict()
        assert d["event_type"] == "task_started"
        assert d["source"] == "test"


# =============================================================================
# Global Instance Tests
# =============================================================================

class TestGlobalInstances:
    def test_get_event_emitter_singleton(self):
        e1 = get_event_emitter()
        e2 = get_event_emitter()
        assert e1 is e2

    def test_get_event_streamer_singleton(self):
        s1 = get_event_streamer()
        s2 = get_event_streamer()
        assert s1 is s2
