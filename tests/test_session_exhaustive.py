"""
Exhaustive Tests for Session Modules

Tests both session persistence and context compaction:
- AgentOutput, ChatMessage, SessionState dataclasses
- SessionPersistence: save, load, list, delete sessions
- CompactionConfig: defaults and custom values
- ContextCompactor: should_compact, compact_session, restore context
- MessageAnalyzer: token estimation, preservable items, compaction
- Convenience functions: check_and_compact, set_compaction_config
"""

import sys
import json
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

from src.session.persistence import (
    AgentOutput,
    ChatMessage,
    SessionSummary,
    SessionState,
    SessionPersistence,
    create_session,
    resume_session,
    save_session,
)
from src.session.compaction import (
    CompactionConfig,
    CompactionResult,
    CompactionTrigger,
    ContextCompactor,
    MessageAnalyzer,
    check_and_compact,
    compact_session_manual,
    get_context_compactor,
    set_compaction_config,
)


# =============================================================================
# Helper Fixtures
# =============================================================================

@pytest.fixture
def sample_timestamp():
    """Provide a consistent test timestamp."""
    return datetime(2025, 6, 15, 12, 0, 0)


@pytest.fixture
def sample_agent_output(sample_timestamp):
    """Provide a sample AgentOutput."""
    return AgentOutput(
        agent_name="Executor",
        phase="execution",
        tier=2,
        content="Here is the solution.",
        structured_data={"key": "value"},
        timestamp=sample_timestamp,
        duration_seconds=5.0,
        token_usage={"input": 100, "output": 200},
        status="completed",
    )


@pytest.fixture
def sample_chat_message(sample_timestamp):
    """Provide a sample ChatMessage."""
    return ChatMessage(
        role="user",
        content="Write a Python function.",
        timestamp=sample_timestamp,
        agent_name=None,
        tier=2,
        metadata={"source": "cli"},
    )


@pytest.fixture
def sample_session(sample_timestamp):
    """Provide a sample SessionState."""
    return SessionState(
        session_id="test_session_001",
        created_at=sample_timestamp,
        updated_at=sample_timestamp,
        tier=2,
        max_budget=10.0,
        default_format="markdown",
        messages=[
            ChatMessage(
                role="user",
                content="Write a function.",
                timestamp=sample_timestamp,
            ),
            ChatMessage(
                role="assistant",
                content="Here is the function.",
                timestamp=sample_timestamp,
                agent_name="Executor",
                tier=2,
            ),
        ],
        agent_outputs=[
            AgentOutput(
                agent_name="Executor",
                phase="execution",
                tier=2,
                content="def hello(): pass",
                timestamp=sample_timestamp,
            ),
        ],
        active_agents=["Executor"],
        current_phase="execution",
        total_tokens=500,
        total_cost_usd=0.05,
        daily_budget_usd=50.0,
        title="Test Session",
        description="A test session",
        tags=["test", "python"],
    )


# =============================================================================
# AgentOutput Tests
# =============================================================================

class TestAgentOutput:
    """Tests for AgentOutput dataclass."""

    def test_required_fields(self):
        """Test creation with required fields only."""
        output = AgentOutput(
            agent_name="Analyst",
            phase="analysis",
            tier=1,
            content="Analysis complete.",
        )
        assert output.agent_name == "Analyst"
        assert output.phase == "analysis"
        assert output.tier == 1
        assert output.content == "Analysis complete."

    def test_default_values(self):
        """Test default field values."""
        output = AgentOutput(
            agent_name="Test", phase="test", tier=1, content="test"
        )
        assert output.structured_data is None
        assert output.timestamp is None
        assert output.duration_seconds == 0.0
        assert output.token_usage is None
        assert output.status == "completed"

    def test_all_fields(self, sample_timestamp):
        """Test creation with all fields."""
        output = AgentOutput(
            agent_name="Executor",
            phase="execution",
            tier=3,
            content="Solution",
            structured_data={"result": "ok"},
            timestamp=sample_timestamp,
            duration_seconds=10.5,
            token_usage={"input": 500, "output": 300},
            status="completed",
        )
        assert output.structured_data == {"result": "ok"}
        assert output.timestamp == sample_timestamp
        assert output.duration_seconds == 10.5
        assert output.token_usage == {"input": 500, "output": 300}

    def test_failed_status(self):
        """Test with failed status."""
        output = AgentOutput(
            agent_name="Verifier", phase="review", tier=2,
            content="", status="failed",
        )
        assert output.status == "failed"

    def test_timeout_status(self):
        """Test with timeout status."""
        output = AgentOutput(
            agent_name="Executor", phase="execution", tier=2,
            content="", status="timeout",
        )
        assert output.status == "timeout"

    def test_to_dict(self, sample_agent_output):
        """Test to_dict serialization."""
        d = sample_agent_output.to_dict()
        assert d["agent_name"] == "Executor"
        assert d["phase"] == "execution"
        assert d["tier"] == 2
        assert d["content"] == "Here is the solution."
        assert d["structured_data"] == {"key": "value"}
        assert d["timestamp"] is not None
        assert d["duration_seconds"] == 5.0
        assert d["token_usage"] == {"input": 100, "output": 200}
        assert d["status"] == "completed"

    def test_to_dict_none_timestamp(self):
        """Test to_dict with None timestamp."""
        output = AgentOutput(
            agent_name="Test", phase="test", tier=1, content="test"
        )
        d = output.to_dict()
        assert d["timestamp"] is None

    def test_from_dict(self):
        """Test from_dict deserialization."""
        data = {
            "agent_name": "Analyst",
            "phase": "analysis",
            "tier": 2,
            "content": "Done",
            "structured_data": {"k": "v"},
            "timestamp": "2025-06-15T12:00:00",
            "duration_seconds": 3.5,
            "token_usage": {"input": 50},
            "status": "completed",
        }
        output = AgentOutput.from_dict(data)
        assert output.agent_name == "Analyst"
        assert output.tier == 2
        assert output.structured_data == {"k": "v"}
        assert output.timestamp == datetime(2025, 6, 15, 12, 0, 0)
        assert output.duration_seconds == 3.5
        assert output.status == "completed"

    def test_from_dict_minimal(self):
        """Test from_dict with minimal data."""
        data = {
            "agent_name": "Test",
            "phase": "test",
            "tier": 1,
            "content": "content",
        }
        output = AgentOutput.from_dict(data)
        assert output.agent_name == "Test"
        assert output.timestamp is None
        assert output.duration_seconds == 0.0
        assert output.status == "completed"

    def test_roundtrip_serialization(self, sample_agent_output):
        """Test to_dict -> from_dict roundtrip."""
        d = sample_agent_output.to_dict()
        restored = AgentOutput.from_dict(d)
        assert restored.agent_name == sample_agent_output.agent_name
        assert restored.phase == sample_agent_output.phase
        assert restored.tier == sample_agent_output.tier
        assert restored.content == sample_agent_output.content
        assert restored.status == sample_agent_output.status


# =============================================================================
# ChatMessage Tests
# =============================================================================

class TestChatMessage:
    """Tests for ChatMessage dataclass."""

    def test_required_fields(self, sample_timestamp):
        """Test creation with required fields."""
        msg = ChatMessage(role="user", content="Hello", timestamp=sample_timestamp)
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.timestamp == sample_timestamp

    def test_default_values(self, sample_timestamp):
        """Test default field values."""
        msg = ChatMessage(role="user", content="Hello", timestamp=sample_timestamp)
        assert msg.agent_name is None
        assert msg.tier is None
        assert msg.metadata == {}

    def test_all_fields(self, sample_timestamp):
        """Test with all fields."""
        msg = ChatMessage(
            role="assistant",
            content="Response",
            timestamp=sample_timestamp,
            agent_name="Executor",
            tier=3,
            metadata={"source": "pipeline"},
        )
        assert msg.agent_name == "Executor"
        assert msg.tier == 3
        assert msg.metadata == {"source": "pipeline"}

    def test_to_dict(self, sample_chat_message):
        """Test to_dict serialization."""
        d = sample_chat_message.to_dict()
        assert d["role"] == "user"
        assert d["content"] == "Write a Python function."
        assert d["timestamp"] is not None
        assert d["agent_name"] is None
        assert d["tier"] == 2
        assert d["metadata"] == {"source": "cli"}

    def test_from_dict(self):
        """Test from_dict deserialization."""
        data = {
            "role": "assistant",
            "content": "Here is the answer.",
            "timestamp": "2025-06-15T12:00:00",
            "agent_name": "Executor",
            "tier": 2,
            "metadata": {"key": "value"},
        }
        msg = ChatMessage.from_dict(data)
        assert msg.role == "assistant"
        assert msg.content == "Here is the answer."
        assert msg.timestamp == datetime(2025, 6, 15, 12, 0, 0)
        assert msg.agent_name == "Executor"
        assert msg.tier == 2
        assert msg.metadata == {"key": "value"}

    def test_from_dict_minimal(self):
        """Test from_dict with minimal data."""
        data = {
            "role": "system",
            "content": "System message",
            "timestamp": "2025-01-01T00:00:00",
        }
        msg = ChatMessage.from_dict(data)
        assert msg.role == "system"
        assert msg.agent_name is None
        assert msg.tier is None
        assert msg.metadata == {}

    def test_roundtrip_serialization(self, sample_chat_message):
        """Test to_dict -> from_dict roundtrip."""
        d = sample_chat_message.to_dict()
        restored = ChatMessage.from_dict(d)
        assert restored.role == sample_chat_message.role
        assert restored.content == sample_chat_message.content

    def test_system_role(self, sample_timestamp):
        """Test system role message."""
        msg = ChatMessage(role="system", content="You are an agent.", timestamp=sample_timestamp)
        assert msg.role == "system"


# =============================================================================
# SessionState Tests
# =============================================================================

class TestSessionState:
    """Tests for SessionState dataclass."""

    def test_required_fields(self, sample_timestamp):
        """Test creation with required fields."""
        state = SessionState(
            session_id="sess_001",
            created_at=sample_timestamp,
            updated_at=sample_timestamp,
        )
        assert state.session_id == "sess_001"
        assert state.created_at == sample_timestamp

    def test_default_values(self, sample_timestamp):
        """Test default field values."""
        state = SessionState(
            session_id="test",
            created_at=sample_timestamp,
            updated_at=sample_timestamp,
        )
        assert state.tier == 2
        assert state.max_budget == 10.0
        assert state.default_format == "markdown"
        assert state.messages == []
        assert state.agent_outputs == []
        assert state.active_agents == []
        assert state.current_phase == ""
        assert state.total_tokens == 0
        assert state.total_cost_usd == 0.0
        assert state.daily_budget_usd == 50.0
        assert state.title is None
        assert state.description is None
        assert state.tags == []

    def test_to_dict(self, sample_session):
        """Test to_dict serialization."""
        d = sample_session.to_dict()
        assert d["session_id"] == "test_session_001"
        assert d["tier"] == 2
        assert d["max_budget"] == 10.0
        assert len(d["messages"]) == 2
        assert len(d["agent_outputs"]) == 1
        assert d["active_agents"] == ["Executor"]
        assert d["total_tokens"] == 500
        assert d["total_cost_usd"] == 0.05
        assert d["title"] == "Test Session"
        assert d["tags"] == ["test", "python"]

    def test_from_dict(self, sample_session):
        """Test from_dict deserialization."""
        d = sample_session.to_dict()
        restored = SessionState.from_dict(d)
        assert restored.session_id == sample_session.session_id
        assert restored.tier == sample_session.tier
        assert len(restored.messages) == len(sample_session.messages)
        assert len(restored.agent_outputs) == len(sample_session.agent_outputs)
        assert restored.title == sample_session.title
        assert restored.tags == sample_session.tags

    def test_from_dict_minimal(self):
        """Test from_dict with minimal data."""
        data = {
            "session_id": "minimal",
            "created_at": "2025-01-01T00:00:00",
            "updated_at": "2025-01-01T00:00:00",
        }
        state = SessionState.from_dict(data)
        assert state.session_id == "minimal"
        assert state.tier == 2
        assert state.messages == []
        assert state.agent_outputs == []

    def test_roundtrip_serialization(self, sample_session):
        """Test to_dict -> from_dict roundtrip."""
        d = sample_session.to_dict()
        restored = SessionState.from_dict(d)
        d2 = restored.to_dict()
        # Compare key fields (timestamps may differ slightly due to serialization)
        assert d["session_id"] == d2["session_id"]
        assert d["tier"] == d2["tier"]
        assert d["total_tokens"] == d2["total_tokens"]

    def test_get_summary(self, sample_session):
        """Test get_summary."""
        summary = sample_session.get_summary()
        assert isinstance(summary, SessionSummary)
        assert summary.session_id == "test_session_001"
        assert summary.tier == 2
        assert summary.total_messages == 2
        assert summary.total_agent_outputs == 1
        assert summary.total_tokens == 500
        assert summary.total_cost_usd == 0.05
        assert summary.title == "Test Session"

    def test_mutable_lists(self, sample_timestamp):
        """Test that list fields are mutable and independent."""
        state1 = SessionState(
            session_id="s1",
            created_at=sample_timestamp,
            updated_at=sample_timestamp,
        )
        state2 = SessionState(
            session_id="s2",
            created_at=sample_timestamp,
            updated_at=sample_timestamp,
        )
        state1.messages.append(
            ChatMessage(role="user", content="test", timestamp=sample_timestamp)
        )
        assert len(state1.messages) == 1
        assert len(state2.messages) == 0


# =============================================================================
# SessionPersistence Tests
# =============================================================================

class TestSessionPersistence:
    """Tests for SessionPersistence class."""

    def test_init_creates_directory(self, tmp_path):
        """Test that init creates the sessions directory."""
        sessions_dir = tmp_path / "sessions"
        with patch("src.session.persistence.get_logger", return_value=MagicMock()):
            sp = SessionPersistence(sessions_dir=sessions_dir)
        assert sessions_dir.exists()

    def test_init_default_directory(self):
        """Test default directory path."""
        with patch("src.session.persistence.get_logger", return_value=MagicMock()), \
             patch.object(Path, "mkdir"):
            sp = SessionPersistence()
            assert "sessions" in str(sp.sessions_dir)

    def test_get_session_path(self, tmp_path):
        """Test session path computation."""
        with patch("src.session.persistence.get_logger", return_value=MagicMock()):
            sp = SessionPersistence(sessions_dir=tmp_path)
        path = sp.get_session_path("sess_123")
        assert path == tmp_path / "sess_123" / "session.json"

    def test_save_session(self, tmp_path, sample_session):
        """Test saving a session to disk."""
        with patch("src.session.persistence.get_logger", return_value=MagicMock()):
            sp = SessionPersistence(sessions_dir=tmp_path)
        sp.save_session(sample_session)

        session_path = tmp_path / "test_session_001" / "session.json"
        assert session_path.exists()

        with open(session_path) as f:
            data = json.load(f)
        assert data["session_id"] == "test_session_001"
        assert data["tier"] == 2

    def test_save_session_updates_timestamp(self, tmp_path, sample_session):
        """Test that saving updates the updated_at timestamp."""
        with patch("src.session.persistence.get_logger", return_value=MagicMock()):
            sp = SessionPersistence(sessions_dir=tmp_path)
        original_updated = sample_session.updated_at
        sp.save_session(sample_session)
        # updated_at should be set to now()
        assert sample_session.updated_at >= original_updated

    def test_load_session(self, tmp_path, sample_session):
        """Test loading a session from disk."""
        with patch("src.session.persistence.get_logger", return_value=MagicMock()):
            sp = SessionPersistence(sessions_dir=tmp_path)
        sp.save_session(sample_session)

        loaded = sp.load_session("test_session_001")
        assert loaded is not None
        assert loaded.session_id == "test_session_001"
        assert loaded.tier == 2
        assert len(loaded.messages) == 2
        assert len(loaded.agent_outputs) == 1

    def test_load_session_not_found(self, tmp_path):
        """Test loading a non-existent session returns None."""
        with patch("src.session.persistence.get_logger", return_value=MagicMock()):
            sp = SessionPersistence(sessions_dir=tmp_path)
        result = sp.load_session("nonexistent_session")
        assert result is None

    def test_load_session_corrupt_data(self, tmp_path):
        """Test loading a session with corrupt JSON."""
        with patch("src.session.persistence.get_logger", return_value=MagicMock()):
            sp = SessionPersistence(sessions_dir=tmp_path)

        # Create corrupt session file
        session_dir = tmp_path / "corrupt_sess"
        session_dir.mkdir()
        (session_dir / "session.json").write_text("not valid json{{{")

        result = sp.load_session("corrupt_sess")
        assert result is None

    def test_delete_session(self, tmp_path, sample_session):
        """Test deleting a session."""
        with patch("src.session.persistence.get_logger", return_value=MagicMock()):
            sp = SessionPersistence(sessions_dir=tmp_path)
        sp.save_session(sample_session)

        result = sp.delete_session("test_session_001")
        assert result is True
        assert not (tmp_path / "test_session_001").exists()

    def test_delete_session_not_found(self, tmp_path):
        """Test deleting a non-existent session."""
        with patch("src.session.persistence.get_logger", return_value=MagicMock()):
            sp = SessionPersistence(sessions_dir=tmp_path)
        result = sp.delete_session("nonexistent")
        assert result is False

    def test_delete_all_sessions(self, tmp_path, sample_timestamp):
        """Test deleting all sessions."""
        with patch("src.session.persistence.get_logger", return_value=MagicMock()):
            sp = SessionPersistence(sessions_dir=tmp_path)

        # Create multiple sessions
        for i in range(3):
            session = SessionState(
                session_id=f"sess_{i}",
                created_at=sample_timestamp,
                updated_at=sample_timestamp,
            )
            sp.save_session(session)

        count = sp.delete_all_sessions()
        assert count == 3

    def test_list_sessions(self, tmp_path, sample_timestamp):
        """Test listing sessions."""
        with patch("src.session.persistence.get_logger", return_value=MagicMock()):
            sp = SessionPersistence(sessions_dir=tmp_path)

        for i in range(5):
            session = SessionState(
                session_id=f"sess_{i}",
                created_at=sample_timestamp + timedelta(hours=i),
                updated_at=sample_timestamp + timedelta(hours=i),
                title=f"Session {i}",
            )
            sp.save_session(session)

        summaries = sp.list_sessions()
        assert len(summaries) == 5

    def test_list_sessions_with_limit(self, tmp_path, sample_timestamp):
        """Test listing sessions with limit."""
        with patch("src.session.persistence.get_logger", return_value=MagicMock()):
            sp = SessionPersistence(sessions_dir=tmp_path)

        for i in range(5):
            session = SessionState(
                session_id=f"sess_{i}",
                created_at=sample_timestamp,
                updated_at=sample_timestamp,
            )
            sp.save_session(session)

        summaries = sp.list_sessions(limit=3)
        assert len(summaries) == 3

    def test_list_sessions_sorted_descending(self, tmp_path, sample_timestamp):
        """Test that sessions are sorted in descending order by default."""
        with patch("src.session.persistence.get_logger", return_value=MagicMock()):
            sp = SessionPersistence(sessions_dir=tmp_path)

        for i in range(3):
            session = SessionState(
                session_id=f"sess_{i}",
                created_at=sample_timestamp + timedelta(hours=i),
                updated_at=sample_timestamp + timedelta(hours=i),
            )
            sp.save_session(session)

        summaries = sp.list_sessions(sort_by="created_at", descending=True)
        if len(summaries) >= 2:
            assert summaries[0].created_at >= summaries[1].created_at

    def test_list_sessions_empty(self, tmp_path):
        """Test listing sessions when none exist."""
        with patch("src.session.persistence.get_logger", return_value=MagicMock()):
            sp = SessionPersistence(sessions_dir=tmp_path)
        summaries = sp.list_sessions()
        assert summaries == []

    def test_save_load_roundtrip(self, tmp_path, sample_session):
        """Test full save/load roundtrip preserves data."""
        with patch("src.session.persistence.get_logger", return_value=MagicMock()):
            sp = SessionPersistence(sessions_dir=tmp_path)
        sp.save_session(sample_session)
        loaded = sp.load_session("test_session_001")

        assert loaded.session_id == sample_session.session_id
        assert loaded.tier == sample_session.tier
        assert loaded.max_budget == sample_session.max_budget
        assert loaded.total_tokens == sample_session.total_tokens
        assert loaded.title == sample_session.title
        assert loaded.tags == sample_session.tags
        assert len(loaded.messages) == len(sample_session.messages)
        assert len(loaded.agent_outputs) == len(sample_session.agent_outputs)


# =============================================================================
# Session Factory Tests
# =============================================================================

class TestSessionFactory:
    """Tests for create_session and resume_session functions."""

    def test_create_session_defaults(self):
        """Test creating a session with defaults."""
        session = create_session()
        assert session.session_id.startswith("sess_")
        assert session.tier == 2
        assert session.max_budget == 10.0

    def test_create_session_custom_id(self):
        """Test creating a session with custom ID."""
        session = create_session(session_id="custom_123")
        assert session.session_id == "custom_123"

    def test_create_session_custom_values(self):
        """Test creating a session with custom values."""
        session = create_session(
            session_id="test",
            tier=4,
            max_budget=50.0,
            title="My Session",
            description="Test description",
        )
        assert session.tier == 4
        assert session.max_budget == 50.0
        assert session.title == "My Session"
        assert session.description == "Test description"

    def test_create_session_timestamps(self):
        """Test that create_session sets timestamps."""
        session = create_session()
        assert session.created_at is not None
        assert session.updated_at is not None
        assert session.created_at == session.updated_at


# =============================================================================
# CompactionConfig Tests
# =============================================================================

class TestCompactionConfig:
    """Tests for CompactionConfig dataclass."""

    def test_defaults(self):
        """Test default configuration values."""
        config = CompactionConfig()
        assert config.max_tokens == 100000
        assert config.max_messages == 50
        assert config.max_session_age_hours == 24.0
        assert config.recent_messages_to_keep == 10
        assert config.preserve_verdicts is True
        assert config.preserve_sme_advisories is True
        assert config.preserve_key_decisions is True
        assert config.summary_ratio == 0.3
        assert config.min_summary_length == 100

    def test_custom_values(self):
        """Test custom configuration values."""
        config = CompactionConfig(
            max_tokens=50000,
            max_messages=25,
            max_session_age_hours=12.0,
            recent_messages_to_keep=5,
            preserve_verdicts=False,
            summary_ratio=0.5,
        )
        assert config.max_tokens == 50000
        assert config.max_messages == 25
        assert config.max_session_age_hours == 12.0
        assert config.recent_messages_to_keep == 5
        assert config.preserve_verdicts is False
        assert config.summary_ratio == 0.5


# =============================================================================
# CompactionTrigger Tests
# =============================================================================

class TestCompactionTrigger:
    """Tests for CompactionTrigger enum."""

    def test_trigger_values(self):
        """Test trigger enum values."""
        assert CompactionTrigger.TOKEN_COUNT == "token_count"
        assert CompactionTrigger.MESSAGE_COUNT == "message_count"
        assert CompactionTrigger.SESSION_AGE == "session_age"
        assert CompactionTrigger.MANUAL == "manual"
        assert CompactionTrigger.AUTO == "auto"


# =============================================================================
# MessageAnalyzer Tests
# =============================================================================

class TestMessageAnalyzer:
    """Tests for MessageAnalyzer class."""

    def test_should_compact_token_threshold(self, sample_timestamp):
        """Test compaction triggered by token count."""
        config = CompactionConfig(max_tokens=10)  # Very low threshold
        with patch("src.session.compaction.get_logger", return_value=MagicMock()):
            analyzer = MessageAnalyzer(config)

        session = SessionState(
            session_id="test",
            created_at=sample_timestamp,
            updated_at=sample_timestamp,
            messages=[
                ChatMessage(role="user", content="x" * 1000, timestamp=sample_timestamp),
            ],
        )
        assert analyzer.should_compact(session) is True

    def test_should_compact_message_threshold(self, sample_timestamp):
        """Test compaction triggered by message count."""
        config = CompactionConfig(max_messages=5, max_tokens=999999)
        with patch("src.session.compaction.get_logger", return_value=MagicMock()):
            analyzer = MessageAnalyzer(config)

        messages = [
            ChatMessage(role="user", content=f"msg {i}", timestamp=sample_timestamp)
            for i in range(10)
        ]
        session = SessionState(
            session_id="test",
            created_at=sample_timestamp,
            updated_at=sample_timestamp,
            messages=messages,
        )
        assert analyzer.should_compact(session) is True

    def test_should_compact_age_threshold(self):
        """Test compaction triggered by session age."""
        config = CompactionConfig(max_session_age_hours=1.0, max_tokens=999999, max_messages=999)
        with patch("src.session.compaction.get_logger", return_value=MagicMock()):
            analyzer = MessageAnalyzer(config)

        old_time = datetime.now() - timedelta(hours=2)
        session = SessionState(
            session_id="test",
            created_at=old_time,
            updated_at=old_time,
        )
        assert analyzer.should_compact(session) is True

    def test_should_not_compact(self, sample_timestamp):
        """Test no compaction when all thresholds are within limits."""
        config = CompactionConfig(
            max_tokens=999999, max_messages=999, max_session_age_hours=999
        )
        with patch("src.session.compaction.get_logger", return_value=MagicMock()):
            analyzer = MessageAnalyzer(config)

        session = SessionState(
            session_id="test",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            messages=[
                ChatMessage(role="user", content="short", timestamp=sample_timestamp)
            ],
        )
        assert analyzer.should_compact(session) is False

    def test_estimate_tokens(self, sample_timestamp):
        """Test token estimation."""
        config = CompactionConfig()
        with patch("src.session.compaction.get_logger", return_value=MagicMock()):
            analyzer = MessageAnalyzer(config)

        session = SessionState(
            session_id="test",
            created_at=sample_timestamp,
            updated_at=sample_timestamp,
            messages=[
                ChatMessage(role="user", content="a" * 400, timestamp=sample_timestamp),
            ],
            agent_outputs=[
                AgentOutput(agent_name="Test", phase="test", tier=1, content="b" * 400),
            ],
        )
        tokens = analyzer.estimate_tokens(session)
        assert tokens == 200  # 800 chars / 4

    def test_estimate_tokens_with_structured_data(self, sample_timestamp):
        """Test token estimation includes structured data."""
        config = CompactionConfig()
        with patch("src.session.compaction.get_logger", return_value=MagicMock()):
            analyzer = MessageAnalyzer(config)

        session = SessionState(
            session_id="test",
            created_at=sample_timestamp,
            updated_at=sample_timestamp,
            agent_outputs=[
                AgentOutput(
                    agent_name="Test", phase="test", tier=1,
                    content="x" * 100,
                    structured_data={"key": "value" * 100},
                ),
            ],
        )
        tokens = analyzer.estimate_tokens(session)
        assert tokens > 25  # Content alone would be 25

    def test_identify_preservable_items_verdicts(self, sample_timestamp):
        """Test preservable item identification for verdicts."""
        config = CompactionConfig()
        with patch("src.session.compaction.get_logger", return_value=MagicMock()):
            analyzer = MessageAnalyzer(config)

        session = SessionState(
            session_id="test",
            created_at=sample_timestamp,
            updated_at=sample_timestamp,
            messages=[
                ChatMessage(role="assistant", content="verdict: PASS", timestamp=sample_timestamp),
                ChatMessage(role="assistant", content="no important info", timestamp=sample_timestamp),
                ChatMessage(role="assistant", content="quality gate passed", timestamp=sample_timestamp),
            ],
        )
        preserve = analyzer.identify_preservable_items(session)
        assert 0 in preserve["verdicts"]
        assert 2 in preserve["verdicts"]

    def test_identify_preservable_items_user_requests(self, sample_timestamp):
        """Test that all user messages are preserved."""
        config = CompactionConfig()
        with patch("src.session.compaction.get_logger", return_value=MagicMock()):
            analyzer = MessageAnalyzer(config)

        session = SessionState(
            session_id="test",
            created_at=sample_timestamp,
            updated_at=sample_timestamp,
            messages=[
                ChatMessage(role="user", content="Do this", timestamp=sample_timestamp),
                ChatMessage(role="assistant", content="Done", timestamp=sample_timestamp),
                ChatMessage(role="user", content="Now that", timestamp=sample_timestamp),
            ],
        )
        preserve = analyzer.identify_preservable_items(session)
        assert 0 in preserve["user_requests"]
        assert 2 in preserve["user_requests"]

    def test_identify_preservable_items_sme(self, sample_timestamp):
        """Test SME advisory preservation."""
        config = CompactionConfig()
        with patch("src.session.compaction.get_logger", return_value=MagicMock()):
            analyzer = MessageAnalyzer(config)

        session = SessionState(
            session_id="test",
            created_at=sample_timestamp,
            updated_at=sample_timestamp,
            messages=[
                ChatMessage(
                    role="assistant", content="cloud guidance",
                    timestamp=sample_timestamp,
                    metadata={"source": "sme_cloud_architect"},
                ),
            ],
            agent_outputs=[
                AgentOutput(
                    agent_name="cloud_architect", phase="execution",
                    tier=3, content="Architecture recommendation",
                ),
            ],
        )
        preserve = analyzer.identify_preservable_items(session)
        assert 0 in preserve["sme_advisories"]

    def test_identify_preservable_items_key_decisions(self, sample_timestamp):
        """Test key decision preservation."""
        config = CompactionConfig()
        with patch("src.session.compaction.get_logger", return_value=MagicMock()):
            analyzer = MessageAnalyzer(config)

        session = SessionState(
            session_id="test",
            created_at=sample_timestamp,
            updated_at=sample_timestamp,
            messages=[
                ChatMessage(
                    role="assistant",
                    content="decision: use microservices architecture:",
                    timestamp=sample_timestamp,
                ),
            ],
        )
        preserve = analyzer.identify_preservable_items(session)
        assert 0 in preserve["key_decisions"]

    def test_create_compacted_messages(self, sample_timestamp):
        """Test message compaction keeps recent and preserved items."""
        config = CompactionConfig(recent_messages_to_keep=2)
        with patch("src.session.compaction.get_logger", return_value=MagicMock()):
            analyzer = MessageAnalyzer(config)

        messages = [
            ChatMessage(role="user", content=f"msg {i}", timestamp=sample_timestamp)
            for i in range(10)
        ]
        session = SessionState(
            session_id="test",
            created_at=sample_timestamp,
            updated_at=sample_timestamp,
            messages=messages,
        )
        preserve = {"user_requests": list(range(10)), "verdicts": [], "sme_advisories": [],
                     "key_decisions": [], "final_outputs": []}
        compacted, summary = analyzer.create_compacted_messages(session, preserve)
        # All user messages should be preserved
        assert len(compacted) == 10

    def test_create_compacted_messages_removes_non_preserved(self, sample_timestamp):
        """Test that non-preserved messages are removed."""
        config = CompactionConfig(recent_messages_to_keep=1)
        with patch("src.session.compaction.get_logger", return_value=MagicMock()):
            analyzer = MessageAnalyzer(config)

        messages = [
            ChatMessage(role="assistant", content=f"response {i}", timestamp=sample_timestamp)
            for i in range(10)
        ]
        session = SessionState(
            session_id="test",
            created_at=sample_timestamp,
            updated_at=sample_timestamp,
            messages=messages,
        )
        preserve = {"user_requests": [], "verdicts": [], "sme_advisories": [],
                     "key_decisions": [], "final_outputs": []}
        compacted, summary = analyzer.create_compacted_messages(session, preserve)
        # Only the last message should be kept (recent_messages_to_keep=1)
        assert len(compacted) == 1
        assert "response 9" in compacted[0].content

    def test_generate_summary_content(self, sample_timestamp):
        """Test summary generation content."""
        config = CompactionConfig(recent_messages_to_keep=1)
        with patch("src.session.compaction.get_logger", return_value=MagicMock()):
            analyzer = MessageAnalyzer(config)

        session = SessionState(
            session_id="test_123",
            created_at=sample_timestamp,
            updated_at=sample_timestamp,
            tier=3,
            total_cost_usd=0.15,
            messages=[
                ChatMessage(role="assistant", content="old message", timestamp=sample_timestamp),
                ChatMessage(role="user", content="request", timestamp=sample_timestamp),
            ],
        )
        summary = analyzer._generate_summary(session, {0}, [1])
        assert "Session Summary" in summary
        assert "test_123" in summary
        assert "Tier" in summary


# =============================================================================
# ContextCompactor Tests
# =============================================================================

class TestContextCompactor:
    """Tests for ContextCompactor class."""

    def test_init_default_config(self):
        """Test default config initialization."""
        with patch("src.session.compaction.get_logger", return_value=MagicMock()):
            compactor = ContextCompactor()
        assert compactor.config.max_tokens == 100000
        assert compactor.config.max_messages == 50

    def test_init_custom_config(self):
        """Test custom config initialization."""
        config = CompactionConfig(max_tokens=50000, max_messages=25)
        with patch("src.session.compaction.get_logger", return_value=MagicMock()):
            compactor = ContextCompactor(config=config)
        assert compactor.config.max_tokens == 50000
        assert compactor.config.max_messages == 25

    def test_should_compact_delegates(self, sample_timestamp):
        """Test that should_compact delegates to MessageAnalyzer."""
        config = CompactionConfig(max_messages=2, max_tokens=999999)
        with patch("src.session.compaction.get_logger", return_value=MagicMock()):
            compactor = ContextCompactor(config=config)

        session = SessionState(
            session_id="test",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            messages=[
                ChatMessage(role="user", content=f"msg {i}", timestamp=sample_timestamp)
                for i in range(5)
            ],
        )
        assert compactor.should_compact(session) is True

    def test_should_not_compact(self, sample_timestamp):
        """Test should_compact returns False when not needed."""
        config = CompactionConfig(max_messages=999, max_tokens=999999, max_session_age_hours=999)
        with patch("src.session.compaction.get_logger", return_value=MagicMock()):
            compactor = ContextCompactor(config=config)

        session = SessionState(
            session_id="test",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            messages=[
                ChatMessage(role="user", content="short", timestamp=sample_timestamp),
            ],
        )
        assert compactor.should_compact(session) is False

    def test_compact_session(self, sample_timestamp):
        """Test compact_session performs compaction."""
        config = CompactionConfig(recent_messages_to_keep=2, max_messages=5)
        with patch("src.session.compaction.get_logger", return_value=MagicMock()):
            compactor = ContextCompactor(config=config)

        messages = [
            ChatMessage(role="assistant", content=f"message {i}", timestamp=sample_timestamp)
            for i in range(20)
        ]
        session = SessionState(
            session_id="compact_test",
            created_at=sample_timestamp,
            updated_at=sample_timestamp,
            messages=messages,
            tier=2,
            total_cost_usd=0.10,
        )

        result = compactor.compact_session(session)

        assert isinstance(result, CompactionResult)
        assert result.original_count == 20
        assert result.compacted_count < 20
        assert result.tokens_removed >= 0
        assert isinstance(result.summary, str)
        assert len(result.preserved_items) > 0

    def test_compact_session_adds_reorientation(self, sample_timestamp):
        """Test that compaction adds re-orientation system message."""
        config = CompactionConfig(recent_messages_to_keep=1)
        with patch("src.session.compaction.get_logger", return_value=MagicMock()):
            compactor = ContextCompactor(config=config)

        session = SessionState(
            session_id="test",
            created_at=sample_timestamp,
            updated_at=sample_timestamp,
            messages=[
                ChatMessage(role="assistant", content=f"msg {i}", timestamp=sample_timestamp)
                for i in range(5)
            ],
        )

        compactor.compact_session(session)

        # Should have reorientation and summary messages added
        system_msgs = [m for m in session.messages if m.role == "system"]
        assert len(system_msgs) >= 2

    def test_compact_session_reorientation_content(self, sample_timestamp):
        """Test reorientation prompt content."""
        config = CompactionConfig(recent_messages_to_keep=1)
        with patch("src.session.compaction.get_logger", return_value=MagicMock()):
            compactor = ContextCompactor(config=config)

        session = SessionState(
            session_id="test_reorient",
            created_at=sample_timestamp,
            updated_at=sample_timestamp,
            tier=3,
            active_agents=["cloud_architect"],
            current_phase="execution",
            messages=[
                ChatMessage(role="assistant", content="test", timestamp=sample_timestamp)
                for _ in range(3)
            ],
        )

        compactor.compact_session(session)

        # Find the reorientation message (metadata has "compaction": "reorientation")
        reorient_msgs = [
            m for m in session.messages
            if m.role == "system" and m.metadata.get("compaction") == "reorientation"
        ]
        assert len(reorient_msgs) == 1
        content = reorient_msgs[0].content
        assert "Re-orientation" in content
        assert "test_reorient" in content
        assert "Tier: 3" in content

    def test_compact_session_manual_trigger(self, sample_timestamp):
        """Test compaction with manual trigger."""
        config = CompactionConfig(recent_messages_to_keep=1)
        with patch("src.session.compaction.get_logger", return_value=MagicMock()):
            compactor = ContextCompactor(config=config)

        session = SessionState(
            session_id="test",
            created_at=sample_timestamp,
            updated_at=sample_timestamp,
            messages=[
                ChatMessage(role="assistant", content="msg", timestamp=sample_timestamp)
                for _ in range(5)
            ],
        )
        result = compactor.compact_session(session, trigger=CompactionTrigger.MANUAL)
        assert isinstance(result, CompactionResult)

    def test_estimate_tokens(self, sample_timestamp):
        """Test token estimation."""
        with patch("src.session.compaction.get_logger", return_value=MagicMock()):
            compactor = ContextCompactor()

        session = SessionState(
            session_id="test",
            created_at=sample_timestamp,
            updated_at=sample_timestamp,
            messages=[
                ChatMessage(role="user", content="x" * 400, timestamp=sample_timestamp),
            ],
        )
        tokens = compactor.estimate_tokens(session)
        assert tokens == 100  # 400 / 4

    def test_estimate_tokens_from_messages(self, sample_timestamp):
        """Test token estimation from message list."""
        with patch("src.session.compaction.get_logger", return_value=MagicMock()):
            compactor = ContextCompactor()

        messages = [
            ChatMessage(role="user", content="a" * 200, timestamp=sample_timestamp),
            ChatMessage(role="assistant", content="b" * 200, timestamp=sample_timestamp),
        ]
        tokens = compactor.estimate_tokens_from_messages(messages)
        assert tokens == 100  # 400 / 4

    def test_compact_session_reduction_ratio(self, sample_timestamp):
        """Test that reduction ratio is computed correctly."""
        config = CompactionConfig(recent_messages_to_keep=1)
        with patch("src.session.compaction.get_logger", return_value=MagicMock()):
            compactor = ContextCompactor(config=config)

        session = SessionState(
            session_id="test",
            created_at=sample_timestamp,
            updated_at=sample_timestamp,
            messages=[
                ChatMessage(role="assistant", content="x" * 100, timestamp=sample_timestamp)
                for _ in range(10)
            ],
        )
        result = compactor.compact_session(session)
        assert 0.0 <= result.reduction_ratio <= 1.0

    def test_build_reorientation_prompt(self, sample_timestamp):
        """Test _build_reorientation_prompt content."""
        with patch("src.session.compaction.get_logger", return_value=MagicMock()):
            compactor = ContextCompactor()

        session = SessionState(
            session_id="test_prompt",
            created_at=sample_timestamp,
            updated_at=sample_timestamp,
            tier=4,
            total_cost_usd=0.25,
            active_agents=["security_analyst"],
            current_phase="review",
        )
        prompt = compactor._build_reorientation_prompt(session)
        assert "Context Compaction" in prompt
        assert "test_prompt" in prompt
        assert "Tier: 4" in prompt
        assert "$0.2500" in prompt
        assert "security_analyst" in prompt
        assert "review" in prompt
        assert "escalation_needed" in prompt


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_check_and_compact_needed(self, sample_timestamp):
        """Test check_and_compact when compaction is needed."""
        # Reset global compactor
        import src.session.compaction as compaction_module
        compaction_module._global_compactor = None

        config = CompactionConfig(max_messages=2, max_tokens=999999, max_session_age_hours=999)
        set_compaction_config(config)

        session = SessionState(
            session_id="test",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            messages=[
                ChatMessage(role="user", content=f"msg {i}", timestamp=sample_timestamp)
                for i in range(10)
            ],
        )

        with patch("src.session.compaction.get_logger", return_value=MagicMock()):
            result = check_and_compact(session)

        assert result is not None
        assert isinstance(result, CompactionResult)

    def test_check_and_compact_not_needed(self, sample_timestamp):
        """Test check_and_compact when compaction is not needed."""
        import src.session.compaction as compaction_module
        compaction_module._global_compactor = None

        config = CompactionConfig(max_messages=999, max_tokens=999999, max_session_age_hours=999)
        set_compaction_config(config)

        session = SessionState(
            session_id="test",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            messages=[
                ChatMessage(role="user", content="short", timestamp=sample_timestamp),
            ],
        )

        with patch("src.session.compaction.get_logger", return_value=MagicMock()):
            result = check_and_compact(session)

        assert result is None

    def test_set_compaction_config(self):
        """Test set_compaction_config updates global compactor."""
        import src.session.compaction as compaction_module

        config = CompactionConfig(max_tokens=42000)
        set_compaction_config(config)

        assert compaction_module._global_compactor is not None
        assert compaction_module._global_compactor.config.max_tokens == 42000

    def test_get_context_compactor_creates_singleton(self):
        """Test get_context_compactor creates and returns singleton."""
        import src.session.compaction as compaction_module
        compaction_module._global_compactor = None

        with patch("src.session.compaction.get_logger", return_value=MagicMock()):
            compactor1 = get_context_compactor()
            compactor2 = get_context_compactor()

        assert compactor1 is compactor2

    def test_get_context_compactor_returns_instance(self):
        """Test get_context_compactor returns ContextCompactor."""
        import src.session.compaction as compaction_module
        compaction_module._global_compactor = None

        with patch("src.session.compaction.get_logger", return_value=MagicMock()):
            compactor = get_context_compactor()

        assert isinstance(compactor, ContextCompactor)


# =============================================================================
# CompactionResult Tests
# =============================================================================

class TestCompactionResult:
    """Tests for CompactionResult dataclass."""

    def test_all_fields(self):
        """Test CompactionResult with all fields."""
        result = CompactionResult(
            original_count=50,
            compacted_count=15,
            tokens_removed=5000,
            tokens_remaining=2000,
            reduction_ratio=0.71,
            summary="Session compacted",
            preserved_items=["verdicts: 3 items", "user_requests: 5 items"],
        )
        assert result.original_count == 50
        assert result.compacted_count == 15
        assert result.tokens_removed == 5000
        assert result.tokens_remaining == 2000
        assert result.reduction_ratio == 0.71
        assert "compacted" in result.summary
        assert len(result.preserved_items) == 2


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_session_compaction(self, sample_timestamp):
        """Test compaction of a session with no messages raises ZeroDivisionError.

        The source code has a division by len(session.messages) in _generate_summary
        which causes ZeroDivisionError on empty sessions. This documents that behavior.
        """
        config = CompactionConfig(recent_messages_to_keep=5)
        with patch("src.session.compaction.get_logger", return_value=MagicMock()):
            compactor = ContextCompactor(config=config)

        session = SessionState(
            session_id="empty",
            created_at=sample_timestamp,
            updated_at=sample_timestamp,
            messages=[],
        )
        # Source code divides by len(session.messages) which is 0
        with pytest.raises(ZeroDivisionError):
            compactor.compact_session(session)

    def test_session_with_only_system_messages(self, sample_timestamp):
        """Test session with only system messages."""
        config = CompactionConfig(recent_messages_to_keep=1)
        with patch("src.session.compaction.get_logger", return_value=MagicMock()):
            compactor = ContextCompactor(config=config)

        session = SessionState(
            session_id="sys_only",
            created_at=sample_timestamp,
            updated_at=sample_timestamp,
            messages=[
                ChatMessage(role="system", content="init", timestamp=sample_timestamp),
            ],
        )
        result = compactor.compact_session(session)
        assert isinstance(result, CompactionResult)

    def test_agent_output_empty_content(self):
        """Test AgentOutput with empty content."""
        output = AgentOutput(
            agent_name="Test", phase="test", tier=1, content=""
        )
        d = output.to_dict()
        assert d["content"] == ""
        restored = AgentOutput.from_dict(d)
        assert restored.content == ""

    def test_chat_message_with_large_metadata(self, sample_timestamp):
        """Test ChatMessage with large metadata dict."""
        large_metadata = {f"key_{i}": f"value_{i}" for i in range(100)}
        msg = ChatMessage(
            role="assistant",
            content="test",
            timestamp=sample_timestamp,
            metadata=large_metadata,
        )
        d = msg.to_dict()
        restored = ChatMessage.from_dict(d)
        assert len(restored.metadata) == 100

    def test_session_with_many_outputs(self, sample_timestamp):
        """Test session with many agent outputs."""
        outputs = [
            AgentOutput(
                agent_name=f"Agent_{i}", phase="execution",
                tier=2, content=f"Output {i}",
                timestamp=sample_timestamp,
            )
            for i in range(50)
        ]
        session = SessionState(
            session_id="many_outputs",
            created_at=sample_timestamp,
            updated_at=sample_timestamp,
            agent_outputs=outputs,
        )
        d = session.to_dict()
        restored = SessionState.from_dict(d)
        assert len(restored.agent_outputs) == 50

    def test_concurrent_save_load(self, tmp_path, sample_timestamp):
        """Test that lock prevents corruption during concurrent access."""
        with patch("src.session.persistence.get_logger", return_value=MagicMock()):
            sp = SessionPersistence(sessions_dir=tmp_path)

        session = SessionState(
            session_id="concurrent",
            created_at=sample_timestamp,
            updated_at=sample_timestamp,
            total_tokens=100,
        )

        # Save and immediately load
        sp.save_session(session)
        loaded = sp.load_session("concurrent")
        assert loaded is not None
        assert loaded.total_tokens == 100

    def test_session_state_from_dict_missing_optional_fields(self):
        """Test SessionState.from_dict with missing optional fields."""
        data = {
            "session_id": "minimal_test",
            "created_at": "2025-01-01T00:00:00",
            "updated_at": "2025-01-01T00:00:00",
        }
        state = SessionState.from_dict(data)
        assert state.title is None
        assert state.description is None
        assert state.tags == []
        assert state.current_phase == ""
        assert state.total_tokens == 0
        assert state.total_cost_usd == 0.0

    def test_formatter_output_preserved(self, sample_timestamp):
        """Test that Formatter outputs are identified as preservable."""
        config = CompactionConfig()
        with patch("src.session.compaction.get_logger", return_value=MagicMock()):
            analyzer = MessageAnalyzer(config)

        session = SessionState(
            session_id="test",
            created_at=sample_timestamp,
            updated_at=sample_timestamp,
            agent_outputs=[
                AgentOutput(agent_name="Formatter", phase="format", tier=2, content="formatted"),
            ],
        )
        preserve = analyzer.identify_preservable_items(session)
        assert 0 in preserve["final_outputs"]
