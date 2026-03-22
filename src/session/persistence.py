"""
Session Persistence - Save and Load Session State

Handles saving and loading of session state including messages,
agent outputs, tier classification, and cost tracking.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from threading import Lock

from src.utils.logging import get_logger


# =============================================================================
# Session Data Structures
# =============================================================================

@dataclass
class AgentOutput:
    """Output from a single agent execution."""
    agent_name: str
    phase: str
    tier: int
    content: str
    structured_data: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None
    duration_seconds: float = 0.0
    token_usage: Optional[Dict[str, int]] = None
    status: str = "completed"  # completed, failed, timeout

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "agent_name": self.agent_name,
            "phase": self.phase,
            "tier": self.tier,
            "content": self.content,
            "structured_data": self.structured_data,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "duration_seconds": self.duration_seconds,
            "token_usage": self.token_usage,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentOutput":
        """Create from dictionary."""
        timestamp = data.get("timestamp")
        if timestamp:
            timestamp = datetime.fromisoformat(timestamp)

        return cls(
            agent_name=data["agent_name"],
            phase=data["phase"],
            tier=data["tier"],
            content=data["content"],
            structured_data=data.get("structured_data"),
            timestamp=timestamp,
            duration_seconds=data.get("duration_seconds", 0.0),
            token_usage=data.get("token_usage"),
            status=data.get("status", "completed"),
        )


@dataclass
class ChatMessage:
    """A chat message in the session."""
    role: str  # user, assistant, system
    content: str
    timestamp: datetime
    agent_name: Optional[str] = None
    tier: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "agent_name": self.agent_name,
            "tier": self.tier,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatMessage":
        """Create from dictionary."""
        timestamp = datetime.fromisoformat(data["timestamp"])

        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=timestamp,
            agent_name=data.get("agent_name"),
            tier=data.get("tier"),
            # Ensure metadata is always a dict, even if stored as False or None
            metadata=data.get("metadata", {}) or {},
        )


@dataclass
class SessionSummary:
    """Summary of a session for quick loading."""
    session_id: str
    created_at: datetime
    updated_at: datetime
    tier: int
    total_messages: int
    total_agent_outputs: int
    total_tokens: int
    total_cost_usd: float
    title: Optional[str] = None
    description: Optional[str] = None


@dataclass
class SessionState:
    """Complete session state."""
    session_id: str
    created_at: datetime
    updated_at: datetime

    # Configuration
    tier: int = 2
    max_budget: float = 10.0
    default_format: str = "markdown"

    # Content
    messages: List[ChatMessage] = field(default_factory=list)
    agent_outputs: List[AgentOutput] = field(default_factory=list)

    # Tracking
    active_agents: List[str] = field(default_factory=list)
    current_phase: str = ""

    # Cost tracking
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    daily_budget_usd: float = 50.0

    # Metadata
    title: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    @property
    def max_budget_usd(self) -> float:
        """Alias for max_budget for compatibility with orchestrator code."""
        return self.max_budget

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "tier": self.tier,
            "max_budget": self.max_budget,
            "default_format": self.default_format,
            "messages": [m.to_dict() for m in self.messages],
            "agent_outputs": [o.to_dict() for o in self.agent_outputs],
            "active_agents": self.active_agents,
            "current_phase": self.current_phase,
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost_usd,
            "daily_budget_usd": self.daily_budget_usd,
            "title": self.title,
            "description": self.description,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionState":
        """Create from dictionary."""
        created_at = datetime.fromisoformat(data["created_at"])
        updated_at = datetime.fromisoformat(data["updated_at"])

        messages = [
            ChatMessage.from_dict(m) for m in data.get("messages", [])
        ]
        agent_outputs = [
            AgentOutput.from_dict(o) for o in data.get("agent_outputs", [])
        ]

        return cls(
            session_id=data["session_id"],
            created_at=created_at,
            updated_at=updated_at,
            tier=data.get("tier", 2),
            max_budget=data.get("max_budget") or data.get("max_budget_usd", 10.0),
            default_format=data.get("default_format", "markdown"),
            messages=messages,
            agent_outputs=agent_outputs,
            active_agents=data.get("active_agents", []),
            current_phase=data.get("current_phase", ""),
            total_tokens=data.get("total_tokens", 0),
            total_cost_usd=data.get("total_cost_usd", 0.0),
            daily_budget_usd=data.get("daily_budget_usd", 50.0),
            title=data.get("title"),
            description=data.get("description"),
            tags=data.get("tags", []),
        )

    def get_summary(self) -> SessionSummary:
        """Get a summary of the session."""
        return SessionSummary(
            session_id=self.session_id,
            created_at=self.created_at,
            updated_at=self.updated_at,
            tier=self.tier,
            total_messages=len(self.messages),
            total_agent_outputs=len(self.agent_outputs),
            total_tokens=self.total_tokens,
            total_cost_usd=self.total_cost_usd,
            title=self.title,
            description=self.description,
        )


# =============================================================================
# Session Persistence
# =============================================================================

class SessionPersistence:
    """
    Handles saving and loading session state.

    Sessions are stored in .claude/sessions/{session_id}/session.json
    """

    def __init__(self, sessions_dir: Optional[Path] = None):
        """
        Initialize session persistence.

        Args:
            sessions_dir: Directory for storing sessions
        """
        if sessions_dir is None:
            # Default to .claude/sessions/
            sessions_dir = Path.cwd() / ".claude" / "sessions"

        self.sessions_dir = sessions_dir
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

        self._lock = Lock()
        self._logger = get_logger("session_persistence")

    def get_session_path(self, session_id: str) -> Path:
        """Get the file path for a session."""
        return self.sessions_dir / session_id / "session.json"

    def save_session(self, session: SessionState) -> None:
        """
        Save session state to disk.

        Args:
            session: Session state to save
        """
        with self._lock:
            session_path = self.get_session_path(session.session_id)
            session_path.parent.mkdir(parents=True, exist_ok=True)

            # Update timestamp
            session.updated_at = datetime.now()

            # Convert to JSON
            data = session.to_dict()

            # Write to file
            with open(session_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)

            self._logger.info(
                "session_saved",
                session_id=session.session_id,
                path=str(session_path),
                messages=len(session.messages),
                outputs=len(session.agent_outputs),
            )

    def load_session(self, session_id: str) -> Optional[SessionState]:
        """
        Load session state from disk.

        Args:
            session_id: Session ID to load

        Returns:
            SessionState if found, None otherwise
        """
        with self._lock:
            session_path = self.get_session_path(session_id)

            if not session_path.exists():
                self._logger.warning("session_not_found", session_id=session_id)
                return None

            try:
                with open(session_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                session = SessionState.from_dict(data)

                self._logger.info(
                    "session_loaded",
                    session_id=session_id,
                    messages=len(session.messages),
                    outputs=len(session.agent_outputs),
                )

                return session

            except Exception as e:
                self._logger.error(
                    "session_load_error",
                    session_id=session_id,
                    error=str(e),
                )
                return None

    def list_sessions(
        self,
        limit: int = 50,
        sort_by: str = "updated_at",
        descending: bool = True,
    ) -> List[SessionSummary]:
        """
        List available sessions.

        Args:
            limit: Maximum sessions to return
            sort_by: Field to sort by (created_at, updated_at, title)
            descending: Sort order

        Returns:
            List of session summaries
        """
        summaries = []

        with self._lock:
            # Find all session directories
            for session_dir in self.sessions_dir.iterdir():
                if not session_dir.is_dir():
                    continue

                session_id = session_dir.name
                session_path = session_dir / "session.json"

                if not session_path.exists():
                    continue

                try:
                    with open(session_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    # Create summary
                    created_at = datetime.fromisoformat(data["created_at"])
                    updated_at = datetime.fromisoformat(data["updated_at"])

                    summary = SessionSummary(
                        session_id=session_id,
                        created_at=created_at,
                        updated_at=updated_at,
                        tier=data.get("tier", 2),
                        total_messages=len(data.get("messages", [])),
                        total_agent_outputs=len(data.get("agent_outputs", [])),
                        total_tokens=data.get("total_tokens", 0),
                        total_cost_usd=data.get("total_cost_usd", 0.0),
                        title=data.get("title"),
                        description=data.get("description"),
                    )

                    summaries.append(summary)

                except Exception as e:
                    self._logger.warning(
                        "session_summary_error",
                        session_id=session_id,
                        error=str(e),
                    )

        # Sort
        reverse = descending
        summaries.sort(key=lambda s: getattr(s, sort_by, datetime.min), reverse=reverse)

        return summaries[:limit]

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.

        Args:
            session_id: Session ID to delete

        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            session_path = self.get_session_path(session_id)

            if not session_path.exists():
                return False

            # Delete session directory
            import shutil
            shutil.rmtree(session_path.parent)

            self._logger.info("session_deleted", session_id=session_id)

            return True

    def delete_all_sessions(self) -> int:
        """
        Delete all sessions.

        Returns:
            Number of sessions deleted
        """
        count = 0

        with self._lock:
            for session_dir in self.sessions_dir.iterdir():
                if session_dir.is_dir():
                    import shutil
                    shutil.rmtree(session_dir)
                    count += 1

        self._logger.info("all_sessions_deleted", count=count)

        return count


# =============================================================================
# Global Instance
# =============================================================================

_global_persistence: Optional[SessionPersistence] = None


def get_session_persistence() -> SessionPersistence:
    """Get the global session persistence instance."""
    global _global_persistence

    if _global_persistence is None:
        _global_persistence = SessionPersistence()

    return _global_persistence


# =============================================================================
# Session Factory
# =============================================================================

def create_session(
    session_id: Optional[str] = None,
    tier: int = 2,
    max_budget: float = 10.0,
    title: Optional[str] = None,
    description: Optional[str] = None,
) -> SessionState:
    """
    Create a new session.

    Args:
        session_id: Optional session ID (generated if not provided)
        tier: Default tier for this session
        max_budget: Maximum budget for this session
        title: Optional title
        description: Optional description

    Returns:
        New SessionState instance
    """
    if session_id is None:
        session_id = f"sess_{int(time.time() * 1000000)}"

    now = datetime.now()

    return SessionState(
        session_id=session_id,
        created_at=now,
        updated_at=now,
        tier=tier,
        max_budget=max_budget,
        title=title,
        description=description,
    )


def resume_session(session_id: str) -> Optional[SessionState]:
    """
    Resume an existing session.

    Args:
        session_id: Session ID to resume

    Returns:
        SessionState if found, None otherwise
    """
    persistence = get_session_persistence()
    return persistence.load_session(session_id)


def save_session(session: SessionState) -> None:
    """
    Save a session to disk.

    Args:
        session: Session state to save
    """
    persistence = get_session_persistence()
    persistence.save_session(session)
