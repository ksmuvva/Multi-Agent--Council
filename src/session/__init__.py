"""
Session Management Module

Provides session persistence and context compaction for long-running
multi-agent conversations.

Main exports:
- SessionState: Complete session state data structure
- ChatMessage: Individual chat message
- AgentOutput: Output from agent execution
- SessionPersistence: Save/load sessions
- ContextCompactor: Compact conversation context
- check_and_compact: Convenience function for auto-compaction
- create_session: Factory for new sessions
- resume_session: Resume from persisted session
- save_session: Persist session to disk
"""

from .persistence import (
    AgentOutput,
    ChatMessage,
    SessionSummary,
    SessionState,
    SessionPersistence,
    create_session,
    resume_session,
    save_session,
)

from .compaction import (
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

__all__ = [
    # Persistence
    "AgentOutput",
    "ChatMessage",
    "SessionSummary",
    "SessionState",
    "SessionPersistence",
    "create_session",
    "resume_session",
    "save_session",
    # Compaction
    "CompactionConfig",
    "CompactionResult",
    "CompactionTrigger",
    "ContextCompactor",
    "MessageAnalyzer",
    "check_and_compact",
    "compact_session_manual",
    "get_context_compactor",
    "set_compaction_config",
]
