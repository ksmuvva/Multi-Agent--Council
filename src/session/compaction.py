"""
Context Compaction - Reduce Conversation History for Long Sessions

Implements intelligent compaction of conversation history to stay within
token limits while preserving important information.
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

from src.utils.logging import get_logger, bind_session
from .persistence import SessionState, ChatMessage, AgentOutput


# =============================================================================
# Compaction Configuration
# =============================================================================

class CompactionTrigger(str, Enum):
    """Triggers for context compaction."""
    TOKEN_COUNT = "token_count"  # Total token limit exceeded
    MESSAGE_COUNT = "message_count"  # Message count threshold
    SESSION_AGE = "session_age"  # Session age threshold
    MANUAL = "manual"  # User requested
    AUTO = "auto"  # Automatic compaction


@dataclass
class CompactionConfig:
    """Configuration for context compaction."""
    # Triggers
    max_tokens: int = 100000  # Trigger compaction at this token count
    max_messages: int = 50  # Trigger compaction at this message count
    max_session_age_hours: float = 24.0  # Trigger compaction for old sessions

    # Compaction behavior
    recent_messages_to_keep: int = 10  # Always keep last N messages
    preserve_verdicts: bool = True  # Always preserve quality verdicts
    preserve_sme_advisories: bool = True  # Always preserve SME outputs
    preserve_key_decisions: bool = True  # Always preserve key decisions

    # Summary settings
    summary_ratio: float = 0.3  # Target size: compacted / original
    min_summary_length: int = 100  # Minimum summary length


# =============================================================================
# Compaction Result
# =============================================================================

@dataclass
class CompactionResult:
    """Result of a context compaction operation."""
    original_count: int
    compacted_count: int
    tokens_removed: int
    tokens_remaining: int
    reduction_ratio: float
    summary: str
    preserved_items: List[str]  # Descriptions of what was preserved


# =============================================================================
# Message Analysis
# =============================================================================

class MessageAnalyzer:
    """Analyzes messages to determine what to preserve."""

    def __init__(self, config: CompactionConfig):
        """
        Initialize analyzer.

        Args:
            config: Compaction configuration
        """
        self.config = config
        self._logger = get_logger("message_analyzer")

    def should_compact(self, session: SessionState) -> bool:
        """
        Determine if compaction is needed.

        Args:
            session: Session to analyze

        Returns:
            True if compaction should be triggered
        """
        # Check token count (estimated)
        estimated_tokens = self.estimate_tokens(session)

        if estimated_tokens > self.config.max_tokens:
            self._logger.info(
                "compaction_triggered_tokens",
                tokens=estimated_tokens,
                threshold=self.config.max_tokens,
            )
            return True

        # Check message count
        if len(session.messages) > self.config.max_messages:
            self._logger.info(
                "compaction_triggered_messages",
                messages=len(session.messages),
                threshold=self.config.max_messages,
            )
            return True

        # Check session age
        session_age = datetime.now() - session.created_at
        if session_age > timedelta(hours=self.config.max_session_age_hours):
            self._logger.info(
                "compaction_triggered_age",
                age_hours=session_age.total_seconds() / 3600,
                threshold=self.config.max_session_age_hours,
            )
            return True

        return False

    def estimate_tokens(self, session: SessionState) -> int:
        """
        Estimate total tokens in session.

        Args:
            session: Session to analyze

        Returns:
            Estimated token count
        """
        # Rough estimate: ~4 characters per token
        # Count message content and agent outputs
        total_chars = 0

        for message in session.messages:
            total_chars += len(message.content)

        for output in session.agent_outputs:
            total_chars += len(output.content)
            if output.structured_data:
                total_chars += len(str(output.structured_data)) // 2  # Rough estimate

        return total_chars // 4

    def identify_preservable_items(self, session: SessionState) -> Dict[str, List[int]]:
        """
        Identify items that should be preserved during compaction.

        Args:
            session: Session to analyze

        Returns:
            Dictionary mapping category to list of indices
        """
        preserve = {
            "verdicts": [],
            "sme_advisories": [],
            "key_decisions": [],
            "user_requests": [],
            "final_outputs": [],
        }

        for i, message in enumerate(session.messages):
            # Check for verdicts
            if "verdict" in message.content.lower() or "quality gate" in message.content.lower():
                preserve["verdicts"].append(i)

            # Check for SME advisories
            if "sme" in message.metadata.get("source", "").lower():
                preserve["sme_advisories"].append(i)

            # Check for key decisions
            decision_keywords = ["decision:", "approach:", "selected:", "architecture:"]
            if any(kw in message.content.lower() for kw in decision_keywords):
                preserve["key_decisions"].append(i)

            # User requests always preserved
            if message.role == "user":
                preserve["user_requests"].append(i)

        # Also check agent outputs
        for i, output in enumerate(session.agent_outputs):
            # SME outputs
            if output.agent_name in ["cloud_architect", "security_analyst", "iam_architect"]:
                preserve["sme_advisories"].append(i)

            # Final outputs (Formatter)
            if output.agent_name == "Formatter":
                preserve["final_outputs"].append(i)

        return preserve

    def create_compacted_messages(
        self,
        session: SessionState,
        preserve: Dict[str, List[int]],
    ) -> Tuple[List[ChatMessage], str]:
        """
        Create compacted message list.

        Args:
            session: Original session
            preserve: Indices of items to preserve

        Returns:
            Tuple of (compacted messages, summary text)
        """
        # Always keep recent messages
        recent_start = max(0, len(session.messages) - self.config.recent_messages_to_keep)

        # Collect messages to preserve (in order)
        preserved_indices = set()
        for category, indices in preserve.items():
            preserved_indices.update(indices)

        # Add recent messages
        for i in range(recent_start, len(session.messages)):
            preserved_indices.add(i)

        # Sort and create compacted messages list
        sorted_indices = sorted(preserved_indices)
        compacted = [session.messages[i] for i in sorted_indices]

        # Generate summary of removed messages
        removed_indices = set(range(len(session.messages))) - preserved_indices
        summary = self._generate_summary(session, removed_indices, sorted_indices)

        return compacted, summary

    def _generate_summary(
        self,
        session: SessionState,
        removed_indices: set,
        preserved_indices: List[int],
    ) -> str:
        """
        Generate a summary of removed messages.

        Args:
            session: Session state
            removed_indices: Indices of removed messages
            preserved_indices: Indices of preserved messages

        Returns:
            Summary text
        """
        summary_parts = [
            f"# Session Summary",
            f"",
            f"**Session ID**: {session.session_id}",
            f"**Created**: {session.created_at.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Tier**: {session.tier}",
            f"**Total Cost**: ${session.total_cost_usd:.4f}",
            f"**Messages Compacted**: {len(removed_indices)} → {len(preserved_indices)}",
            f"**Reduction**: {100 - (100 * len(preserved_indices) / len(session.messages)):.1f}%",
        ]

        # Key preserved items
        summary_parts.extend([
            f"",
            f"## Preserved Information",
            f"",
        ])

        # Add agent outputs summary
        if session.agent_outputs:
            summary_parts.append("### Agent Outputs")
            unique_agents = set(output.agent_name for output in session.agent_outputs)
            for agent in sorted(unique_agents):
                agent_outputs = [o for o in session.agent_outputs if o.agent_name == agent]
                if agent_outputs:
                    latest = agent_outputs[-1]
                    summary_parts.append(f"- **{agent}** ({len(agent_outputs)} calls) - Latest: {latest.phase}")

        # Add key findings
        summary_parts.extend([
            f"",
            f"## Compacted Content",
            f"",
            f"The following {len(removed_indices)} messages were compacted:",
            f"",
        ])

        # Categorize removed messages
        removed_by_role = defaultdict(list)
        for idx in removed_indices:
            msg = session.messages[idx]
            removed_by_role[msg.role].append(idx)

        for role, indices in removed_by_role.items():
            count = len(indices)
            if role == "assistant":
                role = "agent"
            summary_parts.append(f"- {count} {role} message(s)")

        return "\n".join(summary_parts)


# =============================================================================
# Context Compactor
# =============================================================================

class ContextCompactor:
    """
    Compacts conversation context to stay within token limits.

    Implements intelligent compaction that preserves important information
    while reducing overall token count.
    """

    def __init__(self, config: Optional[CompactionConfig] = None):
        """
        Initialize compactor.

        Args:
            config: Compaction configuration
        """
        self.config = config or CompactionConfig()
        self._logger = get_logger("context_compactor")

    def should_compact(self, session: SessionState) -> bool:
        """
        Check if session should be compacted.

        Args:
            session: Session to check

        Returns:
            True if compaction needed
        """
        analyzer = MessageAnalyzer(self.config)
        return analyzer.should_compact(session)

    def compact_session(
        self,
        session: SessionState,
        trigger: CompactionTrigger = CompactionTrigger.AUTO,
    ) -> CompactionResult:
        """
        Compact a session's context.

        Args:
            session: Session to compact
            trigger: What triggered compaction

        Returns:
            Compaction result
        """
        self._logger.info(
            "compaction_started",
            session_id=session.session_id,
            trigger=trigger.value if trigger != CompactionTrigger.AUTO else "auto",
        )

        original_count = len(session.messages)
        original_tokens = self.estimate_tokens(session)

        # Analyze and identify what to preserve
        analyzer = MessageAnalyzer(self.config)
        preserve = analyzer.identify_preservable_items(session)

        # Create compacted messages
        compacted_messages, summary = analyzer.create_compacted_messages(
            session, preserve
        )

        # Calculate result
        compacted_tokens = self.estimate_tokens_from_messages(compacted_messages)
        tokens_removed = original_tokens - compacted_tokens

        result = CompactionResult(
            original_count=original_count,
            compacted_count=len(compacted_messages),
            tokens_removed=tokens_removed,
            tokens_remaining=compacted_tokens,
            reduction_ratio=1.0 - (compacted_tokens / max(original_tokens, 1)),
            summary=summary,
            preserved_items=[
                f"{category}: {len(indices)} items"
                for category, indices in preserve.items()
            ],
        )

        # Update session (create new instance to avoid mutation issues)
        session.messages = compacted_messages
        session.updated_at = datetime.now()

        # Re-read CLAUDE.md for re-orientation post-compaction
        reorientation = self._build_reorientation_prompt(session)

        # Add re-orientation as a system message
        reorientation_message = ChatMessage(
            role="system",
            content=reorientation,
            timestamp=datetime.now(),
            metadata={"compaction": "reorientation", "auto_compaction": True},
        )
        session.messages.insert(0, reorientation_message)

        # Add summary as a system message
        summary_message = ChatMessage(
            role="system",
            content=summary,
            timestamp=datetime.now(),
            metadata={"compaction": "true", "tokens_removed": tokens_removed},
        )
        session.messages.insert(1, summary_message)

        self._logger.info(
            "compaction_complete",
            session_id=session.session_id,
            original_count=original_count,
            compacted_count=len(compacted_messages),
            tokens_removed=tokens_removed,
            reduction_ratio=result.reduction_ratio,
        )

        return result

    def _build_reorientation_prompt(self, session: SessionState) -> str:
        """
        Build a re-orientation prompt by re-reading CLAUDE.md files.

        This ensures agents can recover context after compaction.
        """
        reorientation_parts = [
            "# Context Compaction - Re-orientation",
            "",
            "Your context has been compacted. Here is your re-orientation:",
            "",
        ]

        # Re-read global CLAUDE.md
        try:
            from pathlib import Path
            claude_md = Path("CLAUDE.md")
            if claude_md.exists():
                content = claude_md.read_text(encoding="utf-8")
                # Extract key sections
                reorientation_parts.append("## System Overview (from CLAUDE.md)")
                reorientation_parts.append("")
                # Get first 500 chars of key info
                for line in content.split("\n"):
                    if line.startswith("## ") or line.startswith("# "):
                        reorientation_parts.append(line)
                    elif "Re-orientation" in line or "re-read" in line.lower():
                        reorientation_parts.append(line)
        except Exception:
            pass

        # Add session state
        reorientation_parts.extend([
            "",
            "## Current Session State",
            f"- Session ID: {session.session_id}",
            f"- Tier: {session.tier}",
            f"- Total Cost: ${session.total_cost_usd:.4f}",
            f"- Active Agents: {', '.join(session.active_agents) if session.active_agents else 'None'}",
            f"- Current Phase: {session.current_phase or 'Unknown'}",
            "",
            "## Instructions",
            "- Your output must conform to your assigned Pydantic schema",
            "- Do NOT spawn subagents - only the Orchestrator can do that",
            "- Return results via structured output",
            "- Set `escalation_needed: true` if the task exceeds your capability",
        ])

        return "\n".join(reorientation_parts)

    def estimate_tokens(self, session: SessionState) -> int:
        """Estimate tokens in session."""
        analyzer = MessageAnalyzer(self.config)
        return analyzer.estimate_tokens(session)

    def estimate_tokens_from_messages(self, messages: List[ChatMessage]) -> int:
        """Estimate tokens from a list of messages."""
        total_chars = sum(len(m.content) for m in messages)
        return total_chars // 4


# =============================================================================
# Global Instance
# =============================================================================

_global_compactor: Optional[ContextCompactor] = None


def get_context_compactor() -> ContextCompactor:
    """Get the global context compactor instance."""
    global _global_compactor

    if _global_compactor is None:
        _global_compactor = ContextCompactor()

    return _global_compactor


# =============================================================================
# Convenience Functions
# =============================================================================

def check_and_compact(session: SessionState) -> Optional[CompactionResult]:
    """
    Check if session needs compaction and perform if needed.

    Args:
        session: Session to check

    Returns:
        CompactionResult if compacted, None otherwise
    """
    compactor = get_context_compactor()

    if compactor.should_compact(session):
        return compactor.compact_session(session)

    return None


def compact_session_manual(session_id: str) -> Optional[CompactionResult]:
    """
    Manually trigger compaction for a session.

    Args:
        session_id: Session to compact

    Returns:
        CompactionResult if found and compacted, None otherwise
    """
    from .persistence import resume_session

    session = resume_session(session_id)
    if session is None:
        return None

    compactor = get_context_compactor()
    return compactor.compact_session(session, trigger=CompactionTrigger.MANUAL)


def set_compaction_config(config: CompactionConfig) -> None:
    """
    Update global compaction configuration.

    Args:
        config: New configuration
    """
    global _global_compactor
    _global_compactor = ContextCompactor(config)
