"""
Multi-Agent Reasoning System

A sophisticated multi-agent reasoning system using Claude Agent SDK.
Implements three-tier architecture with Council, Operational Agents,
and dynamic SME personas.
"""

__version__ = "0.1.0"

# Main entry point
from src.agents.orchestrator import (
    OrchestratorAgent,
    create_orchestrator,
)

# Session management
from src.session import (
    SessionState,
    SessionPersistence,
    ContextCompactor,
    create_session,
    resume_session,
    save_session,
    check_and_compact,
)

__all__ = [
    # Main
    "OrchestratorAgent",
    "create_orchestrator",
    # Session
    "SessionState",
    "SessionPersistence",
    "ContextCompactor",
    "create_session",
    "resume_session",
    "save_session",
    "check_and_compact",
]
