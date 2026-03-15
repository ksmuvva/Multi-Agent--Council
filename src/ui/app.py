"""
Multi-Agent Reasoning System - Streamlit UI

Main application entry point for the web-based interface.
Provides interactive access to the multi-agent system with real-time updates.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

import streamlit as st
from streamlit_option_menu import option_menu

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.logging import configure_logging, get_logger
from src.utils.events import (
    EventType,
    get_event_emitter,
    get_event_streamer,
    format_sse_event,
)

_app_logger = get_logger("streamlit_app")


# =============================================================================
# Configuration
# =============================================================================

VERSION = "0.1.0"
APP_TITLE = "Multi-Agent Reasoning System"
APP_ICON = "🤖"

# Page definitions
PAGES = {
    "chat": {
        "title": "Chat",
        "icon": "💬",
        "description": "Interactive chat with the multi-agent system",
    },
    "agents": {
        "title": "Agent Activity",
        "icon": "⚡",
        "description": "Real-time agent activity monitoring",
    },
    "results": {
        "title": "Results",
        "icon": "📊",
        "description": "Browse and inspect past results",
    },
    "skills": {
        "title": "Skills",
        "icon": "🎯",
        "description": "Browse available agent skills",
    },
    "personas": {
        "title": "SME Personas",
        "icon": "👤",
        "description": "Subject matter expert registry",
    },
    "ensembles": {
        "title": "Ensembles",
        "icon": "🎭",
        "description": "Pre-configured agent workflows",
    },
    "debates": {
        "title": "Debates",
        "icon": "⚖️",
        "description": "Self-play debate transcripts",
    },
    "knowledge": {
        "title": "Knowledge Base",
        "icon": "📚",
        "description": "Organizational memory",
    },
    "costs": {
        "title": "Costs",
        "icon": "💰",
        "description": "Token usage and cost tracking",
    },
    "settings": {
        "title": "Settings",
        "icon": "⚙️",
        "description": "Configure system behavior",
    },
}


# =============================================================================
# Session State Initialization
# =============================================================================

def init_session_state() -> None:
    """Initialize Streamlit session state variables."""
    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = None

    # Agent activity tracking
    if "agent_activity" not in st.session_state:
        st.session_state.agent_activity = {
            "council": [],
            "operational": [],
            "sme": [],
        }

    if "active_agents" not in st.session_state:
        st.session_state.active_agents = []

    # Results storage
    if "results" not in st.session_state:
        st.session_state.results = {}

    if "current_result" not in st.session_state:
        st.session_state.current_result = None

    # Event streaming
    if "event_stream_id" not in st.session_state:
        st.session_state.event_stream_id = f"ui_{int(time.time())}"

    # Settings
    if "default_tier" not in st.session_state:
        st.session_state.default_tier = 2  # Standard

    if "default_format" not in st.session_state:
        st.session_state.default_format = "markdown"

    if "max_budget" not in st.session_state:
        st.session_state.max_budget = 10.0  # USD

    if "show_developer_mode" not in st.session_state:
        st.session_state.show_developer_mode = False

    # UI preferences
    if "theme" not in st.session_state:
        st.session_state.theme = "light"

    if "auto_scroll" not in st.session_state:
        st.session_state.auto_scroll = True

    if "compact_mode" not in st.session_state:
        st.session_state.compact_mode = False


# =============================================================================
# Event Handlers
# =============================================================================

def setup_event_streaming() -> None:
    """Setup event streaming for real-time updates."""
    streamer = get_event_streamer()

    # Subscribe to all relevant events
    streamer.create_stream(
        stream_id=st.session_state.event_stream_id,
        event_types=[
            EventType.TASK_STARTED,
            EventType.TASK_PROGRESS,
            EventType.TASK_COMPLETED,
            EventType.AGENT_STARTED,
            EventType.AGENT_COMPLETED,
            EventType.PHASE_STARTED,
            EventType.PHASE_COMPLETED,
            EventType.FINDING_REPORTED,
            EventType.VERDICT_PASSED,
            EventType.VERDICT_FAILED,
            EventType.SME_SPAWNED,
            EventType.ERROR,
        ],
        session_id=st.session_state.current_session_id,
    )


def get_recent_events(limit: int = 50) -> List[Dict[str, Any]]:
    """Get recent events for the current session."""
    streamer = get_event_streamer()
    events = streamer.get_stream_events(
        stream_id=st.session_state.event_stream_id,
        limit=limit,
    )

    # Convert to dict for JSON serialization
    return [
        {
            "event_type": e.event_type.value,
            "timestamp": e.timestamp,
            "source": e.source,
            "data": e.data,
            "event_id": e.event_id,
        }
        for e in events
    ]


# =============================================================================
# UI Components - Sidebar
# =============================================================================

def render_sidebar() -> str:
    """
    Render the sidebar with navigation and system status.

    Returns:
        Selected page name
    """
    with st.sidebar:
        # Header
        st.title(f"{APP_ICON} {APP_TITLE}")
        st.caption(f"v{VERSION}")

        st.markdown("---")

        # Navigation menu
        selected_page = option_menu(
            menu_title=None,
            options=list(PAGES.keys()),
            icons=[PAGES[k]["icon"] for k in PAGES.keys()],
            menu_icon="cast",
            default_index=0,
            orientation="vertical",
            styles={
                "container": {"padding": "0!important", "background-color": "transparent"},
                "icon": {"color": "var(--text-color)", "font-size": "18px"},
                "nav-link": {
                    "font-size": "14px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "rgba(255,255,255,0.1)",
                },
                "nav-link-selected": {"background-color": "#0068c9"},
            },
        )

        st.markdown("---")

        # Quick stats
        st.subheader("Session Status")

        # Message count
        msg_count = len(st.session_state.messages)
        st.metric("Messages", msg_count)

        # Active agents
        active_count = len(st.session_state.active_agents)
        if active_count > 0:
            st.metric("Active Agents", active_count)
        else:
            st.metric("Active Agents", "None")

        # Session info
        if st.session_state.current_session_id:
            st.caption(f"Session: {st.session_state.current_session_id[:8]}...")
        else:
            st.caption("No active session")

        st.markdown("---")

        # Quick actions
        st.subheader("Quick Actions")

        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.current_session_id = None
            st.rerun()

        if st.button("🔄 New Session", use_container_width=True):
            st.session_state.current_session_id = f"sess_{int(time.time())}"
            st.session_state.messages = []
            st.rerun()

        st.markdown("---")

        # System info
        st.subheader("System Info")

        st.caption(f"Default Tier: {st.session_state.default_tier}")
        st.caption(f"Format: {st.session_state.default_format}")
        st.caption(f"Budget: ${st.session_state.max_budget:.2f}")

    return selected_page


# =============================================================================
# UI Components - Main Content
# =============================================================================

def render_page_header(title: str, icon: str, description: str) -> None:
    """Render a page header."""
    st.markdown(f"### {icon} {title}")
    st.caption(description)
    st.markdown("---")


def render_empty_state(
    icon: str,
    title: str,
    message: str,
    action_text: Optional[str] = None,
    action_key: Optional[str] = None,
) -> bool:
    """
    Render an empty state with optional action button.

    Args:
        icon: Icon to display
        title: Empty state title
        message: Descriptive message
        action_text: Optional button text
        action_key: Optional key for button

    Returns:
        True if action button was clicked
    """
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown(f"#### {icon}")
        st.markdown(f"**{title}**")
        st.markdown(f"<div style='text-align: center; color: gray;'>{message}</div>",
                   unsafe_allow_html=True)

        if action_text and action_key:
            if st.button(action_text, key=action_key, use_container_width=True):
                return True

    return False


# =============================================================================
# Page Routing
# =============================================================================

def render_chat_page() -> None:
    """Render the chat page."""
    from src.ui.pages.chat import render_chat_interface

    render_page_header(
        "Chat",
        PAGES["chat"]["icon"],
        PAGES["chat"]["description"],
    )

    render_chat_interface()


def render_agents_page() -> None:
    """Render the agent activity monitoring page."""
    from src.ui.components.agent_panel import render_agent_monitor

    render_page_header(
        "Agent Activity",
        PAGES["agents"]["icon"],
        PAGES["agents"]["description"],
    )

    render_agent_monitor()


def render_results_page() -> None:
    """Render the results browser page."""
    from src.ui.components.results_inspector import render_results_browser

    render_page_header(
        "Results",
        PAGES["results"]["icon"],
        PAGES["results"]["description"],
    )

    render_results_browser()


def render_skills_page() -> None:
    """Render the skills catalogue page."""
    from src.ui.pages.skills import render_skills_catalogue

    render_page_header(
        "Skills",
        PAGES["skills"]["icon"],
        PAGES["skills"]["description"],
    )

    render_skills_catalogue()


def render_personas_page() -> None:
    """Render the SME personas browser page."""
    from src.ui.pages.sme_browser import render_sme_browser

    render_page_header(
        "SME Personas",
        PAGES["personas"]["icon"],
        PAGES["personas"]["description"],
    )

    render_sme_browser()


def render_ensembles_page() -> None:
    """Render the ensembles catalogue page."""
    from src.ui.pages.ensembles import render_ensembles_catalogue

    render_page_header(
        "Ensembles",
        PAGES["ensembles"]["icon"],
        PAGES["ensembles"]["description"],
    )

    render_ensembles_catalogue()


def render_debates_page() -> None:
    """Render the debates transcript viewer page."""
    from src.ui.components.debate_viewer import render_debate_page

    render_page_header(
        "Debates",
        PAGES["debates"]["icon"],
        PAGES["debates"]["description"],
    )

    render_debate_page()


def render_knowledge_page() -> None:
    """Render the knowledge base page."""
    from src.ui.pages.knowledge import render_knowledge_base

    render_page_header(
        "Knowledge Base",
        PAGES["knowledge"]["icon"],
        PAGES["knowledge"]["description"],
    )

    render_knowledge_base()


def render_costs_page() -> None:
    """Render the cost tracking page."""
    from src.ui.components.cost_dashboard import render_cost_dashboard

    render_page_header(
        "Costs",
        PAGES["costs"]["icon"],
        PAGES["costs"]["description"],
    )

    render_cost_dashboard()


def render_settings_page() -> None:
    """Render the settings page."""
    from src.ui.pages.settings import render_settings_panel

    render_page_header(
        "Settings",
        PAGES["settings"]["icon"],
        PAGES["settings"]["description"],
    )

    render_settings_panel()


# =============================================================================
# Main Application
# =============================================================================

def main() -> None:
    """Main application entry point."""
    # Page config
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS
    st.markdown("""
    <style>
        /* Main container padding */
        .main .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
            max-width: 1400px;
        }

        /* Sidebar styling */
        .css-1d391kg {
            background-color: #f8f9fa;
        }

        /* Agent status indicators */
        .status-active {
            color: #28a745;
            font-weight: bold;
        }
        .status-idle {
            color: #6c757d;
        }
        .status-error {
            color: #dc3545;
            font-weight: bold;
        }

        /* Tier badges */
        .tier-1 { background: #e7f5ff; color: #1c7ed6; padding: 2px 8px; border-radius: 4px; }
        .tier-2 { background: #fff4e6; color: #fd7e14; padding: 2px 8px; border-radius: 4px; }
        .tier-3 { background: #f3f0ff; color: #7950f2; padding: 2px 8px; border-radius: 4px; }
        .tier-4 { background: #fff0f6; color: #f06595; padding: 2px 8px; border-radius: 4px; }

        /* Message styling */
        .user-message {
            background: #e3f2fd;
            padding: 12px;
            border-radius: 8px;
            margin: 8px 0;
        }
        .agent-message {
            background: #f5f5f5;
            padding: 12px;
            border-radius: 8px;
            margin: 8px 0;
        }

        /* Code blocks */
        pre {
            background: #f8f9fa;
            padding: 12px;
            border-radius: 4px;
            overflow-x: auto;
        }
    </style>
    """, unsafe_allow_html=True)

    # Configure structured logging for Streamlit
    configure_logging(level="INFO", json_output=False, enable_filtering=True)

    # Initialize session state
    init_session_state()

    # Setup event streaming
    setup_event_streaming()

    # Setup agent panel event subscriptions for real-time updates
    from src.ui.components.agent_panel import setup_agent_event_subscriptions
    setup_agent_event_subscriptions(session_id=st.session_state.current_session_id)

    _app_logger.info("streamlit_app.initialized",
                     session_id=st.session_state.current_session_id)

    # Render sidebar and get selected page
    selected_page = render_sidebar()

    # Route to appropriate page
    page_renderers = {
        "chat": render_chat_page,
        "agents": render_agents_page,
        "results": render_results_page,
        "skills": render_skills_page,
        "personas": render_personas_page,
        "ensembles": render_ensembles_page,
        "debates": render_debates_page,
        "knowledge": render_knowledge_page,
        "costs": render_costs_page,
        "settings": render_settings_page,
    }

    renderer = page_renderers.get(selected_page)
    if renderer:
        renderer()
    else:
        st.error(f"Unknown page: {selected_page}")

    # Auto-refresh for real-time updates
    if st.session_state.active_agents:
        time.sleep(1)
        st.rerun()


if __name__ == "__main__":
    main()
