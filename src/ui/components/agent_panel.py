"""
Agent Panel Component - Three-Tier Agent Activity Display

Real-time visualization of Council, Operational, and SME agent activities.
Provides live updates on agent states, progress, and outputs.
"""

import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum

import streamlit as st

from src.utils.logging import get_logger
from src.utils.events import (
    get_event_emitter,
    Event,
    EventType,
)

_logger = get_logger("agent_panel")


# =============================================================================
# Agent Status Types
# =============================================================================

class AgentStatus(str, Enum):
    """Status of an agent."""
    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentTier(str, Enum):
    """Agent tier classification."""
    COUNCIL = "council"
    OPERATIONAL = "operational"
    SME = "sme"


# =============================================================================
# Agent Activity Data
# =============================================================================

@dataclass
class AgentActivity:
    """Activity data for a single agent."""
    agent_id: str
    agent_name: str
    tier: AgentTier
    status: AgentStatus
    phase: str
    progress: float  # 0.0 to 1.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    output_preview: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> Optional[timedelta]:
        """Get the duration of the agent's activity."""
        if self.start_time:
            end = self.end_time or datetime.now()
            return end - self.start_time
        return None

    @property
    def is_active(self) -> bool:
        """Check if agent is currently active."""
        return self.status in [AgentStatus.STARTING, AgentStatus.RUNNING, AgentStatus.WAITING]

    @property
    def status_emoji(self) -> str:
        """Get emoji for status."""
        return {
            AgentStatus.IDLE: "💤",
            AgentStatus.STARTING: "🔄",
            AgentStatus.RUNNING: "⚡",
            AgentStatus.WAITING: "⏸️",
            AgentStatus.COMPLETED: "✅",
            AgentStatus.FAILED: "❌",
            AgentStatus.CANCELLED: "🚫",
        }.get(self.status, "❓")

    @property
    def status_color(self) -> str:
        """Get CSS color for status."""
        return {
            AgentStatus.IDLE: "#6c757d",
            AgentStatus.STARTING: "#17a2b8",
            AgentStatus.RUNNING: "#28a745",
            AgentStatus.WAITING: "#ffc107",
            AgentStatus.COMPLETED: "#007bff",
            AgentStatus.FAILED: "#dc3545",
            AgentStatus.CANCELLED: "#6c757d",
        }.get(self.status, "#000000")


# =============================================================================
# Session State Management
# =============================================================================

def get_agent_activities() -> Dict[str, List[AgentActivity]]:
    """Get agent activities from session state."""
    if "agent_activities" not in st.session_state:
        st.session_state.agent_activities = {
            AgentTier.COUNCIL: [],
            AgentTier.OPERATIONAL: [],
            AgentTier.SME: [],
        }
    return st.session_state.agent_activities


def add_agent_activity(activity: AgentActivity) -> None:
    """Add an agent activity to the session state."""
    activities = get_agent_activities()

    # Remove existing activity for this agent
    activities[activity.tier] = [
        a for a in activities[activity.tier]
        if a.agent_id != activity.agent_id
    ]

    # Add new activity
    activities[activity.tier].append(activity)

    # Update active agents list
    active = [a for tier_activities in activities.values()
              for a in tier_activities if a.is_active]
    st.session_state.active_agents = [a.agent_id for a in active]


def update_agent_status(
    agent_id: str,
    status: AgentStatus,
    progress: Optional[float] = None,
    output_preview: Optional[str] = None,
    error_message: Optional[str] = None,
) -> None:
    """Update an existing agent activity."""
    activities = get_agent_activities()

    for tier_activities in activities.values():
        for activity in tier_activities:
            if activity.agent_id == agent_id:
                activity.status = status
                if progress is not None:
                    activity.progress = progress
                if output_preview is not None:
                    activity.output_preview = output_preview
                if error_message is not None:
                    activity.error_message = error_message
                if status == AgentStatus.COMPLETED:
                    activity.end_time = datetime.now()
                return


def clear_agent_activities() -> None:
    """Clear all agent activities."""
    st.session_state.agent_activities = {
        AgentTier.COUNCIL: [],
        AgentTier.OPERATIONAL: [],
        AgentTier.SME: [],
    }
    st.session_state.active_agents = []


# =============================================================================
# UI Components
# =============================================================================

def render_tier_header(
    tier: AgentTier,
    activity_count: int,
    active_count: int,
) -> None:
    """Render a tier section header."""
    tier_info = {
        AgentTier.COUNCIL: {
            "name": "Strategic Council",
            "icon": "🏛️",
            "color": "#7950f2",
            "description": "Governance, quality arbitration, and SME selection",
        },
        AgentTier.OPERATIONAL: {
            "name": "Operational Agents",
            "icon": "⚙️",
            "color": "#fd7e14",
            "description": "Analysis, planning, research, and execution",
        },
        AgentTier.SME: {
            "name": "SME Personas",
            "icon": "👤",
            "color": "#20c997",
            "description": "Domain expertise on demand",
        },
    }

    info = tier_info[tier]

    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {info['color']}15 0%, {info['color']}05 100%);
        border-left: 4px solid {info['color']};
        padding: 12px 16px;
        border-radius: 8px;
        margin-bottom: 16px;
    ">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h4 style="margin: 0; color: {info['color']};">
                    {info['icon']} {info['name']}
                </h4>
                <small style="color: #666;">{info['description']}</small>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 24px; font-weight: bold; color: {info['color']};">
                    {active_count}/{activity_count}
                </div>
                <small style="color: #666;">active/total</small>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_agent_card(activity: AgentActivity) -> None:
    """Render a single agent activity card."""
    # Duration display
    duration_str = ""
    if activity.duration:
        total_seconds = int(activity.duration.total_seconds())
        if total_seconds < 60:
            duration_str = f"{total_seconds}s"
        else:
            minutes = total_seconds // 60
            seconds = total_seconds % 60
            duration_str = f"{minutes}m {seconds}s"

    # Progress bar
    progress_color = activity.status_color
    if activity.is_active:
        # Animate progress for active agents
        progress_value = activity.progress
        if activity.status == AgentStatus.STARTING:
            progress_value = min(activity.progress + 0.1, 0.9)
    else:
        progress_value = 1.0 if activity.status == AgentStatus.COMPLETED else activity.progress

    st.markdown(f"""
    <div style="
        border: 1px solid #e0e0e0;
        border-left: 4px solid {progress_color};
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 8px;
        background: white;
        transition: all 0.3s ease;
    ">
        <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 8px;">
            <div>
                <strong>{activity.status_emoji} {activity.agent_name}</strong>
                <div style="font-size: 12px; color: #666; margin-top: 4px;">
                    📍 {activity.phase}
                    {f" • ⏱️ {duration_str}" if duration_str else ""}
                </div>
            </div>
            <div style="text-align: right;">
                <span style="
                    background: {progress_color}20;
                    color: {progress_color};
                    padding: 4px 12px;
                    border-radius: 12px;
                    font-size: 11px;
                    font-weight: bold;
                    text-transform: uppercase;
                ">{activity.status.value}</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Progress bar
    if activity.is_active or activity.progress > 0:
        st.progress(progress_value if activity.is_active else activity.progress)

    # Output preview or error
    if activity.error_message:
        st.error(f"❌ Error: {activity.error_message}")
    elif activity.output_preview:
        with st.expander("📄 Output Preview", expanded=False):
            st.markdown(activity.output_preview[:500] + "..." if len(activity.output_preview) > 500
                       else activity.output_preview)

    # Metadata
    if activity.metadata:
        with st.expander("ℹ️ Details", expanded=False):
            for key, value in activity.metadata.items():
                st.caption(f"**{key}**: {value}")

    st.markdown("</div>", unsafe_allow_html=True)


def render_agent_monitor() -> None:
    """Render the main agent activity monitoring interface."""
    activities = get_agent_activities()
    demo_mode = is_demo_mode()

    # Demo mode warning
    if demo_mode:
        st.warning("""
        ⚠️ **Demo Mode Active**: Displaying simulated agent activities for demonstration purposes.

        Real-time agent monitoring requires:
        - An active session with the orchestrator
        - Event streaming enabled (via setup_agent_event_subscriptions)
        - Agents executing through the multi-agent system
        """)

    # Header
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total = sum(len(v) for v in activities.values())
        st.metric("Total Agents", total)

    with col2:
        active = len(st.session_state.get("active_agents", []))
        st.metric("Active Now", active)

    with col3:
        completed = sum(
            1 for tier_activities in activities.values()
            for a in tier_activities
            if a.status == AgentStatus.COMPLETED
        )
        st.metric("Completed", completed)

    with col4:
        failed = sum(
            1 for tier_activities in activities.values()
            for a in tier_activities
            if a.status == AgentStatus.FAILED
        )
        st.metric("Failed", failed)

    st.markdown("---")

    # Controls
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

    with col2:
        if st.button("🗑️ Clear History", use_container_width=True):
            clear_agent_activities()
            st.rerun()

    with col3:
        auto_refresh = st.checkbox("🔴 Auto-refresh", value=True)

    with col4:
        demo_toggle = st.checkbox("🎭 Demo Mode", value=demo_mode)
        if demo_toggle != demo_mode:
            set_demo_mode(demo_toggle)
            st.rerun()

    st.markdown("---")

    # Render each tier
    for tier in [AgentTier.COUNCIL, AgentTier.OPERATIONAL, AgentTier.SME]:
        tier_activities = activities[tier]
        active_count = sum(1 for a in tier_activities if a.is_active)

        render_tier_header(tier, len(tier_activities), active_count)

        if not tier_activities:
            if demo_mode:
                st.info(f"No {tier.value} agents in demo. Click '🔄 Refresh' to regenerate demo data.")
            else:
                st.info(f"No {tier.value} agents active. Start a task or enable Demo Mode to see activity.")
        else:
            for activity in tier_activities:
                render_agent_card(activity)

    # Auto-refresh
    if auto_refresh and active > 0:
        time.sleep(2)
        st.rerun()


def render_compact_agent_panel() -> None:
    """Render a compact version of the agent panel for sidebar."""
    activities = get_agent_activities()
    active = len(st.session_state.get("active_agents", []))

    if active == 0:
        st.caption("💤 No active agents")
        return

    st.caption(f"⚡ {active} active agent(s)")

    for tier in [AgentTier.OPERATIONAL, AgentTier.COUNCIL, AgentTier.SME]:
        tier_activities = [a for a in activities[tier] if a.is_active]

        if tier_activities:
            for activity in tier_activities:
                # Compact display
                status_emoji = activity.status_emoji
                progress_pct = int(activity.progress * 100)

                st.markdown(f"""
                <div style="font-size: 12px; margin-bottom: 4px;">
                    {status_emoji} <strong>{activity.agent_name}</strong>
                    <span style="float: right; color: {activity.status_color};">
                        {progress_pct}%
                    </span>
                </div>
                """, unsafe_allow_html=True)

                st.progress(activity.progress, min_height=20)


# =============================================================================
# Event Integration
# =============================================================================

def handle_agent_event(event_data: Dict[str, Any]) -> None:
    """
    Handle an agent-related event from the event system.

    Args:
        event_data: Event data dictionary
    """
    event_type = event_data.get("event_type")

    if event_type == "agent_started":
        # Create new activity
        tier_map = {
            "council": AgentTier.COUNCIL,
            "operational": AgentTier.OPERATIONAL,
            "sme": AgentTier.SME,
        }

        tier = tier_map.get(event_data.get("tier", "operational"), AgentTier.OPERATIONAL)

        activity = AgentActivity(
            agent_id=event_data["agent_id"],
            agent_name=event_data["agent_name"],
            tier=tier,
            status=AgentStatus.STARTING,
            phase=event_data.get("phase", "Initializing"),
            progress=0.0,
            start_time=datetime.now(),
        )

        add_agent_activity(activity)

        # Update to running
        update_agent_status(activity.agent_id, AgentStatus.RUNNING)

    elif event_type == "agent_progress":
        update_agent_status(
            agent_id=event_data["agent_id"],
            progress=event_data.get("progress", 0.0),
            output_preview=event_data.get("output"),
        )

    elif event_type == "agent_completed":
        update_agent_status(
            agent_id=event_data["agent_id"],
            status=AgentStatus.COMPLETED,
            progress=1.0,
            output_preview=event_data.get("output"),
        )

    elif event_type == "agent_failed":
        update_agent_status(
            agent_id=event_data["agent_id"],
            status=AgentStatus.FAILED,
            error_message=event_data.get("error"),
        )


# =============================================================================
# Event System Integration
# =============================================================================

def setup_agent_event_subscriptions(session_id: Optional[str] = None) -> None:
    """
    Subscribe to agent lifecycle events from the event system.
    Bridges events into Streamlit session state for real-time display.
    """
    if st.session_state.get("_agent_panel_subscribed"):
        return

    emitter = get_event_emitter()

    # Map agent names to tiers
    council_agents = {"Domain Council Chair", "Quality Arbiter", "Ethics & Safety Advisor"}

    def _on_agent_event(event: Event) -> None:
        """Handle agent events and update session state."""
        source = event.source
        data = event.data
        agent_name = data.get("agent", source)

        # Determine tier
        if agent_name in council_agents:
            tier = AgentTier.COUNCIL
        elif "sme" in agent_name.lower() or data.get("tier") == "sme":
            tier = AgentTier.SME
        else:
            tier = AgentTier.OPERATIONAL

        agent_id = f"{tier.value}_{agent_name.lower().replace(' ', '_')}"

        if event.event_type == EventType.AGENT_STARTED:
            activity = AgentActivity(
                agent_id=agent_id,
                agent_name=agent_name,
                tier=tier,
                status=AgentStatus.RUNNING,
                phase=data.get("phase", "Processing"),
                progress=0.1,
                start_time=datetime.now(),
            )
            add_agent_activity(activity)
            _logger.debug("agent_panel.agent_started", agent=agent_name, tier=tier.value)

        elif event.event_type == EventType.AGENT_COMPLETED:
            update_agent_status(
                agent_id=agent_id,
                status=AgentStatus.COMPLETED,
                progress=1.0,
                output_preview=data.get("output_summary", ""),
            )
            _logger.debug("agent_panel.agent_completed", agent=agent_name)

        elif event.event_type == EventType.AGENT_FAILED:
            update_agent_status(
                agent_id=agent_id,
                status=AgentStatus.FAILED,
                error_message=data.get("error_message", "Unknown error"),
            )
            _logger.debug("agent_panel.agent_failed", agent=agent_name)

        elif event.event_type == EventType.AGENT_PROGRESS:
            update_agent_status(
                agent_id=agent_id,
                status=AgentStatus.RUNNING,
                progress=data.get("progress", 0.5),
                output_preview=data.get("message", ""),
            )

    emitter.subscribe(
        event_types=[
            EventType.AGENT_STARTED,
            EventType.AGENT_COMPLETED,
            EventType.AGENT_FAILED,
            EventType.AGENT_PROGRESS,
        ],
        callback=_on_agent_event,
        subscriber_id="streamlit_agent_panel",
        session_id=session_id,
    )

    st.session_state._agent_panel_subscribed = True
    _logger.info("agent_panel.subscriptions_active", session_id=session_id)


# =============================================================================
# Demo Mode Management
# =============================================================================

def is_demo_mode() -> bool:
    """Check if demo mode is enabled."""
    return st.session_state.get("agent_panel_demo_mode", False)


def set_demo_mode(enabled: bool) -> None:
    """Set demo mode for the agent panel."""
    st.session_state.agent_panel_demo_mode = enabled
    if enabled:
        generate_mock_activities()
        _logger.info("agent_panel.demo_mode_enabled")
    else:
        clear_agent_activities()
        _logger.info("agent_panel.demo_mode_disabled")


# =============================================================================
# Mock Data Generator (for testing/demo)
# =============================================================================

def generate_mock_activities() -> None:
    """Generate mock agent activities for testing/demo purposes.

    NOTE: This is for demonstration only. Real agent activities come from
    the event system via setup_agent_event_subscriptions().
    """
    clear_agent_activities()

    # Council agents
    add_agent_activity(AgentActivity(
        agent_id="council_chair",
        agent_name="Council Chair",
        tier=AgentTier.COUNCIL,
        status=AgentStatus.COMPLETED,
        phase="SME Selection",
        progress=1.0,
        start_time=datetime.now() - timedelta(seconds=45),
        end_time=datetime.now() - timedelta(seconds=15),
        output_preview="Selected IAM Architect and Security Analyst for this task...",
    ))

    # Operational agents
    for i, (name, phase, status, progress) in enumerate([
        ("Analyst", "Task Analysis", AgentStatus.COMPLETED, 1.0),
        ("Planner", "Execution Planning", AgentStatus.COMPLETED, 1.0),
        ("Researcher", "Information Gathering", AgentStatus.RUNNING, 0.6),
        ("Executor", "Solution Generation", AgentStatus.WAITING, 0.0),
        ("Verifier", "Output Verification", AgentStatus.IDLE, 0.0),
    ]):
        activity = AgentActivity(
            agent_id=f"operational_{i}",
            agent_name=name,
            tier=AgentTier.OPERATIONAL,
            status=status,
            phase=phase,
            progress=progress,
            start_time=datetime.now() - timedelta(seconds=30 + i * 10),
        )
        add_agent_activity(activity)

    # SME agents
    add_agent_activity(AgentActivity(
        agent_id="sme_iam_architect",
        agent_name="IAM Architect",
        tier=AgentTier.SME,
        status=AgentStatus.RUNNING,
        phase="Advisory Review",
        progress=0.4,
        start_time=datetime.now() - timedelta(seconds=20),
    ))
