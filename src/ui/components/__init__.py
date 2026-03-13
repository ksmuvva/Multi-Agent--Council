"""
UI Components - Reusable Streamlit Components

Reusable UI components for the Streamlit application.
"""

from .agent_panel import (
    AgentStatus,
    AgentTier,
    AgentActivity,
    get_agent_activities,
    add_agent_activity,
    update_agent_status,
    clear_agent_activities,
    render_agent_monitor,
    render_compact_agent_panel,
)

from .results_inspector import (
    OutputFormat,
    ContentType,
    ResultMetadata,
    AgentResult,
    get_results_store,
    add_result,
    get_result,
    get_recent_results,
    clear_results,
    create_result,
)

from .cost_dashboard import (
    ModelPricing,
    MODEL_PRICING,
    AgentCost,
    CostSession,
    get_cost_sessions,
    get_current_session,
    add_agent_cost,
    get_daily_cost,
    get_weekly_cost,
    render_cost_dashboard,
)

from .debate_viewer import (
    DebatePhase,
    ArgumentType,
    PersuasionStrength,
    Argument,
    DebatePerspective,
    DebateRound,
    DebateConsensus,
    DebateTranscript,
    render_debate_viewer,
)

from .enhanced_filters import (
    FilterConfig,
    AdvancedFilters,
    render_export_buttons,
    render_sort_options,
    render_pagination,
)

__all__ = [
    # Agent panel
    "AgentStatus",
    "AgentTier",
    "AgentActivity",
    "get_agent_activities",
    "add_agent_activity",
    "update_agent_status",
    "clear_agent_activities",
    "render_agent_monitor",
    "render_compact_agent_panel",
    # Results inspector
    "OutputFormat",
    "ContentType",
    "ResultMetadata",
    "AgentResult",
    "get_results_store",
    "add_result",
    "get_result",
    "get_recent_results",
    "clear_results",
    "create_result",
    # Cost dashboard
    "ModelPricing",
    "MODEL_PRICING",
    "AgentCost",
    "CostSession",
    "get_cost_sessions",
    "get_current_session",
    "add_agent_cost",
    "get_daily_cost",
    "get_weekly_cost",
    "render_cost_dashboard",
    # Debate viewer
    "DebatePhase",
    "ArgumentType",
    "PersuasionStrength",
    "Argument",
    "DebatePerspective",
    "DebateRound",
    "DebateConsensus",
    "DebateTranscript",
    "render_debate_viewer",
    # Enhanced filters
    "FilterConfig",
    "AdvancedFilters",
    "render_export_buttons",
    "render_sort_options",
    "render_pagination",
]
