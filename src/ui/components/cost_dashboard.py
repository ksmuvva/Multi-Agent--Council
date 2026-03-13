"""
Cost Dashboard Component - Real-time Cost Tracking

Displays token usage, costs, and budget information for agent operations.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px


# =============================================================================
# Cost Data Structures
# =============================================================================

class ModelPricing(str, Enum):
    """Claude model pricing per 1M tokens."""
    HAIKU = "haiku"
    SONNET = "sonnet"
    OPUS = "opus"


# Pricing (USD per 1M tokens)
MODEL_PRICING = {
    ModelPricing.HAIKU: {"input": 0.25, "output": 1.25},
    ModelPricing.SONNET: {"input": 3.0, "output": 15.0},
    ModelPricing.OPUS: {"input": 15.0, "output": 75.0},
}


@dataclass
class AgentCost:
    """Cost data for a single agent execution."""
    agent_name: str
    model: ModelPricing
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    timestamp: datetime
    tier: int
    phase: str


@dataclass
class CostSession:
    """Cost tracking for a session."""
    session_id: str
    start_time: datetime
    agent_costs: List[AgentCost] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        """Total tokens used in session."""
        return sum(c.total_tokens for c in self.agent_costs)

    @property
    def total_cost(self) -> float:
        """Total cost in USD."""
        return sum(c.cost_usd for c in self.agent_costs)

    @property
    def input_tokens(self) -> int:
        """Total input tokens."""
        return sum(c.input_tokens for c in self.agent_costs)

    @property
    def output_tokens(self) -> int:
        """Total output tokens."""
        return sum(c.output_tokens for c in self.agent_costs)

    @property
    def duration(self) -> timedelta:
        """Session duration."""
        return datetime.now() - self.start_time


# =============================================================================
# Session State Management
# =============================================================================

def get_cost_sessions() -> Dict[str, CostSession]:
    """Get all cost sessions."""
    if "cost_sessions" not in st.session_state:
        st.session_state.cost_sessions = {}
    return st.session_state.cost_sessions


def get_current_session() -> Optional[CostSession]:
    """Get the current cost session."""
    session_id = st.session_state.get("current_session_id")
    if not session_id:
        return None

    sessions = get_cost_sessions()
    if session_id not in sessions:
        sessions[session_id] = CostSession(
            session_id=session_id,
            start_time=datetime.now(),
        )

    return sessions[session_id]


def add_agent_cost(cost: AgentCost) -> None:
    """Add an agent cost to the current session."""
    session = get_current_session()
    if session:
        session.agent_costs.append(cost)


def get_daily_cost(date: Optional[datetime] = None) -> float:
    """Get total cost for a day."""
    if date is None:
        date = datetime.now()

    sessions = get_cost_sessions()
    daily_cost = 0.0

    for session in sessions.values():
        # Check if session was on this day
        if session.start_time.date() == date.date():
            daily_cost += session.total_cost

    return daily_cost


def get_weekly_cost() -> Dict[datetime, float]:
    """Get cost breakdown for the past week."""
    daily_costs = {}
    today = datetime.now().date()

    for i in range(7):
        date = today - timedelta(days=i)
        daily_costs[datetime.combine(date, datetime.min.time())] = 0.0

    sessions = get_cost_sessions()
    for session in sessions.values():
        session_date = datetime.combine(session.start_time.date(), datetime.min.time())
        if session_date in daily_costs:
            daily_costs[session_date] += session.total_cost

    return daily_costs


# =============================================================================
# Rendering Functions
# =============================================================================

def render_cost_dashboard() -> None:
    """Render the main cost dashboard."""
    st.markdown("### 💰 Cost Tracking")
    st.caption("Token usage and cost monitoring")

    st.markdown("---")

    # Current session costs
    render_session_costs()

    st.markdown("---")

    # Charts
    tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Trends", "🔍 Breakdown"])

    with tab1:
        render_cost_overview()

    with tab2:
        render_cost_trends()

    with tab3:
        render_cost_breakdown()


def render_session_costs() -> None:
    """Render current session cost summary."""
    current_session = get_current_session()

    col1, col2, col3, col4 = st.columns(4)

    if current_session and current_session.agent_costs:
        with col1:
            st.metric("Session Cost", f"${current_session.total_cost:.4f}")

        with col2:
            st.metric("Total Tokens", f"{current_session.total_tokens:,}")

        with col3:
            st.metric("Input Tokens", f"{current_session.input_tokens:,}")

        with col4:
            st.metric("Output Tokens", f"{current_session.output_tokens:,}")
    else:
        with col1:
            st.metric("Session Cost", "$0.0000")

        with col2:
            st.metric("Total Tokens", "0")

        with col3:
            st.metric("Input Tokens", "0")

        with col4:
            st.metric("Output Tokens", "0")

    # Budget status
    max_budget = st.session_state.get("max_budget", 10.0)
    current_cost = current_session.total_cost if current_session else 0.0
    budget_remaining = max_budget - current_cost

    budget_col1, budget_col2 = st.columns(2)

    with budget_col1:
        # Budget progress bar
        budget_pct = (current_cost / max_budget) * 100 if max_budget > 0 else 0

        if budget_pct >= 100:
            st.error(f"🚨 Budget exceeded! ${current_cost:.2f} / ${max_budget:.2f}")
        elif budget_pct >= 80:
            st.warning(f"⚠️ {budget_pct:.1f}% of budget used: ${current_cost:.2f} / ${max_budget:.2f}")
        else:
            st.progress(budget_pct / 100)
            st.caption(f"${current_cost:.2f} / ${max_budget:.2f} ({budget_pct:.1f}%)")

    with budget_col2:
        st.metric("Budget Remaining", f"${budget_remaining:.2f}")

    # Daily budget
    daily_budget = st.session_state.get("daily_budget", 50.0)
    daily_cost = get_daily_cost()
    daily_remaining = daily_budget - daily_cost

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Today's Spend", f"${daily_cost:.4f}")

    with col2:
        st.metric("Daily Remaining", f"${daily_remaining:.2f}")


def render_cost_overview() -> None:
    """Render cost overview charts."""
    # Get weekly data
    weekly_data = get_weekly_cost()

    if not any(weekly_data.values()):
        st.info("No cost data available yet")
        return

    # Weekly cost chart
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=list(weekly_data.keys()),
        y=list(weekly_data.values()),
        name="Daily Cost",
        marker_color="#007bff",
    ))

    fig.update_layout(
        title="Daily Cost (Last 7 Days)",
        xaxis_title="Date",
        yaxis_title="Cost (USD)",
        hovermode="x unified",
        height=300,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Summary stats
    total_weekly = sum(weekly_data.values())
    avg_daily = total_weekly / 7

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Weekly Total", f"${total_weekly:.4f}")

    with col2:
        st.metric("Daily Average", f"${avg_daily:.4f}")

    with col3:
        max_day = max(weekly_data.items(), key=lambda x: x[1])
        st.metric("Highest Day", f"${max_day[1]:.4f}", f"{max_day[0].strftime('%a')}")


def render_cost_trends() -> None:
    """Render cost trend charts."""
    current_session = get_current_session()

    if not current_session or not current_session.agent_costs:
        st.info("No session cost data available")
        return

    # Agent cost breakdown
    agent_costs = {}
    for cost in current_session.agent_costs:
        if cost.agent_name not in agent_costs:
            agent_costs[cost.agent_name] = {"tokens": 0, "cost": 0.0}
        agent_costs[cost.agent_name]["tokens"] += cost.total_tokens
        agent_costs[cost.agent_name]["cost"] += cost.cost_usd

    if agent_costs:
        # Sort by cost
        sorted_agents = sorted(agent_costs.items(), key=lambda x: x[1]["cost"], reverse=True)
        agents = [a[0] for a in sorted_agents]
        costs = [a[1]["cost"] for a in sorted_agents]

        # Bar chart
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=agents,
            y=costs,
            marker_color=[{
                ModelPricing.HAIKU: "#28a745",
                ModelPricing.SONNET: "#007bff",
                ModelPricing.OPUS: "#7950f2",
            }.get(ModelPricing.HAIKU, "#007bff") for _ in agents],
            text=[f"${c:.4f}" for c in costs],
            textposition="auto",
        ))

        fig.update_layout(
            title="Cost by Agent",
            xaxis_title="Agent",
            yaxis_title="Cost (USD)",
            height=400,
        )

        st.plotly_chart(fig, use_container_width=True)

        # Token breakdown
        st.markdown("#### Token Breakdown by Agent")

        agent_data = []
        for agent, data in sorted_agents:
            agent_data.append({
                "Agent": agent,
                "Tokens": agent_costs[agent]["tokens"],
                "Cost": f"${agent_costs[agent]['cost']:.4f}",
            })

        st.dataframe(
            agent_data,
            use_container_width=True,
            hide_index=True,
        )


def render_cost_breakdown() -> None:
    """Render detailed cost breakdown."""
    current_session = get_current_session()

    if not current_session or not current_session.agent_costs:
        st.info("No session cost data available")
        return

    # Model breakdown
    model_stats: Dict[ModelPricing, Dict[str, Any]] = {}

    for cost in current_session.agent_costs:
        model = cost.model
        if model not in model_stats:
            model_stats[model] = {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cost": 0.0,
                "calls": 0,
            }

        model_stats[model]["input_tokens"] += cost.input_tokens
        model_stats[model]["output_tokens"] += cost.output_tokens
        model_stats[model]["total_tokens"] += cost.total_tokens
        model_stats[model]["cost"] += cost.cost_usd
        model_stats[model]["calls"] += 1

    # Display model stats
    for model, stats in model_stats.items():
        model_name = model.value.title()

        st.markdown(f"#### {model_name}")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Calls", stats["calls"])

        with col2:
            st.metric("Tokens", f"{stats['total_tokens']:,}")

        with col3:
            st.metric("Input", f"{stats['input_tokens']:,}")

        with col4:
            st.metric("Output", f"{stats['output_tokens']:,}")

        st.metric("Cost", f"${stats['cost']:.4f}")

        # Pricing info
        pricing = MODEL_PRICING[model]
        st.caption(f"Input: ${pricing['input']}/M tokens • Output: ${pricing['output']}/M tokens")

        st.markdown("---")

    # Detailed history
    st.markdown("#### Detailed History")

    history_data = []
    for cost in current_session.agent_costs:
        history_data.append({
            "Agent": cost.agent_name,
            "Model": cost.model.value,
            "Phase": cost.phase,
            "Tier": cost.tier,
            "Input": cost.input_tokens,
            "Output": cost.output_tokens,
            "Total": cost.total_tokens,
            "Cost": f"${cost.cost_usd:.4f}",
            "Time": cost.timestamp.strftime("%H:%M:%S"),
        })

    st.dataframe(
        history_data,
        use_container_width=True,
        hide_index=True,
    )


# =============================================================================
# Mock Data Generation (for testing)
# =============================================================================

def generate_mock_cost_data() -> None:
    """Generate mock cost data for testing."""
    import random

    session = get_current_session()
    if not session:
        return

    agents = ["Analyst", "Planner", "Researcher", "Executor", "Verifier", "Critic", "Formatter"]
    models = [ModelPricing.HAIKU, ModelPricing.SONNET, ModelPricing.OPUS]
    phases = ["Analysis", "Planning", "Research", "Execution", "Verification", "Review", "Format"]

    for _ in range(random.randint(5, 15)):
        model = random.choice(models)
        pricing = MODEL_PRICING[model]

        input_tokens = random.randint(500, 5000)
        output_tokens = random.randint(200, 2000)
        total_tokens = input_tokens + output_tokens

        cost = (
            (input_tokens / 1_000_000) * pricing["input"] +
            (output_tokens / 1_000_000) * pricing["output"]
        )

        agent_cost = AgentCost(
            agent_name=random.choice(agents),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost_usd=cost,
            timestamp=datetime.now() - timedelta(minutes=random.randint(1, 60)),
            tier=random.randint(1, 4),
            phase=random.choice(phases),
        )

        add_agent_cost(agent_cost)

    st.rerun()


def render_developer_cost_tools() -> None:
    """Render developer-only cost data tools. Call from developer settings panel."""
    if st.session_state.get("show_developer_mode", False):
        if st.button("Generate Mock Cost Data", key="mock_cost_btn"):
            generate_mock_cost_data()
