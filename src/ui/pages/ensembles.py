"""
Ensembles Page - Pre-configured Agent Workflows

Browse and view details of ensemble patterns - pre-configured
workflows of multiple agents for common tasks.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

import streamlit as st


# =============================================================================
# Ensemble Types
# =============================================================================

class EnsembleType(str, Enum):
    """Types of ensemble patterns."""
    ARCHITECTURE_REVIEW = "architecture_review_board"
    CODE_SPRINT = "code_sprint"
    RESEARCH_COUNCIL = "research_council"
    DOCUMENT_ASSEMBLY = "document_assembly"
    REQUIREMENTS_WORKSHOP = "requirements_workshop"


@dataclass
class EnsembleAgent:
    """An agent in an ensemble."""
    name: str
    role: str
    description: str
    dependencies: List[str] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class EnsemblePattern:
    """An ensemble pattern definition."""
    pattern_id: str
    name: str
    description: str
    use_case: str
    agents: List[EnsembleAgent]
    estimated_duration: str  # e.g., "5-10 minutes"
    tier_requirement: int
    icon: str = "🎭"
    color: str = "#007bff"


# =============================================================================
# Ensemble Definitions
# =============================================================================

ENSEMBLE_PATTERNS: List[EnsemblePattern] = [
    EnsemblePattern(
        pattern_id="architecture_review_board",
        name="Architecture Review Board",
        description="Comprehensive architecture review with multiple perspectives",
        use_case="Designing system architectures, evaluating technical decisions, reviewing infrastructure patterns",
        agents=[
            EnsembleAgent("Analyst", "Analyzer", "Analyzes requirements and constraints"),
            EnsembleAgent("CouncilChair", "Facilitator", "Selects relevant SMEs"),
            EnsembleAgent("cloud_architect", "SME", "Provides cloud architecture expertise", dependencies=["CouncilChair"]),
            EnsembleAgent("security_analyst", "SME", "Provides security review", dependencies=["CouncilChair"]),
            EnsembleAgent("Planner", "Planner", "Creates evaluation plan", dependencies=["Analyst"]),
            EnsembleAgent("Executor", "Synthesizer", "Synthesizes recommendations", dependencies=["Planner", "cloud_architect", "security_analyst"]),
            EnsembleAgent("Critic", "Reviewer", "Challenges assumptions", dependencies=["Executor"]),
            EnsembleAgent("Reviewer", "Finalizer", "Produces final recommendations", dependencies=["Critic"]),
        ],
        estimated_duration="10-15 minutes",
        tier_requirement=3,
        icon="🏛️",
        color="#7950f2",
    ),
    EnsemblePattern(
        pattern_id="code_sprint",
        name="Code Sprint",
        description="Rapid code generation with quality gates",
        use_case="Feature development, bug fixes, code refactoring",
        agents=[
            EnsembleAgent("Analyst", "Analyzer", "Analyzes requirements"),
            EnsembleAgent("Planner", "Planner", "Creates implementation plan"),
            EnsembleAgent("Executor", "Generator", "Generates code", dependencies=["Planner"]),
            EnsembleAgent("CodeReviewer", "Reviewer", "Reviews for quality and security", dependencies=["Executor"]),
            EnsembleAgent("Verifier", "Validator", "Validates correctness", dependencies=["Executor"]),
            EnsembleAgent("Formatter", "Formatter", "Formats final output", dependencies=["CodeReviewer", "Verifier"]),
        ],
        estimated_duration="5-8 minutes",
        tier_requirement=2,
        icon="💻",
        color="#28a745",
    ),
    EnsemblePattern(
        pattern_id="research_council",
        name="Research Council",
        description="Comprehensive research with multiple SME perspectives",
        use_case="Technology evaluation, market research, competitive analysis",
        agents=[
            EnsembleAgent("Analyst", "Coordinator", "Defines research scope"),
            EnsembleAgent("CouncilChair", "SME Selector", "Selects domain experts"),
            EnsembleAgent("Researcher", "Researcher", "Conducts web research"),
            EnsembleAgent("ai_ml_engineer", "SME", "AI/ML domain perspective", dependencies=["CouncilChair"]),
            EnsembleAgent("data_engineer", "SME", "Data engineering perspective", dependencies=["CouncilChair"]),
            EnsembleAgent("Executor", "Synthesizer", "Synthesizes findings", dependencies=["Researcher", "ai_ml_engineer", "data_engineer"]),
            EnsembleAgent("Verifier", "Fact-checker", "Verifies sources", dependencies=["Executor"]),
        ],
        estimated_duration="8-12 minutes",
        tier_requirement=3,
        icon="🔍",
        color="#17a2b8",
    ),
    EnsemblePattern(
        pattern_id="document_assembly",
        name="Document Assembly",
        description="Multi-format document creation with technical review",
        use_case="Technical documentation, API docs, user guides, reports",
        agents=[
            EnsembleAgent("Analyst", "Planner", "Analyzes documentation requirements"),
            EnsembleAgent("technical_writer", "SME", "Provides writing expertise"),
            EnsembleAgent("Executor", "Author", "Drafts content"),
            EnsembleAgent("Critic", "Reviewer", "Reviews for clarity", dependencies=["Executor"]),
            EnsembleAgent("Formatter", "Publisher", "Formats in multiple outputs", dependencies=["Critic"]),
        ],
        estimated_duration="5-10 minutes",
        tier_requirement=2,
        icon="📚",
        color="#fd7e14",
    ),
    EnsemblePattern(
        pattern_id="requirements_workshop",
        name="Requirements Workshop",
        description="Requirements elicitation and analysis",
        use_case="Gathering requirements, defining acceptance criteria, user stories",
        agents=[
            EnsembleAgent("Clarifier", "Interviewer", "Identifies unclear requirements"),
            EnsembleAgent("business_analyst", "SME", "Provides BA expertise"),
            EnsembleAgent("Analyst", "Analyzer", "Analyzes requirements"),
            EnsembleAgent("Planner", "Structurer", "Structures user stories", dependencies=["Analyst"]),
            EnsembleAgent("Critic", "Validator", "Challenges assumptions", dependencies=["Planner"]),
            EnsembleAgent("Formatter", "Documenter", "Creates requirement document", dependencies=["Critic"]),
        ],
        estimated_duration="6-10 minutes",
        tier_requirement=2,
        icon="📋",
        color="#20c997",
    ),
]


# =============================================================================
# Helper Functions
# =============================================================================

def get_all_ensembles() -> List[EnsemblePattern]:
    """Get all ensemble patterns."""
    return ENSEMBLE_PATTERNS


def get_ensemble(pattern_id: str) -> Optional[EnsemblePattern]:
    """Get an ensemble by ID."""
    for ensemble in ENSEMBLE_PATTERNS:
        if ensemble.pattern_id == pattern_id:
            return ensemble
    return None


# =============================================================================
# Rendering
# =============================================================================

def render_ensembles_catalogue() -> None:
    """Render the ensembles catalogue page."""
    st.markdown("### 🎭 Ensemble Patterns")
    st.caption("Pre-configured agent workflows for common tasks")

    st.markdown("---")

    # Info
    st.info("""
    **Ensemble Patterns** are pre-configured workflows that combine multiple agents
    to accomplish specific types of tasks efficiently. Each pattern specifies:
    - Which agents to use
    - The order of execution
    - Dependencies between agents
    - Expected duration and complexity tier
    """)

    st.markdown("---")

    # Get all ensembles
    all_ensembles = get_all_ensembles()

    # Filter options
    col1, col2 = st.columns(2)

    with col1:
        tier_filter = st.multiselect(
            "Filter by Tier",
            [1, 2, 3, 4],
            default=[1, 2, 3, 4],
        )

    with col2:
        search_query = st.text_input("🔍 Search", placeholder="Search ensembles...")

    st.markdown("---")

    # Filter
    filtered_ensembles = [
        e for e in all_ensembles
        if e.tier_requirement in tier_filter
        and (not search_query or search_query.lower() in e.name.lower() or search_query.lower() in e.description.lower())
    ]

    # Show count
    st.caption(f"Showing {len(filtered_ensembles)} of {len(all_ensembles)} ensemble(s)")

    st.markdown("---")

    # Render ensembles
    for ensemble in filtered_ensembles:
        render_ensemble_card(ensemble)


def render_ensemble_card(ensemble: EnsemblePattern) -> None:
    """Render an ensemble card."""
    st.markdown(f"""
    <div style="
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        background: white;
    ">
        <div style="display: flex; justify-content: space-between; align-items: start;">
            <div>
                <h4 style="margin: 0;">{ensemble.icon} {ensemble.name}</h4>
                <span style="
                    background: {ensemble.color}20;
                    color: {ensemble.color};
                    padding: 4px 12px;
                    border-radius: 12px;
                    font-size: 12px;
                    font-weight: bold;
                ">Tier {ensemble.tier_requirement}</span>
                <span style="color: #666; font-size: 14px; margin-left: 12px;">
                    ⏱️ {ensemble.estimated_duration}
                </span>
            </div>
        </div>
        <p style="margin: 12px 0; color: #333;">{ensemble.description}</p>
        <div style="background: #f8f9fa; padding: 12px; border-radius: 8px; margin-top: 12px;">
            <strong>Use Case:</strong> {ensemble.use_case}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Agent count
    st.caption(f"**{len(ensemble.agents)} agents** in this pattern")

    # Show agents button
    if st.button(f"👥 View Agents ({len(ensemble.agents)})", key=f"agents_{ensemble.pattern_id}"):
        st.session_state.selected_ensemble = ensemble.pattern_id
        st.rerun()


def render_ensemble_detail(pattern_id: str) -> None:
    """Render detailed view of an ensemble."""
    ensemble = get_ensemble(pattern_id)

    if not ensemble:
        st.error(f"Ensemble '{pattern_id}' not found")
        return

    # Header
    st.markdown(f"""
    <div style="
        background: {ensemble.color}10;
        border-left: 4px solid {ensemble.color};
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 24px;
    ">
        <h2 style="margin: 0;">{ensemble.icon} {ensemble.name}</h2>
        <p style="margin: 8px 0 0 0; color: #666;">{ensemble.description}</p>
    </div>
    """, unsafe_allow_html=True)

    # Metadata
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Agents", len(ensemble.agents))
    with col2:
        st.metric("Tier", ensemble.tier_requirement)
    with col3:
        st.metric("Duration", ensemble.estimated_duration)

    # Use case
    st.markdown("### 📋 Use Case")
    st.info(ensemble.use_case)

    # Agent workflow
    st.markdown("### 👥 Agent Workflow")

    # Visual workflow
    render_workflow_diagram(ensemble)

    # Agent details
    st.markdown("### 📝 Agent Details")

    for i, agent in enumerate(ensemble.agents):
        with st.expander(f"{i + 1}. {agent.name} — {agent.role}"):
            st.markdown(f"**{agent.description}**")

            if agent.dependencies:
                st.write(f"**Dependencies:** {', '.join(agent.dependencies)}")
            else:
                st.write("**Dependencies:** None (can start immediately)")

    # Configuration JSON
    with st.expander("🔧 Configuration"):
        config = {
            "pattern_id": ensemble.pattern_id,
            "name": ensemble.name,
            "tier_requirement": ensemble.tier_requirement,
            "agents": [
                {
                    "name": a.name,
                    "role": a.role,
                    "dependencies": a.dependencies,
                }
                for a in ensemble.agents
            ],
        }
        st.json(config)

    # Back button
    if st.button("← Back to Ensembles"):
        st.session_state.selected_ensemble = None
        st.rerun()


def render_workflow_diagram(ensemble: EnsemblePattern) -> None:
    """Render a visual workflow diagram."""
    st.markdown("""
    <style>
        .workflow-node {
            background: white;
            border: 2px solid #007bff;
            border-radius: 8px;
            padding: 8px 16px;
            margin: 4px;
            display: inline-block;
            font-size: 12px;
        }
        .workflow-arrow {
            color: #6c757d;
            font-size: 18px;
            margin: 0 8px;
        }
    </style>
    """, unsafe_allow_html=True)

    # Group agents by dependency level
    levels: List[List[EnsembleAgent]] = []

    remaining = ensemble.agents.copy()
    max_iterations = len(ensemble.agents) + 1

    for _ in range(max_iterations):
        # Find agents with all dependencies satisfied
        ready = []
        for agent in remaining:
            deps_satisfied = all(
                dep not in [a.name for a in remaining]
                for dep in agent.dependencies
            )
            if deps_satisfied:
                ready.append(agent)

        if not ready:
            break

        levels.append(ready)
        for agent in ready:
            remaining.remove(agent)

    # Add any remaining (circular dependencies or orphans)
    if remaining:
        levels.append(remaining)

    # Render levels
    for i, level in enumerate(levels):
        # Create columns for this level
        cols = st.columns(len(level))

        for j, agent in enumerate(level):
            if j < len(cols):
                with cols[j]:
                    st.markdown(f"""
                    <div class="workflow-node">
                        <strong>{agent.name}</strong><br>
                        <small>{agent.role}</small>
                    </div>
                    """, unsafe_allow_html=True)

        # Add arrow between levels
        if i < len(levels) - 1:
            st.markdown("<div style='text-align: center;' class='workflow-arrow'>↓</div>", unsafe_allow_html=True)
