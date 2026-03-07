"""
Debate Viewer Component - Self-Play Debate Visualization

Displays transcripts of adversarial debates between agents with
perspective comparison, argument tracking, and consensus scoring.
"""

from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px


# =============================================================================
# Debate Data Structures
# =============================================================================

class DebatePhase(str, Enum):
    """Phases of a debate."""
    OPENING = "opening"
    ARGUMENT = "argument"
    REBUTTAL = "rebuttal"
    CLOSING = "closing"
    CONSENSUS = "consensus"


class ArgumentType(str, Enum):
    """Types of arguments."""
    FACTUAL = "factual"
    LOGICAL = "logical"
    ETHICAL = "ethical"
    PRACTICAL = "practical"
    TECHNICAL = "technical"


class PersuasionStrength(str, Enum):
    """Strength of an argument."""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


@dataclass
class Argument:
    """A single argument in a debate."""
    argument_id: str
    speaker: str  # Agent name
    content: str
    argument_type: ArgumentType
    strength: PersuasionStrength
    timestamp: datetime
    phase: DebatePhase
    references: List[str] = field(default_factory=list)  # IDs of arguments being responded to
    evidence: List[str] = field(default_factory=list)
    score: float = 0.0  # Assigned quality score


@dataclass
class DebatePerspective:
    """A perspective (side) in a debate."""
    perspective_id: str
    name: str
    description: str
    agent_name: str
    position: str  # The stance being argued
    opening_statement: str
    color: str = "#007bff"


@dataclass
class DebateRound:
    """A round of argument exchange."""
    round_number: int
    phase: DebatePhase
    arguments: List[Argument]
    start_time: datetime
    end_time: Optional[datetime] = None


@dataclass
class DebateConsensus:
    """Consensus outcome of a debate."""
    consensus_reached: bool
    final_position: str
    confidence_score: float  # 0.0 to 1.0
    winning_perspective: Optional[str]
    synthesis: str
    key_agreements: List[str]
    remaining_disagreements: List[str]


@dataclass
class DebateTranscript:
    """Complete transcript of a debate."""
    debate_id: str
    topic: str
    perspectives: List[DebatePerspective]
    rounds: List[DebateRound]
    consensus: Optional[DebateConsensus] = None
    start_time: datetime = None
    end_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_arguments(self) -> int:
        """Total number of arguments."""
        return sum(len(r.arguments) for r in self.rounds)

    @property
    def duration(self) -> Optional[float]:
        """Debate duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


# =============================================================================
# Mock Debate Generator (for testing)
# =============================================================================

def generate_mock_debate() -> DebateTranscript:
    """Generate a mock debate for testing the UI."""
    now = datetime.now()

    perspectives = [
        DebatePerspective(
            perspective_id="prop_1",
            name="Proponent",
            description="Argues in favor of the proposition",
            agent_name="Security Analyst",
            position="All API endpoints should require authentication by default",
            opening_statement="Security by default is essential in modern software development. "
                           "Requiring authentication for all API endpoints prevents unauthorized access "
                           "and reduces the attack surface significantly.",
            color="#28a745",
        ),
        DebatePerspective(
            perspective_id="opp_1",
            name="Opponent",
            description="Argues against the proposition",
            agent_name="UX Designer",
            position="Public APIs have valid use cases that don't require authentication",
            opening_statement="While security is important, mandating authentication for all endpoints "
                           "creates unnecessary friction. Public APIs for catalogs, documentation, "
                           "and status pages serve legitimate purposes without auth.",
            color="#dc3545",
        ),
    ]

    arguments = [
        # Round 1 - Opening
        Argument(
            argument_id="arg_1",
            speaker="Security Analyst",
            content="The OWASP Top 10 consistently lists broken access control as a top vulnerability. "
                   "Default authentication prevents this class of vulnerability entirely.",
            argument_type=ArgumentType.FACTUAL,
            strength=PersuasionStrength.STRONG,
            timestamp=now,
            phase=DebatePhase.OPENING,
            evidence=["OWASP Top 10 2021 - A01:2021"],
            score=0.85,
        ),
        Argument(
            argument_id="arg_2",
            speaker="UX Designer",
            content="Consider a public API like GitHub's README endpoint. It's designed for public access. "
                   "Forcing authentication would break legitimate integrations and increase load.",
            argument_type=ArgumentType.PRACTICAL,
            strength=PersuasionStrength.MODERATE,
            timestamp=now,
            phase=DebatePhase.OPENING,
            references=["arg_1"],
            score=0.70,
        ),
        # Round 2 - Rebuttal
        Argument(
            argument_id="arg_3",
            speaker="Security Analyst",
            content="Opt-in authentication is still superior. Start authenticated, then explicitly expose "
                   "specific endpoints as public. This creates a secure-by-default audit trail.",
            argument_type=ArgumentType.LOGICAL,
            strength=PersuasionStrength.VERY_STRONG,
            timestamp=now,
            phase=DebatePhase.REBUTTAL,
            references=["arg_2"],
            score=0.90,
        ),
        Argument(
            argument_id="arg_4",
            speaker="UX Designer",
            content="That increases development complexity. Teams must remember to mark public endpoints. "
                   "Forgetting means broken integrations. The cost of errors falls on users.",
            argument_type=ArgumentType.PRACTICAL,
            strength=PersuasionStrength.MODERATE,
            timestamp=now,
            phase=DebatePhase.REBUTTAL,
            references=["arg_3"],
            score=0.75,
        ),
    ]

    rounds = [
        DebateRound(
            round_number=1,
            phase=DebatePhase.OPENING,
            arguments=[arguments[0], arguments[1]],
            start_time=now,
        ),
        DebateRound(
            round_number=2,
            phase=DebatePhase.REBUTTAL,
            arguments=[arguments[2], arguments[3]],
            start_time=now,
        ),
    ]

    consensus = DebateConsensus(
        consensus_reached=True,
        final_position="Authentication should be default with explicit opt-out for public endpoints",
        confidence_score=0.75,
        winning_perspective="prop_1",
        synthesis="Both parties agree security is important but disagree on implementation. "
                  "The consensus is to require authentication by default while allowing "
                  "explicitly-marked public endpoints. The security benefit of default-auth "
                  "outweighs the development overhead.",
        key_agreements=[
            "Security is a critical concern",
            "Some endpoints legitimately need public access",
            "The solution should minimize errors",
        ],
        remaining_disagreements=[
            "Whether the development overhead is acceptable",
            "How to handle migration of existing systems",
        ],
    )

    return DebateTranscript(
        debate_id=f"debate_{int(now.timestamp())}",
        topic="Should all API endpoints require authentication by default?",
        perspectives=perspectives,
        rounds=rounds,
        consensus=consensus,
        start_time=now,
        end_time=now,
    )


# =============================================================================
# Rendering Functions
# =============================================================================

def render_debate_viewer() -> None:
    """Render the main debate viewer interface."""
    st.markdown("### 🎭 Debate Viewer")
    st.caption("Self-play debate transcripts with argument analysis")

    st.markdown("---")

    # Load or select debate
    debate = load_debate()

    if not debate:
        render_empty_debates()
        return

    # Debate header
    render_debate_header(debate)

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📜 Transcript",
        "🔍 Analysis",
        "📊 Visualizations",
        "🤝 Consensus",
        "ℹ️ Metadata",
    ])

    with tab1:
        render_debate_transcript(debate)

    with tab2:
        render_debate_analysis(debate)

    with tab3:
        render_debate_visualizations(debate)

    with tab4:
        render_debate_consensus(debate)

    with tab5:
        render_debate_metadata(debate)


def load_debate() -> Optional[DebateTranscript]:
    """Load a debate from session state or generate mock."""
    # Check if we have a debate in session
    if "current_debate" in st.session_state:
        return st.session_state.current_debate

    # Generate mock for demo
    if "demo_debate_loaded" not in st.session_state:
        st.session_state.current_debate = generate_mock_debate()
        st.session_state.demo_debate_loaded = True

    return st.session_state.get("current_debate")


def render_empty_debates() -> None:
    """Render empty state when no debates available."""
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 40px;">
            <div style="font-size: 64px; margin-bottom: 16px;">🎭</div>
            <h3>No Debates Available</h3>
            <p style="color: #666;">
                Self-play debates are generated during Tier 3-4 tasks
                when multiple perspectives need to be evaluated.
            </p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🎲 Load Demo Debate", use_container_width=True):
            st.session_state.current_debate = generate_mock_debate()
            st.session_state.demo_debate_loaded = True
            st.rerun()


def render_debate_header(debate: DebateTranscript) -> None:
    """Render the debate header with overview stats."""
    # Topic
    st.markdown(f"#### 📌 {debate.topic}")

    # Stats
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Rounds", len(debate.rounds))

    with col2:
        st.metric("Arguments", debate.total_arguments)

    with col3:
        duration = debate.duration
        st.metric("Duration", f"{int(duration)}s" if duration else "N/A")

    with col4:
        if debate.consensus:
            confidence = int(debate.consensus.confidence_score * 100)
            st.metric("Consensus", f"{confidence}%")

    # Perspectives
    st.markdown("**Perspectives:**")

    cols = st.columns(len(debate.perspectives))

    for i, perspective in enumerate(debate.perspectives):
        with cols[i]:
            st.markdown(f"""
            <div style="
                background: {perspective.color}15;
                border-left: 4px solid {perspective.color};
                padding: 12px;
                border-radius: 8px;
            ">
                <strong>{perspective.name}</strong>
                <div style="font-size: 12px; color: #666;">{perspective.agent_name}</div>
                <div style="font-size: 11px; margin-top: 4px;">{perspective.position[:60]}...</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")


def render_debate_transcript(debate: DebateTranscript) -> None:
    """Render the debate transcript."""
    show_all = st.checkbox("Show all arguments", value=True)

    for round_num, round_data in enumerate(debate.rounds, 1):
        with st.expander(
            f"🔵 Round {round_num}: {round_data.phase.value.title()} "
            f"({len(round_data.arguments)} arguments)",
            expanded=show_all or round_num == len(debate.rounds),
        ):
            for arg in round_data.arguments:
                render_argument(arg, debate.perspectives)


def render_argument(argument: Argument, perspectives: List[DebatePerspective]) -> None:
    """Render a single argument."""
    # Get speaker's perspective color
    perspective = next((p for p in perspectives if p.agent_name == argument.speaker), None)
    color = perspective.color if perspective else "#6c757d"

    # Argument type badge
    type_colors = {
        ArgumentType.FACTUAL: "#007bff",
        ArgumentType.LOGICAL: "#28a745",
        ArgumentType.ETHICAL: "#fd7e14",
        ArgumentType.PRACTICAL: "#6f42c1",
        ArgumentType.TECHNICAL: "#17a2b8",
    }
    type_color = type_colors.get(argument.argument_type, "#6c757d")

    # Strength indicator
    strength_symbols = {
        PersuasionStrength.WEAK: "💪",
        PersuasionStrength.MODERATE: "💪💪",
        PersuasionStrength.STRONG: "💪💪💪",
        PersuasionStrength.VERY_STRONG: "💪💪💪💪",
    }

    st.markdown(f"""
    <div style="
        border: 1px solid #e0e0e0;
        border-left: 4px solid {color};
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 12px;
    ">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <strong>{argument.speaker}</strong>
                <span style="
                    background: {type_color}20;
                    color: {type_color};
                    padding: 2px 8px;
                    border-radius: 12px;
                    font-size: 11px;
                    margin-left: 8px;
                ">{argument.argument_type.value}</span>
            </div>
            <div style="text-align: right;">
                <span>{strength_symbols.get(argument.strength, '')}</span>
                <span style="color: #666; font-size: 12px;">
                    Score: {argument.score:.2f}
                </span>
            </div>
        </div>
        <div style="margin-top: 8px;">{argument.content}</div>
    """, unsafe_allow_html=True)

    # Evidence
    if argument.evidence:
        with st.expander(f"📚 Evidence ({len(argument.evidence)})"):
            for evidence in argument.evidence:
                st.markdown(f"- {evidence}")

    # References
    if argument.references:
        refs = ", ".join(argument.references)
        st.caption(f"↩️ Responding to: {refs}")

    st.markdown("</div>", unsafe_allow_html=True)


def render_debate_analysis(debate: DebateTranscript) -> None:
    """Render debate analysis."""
    st.markdown("#### Argument Analysis")

    # Arguments by type
    type_counts = {}
    strength_counts = {}

    for round_data in debate.rounds:
        for arg in round_data.arguments:
            type_counts[arg.argument_type.value] = type_counts.get(arg.argument_type.value, 0) + 1
            strength_counts[arg.strength.value] = strength_counts.get(arg.strength.value, 0) + 1

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**By Type:**")
        for arg_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            st.markdown(f"- {arg_type.title()}: {count}")

    with col2:
        st.markdown("**By Strength:**")
        for strength, count in sorted(strength_counts.items()):
            st.markdown(f"- {strength.title()}: {count}")

    st.markdown("---")

    # Average scores by speaker
    speaker_scores = {}

    for round_data in debate.rounds:
        for arg in round_data.arguments:
            if arg.speaker not in speaker_scores:
                speaker_scores[arg.speaker] = []
            speaker_scores[arg.speaker].append(arg.score)

    st.markdown("#### Performance by Speaker")

    for speaker, scores in speaker_scores.items():
        avg = sum(scores) / len(scores)
        st.metric(speaker, f"{avg:.2f}", f"{len(scores)} arguments")

    st.markdown("---")

    # Timeline
    st.markdown("#### Argument Timeline")

    timeline_data = []
    for round_data in debate.rounds:
        for arg in round_data.arguments:
            perspective = next((p for p in debate.perspectives if p.agent_name == arg.speaker), None)
            timeline_data.append({
                "Round": round_data.round_number,
                "Speaker": arg.speaker,
                "Type": arg.argument_type.value,
                "Score": arg.score,
                "Perspective": perspective.name if perspective else "Unknown",
            })

    if timeline_data:
        st.dataframe(timeline_data, use_container_width=True, hide_index=True)


def render_debate_visualizations(debate: DebateTranscript) -> None:
    """Render debate visualizations."""
    # Score progression
    st.markdown("#### Score Progression")

    scores_by_round = {p.name: [] for p in debate.perspectives}
    rounds = []

    for round_data in debate.rounds:
        rounds.append(round_data.round_number)

        round_scores = {p.name: 0.0 for p in debate.perspectives}

        for arg in round_data.arguments:
            perspective = next((p for p in debate.perspectives if p.agent_name == arg.speaker), None)
            if perspective:
                round_scores[perspective.name] += arg.score

        for perspective_name in scores_by_round:
            scores_by_round[perspective_name].append(round_scores[perspective_name])

    # Create line chart
    fig = go.Figure()

    for perspective in debate.perspectives:
        fig.add_trace(go.Scatter(
            x=rounds,
            y=scores_by_round[perspective.name],
            name=perspective.name,
            line=dict(color=perspective.color),
            mode="lines+markers",
        ))

    fig.update_layout(
        title="Cumulative Scores by Round",
        xaxis_title="Round",
        yaxis_title="Cumulative Score",
        hovermode="x unified",
        height=400,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Argument type distribution
    st.markdown("#### Argument Type Distribution")

    type_data = []
    for round_data in debate.rounds:
        for arg in round_data.arguments:
            perspective = next((p for p in debate.perspectives if p.agent_name == arg.speaker), None)
            type_data.append({
                "Speaker": arg.speaker,
                "Type": arg.argument_type.value,
                "Perspective": perspective.name if perspective else "Unknown",
            })

    if type_data:
        fig = px.sunburst(
            type_data,
            path=["Perspective", "Speaker", "Type"],
            title="Argument Distribution by Type and Speaker",
        )

        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Strength distribution
    st.markdown("#### Strength Distribution")

    strength_data = {}
    for perspective in debate.perspectives:
        strength_data[perspective.name] = {s.value: 0 for s in PersuasionStrength}

    for round_data in debate.rounds:
        for arg in round_data.arguments:
            perspective = next((p for p in debate.perspectives if p.agent_name == arg.speaker), None)
            if perspective:
                strength_data[perspective.name][arg.strength.value] += 1

    # Create stacked bar chart
    fig = go.Figure()

    for strength in PersuasionStrength:
        y_values = []
        for perspective in debate.perspectives:
            y_values.append(strength_data[perspective.name][strength.value])

        fig.add_trace(go.Bar(
            name=strength.value.title(),
            x=[p.name for p in debate.perspectives],
            y=y_values,
        ))

    fig.update_layout(
        title="Argument Strength by Perspective",
        xaxis_title="Perspective",
        yaxis_title="Count",
        barmode="stack",
        height=400,
    )

    st.plotly_chart(fig, use_container_width=True)


def render_debate_consensus(debate: DebateTranscript) -> None:
    """Render debate consensus."""
    if not debate.consensus:
        st.info("No consensus reached yet")
        return

    consensus = debate.consensus

    # Consensus status
    if consensus.consensus_reached:
        st.success(f"✅ Consensus Reached (Confidence: {int(consensus.confidence_score * 100)}%)")
    else:
        st.warning("⚠️ No Consensus Reached")

    st.markdown("---")

    # Final position
    st.markdown("#### Final Position")
    st.info(consensus.final_position)

    # Winning perspective
    if consensus.winning_perspective:
        perspective = next((p for p in debate.perspectives if p.perspective_id == consensus.winning_perspective), None)
        if perspective:
            st.markdown(f"**Winning Perspective:** {perspective.name} ({perspective.agent_name})")

    st.markdown("---")

    # Synthesis
    st.markdown("#### Synthesis")
    st.markdown(consensus.synthesis)

    st.markdown("---")

    # Key agreements
    if consensus.key_agreements:
        st.markdown("#### ✅ Key Agreements")
        for agreement in consensus.key_agreements:
            st.markdown(f"- {agreement}")

    # Remaining disagreements
    if consensus.remaining_disagreements:
        st.markdown("#### ⚠️ Remaining Disagreements")
        for disagreement in consensus.remaining_disagreements:
            st.markdown(f"- {disagreement}")


def render_debate_metadata(debate: DebateTranscript) -> None:
    """Render debate metadata."""
    st.markdown("#### Debate Metadata")

    metadata = {
        "Debate ID": debate.debate_id,
        "Topic": debate.topic,
        "Start Time": debate.start_time.isoformat() if debate.start_time else None,
        "End Time": debate.end_time.isoformat() if debate.end_time else None,
        "Duration (s)": debate.duration,
        "Total Rounds": len(debate.rounds),
        "Total Arguments": debate.total_arguments,
        "Perspectives": len(debate.perspectives),
        "Consensus Reached": debate.consensus.consensus_reached if debate.consensus else None,
        "Confidence Score": debate.consensus.confidence_score if debate.consensus else None,
    }

    # Add any custom metadata
    if debate.metadata:
        metadata.update(debate.metadata)

    st.json(metadata)

    # Perspective details
    st.markdown("---")
    st.markdown("#### Perspective Details")

    for perspective in debate.perspectives:
        with st.expander(f"{perspective.name} — {perspective.agent_name}"):
            st.markdown(f"**Position:** {perspective.position}")
            st.markdown(f"**Description:** {perspective.description}")
            st.markdown(f"**Opening Statement:** {perspective.opening_statement}")


# =============================================================================
# Standalone Page Renderer
# =============================================================================

def render_debate_page() -> None:
    """Render the debate page (for use in main app)."""
    render_debate_viewer()
