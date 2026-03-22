"""
SME Browser Page - Subject Matter Expert Registry

Browse and view details of available SME (Subject Matter Expert) personas.
SMEs provide domain expertise on-demand during agent execution.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import sys
from pathlib import Path

import streamlit as st

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.core.sme_registry import (
    SMEPersona as RegistrySMEPersona,
    get_persona,
    get_all_personas as get_registry_personas,
    find_personas_by_keywords,
    find_personas_by_domain,
    get_persona_for_display,
)


# =============================================================================
# SME Data Structures (for display)
# =============================================================================

class InteractionMode(str, Enum):
    """SME interaction modes."""
    ADVISOR = "advisor"
    CO_EXECUTOR = "co-executor"
    DEBATER = "debater"


class SMEDomain(str, Enum):
    """SME domain categories."""
    IAM = "iam"
    CLOUD = "cloud"
    SECURITY = "security"
    DATA = "data"
    AI_ML = "ai_ml"
    TESTING = "testing"
    BUSINESS = "business"
    DOCUMENTATION = "documentation"
    DEVOPS = "devops"
    FRONTEND = "frontend"


@dataclass
class SMEPersona:
    """SME persona data for display."""
    persona_id: str
    name: str
    domain: str
    description: str
    trigger_keywords: List[str]
    skill_files: List[str]
    interaction_modes: List[str]
    default_model: str
    icon: str = "👤"
    color: str = "#007bff"


# Icons for different SME domains
DOMAIN_ICONS = {
    "Identity and Access Management": "🔐",
    "Cloud Infrastructure Architecture": "☁️",
    "Application & Infrastructure Security": "🛡️",
    "Data Engineering": "📊",
    "AI/ML Engineering": "🤖",
    "Test Engineering": "🧪",
    "Business Analysis": "💼",
    "Technical Writing": "📝",
    "DevOps Engineering": "⚙️",
    "Frontend Development & UI/UX": "🎨",
}

# Default icon for unknown domains
DEFAULT_ICON = "👤"


# =============================================================================
# Helper Functions
# =============================================================================

def _get_domain_icon(domain: str) -> str:
    """Get icon for a domain."""
    return DOMAIN_ICONS.get(domain, DEFAULT_ICON)


def _convert_registry_persona(persona: RegistrySMEPersona) -> SMEPersona:
    """Convert a registry SMEPersona to display SMEPersona."""
    # Convert InteractionMode enums to strings
    modes = [m.value if hasattr(m, 'value') else m for m in persona.interaction_modes]

    return SMEPersona(
        persona_id=persona.persona_id,
        name=persona.name,
        domain=persona.domain,
        description=persona.description or persona.name,  # Fallback to name if description is empty
        trigger_keywords=persona.trigger_keywords,
        skill_files=persona.skill_files,
        interaction_modes=modes,
        default_model=persona.default_model,
        icon=_get_domain_icon(persona.domain),
        color="#007bff",
    )


def get_all_personas() -> List[SMEPersona]:
    """Get all SME personas from the real registry."""
    registry_personas = get_registry_personas()  # Returns Dict[str, SMEPersona]
    return [
        _convert_registry_persona(persona)
        for persona in registry_personas.values()
    ]


def get_persona(persona_id: str) -> Optional[SMEPersona]:
    """Get a specific persona by ID from the real registry."""
    persona = get_persona(persona_id)  # From src.core.sme_registry
    if persona:
        return _convert_registry_persona(persona)
    return None


def find_personas_by_keywords(keywords: List[str]) -> List[SMEPersona]:
    """Find personas matching given keywords from the real registry."""
    registry_personas = find_personas_by_keywords(keywords)  # From src.core.sme_registry
    return [
        _convert_registry_persona(persona)
        for persona in registry_personas
    ]


def find_personas_by_domain(domain: str) -> List[SMEPersona]:
    """Find personas by domain from the real registry."""
    registry_personas = find_personas_by_domain([domain])  # From src.core.sme_registry
    return [
        _convert_registry_persona(persona)
        for persona in registry_personas
    ]


def get_domains() -> List[str]:
    """Get list of all domains from the real registry."""
    registry_personas = get_registry_personas()
    domains = set(persona.domain for persona in registry_personas.values())
    return sorted(domains)


# =============================================================================
# Rendering
# =============================================================================

def render_sme_browser() -> None:
    """Render the SME personas browser page."""
    st.markdown("### 👤 SME Personas")
    st.caption("Subject Matter Experts available for on-demand consultation")

    st.markdown("---")

    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        domains = get_domains()
        domain_filter = st.multiselect(
            "Filter by Domain",
            domains,
            default=domains,
        )

    with col2:
        # Get all unique interaction modes from personas
        all_personas = get_all_personas()
        all_modes = set()
        for persona in all_personas:
            all_modes.update(persona.interaction_modes)
        mode_filter = st.multiselect(
            "Filter by Interaction Mode",
            sorted(all_modes),
            default=list(all_modes),
        )

    with col3:
        search_query = st.text_input("🔍 Search", placeholder="Search personas...")

    st.markdown("---")

    # Get all personas
    all_personas = get_all_personas()

    # Filter
    filtered_personas = []

    for persona in all_personas:
        # Domain filter
        if domain_filter and persona.domain not in domain_filter:
            continue

        # Mode filter
        if mode_filter:
            persona_modes = [m if isinstance(m, str) else m.value for m in persona.interaction_modes]
            if not any(m in mode_filter for m in persona_modes):
                continue

        # Search filter
        if search_query:
            query_lower = search_query.lower()
            searchable = (
                persona.name.lower() +
                persona.domain.lower() +
                persona.description.lower() +
                " ".join(persona.trigger_keywords).lower()
            )
            if query_lower not in searchable:
                continue

        filtered_personas.append(persona)

    # Show count
    st.caption(f"Showing {len(filtered_personas)} of {len(all_personas)} persona(s)")

    st.markdown("---")

    # View toggle
    view_mode = st.radio(
        "View Mode",
        options=["Cards", "Table", "Compact"],
        horizontal=True,
        label_visibility="collapsed",
    )

    # Render personas
    if view_mode == "Cards":
        render_persona_cards(filtered_personas)
    elif view_mode == "Table":
        render_persona_table(filtered_personas)
    else:
        render_persona_compact(filtered_personas)


def render_persona_cards(personas: List[SMEPersona]) -> None:
    """Render personas as cards."""
    # Determine active SMEs from session state
    active_smes = st.session_state.get("active_smes", st.session_state.get("last_active_smes", []))

    cols = st.columns(2)

    for i, persona in enumerate(personas):
        with cols[i % 2]:
            is_active = persona.persona_id in active_smes or persona.name in active_smes
            border_color = "#28a745" if is_active else "#e0e0e0"
            border_width = "2px" if is_active else "1px"
            active_badge_html = (
                '<span style="background: #28a745; color: white; padding: 2px 10px; '
                'border-radius: 10px; font-size: 11px; margin-left: 8px; font-weight: bold;">'
                'ACTIVE</span>'
                if is_active else ""
            )

            st.markdown(f"""
            <div style="
                border: {border_width} solid {border_color};
                border-radius: 12px;
                padding: 16px;
                margin-bottom: 16px;
                background: {"#f0fff0" if is_active else "white"};
            ">
                <div style="display: flex; align-items: center; margin-bottom: 12px;">
                    <span style="font-size: 32px; margin-right: 12px;">{persona.icon}</span>
                    <div>
                        <h4 style="margin: 0;">{persona.name}{active_badge_html}</h4>
                        <small style="color: {persona.color};">{persona.domain}</small>
                    </div>
                </div>
                <p style="margin: 8px 0; color: #333;">{persona.description}</p>
            </div>
            """, unsafe_allow_html=True)

            # Interaction modes
            modes_badges = []
            for mode in persona.interaction_modes:
                mode_colors = {
                    InteractionMode.ADVISOR: "#28a745",
                    InteractionMode.CO_EXECUTOR: "#007bff",
                    InteractionMode.DEBATER: "#fd7e14",
                }
                color = mode_colors.get(mode, "#6c757d")
                modes_badges.append(
                    f"<span style='background: {color}20; color: {color}; "
                    f"padding: 2px 8px; border-radius: 12px; font-size: 11px;'>"
                    f"{mode.value.replace('-', ' ').title()}</span>"
                )
            st.markdown(f"<div>{' '.join(modes_badges)}</div>", unsafe_allow_html=True)

            # Keywords
            with st.expander(f"🔑 Trigger Keywords ({len(persona.trigger_keywords)})"):
                st.write(", ".join(persona.trigger_keywords))

            # Skills
            if persona.skill_files:
                with st.expander(f"🎯 Skills ({len(persona.skill_files)})"):
                    for skill in persona.skill_files:
                        st.write(f"• `{skill}`")

            # Model
            st.caption(f"Default Model: `{persona.default_model}`")

            # View details button
            if st.button(f"📖 View Details", key=f"view_{persona.persona_id}"):
                st.session_state.selected_persona = persona.persona_id
                st.rerun()


def render_persona_table(personas: List[SMEPersona]) -> None:
    """Render personas as a table."""
    # Create table data
    table_data = []

    for persona in personas:
        table_data.append({
            "Persona": f"{persona.icon} {persona.name}",
            "Domain": persona.domain,
            "Modes": ", ".join(m.value for m in persona.interaction_modes),
            "Model": persona.default_model,
            "Keywords": len(persona.trigger_keywords),
        })

    st.dataframe(
        table_data,
        use_container_width=True,
        hide_index=True,
    )


def render_persona_compact(personas: List[SMEPersona]) -> None:
    """Render personas in compact list view."""
    for persona in personas:
        with st.expander(f"{persona.icon} {persona.name} — {persona.domain}"):
            st.markdown(f"**{persona.description}**")

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Interaction Modes:**")
                for mode in persona.interaction_modes:
                    st.write(f"• {mode.value.replace('-', ' ').title()}")

            with col2:
                st.write("**Skills:**")
                for skill in persona.skill_files:
                    st.write(f"• `{skill}`")

            st.write(f"**Default Model:** `{persona.default_model}`")


def render_persona_detail(persona_id: str) -> None:
    """Render detailed view of a persona."""
    persona = get_persona(persona_id)

    if not persona:
        st.error(f"Persona '{persona_id}' not found")
        return

    # Header
    st.markdown(f"""
    <div style="
        background: {persona.color}10;
        border-left: 4px solid {persona.color};
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 24px;
    ">
        <div style="display: flex; align-items: center;">
            <span style="font-size: 48px; margin-right: 16px;">{persona.icon}</span>
            <div>
                <h2 style="margin: 0;">{persona.name}</h2>
                <p style="margin: 4px 0 0 0; color: {persona.color}; font-weight: bold;">
                    {persona.domain}
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Description
    st.markdown("### Description")
    st.markdown(persona.description)

    # Metadata
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Interaction Modes", len(persona.interaction_modes))

    with col2:
        st.metric("Skills", len(persona.skill_files))

    with col3:
        st.metric("Trigger Keywords", len(persona.trigger_keywords))

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Modes", "Keywords", "Skills", "Configuration"])

    with tab1:
        st.markdown("### Interaction Modes")
        for mode in persona.interaction_modes:
            mode_descriptions = {
                InteractionMode.ADVISOR: (
                    "**Advisor Mode**: Provides domain review, recommendations, "
                    "and guidance without directly contributing content."
                ),
                InteractionMode.CO_EXECUTOR: (
                    "**Co-executor Mode**: Actively contributes sections of content "
                    "based on domain expertise."
                ),
                InteractionMode.DEBATER: (
                    "**Debater Mode**: Argues specific positions in adversarial "
                    "debate to stress-test solutions."
                ),
            }
            st.info(mode_descriptions.get(mode, f"**{mode.value}**"))

    with tab2:
        st.markdown("### Trigger Keywords")
        st.markdown("These keywords trigger the automatic selection of this SME:")
        for keyword in persona.trigger_keywords:
            st.markdown(f"- `{keyword}`")

    with tab3:
        st.markdown("### Skills")
        if persona.skill_files:
            st.markdown("This SME has access to the following skills:")
            for skill in persona.skill_files:
                st.markdown(f"- **{skill}**")
        else:
            st.info("No specific skills assigned")

    with tab4:
        st.markdown("### Configuration")
        st.json({
            "persona_id": persona.persona_id,
            "name": persona.name,
            "domain": persona.domain,
            "default_model": persona.default_model,
            "interaction_modes": [m.value for m in persona.interaction_modes],
            "trigger_keywords": persona.trigger_keywords,
            "skill_files": persona.skill_files,
        })

    # Back button
    if st.button("← Back to Personas"):
        st.session_state.selected_persona = None
        st.rerun()
