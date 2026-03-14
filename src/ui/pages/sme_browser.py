"""
SME Browser Page - Subject Matter Expert Registry

Browse and view details of available SME (Subject Matter Expert) personas.
SMEs provide domain expertise on-demand during agent execution.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

import streamlit as st

from src.core.sme_registry import InteractionMode


# =============================================================================
# SME Data Structures
# =============================================================================


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
    """SME persona data."""
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


# =============================================================================
# SME Registry (subset of full registry)
# =============================================================================

SME_REGISTRY_DATA: List[Dict[str, Any]] = [
    {
        "persona_id": "iam_architect",
        "name": "IAM Architect",
        "domain": "Identity and Access Management",
        "description": "Expert in SailPoint, CyberArk, RBAC, identity governance, and privileged access management. Designs secure identity architectures for enterprise environments.",
        "trigger_keywords": ["sailpoint", "cyberark", "rbac", "identity", "azure ad", "okta", "ldap", "privileged access"],
        "skill_files": ["sailpoint-test-engineer", "azure-architect"],
        "interaction_modes": ["advisor", "co-executor"],
        "default_model": "sonnet",
        "icon": "🔐",
        "color": "#7950f2",
    },
    {
        "persona_id": "cloud_architect",
        "name": "Cloud Architect",
        "domain": "Cloud Infrastructure",
        "description": "Specializes in AWS, Azure, and GCP infrastructure design. Expert in serverless, containers, Kubernetes, cloud-native patterns, and multi-cloud strategies.",
        "trigger_keywords": ["aws", "azure", "gcp", "cloud", "serverless", "lambda", "ec2", "kubernetes", "eks", "aks"],
        "skill_files": ["azure-architect", "aws-solutions-architect"],
        "interaction_modes": ["advisor", "co-executor", "debater"],
        "default_model": "sonnet",
        "icon": "☁️",
        "color": "#007bff",
    },
    {
        "persona_id": "security_analyst",
        "name": "Security Analyst",
        "domain": "Security",
        "description": "Focuses on application security, threat modeling, secure coding practices, vulnerability assessment, and security testing. Experienced with OWASP Top 10 and security frameworks.",
        "trigger_keywords": ["security", "vulnerability", "threat", "owasp", "penetration", "security testing", "xss", "csrf", "sqli"],
        "skill_files": ["security-specialist"],
        "interaction_modes": ["advisor", "debater"],
        "default_model": "sonnet",
        "icon": "🛡️",
        "color": "#dc3545",
    },
    {
        "persona_id": "data_engineer",
        "name": "Data Engineer",
        "domain": "Data Engineering",
        "description": "Expert in data pipelines, ETL/ELT processes, data modeling, data warehousing, and big data technologies. Works with SQL, NoSQL, and streaming data platforms.",
        "trigger_keywords": ["data", "etl", "elt", "pipeline", "warehouse", "sql", "nosql", "spark", "kafka", "airflow"],
        "skill_files": ["data-engineering"],
        "interaction_modes": ["advisor", "co-executor"],
        "default_model": "sonnet",
        "icon": "📊",
        "color": "#20c997",
    },
    {
        "persona_id": "ai_ml_engineer",
        "name": "AI/ML Engineer",
        "domain": "AI/Machine Learning",
        "description": "Specializes in machine learning model development, MLOps, feature engineering, and model deployment. Works with TensorFlow, PyTorch, scikit-learn, and ML platforms.",
        "trigger_keywords": ["ml", "machine learning", "ai", "model", "training", "inference", "tensorflow", "pytorch", "mlops"],
        "skill_files": ["ml-engineering"],
        "interaction_modes": ["advisor", "co-executor", "debater"],
        "default_model": "sonnet",
        "icon": "🤖",
        "color": "#fd7e14",
    },
    {
        "persona_id": "test_engineer",
        "name": "Test Engineer",
        "domain": "Quality Assurance",
        "description": "Expert in test automation, test strategy, performance testing, and quality assurance. Works with pytest, unittest, Selenium, and testing frameworks.",
        "trigger_keywords": ["test", "testing", "pytest", "unittest", "selenium", "automation", "quality", "tdd", "bdd"],
        "skill_files": ["test-automation"],
        "interaction_modes": ["advisor", "co-executor"],
        "default_model": "haiku",
        "icon": "🧪",
        "color": "#6f42c1",
    },
    {
        "persona_id": "business_analyst",
        "name": "Business Analyst",
        "domain": "Business Analysis",
        "description": "Expert in requirements gathering, stakeholder management, process analysis, and business documentation. Bridges technical and business domains.",
        "trigger_keywords": ["requirements", "stakeholder", "business", "process", "analysis", "user stories", "acceptance criteria"],
        "skill_files": ["business-analysis"],
        "interaction_modes": ["advisor", "debater"],
        "default_model": "sonnet",
        "icon": "📋",
        "color": "#17a2b8",
    },
    {
        "persona_id": "technical_writer",
        "name": "Technical Writer",
        "domain": "Documentation",
        "description": "Specializes in technical documentation, API documentation, user guides, and knowledge base creation. Ensures clear, accurate, and accessible documentation.",
        "trigger_keywords": ["documentation", "docs", "api docs", "user guide", "knowledge base", "readme", "technical writing"],
        "skill_files": ["technical-writing"],
        "interaction_modes": ["advisor", "co-executor"],
        "default_model": "haiku",
        "icon": "📝",
        "color": "#28a745",
    },
    {
        "persona_id": "devops_engineer",
        "name": "DevOps Engineer",
        "domain": "DevOps",
        "description": "Expert in CI/CD pipelines, infrastructure as code, containerization, orchestration, and deployment automation. Works with Docker, Kubernetes, Jenkins, and GitOps.",
        "trigger_keywords": ["devops", "cicd", "docker", "kubernetes", "jenkins", "pipeline", "deployment", "infrastructure as code", "gitops"],
        "skill_files": ["devops-engineering"],
        "interaction_modes": ["advisor", "co-executor"],
        "default_model": "sonnet",
        "icon": "⚙️",
        "color": "#6c757d",
    },
    {
        "persona_id": "frontend_developer",
        "name": "Frontend Developer",
        "domain": "Frontend Development",
        "description": "Specializes in modern web development, React, component design, state management, and user experience. Expert in TypeScript, CSS, and responsive design.",
        "trigger_keywords": ["frontend", "react", "vue", "angular", "typescript", "css", "component", "ui", "ux"],
        "skill_files": ["frontend-development"],
        "interaction_modes": ["advisor", "co-executor", "debater"],
        "default_model": "sonnet",
        "icon": "🎨",
        "color": "#e83e8c",
    },
]


# =============================================================================
# Helper Functions
# =============================================================================

def get_all_personas() -> List[SMEPersona]:
    """Get all SME personas."""
    return [
        SMEPersona(**data)
        for data in SME_REGISTRY_DATA
    ]


def get_persona(persona_id: str) -> Optional[SMEPersona]:
    """Get a specific persona by ID."""
    for data in SME_REGISTRY_DATA:
        if data["persona_id"] == persona_id:
            return SMEPersona(**data)
    return None


def find_personas_by_keywords(keywords: List[str]) -> List[SMEPersona]:
    """Find personas matching given keywords."""
    keywords_lower = [k.lower() for k in keywords]

    matching = []
    for data in SME_REGISTRY_DATA:
        trigger_keywords = [k.lower() for k in data["trigger_keywords"]]
        if any(kw in trigger_keywords for kw in keywords_lower):
            matching.append(SMEPersona(**data))

    return matching


def find_personas_by_domain(domain: str) -> List[SMEPersona]:
    """Find personas by domain."""
    domain_lower = domain.lower()

    matching = []
    for data in SME_REGISTRY_DATA:
        if domain_lower in data["domain"].lower():
            matching.append(SMEPersona(**data))

    return matching


def get_domains() -> List[str]:
    """Get list of all domains."""
    domains = set()
    for data in SME_REGISTRY_DATA:
        domains.add(data["domain"])
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
        # Get all unique interaction modes
        all_modes = set()
        for data in SME_REGISTRY_DATA:
            all_modes.update(data["interaction_modes"])
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
