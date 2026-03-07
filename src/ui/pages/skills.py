"""
Skills Page - Browse Available Agent Skills

Displays all available agent skills with their descriptions,
categories, and usage information.
"""

from pathlib import Path
from typing import Dict, List, Any
import yaml

import streamlit as st


# =============================================================================
# Skill Discovery
# =============================================================================

def discover_skills() -> List[Dict[str, Any]]:
    """
    Discover all available skills from the .claude/skills directory.

    Returns:
        List of skill metadata dictionaries
    """
    skills_dir = Path.cwd() / ".claude" / "skills"

    if not skills_dir.exists():
        return []

    skills = []

    for skill_path in skills_dir.iterdir():
        if skill_path.is_dir() and (skill_path / "SKILL.md").exists():
            skill_file = skill_path / "SKILL.md"

            # Read and parse frontmatter
            content = skill_file.read_text(encoding="utf-8")

            # Extract YAML frontmatter
            if content.startswith("---"):
                try:
                    # Find end of frontmatter
                    end_marker = content.find("\n---", 3)
                    if end_marker == -1:
                        end_marker = content.find("\n...", 3)

                    if end_marker > 0:
                        frontmatter = content[3:end_marker]
                        metadata = yaml.safe_load(frontmatter)

                        # Extract description from content
                        description = ""
                        if end_marker > 0:
                            body = content[end_marker + 4:].strip()
                            # Get first paragraph
                            paragraphs = body.split("\n\n")
                            if paragraphs:
                                description = paragraphs[0].strip()

                        skills.append({
                            "name": metadata.get("name", skill_path.name),
                            "description": metadata.get("description", description),
                            "version": metadata.get("version", "1.0.0"),
                            "category": metadata.get("category", "general"),
                            "tags": metadata.get("tags", []),
                            "path": str(skill_path),
                        })
                except Exception as e:
                    # Skip invalid skills
                    pass

    return skills


def get_skill_content(skill_name: str) -> str:
    """Get the full content of a skill."""
    skills_dir = Path.cwd() / ".claude" / "skills"
    skill_path = skills_dir / skill_name / "SKILL.md"

    if skill_path.exists():
        return skill_path.read_text(encoding="utf-8")

    return ""


# =============================================================================
# Rendering
# =============================================================================

def render_skills_catalogue() -> None:
    """Render the skills catalogue page."""
    st.markdown("### 🎯 Agent Skills")

    # Get all skills
    all_skills = discover_skills()

    if not all_skills:
        st.info("""
        No skills found. Skills should be located in `.claude/skills/*/SKILL.md`.

        Each skill should have a YAML frontmatter with:
        - name: Skill name
        - description: What the skill does
        - version: Version number
        - category: Skill category
        - tags: Relevant keywords
        """)
        return

    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        categories = sorted(set(s.get("category", "general") for s in all_skills))
        category_filter = st.multiselect(
            "Filter by Category",
            categories,
            default=categories,
        )

    with col2:
        all_tags = sorted(set(tag for s in all_skills for tag in s.get("tags", [])))
        tag_filter = st.multiselect(
            "Filter by Tag",
            all_tags,
        )

    with col3:
        search_query = st.text_input("🔍 Search", placeholder="Search skills...")

    st.markdown("---")

    # Filter skills
    filtered_skills = all_skills.copy()

    if category_filter:
        filtered_skills = [s for s in filtered_skills if s.get("category") in category_filter]

    if tag_filter:
        filtered_skills = [
            s for s in filtered_skills
            if any(tag in s.get("tags", []) for tag in tag_filter)
        ]

    if search_query:
        query_lower = search_query.lower()
        filtered_skills = [
            s for s in filtered_skills
            if query_lower in s.get("name", "").lower()
            or query_lower in s.get("description", "").lower()
        ]

    # Show count
    st.caption(f"Showing {len(filtered_skills)} of {len(all_skills)} skill(s)")

    st.markdown("---")

    # Group by category
    by_category: Dict[str, List[Dict]] = {}
    for skill in filtered_skills:
        category = skill.get("category", "general")
        if category not in by_category:
            by_category[category] = []
        by_category[category].append(skill)

    # Display skills by category
    for category, skills in sorted(by_category.items()):
        # Category header
        category_info = {
            "core": {"icon": "🔧", "color": "#007bff"},
            "development": {"icon": "💻", "color": "#28a745"},
            "testing": {"icon": "🧪", "color": "#fd7e14"},
            "research": {"icon": "🔍", "color": "#6f42c1"},
            "documentation": {"icon": "📚", "color": "#20c997"},
            "requirements": {"icon": "📋", "color": "#17a2b8"},
            "architecture": {"icon": "🏗️", "color": "#dc3545"},
        }.get(category, {"icon": "📦", "color": "#6c757d"})

        st.markdown(f"""
        <div style="
            background: {category_info['color']}10;
            border-left: 4px solid {category_info['color']};
            padding: 12px 16px;
            border-radius: 8px;
            margin-bottom: 16px;
            margin-top: 24px;
        ">
            <h4 style="margin: 0; color: {category_info['color']};">
                {category_info['icon']} {category.title()}
            </h4>
        </div>
        """, unsafe_allow_html=True)

        # Skills in this category
        for skill in skills:
            render_skill_card(skill)


# Mapping of skill names to the agents that use them
SKILL_AGENT_MAP: Dict[str, List[str]] = {
    "code-generation": ["Executor", "Code Reviewer"],
    "document-creation": ["Formatter"],
    "test-case-generation": ["Test Engineer SME"],
    "web-research": ["Researcher"],
    "requirements-engineering": ["Analyst", "Business Analyst SME"],
    "architecture-design": ["Planner", "Cloud Architect SME"],
    "multi-agent-reasoning": ["Orchestrator"],
}

# SME registry for skill_files lookup
SME_SKILL_MAP: Dict[str, List[str]] = {
    "sailpoint-test-engineer": ["IAM Architect"],
    "azure-architect": ["IAM Architect", "Cloud Architect"],
    "aws-solutions-architect": ["Cloud Architect"],
    "security-specialist": ["Security Analyst"],
    "data-engineering": ["Data Engineer"],
    "ml-engineering": ["AI/ML Engineer"],
    "test-automation": ["Test Engineer"],
    "business-analysis": ["Business Analyst"],
    "technical-writing": ["Technical Writer"],
    "devops-engineering": ["DevOps Engineer"],
    "frontend-development": ["Frontend Developer"],
}


def render_skill_card(skill: Dict[str, Any]) -> None:
    """Render a single skill card."""
    name = skill.get("name", "Unknown")
    description = skill.get("description", "")
    version = skill.get("version", "1.0.0")
    tags = skill.get("tags", [])

    st.markdown(f"""
    <div style="
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
        background: white;
        transition: box-shadow 0.2s;
    ">
        <div style="display: flex; justify-content: space-between; align-items: start;">
            <div>
                <h5 style="margin: 0;">{name}</h5>
                <small style="color: #666;">v{version}</small>
            </div>
        </div>
        <p style="margin: 8px 0; color: #333;">{description}</p>
    </div>
    """, unsafe_allow_html=True)

    # Agent assignments
    agents = SKILL_AGENT_MAP.get(name, [])
    if agents:
        agents_html = " ".join([
            f"<span style='background: #007bff20; color: #007bff; padding: 2px 8px; "
            f"border-radius: 12px; font-size: 11px; margin-right: 4px;'>{agent}</span>"
            for agent in agents
        ])
        st.markdown(
            f"<div style='margin-bottom: 6px;'><strong style='font-size: 12px;'>Agents:</strong> {agents_html}</div>",
            unsafe_allow_html=True,
        )

    # SME persona assignments
    sme_users = SME_SKILL_MAP.get(name, [])
    if sme_users:
        sme_html = " ".join([
            f"<span style='background: #28a74520; color: #28a745; padding: 2px 8px; "
            f"border-radius: 12px; font-size: 11px; margin-right: 4px;'>{sme}</span>"
            for sme in sme_users
        ])
        st.markdown(
            f"<div style='margin-bottom: 6px;'><strong style='font-size: 12px;'>SMEs:</strong> {sme_html}</div>",
            unsafe_allow_html=True,
        )

    # Tags
    if tags:
        tags_html = " ".join([
            f"<span style='background: #e9ecef; color: #495057; padding: 2px 8px; "
            f"border-radius: 12px; font-size: 11px; margin-right: 4px;'>#{tag}</span>"
            for tag in tags[:5]
        ])
        st.markdown(f"<div style='margin-bottom: 8px;'>{tags_html}</div>", unsafe_allow_html=True)

    # SKILL.md content preview
    skill_path = skill.get("path", "")
    if skill_path:
        skill_content = get_skill_content(Path(skill_path).name)
        if skill_content:
            # Strip frontmatter for preview
            body = skill_content
            if body.startswith("---"):
                end_marker = body.find("\n---", 3)
                if end_marker > 0:
                    body = body[end_marker + 4:].strip()
            preview = body[:500]
            if len(body) > 500:
                preview += "..."
            with st.expander("SKILL.md Preview"):
                st.markdown(preview)

    # View details button
    if st.button(f"View Details", key=f"view_{skill.get('path', '')}"):
        render_skill_detail(skill)


def render_skill_detail(skill: Dict[str, Any]) -> None:
    """Render detailed view of a skill."""
    name = skill.get("name", "Unknown")
    skill_path = skill.get("path", "")

    content = get_skill_content(Path(skill_path).name)

    st.markdown(f"## {name}")

    if content:
        st.markdown(content)
    else:
        st.warning("Skill content not found")

    if st.button("← Back to Catalogue"):
        st.rerun()
