"""
Knowledge Base Page - Organizational Memory

Browse and search the knowledge base where the Memory Curator agent
stores extracted insights, patterns, and decisions.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import re

import streamlit as st


# =============================================================================
# Knowledge Entry Types
# =============================================================================

class KnowledgeCategory(str, Enum):
    """Knowledge entry categories."""
    ARCHITECTURAL_DECISION = "architectural_decision"
    CODE_PATTERN = "code_pattern"
    DOMAIN_INSIGHT = "domain_insight"
    BEST_PRACTICE = "best_practice"
    LESSON_LEARNED = "lesson_learned"
    REQUIREMENT = "requirement"
    SOLUTION = "solution"
    TERMINOLOGY = "terminology"


@dataclass
class KnowledgeEntry:
    """A knowledge base entry."""
    id: str
    title: str
    category: KnowledgeCategory
    content: str
    tags: List[str]
    created_at: datetime
    source: Optional[str] = None
    related_entries: List[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.related_entries is None:
            self.related_entries = []
        if self.metadata is None:
            self.metadata = {}


# =============================================================================
# Knowledge Base Access
# =============================================================================

KNOWLEDGE_DIR = Path("docs/knowledge")


def get_knowledge_files() -> List[Path]:
    """Get all knowledge markdown files."""
    if not KNOWLEDGE_DIR.exists():
        return []

    return list(KNOWLEDGE_DIR.glob("*.md"))


def parse_knowledge_file(file_path: Path) -> Optional[KnowledgeEntry]:
    """Parse a knowledge file into an entry."""
    try:
        content = file_path.read_text(encoding="utf-8")

        # Extract YAML frontmatter
        frontmatter = {}
        body = content

        if content.startswith("---"):
            end_marker = content.find("\n---", 3)
            if end_marker == -1:
                end_marker = content.find("\n...", 3)

            if end_marker > 0:
                import yaml
                frontmatter_text = content[3:end_marker]
                try:
                    frontmatter = yaml.safe_load(frontmatter_text) or {}
                except:
                    pass
                body = content[end_marker + 4:].strip()

        # Create entry
        return KnowledgeEntry(
            id=file_path.stem,
            title=frontmatter.get("title", file_path.stem),
            category=frontmatter.get("category", "domain_insight"),
            content=body,
            tags=frontmatter.get("tags", []),
            created_at=datetime.fromisoformat(frontmatter.get("created_at", datetime.now().isoformat())),
            source=frontmatter.get("source"),
            related_entries=frontmatter.get("related_entries", []),
            metadata=frontmatter,
        )
    except Exception as e:
        return None


def get_all_knowledge() -> List[KnowledgeEntry]:
    """Get all knowledge entries."""
    entries = []

    for file_path in get_knowledge_files():
        entry = parse_knowledge_file(file_path)
        if entry:
            entries.append(entry)

    # Sort by created date, newest first
    entries.sort(key=lambda e: e.created_at, reverse=True)
    return entries


def search_knowledge(query: str) -> List[KnowledgeEntry]:
    """Search knowledge entries by query."""
    all_entries = get_all_knowledge()

    if not query:
        return all_entries

    query_lower = query.lower()
    results = []

    for entry in all_entries:
        # Search in title, content, tags
        if (
            query_lower in entry.title.lower()
            or query_lower in entry.content.lower()
            or any(query_lower in tag.lower() for tag in entry.tags)
        ):
            results.append(entry)

    return results


def get_entries_by_category(category: str) -> List[KnowledgeEntry]:
    """Get entries filtered by category."""
    all_entries = get_all_knowledge()
    return [e for e in all_entries if e.category == category]


# =============================================================================
# Rendering
# =============================================================================

def render_knowledge_base() -> None:
    """Render the knowledge base page."""
    st.markdown("### 📚 Knowledge Base")
    st.caption("Organizational memory and extracted insights")

    st.markdown("---")

    # Check if knowledge directory exists
    if not KNOWLEDGE_DIR.exists():
        st.warning(f"""
        Knowledge directory not found at `{KNOWLEDGE_DIR}`.

        The Memory Curator agent will store extracted knowledge here.
        """)
        return

    # Get stats
    all_entries = get_all_knowledge()

    # Stats cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Entries", len(all_entries))

    with col2:
        categories = set(e.category for e in all_entries)
        st.metric("Categories", len(categories))

    with col3:
        total_tags = sum(len(e.tags) for e in all_entries)
        st.metric("Total Tags", total_tags)

    with col4:
        # Count entries from last 7 days
        recent = sum(
            1 for e in all_entries
            if (datetime.now() - e.created_at).days <= 7
        )
        st.metric("Last 7 Days", recent)

    st.markdown("---")

    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        category_filter = st.selectbox(
            "Filter by Category",
            options=["All"] + [c.value for c in KnowledgeCategory],
            format_func=lambda x: x.replace("_", " ").title() if x != "All" else "All Categories",
        )

    with col2:
        if all_entries:
            all_tags = sorted(set(tag for e in all_entries for tag in e.tags))
            tag_filter = st.multiselect(
                "Filter by Tag",
                all_tags,
            )
        else:
            tag_filter = []

    with col3:
        search_query = st.text_input("🔍 Search", placeholder="Search knowledge...")

    st.markdown("---")

    # Filter entries
    filtered_entries = all_entries.copy()

    if category_filter != "All":
        filtered_entries = [e for e in filtered_entries if e.category == category_filter]

    if tag_filter:
        filtered_entries = [
            e for e in filtered_entries
            if any(tag in e.tags for tag in tag_filter)
        ]

    if search_query:
        filtered_entries = search_knowledge(search_query)
        # Re-apply other filters
        if category_filter != "All":
            filtered_entries = [e for e in filtered_entries if e.category == category_filter]
        if tag_filter:
            filtered_entries = [
                e for e in filtered_entries
                if any(tag in e.tags for tag in tag_filter)
            ]

    # Show count
    st.caption(f"Showing {len(filtered_entries)} of {len(all_entries)} entry/entries")

    st.markdown("---")

    # Render entries
    if not filtered_entries:
        render_empty_knowledge()
    else:
        # View toggle
        view_mode = st.radio(
            "View Mode",
            options=["Cards", "List"],
            horizontal=True,
            label_visibility="collapsed",
        )

        if view_mode == "Cards":
            render_knowledge_cards(filtered_entries)
        else:
            render_knowledge_list(filtered_entries)


def render_knowledge_cards(entries: List[KnowledgeEntry]) -> None:
    """Render knowledge entries as cards."""
    cols = st.columns(2)

    for i, entry in enumerate(entries):
        with cols[i % 2]:
            render_knowledge_card(entry)


def render_knowledge_card(entry: KnowledgeEntry) -> None:
    """Render a single knowledge card."""
    # Category colors
    category_colors = {
        KnowledgeCategory.ARCHITECTURAL_DECISION: "#7950f2",
        KnowledgeCategory.CODE_PATTERN: "#28a745",
        KnowledgeCategory.DOMAIN_INSIGHT: "#17a2b8",
        KnowledgeCategory.BEST_PRACTICE: "#007bff",
        KnowledgeCategory.LESSON_LEARNED: "#fd7e14",
        KnowledgeCategory.REQUIREMENT: "#dc3545",
        KnowledgeCategory.SOLUTION: "#20c997",
        KnowledgeCategory.TERMINOLOGY: "#6c757d",
    }

    color = category_colors.get(entry.category, "#007bff")

    st.markdown(f"""
    <div style="
        border: 1px solid #e0e0e0;
        border-left: 4px solid {color};
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 16px;
        background: white;
    ">
        <h5 style="margin: 0;">{entry.title}</h5>
        <small style="color: {color};">{entry.category.replace('_', ' ').title()}</small>
        <div style="font-size: 12px; color: #666; margin-top: 4px;">
            📅 {entry.created_at.strftime('%Y-%m-%d')}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Preview
    preview = entry.content[:200] + "..." if len(entry.content) > 200 else entry.content
    st.caption(preview)

    # Tags
    if entry.tags:
        tags_html = " ".join([
            f"<span style='background: #e9ecef; color: #495057; padding: 2px 8px; "
            f"border-radius: 12px; font-size: 11px; margin-right: 4px;'>#{tag}</span>"
            for tag in entry.tags[:3]
        ])
        st.markdown(f"<div style='margin-bottom: 8px;'>{tags_html}</div>", unsafe_allow_html=True)

    # View details button
    if st.button(f"📖 View Details", key=f"view_{entry.id}"):
        st.session_state.selected_entry = entry.id
        st.rerun()


def render_knowledge_list(entries: List[KnowledgeEntry]) -> None:
    """Render knowledge entries as a list."""
    for entry in entries:
        with st.expander(f"📄 {entry.title}"):
            st.markdown(f"**Category:** {entry.category.replace('_', ' ').title()}")
            st.markdown(f"**Created:** {entry.created_at.strftime('%Y-%m-%d %H:%M')}")
            if entry.source:
                st.markdown(f"**Source:** {entry.source}")

            st.markdown("---")
            st.markdown(entry.content)

            if entry.tags:
                st.markdown("**Tags:** " + ", ".join(f"`#{tag}`" for tag in entry.tags))


def render_empty_knowledge() -> None:
    """Render empty state for knowledge base."""
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 40px;">
            <div style="font-size: 64px; margin-bottom: 16px;">📚</div>
            <h3>Knowledge Base Empty</h3>
            <p style="color: #666;">
                The Memory Curator agent will automatically populate this
                knowledge base with extracted insights from your interactions.
            </p>
        </div>
        """, unsafe_allow_html=True)


def render_entry_detail(entry_id: str) -> None:
    """Render detailed view of a knowledge entry."""
    all_entries = get_all_knowledge()

    entry = next((e for e in all_entries if e.id == entry_id), None)

    if not entry:
        st.error(f"Entry '{entry_id}' not found")
        return

    # Header
    st.markdown(f"## 📄 {entry.title}")

    # Metadata
    col1, col2, col3 = st.columns(3)

    with col1:
        st.caption(f"**Category:** {entry.category.replace('_', ' ').title()}")

    with col2:
        st.caption(f"**Created:** {entry.created_at.strftime('%Y-%m-%d %H:%M')}")

    with col3:
        if entry.source:
            st.caption(f"**Source:** {entry.source}")

    st.markdown("---")

    # Tags
    if entry.tags:
        st.markdown("**Tags:** " + ", ".join(f"`#{tag}`" for tag in entry.tags))

    st.markdown("---")

    # Content
    st.markdown(entry.content)

    # Related entries
    if entry.related_entries:
        st.markdown("---")
        st.markdown("### 🔗 Related Entries")

        for related_id in entry.related_entries:
            related = next((e for e in get_all_knowledge() if e.id == related_id), None)
            if related:
                if st.button(f"📄 {related.title}", key=f"related_{related.id}"):
                    st.session_state.selected_entry = related.id
                    st.rerun()

    # Metadata
    if entry.metadata:
        with st.expander("🔧 Metadata"):
            # Filter out internal fields
            display_metadata = {k: v for k, v in entry.metadata.items()
                             if k not in ["title", "category", "tags", "created_at", "source", "related_entries"]}
            if display_metadata:
                st.json(display_metadata)

    # Back button
    if st.button("← Back to Knowledge Base"):
        st.session_state.selected_entry = None
        st.rerun()
