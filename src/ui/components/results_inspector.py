"""
Results Inspector Component - Structured Output Viewer

Displays formatted agent outputs with support for multiple content types
including code, documents, markdown, and structured data.
"""

import json
import base64
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

import streamlit as st
import streamlit.components.v1 as components


# =============================================================================
# Result Types
# =============================================================================

class OutputFormat(str, Enum):
    """Output format types."""
    MARKDOWN = "markdown"
    HTML = "html"
    CODE = "code"
    JSON = "json"
    DOCX = "docx"
    PDF = "pdf"
    XLSX = "xlsx"
    PPTX = "pptx"


class ContentType(str, Enum):
    """Content categories."""
    TEXT = "text"
    CODE = "code"
    DOCUMENT = "document"
    DATA = "data"
    DIAGRAM = "diagram"
    MIXED = "mixed"


# =============================================================================
# Result Data Structures
# =============================================================================

@dataclass
class ResultMetadata:
    """Metadata for a result."""
    result_id: str
    timestamp: datetime
    agent_name: str
    phase: str
    tier: int
    format: OutputFormat
    content_type: ContentType
    title: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = None
    file_path: Optional[str] = None
    token_usage: Optional[Dict[str, int]] = None
    duration_seconds: Optional[float] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass
class AgentResult:
    """A complete result from an agent."""
    metadata: ResultMetadata
    content: str
    structured_data: Optional[Dict[str, Any]] = None
    artifacts: List[Dict[str, Any]] = None
    related_results: List[str] = None

    def __post_init__(self):
        if self.artifacts is None:
            self.artifacts = []
        if self.related_results is None:
            self.related_results = []


# =============================================================================
# Session State Management
# =============================================================================

def get_results_store() -> Dict[str, AgentResult]:
    """Get the results store from session state."""
    if "results_store" not in st.session_state:
        st.session_state.results_store = {}
    return st.session_state.results_store


def add_result(result: AgentResult) -> None:
    """Add a result to the store."""
    store = get_results_store()
    store[result.metadata.result_id] = result

    # Update results index
    if "results_index" not in st.session_state:
        st.session_state.results_index = []

    if result.metadata.result_id not in st.session_state.results_index:
        st.session_state.results_index.insert(0, result.metadata.result_id)


def get_result(result_id: str) -> Optional[AgentResult]:
    """Get a result by ID."""
    return get_results_store().get(result_id)


def get_recent_results(limit: int = 20) -> List[AgentResult]:
    """Get recent results."""
    index = st.session_state.get("results_index", [])
    store = get_results_store()

    results = []
    for result_id in index[:limit]:
        if result_id in store:
            results.append(store[result_id])

    return results


def clear_results() -> None:
    """Clear all results."""
    st.session_state.results_store = {}
    st.session_state.results_index = []


# =============================================================================
# Rendering Functions
# =============================================================================

def render_markdown(content: str, title: Optional[str] = None) -> None:
    """Render markdown content."""
    if title:
        st.markdown(f"#### {title}")
    st.markdown(content)


def render_code(code: str, language: str = "python", title: Optional[str] = None) -> None:
    """Render code with syntax highlighting."""
    if title:
        st.markdown(f"#### {title}")

    st.code(code, language=language, line_numbers=True)


def render_json(data: Union[Dict, List, str], title: Optional[str] = None) -> None:
    """Render JSON data."""
    if title:
        st.markdown(f"#### {title}")

    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            st.error("Invalid JSON")
            return

    # Pretty print
    st.json(data)


def render_html(html: str, title: Optional[str] = None, height: int = 400) -> None:
    """Render HTML content."""
    if title:
        st.markdown(f"#### {title}")

    components.html(html, height=height, scrolling=True)


def render_structured_data(
    data: Dict[str, Any],
    schema_name: Optional[str] = None,
) -> None:
    """Render structured/validated data from Pydantic schemas."""
    if schema_name:
        st.caption(f"Schema: `{schema_name}`")

    # Use expander for nested data
    for key, value in data.items():
        if isinstance(value, dict):
            with st.expander(f"📋 {key}"):
                render_structured_data(value)
        elif isinstance(value, list):
            st.markdown(f"**{key}**: {len(value)} items")
            if value and isinstance(value[0], dict):
                for i, item in enumerate(value[:5]):  # Show first 5
                    with st.expander(f"Item {i + 1}"):
                        render_structured_data(item)
                if len(value) > 5:
                    st.caption(f"... and {len(value) - 5} more items")
            else:
                st.markdown(f"- `{value}`")
        else:
            st.markdown(f"**{key}**: `{value}`")


def render_download_button(
    content: str,
    filename: str,
    mime_type: str = "text/plain",
    label: str = "📥 Download",
) -> None:
    """Render a download button."""
    st.download_button(
        label=label,
        data=content,
        file_name=filename,
        mime=mime_type,
        use_container_width=True,
    )


def render_file_preview(file_path: str, title: Optional[str] = None) -> None:
    """Render a preview of a generated file."""
    path = Path(file_path)

    if not path.exists():
        st.warning(f"File not found: {file_path}")
        return

    if title:
        st.markdown(f"#### {title}")

    # File info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Size", f"{path.stat().st_size / 1024:.1f} KB")
    with col2:
        st.metric("Type", path.suffix.upper())
    with col3:
        st.metric("Location", str(path.parent))

    # Read and display based on type
    if path.suffix in [".txt", ".md", ".py", ".js", ".html", ".css", ".json"]:
        with st.expander("📄 View Contents", expanded=True):
            content = path.read_text(encoding="utf-8")
            if path.suffix == ".json":
                render_json(content)
            elif path.suffix in [".py", ".js"]:
                render_code(content, language=path.suffix[1:])
            else:
                st.markdown(content)

    elif path.suffix in [".png", ".jpg", ".jpeg", ".gif", ".svg"]:
        st.image(str(path))

    # Download button
    with open(path, "rb") as f:
        st.download_button(
            label=f"📥 Download {path.name}",
            data=f.read(),
            file_name=path.name,
            mime="application/octet-stream",
            use_container_width=True,
        )


# =============================================================================
# Main Result Inspector
# =============================================================================

def render_result_card(result: AgentResult, compact: bool = False) -> None:
    """Render a single result card."""
    meta = result.metadata

    # Determine agent type and colour
    council_keywords = ["Council", "Arbiter", "Ethics"]
    sme_names = [
        "IAM Architect", "Cloud Architect", "Security Analyst",
        "Data Engineer", "AI/ML Engineer", "Test Engineer",
        "Business Analyst", "Technical Writer", "DevOps Engineer",
        "Frontend Developer",
    ]

    if any(kw in meta.agent_name for kw in council_keywords):
        agent_type_color = "#FFD700"  # Gold for Council
        agent_type_label = "Council"
    elif meta.agent_name in sme_names:
        agent_type_color = "#28a745"  # Green for SME
        agent_type_label = "SME"
    else:
        agent_type_color = "#007bff"  # Blue for Operational
        agent_type_label = "Operational"

    if compact:
        # Compact card for list view
        st.markdown(f"""
        <div style="
            border: 1px solid #e0e0e0;
            border-left: 4px solid {agent_type_color};
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 8px;
            cursor: pointer;
        ">
            <div style="display: flex; justify-content: space-between;">
                <strong>{meta.title or meta.agent_name}</strong>
                <span style="background: {agent_type_color}; color: white; padding: 1px 6px;
                    border-radius: 8px; font-size: 10px;">{agent_type_label}</span>
                <small>{meta.timestamp.strftime("%H:%M")}</small>
            </div>
            <div style="font-size: 12px; color: #666; margin-top: 4px;">
                Tier {meta.tier} • {meta.phase} • {meta.format.value}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Full card
        status_emoji = {
            ContentType.TEXT: "📝",
            ContentType.CODE: "💻",
            ContentType.DOCUMENT: "📄",
            ContentType.DATA: "📊",
            ContentType.DIAGRAM: "🎨",
            ContentType.MIXED: "🔀",
        }.get(meta.content_type, "📦")

        st.markdown(f"""
        <div style="
            border: 1px solid #e0e0e0;
            border-left: 4px solid {agent_type_color};
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 16px;
            background: white;
        ">
            <div style="display: flex; justify-content: space-between; align-items: start;">
                <div>
                    <h3 style="margin: 0;">{status_emoji} {meta.title or meta.agent_name}</h3>
                    <div style="color: #666; margin-top: 4px;">
                        <strong>{meta.agent_name}</strong> • {meta.phase}
                        <span style="background: {agent_type_color}; color: white; padding: 2px 8px;
                            border-radius: 10px; font-size: 11px; margin-left: 8px;">{agent_type_label}</span>
                    </div>
                </div>
                <div style="text-align: right;">
                    <span class="tier-{meta.tier}">Tier {meta.tier}</span>
                    <div style="margin-top: 4px; font-size: 12px; color: #666;">
                        {meta.timestamp.strftime("%Y-%m-%d %H:%M:%S")}
                    </div>
                </div>
            </div>

            {f"<p style='margin-top: 12px; color: #666;'>{meta.description}</p>" if meta.description else ""}
        </div>
        """, unsafe_allow_html=True)

        # Tags
        if meta.tags:
            tags_html = " ".join([
                f"<span style='background: #e3f2fd; color: #1976d2; padding: 2px 8px; "
                f"border-radius: 12px; font-size: 11px; margin-right: 4px;'>#{tag}</span>"
                for tag in meta.tags
            ])
            st.markdown(f"<div style='margin-bottom: 12px;'>{tags_html}</div>", unsafe_allow_html=True)

        # Token usage
        if meta.token_usage:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Input Tokens", meta.token_usage.get("input", 0))
            with col2:
                st.metric("Output Tokens", meta.token_usage.get("output", 0))
            with col3:
                total = meta.token_usage.get("input", 0) + meta.token_usage.get("output", 0)
                st.metric("Total", total)

        # Content tabs
        tab1, tab2, tab3 = st.tabs(["📄 Content", "📊 Structured", "ℹ️ Metadata"])

        with tab1:
            render_content_by_format(result)

        with tab2:
            if result.structured_data:
                render_structured_data(result.structured_data)
            else:
                st.info("No structured data available")

        with tab3:
            st.json({
                "result_id": meta.result_id,
                "timestamp": meta.timestamp.isoformat(),
                "agent": meta.agent_name,
                "phase": meta.phase,
                "tier": meta.tier,
                "format": meta.format.value,
                "content_type": meta.content_type.value,
                "duration_seconds": meta.duration_seconds,
                "file_path": meta.file_path,
            })

        # Artifacts
        if result.artifacts:
            st.markdown("### 📎 Artifacts")
            for artifact in result.artifacts:
                if artifact.get("type") == "file":
                    render_file_preview(artifact["path"], artifact.get("name"))
                elif artifact.get("type") == "code":
                    render_code(artifact["content"], artifact.get("language", "python"), artifact.get("name"))


def render_content_by_format(result: AgentResult) -> None:
    """Render content based on format type."""
    meta = result.metadata
    content = result.content

    if meta.format == OutputFormat.MARKDOWN:
        render_markdown(content, meta.title)

    elif meta.format == OutputFormat.CODE:
        language = "python"
        if meta.title:
            # Try to infer language from title
            title_lower = meta.title.lower()
            if "javascript" in title_lower or "js" in title_lower:
                language = "javascript"
            elif "typescript" in title_lower or "ts" in title_lower:
                language = "typescript"
            elif "sql" in title_lower:
                language = "sql"
            elif "html" in title_lower:
                language = "html"
            elif "css" in title_lower:
                language = "css"

        render_code(content, language, meta.title)

    elif meta.format == OutputFormat.JSON:
        render_json(content, meta.title)

    elif meta.format == OutputFormat.HTML:
        render_html(content, meta.title)

    else:
        st.markdown(content)

    # Highlight flagged/unverified claims
    if result.structured_data:
        flagged = result.structured_data.get("flagged_claims", [])
        unverified = result.structured_data.get("unverified_claims", [])

        if flagged:
            st.markdown("#### Flagged Claims")
            for claim in flagged:
                claim_text = claim if isinstance(claim, str) else claim.get("claim", str(claim))
                st.markdown(
                    f'<div style="background: #ffcccc; border-left: 4px solid #dc3545; '
                    f'padding: 8px 12px; margin-bottom: 6px; border-radius: 4px; color: #721c24;">'
                    f'<strong>Flagged:</strong> {claim_text}</div>',
                    unsafe_allow_html=True,
                )

        if unverified:
            st.markdown("#### Unverified Claims")
            for claim in unverified:
                claim_text = claim if isinstance(claim, str) else claim.get("claim", str(claim))
                st.markdown(
                    f'<div style="background: #ffe0e0; border-left: 4px solid #e74c3c; '
                    f'padding: 8px 12px; margin-bottom: 6px; border-radius: 4px; color: #721c24;">'
                    f'<strong>Unverified:</strong> {claim_text}</div>',
                    unsafe_allow_html=True,
                )

    # Download button
    render_download_button(
        content,
        f"{meta.title or 'output'}.{meta.format.value}",
        label=f"📥 Download {meta.format.value.upper()}",
    )


def render_results_browser() -> None:
    """Render the main results browser interface."""
    st.markdown("### Browse Results")

    # Filters
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        tier_filter = st.multiselect(
            "Filter by Tier",
            [1, 2, 3, 4],
            default=[1, 2, 3, 4],
        )

    with col2:
        format_filter = st.multiselect(
            "Filter by Format",
            [f.value for f in OutputFormat],
            default=[f.value for f in OutputFormat],
        )

    with col3:
        search_query = st.text_input("🔍 Search", placeholder="Search results...")

    with col4:
        limit = st.number_input("Results", min_value=5, max_value=100, value=20)

    st.markdown("---")

    # Get and filter results
    results = get_recent_results(limit=100)

    # Apply filters
    if tier_filter:
        results = [r for r in results if r.metadata.tier in tier_filter]
    if format_filter:
        results = [r for r in results if r.metadata.format.value in format_filter]
    if search_query:
        query_lower = search_query.lower()
        results = [
            r for r in results
            if query_lower in (r.metadata.title or "").lower()
            or query_lower in r.content.lower()
            or any(query_lower in tag.lower() for tag in r.metadata.tags)
        ]

    results = results[:limit]

    # Show count
    st.caption(f"Showing {len(results)} result(s)")

    # Clear button
    if st.button("🗑️ Clear All Results"):
        clear_results()
        st.rerun()

    st.markdown("---")

    # Render results
    if not results:
        render_empty_results()
    else:
        for result in results:
            render_result_card(result)
            # Expander per subagent showing structured output
            with st.expander(f"Structured output: {result.metadata.agent_name}", expanded=False):
                if result.structured_data:
                    st.json(result.structured_data)
                else:
                    st.info("No structured data available for this agent.")
            st.markdown("---")


def render_empty_results() -> None:
    """Render empty state for results."""
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 40px;">
            <div style="font-size: 64px; margin-bottom: 16px;">📭</div>
            <h3>No Results Yet</h3>
            <p style="color: #666;">
                Results from agent interactions will appear here.
            </p>
        </div>
        """, unsafe_allow_html=True)


def render_current_result() -> None:
    """Render the current active result."""
    result_id = st.session_state.get("current_result")

    if not result_id:
        st.info("No result selected")
        return

    result = get_result(result_id)
    if not result:
        st.warning("Result not found")
        return

    render_result_card(result)


# =============================================================================
# Result Creation Helper
# =============================================================================

def create_result(
    agent_name: str,
    phase: str,
    tier: int,
    content: str,
    output_format: OutputFormat = OutputFormat.MARKDOWN,
    title: Optional[str] = None,
    description: Optional[str] = None,
    structured_data: Optional[Dict[str, Any]] = None,
) -> AgentResult:
    """
    Create a new agent result.

    Args:
        agent_name: Name of the agent
        phase: Phase in which result was generated
        tier: Complexity tier
        content: Result content
        output_format: Output format type
        title: Optional title
        description: Optional description
        structured_data: Optional structured/validated data

    Returns:
        AgentResult instance
    """
    result_id = f"res_{int(datetime.now().timestamp() * 1000000)}"

    # Detect content type
    content_type = ContentType.TEXT
    if output_format == OutputFormat.CODE:
        content_type = ContentType.CODE
    elif output_format == OutputFormat.JSON:
        content_type = ContentType.DATA
    elif structured_data and content:
        content_type = ContentType.MIXED

    metadata = ResultMetadata(
        result_id=result_id,
        timestamp=datetime.now(),
        agent_name=agent_name,
        phase=phase,
        tier=tier,
        format=output_format,
        content_type=content_type,
        title=title,
        description=description,
    )

    return AgentResult(
        metadata=metadata,
        content=content,
        structured_data=structured_data,
    )
