"""
Enhanced Filters Component - Advanced filtering UI

Provides advanced filtering capabilities for search and filtering
across the UI components.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass

import streamlit as st


@dataclass
class FilterConfig:
    """Configuration for a filter."""
    name: str
    filter_type: str  # "select", "multiselect", "text", "date_range", "number_range"
    options: Optional[List[Any]] = None
    default: Any = None
    help_text: Optional[str] = None


class AdvancedFilters:
    """Advanced filtering UI component."""

    def __init__(self, filter_configs: List[FilterConfig]):
        """
        Initialize advanced filters.

        Args:
            filter_configs: List of filter configurations
        """
        self.filter_configs = {fc.name: fc for fc in filter_configs}
        self.filter_values = {}

    def render(self) -> Dict[str, Any]:
        """
        Render the filter UI and return current filter values.

        Returns:
            Dictionary of filter name to value
        """
        st.markdown("### 🔍 Advanced Filters")

        cols = st.columns(2)

        for i, (name, config) in enumerate(self.filter_configs.items()):
            with cols[i % 2]:
                value = self._render_filter(config)
                self.filter_values[name] = value

        return self.filter_values

    def _render_filter(self, config: FilterConfig) -> Any:
        """Render a single filter."""
        label = config.name.replace("_", " ").title()

        if config.filter_type == "select":
            return st.selectbox(
                label,
                options=config.options or [],
                index=0 if config.default is None else (
                    config.options.index(config.default) if config.default in config.options else 0
                ),
                help=config.help_text,
                key=f"filter_{config.name}",
            )

        elif config.filter_type == "multiselect":
            default = config.default if config.default is not None else config.options
            return st.multiselect(
                label,
                options=config.options or [],
                default=default,
                help=config.help_text,
                key=f"filter_{config.name}",
            )

        elif config.filter_type == "text":
            return st.text_input(
                label,
                value=config.default or "",
                help=config.help_text,
                key=f"filter_{config.name}",
            )

        elif config.filter_type == "number_range":
            col1, col2 = st.columns(2)
            with col1:
                min_val = st.number_input(
                    f"{label} Min",
                    value=config.default.get("min", 0) if config.default else 0,
                    key=f"filter_{config.name}_min",
                )
            with col2:
                max_val = st.number_input(
                    f"{label} Max",
                    value=config.default.get("max", 100) if config.default else 100,
                    key=f"filter_{config.name}_max",
                )
            return {"min": min_val, "max": max_val}

        elif config.filter_type == "date_range":
            col1, col2 = st.columns(2)
            with col1:
                min_date = st.date_input(
                    f"{label} Start",
                    value=config.default.get("start", datetime.now() - timedelta(days=30)) if config.default else datetime.now() - timedelta(days=30),
                    key=f"filter_{config.name}_start",
                )
            with col2:
                max_date = st.date_input(
                    f"{label} End",
                    value=config.default.get("end", datetime.now()) if config.default else datetime.now(),
                    key=f"filter_{config.name}_end",
                )
            return {"start": min_date, "end": max_date}

        return None

    def get_active_filters(self) -> Dict[str, Any]:
        """Get only non-default filter values."""
        active = {}
        for name, value in self.filter_values.items():
            if value and value != self.filter_configs[name].default:
                active[name] = value
        return active

    def clear_all(self) -> None:
        """Clear all filters to defaults."""
        for name, config in self.filter_configs.items():
            self.filter_values[name] = config.default


def render_export_buttons(
    data: Any,
    formats: List[str] = None,
    filename_prefix: str = "export",
) -> None:
    """
    Render export buttons for data.

    Args:
        data: Data to export (list, dict, or string)
        formats: List of formats to support
        filename_prefix: Prefix for export filename
    """
    if formats is None:
        formats = ["json", "csv", "markdown"]

    st.markdown("### 📥 Export")

    cols = st.columns(len(formats))

    for i, fmt in enumerate(formats):
        with cols[i]:
            if st.button(fmt.upper(), use_container_width=True, key=f"export_{fmt}"):
                exported_data = _convert_to_format(data, fmt)
                filename = f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{fmt}"

                st.download_button(
                    label=f"Download {fmt.upper()}",
                    data=exported_data,
                    file_name=filename,
                    mime=f"application/{fmt}" if fmt != "csv" else "text/csv",
                    use_container_width=True,
                )


def _convert_to_format(data: Any, format_type: str) -> str:
    """Convert data to specified format."""
    import json

    if format_type == "json":
        return json.dumps(data, indent=2, default=str)

    elif format_type == "csv":
        import csv
        from io import StringIO

        if isinstance(data, list) and data and isinstance(data[0], dict):
            output = StringIO()
            writer = csv.DictWriter(output, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
            return output.getvalue()
        return str(data)

    elif format_type == "markdown":
        if isinstance(data, list):
            lines = []
            for item in data:
                if isinstance(item, dict):
                    for k, v in item.items():
                        lines.append(f"**{k}**: {v}")
                    lines.append("---")
                else:
                    lines.append(f"- {item}")
            return "\n".join(lines)
        return str(data)

    return str(data)


def render_sort_options(
    sort_field: str,
    sort_order: str = "desc",
    available_fields: List[str] = None,
) -> tuple:
    """
    Render sort options UI.

    Args:
        sort_field: Current sort field
        sort_order: Current sort order ("asc" or "desc")
        available_fields: List of available fields to sort by

    Returns:
        Tuple of (field, order)
    """
    col1, col2 = st.columns([3, 1])

    with col1:
        field = st.selectbox(
            "Sort by",
            options=available_fields or ["date", "name", "tier", "status"],
            index=0 if available_fields is None else (
                available_fields.index(sort_field) if sort_field in available_fields else 0
            ),
    )

    with col2:
        order = st.selectbox(
            "Order",
            options=["desc", "asc"],
            index=0 if sort_order == "desc" else 1,
            format_func=lambda x: "↓ Desc" if x == "desc" else "↑ Asc",
        )

    return field, order


def render_pagination(
    total_items: int,
    items_per_page: int = 20,
    current_page: int = 1,
) -> tuple:
    """
    Render pagination controls.

    Args:
        total_items: Total number of items
        items_per_page: Items to show per page
        current_page: Current page number (1-indexed)

    Returns:
        Tuple of (page, start_idx, end_idx)
    """
    total_pages = max(1, (total_items + items_per_page - 1) // items_per_page)

    col1, col2, col3 = st.columns([1, 3, 1])

    with col1:
        if st.button("← Previous", disabled=(current_page <= 1)):
            current_page = max(1, current_page - 1)
            st.rerun()

    with col2:
        page = st.select_slider(
            "Page",
            options=list(range(1, total_pages + 1)),
            value=min(current_page, total_pages),
            format_func=lambda x: f"Page {x} of {total_pages}",
        )

    with col3:
        if st.button("Next →", disabled=(current_page >= total_pages)):
            current_page = min(total_pages, current_page + 1)
            st.rerun()

    start_idx = (page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, total_items)

    return page, start_idx, end_idx
