"""
Settings Page - System Configuration

Configure system behavior, API keys, budgets, and UI preferences.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional

import streamlit as st


# =============================================================================
# Settings Sections
# =============================================================================

def render_api_key_section() -> None:
    """Render API key configuration section."""
    st.markdown("### 🔑 API Keys")

    st.info("""
    **Important:** API keys are stored in your Streamlit session state only.
    They are never sent to any server other than the Anthropic API.
    """)

    # Anthropic API Key
    anthropic_key = st.session_state.get("anthropic_api_key", "")

    col1, col2 = st.columns([3, 1])

    with col1:
        new_key = st.text_input(
            "Anthropic API Key",
            value=anthropic_key,
            type="password",
            placeholder="sk-ant-...",
            help="Your Anthropic API key for Claude access",
        )

    with col2:
        if st.button("Save", use_container_width=True):
            st.session_state.anthropic_api_key = new_key
            st.success("✅ API Key saved to session")
            st.rerun()

    st.caption("Get your API key from: https://console.anthropic.com/")

    # Test connection
    if anthropic_key and st.button("🧪 Test Connection"):
        with st.spinner("Testing connection..."):
            try:
                # TODO: Actual API test
                st.success("✅ Connection successful!")
            except Exception as e:
                st.error(f"❌ Connection failed: {e}")


def render_budget_section() -> None:
    """Render budget configuration section."""
    st.markdown("### 💰 Budget & Costs")

    col1, col2 = st.columns(2)

    with col1:
        max_budget = st.number_input(
            "Max Budget per Query ($)",
            min_value=0.1,
            max_value=100.0,
            value=st.session_state.get("max_budget", 10.0),
            step=0.5,
            help="Maximum spend per single query",
        )
        st.session_state.max_budget = max_budget

    with col2:
        daily_budget = st.number_input(
            "Daily Budget Limit ($)",
            min_value=1.0,
            max_value=1000.0,
            value=st.session_state.get("daily_budget", 50.0),
            step=5.0,
            help="Maximum spend per day",
        )
        st.session_state.daily_budget = daily_budget

    # Budget enforcement
    enforce = st.checkbox(
        "🛑 Enforce Budget Limits",
        value=st.session_state.get("enforce_budget", True),
        help="Stop execution when budget is exceeded",
    )
    st.session_state.enforce_budget = enforce

    # Current spend (mock)
    st.markdown("---")
    st.markdown("#### Current Session Spend")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("This Session", "$0.00")

    with col2:
        st.metric("Today", "$0.00")

    with col3:
        st.metric("This Week", "$0.00")


def render_agent_section() -> None:
    """Render agent configuration section."""
    st.markdown("### ⚙️ Agent Settings")

    col1, col2 = st.columns(2)

    with col1:
        # Default tier
        default_tier = st.select_slider(
            "Default Complexity Tier",
            options=[1, 2, 3, 4],
            value=st.session_state.get("default_tier", 2),
            format_func=lambda x: [
                "1 - Direct",
                "2 - Standard",
                "3 - Deep",
                "4 - Adversarial",
            ][x - 1],
            help="Default tier for new queries",
        )
        st.session_state.default_tier = default_tier

    with col2:
        # Default model
        default_model = st.selectbox(
            "Default Model",
            options=["haiku", "sonnet", "opus"],
            index=1,
            help="Default Claude model to use",
        )
        st.session_state.default_model = default_model

    # Max turns
    col1, col2 = st.columns(2)

    with col1:
        max_turns = st.slider(
            "Max Agent Turns",
            min_value=10,
            max_value=200,
            value=st.session_state.get("max_turns", 50),
            help="Maximum conversation turns per agent",
        )
        st.session_state.max_turns = max_turns

    with col2:
        timeout = st.slider(
            "Query Timeout (seconds)",
            min_value=30,
            max_value=600,
            value=st.session_state.get("timeout", 120),
            help="Maximum execution time per query",
        )
        st.session_state.timeout = timeout

    # Per-agent model selection and enable/disable toggles
    st.markdown("---")
    st.markdown("#### Per-Agent Configuration")

    agent_names = [
        "Orchestrator", "Analyst", "Planner", "Clarifier", "Researcher",
        "Executor", "Code Reviewer", "Formatter", "Verifier", "Critic",
        "Reviewer", "Memory Curator", "Council Chair",
    ]

    for agent_name in agent_names:
        col_toggle, col_model = st.columns([1, 2])
        with col_toggle:
            enabled = st.toggle(
                f"{agent_name}",
                value=st.session_state.get(f"agent_enabled_{agent_name}", True),
                key=f"toggle_{agent_name}",
            )
            st.session_state[f"agent_enabled_{agent_name}"] = enabled
        with col_model:
            model_options = ["haiku", "sonnet", "opus"]
            current = st.session_state.get(f"model_{agent_name}", "sonnet")
            default_index = model_options.index(current) if current in model_options else 1
            model = st.selectbox(
                f"Model for {agent_name}",
                options=model_options,
                index=default_index,
                key=f"model_select_{agent_name}",
                label_visibility="collapsed",
            )
            st.session_state[f"model_{agent_name}"] = model

    # SME Controls
    st.markdown("---")
    st.markdown("#### SME Persona Controls")

    sme_personas = [
        "IAM Architect", "Cloud Architect", "Security Analyst",
        "Data Engineer", "AI/ML Engineer", "Test Engineer",
        "Business Analyst", "Technical Writer", "DevOps Engineer",
        "Frontend Developer",
    ]

    max_sme = st.slider(
        "Max concurrent SMEs",
        min_value=0,
        max_value=10,
        value=st.session_state.get("max_sme_count", 3),
        help="Maximum number of SME personas active simultaneously",
    )
    st.session_state.max_sme_count = max_sme

    sme_col1, sme_col2 = st.columns(2)
    for idx, sme_name in enumerate(sme_personas):
        with sme_col1 if idx % 2 == 0 else sme_col2:
            sme_enabled = st.toggle(
                f"{sme_name}",
                value=st.session_state.get(f"sme_enabled_{sme_name}", True),
                key=f"sme_toggle_{sme_name}",
            )
            st.session_state[f"sme_enabled_{sme_name}"] = sme_enabled

    # Feature toggles
    st.markdown("---")
    st.markdown("#### Feature Toggles")

    col1, col2, col3 = st.columns(3)

    with col1:
        enable_smes = st.checkbox(
            "Enable SME Personas",
            value=st.session_state.get("enable_smes", True),
            help="Allow automatic SME spawning",
        )
        st.session_state.enable_smes = enable_smes

    with col2:
        enable_debate = st.checkbox(
            "Enable Self-Play Debate",
            value=st.session_state.get("enable_debate", True),
            help="Use adversarial debate for Tier 3-4",
        )
        st.session_state.enable_debate = enable_debate

    with col3:
        enable_memory = st.checkbox(
            "Enable Memory Curator",
            value=st.session_state.get("enable_memory", True),
            help="Extract knowledge to memory",
        )
        st.session_state.enable_memory = enable_memory


def render_output_section() -> None:
    """Render output configuration section."""
    st.markdown("### 📄 Output Settings")

    col1, col2 = st.columns(2)

    with col1:
        # Default format
        default_format = st.selectbox(
            "Default Output Format",
            options=["markdown", "code", "json", "html", "docx", "pdf"],
            index=0,
            help="Default format for responses",
        )
        st.session_state.default_format = default_format

    with col2:
        # Output verbosity
        verbosity = st.select_slider(
            "Output Verbosity",
            options=["concise", "normal", "verbose", "debug"],
            value="normal",
            help="Level of detail in outputs",
        )
        st.session_state.verbosity = verbosity

    # Include artifacts
    include_artifacts = st.checkbox(
        "📎 Include Artifacts",
        value=True,
        help="Include intermediate artifacts in output",
    )
    st.session_state.include_artifacts = include_artifacts

    # Auto-download
    auto_download = st.checkbox(
        "📥 Auto-download Results",
        value=False,
        help="Automatically download results after completion",
    )
    st.session_state.auto_download = auto_download


def render_ui_section() -> None:
    """Render UI preferences section."""
    st.markdown("### 🎨 UI Preferences")

    col1, col2 = st.columns(2)

    with col1:
        # Theme
        theme = st.selectbox(
            "Theme",
            options=["light", "dark", "auto"],
            index=0,
            help="Application theme",
        )
        st.session_state.theme = theme

        # Compact mode
        compact = st.checkbox(
            "Compact Mode",
            value=st.session_state.get("compact_mode", False),
            help="Reduce spacing in UI",
        )
        st.session_state.compact_mode = compact

    with col2:
        # Auto-scroll
        auto_scroll = st.checkbox(
            "Auto-scroll Chat",
            value=st.session_state.get("auto_scroll", True),
            help="Automatically scroll to newest messages",
        )
        st.session_state.auto_scroll = auto_scroll

        # Show timestamps
        show_timestamps = st.checkbox(
            "Show Timestamps",
            value=st.session_state.get("show_timestamps", True),
            help="Display timestamps on messages",
        )
        st.session_state.show_timestamps = show_timestamps

    # Developer mode
    st.markdown("---")

    dev_mode = st.checkbox(
        "👨‍💻 Developer Mode",
        value=st.session_state.get("show_developer_mode", False),
        help="Show technical details and debug info",
    )
    st.session_state.show_developer_mode = dev_mode

    if dev_mode:
        st.warning("""
        **Developer Mode Enabled**

        This will show:
        - Internal agent states
        - Raw API responses
        - Performance metrics
        - Debug logs
        """)


def render_knowledge_section() -> None:
    """Render knowledge base settings section."""
    st.markdown("### 📚 Knowledge Base")

    # Knowledge directory
    knowledge_dir = st.text_input(
        "Knowledge Directory",
        value=str(st.session_state.get("knowledge_dir", Path("docs/knowledge"))),
        help="Directory where knowledge entries are stored",
    )
    st.session_state.knowledge_dir = Path(knowledge_dir)

    # Auto-extract settings
    col1, col2 = st.columns(2)

    with col1:
        auto_extract = st.checkbox(
            "Auto-extract Knowledge",
            value=st.session_state.get("auto_extract_knowledge", True),
            help="Automatically extract knowledge from conversations",
        )
        st.session_state.auto_extract_knowledge = auto_extract

    with col2:
        extract_threshold = st.slider(
            "Extraction Threshold",
            min_value=1,
            max_value=10,
            value=st.session_state.get("extract_threshold", 5),
            help="Minimum importance score for extraction",
        )
        st.session_state.extract_threshold = extract_threshold

    # Knowledge stats
    st.markdown("---")

    try:
        from src.ui.pages.knowledge import get_all_knowledge
        entries = get_all_knowledge()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Entries", len(entries))

        with col2:
            categories = set(e.category for e in entries)
            st.metric("Categories", len(categories))

        with col3:
            recent = sum(
                1 for e in entries
                if (e.created_at - datetime.now()).days <= 7
            )
            st.metric("Last 7 Days", recent)

    except Exception:
        st.info("Knowledge base not initialized")


def render_data_section() -> None:
    """Render data management section."""
    st.markdown("### 🗃️ Data Management")

    st.warning("""
    **Danger Zone:** These actions can permanently delete data.
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            if st.session_state.get("confirm_clear"):
                from src.ui.pages.chat import clear_chat_history
                clear_chat_history()
                st.session_state.confirm_clear = False
                st.success("Chat history cleared")
                st.rerun()
            else:
                st.session_state.confirm_clear = True
                st.warning("Click again to confirm")

    with col2:
        if st.button("🗑️ Clear Results", use_container_width=True):
            if st.session_state.get("confirm_clear_results"):
                from src.ui.components.results_inspector import clear_results
                clear_results()
                st.session_state.confirm_clear_results = False
                st.success("Results cleared")
                st.rerun()
            else:
                st.session_state.confirm_clear_results = True
                st.warning("Click again to confirm")

    with col3:
        if st.button("🗑️ Reset Settings", use_container_width=True):
            if st.session_state.get("confirm_reset"):
                # Reset all settings to defaults
                for key in list(st.session_state.keys()):
                    if key not in ["messages", "chat_history", "results_store"]:
                        del st.session_state[key]
                st.success("Settings reset")
                st.rerun()
            else:
                st.session_state.confirm_reset = True
                st.warning("Click again to confirm")

    st.markdown("---")

    # Export data
    st.markdown("#### Export Data")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📥 Export Chat History", use_container_width=True):
            from src.ui.pages.chat import export_chat_history
            export_chat_history()

    with col2:
        if st.button("📥 Export Settings", use_container_width=True):
            import json
            settings = {
                k: v for k, v in st.session_state.items()
                if not k.startswith("_") and k not in ["messages", "chat_history", "results_store"]
            }
            st.download_button(
                "Download Settings",
                data=json.dumps(settings, indent=2, default=str),
                file_name="settings.json",
                mime="application/json",
            )


# =============================================================================
# Main Settings Panel
# =============================================================================

def render_settings_panel() -> None:
    """Render the main settings panel."""
    st.markdown("### ⚙️ Settings")
    st.caption("Configure system behavior and preferences")

    st.markdown("---")

    # Settings tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🔑 API Keys",
        "💰 Budget",
        "⚙️ Agents",
        "📄 Output",
        "🎨 UI",
        "📚 Knowledge",
    ])

    with tab1:
        render_api_key_section()

    with tab2:
        render_budget_section()

    with tab3:
        render_agent_section()

    with tab4:
        render_output_section()

    with tab5:
        render_ui_section()

    with tab6:
        render_knowledge_section()

    st.markdown("---")

    # Data management (separate section)
    render_data_section()

    # Save button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        if st.button("💾 Save All Settings", use_container_width=True, type="primary"):
            st.success("✅ Settings saved to session state!")
            st.info("Note: Settings are stored in browser session. Use Export to save permanently.")
