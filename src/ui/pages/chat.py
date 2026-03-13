"""
Chat Page - Interactive Multi-Agent Chat Interface

Provides the main chat interface for interacting with the multi-agent
reasoning system, including message history, streaming output, and
agent activity visualization.
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import streamlit as st


# =============================================================================
# Message Types
# =============================================================================

class MessageRole(str, Enum):
    """Role of a message sender."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    AGENT = "agent"


class MessageStatus(str, Enum):
    """Status of a message."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# =============================================================================
# Message Data Structures
# =============================================================================

@dataclass
class ChatMessage:
    """A chat message."""
    message_id: str
    role: MessageRole
    content: str
    timestamp: datetime
    status: MessageStatus = MessageStatus.COMPLETED
    agent_name: Optional[str] = None
    tier: Optional[int] = None
    metadata: Dict[str, Any] = None
    artifacts: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.artifacts is None:
            self.artifacts = []


# =============================================================================
# Session State Management
# =============================================================================

def get_chat_history() -> List[ChatMessage]:
    """Get chat history from session state."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    return st.session_state.chat_history


def add_message(message: ChatMessage) -> None:
    """Add a message to chat history."""
    history = get_chat_history()
    history.append(message)

    # Also sync with messages list for compatibility
    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.session_state.messages.append({
        "role": message.role.value,
        "content": message.content,
        "timestamp": message.timestamp.isoformat(),
    })


def get_last_message(role: Optional[MessageRole] = None) -> Optional[ChatMessage]:
    """Get the last message, optionally filtered by role."""
    history = get_chat_history()

    if role:
        for msg in reversed(history):
            if msg.role == role:
                return msg
    else:
        return history[-1] if history else None

    return None


def clear_chat_history() -> None:
    """Clear chat history."""
    st.session_state.chat_history = []
    st.session_state.messages = []


# =============================================================================
# UI Components - Chat Input
# =============================================================================

def render_chat_input() -> Optional[str]:
    """
    Render the chat input area with options.

    Returns:
        The submitted prompt, or None if no input
    """
    st.markdown("""
    <style>
        .chat-input-container {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background: white;
            padding: 16px;
            border-top: 1px solid #e0e0e0;
            z-index: 999;
        }
    </style>
    """, unsafe_allow_html=True)

    # Input options container
    with st.container():
        col1, col2, col3, col4 = st.columns([3, 2, 2, 1])

        with col1:
            tier = st.select_slider(
                "Complexity Tier",
                options=[1, 2, 3, 4],
                value=st.session_state.get("default_tier", 2),
                format_func=lambda x: f"Tier {x}",
                help="1=Direct, 2=Standard, 3=Deep, 4=Adversarial",
            )

        with col2:
            output_format = st.selectbox(
                "Output Format",
                options=["markdown", "code", "json", "html"],
                index=0,
                help="Format of the response",
            )

        with col3:
            file_upload = st.file_uploader(
                "Attach File",
                type=["txt", "md", "py", "js", "json", "csv"],
                help="Attach context or requirements",
            )

        with col4:
            st.write("")  # Spacer
            auto_submit = st.checkbox("Auto-submit", value=False)

    # Chat input
    prompt = st.chat_input(
        "Enter your request...",
        key="chat_input",
        accept_file=False,
    )

    # Store options for processing
    if prompt:
        st.session_state.last_chat_options = {
            "tier": tier,
            "format": output_format,
            "file": file_upload,
        }

    return prompt


def render_advanced_options() -> Dict[str, Any]:
    """Render advanced options in an expander."""
    options = {}

    with st.expander("⚙️ Advanced Options", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            options["max_turns"] = st.slider(
                "Max Turns",
                min_value=10,
                max_value=200,
                value=50,
                help="Maximum agent conversation turns",
            )

            options["timeout"] = st.slider(
                "Timeout (seconds)",
                min_value=30,
                max_value=600,
                value=120,
                help="Maximum execution time",
            )

        with col2:
            options["budget"] = st.number_input(
                "Max Budget ($)",
                min_value=0.1,
                max_value=100.0,
                value=st.session_state.get("max_budget", 10.0),
                step=0.5,
                help="Maximum cost in USD",
            )

            options["verbose"] = st.checkbox(
                "Verbose Output",
                value=st.session_state.get("show_developer_mode", False),
                help="Show detailed agent logs",
            )

        options["save_to_knowledge"] = st.checkbox(
            "💾 Save to Knowledge Base",
            value=False,
            help="Extract and save learnings",
        )

        options["enable_smes"] = st.checkbox(
            "👤 Enable SMEs",
            value=True,
            help="Allow SME persona spawning",
        )

        options["ensemble"] = st.selectbox(
            "Ensemble Pattern",
            options=["None", "Architecture Review Board", "Code Sprint", "Research Council",
                     "Document Assembly", "Requirements Workshop"],
            index=0,
        )

    return options


# =============================================================================
# UI Components - Message Display
# =============================================================================

def render_message(message: ChatMessage) -> None:
    """Render a single chat message."""
    role = message.role

    # Style based on role
    if role == MessageRole.USER:
        st.chat_message("user").write(message.content)

    elif role == MessageRole.ASSISTANT:
        with st.chat_message("assistant"):
            # Status indicator
            if message.status == MessageStatus.PROCESSING:
                st.info("🔄 Processing...")
            elif message.status == MessageStatus.FAILED:
                st.error("❌ Processing failed")

            # Main content
            st.markdown(message.content)

            # Metadata
            if message.metadata:
                metadata_cols = st.columns(4)
                with metadata_cols[0]:
                    if message.tier:
                        st.caption(f"Tier: {message.tier}")
                with metadata_cols[1]:
                    if message.metadata.get("duration"):
                        st.caption(f"Duration: {message.metadata['duration']:.1f}s")
                with metadata_cols[2]:
                    if message.metadata.get("tokens"):
                        st.caption(f"Tokens: {message.metadata['tokens']}")
                with metadata_cols[3]:
                    if message.metadata.get("cost"):
                        st.caption(f"Cost: ${message.metadata['cost']:.4f}")

            # Artifact download buttons
            if message.artifacts:
                for artifact in message.artifacts:
                    a_name = artifact.get("name", "artifact")
                    a_content = artifact.get("content", "")
                    a_type = artifact.get("type", "file")
                    if a_type == "file" and artifact.get("path"):
                        a_path = Path(artifact["path"])
                        if a_path.exists():
                            with open(a_path, "rb") as f:
                                st.download_button(
                                    label=f"Download {a_name}",
                                    data=f.read(),
                                    file_name=a_path.name,
                                    mime="application/octet-stream",
                                    key=f"dl_{message.message_id}_{a_name}",
                                )
                    elif a_content:
                        st.download_button(
                            label=f"Download {a_name}",
                            data=a_content,
                            file_name=a_name,
                            mime="text/plain",
                            key=f"dl_{message.message_id}_{a_name}",
                        )

    elif role == MessageRole.AGENT:
        with st.chat_message("assistant"):
            # Agent-specific header
            agent_name = message.agent_name or "Agent"
            st.caption(f"🤖 **{agent_name}**")

            st.markdown(message.content)

    elif role == MessageRole.SYSTEM:
        st.chat_message("system").info(message.content)


def render_message_history() -> None:
    """Render the entire message history."""
    history = get_chat_history()

    for message in history:
        render_message(message)


def render_streaming_response(
    response_generator,
    agent_name: str = "Assistant",
) -> str:
    """
    Render a streaming response.

    Args:
        response_generator: Generator yielding response chunks
        agent_name: Name of the responding agent

    Returns:
        The complete response
    """
    with st.chat_message("assistant"):
        st.caption(f"🤖 **{agent_name}**")

        # Placeholder for streaming content
        placeholder = st.empty()
        full_response = ""

        # Stream response
        for chunk in response_generator:
            full_response += chunk
            placeholder.markdown(full_response + "▌")

        # Final update without cursor
        placeholder.markdown(full_response)

    return full_response


# =============================================================================
# UI Components - Agent Activity During Chat
# =============================================================================

def render_chat_agent_panel() -> None:
    """Render a compact agent activity panel during chat."""
    from src.ui.components.agent_panel import (
        get_agent_activities,
        AgentTier,
    )

    activities = get_agent_activities()
    active_count = len(st.session_state.get("active_agents", []))

    if active_count == 0:
        return

    with st.expander(f"⚡ Agent Activity ({active_count} active)", expanded=False):
        # Show active agents
        for tier in [AgentTier.OPERATIONAL, AgentTier.COUNCIL, AgentTier.SME]:
            tier_activities = [a for a in activities[tier] if a.is_active]

            if tier_activities:
                st.markdown(f"**{tier.value.title()}**")

                for activity in tier_activities:
                    cols = st.columns([3, 1, 1])
                    with cols[0]:
                        st.write(f"{activity.status_emoji} {activity.agent_name}")
                    with cols[1]:
                        st.caption(activity.phase)
                    with cols[2]:
                        st.caption(f"{int(activity.progress * 100)}%")

                st.progress(activity.progress)


def render_progress_indicator(
    phase: str,
    progress: float,
    agent_name: str,
) -> None:
    """Render a progress indicator for the current operation."""
    with st.container():
        col1, col2 = st.columns([4, 1])

        with col1:
            st.write(f"🔄 {phase}")

        with col2:
            st.write(f"{int(progress * 100)}%")

        st.progress(progress)


# =============================================================================
# Chat Interface
# =============================================================================

def render_chat_interface() -> None:
    """Render the main chat interface."""
    # Initialize session if needed
    if "current_session_id" not in st.session_state or not st.session_state.current_session_id:
        st.session_state.current_session_id = f"chat_{int(time.time())}"

    # Header
    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        st.markdown("### 💬 Chat")

    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            clear_chat_history()
            st.rerun()

    with col3:
        if st.button("💾 Export", use_container_width=True):
            export_chat_history()

    st.markdown("---")

    # Advanced options
    options = render_advanced_options()

    # Message history
    render_message_history()

    # Agent activity panel (if active)
    if st.session_state.get("active_agents"):
        render_chat_agent_panel()

    # Chat input
    prompt = render_chat_input()

    # Process input
    if prompt:
        process_user_input(prompt, options)


# =============================================================================
# Processing
# =============================================================================

def process_user_input(prompt: str, options: Dict[str, Any]) -> None:
    """
    Process user input through the multi-agent system.

    Args:
        prompt: User's input prompt
        options: Processing options
    """
    # Get last chat options
    chat_options = st.session_state.get("last_chat_options", {})

    tier = chat_options.get("tier", st.session_state.get("default_tier", 2))
    output_format = chat_options.get("format", "markdown")
    file_upload = chat_options.get("file")

    # Read uploaded file content and include it
    file_content = None
    file_name = None
    if file_upload is not None:
        file_content = file_upload.read().decode("utf-8", errors="replace")
        file_name = file_upload.name

    # Build prompt with file context if attached
    full_prompt = prompt
    if file_content:
        full_prompt = (
            f"{prompt}\n\n--- Attached File: {file_name} ---\n{file_content}"
        )

    # Create user message
    user_message = ChatMessage(
        message_id=f"msg_{int(time.time() * 1000000)}",
        role=MessageRole.USER,
        content=full_prompt,
        timestamp=datetime.now(),
        metadata={"attached_file": file_name} if file_name else {},
    )

    add_message(user_message)

    # Process with orchestrator
    execute_orchestrator_request(full_prompt, tier, output_format, options)

    st.rerun()
    return  # st.rerun() halts execution; explicit return for clarity


def execute_orchestrator_request(
    prompt: str,
    tier: int,
    output_format: str,
    options: Dict[str, Any],
) -> None:
    """
    Execute a request through the real multi-agent orchestrator.

    Falls back to direct Anthropic API if orchestrator fails,
    and finally to a clear error message.

    Args:
        prompt: User's prompt (including any attached file content)
        tier: Complexity tier
        output_format: Desired output format
        options: Processing options
    """
    start_time = time.time()

    # Create a processing message
    processing_message = ChatMessage(
        message_id=f"msg_{int(time.time() * 1000000)}",
        role=MessageRole.ASSISTANT,
        content="",
        timestamp=datetime.now(),
        status=MessageStatus.PROCESSING,
        tier=tier,
        metadata={"tier": tier},
    )
    add_message(processing_message)

    response_text = ""
    response_metadata: Dict[str, Any] = {"tier": tier}

    # Try 1: Real orchestrator
    try:
        from src.agents.orchestrator import create_orchestrator

        budget = options.get("budget", 10.0)
        verbose = options.get("verbose", False)

        orchestrator = create_orchestrator(
            max_budget_usd=budget,
            verbose=verbose,
        )

        result = orchestrator.execute(
            user_prompt=prompt,
            tier_level=tier,
            format=output_format,
        )

        response_text = result.get("formatted_output", result.get("raw_output", ""))
        response_metadata.update({
            "duration": result.get("duration_seconds", time.time() - start_time),
            "tokens": result.get("metadata", {}).get("total_tokens", 0),
            "cost": result.get("total_cost_usd", 0),
            "agents_used": result.get("metadata", {}).get("agents_used", []),
            "smes_used": result.get("metadata", {}).get("smes_used", []),
            "source": "orchestrator",
        })

    except Exception as orch_error:
        # Try 2: Direct Anthropic API fallback
        try:
            from anthropic import Anthropic

            client = Anthropic()
            api_response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                system=(
                    f"You are a helpful assistant. The user's request is at "
                    f"complexity tier {tier}. Provide a thorough response."
                ),
                messages=[{"role": "user", "content": prompt}],
            )

            for block in api_response.content:
                if hasattr(block, "text"):
                    response_text += block.text

            total_tokens = (
                api_response.usage.input_tokens + api_response.usage.output_tokens
            )
            cost = (
                api_response.usage.input_tokens / 1_000_000 * 3.0
                + api_response.usage.output_tokens / 1_000_000 * 15.0
            )

            response_metadata.update({
                "duration": time.time() - start_time,
                "tokens": total_tokens,
                "cost": cost,
                "source": "anthropic_api_fallback",
            })

        except Exception:
            # Try 3: Clear error with guidance
            response_text = (
                f"**Unable to process request**\n\n"
                f"The multi-agent orchestrator and Anthropic API are both "
                f"unavailable. Please check:\n\n"
                f"1. `ANTHROPIC_API_KEY` is set in your `.env` file\n"
                f"2. The `anthropic` package is installed (`pip install anthropic`)\n"
                f"3. Network connectivity to the Anthropic API\n\n"
                f"**Orchestrator error:** {orch_error}\n\n"
                f"Configure your API key in the Settings page to enable "
                f"the multi-agent reasoning system."
            )
            response_metadata.update({
                "duration": time.time() - start_time,
                "error": str(orch_error),
                "source": "unavailable",
            })

    # Update the processing message with the response
    processing_message.content = response_text
    processing_message.status = MessageStatus.COMPLETED
    processing_message.metadata = response_metadata


# =============================================================================
# Export
# =============================================================================

def export_chat_history() -> None:
    """Export chat history to a file."""
    history = get_chat_history()

    if not history:
        st.info("No chat history to export")
        return

    # Format as markdown
    export_content = "# Chat History\n\n"
    export_content += f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    export_content += f"Session: {st.session_state.get('current_session_id', 'unknown')}\n\n"
    export_content += "---\n\n"

    for msg in history:
        role_header = {
            MessageRole.USER: "## 👤 User",
            MessageRole.ASSISTANT: "## 🤖 Assistant",
            MessageRole.SYSTEM: "## 📋 System",
            MessageRole.AGENT: f"## 🎭 {msg.agent_name or 'Agent'}",
        }.get(msg.role, f"## {msg.role.value}")

        export_content += f"{role_header}\n\n"
        export_content += f"{msg.content}\n\n"
        export_content += f"*{msg.timestamp.strftime('%H:%M:%S')}*\n\n"
        export_content += "---\n\n"

    # Offer download
    st.download_button(
        label="📥 Download Chat History",
        data=export_content,
        file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown",
    )
