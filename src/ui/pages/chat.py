"""
Chat Page - Interactive Multi-Agent Chat Interface

Provides the main chat interface for interacting with the multi-agent
reasoning system, including message history, streaming output, and
agent activity visualization.
"""

import time
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import streamlit as st

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agents.orchestrator import OrchestratorAgent
from src.config import get_settings, get_api_key, get_provider


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
        # Handle cases where metadata might be False or other non-dict values
        if not isinstance(self.metadata, dict):
            self.metadata = {}
        if not isinstance(self.artifacts, list):
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

    # Process with orchestrator (no rerun - let the message display naturally)
    execute_with_orchestrator(prompt, tier, output_format, options)


def execute_with_orchestrator(
    prompt: str,
    tier: int,
    output_format: str,
    options: Dict[str, Any],
) -> None:
    """
    Execute user request through the real multi-agent orchestrator.

    Args:
        prompt: User's prompt
        tier: Complexity tier
        output_format: Desired output format
        options: Processing options
    """
    try:
        # Get settings
        settings = get_settings()

        # Create a processing message
        processing_message = ChatMessage(
            message_id=f"msg_{int(time.time() * 1000000)}",
            role=MessageRole.ASSISTANT,
            content="🔄 *Starting multi-agent processing...*",
            timestamp=datetime.now(),
            status=MessageStatus.PROCESSING,
            tier=tier,
            metadata={"tier": tier},
        )
        add_message(processing_message)

        # Create orchestrator instance
        orchestrator = OrchestratorAgent(
            api_key=get_api_key(),
            max_budget_usd=options.get("budget", settings.max_budget),
            max_revisions=2,
            max_debate_rounds=2,
            verbose=options.get("verbose", False),
            enable_persistence=True,
            enable_auto_compact=True,
        )

        # Get session ID
        session_id = st.session_state.get("current_session_id", f"chat_{int(time.time())}")

        # Execute with orchestrator
        with st.spinner("Agents are working on your request..."):
            result = orchestrator.execute(
                user_prompt=prompt,
                tier_level=tier,
                session_id=session_id,
                format=output_format,
            )

        # Debug: Check result type
        if not isinstance(result, dict):
            raise TypeError(f"Orchestrator returned {type(result).__name__} instead of dict: {result}")

        # Extract response data safely
        response_content = result.get("formatted_output") or result.get("raw_output") or ""
        duration = result.get("duration_seconds", 0) if isinstance(result.get("duration_seconds"), (int, float)) else 0
        cost = result.get("total_cost_usd", 0) if isinstance(result.get("total_cost_usd"), (int, float)) else 0

        # Get metadata safely
        raw_metadata = result.get("metadata")
        if isinstance(raw_metadata, dict):
            metadata = raw_metadata
        else:
            metadata = {}

        # Update processing message with result
        processing_message.content = response_content
        processing_message.status = MessageStatus.COMPLETED
        processing_message.metadata = {
            "tier": tier,
            "duration": duration,
            "cost": cost,
            "tokens": metadata.get("total_tokens", 0) if isinstance(metadata, dict) else 0,
            "agents_used": metadata.get("agents_used", []) if isinstance(metadata, dict) else [],
            "smes_used": metadata.get("smes_used", []) if isinstance(metadata, dict) else [],
        }

        # Update session state with agent activity
        if isinstance(metadata, dict) and "agents_used" in metadata:
            st.session_state.active_agents = metadata["agents_used"]

    except Exception as e:
        import traceback
        # Handle error with full traceback for debugging
        tb_str = traceback.format_exc()
        error_message = ChatMessage(
            message_id=f"msg_{int(time.time() * 1000000)}",
            role=MessageRole.ASSISTANT,
            content=f"❌ **Error**: {str(e)}\n\n```\n{tb_str}\n```",
            timestamp=datetime.now(),
            status=MessageStatus.FAILED,
            tier=tier,
        )
        # Replace the processing message with error
        history = get_chat_history()
        if isinstance(history, list) and history and history[-1].role == MessageRole.ASSISTANT:
            history[-1] = error_message
        else:
            add_message(error_message)


def simulate_agent_response(
    prompt: str,
    tier: int,
    output_format: str,
    options: Dict[str, Any],
) -> None:
    """
    Simulate an agent response (placeholder for actual integration).

    Args:
        prompt: User's prompt
        tier: Complexity tier
        output_format: Desired output format
        options: Processing options
    """
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

    # Simulate processing delay
    time.sleep(1)

    # Update with response
    response = generate_mock_response(prompt, tier, output_format)

    processing_message.content = response
    processing_message.status = MessageStatus.COMPLETED
    processing_message.metadata = {
        "tier": tier,
        "duration": 1.5,
        "tokens": 150 + tier * 100,
        "cost": 0.01 + tier * 0.005,
    }

    # Show download buttons for any artifacts in the response
    if processing_message.artifacts:
        for artifact in processing_message.artifacts:
            artifact_name = artifact.get("name", "artifact")
            artifact_content = artifact.get("content", "")
            artifact_type = artifact.get("type", "file")
            if artifact_type == "file" and artifact.get("path"):
                file_path = Path(artifact["path"])
                if file_path.exists():
                    with open(file_path, "rb") as f:
                        st.download_button(
                            label=f"Download {artifact_name}",
                            data=f.read(),
                            file_name=file_path.name,
                            mime="application/octet-stream",
                            key=f"dl_{processing_message.message_id}_{artifact_name}",
                        )
            elif artifact_content:
                st.download_button(
                    label=f"Download {artifact_name}",
                    data=artifact_content,
                    file_name=artifact_name,
                    mime="text/plain",
                    key=f"dl_{processing_message.message_id}_{artifact_name}",
                )

    st.rerun()


def generate_mock_response(prompt: str, tier: int, output_format: str) -> str:
    """Generate a mock response (for testing)."""
    tier_descriptions = {
        1: "Direct",
        2: "Standard",
        3: "Deep",
        4: "Adversarial",
    }

    response = f"""# Response to: "{prompt[:50]}..."

**Tier:** {tier_descriptions.get(tier, tier)}
**Format:** {output_format}

This is a simulated response. The actual multi-agent system will provide
comprehensive responses based on the complexity tier selected.

## What happens at Tier {tier}?

"""

    if tier == 1:
        response += """
- **Executor Agent**: Generates the direct response
- **Formatter Agent**: Formats output

This tier handles simple, well-defined requests.
"""
    elif tier == 2:
        response += """
- **Analyst Agent**: Analyzes the request
- **Planner Agent**: Creates execution plan
- **Executor Agent**: Generates solution
- **Verifier Agent**: Validates output
- **Formatter Agent**: Formats output

This tier handles standard tasks requiring research and planning.
"""
    elif tier == 3:
        response += """
- **Council Chair**: Selects relevant SMEs
- **SME Personas**: Provide domain expertise
- **Analyst Agent**: Deep analysis
- **Planner Agent**: Comprehensive planning
- **Researcher Agent**: Extensive research
- **Executor Agent**: Detailed solution
- **Verifier Agent**: Thorough validation
- **Critic Agent**: Quality review
- **Formatter Agent**: Polished output

This tier handles complex, multi-domain tasks.
"""
    else:  # tier 4
        response += """
- **Full Council**: Governance and oversight
- **Quality Arbiter**: Sets quality standards
- **Ethics Advisor**: Reviews for concerns
- **Multiple SMEs**: Cross-domain expertise
- **Self-Play Debate**: Multi-perspective reasoning
- **Adversarial Critic**: Stress-tests solutions
- **Full Agent Pipeline**: All operational agents
- **Reviewer Agent**: Final quality gate
- **Formatter Agent**: Publication-ready output

This tier handles adversarial, high-stakes tasks requiring the highest quality.
"""

    response += f"""
## Next Steps

To use the actual multi-agent system:
1. Configure your API keys in Settings
2. The system will automatically route based on tier
3. Watch the Agent Activity panel for real-time updates
4. Review results in the Results tab

---

*Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

    return response


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
