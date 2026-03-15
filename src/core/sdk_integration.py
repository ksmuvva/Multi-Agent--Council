"""
SDK Integration Module - Claude Agent SDK Configuration

Provides ClaudeAgentOptions configuration, allowedTools declarations,
outputFormat schemas, MCP server registration, and SDK query wrappers.
"""

import json
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum

from src.config import get_model_for_agent, get_settings
from src.utils.logging import get_logger


# =============================================================================
# Agent SDK Configuration Types
# =============================================================================

class PermissionMode(str, Enum):
    """SDK permission modes for subagents."""
    DEFAULT = "default"
    ACCEPT_EDITS = "acceptEdits"


@dataclass
class ClaudeAgentOptions:
    """
    Configuration options for a Claude Agent SDK subagent.

    Maps to the SDK's query() / Task tool parameters.
    """
    name: str
    model: str
    system_prompt: str
    max_turns: int = 30
    allowed_tools: List[str] = field(default_factory=list)
    output_format: Optional[Dict[str, Any]] = None
    setting_sources: List[str] = field(default_factory=lambda: ["user", "project"])
    permission_mode: PermissionMode = PermissionMode.DEFAULT
    append_system_prompt: Optional[str] = None

    def to_sdk_kwargs(self) -> Dict[str, Any]:
        """Convert to kwargs for SDK query() / Task tool invocation."""
        kwargs = {
            "name": self.name,
            "model": self.model,
            "system_prompt": self.system_prompt,
            "max_turns": self.max_turns,
        }
        if self.allowed_tools:
            kwargs["allowed_tools"] = self.allowed_tools
        if self.output_format:
            kwargs["output_format"] = self.output_format
        if self.setting_sources:
            kwargs["setting_sources"] = self.setting_sources
        if self.permission_mode != PermissionMode.DEFAULT:
            kwargs["permission_mode"] = self.permission_mode.value
        if self.append_system_prompt:
            kwargs["append_system_prompt"] = self.append_system_prompt
        return kwargs


# =============================================================================
# Per-Agent Allowed Tools (Least-Privilege)
# =============================================================================

AGENT_ALLOWED_TOOLS: Dict[str, List[str]] = {
    # Operational Agents
    "analyst": [
        "Read", "Glob", "Grep",
    ],
    "planner": [
        "Read", "Glob",
    ],
    "clarifier": [],
    "researcher": [
        "WebSearch", "WebFetch", "Read",
    ],
    "executor": [
        "Read", "Write", "Edit", "Bash", "Glob", "Grep",
        "Skill",
    ],
    "code_reviewer": [
        "Read", "Glob", "Grep", "Bash",
    ],
    "formatter": [
        "Read", "Write", "Bash", "Skill",
    ],
    "verifier": [
        "Read", "WebSearch", "WebFetch",
    ],
    "critic": [
        "Read", "Grep",
    ],
    "reviewer": [
        "Read", "Glob", "Grep",
    ],
    "memory_curator": [
        "Read", "Write", "Glob",
    ],
    # Council Agents
    "council_chair": [],
    "quality_arbiter": [],
    "ethics_advisor": [],
    # SME Personas (base set - extended per-persona)
    "sme_default": [
        "Read", "Glob", "Grep", "Skill",
    ],
}


# =============================================================================
# Per-Agent Output Format (JSON Schema from Pydantic)
# =============================================================================

def _get_output_schema(agent_name: str) -> Optional[Dict[str, Any]]:
    """
    Get the JSON Schema for an agent's output format.

    Uses Pydantic model_json_schema() to generate SDK-compatible schemas.
    """
    schema_map = {
        "analyst": "TaskIntelligenceReport",
        "planner": "ExecutionPlan",
        "clarifier": "ClarificationRequest",
        "researcher": "EvidenceBrief",
        "code_reviewer": "CodeReviewReport",
        "verifier": "VerificationReport",
        "critic": "CritiqueReport",
        "reviewer": "ReviewVerdict",
        "council_chair": "SMESelectionReport",
        "quality_arbiter": "QualityVerdict",
        "ethics_advisor": "EthicsReview",
    }

    model_name = schema_map.get(agent_name)
    if not model_name:
        return None

    try:
        import src.schemas as schemas
        model_class = getattr(schemas, model_name, None)
        if model_class and hasattr(model_class, "model_json_schema"):
            return model_class.model_json_schema()
    except Exception as e:
        get_logger("sdk_integration").debug(f"Failed to load schema '{model_name}': {e}")

    return None


# =============================================================================
# Agent Configuration Builder
# =============================================================================

def build_agent_options(
    agent_name: str,
    system_prompt: str,
    agent_role: str = "",
    model_override: Optional[str] = None,
    max_turns_override: Optional[int] = None,
    extra_tools: Optional[List[str]] = None,
    extra_system_prompt: Optional[str] = None,
) -> ClaudeAgentOptions:
    """
    Build ClaudeAgentOptions for a specific agent.

    Args:
        agent_name: Normalized agent name (e.g., "analyst", "executor")
        system_prompt: The agent's system prompt
        agent_role: Optional role qualifier (e.g., "chair", "arbiter")
        model_override: Override the configured model
        max_turns_override: Override max turns
        extra_tools: Additional tools beyond the default set
        extra_system_prompt: Additional system prompt instructions

    Returns:
        Configured ClaudeAgentOptions
    """
    settings = get_settings()

    # Determine model
    model = model_override or get_model_for_agent(agent_name)

    # Determine max turns
    if max_turns_override:
        max_turns = max_turns_override
    elif agent_name == "orchestrator":
        max_turns = settings.max_turns_orchestrator
    elif agent_name == "executor":
        max_turns = settings.max_turns_executor
    else:
        max_turns = settings.max_turns_subagent

    # Build allowed tools list
    tool_key = agent_name
    if agent_role:
        role_key = f"{agent_name}_{agent_role}"
        if role_key in AGENT_ALLOWED_TOOLS:
            tool_key = role_key
    allowed_tools = list(AGENT_ALLOWED_TOOLS.get(tool_key, []))
    if extra_tools:
        allowed_tools.extend(extra_tools)

    # Get output format schema
    output_format = _get_output_schema(
        f"{agent_name}_{agent_role}" if agent_role else agent_name
    )

    # Build display name
    display_name = agent_name.replace("_", " ").title()
    if agent_role:
        display_name = f"{agent_role.replace('_', ' ').title()}"

    # Permission mode: Executor gets acceptEdits for file operations
    permission_mode = PermissionMode.DEFAULT
    if agent_name == "executor":
        permission_mode = PermissionMode.ACCEPT_EDITS

    return ClaudeAgentOptions(
        name=display_name,
        model=model,
        system_prompt=system_prompt,
        max_turns=max_turns,
        allowed_tools=allowed_tools,
        output_format=output_format,
        setting_sources=["user", "project"],
        permission_mode=permission_mode,
        append_system_prompt=extra_system_prompt,
    )


# =============================================================================
# SDK Query Wrapper
# =============================================================================

def spawn_subagent(
    options: ClaudeAgentOptions,
    input_data: str,
    retry_count: int = 0,
    max_retries: int = 2,
) -> Dict[str, Any]:
    """
    Spawn a subagent using the Claude Agent SDK.

    Wraps the SDK's Task tool / query() method with retry logic,
    cost tracking, and structured output validation.

    Args:
        options: Agent configuration
        input_data: The prompt/data to send to the subagent
        retry_count: Current retry attempt (for internal recursion)
        max_retries: Maximum retry attempts

    Returns:
        Dictionary with status, output, tokens_used, cost_usd
    """
    import time

    start_time = time.time()

    try:
        # Attempt SDK query
        sdk_kwargs = options.to_sdk_kwargs()
        result = _execute_sdk_query(sdk_kwargs, input_data)

        end_time = time.time()

        # Parse and validate output
        output = result.get("output", "")
        tokens_used = result.get("tokens_used", 0)
        cost_usd = result.get("cost_usd", 0.0)

        # Validate against output schema if configured
        if options.output_format and output:
            validated = _validate_output(output, options.output_format)
            if not validated and retry_count < max_retries:
                # Retry with schema reminder
                return spawn_subagent(
                    options=options,
                    input_data=f"{input_data}\n\nIMPORTANT: Your response MUST conform to the JSON schema provided in output_format.",
                    retry_count=retry_count + 1,
                    max_retries=max_retries,
                )

        return {
            "status": "success",
            "output": output,
            "tokens_used": tokens_used,
            "cost_usd": cost_usd,
            "duration_ms": int((end_time - start_time) * 1000),
            "model": options.model,
            "retries": retry_count,
        }

    except Exception as e:
        # Retry on transient errors
        if retry_count < max_retries:
            # Exponential backoff
            wait_seconds = 2 ** (retry_count + 1)
            time.sleep(wait_seconds)
            return spawn_subagent(
                options=options,
                input_data=input_data,
                retry_count=retry_count + 1,
                max_retries=max_retries,
            )

        return {
            "status": "error",
            "output": None,
            "error": str(e),
            "tokens_used": 0,
            "cost_usd": 0.0,
            "duration_ms": int((time.time() - start_time) * 1000),
            "model": options.model,
            "retries": retry_count,
        }


def _execute_sdk_query(
    sdk_kwargs: Dict[str, Any],
    input_data: str,
) -> Dict[str, Any]:
    """
    Execute the actual SDK query.

    This method interfaces with the Claude Agent SDK's query() method
    or Task tool depending on the execution context.

    In production, this calls claude_agent_sdk.query().
    Falls back to direct Anthropic API calls if SDK is not available.
    """
    try:
        # Try Claude Agent SDK first
        from claude_agent_sdk import query as sdk_query

        result = sdk_query(
            prompt=input_data,
            system=sdk_kwargs.get("system_prompt", ""),
            model=sdk_kwargs.get("model", "claude-sonnet-4-20250514"),
            max_turns=sdk_kwargs.get("max_turns", 30),
            allowed_tools=sdk_kwargs.get("allowed_tools"),
            output_format=sdk_kwargs.get("output_format"),
        )

        return {
            "output": result.get("response", result.get("output", "")),
            "tokens_used": result.get("usage", {}).get("total_tokens", 0),
            "cost_usd": result.get("cost", 0.0),
        }

    except ImportError:
        # Check if we should use GLM API instead
        settings = get_settings()
        if settings.llm_provider.value == "glm":
            return _execute_glm_api(sdk_kwargs, input_data)
        # Fall back to direct Anthropic API
        return _execute_anthropic_api(sdk_kwargs, input_data)


def _execute_glm_api(
    sdk_kwargs: Dict[str, Any],
    input_data: str,
) -> Dict[str, Any]:
    """
    Execute via ZhipuAI GLM API (OpenAI-compatible endpoint).

    Used when MAS_LLM_PROVIDER=glm.
    """
    import httpx
    import time as _time

    settings = get_settings()
    api_key = settings.get_api_key()
    base_url = settings.get_base_url() or "https://open.bigmodel.cn/api/paas/v4"
    model = sdk_kwargs.get("model", "glm-4-plus")

    messages = []
    system_prompt = sdk_kwargs.get("system_prompt", "")
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": input_data})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 4096,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    logger = get_logger("sdk_integration")

    for attempt in range(3):
        try:
            with httpx.Client(timeout=120) as client:
                resp = client.post(
                    f"{base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                )

            if resp.status_code != 200:
                logger.warning(
                    "glm_api.error",
                    status=resp.status_code,
                    body=resp.text[:200],
                    attempt=attempt + 1,
                )
                if resp.status_code in (429, 500, 502, 503):
                    _time.sleep(2 ** (attempt + 1))
                    continue
                break

            data = resp.json()
            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})
            usage = data.get("usage", {})

            total_tokens = usage.get("total_tokens", 0)
            # GLM pricing estimate (approx)
            cost = total_tokens / 1_000_000 * 2.0

            return {
                "output": message.get("content", ""),
                "tokens_used": total_tokens,
                "cost_usd": cost,
            }

        except Exception as e:
            logger.warning("glm_api.retry", error=str(e), attempt=attempt + 1)
            _time.sleep(2 ** (attempt + 1))

    # All retries exhausted - fall back to simulation
    logger.error("glm_api.all_retries_exhausted", model=model)
    return _simulate_response(sdk_kwargs, input_data)


def _execute_anthropic_api(
    sdk_kwargs: Dict[str, Any],
    input_data: str,
) -> Dict[str, Any]:
    """
    Fall back to direct Anthropic API calls.

    Used when claude_agent_sdk is not available.
    """
    try:
        from anthropic import Anthropic

        client = Anthropic()

        messages = [{"role": "user", "content": input_data}]

        api_kwargs = {
            "model": sdk_kwargs.get("model", "claude-sonnet-4-20250514"),
            "max_tokens": 4096,
            "system": sdk_kwargs.get("system_prompt", ""),
            "messages": messages,
        }

        response = client.messages.create(**api_kwargs)

        # Extract text content
        output = ""
        for block in response.content:
            if hasattr(block, "text"):
                output += block.text

        # Calculate tokens
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        total_tokens = input_tokens + output_tokens

        # Estimate cost (Claude Sonnet pricing)
        cost = (input_tokens / 1_000_000 * 3.0) + (output_tokens / 1_000_000 * 15.0)

        return {
            "output": output,
            "tokens_used": total_tokens,
            "cost_usd": cost,
        }

    except ImportError:
        # Check GLM before simulation
        settings = get_settings()
        if settings.llm_provider.value == "glm":
            return _execute_glm_api(sdk_kwargs, input_data)
        # No SDK available - return simulated response
        return _simulate_response(sdk_kwargs, input_data)


def _simulate_response(
    sdk_kwargs: Dict[str, Any],
    input_data: str,
) -> Dict[str, Any]:
    """
    Simulate a response when no API is available.

    This is the fallback for development/testing without API keys.
    WARNING: Not for production use - returns simulated data.
    """
    get_logger("sdk_integration").warning(
        f"Using simulated response - no API available for agent '{sdk_kwargs.get('name', 'Agent')}'"
    )
    agent_name = sdk_kwargs.get("name", "Agent")
    return {
        "output": f"[Simulated output from {agent_name}] Processed: {input_data[:200]}...",
        "tokens_used": 500,
        "cost_usd": 0.005,
    }


def _validate_output(output: Any, schema: Dict[str, Any]) -> bool:
    """Validate output against JSON Schema."""
    if not output:
        return False

    try:
        # Try to parse as JSON
        if isinstance(output, str):
            parsed = json.loads(output)
        elif isinstance(output, dict):
            parsed = output
        else:
            return False

        # Basic schema validation - check required fields
        required = schema.get("required", [])
        if required:
            return all(key in parsed for key in required)

        return True

    except (json.JSONDecodeError, TypeError):
        return False


# =============================================================================
# MCP Server Registration
# =============================================================================

def create_sdk_mcp_server() -> Dict[str, Any]:
    """
    Create and register an MCP server with the SDK for custom tools.

    Returns:
        MCP server configuration dictionary
    """
    try:
        from src.tools.custom_tools import get_all_tools
    except ImportError:
        get_logger("sdk_integration").warning("Custom tools module not available")
        return {"name": "multi-agent-reasoning", "version": "1.0.0", "tools": [], "tool_count": 0}

    tools = get_all_tools()

    # Build MCP tool definitions
    mcp_tools = []
    for name, metadata in tools.items():
        tool_def = {
            "name": name,
            "description": metadata.description,
            "parameters": {
                "type": "object",
                "properties": {
                    param: {"type": "string", "description": desc}
                    for param, desc in metadata.parameters.items()
                },
            },
        }
        mcp_tools.append(tool_def)

    server_config = {
        "name": "multi-agent-reasoning",
        "version": "1.0.0",
        "tools": mcp_tools,
        "tool_count": len(mcp_tools),
    }

    return server_config


# =============================================================================
# Skill Integration
# =============================================================================

def get_skills_for_agent(agent_name: str) -> List[str]:
    """
    Get the skill names that should be loaded for an agent.

    Maps agents to their assigned skills from the skill system.
    """
    AGENT_SKILLS: Dict[str, List[str]] = {
        "executor": ["code-generation"],
        "formatter": ["document-creation"],
        "analyst": ["requirements-engineering"],
        "planner": ["architecture-design"],
        "researcher": ["web-research"],
        "code_reviewer": ["code-generation"],
        "orchestrator": ["multi-agent-reasoning"],
    }
    return AGENT_SKILLS.get(agent_name, [])


def get_skills_for_sme(persona_id: str) -> List[str]:
    """Get skills for an SME persona from the registry."""
    from src.core.sme_registry import get_persona
    persona = get_persona(persona_id)
    if persona:
        return persona.skill_files
    return []
