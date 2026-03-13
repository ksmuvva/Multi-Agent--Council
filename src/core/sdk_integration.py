"""
SDK Integration Module - Claude Agent SDK Configuration

Provides ClaudeAgentOptions configuration, allowedTools declarations,
outputFormat schemas, MCP server registration, and SDK query wrappers.
"""

import json
import os
from typing import Any, Dict, List, Optional
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
            import time as time_mod
            wait_seconds = 2 ** (retry_count + 1)
            time_mod.sleep(wait_seconds)
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
        # Fall back to direct Anthropic API
        return _execute_anthropic_api(sdk_kwargs, input_data)


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
    from src.tools.custom_tools import get_all_tools

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

AGENT_SKILLS: Dict[str, List[str]] = {
    "executor": ["code-generation"],
    "formatter": ["document-creation"],
    "analyst": ["requirements-engineering"],
    "planner": ["architecture-design"],
    "researcher": ["web-research"],
    "code_reviewer": ["code-generation"],
    "orchestrator": ["multi-agent-reasoning"],
}

# Active per-task skill overrides (set via override_agent_skills)
_task_skill_overrides: Dict[str, List[str]] = {}


def get_skills_for_agent(
    agent_name: str,
    task_context: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    Get the skill names that should be loaded for an agent.

    Supports per-task skill overrides via task_context or the global
    override registry. Priority order:
    1. task_context["skill_overrides"][agent_name] (per-task override)
    2. _task_skill_overrides[agent_name] (global session override)
    3. AGENT_SKILLS[agent_name] (default assignment)

    Args:
        agent_name: Normalized agent name
        task_context: Optional task context with skill_overrides dict
    """
    # Priority 1: Per-task override from context
    if task_context:
        overrides = task_context.get("skill_overrides", {})
        if agent_name in overrides:
            return list(overrides[agent_name])

    # Priority 2: Global session override
    if agent_name in _task_skill_overrides:
        return list(_task_skill_overrides[agent_name])

    # Priority 3: Default assignment
    return list(AGENT_SKILLS.get(agent_name, []))


def override_agent_skills(
    agent_name: str,
    skills: List[str],
) -> None:
    """
    Override the skill assignment for an agent for the current session.

    This enables per-task skill customization as required by FR-028.

    Args:
        agent_name: Normalized agent name (e.g., "executor", "analyst")
        skills: List of skill names to assign (must exist in .claude/skills/)
    """
    _task_skill_overrides[agent_name] = list(skills)


def clear_skill_overrides() -> None:
    """Clear all per-task skill overrides, reverting to defaults."""
    _task_skill_overrides.clear()


def get_skill_assignments() -> Dict[str, List[str]]:
    """
    Get the current effective skill assignments for all agents.

    Merges defaults with any active overrides.
    """
    effective = dict(AGENT_SKILLS)
    effective.update(_task_skill_overrides)
    return effective


def get_skills_for_sme(persona_id: str) -> List[str]:
    """Get skills for an SME persona from the registry."""
    from src.core.sme_registry import get_persona
    persona = get_persona(persona_id)
    if persona:
        return persona.skill_files
    return []


# =============================================================================
# Skill Chaining (FR-029)
# =============================================================================

@dataclass
class SkillChainStep:
    """A step in a skill chain."""
    skill_name: str
    agent_name: str
    input_from: Optional[str] = None  # Previous step's agent_name to use output from
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillChain:
    """
    A configurable skill chain that defines the order of skill
    invocations across agents in the pipeline.

    Chains allow skills to reference outputs from prior skill invocations,
    enabling composed workflows like:
      requirements-engineering -> architecture-design -> code-generation
    """
    name: str
    description: str
    steps: List[SkillChainStep] = field(default_factory=list)

    def add_step(
        self,
        skill_name: str,
        agent_name: str,
        input_from: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> "SkillChain":
        """Add a step to the chain. Returns self for fluent chaining."""
        self.steps.append(SkillChainStep(
            skill_name=skill_name,
            agent_name=agent_name,
            input_from=input_from,
            config=config or {},
        ))
        return self

    def get_skill_overrides(self) -> Dict[str, List[str]]:
        """Convert chain steps to skill overrides for agents."""
        overrides: Dict[str, List[str]] = {}
        for step in self.steps:
            if step.agent_name not in overrides:
                overrides[step.agent_name] = []
            if step.skill_name not in overrides[step.agent_name]:
                overrides[step.agent_name].append(step.skill_name)
        return overrides

    def get_input_mapping(self) -> Dict[str, str]:
        """Get the input dependency mapping for the chain."""
        mapping = {}
        for step in self.steps:
            if step.input_from:
                mapping[step.agent_name] = step.input_from
        return mapping


# Built-in skill chains
BUILTIN_CHAINS: Dict[str, SkillChain] = {
    "full_development": SkillChain(
        name="full_development",
        description="Full development workflow: requirements -> architecture -> code -> tests",
        steps=[
            SkillChainStep(
                skill_name="requirements-engineering",
                agent_name="analyst",
            ),
            SkillChainStep(
                skill_name="architecture-design",
                agent_name="planner",
                input_from="analyst",
            ),
            SkillChainStep(
                skill_name="code-generation",
                agent_name="executor",
                input_from="planner",
            ),
            SkillChainStep(
                skill_name="test-case-generation",
                agent_name="code_reviewer",
                input_from="executor",
            ),
        ],
    ),
    "research_and_document": SkillChain(
        name="research_and_document",
        description="Research a topic and produce documentation",
        steps=[
            SkillChainStep(
                skill_name="web-research",
                agent_name="researcher",
            ),
            SkillChainStep(
                skill_name="document-creation",
                agent_name="formatter",
                input_from="researcher",
            ),
        ],
    ),
    "architecture_review": SkillChain(
        name="architecture_review",
        description="Architecture design with multi-agent review",
        steps=[
            SkillChainStep(
                skill_name="requirements-engineering",
                agent_name="analyst",
            ),
            SkillChainStep(
                skill_name="architecture-design",
                agent_name="planner",
                input_from="analyst",
            ),
            SkillChainStep(
                skill_name="multi-agent-reasoning",
                agent_name="orchestrator",
                input_from="planner",
            ),
        ],
    ),
}


def get_skill_chain(chain_name: str) -> Optional[SkillChain]:
    """Get a skill chain by name."""
    return BUILTIN_CHAINS.get(chain_name)


def list_skill_chains() -> Dict[str, str]:
    """List available skill chains with descriptions."""
    return {name: chain.description for name, chain in BUILTIN_CHAINS.items()}


def apply_skill_chain(chain_name: str) -> Optional[SkillChain]:
    """
    Apply a skill chain, setting up skill overrides for all agents in the chain.

    Args:
        chain_name: Name of the chain to apply

    Returns:
        The SkillChain if found, None otherwise
    """
    chain = get_skill_chain(chain_name)
    if chain is None:
        return None

    # Apply skill overrides from the chain
    overrides = chain.get_skill_overrides()
    for agent_name, skills in overrides.items():
        override_agent_skills(agent_name, skills)

    return chain


def create_custom_chain(
    name: str,
    description: str,
    steps: List[Dict[str, Any]],
) -> SkillChain:
    """
    Create a custom skill chain from a list of step definitions.

    Args:
        name: Chain name
        description: Chain description
        steps: List of dicts with keys: skill_name, agent_name, input_from (optional)

    Returns:
        The created SkillChain
    """
    chain = SkillChain(name=name, description=description)
    for step in steps:
        chain.add_step(
            skill_name=step["skill_name"],
            agent_name=step["agent_name"],
            input_from=step.get("input_from"),
            config=step.get("config", {}),
        )

    # Register it
    BUILTIN_CHAINS[name] = chain
    return chain
