"""
MCP Tools - Custom Model Context Protocol Tools

Custom tools for in-process agent operations using @tool decorator.
Tools provide extended capabilities to agents with proper documentation.
"""

import functools
import json
import time
from typing import Callable, Any, Dict, List, Optional, get_type_hints
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


# =============================================================================
# Tool Registry
# =============================================================================

class ToolCategory(str, Enum):
    """Categories of MCP tools."""
    SYSTEM = "system"          # System operations
    KNOWLEDGE = "knowledge"    # Knowledge base access
    ANALYSIS = "analysis"      # Code/data analysis
    EXECUTION = "execution"    # Task execution
    MONITORING = "monitoring"  # Cost and performance tracking
    SME = "sme"                # SME registry operations


@dataclass
class ToolMetadata:
    """Metadata about an MCP tool."""
    name: str
    description: str
    category: ToolCategory
    parameters: Dict[str, str]  # name -> description
    return_type: str
    author: str = "Multi-Agent System"
    version: str = "1.0.0"
    examples: List[str] = field(default_factory=list)


class ToolRegistry:
    """Registry of all available MCP tools."""

    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._metadata: Dict[str, ToolMetadata] = {}

    def register(
        self,
        name: str,
        func: Callable,
        metadata: ToolMetadata,
    ) -> Callable:
        """Register a tool function."""
        self._tools[name] = func
        self._metadata[name] = metadata
        return func

    def get_tool(self, name: str) -> Optional[Callable]:
        """Get a tool by name."""
        return self._tools.get(name)

    def get_metadata(self, name: str) -> Optional[ToolMetadata]:
        """Get tool metadata."""
        return self._metadata.get(name)

    def list_tools(self, category: Optional[ToolCategory] = None) -> List[str]:
        """List all tools, optionally filtered by category."""
        if category:
            return [
                name for name, meta in self._metadata.items()
                if meta.category == category
            ]
        return list(self._tools.keys())

    def get_all_metadata(self) -> Dict[str, ToolMetadata]:
        """Get metadata for all tools."""
        return self._metadata.copy()


# Global registry
_tool_registry = ToolRegistry()


# =============================================================================
# Tool Decorator
# =============================================================================

def tool(
    name: str,
    description: str,
    category: ToolCategory,
    parameters: Optional[Dict[str, str]] = None,
    examples: Optional[List[str]] = None,
):
    """
    Decorator to register an MCP tool.

    Args:
        name: Unique name for the tool
        description: What the tool does
        category: Tool category
        parameters: Parameter descriptions
        examples: Usage examples

    Usage:
        @tool(name="get_user", description="Get user info", category=ToolCategory.SYSTEM)
        def get_user(user_id: str) -> dict:
            return {"user_id": user_id, "name": "John Doe"}
    """
    def decorator(func: Callable) -> Callable:
        # Infer parameter types from function signature
        type_hints = get_type_hints(func)
        func_params = {}

        if parameters is None:
            # Build from type hints
            for param_name, param_type in type_hints.items():
                if param_name != "return":
                    param_type_str = str(param_type).replace("typing.", "")
                    func_params[param_name] = f"({param_type_str})"
        else:
            func_params = parameters

        # Build return type description
        return_type = str(type_hints.get("return", "Any"))
        return_type = return_type.replace("typing.", "")

        # Create metadata
        metadata = ToolMetadata(
            name=name,
            description=description,
            category=category,
            parameters=func_params,
            return_type=return_type,
            examples=examples or [],
        )

        # Register the tool
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        # Register in global registry
        _tool_registry.register(name, wrapper, metadata)

        return wrapper

    return decorator


def get_tool(name: str) -> Optional[Callable]:
    """Get a tool from the registry."""
    return _tool_registry.get_tool(name)


def list_tools(category: Optional[ToolCategory] = None) -> List[str]:
    """List available tools."""
    return _tool_registry.list_tools(category)


def get_tool_metadata(name: str) -> Optional[ToolMetadata]:
    """Get tool metadata."""
    return _tool_registry.get_metadata(name)


def get_all_tools() -> Dict[str, ToolMetadata]:
    """Get all tool metadata."""
    return _tool_registry.get_all_metadata()


# =============================================================================
# Built-in MCP Tools
# =============================================================================

@tool(
    name="sme_query_registry",
    description="Query the SME registry to find personas by keywords or domain",
    category=ToolCategory.SME,
    parameters={
        "keywords": "List of keywords to search for",
        "domain": "Domain to filter by (optional)",
    },
    examples=[
        "sme_query_registry(keywords=['cloud', 'aws'])",
        "sme_query_registry(domain='security')",
    ],
)
def sme_query_registry(
    keywords: Optional[List[str]] = None,
    domain: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Query the SME registry for matching personas.

    Args:
        keywords: Optional list of keywords to search for
        domain: Optional domain to filter by

    Returns:
        Dictionary with matching SME personas
    """
    from src.core.sme_registry import (
        find_personas_by_keywords,
        find_personas_by_domain,
        get_persona_for_display,
    )

    results = []

    if keywords:
        personas = find_personas_by_keywords(keywords)
    elif domain:
        personas = find_personas_by_domain([domain])
    else:
        personas = []

    for persona in personas:
        display_info = get_persona_for_display(persona.persona_id)
        if display_info:
            results.append(display_info)

    return {
        "count": len(results),
        "personas": results,
    }


@tool(
    name="sme_get_persona",
    description="Get detailed information about a specific SME persona",
    category=ToolCategory.SME,
    parameters={
        "persona_id": "The persona ID (e.g., 'cloud_architect')",
    },
    examples=[
        "sme_get_persona(persona_id='cloud_architect')",
        "sme_get_persona(persona_id='security_analyst')",
    ],
)
def sme_get_persona(persona_id: str) -> Dict[str, Any]:
    """
    Get detailed information about an SME persona.

    Args:
        persona_id: The persona ID

    Returns:
        Dictionary with persona details
    """
    from src.core.sme_registry import (
        get_persona,
        get_persona_for_display,
    )

    persona = get_persona(persona_id)
    if persona is None:
        return {
            "error": f"Persona '{persona_id}' not found",
            "available_personas": list(
                __import__('src.core.sme_registry', fromlist=['SME_REGISTRY'])
                .SME_REGISTRY.keys()
            ),
        }

    display_info = get_persona_for_display(persona_id)

    return {
        "persona_id": persona.persona_id,
        "name": persona.name,
        "domain": persona.domain,
        "description": persona.description,
        "trigger_keywords": persona.trigger_keywords,
        "skill_files": persona.skill_files,
        "interaction_modes": [m.value for m in persona.interaction_modes],
        "default_model": persona.default_model,
    }


@tool(
    name="knowledge_retrieve",
    description="Retrieve relevant knowledge entries from the knowledge base",
    category=ToolCategory.KNOWLEDGE,
    parameters={
        "query": "Search query for knowledge retrieval",
        "limit": "Maximum number of entries to return (default: 5)",
    },
    examples=[
        "knowledge_retrieve(query='architecture decision', limit=3)",
        "knowledge_retrieve(query='python async patterns')",
    ],
)
def knowledge_retrieve(
    query: str,
    limit: int = 5,
) -> Dict[str, Any]:
    """
    Retrieve relevant knowledge from the knowledge base.

    Args:
        query: Search query
        limit: Maximum results

    Returns:
        Dictionary with matching knowledge entries
    """
    from src.agents.memory_curator import MemoryCuratorAgent

    curator = MemoryCuratorAgent()
    results = curator.retrieve_knowledge(query, limit=limit)

    return {
        "query": query,
        "count": len(results),
        "results": results,
    }


@tool(
    name="knowledge_list",
    description="List all knowledge entries in the knowledge base",
    category=ToolCategory.KNOWLEDGE,
    parameters={
        "category": "Optional category filter (e.g., 'architectural_decision', 'code_pattern')",
    },
    examples=[
        "knowledge_list()",
        "knowledge_list(category='architectural_decision')",
    ],
)
def knowledge_list(category: Optional[str] = None) -> Dict[str, Any]:
    """
    List all knowledge entries.

    Args:
        category: Optional category filter

    Returns:
        Dictionary with all knowledge entries
    """
    from src.agents.memory_curator import MemoryCuratorAgent

    curator = MemoryCuratorAgent()
    entries = curator.list_knowledge()

    if category:
        entries = [e for e in entries if e.get("category") == category]

    return {
        "count": len(entries),
        "entries": entries,
    }


@tool(
    name="cost_estimate",
    description="Estimate token cost for agent operations",
    category=ToolCategory.MONITORING,
    parameters={
        "agents": "List of agent names and their estimated turns",
        "tier": "Tier level (affects model selection and token count)",
    },
    examples=[
        "cost_estimate(agents=[('Executor', 10), ('Verifier', 5)], tier=2)",
        "cost_estimate(agents=[('Executor', 30)], tier=4)",
    ],
)
def cost_estimate(
    agents: List[tuple[str, int]],
    tier: int = 2,
) -> Dict[str, Any]:
    """
    Estimate the cost of agent operations.

    Args:
        agents: List of (agent_name, turns) tuples
        tier: Tier level for model selection

    Returns:
        Dictionary with cost breakdown
    """
    # Token costs per 1M tokens (approximate)
    INPUT_COSTS = {
        "claude-3-5-opus-20240507": 15.0,
        "claude-3-5-sonnet-20241022": 3.0,
        "claude-3-5-haiku-20250101": 0.25,
    }

    OUTPUT_COSTS = {
        "claude-3-5-opus-20240507": 75.0,
        "claude-3-5-sonnet-20241022": 15.0,
        "claude-3-5-haiku-20250101": 1.25,
    }

    # Model selection by tier
    TIER_MODELS = {
        1: "claude-3-5-haiku-20250101",  # Fast, cheap
        2: "claude-3-5-sonnet-20241022",
        3: "claude-3-5-opus-20240507",
        4: "claude-3-5-opus-20240507",  # Adversarial needs best
    }

    model = TIER_MODELS.get(tier, "claude-3-5-sonnet-20241022")

    # Estimate tokens per turn
    TOKENS_PER_TURN = {
        "Analyst": 1000,
        "Planner": 800,
        "Clarifier": 600,
        "Researcher": 2000,  # Might use WebSearch
        "Executor": 1500,
        "CodeReviewer": 1200,
        "Formatter": 500,
        "Verifier": 1000,
        "Critic": 1200,
        "Reviewer": 800,
        "MemoryCurator": 400,
        "CouncilChair": 600,
        "QualityArbiter": 600,
        "EthicsAdvisor": 800,
    }

    # Calculate costs
    total_input_tokens = 0
    total_output_tokens = 0
    agent_costs = []

    for agent_name, turns in agents:
        tokens_per_turn = TOKENS_PER_TURN.get(agent_name, 1000)
        input_tokens = tokens_per_turn * turns
        output_tokens = tokens_per_turn * turns * 0.8  # Output typically 80% of input

        agent_cost = (
            (input_tokens / 1_000_000) * INPUT_COSTS[model] +
            (output_tokens / 1_000_000) * OUTPUT_COSTS[model]
        )

        agent_costs.append({
            "agent": agent_name,
            "turns": turns,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": round(agent_cost, 4),
            "model": model,
        })

        total_input_tokens += input_tokens
        total_output_tokens += output_tokens

    total_cost = sum(c["cost_usd"] for c in agent_costs)

    return {
        "tier": tier,
        "model": model,
        "total_tokens": total_input_tokens + total_output_tokens,
        "total_cost_usd": round(total_cost, 2),
        "agent_breakdown": agent_costs,
    }


@tool(
    name="system_get_status",
    description="Get current system status and configuration",
    category=ToolCategory.SYSTEM,
    parameters={},
    examples=["system_get_status()"],
)
def system_get_status() -> Dict[str, Any]:
    """
    Get current system status.

    Returns:
        Dictionary with system status
    """
    from src.core.complexity import get_active_agents, get_council_agents
    from src.core.sme_registry import get_registry_stats
    from src.core.ensemble import ENSEMBLE_REGISTRY

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "available_operational_agents": [
            "Analyst", "Planner", "Clarifier", "Researcher",
            "Executor", "CodeReviewer", "Formatter",
            "Verifier", "Critic", "Reviewer", "MemoryCurator",
        ],
        "tier1_agents": get_active_agents(1),
        "tier2_agents": get_active_agents(2),
        "tier3_agents": get_active_agents(3),
        "tier4_agents": get_active_agents(4),
        "council_agents": get_council_agents(4),
        "sme_personas": get_registry_stats(),
        "ensemble_patterns": list(ENSEMBLE_REGISTRY.keys()),
        "mcp_tools": list_tools(),
    }


# =============================================================================
# Tool Execution
# =============================================================================

def execute_tool(
    tool_name: str,
    **kwargs
) -> Any:
    """
    Execute an MCP tool.

    Args:
        tool_name: Name of the tool to execute
        **kwargs: Tool parameters

    Returns:
        Tool execution result
    """
    tool_func = get_tool(tool_name)

    if tool_func is None:
        return {
            "error": f"Tool '{tool_name}' not found",
            "available_tools": list_tools(),
        }

    try:
        return tool_func(**kwargs)
    except Exception as e:
        return {
            "error": str(e),
            "tool": tool_name,
        }


# =============================================================================
# MCP Server Registration
# =============================================================================

_mcp_server_instance = None


def create_and_register_mcp_server() -> Dict[str, Any]:
    """
    Create and register the MCP server with the Claude Agent SDK.

    Returns the server configuration for use in agent allowedTools.
    This should be called once at system startup.
    """
    global _mcp_server_instance

    if _mcp_server_instance is not None:
        return _mcp_server_instance

    tools = get_all_tools()

    # Build MCP-compatible tool definitions
    mcp_tools = []
    for name, metadata in tools.items():
        tool_def = {
            "name": f"mcp__multi_agent__{name}",
            "description": metadata.description,
            "input_schema": {
                "type": "object",
                "properties": {
                    param: {"type": "string", "description": desc}
                    for param, desc in metadata.parameters.items()
                },
            },
        }
        mcp_tools.append(tool_def)

    _mcp_server_instance = {
        "server_name": "multi-agent-reasoning",
        "version": "1.0.0",
        "tools": mcp_tools,
        "tool_count": len(mcp_tools),
        "registered": True,
    }

    return _mcp_server_instance


def get_mcp_tool_names() -> List[str]:
    """Get the list of MCP tool names for use in allowedTools."""
    server = create_and_register_mcp_server()
    return [t["name"] for t in server.get("tools", [])]


# =============================================================================
# Convenience Functions
# =============================================================================

def create_tool_summary() -> str:
    """Create a formatted summary of all available tools."""
    tools = get_all_tools()

    lines = ["# Available MCP Tools\n"]

    # Group by category
    by_category: Dict[ToolCategory, List[ToolMetadata]] = {}
    for meta in tools.values():
        if meta.category not in by_category:
            by_category[meta.category] = []
        by_category[meta.category].append(meta)

    for category in ToolCategory:
        if category not in by_category:
            continue

        lines.append(f"\n## {category.value.title()}\n")

        for meta in by_category[category]:
            lines.append(f"### {meta.name}")
            lines.append(f"**Description:** {meta.description}")
            lines.append(f"**Parameters:**")
            for param, desc in meta.parameters.items():
                lines.append(f"  - `{param}`: {desc}")
            lines.append(f"**Returns:** {meta.return_type}")
            if meta.examples:
                lines.append("**Examples:**")
                for example in meta.examples:
                    lines.append(f"  ```python")
                    lines.append(f"  {example}")
                    lines.append(f"  ```")
            lines.append("")

    return "\n".join(lines)
