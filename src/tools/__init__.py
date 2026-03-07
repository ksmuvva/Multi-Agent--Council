"""
MCP Tools - Custom Model Context Protocol Tools

This package contains custom tools for in-process agent operations.
All tools are registered using the @tool decorator for auto-discovery.
"""

from .custom_tools import (
    ToolCategory,
    ToolMetadata,
    ToolRegistry,
    tool,
    get_tool,
    list_tools,
    get_tool_metadata,
    get_all_tools,
    execute_tool,
    create_tool_summary,
    # Built-in tools
    sme_query_registry,
    sme_get_persona,
    knowledge_retrieve,
    knowledge_list,
    cost_estimate,
    system_get_status,
)

__all__ = [
    # Classes
    "ToolCategory",
    "ToolMetadata",
    "ToolRegistry",
    # Decorator
    "tool",
    # Query functions
    "get_tool",
    "list_tools",
    "get_tool_metadata",
    "get_all_tools",
    # Execution
    "execute_tool",
    # Utility
    "create_tool_summary",
    # Built-in tools
    "sme_query_registry",
    "sme_get_persona",
    "knowledge_retrieve",
    "knowledge_list",
    "cost_estimate",
    "system_get_status",
]
