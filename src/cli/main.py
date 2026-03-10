"""
Multi-Agent Reasoning System - CLI Interface

Command-line interface for interacting with the multi-agent reasoning system.
Built with Typer for rich CLI features.
"""

import typer
from typing import Optional, List
from pathlib import Path
import sys
import json
from datetime import datetime

VERSION = "0.1.0"


def get_banner() -> str:
    """Return the CLI banner text."""
    return f"Multi-Agent Reasoning System v{VERSION}"


from src.agents.orchestrator import create_orchestrator
from src.core.complexity import classify_complexity
from src.core.ensemble import suggest_ensemble
from src.utils.events import (
    emit_task_started,
    emit_task_progress,
    emit_task_completed,
    emit_system_message,
    emit_error,
    format_sse_event,
    EventType,
)

# Create CLI app
app = typer.Typer(
    name="mas",
    help="Multi-Agent Reasoning System - AI-powered task execution",
    add_completion=True,
    no_args_is_help=True,
)


@app.callback()
def setup(
    ctx: typer.Context,
    verbose: bool = False,
    config_file: Optional[str] = None,
):
    """
    Initialize the CLI with configuration.

    Sets up logging, loads environment, and initializes the orchestrator.
    """
    # Set up verbose logging if requested
    if verbose:
        import logging
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    # Load configuration
    if config_file:
        import yaml
        with open(config_file) as f:
            config = yaml.safe_load(f)
        ctx.obj = {"config": config}
    else:
        ctx.obj = {"config": {}}


# =============================================================================
# Main Commands
# =============================================================================

@app.command()
def query(
    prompt: str = typer.Argument(..., help="The task or question to execute"),
    tier: Optional[int] = typer.Option(None, min=1, max=4, help="Force specific tier (1-4)"),
    format: str = typer.Option("markdown", help="Output format"),
    file: Optional[Path] = typer.Option(None, "--file", "-o", help="Save output to file"),
    input_file: Optional[Path] = typer.Option(None, "--input-file", "-i", help="Attach input file for multimodal processing"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    session_id: Optional[str] = typer.Option(None, help="Resume existing session"),
) -> None:
    """
    Execute a query using the multi-agent reasoning system.

    The system will analyze your request, classify complexity,
    and execute with appropriate agents for the tier.

    Use --input-file to attach files for multimodal input (code, documents, data).
    """
    ctx = typer.Context

    # Emit task started event
    session_id_to_use = session_id or f"cli_{int(datetime.now().timestamp())}"
    emit_task_started("cli", prompt, tier or 2, session_id_to_use)

    # Handle multimodal input file
    file_path_str = None
    if input_file:
        if not input_file.exists():
            typer.echo(f"Error: Input file '{input_file}' not found.", err=True)
            raise typer.Exit(code=1)
        file_path_str = str(input_file)
        if verbose:
            typer.echo(f"[SYSTEM] Input file attached: {input_file}")

    try:
        # Create orchestrator
        orchestrator = create_orchestrator()

        # Show what we're doing
        if verbose:
            typer.echo(f"[SYSTEM] Session ID: {session_id_to_use}")
            if tier:
                typer.echo(f"[SYSTEM] Tier: {tier} (forced)")
            else:
                typer.echo(f"[SYSTEM] Auto-detecting tier...")

        # Classify complexity if tier not specified
        if tier is None:
            classification = classify_complexity(prompt)
            tier = classification.tier
            if verbose:
                typer.echo(f"[SYSTEM] Detected tier {tier}: {classification.reasoning}")

        # Execute query
        emit_task_progress("cli", 0.1, "Initializing agents...", session_id_to_use)

        result = orchestrator.execute(
            user_prompt=prompt,
            tier_level=tier,
            session_id=session_id_to_use,
            format=format,
            file_path=file_path_str,
        )

        emit_task_progress("cli", 0.9, "Finalizing output...", session_id_to_use)

        # Output result
        if file:
            file.parent.mkdir(parents=True, exist_ok=True)
            with open(file, 'w') as f:
                if format == "json":
                    json.dump(result, f, indent=2)
                else:
                    f.write(result.get("formatted_output", result))
            typer.echo(f"\n[SUCCESS] Output saved to: {file}")
        else:
            typer.echo(result.get("formatted_output", str(result)))

        # Emit completion
        emit_task_completed(
            "cli",
            result.get("summary", "Query completed"),
            result.get("duration_seconds", 0),
            session_id_to_use,
        )

    except Exception as e:
        emit_error("cli", str(e), session_id_to_use)
        typer.echo(f"[ERROR] {e}", err=True)
        raise typer.Exit(1)


@app.command()
def chat(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    session_id: Optional[str] = typer.Option(None, help="Resume existing chat session"),
) -> None:
    """
    Interactive chat mode with the multi-agent system.

    Enter a conversational loop where you can ask questions
    and receive answers. The system maintains context across turns.
    """
    from rich.console import Console
    from rich.prompt import Prompt
    from rich.markdown import Markdown

    console = Console()
    session_id_to_use = session_id or f"chat_{int(datetime.now().timestamp())}"

    typer.echo(f"[SYSTEM] Starting chat session: {session_id_to_use}")
    typer.echo("[SYSTEM] Type 'exit' or 'quit' to end the session\n")

    emit_system_message(f"Chat session started: {session_id_to_use}", "info", session_id_to_use)

    # Create orchestrator once and reuse across the session
    orchestrator = create_orchestrator()

    while True:
        try:
            # Get user input
            prompt = Prompt.ask(
                "\n[You]",
                console=console,
                default="",
                show_default=False,
            )

            if not prompt:
                continue

            # Check for exit commands
            if prompt.lower() in ["exit", "quit", "q", "x"]:
                break

            # Auto-detect complexity tier
            classification = classify_complexity(prompt)
            tier_level = int(classification.tier)

            # Process the query
            emit_task_started("cli", prompt, tier_level, session_id_to_use)

            console.print(
                f"[System] Processing (Tier {tier_level}: {classification.reasoning})...\n",
                style="dim",
            )

            result = orchestrator.execute(
                user_prompt=prompt,
                tier_level=tier_level,
                session_id=session_id_to_use,
                format="markdown",
            )

            # Display result
            console.print("\n[System] Response:", style="bold dim")
            console.print(Markdown(result.get("formatted_output", str(result))))
            console.print("")  # Blank line

            emit_task_completed(
                "cli",
                "Response delivered",
                result.get("duration_seconds", 0),
                session_id_to_use,
            )

        except KeyboardInterrupt:
            console.print("\n[SYSTEM] Interrupted by user")
            break
        except EOFError:
            break
        except Exception as e:
            console.print(f"[ERROR] {e}", style="red")
            emit_error("cli", str(e), session_id_to_use)

    typer.echo(f"\n[SYSTEM] Chat session ended: {session_id_to_use}")
    emit_system_message(f"Chat session ended: {session_id_to_use}", "info", session_id_to_use)


@app.command()
def analyze(
    task: str = typer.Argument(..., help="Task to analyze"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed analysis"),
) -> None:
    """
    Analyze a task without executing it.

    Shows how the system would approach the task, which agents would be
    involved, and provides estimates for time and cost.
    """
    # Classify complexity
    classification = classify_complexity(task)

    typer.echo(f"\n{'='*60}")
    typer.echo(f"Task Analysis")
    typer.echo(f"{'='*60}\n")

    typer.echo(f"Task: {task[:100]}...")
    typer.echo(f"")
    typer.echo(f"Complexity Classification:")
    typer.echo(f"  Tier: {classification.tier} - {classification.tier.name}")
    typer.echo(f"  Reasoning: {classification.reasoning}")
    typer.echo(f"")
    typer.echo(f"Estimated Execution:")
    typer.echo(f"  Agents: {classification.estimated_agents}")
    typer.echo(f"  Requires Council: {classification.requires_council}")
    typer.echo(f"  Requires SMEs: {classification.requires_smes}")
    typer.echo(f"  Confidence: {classification.confidence:.0%}")
    typer.echo(f"")

    # Suggest ensemble
    ensemble = suggest_ensemble(task)
    if ensemble:
        typer.echo(f"Suggested Ensemble: {ensemble.name}")
        typer.echo(f"  Description: {ensemble.description}")
        typer.echo(f"")

    # Show active agents for tier
    from src.core.complexity import get_active_agents
    agents = get_active_agents(classification.tier)

    typer.echo(f"Active Agents for Tier {classification.tier}:")
    for agent in agents:
        typer.echo(f"  - {agent}")
    typer.echo(f"")

    # Show cost estimate
    from src.tools import cost_estimate

    # Estimate turns per agent (rough estimate)
    agent_estimates = []
    for agent in agents[:5]:  # Limit to 5 for estimate
        agent_estimates.append((agent, 10))  # Assume 10 turns

    cost_result = cost_estimate(agent_estimates, classification.tier)

    typer.echo(f"Cost Estimate:")
    typer.echo(f"  Model: {cost_result['model']}")
    typer.echo(f"  Total tokens: {cost_result['total_tokens']:,}")
    typer.echo(f"  Estimated cost: ${cost_result['total_cost_usd']:.2f} USD")
    typer.echo(f"")

    # Show SME requirements if applicable
    if classification.requires_smes:
        from src.core.sme_registry import find_personas_by_keywords
        smes = find_personas_by_keywords(task.split()[:5])

        if smes:
            typer.echo(f"Suggested SMEs:")
            for sme in smes[:3]:
                typer.echo(f"  - {sme.name}: {sme.domain}")
            typer.echo(f"")


@app.command()
def tools(
    category: Optional[str] = typer.Option(None, help="Filter by category"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show full details"),
) -> None:
    """
    List available MCP tools.

    Shows all custom tools available for agent operations.
    """
    from src.tools import get_all_tools

    tools = get_all_tools()

    # Group by category
    from src.tools import ToolCategory
    by_category: dict = {}
    for name, meta in tools.items():
        cat = meta.category.value
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(name)

    typer.echo(f"\n{'='*60}")
    typer.echo("Available MCP Tools")
    typer.echo(f"{'='*60}\n")

    if category and category in by_category:
        # Show specific category
        cat_tools = by_category[category]
        typer.echo(f"\n{category.upper()} ({len(cat_tools)} tools):\n")

        for tool_name in cat_tools:
            meta = tools[tool_name]
            typer.echo(f"  {tool_name}")
            typer.echo(f"    Description: {meta.description}")
            if verbose:
                typer.echo(f"    Parameters:")
                for param, desc in meta.parameters.items():
                    typer.echo(f"      {param}: {desc}")
                typer.echo(f"    Returns: {meta.return_type}")
                if meta.examples:
                    typer.echo(f"    Examples:")
                    for example in meta.examples[:2]:
                        typer.echo(f"      {example}")
            typer.echo("")
    else:
        # Show summary by category
        for cat, tool_names in by_category.items():
            typer.echo(f"\n{cat.upper()} ({len(tool_names)}):")
            typer.echo(f"  {', '.join(tool_names)}")

    if not category and verbose:
        typer.echo("\nUse --category to filter or --verbose for details.")


@app.command()
def knowledge(
    action: str = typer.Argument("list", help="Action: list, search, retrieve"),
    query: Optional[str] = typer.Option(None, help="Search query for 'retrieve' action"),
    limit: int = typer.Option(5, help="Limit results"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show full details"),
) -> None:
    """
    Interact with the knowledge base.

    List entries, search for specific topics, or retrieve relevant knowledge.
    """
    from src.agents.memory_curator import MemoryCuratorAgent

    curator = MemoryCuratorAgent()

    if action == "list":
        entries = curator.list_knowledge()
        typer.echo(f"\n{'='*60}")
        typer.echo("Knowledge Base Entries")
        typer.echo(f"{'='*60}\n")
        typer.echo(f"Total entries: {len(entries)}\n")

        for entry in entries[:20]:  # Limit to 20
            typer.echo(f"  📄 {entry.get('topic', 'Unknown')}")
            typer.echo(f"     Category: {entry.get('category', 'N/A')}")
            typer.echo(f"     Date: {entry.get('date', 'N/A')}")
            typer.echo(f"     Tags: {', '.join(entry.get('tags', []))}")
            typer.echo("")

        if len(entries) > 20:
            typer.echo(f"... and {len(entries) - 20} more entries")

    elif action == "retrieve" or action == "search":
        if not query:
            typer.echo("Error: --query required for retrieve/search action")
            raise typer.Exit(1)

        if action == "retrieve" or action == "search":
            results = curator.retrieve_knowledge(query, limit=limit)
            typer.echo(f"\n{'='*60}")
            typer.echo(f"Knowledge Search: '{query}'")
            typer.echo(f"{'='*60}\n")
            typer.echo(f"Found {len(results)} results:\n")

        for i, result in enumerate(results, 1):
            typer.echo(f"{i}. 📄 {result.get('topic', 'Unknown')}")
            typer.echo(f"   Score: {result.get('score', 0):.2f}")
            typer.echo(f"   Summary: {result.get('summary', 'N/A')[:80]}...")
            typer.echo("")

    else:
        typer.echo(f"Unknown action: {action}")
        typer.echo("Valid actions: list, retrieve, search")


@app.command()
def personas(
    action: str = typer.Argument("list", help="Action: list, query"),
    query: Optional[str] = typer.Option(None, help="Search query"),
) -> None:
    """
    Browse SME personas.

    List all available SME personas or search for specific domains.
    """
    from src.core.sme_registry import (
        get_all_personas,
        find_personas_by_keywords,
        get_persona_for_display,
    )

    if action == "list":
        personas = get_all_personas()
        typer.echo(f"\n{'='*60}")
        typer.echo("SME Personas")
        typer.echo(f"{'='*60}\n")
        typer.echo(f"Total personas: {len(personas)}\n")

        # Group by domain
        from collections import defaultdict
        by_domain = defaultdict(list)
        for persona in personas.values():
            by_domain[persona.domain].append(persona)

        for domain, domain_personas in sorted(by_domain.items()):
            typer.echo(f"\n📚 {domain}:")
            for persona in domain_personas:
                typer.echo(f"  • {persona.name}")
                typer.echo(f"      ID: {persona.persona_id}")
                typer.echo(f"      Skills: {', '.join(persona.skill_files)}")
                typer.echo(f"      Modes: {', '.join(m.value for m in persona.interaction_modes)}")
        typer.echo("")

    elif action == "query":
        if not query:
            typer.echo("Error: --query required for query action")
            raise typer.Exit(1)

        personas = find_personas_by_keywords(query.split())
        typer.echo(f"\n{'='*60}")
        typer.echo(f"SME Search: '{query}'")
        typer.echo(f"{'='*60}\n")
        typer.echo(f"Found {len(personas)} matching personas:\n")

        for i, persona in enumerate(personas, 1):
            typer.echo(f"{i}. {persona.name}")
            typer.echo(f"   Domain: {persona.domain}")
            typer.echo(f"   Keywords: {', '.join(persona.trigger_keywords[:5])}...")
            typer.echo("")


@app.command()
def cost(
    agents: List[str] = typer.Option(..., help="Agent names"),
    turns: List[int] = typer.Option(..., help="Turns per agent"),
    tier: int = typer.Option(2, help="Tier level for cost estimation"),
) -> None:
    """
    Estimate cost for agent operations.

    Calculate estimated token usage and cost for specified agents and turns.
    """
    if len(agents) != len(turns):
        typer.echo("Error: Must provide equal number of --agents and --turns")
        raise typer.Exit(1)

    # Build agent estimates
    agent_estimates = list(zip(agents, turns))

    # Get cost estimate
    from src.tools import cost_estimate
    result = cost_estimate(agent_estimates, tier)

    typer.echo(f"\n{'='*60}")
    typer.echo("Cost Estimate")
    typer.echo(f"{'='*60}\n")

    typer.echo(f"Configuration:")
    typer.echo(f"  Tier: {tier}")
    typer.echo(f"  Model: {result['model']}")
    typer.echo(f"  Total tokens: {result['total_tokens']:,}")
    typer.echo(f"")

    typer.echo(f"Cost Breakdown:\n")
    for item in result["agent_breakdown"]:
        typer.echo(f"  {item['agent']}:")
        typer.echo(f"    Turns: {item['turns']}")
        typer.echo(f"    Input tokens: {item['input_tokens']:,}")
        typer.echo(f"    Output tokens: {item['output_tokens']:,}")
        typer.echo(f"    Cost: ${item['cost_usd']:.4f}")
        typer.echo("")

    typer.echo(f"Total estimated cost: ${result['total_cost_usd']:.2f} USD")


@app.command()
def ensembles(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show details"),
) -> None:
    """
    List available ensemble patterns.

    Shows pre-configured agent workflows for common tasks.
    """
    from src.core.ensemble import (
        get_all_ensembles,
        ENSEMBLE_REGISTRY,
    )

    ensembles = get_all_ensembles()

    typer.echo(f"\n{'='*60}")
    typer.echo("Ensemble Patterns")
    typer.echo(f"{'='*60}\n")
    typer.echo(f"Total patterns: {len(ensembles)}\n")

    for ensemble_type, ensemble in ensembles.items():
        typer.echo(f"📋 {ensemble.name}")
        typer.echo(f"   Type: {ensemble_type.value}")
        typer.echo(f"   Description: {ensemble.description}")

        if verbose:
            config = ensemble.get_config()
            typer.echo(f"   Tier: {config.tier_level}")
            typer.echo(f"   Required SMEs: {', '.join(config.required_smes)}")
            typer.echo(f"   Quality Gates: {', '.join(config.quality_gates)}")
            typer.echo(f"   Success Criteria:")
            for criteria in config.success_criteria:
                typer.echo(f"      • {criteria}")

        typer.echo("")


# =============================================================================
# Utility Commands
# =============================================================================

@app.command()
def version() -> None:
    """Show version information."""
    try:
        with open("pyproject.toml") as f:
            import re
            content = f.read()
            match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', content)
            if match:
                typer.echo(f"Multi-Agent Reasoning System v{match.group(1)}")
            else:
                typer.echo("Multi-Agent Reasoning System (development version)")
    except FileNotFoundError:
        typer.echo("Multi-Agent Reasoning System (development version)")

    # Show Python version
    typer.echo(f"\nPython: {sys.version.split()[0]}")
    typer.echo(f"Platform: {sys.platform}")


@app.command()
def status() -> None:
    """Show system status and configuration."""
    from src.tools import system_get_status

    status = system_get_status()

    typer.echo(f"\n{'='*60}")
    typer.echo("System Status")
    typer.echo(f"{'='*60}\n")
    typer.echo(f"Timestamp: {status['timestamp']}\n")

    typer.echo(f"Agents: {len(status['tier1_agents']) + len(status['tier2_agents'])}")
    typer.echo(f"  Tier 1 (Direct): {', '.join(status['tier1_agents'])}")
    typer.echo(f"  Tier 2 (Standard): {', '.join(status['tier2_agents'])}")
    typer.echo(f"  Tier 3 (Deep): {len(status['tier3_agents'])} agents")
    typer.echo(f"  Tier 4 (Adversarial): {len(status['tier4_agents'])} agents\n")

    typer.echo(f"Council Agents:")
    for agent in status["council_agents"]:
        typer.echo(f"  • {agent}")
    typer.echo("")

    typer.echo(f"SME Personas: {status['sme_personas']['total_personas']}")
    for persona_id in status["sme_personas"]["persona_ids"]:
        typer.echo(f"  • {persona_id}")
    typer.echo("")

    typer.echo(f"Ensemble Patterns: {len(status['ensemble_patterns'])}")
    for pattern in status["ensemble_patterns"]:
        typer.echo(f"  • {pattern.value}")
    typer.echo("")

    typer.echo(f"MCP Tools: {len(status['mcp_tools'])}")
    for tool in status["mcp_tools"]:
        typer.echo(f"  • {tool}")


@app.command()
def sessions(
    action: str = typer.Argument("list", help="Action: list, show, delete, compact"),
    session_id: Optional[str] = typer.Option(None, help="Session ID for show/delete/compact"),
    limit: int = typer.Option(20, help="Limit results for list"),
    sort_by: str = typer.Option("updated_at", help="Sort field: created_at, updated_at, title"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show full details"),
) -> None:
    """
    Manage sessions.

    List, show details, delete, or compact sessions.
    """
    from src.session import (
        SessionPersistence,
        get_context_compactor,
        CompactionTrigger,
    )

    persistence = SessionPersistence()

    if action == "list":
        summaries = persistence.list_sessions(limit=limit, sort_by=sort_by)

        typer.echo(f"\n{'='*60}")
        typer.echo("Sessions")
        typer.echo(f"{'='*60}\n")
        typer.echo(f"Total sessions: {len(summaries)}\n")

        for i, summary in enumerate(summaries, 1):
            typer.echo(f"{i}. {summary.session_id}")
            typer.echo(f"   Created: {summary.created_at.strftime('%Y-%m-%d %H:%M')}")
            typer.echo(f"   Updated: {summary.updated_at.strftime('%Y-%m-%d %H:%M')}")
            typer.echo(f"   Tier: {summary.tier}")
            typer.echo(f"   Messages: {summary.total_messages}")
            typer.echo(f"   Outputs: {summary.total_agent_outputs}")
            typer.echo(f"   Tokens: {summary.total_tokens:,}")
            typer.echo(f"   Cost: ${summary.total_cost_usd:.4f}")

            if summary.title:
                typer.echo(f"   Title: {summary.title}")
            if verbose and summary.description:
                typer.echo(f"   Description: {summary.description[:80]}...")
            typer.echo("")

    elif action == "show":
        if not session_id:
            typer.echo("Error: --session-id required for show action")
            raise typer.Exit(1)

        from src.session import resume_session
        session = resume_session(session_id)

        if not session:
            typer.echo(f"Session not found: {session_id}")
            raise typer.Exit(1)

        typer.echo(f"\n{'='*60}")
        typer.echo(f"Session: {session_id}")
        typer.echo(f"{'='*60}\n")

        typer.echo(f"Created: {session.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        typer.echo(f"Updated: {session.updated_at.strftime('%Y-%m-%d %H:%M:%S')}")
        typer.echo(f"Tier: {session.tier}")
        typer.echo(f"Max Budget: ${session.max_budget_usd:.2f}")
        typer.echo(f"")
        typer.echo(f"Messages: {len(session.messages)}")
        typer.echo(f"Agent Outputs: {len(session.agent_outputs)}")
        typer.echo(f"Total Tokens: {session.total_tokens:,}")
        typer.echo(f"Total Cost: ${session.total_cost_usd:.4f}")
        typer.echo(f"")

        if session.title:
            typer.echo(f"Title: {session.title}")
        if session.description:
            typer.echo(f"Description: {session.description}")
        if session.tags:
            typer.echo(f"Tags: {', '.join(session.tags)}")
        typer.echo("")

        if session.active_agents:
            typer.echo(f"Active Agents: {', '.join(session.active_agents)}")
        if session.current_phase:
            typer.echo(f"Current Phase: {session.current_phase}")

        if verbose:
            typer.echo(f"\n--- Messages ({len(session.messages)}) ---")
            for i, msg in enumerate(session.messages[-10:], 1):  # Last 10
                role = msg.role.upper()
                preview = msg.content[:100].replace("\n", " ")
                typer.echo(f"{i}. [{role}] {preview}...")

            typer.echo(f"\n--- Agent Outputs ({len(session.agent_outputs)}) ---")
            for i, output in enumerate(session.agent_outputs[-10:], 1):  # Last 10
                typer.echo(f"{i}. [{output.agent_name}] {output.phase} - {output.status}")

    elif action == "delete":
        if not session_id:
            typer.echo("Error: --session-id required for delete action")
            raise typer.Exit(1)

        if persistence.delete_session(session_id):
            typer.echo(f"Session deleted: {session_id}")
        else:
            typer.echo(f"Session not found: {session_id}")
            raise typer.Exit(1)

    elif action == "compact":
        if not session_id:
            typer.echo("Error: --session-id required for compact action")
            raise typer.Exit(1)

        from src.session import compact_session_manual

        result = compact_session_manual(session_id)

        if result:
            typer.echo(f"\n{'='*60}")
            typer.echo(f"Session Compacted: {session_id}")
            typer.echo(f"{'='*60}\n")

            typer.echo(f"Messages: {result.original_count} → {result.compacted_count}")
            typer.echo(f"Tokens Removed: {result.tokens_removed:,}")
            typer.echo(f"Tokens Remaining: {result.tokens_remaining:,}")
            typer.echo(f"Reduction: {result.reduction_ratio*100:.1f}%")

            if verbose:
                typer.echo(f"\nSummary:\n{result.summary}")
                typer.echo(f"\nPreserved Items:")
                for item in result.preserved_items:
                    typer.echo(f"  - {item}")
        else:
            typer.echo(f"Session not found or compaction not needed: {session_id}")
            raise typer.Exit(1)

    else:
        typer.echo(f"Unknown action: {action}")
        typer.echo("Valid actions: list, show, delete, compact")
        raise typer.Exit(1)


@app.command()
def test(
    test_name: str = typer.Argument(..., help="Test to run"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show test output"),
) -> None:
    """
    Run system tests.

    Execute unit or integration tests to verify system functionality.
    """
    import subprocess

    typer.echo(f"Running test: {test_name}")

    # Build pytest command
    cmd = [
        sys.executable, "-m", "pytest",
        f"tests/{test_name}" if test_name != "all" else "tests/",
        "-v" if verbose else "-q",
    ]

    if verbose:
        typer.echo(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=False)

    typer.echo(f"\nTest completed with exit code: {result.returncode}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
