# Multi-Agent Reasoning System - Global Configuration

This is the root configuration file for the Multi-Agent Reasoning System. It provides global instructions that apply to all agents in the system.

## System Overview

This is a **Multi-Agent Reasoning System** built with the Claude Agent SDK. It implements a three-tier architecture:

1. **Strategic Council** (3 agents): Governance, SME selection, quality arbitration, ethics review
2. **Operational Agents** (12 agents): Core execution - Analyst, Planner, Clarifier, Researcher, Executor, Code Reviewer, Formatter, Verifier, Critic, Reviewer, Memory Curator
3. **Dynamic SME Personas** (10+ on-demand): Domain experts spawned by the Council

## Architecture

```
User (CLI/Streamlit) → Orchestrator → [Council/SMEs if Tier 3-4] → Operational Agents → Output
```

## Key Principles

1. **Complexity-Based Routing**: Tasks are classified into 4 tiers (Direct/Standard/Deep/Adversarial)
2. **Agent Isolation**: Each subagent has isolated context via the Task tool
3. **Structured Output**: All agents return JSON via Pydantic schemas
4. **Quality Assurance**: Verifier + Critic + Reviewer + optional Council arbitration
5. **Cost Control**: Budget enforcement with configurable per-agent limits

## Global Instructions

### For All Agents

1. **Always return structured output** using your assigned Pydantic schema
2. **Use allowed tools only** - do not attempt to use tools outside your allowedTools list
3. **Stay in character** - each agent has a specific role and perspective
4. **Escalate when needed** - if you encounter something beyond your capability, set `escalation_needed: true`
5. **Be concise but thorough** - provide enough context without unnecessary verbosity

### For Subagents (Operational, Council, SME)

1. **Do NOT spawn your own subagents** - only the Orchestrator can spawn agents
2. **Return results to Orchestrator** - all communication flows through the parent
3. **Use assigned skills** - invoke the Skill tool to load assigned SKILL.md files
4. **Validate your output** - ensure it matches your assigned schema before returning

### For the Orchestrator (Parent Agent)

1. **Classify complexity FIRST** - before spawning any subagents
2. **Consult Council on Tier 3-4** - spawn Domain Council Chair before operational agents
3. **Aggregate all results** - synthesize subagent outputs into unified user-facing response
4. **Handle failures gracefully** - retry non-critical agents once, critical agents twice
5. **Enforce budget** - stop if max_budget_usd is reached

## Tier Classification

| Tier | Description | Active Agents | Council | SMEs |
|------|-------------|---------------|---------|------|
| 1 (Direct) | Simple, straightforward | 3 | No | No |
| 2 (Standard) | Moderate complexity | 7 | No | No |
| 3 (Deep) | Complex, domain-specific | 10-15 | Chair only | 1-3 |
| 4 (Adversarial) | High stakes, sensitive | 13-18 | Full Council | 1-3 |

## Skill System

- Skills are defined in `.claude/skills/{name}/SKILL.md`
- Auto-discovered via `setting_sources=["user", "project"]`
- Agents invoke skills via the Skill tool
- SME personas load skills based on their registry configuration

## Configuration Files

- `config/agents/{agent}/CLAUDE.md` - Per-agent system prompts
- `config/sme/{persona}.md` - SME persona system prompt templates
- `.env` - Environment variables (API keys, model settings, budgets)

## Quick Reference

- **Project root**: Repository root (where this CLAUDE.md resides)
- **CLI entry**: `src/cli/main.py`
- **Streamlit app**: `src/ui/app.py`
- **Schemas**: `src/schemas/`
- **Core logic**: `src/core/`

## Context Compaction & Re-orientation

When context compaction occurs during long sessions:

1. **Re-read this CLAUDE.md** to restore system-wide instructions
2. **Re-read your agent's CLAUDE.md** from `config/agents/{your_agent}/CLAUDE.md`
3. **Remember your role**: You are part of a multi-agent system with isolated context
4. **Restore state**: Check the session state for tier classification, active SMEs, and budget
5. **Continue from last checkpoint**: Resume from the last completed pipeline phase

### Re-orientation Instructions (Post-Compaction)

If you notice your context has been compacted:
- You are an agent in the Multi-Agent Reasoning System
- Your output must conform to your assigned Pydantic schema
- Do NOT spawn subagents - only the Orchestrator can do that
- Return results to the Orchestrator via structured output
- Check `escalation_needed` if the task exceeds your capability

## Development Notes

When working on this system:

1. **Read the FRD**: `FRD_MultiAgent_Prototype_v4.docx` contains all 62 functional requirements
2. **Check the plan**: See implementation phases in the project plan
3. **Follow patterns**: Existing code patterns should guide new implementations
4. **Test incrementally**: Each phase should be testable before moving to the next
5. **Document changes**: Update relevant docs when making changes
