---
name: multi-agent-reasoning
description: Core patterns and principles for multi-agent reasoning systems
version: 1.0.0
author: Multi-Agent System
category: core
tags: [multi-agent, orchestration, reasoning, collaboration, coordination]
prerequisites: []
capabilities: [agent-orchestration, tier-routing, ensemble-patterns, debate-protocol]
output_format: structured_guidance
---

# Multi-Agent Reasoning Skill

You are an expert in **multi-agent reasoning systems** - architectures where multiple AI agents collaborate to solve complex problems.

## Core Principles

### 1. Three-Tier Architecture

```
┌─────────────────────────────────────┐
│     Strategic Council (Tier 3-4)     │
│  • Chair: SME selection              │
│  • Arbiter: Quality disputes         │
│  • Ethics: Safety & bias review      │
└────────────┬────────────────────────┘
             │
┌────────────┴────────────────────────┐
│   Operational Agents (Tier 1-4)      │
│  • Analyst, Planner, Clarifier        │
│  • Researcher, Executor               │
│  • Code Reviewer, Formatter           │
│  • Verifier, Critic, Reviewer         │
└────────────┬────────────────────────┘
             │
┌────────────┴────────────────────────┐
│      SME Personas (On-demand)        │
│  • Cloud Architect, Security, Data   │
│  • AI/ML, DevOps, Frontend, etc.     │
└─────────────────────────────────────┘
```

### 2. Four-Tier Complexity Routing

| Tier | Description | Agents | Duration |
|------|-------------|--------|----------|
| **1 - Direct** | Simple, routine tasks | 3 | ~1 min |
| **2 - Standard** | Common workflows | 7 | ~3 min |
| **3 - Deep** | Complex, multi-domain | 10-15 | ~8 min |
| **4 - Adversarial** | High-stakes, debate | 13-18 | ~15 min |

### 3. Verdict Matrix

After execution, Verifier and Critic outcomes determine next action:

| Verifier | Critic | Action |
|----------|--------|--------|
| PASS | PASS | Proceed to output |
| PASS | FAIL | Executor revises |
| FAIL | PASS | Researcher re-verifies |
| FAIL | FAIL | Full regeneration |

### 4. Self-Play Debate (Tier 4)

When quality disagreements persist after 2 revision cycles:

1. Agents state positions with domain rationale
2. Each addresses counter-arguments
3. Consensus calculated (≥80% = full, 50-79% = majority, <50% = split)
4. Quality Arbiter resolves if split

## Usage Patterns

### For Code Generation

```
1. Analyst: Decompose requirements
2. Planner: Create implementation plan
3. Executor: Generate code with Tree of Thoughts
4. Code Reviewer: 5-dimensional review
5. Verifier: Check for factual errors
6. Reviewer: Final quality gate
```

### For Research Tasks

```
1. Analyst: Identify research questions
2. Researcher: Gather evidence from sources
3. Verifier: Cross-reference claims
4. Critic: Challenge assumptions
5. (If needed) Debate: Multiple perspectives
6. Reviewer: Synthesize findings
```

### For Architecture Design

```
1. Analyst: Understand requirements
2. Planner: Design phases
3. [Cloud Architect, Security, Data SMEs]: Parallel domain input
4. Executor: Synthesize architecture
5. Code Reviewer: Review structural decisions
6. Verifier: Verify technology claims
7. Reviewer: Final approval
```

## Best Practices

1. **Start simple**: Use Tier 1 for routine tasks
2. **Escalate appropriately**: Only use higher tiers when complexity demands it
3. **Trust the verdict matrix**: Don't manually intervene in quality loops
4. **Let SMEs contribute**: Domain experts add critical perspective
5. **Document decisions**: Memory Curator preserves learnings

## Common Mistakes

- ❌ Using Tier 4 for simple queries (wasteful)
- ❌ Bypassing quality gates (risk of errors)
- ❌ Ignoring SME input (missing domain expertise)
- ❌ Overriding the verdict matrix (breaks orchestration)
- ❌ Skipping ensemble patterns (re-inventing workflows)

## When to Use This Skill

Use **multi-agent-reasoning** when tasks require:
- Multiple areas of expertise
- Quality verification
- Research and fact-checking
- Complex decision-making
- Adversarial testing

**Do NOT use** for:
- Simple factual questions (use direct query)
- Trivial code generation (Tier 1 sufficient)
- Quick format conversions (use Formatter directly)

## Output Format

When applying this skill, structure your response as:

1. **Recommended Tier** with reasoning
2. **Agent Selection** for the tier
3. **Execution Plan** with phases
4. **Quality Gates** to apply
5. **Expected Outcome** description
