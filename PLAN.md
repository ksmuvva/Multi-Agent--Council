# Full ReAct Conversion Plan - Multi-Agent Council System

## Problem Statement

All 13 agents are **procedural Python** with hardcoded regex/keyword heuristics. None make LLM API calls. The orchestrator's `_spawn_agent()` calls `spawn_subagent()` which sends the task to an LLM via the SDK, but the **agent classes themselves** (AnalystAgent, CriticAgent, etc.) are never invoked by the orchestrator — they're standalone facades.

## Architecture Decision: Two Execution Modes

We will NOT delete the existing procedural logic. Instead, we introduce a **dual-mode** design:

```
Agent.execute(task, mode="react")   → LLM-driven ReAct loop
Agent.execute(task, mode="local")   → Current procedural logic (fallback, testing, offline)
```

This preserves backward compatibility, allows offline/testing use, and enables incremental rollout.

## Implementation Plan

### Phase 1: ReAct Base Infrastructure (`src/core/react.py`)

Create a reusable ReAct loop engine that all agents will share.

**New file: `src/core/react.py`**

```python
class ReactLoop:
    """
    Generic ReAct (Reasoning + Acting) execution loop.

    Cycle: Thought → Action → Observation → Thought → ... → Final Answer
    """
    def __init__(self, agent_name, system_prompt, tools, output_schema,
                 model, max_iterations=10, max_turns=30):
        ...

    def run(self, task_input: str, context: dict = None) -> dict:
        """
        Execute the ReAct loop:
        1. Send system_prompt + task to LLM
        2. Parse response for Thought/Action/Final Answer
        3. If Action → execute tool → feed Observation back
        4. If Final Answer → validate against output_schema → return
        5. Repeat until max_iterations or Final Answer
        """
        ...
```

**Key design choices:**
- Uses `spawn_subagent()` from existing `sdk_integration.py` for LLM calls (supports Claude SDK, Anthropic API, GLM fallback)
- Tool execution via a `ToolExecutor` class that wraps real tool calls
- Structured output enforcement: the final answer MUST parse into the agent's Pydantic schema
- Budget-aware: tracks tokens/cost per iteration, can early-stop
- Full logging integration: logs each Thought, Action, Observation via existing `structlog` + event system
- Timeout protection via `max_iterations`

### Phase 2: Tool Executor (`src/core/tool_executor.py`)

Create a tool execution layer that agents can use within their ReAct loops.

**New file: `src/core/tool_executor.py`**

```python
class ToolExecutor:
    """Execute tools available to agents in the ReAct loop."""

    def __init__(self, allowed_tools: list[str]):
        self.allowed_tools = set(allowed_tools)

    def execute(self, tool_name: str, tool_input: dict) -> str:
        """
        Execute a tool and return the observation string.

        Supported tools:
        - WebSearch: search the web
        - WebFetch: fetch URL content
        - Read: read a file
        - Write: write a file
        - Glob: find files by pattern
        - Grep: search file contents
        - Bash: run shell command
        """
        ...
```

This maps tool names from `AGENT_ALLOWED_TOOLS` in `sdk_integration.py` to actual implementations. The tools already exist in the SDK — we just need callable wrappers.

### Phase 3: Convert Each Agent (13 agents)

For each agent, we add a `_react_execute()` method alongside the existing procedural method. The main entry point (`analyze()`, `execute()`, `verify()`, etc.) gains a `mode` parameter.

#### 3.1: Analyst Agent (`src/agents/analyst.py`)
- **ReAct tools**: `Read`, `Glob`, `Grep`
- **System prompt**: "You are the Analyst. Examine the user's request. Use tools to inspect files if referenced. Reason about intent, modality, complexity tier, and sub-tasks. Return a TaskIntelligenceReport JSON."
- **What LLM replaces**: `_detect_modality()`, `_infer_intent()`, `_decompose_tasks()`, `_suggest_tier()` — all become LLM reasoning
- **What stays algorithmic**: Nothing — all should be LLM-driven

#### 3.2: Clarifier Agent (`src/agents/clarifier.py`)
- **ReAct tools**: None (pure reasoning)
- **System prompt**: "You are the Clarifier. Given an analyst report, identify ambiguities and formulate clarifying questions ranked by impact. Return a ClarificationRequest JSON."
- **What LLM replaces**: Question generation, severity ranking, answer option suggestions
- **What stays algorithmic**: Priority weight formula (optional)

#### 3.3: Planner Agent (`src/agents/planner.py`)
- **ReAct tools**: `Read`, `Glob`
- **System prompt**: "You are the Planner. Create a step-by-step execution plan with agent assignments, parallelization groups, critical path, SME requirements, and risk factors. Return an ExecutionPlan JSON."
- **What LLM replaces**: Agent assignment, duration estimation, risk identification, SME selection
- **What stays algorithmic**: Critical path graph calculation (utility function the LLM can call)

#### 3.4: Researcher Agent (`src/agents/researcher.py`)
- **ReAct tools**: `WebSearch`, `WebFetch`, `Read`
- **System prompt**: "You are the Researcher. Search for evidence, fetch authoritative sources, cross-reference claims, and identify conflicts/gaps. Return an EvidenceBrief JSON."
- **What LLM replaces**: Query generation, source evaluation, finding synthesis, conflict detection
- **What stays algorithmic**: Jaccard similarity (utility), confidence scoring formula
- **NOTE**: This agent benefits MOST from ReAct — it already has WebSearch/WebFetch tools configured

#### 3.5: Executor Agent (`src/agents/executor.py`)
- **ReAct tools**: `Read`, `Write`, `Edit`, `Bash`, `Glob`, `Grep`, `Skill`
- **System prompt**: "You are the Executor. Use Tree of Thoughts to generate multiple approaches, evaluate each, select the best, and implement the solution. Use tools to read context, write files, run commands. Return your solution."
- **What LLM replaces**: Problem decomposition, approach generation, scoring, code/document generation (the entire 1900-line template system)
- **What stays algorithmic**: Scoring weight matrix (as reference for LLM), output validation checks
- **NOTE**: This is the highest-impact conversion. The ~1400 lines of code generation templates become a single LLM call with tools.

#### 3.6: Code Reviewer Agent (`src/agents/code_reviewer.py`)
- **ReAct tools**: `Read`, `Glob`, `Grep`, `Bash`
- **System prompt**: "You are the Code Reviewer. Analyze code for security vulnerabilities, performance issues, style compliance, error handling, and test coverage. Use tools to read source files and run linters. Return a CodeReviewReport JSON."
- **What LLM replaces**: Security pattern detection, performance analysis, style checking
- **What stays algorithmic**: None — all pattern detection becomes LLM reasoning

#### 3.7: Formatter Agent (`src/agents/formatter.py`)
- **ReAct tools**: `Read`, `Write`, `Bash`, `Skill`
- **System prompt**: "You are the Formatter. Transform raw content into the requested output format (markdown, code, JSON, YAML, etc.). Use Write to create output files. Return the formatted content."
- **What LLM replaces**: Format detection, template selection, content transformation
- **What stays algorithmic**: Binary format conversion (DOCX/PDF/XLSX via python-docx/etc.)

#### 3.8: Verifier Agent (`src/agents/verifier.py`)
- **ReAct tools**: `Read`, `WebSearch`, `WebFetch`
- **System prompt**: "You are the Verifier (Hallucination Guard). Extract every factual claim, verify each against sources using WebSearch, score confidence, and flag fabrication risks. Return a VerificationReport JSON."
- **What LLM replaces**: Claim extraction (currently regex), claim verification logic, hallucination pattern detection
- **What stays algorithmic**: Reliability threshold (0.7), PASS/FAIL verdict formula

#### 3.9: Critic Agent (`src/agents/critic.py`)
- **ReAct tools**: `Read`, `Grep`
- **System prompt**: "You are the Critic (Devil's Advocate). Apply 5 attack vectors: Logic, Completeness, Quality, Contradiction Scan, Red Team. Challenge every assumption. Return a CritiqueReport JSON."
- **What LLM replaces**: All 5 attack vector implementations, fallacy detection, red team argumentation
- **What stays algorithmic**: None — adversarial reasoning is ideal for LLM

#### 3.10: Reviewer Agent (`src/agents/reviewer.py`)
- **ReAct tools**: `Read`, `Glob`, `Grep`
- **System prompt**: "You are the Reviewer (Quality Gate). Evaluate output against 6 quality gates: completeness, consistency, verifier sign-off, critic findings, readability, code review. Apply the verdict matrix. Return a ReviewVerdict JSON."
- **What LLM replaces**: Completeness checking, consistency analysis, readability assessment
- **What stays algorithmic**: Verdict matrix lookup (Verifier×Critic → Action)

#### 3.11: Memory Curator Agent (`src/agents/memory_curator.py`)
- **ReAct tools**: `Read`, `Write`, `Glob`
- **System prompt**: "You are the Memory Curator. Extract key decisions, patterns, domain insights, and lessons learned from the session. Write knowledge files to docs/knowledge/. Return an ExtractionResult JSON."
- **What LLM replaces**: Decision extraction, pattern identification, insight capture
- **What stays algorithmic**: YAML frontmatter generation, file naming

#### 3.12: Council Chair Agent (`src/agents/council.py`)
- **ReAct tools**: None (pure reasoning)
- **System prompt**: "You are the Domain Council Chair. Analyze the task to identify required domains, select appropriate SME personas from the registry, determine interaction modes, and create a collaboration plan. Return an SMESelectionReport JSON."
- **What LLM replaces**: Domain identification, SME selection logic, interaction mode determination
- **What stays algorithmic**: SME registry lookups (utility function)

#### 3.13: SME Spawner Agent (`src/agents/sme_spawner.py`)
- **ReAct tools**: `Read`, `Glob`, `Grep`, `Skill`
- **System prompt**: Dynamic per-persona from `config/sme/{persona}.md`
- **What LLM replaces**: Domain analysis, content generation, adversarial argumentation
- **What stays algorithmic**: Persona registry lookup, skill file loading

### Phase 4: Wire ReAct Mode into Orchestrator

Modify `src/agents/orchestrator.py`:

1. **`_spawn_agent()`** gains a `use_react` parameter (default: `True`)
2. When `use_react=True`:
   - Instantiate the agent class (e.g., `AnalystAgent`)
   - Call its `analyze(task, mode="react")` which uses `ReactLoop`
   - The ReactLoop internally calls `spawn_subagent()` for LLM + executes tools locally
3. When `use_react=False` (fallback):
   - Use current behavior: call `spawn_subagent()` directly with system prompt
4. **Configuration**: Add `MAS_REACT_MODE=true|false` env var

### Phase 5: Logging Integration

The ReAct loop emits events at each step:

```
[THOUGHT]      agent=researcher, iteration=1, thought="I need to search for..."
[ACTION]       agent=researcher, iteration=1, tool=WebSearch, input={query: "..."}
[OBSERVATION]  agent=researcher, iteration=1, tool=WebSearch, result_length=1234
[THOUGHT]      agent=researcher, iteration=2, thought="The search results show..."
[FINAL_ANSWER] agent=researcher, iteration=2, schema=EvidenceBrief, valid=true
```

Uses existing `structlog` + `EventEmitter` infrastructure. New event types:
- `EventType.REACT_THOUGHT`
- `EventType.REACT_ACTION`
- `EventType.REACT_OBSERVATION`

These display in both CLI (verbose mode) and Streamlit (agent panel).

### Phase 6: Testing

- Unit tests for `ReactLoop` with mocked LLM responses
- Integration test: one full pipeline run in react mode
- Fallback test: verify `mode="local"` still works identically

## File Changes Summary

| Action | File | Description |
|--------|------|-------------|
| CREATE | `src/core/react.py` | ReAct loop engine |
| CREATE | `src/core/tool_executor.py` | Tool execution layer |
| MODIFY | `src/agents/analyst.py` | Add `_react_analyze()` + mode param |
| MODIFY | `src/agents/clarifier.py` | Add `_react_formulate()` + mode param |
| MODIFY | `src/agents/planner.py` | Add `_react_plan()` + mode param |
| MODIFY | `src/agents/researcher.py` | Add `_react_research()` + mode param |
| MODIFY | `src/agents/executor.py` | Add `_react_execute()` + mode param |
| MODIFY | `src/agents/code_reviewer.py` | Add `_react_review()` + mode param |
| MODIFY | `src/agents/formatter.py` | Add `_react_format()` + mode param |
| MODIFY | `src/agents/verifier.py` | Add `_react_verify()` + mode param |
| MODIFY | `src/agents/critic.py` | Add `_react_critique()` + mode param |
| MODIFY | `src/agents/reviewer.py` | Add `_react_review()` + mode param |
| MODIFY | `src/agents/memory_curator.py` | Add `_react_extract()` + mode param |
| MODIFY | `src/agents/council.py` | Add `_react_select()` + mode param |
| MODIFY | `src/agents/sme_spawner.py` | Add `_react_spawn()` + mode param |
| MODIFY | `src/agents/orchestrator.py` | Wire `use_react` parameter |
| MODIFY | `src/utils/events.py` | Add REACT_THOUGHT/ACTION/OBSERVATION event types |
| MODIFY | `src/cli/main.py` | Display ReAct events in verbose mode |
| MODIFY | `src/config/__init__.py` | Add `MAS_REACT_MODE` setting |

## Implementation Order

1. `src/core/react.py` + `src/core/tool_executor.py` (foundation)
2. `src/utils/events.py` + `src/config/__init__.py` (infrastructure)
3. `src/agents/researcher.py` (highest value — uses WebSearch/WebFetch)
4. `src/agents/executor.py` (second highest — replaces 1400 lines of templates)
5. `src/agents/verifier.py` (fact-checking with WebSearch)
6. `src/agents/critic.py` (adversarial reasoning)
7. `src/agents/analyst.py`, `src/agents/planner.py`, `src/agents/clarifier.py`
8. `src/agents/code_reviewer.py`, `src/agents/reviewer.py`
9. `src/agents/formatter.py`, `src/agents/memory_curator.py`
10. `src/agents/council.py`, `src/agents/sme_spawner.py`
11. `src/agents/orchestrator.py` (wire everything together)
12. `src/cli/main.py` (display ReAct events)
