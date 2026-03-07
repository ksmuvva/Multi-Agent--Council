# Vibe Coding Prompts

This document contains AI-ready prompts for building the Multi-Agent Reasoning System. Each prompt is self-contained and can be fed directly to Claude Code.

**FRD Version:** 4.0
**Total Requirements:** 62 (FR-001 to FR-062)

---

## Table of Contents

1. [Core Agents (FR-001 to FR-015)](#core-agents)
2. [SME System (FR-016 to FR-019)](#sme-system)
3. [SDK Configuration (FR-020 to FR-021)](#sdk-configuration)
4. [Complexity & Skills (FR-022 to FR-029)](#complexity--skills)
5. [Pipeline & Patterns (FR-030 to FR-033)](#pipeline--patterns)
6. [SDK Integration (FR-034 to FR-038)](#sdk-integration)
7. [Output & Session (FR-039 to FR-041)](#output--session)
8. [CLI & UI (FR-042 to FR-051)](#cli--ui)
9. [Infrastructure (FR-052 to FR-062)](#infrastructure)

---

## Core Agents

### FR-001: Orchestrator Agent

```
Create the Orchestrator agent in src/agents/orchestrator.py. It is the parent agent that:
1. Receives user queries via CLI or Streamlit
2. Classifies task complexity into 4 tiers (Direct/Standard/Deep/Adversarial)
3. Spawns subagents via the Claude Agent SDK Task tool based on tier
4. Aggregates results from all subagents into a unified response
5. Enforces budget limits and handles failures with retry logic (1x non-critical, 2x critical)
6. Tracks cost across all spawned subagents

Use ClaudeAgentOptions for each subagent with model, allowedTools, outputFormat, max_turns, setting_sources.
The orchestrator must never be spawned as a subagent itself.
```

### FR-002: Task Analyst Subagent

```
Create the Task Analyst agent in src/agents/analyst.py. This agent:
1. Receives the raw user query from the Orchestrator
2. Produces a TaskIntelligenceReport (Pydantic schema) containing:
   - domain_classification: which domain(s) the task falls into
   - complexity_score: 1-10 rating
   - ambiguity_flags: list of unclear aspects
   - key_entities: extracted entities from the query
   - recommended_tier: suggested complexity tier
   - sme_recommendations: which SME personas would help
3. allowedTools: Read, Glob, Grep (for codebase analysis)
4. Returns structured JSON output matching TaskIntelligenceReport schema
```

### FR-003: Planner Subagent

```
Create the Planner agent in src/agents/planner.py. This agent:
1. Receives the TaskIntelligenceReport from the Analyst
2. Produces an ExecutionPlan (Pydantic schema) containing:
   - phases: ordered list of execution phases
   - agent_assignments: which agents handle each phase
   - dependencies: phase dependency graph
   - estimated_complexity: per-phase complexity
   - risk_factors: identified risks
   - fallback_strategies: alternatives if phases fail
3. allowedTools: Read, Glob
4. Must account for the tier classification when planning
```

### FR-004: Clarifier Subagent

```
Create the Clarifier agent in src/agents/clarifier.py. This agent:
1. Analyzes the user query for ambiguities
2. Produces a ClarificationRequest (Pydantic schema) with:
   - questions: list of clarifying questions
   - assumptions: assumptions made if no clarification provided
   - default_values: sensible defaults for each ambiguity
   - confidence_score: how confident the system is without clarification
3. allowedTools: none (pure reasoning)
4. If confidence > 0.8, skip clarification and proceed with assumptions
```

### FR-005: Web Researcher Subagent

```
Create the Web Researcher agent in src/agents/researcher.py. This agent:
1. Conducts web research to gather evidence for the task
2. Produces an EvidenceBrief (Pydantic schema) with:
   - sources: list of URLs with titles and relevance scores
   - key_findings: summarized findings from research
   - contradictions: conflicting information found
   - confidence_level: overall confidence in findings
   - citations: properly formatted citations
3. allowedTools: WebSearch, WebFetch, Read
4. Must verify sources and flag potential misinformation
```

### FR-006: Solution Executor Subagent

```
Create the Executor agent in src/agents/executor.py. This agent:
1. Implements the solution based on the execution plan
2. Has the broadest tool access of any subagent
3. Produces solution artifacts (code, documents, etc.)
4. allowedTools: Read, Write, Edit, Bash, Glob, Grep, Skill
5. permission_mode: acceptEdits (for file operations)
6. Can invoke skills via the Skill tool for specialized tasks
7. Must follow the plan phases and report progress
```

### FR-007: Code Reviewer Subagent

```
Create the Code Reviewer agent in src/agents/code_reviewer.py. This agent:
1. Reviews code produced by the Executor
2. Produces a CodeReviewReport (Pydantic schema) with:
   - issues: list of issues found (severity, location, description)
   - suggestions: improvement suggestions
   - security_concerns: security-related findings
   - quality_score: overall code quality score (1-10)
   - approved: boolean approval decision
3. allowedTools: Read, Glob, Grep, Bash (for running linters/tests)
4. Must check for OWASP top 10 vulnerabilities
```

### FR-008: Formatter Subagent

```
Create the Formatter agent in src/agents/formatter.py. This agent:
1. Takes the raw output and formats it for the user
2. Supports multiple output formats: markdown, DOCX, PDF, XLSX, PPTX
3. Applies consistent styling and structure
4. allowedTools: Read, Write, Bash, Skill
5. Can invoke the document-creation skill for complex formats
6. Must handle code blocks, tables, and images appropriately
```

### FR-009: Hallucination Guard (Verifier)

```
Create the Verifier agent in src/agents/verifier.py. This agent:
1. Checks all factual claims in the output for accuracy
2. Produces a VerificationReport (Pydantic schema) with:
   - verified_claims: claims confirmed as accurate
   - flagged_claims: claims that could not be verified (with reasons)
   - confidence_score: overall verification confidence
   - sources_checked: what sources were used for verification
3. allowedTools: Read, WebSearch, WebFetch
4. Flags any potential hallucinations for the Verdict Matrix
```

### FR-010: Adversarial Critic

```
Create the Critic agent in src/agents/critic.py. This agent:
1. Actively tries to find flaws in the solution
2. Produces a CritiqueReport (Pydantic schema) with:
   - weaknesses: identified weaknesses
   - attack_vectors: potential failure modes
   - missing_considerations: overlooked aspects
   - improvement_suggestions: specific improvements
   - overall_assessment: severity rating
3. allowedTools: Read, Grep
4. Must adopt an adversarial perspective - assume the solution is wrong
```

### FR-011: Final Reviewer Subagent

```
Create the Final Reviewer agent in src/agents/reviewer.py. This agent:
1. Performs the final quality check before output
2. Evaluates the Verdict Matrix (Verifier x Critic results)
3. Produces a ReviewVerdict (Pydantic schema) with:
   - verdict: PASS / PASS_WITH_CAVEATS / REVISE / REJECT / ESCALATE
   - revision_instructions: what to fix if REVISE
   - caveats: warnings if PASS_WITH_CAVEATS
   - quality_score: final quality score
4. allowedTools: Read, Glob, Grep
5. If verdict is REVISE, Orchestrator must re-invoke Executor with revision instructions
```

### FR-012: Memory Curator

```
Create the Memory Curator agent in src/agents/memory_curator.py. This agent:
1. Manages persistent memory across sessions
2. Stores and retrieves relevant context from previous sessions
3. Produces memory artifacts (key decisions, patterns, preferences)
4. allowedTools: Read, Write, Glob
5. Maintains a structured memory store in the project directory
6. Must summarize and compress old memories to stay within context limits
```

### FR-013: Domain Council Chair

```
Create the Council Chair in src/agents/council.py (CouncilChair class). This agent:
1. Activated only for Tier 3-4 tasks
2. Selects which SME personas to spawn based on task domain
3. Produces an SMESelectionReport with:
   - selected_smes: list of SME persona IDs to spawn
   - rationale: why each SME was selected
   - interaction_mode: consult / debate / co-author per SME
4. allowedTools: none (pure reasoning based on registry)
5. Coordinates SME contributions and resolves conflicts
```

### FR-014: Quality Arbiter

```
Create the Quality Arbiter in src/agents/council.py (QualityArbiter class). This agent:
1. Activated only for Tier 4 (Adversarial) tasks
2. Mediates disputes between Verifier and Critic
3. Produces a QualityVerdict with:
   - final_decision: the arbiter's ruling
   - reasoning: detailed reasoning for the decision
   - dissenting_views: any views that were overruled
4. allowedTools: none (pure reasoning)
5. Has final authority on quality disputes
```

### FR-015: Ethics and Safety Advisor

```
Create the Ethics Advisor in src/agents/council.py (EthicsAdvisor class). This agent:
1. Activated for Tier 4 tasks or when ethical concerns are flagged
2. Reviews output for ethical implications
3. Produces an EthicsReview with:
   - ethical_concerns: identified ethical issues
   - bias_assessment: potential biases in the output
   - safety_risks: safety-related concerns
   - recommendations: specific recommendations
   - approved: boolean approval
4. allowedTools: none (pure reasoning)
5. Can veto output if serious ethical concerns are found
```

---

## SME System

### FR-016: SME Persona Registry

```
Create the SME Persona Registry in src/core/sme_registry.py. This module:
1. Defines the SMEPersona dataclass with:
   - persona_id, display_name, domain, expertise_areas
   - system_prompt_template, skill_files, interaction_modes
2. Maintains a global PERSONA_REGISTRY dictionary
3. Provides register_persona(), get_persona(), list_personas(), get_personas_for_domain()
4. Loads persona system prompts from config/sme/{persona_id}.md
5. Supports dynamic registration of custom personas
```

### FR-017: SME Persona Spawning

```
Implement SME persona spawning in the Orchestrator. When the Council Chair selects SMEs:
1. Look up persona in PERSONA_REGISTRY
2. Build ClaudeAgentOptions with persona's system_prompt_template
3. Set allowedTools from persona configuration + sme_default tools
4. Spawn via spawn_subagent() with the persona's model preference
5. Collect SME output and feed back to operational agents
6. Track SME cost separately from operational agent cost
```

### FR-018: SME Interaction Modes

```
Implement 3 SME interaction modes in the Orchestrator:
1. CONSULT: SME provides expert opinion, agents decide whether to incorporate
2. DEBATE: SME engages in structured debate with other SMEs (self-play)
3. CO_AUTHOR: SME actively participates in solution creation alongside agents

Each mode affects how SME output is integrated into the pipeline:
- CONSULT: output appended as context to downstream agents
- DEBATE: positions collected, consensus synthesized by Council Chair
- CO_AUTHOR: SME output merged into Executor's solution
```

### FR-019: Built-in SME Persona Library

```
Register 10 built-in SME personas in the registry:
1. software-architect: System design, patterns, trade-offs
2. security-analyst: Vulnerability assessment, threat modeling
3. data-scientist: ML/AI, statistics, data pipelines
4. ux-designer: User experience, accessibility, design systems
5. devops-engineer: CI/CD, infrastructure, deployment
6. technical-writer: Documentation, API docs, tutorials
7. qa-engineer: Testing strategies, quality metrics
8. database-expert: Schema design, query optimization
9. ml-engineer: Model training, inference optimization
10. cloud-architect: Cloud services, scalability, cost optimization

Each persona needs a config/sme/{id}.md system prompt template.
```

---

## SDK Configuration

### FR-020: Subagent Context Isolation

```
Ensure all subagents have isolated context:
1. Each subagent is spawned via the Claude Agent SDK Task tool
2. Subagents have independent context windows (no shared memory)
3. Communication flows only through the Orchestrator (hub-and-spoke)
4. allowedTools enforces least-privilege per agent (AGENT_ALLOWED_TOOLS map)
5. Subagents cannot spawn their own subagents
6. Session state is NOT shared between subagents
```

### FR-021: Structured Output via JSON Schema

```
Implement structured output for all agents:
1. Define Pydantic v2 models in src/schemas/ for each agent's output
2. Use model_json_schema() to generate JSON Schema for SDK outputFormat
3. Configure outputFormat in ClaudeAgentOptions for each agent
4. Validate agent output against schema before accepting
5. Retry with schema reminder if validation fails (up to max_retries)
6. The _get_output_schema() function maps agent names to their Pydantic models
```

---

## Complexity & Skills

### FR-022: Four-Tier Complexity Classification

```
Implement complexity classification in src/core/complexity.py:
1. Tier 1 (Direct): Simple queries, 3 agents (Analyst, Executor, Formatter)
2. Tier 2 (Standard): Moderate tasks, 7 agents (+ Planner, Researcher, Reviewer, Verifier)
3. Tier 3 (Deep): Complex tasks, 10-15 agents (+ Council Chair, 1-3 SMEs)
4. Tier 4 (Adversarial): High-stakes tasks, 13-18 agents (Full Council + SMEs + Critic)

Classification uses keyword analysis, entity count, domain detection, and ambiguity scoring.
Provide classify_complexity(query) -> ComplexityResult with tier, confidence, reasoning.
```

### FR-023: Mid-Execution Tier Escalation

```
Implement tier escalation in the Orchestrator:
1. After each agent completes, check output for escalation_needed flag
2. If any agent sets escalation_needed=True, bump tier up by 1
3. Spawn additional agents appropriate for the new tier
4. Re-run affected pipeline phases with expanded agent set
5. Log escalation events with reasoning
6. Never escalate beyond Tier 4
```

### FR-024: Agent Skills via SKILL.md

```
Configure skills for agents via the SDK:
1. Skills are defined in .claude/skills/{skill-name}/SKILL.md
2. Set setting_sources=["user", "project"] in ClaudeAgentOptions
3. SDK auto-discovers SKILL.md files in the project
4. Agents invoke skills via the Skill tool
5. Each SKILL.md has YAML frontmatter with name, description, and instructions
```

### FR-025: Orchestrator Skill Selection

```
Implement skill selection in the Orchestrator:
1. get_skills_for_agent(agent_name) returns applicable skill names
2. Inject skill references into agent system prompts
3. Agents are told which skills they can invoke
4. AGENT_SKILLS map defines the default skill assignments
5. Skills loaded based on task type and agent role
```

### FR-026: Built-in Skill Library

```
Create 7 built-in skills in .claude/skills/:
1. code-generation: Clean, secure, maintainable code patterns
2. document-creation: Professional document generation (DOCX, PDF, etc.)
3. requirements-engineering: Requirements elicitation and analysis
4. architecture-design: Software architecture design patterns
5. web-research: Web research with source verification
6. test-case-generation: Comprehensive test case generation
7. multi-agent-reasoning: Core multi-agent reasoning patterns

Each skill has a SKILL.md with frontmatter and detailed instructions.
```

### FR-027: Skill Authoring Template

```
Create a skill template in .claude/skills/_template/:
1. SKILL.md with YAML frontmatter template:
   ---
   name: "Skill Name"
   description: "What this skill does"
   version: "1.0.0"
   ---
2. Instructions section with step-by-step guidance
3. Examples section with usage examples
4. README explaining how to create custom skills
```

### FR-028: Skill-per-Agent Assignment

```
Implement skill-per-agent assignment:
1. AGENT_SKILLS map in sdk_integration.py assigns skills to agents
2. executor -> code-generation
3. formatter -> document-creation
4. analyst -> requirements-engineering
5. planner -> architecture-design
6. researcher -> web-research
7. code_reviewer -> code-generation
8. orchestrator -> multi-agent-reasoning
Skills are injected into system prompts when building ClaudeAgentOptions.
```

### FR-029: Skill Chaining

```
Implement skill chaining through the pipeline:
1. Pipeline phases chain outputs via _build_agent_input()
2. Each agent receives previous agents' outputs as context
3. Skills can reference outputs from prior skill invocations
4. The pipeline tracks which skills were used and their outputs
5. Downstream agents can build on upstream skill results
```

---

## Pipeline & Patterns

### FR-030: Eight-Phase Execution Pipeline

```
Implement the 8-phase pipeline in src/core/pipeline.py:
1. Task Intelligence: Analyst produces TaskIntelligenceReport
2. Council Review: Council Chair selects SMEs (Tier 3-4 only)
3. Planning: Planner produces ExecutionPlan
4. Research: Researcher produces EvidenceBrief
5. Solution: Executor implements the solution
6. Review: Code Reviewer + Verifier + Critic evaluate
7. Revision: Executor re-implements if Verdict Matrix requires it
8. Final Review: Final Reviewer produces ReviewVerdict

Each phase maps to specific agents. Pipeline is configurable per tier.
```

### FR-031: Verdict Matrix

```
Implement the Verdict Matrix in src/core/verdict.py:
A 2x2 matrix combining Verifier and Critic assessments:

| | Critic: No Issues | Critic: Issues Found |
|---|---|---|
| Verifier: Confirmed | PASS | PASS_WITH_CAVEATS |
| Verifier: Flagged | REVISE | ESCALATE |

Plus a 5th action: REJECT when both find critical issues.
The matrix feeds into the Final Reviewer's verdict decision.
If REVISE, Orchestrator calls _re_execute_phase() to re-invoke Executor.
```

### FR-032: Self-Play Debate Protocol

```
Implement the debate protocol in src/core/ensemble.py:
1. Spawn 2-3 agents with different perspectives on the same question
2. Each agent produces a position statement with supporting evidence
3. Agents respond to each other's positions (via Orchestrator relay)
4. Continue for configurable number of rounds (default: 2)
5. Council Chair synthesizes final position from debate
6. Record debate history for the Debate Viewer UI
```

### FR-033: Ensemble Patterns

```
Implement ensemble reasoning patterns in src/core/ensemble.py:
1. Consensus: Multiple agents must agree (configurable threshold)
2. Majority Vote: Most common answer wins
3. Weighted: Agents have different weights based on expertise
4. Debate: Adversarial argumentation (see FR-032)
5. Cascade: Sequential refinement through multiple agents

Each pattern takes a list of agent outputs and produces a synthesized result.
```

---

## SDK Integration

### FR-034: Agent SDK Query Configuration

```
Implement ClaudeAgentOptions in src/core/sdk_integration.py:
@dataclass
class ClaudeAgentOptions:
    name: str                    # Display name
    model: str                   # Model ID (e.g., claude-sonnet-4-20250514)
    system_prompt: str           # Agent's system prompt
    max_turns: int = 30          # Safety limit
    allowed_tools: List[str]     # Least-privilege tool list
    output_format: Dict          # JSON Schema from Pydantic
    setting_sources: List[str]   # ["user", "project"] for SKILL.md
    permission_mode: str         # "default" or "acceptEdits"
    append_system_prompt: str    # Additional instructions

Provide to_sdk_kwargs() to convert to SDK query() parameters.
```

### FR-035: CLAUDE.md Configuration

```
Create CLAUDE.md configuration files:
1. Root CLAUDE.md: Global instructions for all agents
2. config/agents/{agent}/CLAUDE.md: Per-agent system prompts (13 files)
3. config/sme/{persona}.md: SME persona templates (10 files)
4. Include re-orientation instructions for post-compaction recovery
5. Each per-agent CLAUDE.md defines the agent's role, constraints, and output format
```

### FR-036: Custom MCP Tools

```
Implement custom MCP tools in src/tools/custom_tools.py:
1. Define tools using a ToolMetadata dataclass (name, description, parameters, handler)
2. create_and_register_mcp_server(): Build MCP-compatible tool definitions
3. get_mcp_tool_names(): Return tool names for allowedTools configuration
4. Register tools: analyze_complexity, query_memory, search_evidence,
   validate_output, get_session_context, calculate_cost, format_output
5. Return MCP server config with tool_count for SDK registration
```

### FR-037: Per-Agent Model Selection

```
Implement per-agent model selection:
1. Default models configured in src/config.py via environment variables
2. MODEL_ORCHESTRATOR, MODEL_COUNCIL, MODEL_OPERATIONAL, MODEL_SME
3. get_model_for_agent(agent_name) returns the appropriate model
4. Settings panel allows per-agent model override
5. Model selection flows through build_agent_options() to ClaudeAgentOptions
```

### FR-038: Multimodal Input

```
Implement multimodal input support:
1. CLI: --input-file / -i option accepts file paths
2. Supported formats: .txt, .md, .py, .js, .ts, .docx, .pdf, .png, .jpg, .xlsx, .pptx
3. Orchestrator _load_input_content() reads file contents
4. Text files: included inline with code blocks
5. Binary files: referenced with metadata (size, type)
6. File content appended to user prompt for all downstream agents
```

---

## Output & Session

### FR-039: Multi-Format Output

```
Implement multi-format output:
1. Formatter agent supports: markdown, DOCX, PDF, XLSX, PPTX
2. CLI --file option saves output to specified path
3. Format auto-detected from file extension
4. Streamlit provides download buttons for each format
5. Default output is always markdown to terminal/chat
```

### FR-040: Session Management

```
Implement session management in src/session/:
1. SessionManager: create, resume, list, delete sessions
2. Each session has a unique ID, creation timestamp, and state
3. Session state includes: tier, cost, active agents, phase, outputs
4. Sessions persisted to disk (JSON) for resume capability
5. Session context passed to all subagents for continuity
```

### FR-041: Context Compaction

```
Implement context compaction in src/session/compaction.py:
1. ContextCompactor monitors context window usage
2. Auto-triggers compaction when approaching limits
3. _build_reorientation_prompt() re-reads CLAUDE.md after compaction
4. Re-orientation includes: session state, active agents, current phase
5. Ensures agents don't lose critical context after compaction
6. Add re-orientation instructions to root CLAUDE.md
```

---

## CLI & UI

### FR-042: CLI Interface

```
Implement the CLI in src/cli/main.py using Typer:
1. mas query "prompt" - Run a query through the system
2. mas query "prompt" --tier 3 - Force a specific tier
3. mas query "prompt" -i file.py - Attach input file
4. mas query "prompt" --file output.md - Save output to file
5. mas session list/resume/delete - Session management
6. mas config show/set - Configuration management
7. Entry point: mas = "src.cli.main:app" in pyproject.toml
```

### FR-043: Streamlit Chat Interface

```
Implement the chat interface in src/ui/pages/chat.py:
1. Chat message display with user/assistant roles
2. Text input with send button
3. File upload widget wired to orchestrator --input-file
4. Session selector sidebar
5. Tier indicator showing current complexity level
6. Cost display updated after each agent completes
7. Streaming output placeholder for real-time display
```

### FR-044: Streamlit Agent Activity Panel

```
Implement the agent activity panel in src/ui/components/:
1. Show all active agents with status indicators
2. Real-time status: idle, running, complete, error
3. Agent timeline showing execution order
4. Per-agent cost and token usage display
5. Expandable sections for agent output details
6. Color-coded by tier: Council=gold, Operational=blue, SME=green
```

### FR-045: Streamlit Results Inspector

```
Implement the Results Inspector in src/ui/components/results_inspector.py:
1. Display aggregated results with tier colour-coding
2. Council results highlighted in gold
3. Operational agent results in blue
4. SME contributions in green
5. Flagged claims highlighted in red (from Verifier)
6. st.expander for each subagent's detailed output
7. Quality score visualization
```

### FR-046: Streamlit Debate Viewer

```
Implement the Debate Viewer in src/ui/components/debate_viewer.py:
1. Display debate positions with colour-coded cards
2. SME domain badges showing expertise area
3. Consensus badge when agreement is reached
4. Arbiter verdict display with reasoning
5. Round-by-round debate progression
6. Plotly chart for position strength visualization
```

### FR-047: Streamlit Cost Dashboard

```
Implement the Cost Dashboard in src/ui/components/:
1. Total session cost display
2. Per-agent cost breakdown (bar chart)
3. Cost over time (line chart)
4. Budget utilization percentage
5. Token usage breakdown (input vs output)
6. Model cost comparison
7. Real-time updates during execution
```

### FR-048: Streamlit Skill Catalogue

```
Implement the Skill Catalogue in src/ui/pages/skills.py:
1. List all available skills from .claude/skills/
2. Parse YAML frontmatter for skill metadata
3. Display skill name, description, version
4. Show agent/SME assignments for each skill
5. SKILL.md content preview with syntax highlighting
6. Skill creation wizard (link to template)
```

### FR-049: Streamlit SME Persona Browser

```
Implement the SME Persona Browser in src/ui/pages/sme_browser.py:
1. List all registered SME personas from the registry
2. Show persona details: name, domain, expertise areas
3. Active SME highlighting (currently spawned)
4. Interaction mode indicators (consult/debate/co-author)
5. Persona system prompt preview
6. Custom persona creation form
```

### FR-050: Streamlit Settings Panel

```
Implement the Settings Panel in src/ui/pages/settings.py:
1. Per-agent model selection dropdowns
2. Agent enable/disable toggles
3. SME controls (activate/deactivate personas)
4. Budget configuration (max_budget_usd)
5. Tier override settings
6. Model defaults configuration
7. Save/load settings to session state
```

### FR-051: Streamlit File Upload/Download

```
Implement file upload and download in Streamlit:
1. File upload widget in chat page
2. Upload wired to orchestrator via --input-file equivalent
3. Download buttons for output artifacts
4. Support formats: MD, DOCX, PDF, XLSX, PPTX
5. Download buttons appear after query completion
6. File preview for uploaded files
```

---

## Infrastructure

### FR-052: Agent Activity Logging

```
Implement structured logging in src/utils/logging.py:
1. Use structlog for structured JSON logging
2. Log agent lifecycle: spawn, execute, complete, error
3. Log inter-agent communication (input/output summaries)
4. Log cost events: token usage, cost calculation
5. Log escalation events with reasoning
6. Configurable log levels per component
```

### FR-053: Cost Tracking and Budget

```
Implement cost tracking in src/utils/cost.py:
1. CostTracker class tracking per-agent and total cost
2. Token-based cost calculation per model
3. Budget enforcement with configurable max_budget_usd
4. Budget warning at 80% utilization
5. Hard stop at 100% budget
6. Cost report generation for session summary
```

### FR-054: Subagent Failure Handling

```
Implement failure handling in the Orchestrator:
1. Non-critical agents: retry 1x with exponential backoff
2. Critical agents (Executor, Verifier): retry 2x
3. Graceful degradation: skip non-critical agents on persistent failure
4. Error propagation: bubble up critical failures to user
5. Fallback chain: SDK -> Anthropic API -> simulation
6. Log all failures with context for debugging
```

### FR-055: max_turns Safety

```
Implement max_turns safety limits:
1. Orchestrator: max_turns from settings (default 100)
2. Executor: max_turns from settings (default 50)
3. Other subagents: max_turns from settings (default 30)
4. Configured in ClaudeAgentOptions per agent
5. Prevents runaway agent loops
6. Log warning when agent approaches max_turns
```

### FR-056: Project Structure

```
Ensure correct project structure:
src/
  agents/       - All agent implementations
  core/         - Core logic (pipeline, complexity, ensemble, verdict, SDK)
  schemas/      - Pydantic v2 models for all agent outputs
  tools/        - Custom MCP tools
  session/      - Session management and compaction
  ui/           - Streamlit app, pages, components
  cli/          - Typer CLI
  config.py     - Settings and configuration
  utils/        - Logging, cost tracking
tests/
  unit/         - Per-agent unit tests
  integration/  - Tier workflow integration tests
config/
  agents/       - Per-agent CLAUDE.md files
  sme/          - SME persona templates
.claude/skills/ - Skill definitions
docs/           - Documentation
```

### FR-057: Environment Configuration

```
Create .env.example with all configuration variables:
ANTHROPIC_API_KEY=your-key-here
MODEL_ORCHESTRATOR=claude-sonnet-4-20250514
MODEL_COUNCIL=claude-sonnet-4-20250514
MODEL_OPERATIONAL=claude-sonnet-4-20250514
MODEL_SME=claude-sonnet-4-20250514
MAX_BUDGET_USD=10.00
MAX_TURNS_ORCHESTRATOR=100
MAX_TURNS_EXECUTOR=50
MAX_TURNS_SUBAGENT=30
LOG_LEVEL=INFO
SESSION_DIR=.sessions
```

### FR-058: Dependencies

```
Maintain dependencies in pyproject.toml and requirements.txt:
- claude-agent-sdk>=0.1.0
- anthropic>=0.40.0
- pydantic>=2.0.0, pydantic-settings>=2.0.0
- typer>=0.12.0
- streamlit>=1.40.0
- structlog>=24.0.0
- python-dotenv>=1.0.0
- python-docx>=1.0.0, openpyxl>=3.1.0, python-pptx>=0.6.0
- pillow>=10.0.0
- plotly>=5.0.0
- pyyaml>=6.0.0
Dev: pytest, pytest-asyncio, pytest-cov, pytest-mock, black, ruff, mypy
```

### FR-059: Unit Tests

```
Create unit tests in tests/unit/:
1. One test file per agent (test_analyst.py, test_planner.py, etc.)
2. Minimum 5 tests per agent covering:
   - Agent initialization
   - System prompt generation
   - Input processing
   - Output schema validation
   - Error handling
3. Use pytest with fixtures for common setup
4. Mock SDK calls to avoid API dependencies
5. Target: 180+ total unit tests
```

### FR-060: Integration Tests

```
Create integration tests in tests/integration/:
1. test_tier_workflows.py: End-to-end tier 1-4 workflows
2. Gated by MAS_RUN_INTEGRATION=true environment variable
3. Uses pytestmark = pytest.mark.skipif for gating
4. Tests actual agent spawning and pipeline execution
5. Validates inter-agent data flow
6. Checks budget enforcement and failure handling
```

### FR-061: README and Quick Start

```
Create comprehensive README.md:
1. Project overview with architecture diagram (Mermaid)
2. Quick start guide (install, configure, run)
3. CLI usage examples
4. Streamlit UI screenshots/description
5. SME persona creation guide with template
6. Configuration reference
7. Development setup instructions
```

### FR-062: Vibe Coding Prompts

```
Create docs/vibe-prompts.md (this document):
1. Self-contained prompt for each of the 62 functional requirements
2. Each prompt provides full context for Claude Code implementation
3. Prompts are ordered by implementation dependency
4. Table of contents for easy navigation
5. Can be fed directly to Claude Code for implementation
```

---

*Generated from FRD_MultiAgent_Prototype_v4.docx - All 62 functional requirements covered.*
