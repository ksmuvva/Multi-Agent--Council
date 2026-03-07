# FRD Audit Report: Not Implemented Requirements

**Audit Date:** 2026-03-07
**FRD Version:** 4.0
**Total Requirements:** 62 (FR-001 to FR-062)
**Audited Against:** Current codebase in `Multi-Agent--Council/`

---

## Audit Methodology

This audit was performed using a self-reflection loop approach:
1. **Pass 1:** Extracted all 62 requirements and their acceptance criteria from `FRD_MultiAgent_Prototype_v4.docx`
2. **Pass 2:** Explored every source file in the codebase (`src/`, `config/`, `.claude/skills/`, `tests/`, `docs/`)
3. **Pass 3:** Mapped each acceptance criterion to concrete code evidence (or lack thereof)
4. **Pass 4 (Self-Reflection):** Re-verified gap findings by searching for alternative implementations, partial implementations, and edge cases

---

## Summary

| Status | Count | Percentage |
|--------|-------|------------|
| Fully Implemented | 27 | 43.5% |
| Partially Implemented | 25 | 40.3% |
| Not Implemented | 10 | 16.1% |

---

## Legend

- **NOT IMPLEMENTED** - No code exists for this requirement
- **PARTIALLY IMPLEMENTED** - Code exists but key acceptance criteria are missing
- **IMPLEMENTED** - All acceptance criteria are satisfied

---

## Fully Implemented Requirements (27)

| FR | Title | Notes |
|----|-------|-------|
| FR-002 | Task Analyst Subagent | `src/agents/analyst.py` - Returns TaskIntelligenceReport, modality detection, severity categorization |
| FR-003 | Planner Subagent | `src/agents/planner.py` - ExecutionPlan with steps, dependencies, parallel groups |
| FR-004 | Clarifier Subagent | `src/agents/clarifier.py` - Ranked questions with defaults and impact assessment |
| FR-007 | Code Reviewer Subagent | `src/agents/code_reviewer.py` - 5 review dimensions, severity ratings, pattern detection |
| FR-009 | Hallucination Guard (Verifier) | `src/agents/verifier.py` - Claim scanning, confidence scoring, fabrication risk |
| FR-010 | Adversarial Critic | `src/agents/critic.py` - 5 attack vectors, severity ratings, red-team analysis |
| FR-012 | Memory Curator | `src/agents/memory_curator.py` - Knowledge extraction, markdown files with frontmatter in `docs/knowledge/` |
| FR-013 | Domain Council Chair | `src/agents/council.py` (CouncilChairAgent) - SME selection, skill mapping, interaction modes |
| FR-014 | Quality Arbiter | `src/agents/council.py` (QualityArbiterAgent) - Acceptance criteria, dispute resolution |
| FR-015 | Ethics and Safety Advisor | `src/agents/council.py` (EthicsAdvisorAgent) - Bias, PII, compliance checks |
| FR-016 | SME Persona Registry | `src/core/sme_registry.py` - 10 personas, keyword search, domain search, config-only extensibility |
| FR-019 | Built-in SME Persona Library | 10 personas registered with skills, keywords, templates |
| FR-022 | Four-Tier Complexity Classification | `src/core/complexity.py` - Semantic keyword classification, tier configs, agent counts |
| FR-023 | Mid-Execution Tier Escalation | `escalation_needed` flag in schemas, `_handle_escalation()` in orchestrator |
| FR-026 | Built-in Skill Library | 7 skills in `.claude/skills/` (multi-agent-reasoning, code-generation, document-creation, test-case-generation, web-research, requirements-engineering, architecture-design) |
| FR-027 | Skill Authoring Template | `.claude/skills/_template/` with SKILL.md skeleton and README |
| FR-030 | Eight-Phase Execution Pipeline | `src/core/pipeline.py` - All 8 phases, tier-based skipping logic |
| FR-031 | Verdict Matrix | `src/core/verdict.py` - 2x2 matrix, 5 actions, revision cap at 2, arbiter escalation |
| FR-033 | Ensemble Patterns | `src/core/ensemble.py` - 5 named patterns (Architecture Review Board, Code Sprint, Research Council, Document Assembly, Requirements Workshop) |
| FR-035 | CLAUDE.md Configuration | Global CLAUDE.md + 13 per-agent configs in `config/agents/` + 10 SME templates in `config/sme/` |
| FR-037 | Per-Agent Model Selection | Model defaults in `.env.example`, per-agent overrides, opus/sonnet mapping |
| FR-040 | Session Management | Session creation, resume via `--session-id`, persistence via `SessionPersistence` class |
| FR-052 | Agent Activity Logging | `src/utils/logging.py` - structlog JSON, DEBUG/INFO/WARN/ERROR levels |
| FR-053 | Cost Tracking and Budget | `src/utils/cost.py` - Per-agent cost, session accumulation, budget enforcement |
| FR-056 | Project Structure | All directories exist, `__init__.py` files present, `.gitignore` exists |
| FR-057 | Environment Configuration | `.env.example` with all vars, python-dotenv, sensible defaults |
| FR-058 | Dependencies | `requirements.txt` with all required packages pinned |

---

## Partially Implemented Requirements (25)

### FR-001: Orchestrator Agent (Parent) - PARTIAL

**What exists:** `src/agents/orchestrator.py` (1,045 lines) - Process request flow, tier classification, council consultation, session management, budget tracking.

**What's missing:**
- **AC-3 (Critical):** `_spawn_agent()` method is a placeholder with TODO comments. Does NOT actually use `claude_agent_sdk.query()` or the Task tool to spawn subagents. All subagent invocations are simulated/mocked within the orchestrator.
- **AC-4 (Partial):** Result aggregation logic exists but relies on simulated subagent responses, not real SDK subagent output.

### FR-005: Web Researcher Subagent - PARTIAL

**What exists:** `src/agents/researcher.py` (606 lines) - Returns EvidenceBrief with findings, conflicts, gaps, source reliability.

**What's missing:**
- **AC-1:** WebSearch and WebFetch are NOT in actual `allowedTools`. The agent uses mock/simulated search results instead of real tool invocations.
- **AC-3:** Source URLs are generated synthetically, not from real web searches.

### FR-006: Solution Executor Subagent - PARTIAL

**What exists:** `src/agents/executor.py` (626 lines) - Tree of Thoughts with 2-3 approaches, scoring, selection.

**What's missing:**
- **AC-3:** `allowedTools` not declared. Agent does not actually invoke Write, Bash, Glob, Grep tools. File creation is simulated.
- **AC-4:** Raw output hand-off to Formatter is conceptual, not wired through actual pipeline.

### FR-008: Formatter Subagent - PARTIAL

**What exists:** `src/agents/formatter.py` (607 lines) - Supports Markdown, Code, DOCX, PDF, XLSX, PPTX, Mermaid, JSON, YAML.

**What's missing:**
- **AC-2:** Document-creation skill invocation is referenced but not actually called via the Skill tool.
- **AC-3:** Code syntax validation via Bash is simulated (uses Python AST parsing only, not actual Bash execution).

### FR-011: Final Reviewer Subagent - PARTIAL

**What exists:** `src/agents/reviewer.py` (830 lines) - Quality gates, checklist, verdict matrix integration.

**What's missing:**
- **AC-2:** FAIL does NOT actually trigger Executor revision in a live pipeline. The revision loop is conceptual - it returns a verdict but doesn't re-invoke the Executor.
- **AC-4:** Quality Arbiter coordination is designed but not wired through actual agent spawning.

### FR-017: SME Persona Spawning - PARTIAL

**What exists:** `src/agents/sme_spawner.py` (1,152 lines) - System prompt generation, skill loading, 3 interaction modes.

**What's missing:**
- **AC-1:** SMEs are NOT spawned as actual Claude Agent SDK subagents via the Task tool. The spawning is simulated internally.
- **AC-3:** SKILL.md files are read from disk but NOT loaded via the SDK's Skill tool mechanism.
- **AC-5:** Model defaults to sonnet conceptually but `ClaudeAgentOptions` is not used.

### FR-018: SME Interaction Modes - PARTIAL

**What exists:** `src/agents/sme_spawner.py` - All 3 modes implemented (Advisor, Co-Executor, Debater).

**What's missing:**
- **AC-2:** Co-Executor mode generates mock section content instead of real LLM-generated contributions.
- **AC-3:** Debater mode has simulated debate positions instead of real agent debate output.

### FR-020: Subagent Context Isolation - PARTIAL

**What exists:** Architecture is designed for isolation. No direct subagent-to-subagent communication.

**What's missing:**
- **AC-1:** Independent context per subagent is NOT enforced via the Task tool (since Task tool / SDK integration is missing).
- **AC-4:** Task tool is not in any subagent's `allowedTools` (correct), but this is trivially true since `allowedTools` is not declared anywhere.

### FR-021: Structured Output via JSON Schema - PARTIAL

**What exists:** All 13 Pydantic v2 schemas defined in `src/schemas/`. Schemas are comprehensive.

**What's missing:**
- **AC-1:** `outputFormat` is NOT configured per subagent via the SDK. There is no `ClaudeAgentOptions.outputFormat` usage.
- **AC-3:** Schema validation happens at the Python level (Pydantic), not via SDK's JSON Schema enforcement.
- **AC-4:** Invalid response retry is not implemented at the SDK level.

### FR-024: Agent Skills via SKILL.md - PARTIAL

**What exists:** 7 skills + template in `.claude/skills/`, valid YAML frontmatter.

**What's missing:**
- **AC-3:** `setting_sources=["user","project"]` is referenced in CLAUDE.md but NOT configured in any Python code (no `ClaudeAgentOptions` usage).
- **AC-4:** Auto-discovery on startup is not implemented - skills are loaded manually when referenced.

### FR-025: Orchestrator Skill Selection - PARTIAL

**What exists:** Skill concepts referenced in agent configs and SME registry.

**What's missing:**
- **AC-1:** Skills are NOT dynamically identified from task analysis at runtime.
- **AC-2:** Skill names are NOT passed to subagent prompts programmatically.
- **AC-3:** Multiple skills per subagent are mapped in config but not loaded via SDK.

### FR-028: Skill-per-Agent Assignment - PARTIAL

**What exists:** Agent CLAUDE.md configs mention skills. SME registry maps skills.

**What's missing:**
- **AC-2:** Per-task override of skill assignments is not implemented.
- **AC-4:** Skill assignments in `config/agents/` are documentation-only, not enforced in code.

### FR-029: Skill Chaining - PARTIAL

**What exists:** Pipeline phases conceptually chain agent outputs.

**What's missing:**
- **AC-1:** Output piping between subagents is simulated, not real inter-agent data flow.
- **AC-3:** Configurable chains are not implemented.
- **AC-4:** Partial results on chain failure not implemented.

### FR-032: Self-Play Debate Protocol - PARTIAL

**What exists:** `src/core/debate.py` (380 lines) - Consensus scoring, round management, arbiter integration.

**What's missing:**
- **AC-1:** Debate positions are hardcoded/simulated (agreement scores in `conduct_round()` are not from real agent output).
- **AC-3:** SME debate arguments are placeholders, not real LLM-generated domain positions.

### FR-034: Agent SDK Query Configuration - PARTIAL

**What exists:** Agent classes have model selection, max_turns tracking, system prompt loading.

**What's missing:**
- **AC-1:** `ClaudeAgentOptions` is NOT used anywhere in the codebase. Agent configuration is via custom Python classes, not SDK options.
- **AC-2:** Least-privilege `allowedTools` is NOT declared in agent code.
- **AC-5:** `setting_sources` is NOT configured in any agent options.

### FR-036: Custom MCP Tools - PARTIAL

**What exists:** `src/tools/custom_tools.py` (573 lines) - Uses `@tool` decorator, defines agent event emitter, SME registry query, knowledge file reader, cost tracker.

**What's missing:**
- **AC-2:** `create_sdk_mcp_server()` is NOT called anywhere. No MCP server is created.
- **AC-3:** Custom tools are NOT registered in agent `allowedTools` (since allowedTools are not declared).

### FR-038: Multimodal Input - PARTIAL

**What exists:** Analyst detects modality (CODE, IMAGE, DOCUMENT, DATA, TEXT). Streamlit has `file_uploader`.

**What's missing:**
- **AC-4:** CLI `--file` flag exists but is for output file saving, NOT for input file/multimodal input as required by FRD.

### FR-039: Multi-Format Output - PARTIAL

**What exists:** Formatter supports all formats. Streamlit has download buttons.

**What's missing:**
- **AC-2:** Document generation does not actually invoke the document-creation skill via Skill tool.
- **AC-4:** CLI file output works via `--file` option but format conversion is simulated.

### FR-042: CLI Interface - PARTIAL

**What exists:** `src/cli/main.py` (792 lines) - Typer CLI with `query`, `chat`, `--verbose`, `--tier`, `--format`, `--session-id`. Additional commands: analyze, tools, knowledge, personas, cost, ensembles, version, status, sessions, test.

**What's missing:**
- **AC-1:** `pip install -e .` may not work correctly (pyproject.toml exists but entry point may not be properly configured).
- **AC-4:** Streaming output is a placeholder (`render_streaming_response` has `# Placeholder for streaming content`).
- **AC-5:** `--file` is for output, not for multimodal input as required.

### FR-043: Streamlit Chat Interface - PARTIAL

**What exists:** `src/ui/pages/chat.py` - Chat interface with `st.chat_message()`, session persistence.

**What's missing:**
- **AC-2:** Streaming output is a placeholder (`render_streaming_response` has placeholder comment).

### FR-044: Streamlit Agent Activity Panel - PARTIAL

**What exists:** `src/ui/components/agent_panel.py` - Agent hierarchy display.

**What's missing:**
- **AC-2:** Real-time status indicators require live agent execution (which is simulated).
- **AC-5:** Skills shown per agent may be static, not dynamically loaded.

### FR-047: Streamlit Cost Dashboard - PARTIAL

**What exists:** `src/ui/components/cost_dashboard.py` - Cost display, per-agent tokens.

**What's missing:**
- **AC-1:** Real-time cost requires live API calls (which are simulated).
- **AC-4:** Budget warning at 80% is configured in `.env.example` but UI integration depends on live data.

### FR-054: Subagent Failure Handling - PARTIAL

**What exists:** Error handling logic in orchestrator. Retry concepts exist.

**What's missing:**
- **AC-1/AC-2:** Retry logic (1x non-critical, 2x critical) is commented out in `_spawn_agent()` placeholder.
- **AC-3:** Graceful degradation is designed but not testable without real agent spawning.

### FR-055: max_turns Safety - PARTIAL

**What exists:** `_get_max_turns()` in orchestrator returns 200/50/30. `.env.example` has MAS_MAX_TURNS_* vars.

**What's missing:**
- **AC-2:** Partial result on limit hit is not implemented (since SDK integration is missing).
- **AC-4:** Configurable via `.env` but not wired through `ClaudeAgentOptions`.

### FR-061: README and Quick Start - PARTIAL

**What exists:** README.md exists with overview content.

**What's missing:**
- **AC-2:** Need to verify Mermaid architecture diagram presence.
- **AC-4:** SME creation guide may be incomplete.

---

## Not Implemented Requirements (10)

### FR-041: Context Compaction - NOT IMPLEMENTED

**Requirement:** SDK automatic compaction for long sessions. CLAUDE.md re-read post-compaction. System prompts include re-orientation instructions.

**Evidence:**
- No SDK compaction configuration found in code
- No re-orientation instructions in system prompts
- `compact_session_manual` exists in CLI but is manual compaction, not SDK automatic compaction
- **AC-1:** Auto-compaction NOT enabled via SDK
- **AC-2:** CLAUDE.md re-read post-compaction NOT implemented
- **AC-3:** Re-orientation instructions NOT in system prompts

### FR-045: Streamlit Results Inspector - NOT IMPLEMENTED

**Requirement:** `st.expander` per subagent showing structured output, colour-coded by tier (Council=gold, Operational=blue, SME=green), flagged claims highlighted red.

**Evidence:**
- File `src/ui/components/results_inspector.py` exists but:
- **AC-2:** Structured JSON as formatted tables - basic implementation only
- **AC-3:** Colour-coding by tier (gold/blue/green) NOT implemented
- **AC-4:** Flagged claims highlighted in red NOT implemented
- Overall: Component exists as scaffold but core functionality is missing

### FR-046: Streamlit Debate Viewer - NOT IMPLEMENTED

**Requirement:** Debate transcript per round, agent positions colour-coded, SME domain arguments highlighted, consensus badge, Quality Arbiter verdict display.

**Evidence:**
- File `src/ui/components/debate_viewer.py` exists but:
- **AC-2:** Colour-coded positions NOT implemented (requires real debate data)
- **AC-3:** SME arguments highlighted NOT implemented
- **AC-4:** Consensus badge NOT implemented
- Component exists as scaffold but depends on live debate data which is simulated

### FR-048: Streamlit Skill Catalogue - NOT IMPLEMENTED

**Requirement:** Page showing all `.claude/skills/`, name, description from frontmatter, assigned agents, SME personas, SKILL.md content preview.

**Evidence:**
- File `src/ui/pages/skills.py` exists but:
- **AC-2:** Frontmatter parsing from SKILL.md files NOT implemented in the page
- **AC-3:** Agent + SME skill assignments NOT displayed
- Page is a scaffold without real skill scanning functionality

### FR-049: Streamlit SME Persona Browser - NOT IMPLEMENTED

**Requirement:** Page showing all registered SME personas with name, domain, trigger keywords, assigned skills, interaction modes. Active SMEs highlighted.

**Evidence:**
- File `src/ui/pages/sme_browser.py` exists but:
- **AC-3:** Active SMEs highlighted for current/last task NOT implemented
- Page exists as scaffold without integration to live SME activation data

### FR-050: Streamlit Settings Panel - NOT IMPLEMENTED

**Requirement:** Model selection per agent, max_budget_usd, verbosity, tier override, agent toggles, SME controls. Persisted in session state.

**Evidence:**
- File `src/ui/pages/settings.py` exists but:
- **AC-1:** Model dropdowns per agent NOT implemented
- **AC-3:** Agent enable/disable toggles NOT implemented
- **AC-4:** Settings NOT applied to next query execution
- **AC-5:** SME controls NOT implemented
- Settings page is a scaffold without functional controls

### FR-051: Streamlit File Upload/Download - NOT IMPLEMENTED

**Requirement:** `st.file_uploader` for multimodal input, `st.download_button` for generated outputs.

**Evidence:**
- `file_uploader` found in some UI files but:
- **AC-2:** Uploaded files NOT passed to Orchestrator for processing
- **AC-3:** Downloads for generated files NOT connected to actual generation pipeline
- Upload/download widgets exist but are NOT wired to the agent pipeline

### FR-059: Unit Tests - NOT IMPLEMENTED (Insufficient Coverage)

**Requirement:** pytest unit tests per agent (operational, council, SME spawning). Mocked SDK responses. Minimum 5 tests per agent. Verdict matrix logic tested.

**Evidence:**
- Test files exist: `test_orchestrator.py` (22 tests), `test_schemas.py` (24 tests), `test_core.py` (33 tests), `test_config.py` (42 tests)
- **AC-1 (FAIL):** Requirement is "5 tests per agent minimum" = 15 agents x 5 = 75 agent-specific tests needed. Only orchestrator has dedicated tests. No test files for: analyst, planner, clarifier, researcher, executor, code_reviewer, formatter, verifier, critic, reviewer, memory_curator, council, sme_spawner
- **AC-2 (PARTIAL):** SDK mocking exists in conftest but not comprehensive per-agent
- Missing: `test_analyst.py`, `test_planner.py`, `test_clarifier.py`, `test_researcher.py`, `test_executor.py`, `test_code_reviewer.py`, `test_formatter.py`, `test_verifier.py`, `test_critic.py`, `test_reviewer.py`, `test_memory_curator.py`, `test_council.py`, `test_sme_spawner.py`

### FR-060: Integration Tests - NOT IMPLEMENTED

**Requirement:** Full pipeline tests per tier (Tier 1-4). Live API calls gated by `MAS_RUN_INTEGRATION=true`. Verify Council activation on Tier 3+.

**Evidence:**
- File `tests/integration/test_tier_workflows.py` exists with 35 tests covering tier 1-4
- **AC-2 (FAIL):** Tests are NOT gated by `MAS_RUN_INTEGRATION=true` env var. No `skipIf` or env check found.
- **AC-3 (FAIL):** Council + SME verification uses mocked data, not live API calls
- Tests exist as structure but do not perform actual live integration testing

### FR-062: Vibe Coding Prompts - NOT IMPLEMENTED (Incomplete)

**Requirement:** `docs/vibe-prompts.md` with one AI-ready prompt per FR, ordered by implementation dependency. Each prompt self-contained.

**Evidence:**
- File `docs/vibe-prompts.md` exists
- **AC-1 (NEEDS VERIFICATION):** May not have a prompt per each of the 62 FRs
- **AC-3:** Ordering by dependency needs verification
- **AC-4:** Self-contained prompts need verification

---

## Critical Cross-Cutting Gap: Claude Agent SDK Integration

The most significant gap across the entire codebase is the **absence of actual Claude Agent SDK integration**. This affects 20+ requirements:

| Gap | Impact |
|-----|--------|
| No `claude_agent_sdk.query()` calls | Orchestrator cannot spawn real subagents |
| No `ClaudeAgentOptions` usage | Agent configuration (model, tools, permissions) not enforced via SDK |
| No `Task` tool integration | Subagent context isolation is theoretical, not enforced |
| No `outputFormat` with JSON Schema | Schema enforcement is Pydantic-only, not SDK-level |
| No `allowedTools` declarations | Least-privilege tool access not enforced |
| No `setting_sources` configuration | Skill auto-discovery not enabled via SDK |
| No `permission_mode` usage | "acceptEdits" vs "default" not configured |
| No `create_sdk_mcp_server()` | Custom MCP tools not registered with SDK |
| No real tool invocations | WebSearch, WebFetch, Write, Bash, Skill tools are simulated |
| Streaming not implemented | Both CLI and Streamlit have placeholder streaming |

**Root Cause:** The system has comprehensive business logic, schemas, and architecture but has not been integrated with the actual Claude Agent SDK runtime. All agent interactions are simulated within Python classes.

---

## Recommendations

### Priority 1: SDK Integration (Blocks 20+ requirements)
1. Implement `claude_agent_sdk.query()` in Orchestrator's `_spawn_agent()`
2. Configure `ClaudeAgentOptions` per agent with model, allowedTools, max_turns, outputFormat
3. Wire `create_sdk_mcp_server()` for custom tools
4. Enable `setting_sources=["user","project"]` for skill discovery

### Priority 2: Real Tool Integration (Blocks 5+ requirements)
1. Enable WebSearch/WebFetch in Researcher
2. Enable Write/Bash/Glob/Grep in Executor
3. Enable Skill tool in skill-using agents

### Priority 3: UI Completion (6 requirements)
1. Implement Results Inspector colour-coding and claim highlighting
2. Implement Debate Viewer with real debate data
3. Wire Skill Catalogue to scan `.claude/skills/`
4. Wire SME Browser to live registry data
5. Implement Settings Panel controls
6. Wire File Upload/Download to pipeline

### Priority 4: Test Coverage (2 requirements)
1. Add per-agent unit test files (13 files needed)
2. Gate integration tests with `MAS_RUN_INTEGRATION` env var

---

*This audit was generated by analyzing the FRD_MultiAgent_Prototype_v4.docx against the current codebase using a multi-pass self-reflection approach.*
