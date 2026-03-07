"""
Orchestrator Agent - Parent Agent

The single point of entry for all user requests and coordinator
of all subagents in the Multi-Agent Reasoning System.
"""

import os
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime

from src.core.complexity import (
    TierLevel,
    TierClassification,
    classify_complexity,
    should_escalate,
    get_escalated_tier,
    get_active_agents,
    get_council_agents,
)
from src.core.pipeline import (
    ExecutionPipeline,
    PipelineBuilder,
    create_execution_context,
    Phase,
)
from src.core.verdict import (
    evaluate_verdict_matrix,
    MatrixAction,
    calculate_phase_cost_estimate,
)
from src.core.debate import (
    DebateProtocol,
    trigger_debate,
    get_debate_participants,
)
from src.core.sme_registry import (
    SME_REGISTRY,
    find_personas_by_keywords,
    get_persona_for_display,
)
from src.schemas.analyst import TaskIntelligenceReport
from src.session import (
    SessionPersistence,
    ChatMessage,
    AgentOutput as SessionAgentOutput,
    check_and_compact,
    CompactionConfig,
)
from src.config import (
    get_settings,
    get_model_for_agent,
    get_api_key,
    get_provider,
)


@dataclass
class AgentExecution:
    """Record of an agent execution."""
    agent_name: str
    start_time: float
    end_time: Optional[float] = None
    status: str = "pending"  # pending, running, complete, failed
    output: Any = None
    error: Optional[str] = None
    tokens_used: int = 0
    cost_usd: float = 0.0


@dataclass
class SessionState:
    """State for an execution session."""
    session_id: str
    user_prompt: str
    tier_classification: Optional[TierClassification] = None
    current_tier: TierLevel = TierLevel.STANDARD
    revision_cycle: int = 0
    max_revisions: int = 2
    debate_rounds: int = 0
    max_debate_rounds: int = 2
    total_cost_usd: float = 0.0
    max_budget_usd: float = 5.0
    budget_warning_threshold: float = 0.8
    agent_executions: List[AgentExecution] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    active_smes: List[str] = field(default_factory=list)
    council_activated: bool = False
    escalation_history: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        """Get session duration in seconds."""
        end = self.end_time or time.time()
        return end - self.start_time

    @property
    def budget_utilization(self) -> float:
        """Get budget utilization as a ratio (0-1)."""
        return self.total_cost_usd / self.max_budget_usd if self.max_budget_usd > 0 else 0

    def should_warn_budget(self) -> bool:
        """Check if budget warning should be issued."""
        return self.budget_utilization >= self.budget_warning_threshold

    def is_budget_exceeded(self) -> bool:
        """Check if budget is exceeded."""
        return self.total_cost_usd >= self.max_budget_usd


class OrchestratorAgent:
    """
    The Orchestrator is the parent agent that coordinates all subagents.

    Uses Claude Agent SDK query() method for parent-child communication.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_budget_usd: Optional[float] = None,
        max_revisions: Optional[int] = None,
        max_debate_rounds: Optional[int] = None,
        verbose: Optional[bool] = None,
        enable_persistence: Optional[bool] = None,
        enable_auto_compact: Optional[bool] = None,
    ):
        """
        Initialize the Orchestrator.

        Args:
            api_key: API key (defaults to configured provider's API key from settings)
            max_budget_usd: Maximum session budget in USD (defaults to settings)
            max_revisions: Maximum revision cycles (defaults to settings)
            max_debate_rounds: Maximum debate rounds (defaults to settings)
            verbose: Enable verbose logging (defaults to settings)
            enable_persistence: Enable session persistence to disk (defaults to settings)
            enable_auto_compact: Enable automatic context compaction (defaults to settings)
        """
        # Load centralized settings
        self.settings = get_settings()

        # Use provided values or fall back to settings
        self.api_key = api_key or get_api_key()
        self.max_budget_usd = max_budget_usd if max_budget_usd is not None else self.settings.max_budget
        self.max_revisions = max_revisions if max_revisions is not None else 2
        self.max_debate_rounds = max_debate_rounds if max_debate_rounds is not None else 2
        self.verbose = verbose if verbose is not None else self.settings.debug
        self.enable_persistence = enable_persistence if enable_persistence is not None else self.settings.session_persistence
        self.enable_auto_compact = enable_auto_compact if enable_auto_compact is not None else True

        # Session persistence
        self.persistence = SessionPersistence() if self.enable_persistence else None
        self.compaction_config = CompactionConfig() if self.enable_auto_compact else None

        # Model configurations - now provider-agnostic
        self.orchestrator_model = get_model_for_agent("orchestrator")
        self.council_model = get_model_for_agent("council")
        self.operational_model = get_model_for_agent("analyst")  # Default operational
        self.sme_model = get_model_for_agent("sme")

        # Max turns per agent - from settings
        self.max_turns_orchestrator = self.settings.max_turns_orchestrator
        self.max_turns_subagent = self.settings.max_turns_subagent
        self.max_turns_executor = self.settings.max_turns_executor

        # Store provider info
        self.provider = get_provider()
        self.provider_config = self.settings.get_provider_config()

    # ========================================================================
    # Main Entry Point
    # ========================================================================

    def process_request(
        self,
        user_prompt: str,
        session_id: Optional[str] = None,
        file_path: Optional[str] = None,
        tier_override: Optional[int] = None,
        resume_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Process a user request through the multi-agent system.

        Args:
            user_prompt: The user's request
            session_id: Optional session ID for persistence
            file_path: Optional file attachment for multimodal input
            tier_override: Optional override for tier classification
            resume_context: Optional context from a resumed session

        Returns:
            Dictionary with response, metadata, and execution details
        """
        # Create session state
        session = self._create_session(
            user_prompt=user_prompt,
            session_id=session_id,
            resume_context=resume_context,
        )

        try:
            # Load multimodal content if provided
            input_content = self._load_input_content(user_prompt, file_path)

            # Step 1: Classify complexity (or use override)
            if tier_override:
                session.tier_classification = self._create_override_classification(
                    tier_override
                )
            else:
                session.tier_classification = classify_complexity(
                    user_prompt=input_content
                )

            session.current_tier = session.tier_classification.tier

            if self.verbose:
                self._log(f"Tier {session.current_tier}: {session.tier_classification.reasoning}")

            # Step 2: Check budget
            if session.is_budget_exceeded():
                return self._budget_exceeded_response(session)

            # Step 3: Council consultation for Tier 3-4
            if session.current_tier >= TierLevel.DEEP:
                self._consult_council(session, input_content)

                # Check budget again after Council
                if session.is_budget_exceeded():
                    return self._budget_exceeded_response(session)

            # Step 4: Create pipeline
            pipeline = PipelineBuilder.for_tier(session.current_tier)
            pipeline.max_revisions = session.max_revisions
            pipeline.max_debate_rounds = session.max_debate_rounds

            # Step 5: Execute pipeline
            execution_context = create_execution_context(
                user_prompt=input_content,
                tier_classification=session.tier_classification,
                session_id=session.session_id,
                additional_context={
                    "active_smes": session.active_smes,
                    "council_activated": session.council_activated,
                },
            )

            final_state = self._execute_pipeline(
                pipeline=pipeline,
                session=session,
                context=execution_context,
            )

            # Step 6: Generate final response
            response = self._generate_final_response(
                session=session,
                pipeline_state=final_state,
            )

            return response

        except Exception as e:
            if self.verbose:
                self._log(f"Error processing request: {e}")
            return self._error_response(session, str(e))

        finally:
            session.end_time = time.time()
            # Auto-save session
            if self.enable_persistence:
                self._save_session_state(session, response if 'response' in locals() else None)

    def execute(
        self,
        user_prompt: str,
        tier_level: int = 2,
        session_id: Optional[str] = None,
        format: str = "markdown",
        file_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute a query through the multi-agent system.

        This is the main entry point called by the CLI.

        Args:
            user_prompt: The user's query
            tier_level: Tier level (1-4) to use
            session_id: Optional session ID for persistence
            format: Output format (markdown, json, text)
            file_path: Optional file to attach

        Returns:
            Dictionary with formatted response and metadata
        """
        # Process the request
        result = self.process_request(
            user_prompt=user_prompt,
            session_id=session_id,
            file_path=file_path,
            tier_override=tier_level,
        )

        # Format output based on requested format
        if format == "json":
            import json
            formatted_output = json.dumps(result, indent=2, default=str)
        elif format == "text":
            formatted_output = result.get("response", str(result))
        else:  # markdown (default)
            formatted_output = self._format_as_markdown(result)

        return {
            "formatted_output": formatted_output,
            "raw_output": result.get("response"),
            "summary": result.get("metadata", {}).get("summary", "Query completed"),
            "duration_seconds": result.get("metadata", {}).get("duration_seconds", 0),
            "total_cost_usd": result.get("metadata", {}).get("total_cost_usd", 0),
            **result,
        }

    def _format_as_markdown(self, result: Dict[str, Any]) -> str:
        """Format result as markdown."""
        metadata = result.get("metadata", {})
        response = result.get("response", "")

        sections = []

        # Add header
        sections.append(f"# Multi-Agent Reasoning Result\n")
        sections.append(f"**Session ID**: `{metadata.get('session_id', 'N/A')}`\n")
        sections.append(f"**Tier**: {metadata.get('tier', 'N/A')}\n")
        sections.append(f"**Duration**: {metadata.get('duration_seconds', 0):.2f} seconds\n")
        sections.append(f"**Cost**: ${metadata.get('total_cost_usd', 0):.4f} USD\n")

        # Add agents used
        agents_used = metadata.get('agents_used', [])
        if agents_used:
            sections.append(f"\n**Agents Used**: {', '.join(agents_used)}\n")

        # Add SMEs used
        smes_used = metadata.get('smes_used', [])
        if smes_used:
            sections.append(f"\n**SME Consultants**: {', '.join(smes_used)}\n")

        # Add response
        sections.append(f"\n---\n\n## Response\n\n{response}")

        return "\n".join(sections)

    # ========================================================================
    # Session Persistence
    # ========================================================================

    def _save_session_state(
        self,
        session: SessionState,
        response: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save session state to disk.

        Args:
            session: Internal session state
            response: Optional response to include in messages
        """
        if not self.persistence:
            return

        try:
            # Convert internal SessionState to persistence SessionState
            from src.session import SessionState as PersistSessionState

            # Build messages list
            messages = []
            for execution in session.agent_executions:
                if execution.output:
                    # Add agent output as message
                    messages.append(ChatMessage(
                        role="assistant",
                        content=str(execution.output),
                        timestamp=datetime.fromtimestamp(execution.start_time),
                        agent_name=execution.agent_name,
                        tier=int(session.current_tier),
                        metadata={"status": execution.status, "tokens": execution.tokens_used},
                    ))

            # Add final response if provided
            if response and "response" in response:
                messages.append(ChatMessage(
                    role="assistant",
                    content=response["response"],
                    timestamp=datetime.now(),
                    agent_name="Orchestrator",
                    tier=int(session.current_tier),
                ))

            # Build agent outputs list
            agent_outputs = []
            for execution in session.agent_executions:
                agent_outputs.append(SessionAgentOutput(
                    agent_name=execution.agent_name,
                    phase="execution",
                    tier=int(session.current_tier),
                    content=str(execution.output) if execution.output else "",
                    structured_data=execution.output if isinstance(execution.output, dict) else None,
                    timestamp=datetime.fromtimestamp(execution.start_time),
                    duration_seconds=execution.end_time - execution.start_time if execution.end_time else 0,
                    token_usage={"total": execution.tokens_used} if execution.tokens_used else None,
                    status=execution.status,
                ))

            # Create persistence session
            persist_session = PersistSessionState(
                session_id=session.session_id,
                created_at=datetime.fromtimestamp(session.start_time),
                updated_at=datetime.now(),
                tier=int(session.current_tier),
                max_budget=session.max_budget_usd,
                messages=messages,
                agent_outputs=agent_outputs,
                active_agents=session.active_smes,
                current_phase="",
                total_tokens=sum(e.tokens_used for e in session.agent_executions),
                total_cost_usd=session.total_cost_usd,
                daily_budget_usd=session.max_budget_usd,
            )

            # Save to disk
            self.persistence.save_session(persist_session)

            # Check if compaction is needed
            if self.enable_auto_compact:
                compaction_result = check_and_compact(persist_session)
                if compaction_result and self.verbose:
                    self._log(
                        f"Session compacted: {compaction_result.original_count} → "
                        f"{compaction_result.compacted_count} messages "
                        f"({compaction_result.reduction_ratio*100:.1f}% reduction)"
                    )

        except Exception as e:
            if self.verbose:
                self._log(f"Error saving session: {e}")

    def load_session(self, session_id: str) -> Optional[SessionState]:
        """
        Load a session from disk.

        Args:
            session_id: Session ID to load

        Returns:
            SessionState if found, None otherwise
        """
        if not self.persistence:
            return None

        try:
            persist_session = self.persistence.load_session(session_id)
            if not persist_session:
                return None

            # Convert to internal SessionState
            session = SessionState(
                session_id=persist_session.session_id,
                user_prompt="",  # Will be set on resume
                max_budget_usd=persist_session.max_budget,
                max_revisions=self.max_revisions,
                max_debate_rounds=self.max_debate_rounds,
            )

            # Restore state
            session.current_tier = TierLevel(persist_session.tier)
            session.total_cost_usd = persist_session.total_cost_usd
            session.active_smes = persist_session.active_agents

            return session

        except Exception as e:
            if self.verbose:
                self._log(f"Error loading session: {e}")
            return None

    # ========================================================================
    # Session Management
    # ========================================================================

    def _create_session(
        self,
        user_prompt: str,
        session_id: Optional[str] = None,
        resume_context: Optional[Dict[str, Any]] = None,
    ) -> SessionState:
        """Create a new session state."""
        if session_id is None:
            session_id = f"session_{int(time.time())}"

        session = SessionState(
            session_id=session_id,
            user_prompt=user_prompt,
            max_budget_usd=self.max_budget_usd,
            max_revisions=self.max_revisions,
            max_debate_rounds=self.max_debate_rounds,
        )

        # Load resume context if provided
        if resume_context:
            session.total_cost_usd = resume_context.get("total_cost_usd", 0)
            session.revision_cycle = resume_context.get("revision_cycle", 0)
            session.active_smes = resume_context.get("active_smes", [])
            session.council_activated = resume_context.get("council_activated", False)

        return session

    def get_session_context(self, session: SessionState) -> Dict[str, Any]:
        """Get context for session resumption."""
        return {
            "session_id": session.session_id,
            "total_cost_usd": session.total_cost_usd,
            "revision_cycle": session.revision_cycle,
            "active_smes": session.active_smes,
            "council_activated": session.council_activated,
            "duration_seconds": session.duration_seconds,
        }

    # ========================================================================
    # Council Consultation
    # ========================================================================

    def _consult_council(self, session: SessionState, input_content: str) -> None:
        """
        Consult the Strategic Council for Tier 3-4 tasks.

        Spawns Domain Council Chair first for SME selection.
        """
        session.council_activated = True

        if self.verbose:
            self._log(f"Consulting Council for Tier {session.current_tier} task")

        # Spawn Domain Council Chair
        chair_result = self._spawn_agent(
            session=session,
            agent_name="Domain Council Chair",
            system_prompt_template="config/agents/council/CLAUDE.md",
            agent_role="chair",
            input_data=input_content,
            model=self.council_model,
        )

        if chair_result["status"] == "success":
            # Process SME selection
            smes = self._extract_sme_selection(chair_result["output"])
            session.active_smes = smes

            if self.verbose:
                self._log(f"Council selected SMEs: {smes}")

        # On Tier 4, also spawn Quality Arbiter and Ethics Advisor
        if session.current_tier == TierLevel.ADVERSARIAL:
            self._spawn_quality_arbiter(session, input_content)

            # Check if Ethics Advisor is needed
            if self._requires_ethics_review(input_content):
                self._spawn_ethics_advisor(session, input_content)

    def _extract_sme_selection(self, council_output: Any) -> List[str]:
        """Extract selected SME personas from Council output."""
        if isinstance(council_output, dict):
            return council_output.get("selected_smes", [])
        return []

    def _spawn_quality_arbiter(self, session: SessionState, input_content: str) -> None:
        """Spawn Quality Arbiter to set acceptance criteria."""
        result = self._spawn_agent(
            session=session,
            agent_name="Quality Arbiter",
            system_prompt_template="config/agents/council/CLAUDE.md",
            agent_role="arbiter",
            input_data=input_content,
            model=self.council_model,
        )
        # Store quality standards for later use
        session.agent_executions[-1].output = result.get("output")

    def _requires_ethics_review(self, input_content: str) -> bool:
        """Check if Ethics Advisor review is needed."""
        ethics_keywords = [
            "personal data", "pii", "medical", "health",
            "legal", "compliance", "government", "security",
        ]
        content_lower = input_content.lower()
        return any(kw in content_lower for kw in ethics_keywords)

    def _spawn_ethics_advisor(self, session: SessionState, input_content: str) -> None:
        """Spawn Ethics & Safety Advisor."""
        result = self._spawn_agent(
            session=session,
            agent_name="Ethics & Safety Advisor",
            system_prompt_template="config/agents/council/CLAUDE.md",
            agent_role="ethics",
            input_data=input_content,
            model=self.council_model,
        )
        session.agent_executions[-1].output = result.get("output")

    # ========================================================================
    # Pipeline Execution
    # ========================================================================

    def _execute_pipeline(
        self,
        pipeline: ExecutionPipeline,
        session: SessionState,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute the pipeline through all phases.

        This is a simplified implementation. The full implementation
        would use the Claude Agent SDK's Task tool to spawn subagents.
        """
        phases = pipeline._get_phases_for_tier()

        for phase in phases:
            # Check budget before each phase
            if session.is_budget_exceeded():
                if self.verbose:
                    self._log("Budget exceeded - stopping pipeline")
                break

            # Get agents for this phase
            agents = pipeline._get_agents_for_phase(phase)

            # Execute agents in this phase
            for agent_name in agents:
                result = self._spawn_agent(
                    session=session,
                    agent_name=agent_name,
                    system_prompt_template=f"config/agents/{agent_name.lower()}/CLAUDE.md",
                    input_data=context["user_prompt"],
                    model=self._get_model_for_agent(agent_name),
                )

                # Track execution
                session.agent_executions.append(AgentExecution(
                    agent_name=agent_name,
                    start_time=time.time(),
                    end_time=time.time(),
                    status=result["status"],
                    output=result.get("output"),
                    error=result.get("error"),
                    tokens_used=result.get("tokens_used", 0),
                    cost_usd=result.get("cost_usd", 0.0),
                ))

                # Check for escalation
                if result.get("escalation_needed"):
                    self._handle_escalation(session, result)

            # Handle verdict matrix for Phase 6
            if phase == Phase.PHASE_6_REVIEW:
                action = self._evaluate_verdict(session)
                if action == MatrixAction.EXECUTOR_REVISE:
                    # Continue to Phase 7
                    continue
                elif action in [MatrixAction.RESEARCHER_REVERIFY, MatrixAction.FULL_REGENERATION]:
                    # Loop back
                    pass

            # Handle debate if triggered
            if self._should_trigger_debate(session, phase):
                self._conduct_debate(session, context)

        return pipeline.state

    # ========================================================================
    # Agent Spawning
    # ========================================================================

    def _spawn_agent(
        self,
        session: SessionState,
        agent_name: str,
        system_prompt_template: str,
        agent_role: str = "",
        input_data: Any = None,
        model: str = None,
    ) -> Dict[str, Any]:
        """
        Spawn a subagent using the Claude Agent SDK.

        This is a placeholder implementation. The full implementation
        would use the SDK's Task tool or subagent API.
        """
        # Load system prompt
        system_prompt = self._load_system_prompt(system_prompt_template, agent_role)

        # Determine model
        if model is None:
            model = self.operational_model

        # In a real implementation, this would call:
        # result = agent.task(
        #     name=agent_name,
        #     system_prompt=system_prompt,
        #     input_data=input_data,
        #     model=model,
        #     max_turns=self._get_max_turns(agent_name),
        # )

        # Placeholder return
        return {
            "status": "success",
            "output": f"Output from {agent_name}",
            "tokens_used": 1000,
            "cost_usd": 0.01,
        }

    def _load_system_prompt(
        self,
        template_path: str,
        agent_role: str = ""
    ) -> str:
        """Load system prompt from template file."""
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Extract role-specific section if agent_role is specified
            if agent_role:
                role_marker = f"# {agent_role.title()}"
                sections = content.split(f"\n{role_marker}")
                if len(sections) > 1:
                    # Get the role section (until next marker or end)
                    role_section = sections[1].split("\n# ")[0]
                    return role_section.strip()

            return content
        except FileNotFoundError:
            # Return default prompt if file not found
            return f"You are the {agent_role or 'agent'}."

    def _get_model_for_agent(self, agent_name: str) -> str:
        """
        Get the model to use for an agent.

        Now uses centralized settings for provider-agnostic model selection.
        """
        # Normalize agent name to match settings keys
        normalized_name = agent_name.lower().replace(" ", "_").replace("-", "_")

        # Use centralized settings
        return get_model_for_agent(normalized_name)

    def _get_max_turns(self, agent_name: str) -> int:
        """Get max turns for an agent."""
        if agent_name == "Orchestrator":
            return self.max_turns_orchestrator
        elif agent_name == "Executor":
            return self.max_turns_executor
        else:
            return self.max_turns_subagent

    # ========================================================================
    # Verdict & Debate
    # ========================================================================

    def _evaluate_verdict(self, session: SessionState) -> MatrixAction:
        """Evaluate verdict matrix from review agent results."""
        # Extract verdicts from Verifier and Critic results
        verifier_verdict = "PASS"
        critic_verdict = "PASS"

        for execution in session.agent_executions:
            if execution.agent_name == "Verifier":
                verifier_verdict = self._parse_verdict(execution.output)
            elif execution.agent_name == "Critic":
                critic_verdict = self._parse_verdict(execution.output)

        outcome = evaluate_verdict_matrix(
            verifier_verdict=verifier_verdict,
            critic_verdict=critic_verdict,
            revision_cycle=session.revision_cycle,
            max_revisions=session.max_revisions,
            tier_level=session.current_tier,
        )

        return outcome.action

    def _parse_verdict(self, output: Any) -> str:
        """Parse verdict from agent output."""
        if isinstance(output, dict):
            return output.get("verdict", "PASS").upper()
        return "PASS"

    def _should_trigger_debate(self, session: SessionState, phase: Phase) -> bool:
        """Check if debate should be triggered."""
        if session.current_tier >= TierLevel.ADVERSARIAL:
            return True

        # Check for disagreement in review phase
        if phase == Phase.PHASE_6_REVIEW:
            verifier_result = next(
                (e for e in session.agent_executions if e.agent_name == "Verifier"),
                None
            )
            critic_result = next(
                (e for e in session.agent_executions if e.agent_name == "Critic"),
                None
            )

            if verifier_result and critic_result:
                v_verdict = self._parse_verdict(verifier_result.output)
                c_verdict = self._parse_verdict(critic_result.output)
                return v_verdict != c_verdict

        return False

    def _conduct_debate(self, session: SessionState, context: Dict[str, Any]) -> None:
        """Conduct a debate session."""
        protocol = DebateProtocol(
            max_rounds=session.max_debate_rounds,
            consensus_threshold=0.8,
        )

        # Add participants
        protocol.add_participant("Executor")
        protocol.add_participant("Critic")
        protocol.add_participant("Verifier")

        # Add SMEs
        for sme in session.active_smes:
            protocol.add_sme_participant(sme)

        # Conduct debate rounds (simplified)
        for _ in range(session.max_debate_rounds):
            # In real implementation, would spawn agents for debate
            session.debate_rounds += 1

            # Check if consensus reached
            # protocol.conduct_round(...)
            # if not protocol.should_continue_debate(...):
            #     break

        outcome = protocol.get_outcome()

        if self.verbose:
            self._log(f"Debate outcome: {outcome.consensus_level}")

    # ========================================================================
    # Escalation
    # ========================================================================

    def _handle_escalation(
        self,
        session: SessionState,
        agent_result: Dict[str, Any]
    ) -> None:
        """Handle mid-execution escalation."""
        if self.verbose:
            self._log(f"Escalation requested by agent")

        session.escalation_history.append({
            "timestamp": time.time(),
            "tier": session.current_tier,
            "reason": agent_result.get("escalation_reason"),
        })

        # Re-evaluate tier
        new_tier = get_escalated_tier(session.current_tier)

        if new_tier != session.current_tier:
            session.current_tier = new_tier
            if self.verbose:
                self._log(f"Escalated to Tier {new_tier}")

            # Activate Council if needed
            if new_tier >= TierLevel.DEEP and not session.council_activated:
                self._consult_council(session, session.user_prompt)

    # ========================================================================
    # Response Generation
    # ========================================================================

    def _generate_final_response(
        self,
        session: SessionState,
        pipeline_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate the final response for the user."""
        # Collect outputs from agents
        outputs = []
        for execution in session.agent_executions:
            if execution.status == "complete" and execution.output:
                outputs.append(execution.output)

        # Format final response
        response = {
            "response": self._format_response(outputs, session),
            "session_id": session.session_id,
            "metadata": {
                "tier": int(session.current_tier),
                "tier_reasoning": session.tier_classification.reasoning if session.tier_classification else "",
                "agents_used": [e.agent_name for e in session.agent_executions],
                "smes_used": session.active_smes,
                "duration_seconds": round(session.duration_seconds, 2),
                "total_cost_usd": round(session.total_cost_usd, 4),
                "revision_cycles": session.revision_cycle,
                "debate_rounds": session.debate_rounds,
            },
        }

        return response

    def _format_response(self, outputs: List[Any], session: SessionState) -> str:
        """Format outputs into a user-friendly response."""
        # In a real implementation, this would synthesize the outputs
        # into a coherent response using the Formatter agent
        if not outputs:
            return "I apologize, but I wasn't able to generate a response."

        # Simple concatenation for now
        parts = []
        for output in outputs:
            if isinstance(output, str):
                parts.append(output)
            elif isinstance(output, dict):
                parts.append(output.get("content", str(output)))

        return "\n\n".join(parts)

    # ========================================================================
    # Input/Output Utilities
    # ========================================================================

    def _load_input_content(
        self,
        user_prompt: str,
        file_path: Optional[str] = None
    ) -> str:
        """Load and combine input content."""
        content = user_prompt

        if file_path:
            try:
                # In real implementation, would read and process the file
                content += f"\n\n[File: {file_path}]"
            except Exception as e:
                if self.verbose:
                    self._log(f"Error loading file: {e}")

        return content

    def _create_override_classification(self, tier: int) -> TierClassification:
        """Create a classification from tier override."""
        from src.core.complexity import TierClassification as TC

        tier_enum = TierLevel(tier)
        config = {
            TierLevel.DIRECT: {"agent_count": 3, "requires_council": False, "requires_smes": False},
            TierLevel.STANDARD: {"agent_count": 7, "requires_council": False, "requires_smes": False},
            TierLevel.DEEP: {"agent_count": 12, "requires_council": True, "requires_smes": True},
            TierLevel.ADVERSARIAL: {"agent_count": 18, "requires_council": True, "requires_smes": True},
        }

        cfg = config[tier_enum]

        return TC(
            tier=tier_enum,
            reasoning=f"Manual override to Tier {tier}",
            confidence=1.0,
            estimated_agents=cfg["agent_count"],
            requires_council=cfg["requires_council"],
            requires_smes=cfg["requires_smes"],
            suggested_sme_count=3 if cfg["requires_smes"] else 0,
        )

    def _budget_exceeded_response(self, session: SessionState) -> Dict[str, Any]:
        """Generate response for budget exceeded."""
        return {
            "response": (
                f"I apologize, but the session budget of ${session.max_budget_usd:.2f} "
                f"has been exceeded (${session.total_cost_usd:.4f} used). "
                "Please provide a new session with a higher budget if needed."
            ),
            "session_id": session.session_id,
            "metadata": {
                "error": "budget_exceeded",
                "total_cost_usd": session.total_cost_usd,
                "max_budget_usd": session.max_budget_usd,
            },
        }

    def _error_response(self, session: SessionState, error: str) -> Dict[str, Any]:
        """Generate error response."""
        return {
            "response": (
                f"I apologize, but an error occurred while processing your request: {error}"
            ),
            "session_id": session.session_id,
            "metadata": {
                "error": error,
                "duration_seconds": session.duration_seconds,
            },
        }

    def _log(self, message: str) -> None:
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            print(f"[Orchestrator] {message}")


# =============================================================================
# Convenience Functions
# =============================================================================

def create_orchestrator(
    api_key: Optional[str] = None,
    max_budget_usd: float = 5.0,
    verbose: bool = False,
) -> OrchestratorAgent:
    """
    Create a configured Orchestrator agent.

    Args:
        api_key: Anthropic API key
        max_budget_usd: Maximum session budget
        verbose: Enable verbose logging

    Returns:
        Configured OrchestratorAgent instance
    """
    return OrchestratorAgent(
        api_key=api_key,
        max_budget_usd=max_budget_usd,
        verbose=verbose,
    )
