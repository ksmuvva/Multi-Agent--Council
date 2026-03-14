"""
Pipeline Orchestration Module

Implements the eight-phase execution pipeline for the Multi-Agent
Reasoning System. Uses a state-machine approach for non-linear phase
transitions (revision loops, re-verification, full regeneration).

Supports parallel agent execution within phases via asyncio.
"""

import asyncio
from enum import Enum
from typing import List, Optional, Dict, Any, Callable
from pydantic import BaseModel, Field, ConfigDict
from dataclasses import dataclass

from .complexity import TierLevel, TierClassification
from .verdict import MatrixAction, Verdict, evaluate_verdict_matrix
from .debate import DebateProtocol, ConsensusLevel
from src.utils.logging import get_logger


class Phase(str, Enum):
    """Pipeline phases."""
    PHASE_1_TASK_INTELLIGENCE = "Phase 1: Task Intelligence"
    PHASE_2_COUNCIL_CONSULTATION = "Phase 2: Council Consultation"
    PHASE_3_PLANNING = "Phase 3: Planning"
    PHASE_4_RESEARCH = "Phase 4: Research"
    PHASE_5_SOLUTION_GENERATION = "Phase 5: Solution Generation"
    PHASE_6_REVIEW = "Phase 6: Review"
    PHASE_7_REVISION = "Phase 7: Revision"
    PHASE_8_FINAL_REVIEW_FORMATTING = "Phase 8: Final Review + Formatting"


class PhaseStatus(str, Enum):
    """Status of a pipeline phase."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class AgentResult:
    """Result from an agent execution."""
    agent_name: str
    status: str
    output: Any
    duration_ms: int
    error: Optional[str] = None
    tokens_used: int = 0


@dataclass
class PhaseResult:
    """Result from a pipeline phase."""
    phase: Phase
    status: PhaseStatus
    agent_results: List[AgentResult]
    duration_ms: int
    output: Any = None
    error: Optional[str] = None


class PipelineState(BaseModel):
    """Current state of the pipeline."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "current_phase": "Phase 5: Solution Generation",
                "completed_phases": [
                    "Phase 1: Task Intelligence",
                    "Phase 3: Planning"
                ],
                "tier_level": 3,
                "revision_cycle": 0,
                "debate_rounds": 0,
                "total_cost_usd": 0.45,
                "total_tokens": 12500,
            }
        }
    )

    current_phase: Optional[Phase] = None
    completed_phases: List[Phase] = Field(default_factory=list)
    tier_level: TierLevel = Field(default=TierLevel.STANDARD)
    revision_cycle: int = Field(default=0, ge=0)
    debate_rounds: int = Field(default=0, ge=0)
    total_cost_usd: float = Field(default=0.0, ge=0.0)
    total_tokens: int = Field(default=0, ge=0)
    start_time: Optional[float] = None
    end_time: Optional[float] = None


class ExecutionPipeline:
    """
    Orchestrates the eight-phase execution pipeline.

    Uses a state-machine (while-loop with explicit phase index) instead
    of a for-loop, enabling non-linear transitions for revision cycles,
    re-verification, and full regeneration.

    Phase structure by tier:
    - Tier 1: Phase 5 + Phase 8 only
    - Tier 2: Phase 1, 3, (4 if needed), 5, 6, 8 (skips 2, 4, 7)
    - Tier 3-4: All phases (1-8) with full Council/SME participation
    """

    def __init__(
        self,
        tier_level: TierLevel = TierLevel.STANDARD,
        max_revisions: int = 2,
        max_debate_rounds: int = 2,
        agent_executor: Optional[Callable] = None,
    ):
        """
        Initialize the pipeline.

        Args:
            tier_level: The complexity tier (1-4)
            max_revisions: Maximum revision cycles
            max_debate_rounds: Maximum debate rounds
            agent_executor: Required callable for agent execution. If not
                provided at init, must be passed to run_pipeline().
        """
        self.tier_level = tier_level
        self.max_revisions = max_revisions
        self.max_debate_rounds = max_debate_rounds
        self.state = PipelineState(tier_level=tier_level)
        self.phase_results: Dict[Phase, PhaseResult] = {}
        self.debate_protocol: Optional[DebateProtocol] = None
        self._agent_executor = agent_executor

    # ========================================================================
    # Phase Execution
    # ========================================================================

    def execute_phase(
        self,
        phase: Phase,
        agent_executor: Callable,
        context: Dict[str, Any]
    ) -> PhaseResult:
        """
        Execute a single pipeline phase with parallel agent support.

        Agents within the same phase are executed in parallel using
        asyncio.gather() when possible.

        Args:
            phase: The phase to execute
            agent_executor: Function to execute agents
            context: Execution context

        Returns:
            PhaseResult with execution results
        """
        # Check if phase should be skipped
        if self._should_skip_phase(phase):
            return PhaseResult(
                phase=phase,
                status=PhaseStatus.SKIPPED,
                agent_results=[],
                duration_ms=0
            )

        # Update state
        self.state.current_phase = phase

        # Get agents for this phase (H2 fix: pass context to get review agents)
        agents = self._get_agents_for_phase(phase, context)

        # Execute agents in parallel within the phase (C2 fix)
        agent_results = self._execute_agents_parallel(agents, phase, agent_executor, context)

        # Determine status
        status = self._determine_phase_status(agent_results)

        # Record result (L2 fix: extract ALL agent outputs, not just first)
        phase_result = PhaseResult(
            phase=phase,
            status=status,
            agent_results=agent_results,
            duration_ms=sum(r.duration_ms for r in agent_results),
            output=self._extract_phase_output(agent_results)
        )

        self.phase_results[phase] = phase_result

        if status == PhaseStatus.COMPLETE:
            self.state.completed_phases.append(phase)

        return phase_result

    def _execute_agents_parallel(
        self,
        agents: List[str],
        phase: Phase,
        agent_executor: Callable,
        context: Dict[str, Any],
    ) -> List[AgentResult]:
        """
        Execute agents in parallel using asyncio when multiple agents
        are in the same phase, falling back to sequential for single agents.
        """
        if len(agents) <= 1:
            # Single agent — run directly
            results = []
            for agent_name in agents:
                result = agent_executor(
                    agent_name=agent_name,
                    phase=phase,
                    context=context,
                )
                results.append(result)
            return results

        # Multiple agents — try parallel execution
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Already in async context — run sequentially to avoid nesting
            return self._execute_agents_sequential(agents, phase, agent_executor, context)

        try:
            return asyncio.run(
                self._execute_agents_async(agents, phase, agent_executor, context)
            )
        except RuntimeError:
            # Fallback to sequential if asyncio.run fails
            return self._execute_agents_sequential(agents, phase, agent_executor, context)

    async def _execute_agents_async(
        self,
        agents: List[str],
        phase: Phase,
        agent_executor: Callable,
        context: Dict[str, Any],
    ) -> List[AgentResult]:
        """Execute agents in parallel using asyncio.gather()."""
        loop = asyncio.get_event_loop()

        async def run_agent(agent_name: str) -> AgentResult:
            return await loop.run_in_executor(
                None,
                lambda: agent_executor(
                    agent_name=agent_name,
                    phase=phase,
                    context=context,
                ),
            )

        tasks = [run_agent(name) for name in agents]
        return list(await asyncio.gather(*tasks))

    def _execute_agents_sequential(
        self,
        agents: List[str],
        phase: Phase,
        agent_executor: Callable,
        context: Dict[str, Any],
    ) -> List[AgentResult]:
        """Execute agents sequentially (fallback)."""
        results = []
        for agent_name in agents:
            result = agent_executor(
                agent_name=agent_name,
                phase=phase,
                context=context,
            )
            results.append(result)
        return results

    # ========================================================================
    # Pipeline Flow (H1 fix: state-machine with while-loop)
    # ========================================================================

    def run_pipeline(
        self,
        agent_executor: Optional[Callable] = None,
        initial_context: Optional[Dict[str, Any]] = None
    ) -> PipelineState:
        """
        Run the complete pipeline based on tier level.

        Uses a state-machine (while-loop with explicit phase index) to
        support non-linear transitions: revision loops, re-verification,
        and full regeneration.

        Args:
            agent_executor: Function to execute agents (uses init value if None)
            initial_context: Initial execution context

        Returns:
            Final PipelineState

        Raises:
            ValueError: If no agent_executor is available
        """
        import time

        executor = agent_executor or self._agent_executor
        if executor is None:
            raise ValueError(
                "agent_executor is required. Pass it to __init__() or run_pipeline()."
            )

        context = initial_context or {}
        # Store executor in context so verdict actions can re-invoke phases (M5 fix)
        context["agent_executor"] = executor

        self.state.start_time = time.time()

        # Determine phases to run
        phases = self._get_phases_for_tier()

        # State-machine loop (H1 fix: replaces broken for-loop)
        phase_idx = 0
        while phase_idx < len(phases):
            phase = phases[phase_idx]

            # Execute phase
            result = self.execute_phase(phase, executor, context)

            # Check for errors
            if result.status == PhaseStatus.FAILED:
                if not self._handle_phase_failure(phase, result):
                    break

            # Special handling for Phase 6 (Review) — verdict matrix
            if phase == Phase.PHASE_6_REVIEW and result.status == PhaseStatus.COMPLETE:
                action = self._evaluate_verdict_matrix(result)

                if action == MatrixAction.PROCEED_TO_FORMATTER:
                    # Skip Phase 7, jump to Phase 8
                    phase_8_idx = None
                    for i, p in enumerate(phases):
                        if p == Phase.PHASE_8_FINAL_REVIEW_FORMATTING:
                            phase_8_idx = i
                            break
                    if phase_8_idx is not None:
                        phase_idx = phase_8_idx
                        continue

                elif action == MatrixAction.EXECUTOR_REVISE:
                    if self.state.revision_cycle < self.max_revisions:
                        # Jump back to Phase 5 (Solution Generation)
                        self.state.revision_cycle += 1
                        self._reset_phases_for_revision()
                        phase_5_idx = None
                        for i, p in enumerate(phases):
                            if p == Phase.PHASE_5_SOLUTION_GENERATION:
                                phase_5_idx = i
                                break
                        if phase_5_idx is not None:
                            phase_idx = phase_5_idx
                            continue
                    # else: max revisions reached, fall through to next phase

                elif action in (MatrixAction.RESEARCHER_REVERIFY, MatrixAction.FULL_REGENERATION):
                    self._handle_verdict_action(action, context)
                    # After handling, re-run from Phase 5
                    self.state.revision_cycle += 1
                    phase_5_idx = None
                    for i, p in enumerate(phases):
                        if p == Phase.PHASE_5_SOLUTION_GENERATION:
                            phase_5_idx = i
                            break
                    if phase_5_idx is not None and self.state.revision_cycle <= self.max_revisions:
                        phase_idx = phase_5_idx
                        continue

                elif action == MatrixAction.QUALITY_ARBITER:
                    self._invoke_quality_arbiter(context)
                    # Proceed to formatting after arbiter

            # Normal progression
            phase_idx += 1

        self.state.end_time = time.time()
        return self.state

    def _reset_phases_for_revision(self) -> None:
        """Clear previous Phase 5/6/7 results to allow re-execution."""
        for phase_to_reset in [
            Phase.PHASE_5_SOLUTION_GENERATION,
            Phase.PHASE_6_REVIEW,
            Phase.PHASE_7_REVISION,
        ]:
            if phase_to_reset in self.phase_results:
                del self.phase_results[phase_to_reset]
            if phase_to_reset in self.state.completed_phases:
                self.state.completed_phases.remove(phase_to_reset)

    # ========================================================================
    # Phase Configuration
    # ========================================================================

    def _should_skip_phase(self, phase: Phase) -> bool:
        """Determine if a phase should be skipped based on tier."""
        # Tier 1 skips most phases
        if self.tier_level == TierLevel.DIRECT:
            return phase not in [
                Phase.PHASE_5_SOLUTION_GENERATION,
                Phase.PHASE_8_FINAL_REVIEW_FORMATTING
            ]

        # Tier 2 skips Council, Research, Revision
        if self.tier_level == TierLevel.STANDARD:
            return phase in [
                Phase.PHASE_2_COUNCIL_CONSULTATION,
                Phase.PHASE_4_RESEARCH,
                Phase.PHASE_7_REVISION
            ]

        return False

    def _get_phases_for_tier(self) -> List[Phase]:
        """Get the list of phases to run for the current tier."""
        all_phases = [
            Phase.PHASE_1_TASK_INTELLIGENCE,
            Phase.PHASE_2_COUNCIL_CONSULTATION,
            Phase.PHASE_3_PLANNING,
            Phase.PHASE_4_RESEARCH,
            Phase.PHASE_5_SOLUTION_GENERATION,
            Phase.PHASE_6_REVIEW,
            Phase.PHASE_7_REVISION,
            Phase.PHASE_8_FINAL_REVIEW_FORMATTING,
        ]

        # Filter based on tier
        return [p for p in all_phases if not self._should_skip_phase(p)]

    def _get_agents_for_phase(
        self, phase: Phase, context: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Get the list of agents for a phase (H2 fix: passes context)."""
        phase_agents = {
            Phase.PHASE_1_TASK_INTELLIGENCE: ["Task Analyst"],
            Phase.PHASE_2_COUNCIL_CONSULTATION: self._get_council_agents(),
            Phase.PHASE_3_PLANNING: ["Planner"],
            Phase.PHASE_4_RESEARCH: ["Researcher"],
            Phase.PHASE_5_SOLUTION_GENERATION: ["Executor"],
            Phase.PHASE_6_REVIEW: self._get_review_agents(context),
            Phase.PHASE_7_REVISION: ["Executor"],
            Phase.PHASE_8_FINAL_REVIEW_FORMATTING: ["Reviewer", "Formatter"],
        }
        return phase_agents.get(phase, [])

    def _get_council_agents(self) -> List[str]:
        """Get Council agents based on tier."""
        if self.tier_level == TierLevel.ADVERSARIAL:
            return [
                "Domain Council Chair",
                "Quality Arbiter",
                "Ethics & Safety Advisor"
            ]
        elif self.tier_level >= TierLevel.DEEP:
            return ["Domain Council Chair"]
        return []

    def _get_review_agents(self, context: Optional[Dict[str, Any]] = None) -> List[str]:
        """Get review agents (run in parallel within Phase 6)."""
        agents = ["Verifier", "Critic"]

        # Add Code Reviewer if code was generated in the solution phase
        if context and context.get("code_generated", False):
            agents.append("Code Reviewer")

        return agents

    # ========================================================================
    # Verdict Matrix & Debate
    # ========================================================================

    def _evaluate_verdict_matrix(self, phase_result: PhaseResult) -> MatrixAction:
        """Evaluate verdict matrix based on review results."""
        # Extract verdicts from agent results
        verifier_verdict = Verdict.PASS
        critic_verdict = Verdict.PASS

        for result in phase_result.agent_results:
            if result.agent_name == "Verifier":
                verifier_verdict = self._parse_verdict(result.output)
            elif result.agent_name == "Critic":
                critic_verdict = self._parse_verdict(result.output)

        # Evaluate matrix
        outcome = evaluate_verdict_matrix(
            verifier_verdict=verifier_verdict,
            critic_verdict=critic_verdict,
            revision_cycle=self.state.revision_cycle,
            max_revisions=self.max_revisions,
            tier_level=self.tier_level
        )

        return outcome.action

    def _parse_verdict(self, output: Any) -> Verdict:
        """Parse a verdict from agent output."""
        if isinstance(output, dict):
            verdict_str = output.get("verdict", "PASS").upper()
            return Verdict.PASS if verdict_str == "PASS" else Verdict.FAIL
        return Verdict.PASS

    def _handle_verdict_action(
        self,
        action: MatrixAction,
        context: Dict[str, Any]
    ) -> None:
        """Handle a verdict matrix action (re-verify or regenerate)."""
        logger = get_logger(__name__)
        agent_executor = context.get("agent_executor")

        if not agent_executor:
            raise ValueError(
                "agent_executor missing from context; cannot handle verdict action. "
                "Ensure agent_executor is set via run_pipeline() or __init__()."
            )

        if action == MatrixAction.RESEARCHER_REVERIFY:
            logger.info("Verdict action: RESEARCHER_REVERIFY - re-running research phase on flagged claims")
            flagged_claims = self._extract_flagged_claims()
            context["flagged_claims"] = flagged_claims
            context["reverify_mode"] = True

            if Phase.PHASE_4_RESEARCH in self.phase_results:
                del self.phase_results[Phase.PHASE_4_RESEARCH]
                if Phase.PHASE_4_RESEARCH in self.state.completed_phases:
                    self.state.completed_phases.remove(Phase.PHASE_4_RESEARCH)

            research_result = self.execute_phase(
                Phase.PHASE_4_RESEARCH, agent_executor, context
            )
            logger.info(
                f"Research re-verification complete: status={research_result.status}"
            )

        elif action == MatrixAction.FULL_REGENERATION:
            logger.info("Verdict action: FULL_REGENERATION - resetting to solution generation phase")
            review_result = self.phase_results.get(Phase.PHASE_6_REVIEW)
            if review_result:
                feedback = [
                    r.output for r in review_result.agent_results
                    if r.output and r.status != "error"
                ]
                context["regeneration_feedback"] = feedback
                context["regeneration_cycle"] = self.state.revision_cycle + 1

            self._reset_phases_for_revision()

        elif action == MatrixAction.QUALITY_ARBITER:
            logger.info("Verdict action: QUALITY_ARBITER - invoking arbiter for dispute resolution")
            self._invoke_quality_arbiter(context)

    def _extract_flagged_claims(self) -> List[str]:
        """Extract flagged claims from the review phase results."""
        flagged = []
        review_result = self.phase_results.get(Phase.PHASE_6_REVIEW)
        if review_result:
            for agent_result in review_result.agent_results:
                if agent_result.output and isinstance(agent_result.output, dict):
                    claims = agent_result.output.get("flagged_claims", [])
                    if claims:
                        flagged.extend(claims)
                    issues = agent_result.output.get("issues", [])
                    if issues:
                        flagged.extend(issues)
        return flagged

    # ========================================================================
    # Debate Protocol
    # ========================================================================

    def initiate_debate(self, context: Dict[str, Any]) -> DebateProtocol:
        """Initiate a debate session."""
        self.debate_protocol = DebateProtocol(
            max_rounds=self.max_debate_rounds,
            consensus_threshold=0.8
        )

        self.debate_protocol.add_participant("Executor")
        self.debate_protocol.add_participant("Critic")
        self.debate_protocol.add_participant("Verifier")

        smes = context.get("active_smes", [])
        for sme in smes:
            self.debate_protocol.add_sme_participant(sme)

        return self.debate_protocol

    # ========================================================================
    # Utilities
    # ========================================================================

    def _determine_phase_status(self, agent_results: List[AgentResult]) -> PhaseStatus:
        """Determine phase status from agent results."""
        for result in agent_results:
            if result.status == "error":
                if result.agent_name in ["Verifier", "Domain Council Chair"]:
                    return PhaseStatus.FAILED

        for result in agent_results:
            if result.status == "success":
                return PhaseStatus.COMPLETE

        return PhaseStatus.FAILED

    def _extract_phase_output(self, agent_results: List[AgentResult]) -> Any:
        """
        Extract meaningful output from agent results.

        L2 fix: Returns all agent outputs as a dict keyed by agent name
        when multiple agents produced output, or a single output for
        single-agent phases.
        """
        outputs = {r.agent_name: r.output for r in agent_results if r.output}
        if len(outputs) == 0:
            return None
        if len(outputs) == 1:
            return next(iter(outputs.values()))
        return outputs

    def _handle_phase_failure(
        self,
        phase: Phase,
        result: PhaseResult
    ) -> bool:
        """Handle a phase failure. Returns True if should continue."""
        if phase in [Phase.PHASE_4_RESEARCH]:
            return True
        return False

    def _invoke_quality_arbiter(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Invoke Quality Arbiter for dispute resolution."""
        logger = get_logger(__name__)

        review_result = self.phase_results.get(Phase.PHASE_6_REVIEW)
        critic_output = None
        verifier_output = None

        if review_result:
            for agent_result in review_result.agent_results:
                if agent_result.agent_name == "Critic":
                    critic_output = agent_result.output
                elif agent_result.agent_name == "Verifier":
                    verifier_output = agent_result.output

        arbitration_request = {
            "type": "quality_arbitration",
            "verifier_output": verifier_output,
            "critic_output": critic_output,
            "revision_cycle": self.state.revision_cycle,
            "tier_level": str(self.tier_level),
            "user_prompt": context.get("user_prompt", ""),
            "solution_output": self.phase_results.get(
                Phase.PHASE_5_SOLUTION_GENERATION, PhaseResult(
                    phase=Phase.PHASE_5_SOLUTION_GENERATION,
                    status=PhaseStatus.PENDING,
                    agent_results=[],
                    duration_ms=0,
                )
            ).output,
        }

        agent_executor = context.get("agent_executor")
        if agent_executor:
            logger.info("Invoking Quality Arbiter with conflict data")
            arbiter_result = agent_executor(
                agent_name="Quality Arbiter",
                phase=Phase.PHASE_6_REVIEW,
                context={**context, "arbitration_request": arbitration_request},
            )
            logger.info(f"Quality Arbiter decision: status={arbiter_result.status}")

            context["arbiter_decision"] = arbiter_result.output
            context["arbiter_invoked"] = True
            return arbiter_result.output
        else:
            raise ValueError(
                "agent_executor missing from context; cannot invoke Quality Arbiter."
            )


class PipelineBuilder:
    """Builder for creating configured pipelines."""

    @staticmethod
    def for_tier(tier: TierLevel, agent_executor: Optional[Callable] = None) -> ExecutionPipeline:
        """Create a pipeline for a specific tier."""
        return ExecutionPipeline(tier_level=tier, agent_executor=agent_executor)

    @staticmethod
    def from_classification(
        classification: TierClassification,
        agent_executor: Optional[Callable] = None,
    ) -> ExecutionPipeline:
        """Create a pipeline from a TierClassification."""
        return ExecutionPipeline(tier_level=classification.tier, agent_executor=agent_executor)


# =============================================================================
# Pipeline Execution Functions
# =============================================================================

def create_execution_context(
    user_prompt: str,
    tier_classification: TierClassification,
    session_id: Optional[str] = None,
    additional_context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create the execution context for a pipeline run.

    Args:
        user_prompt: The user's original request
        tier_classification: The tier classification result
        session_id: Optional session ID for persistence
        additional_context: Any additional context

    Returns:
        Complete execution context
    """
    context = {
        "user_prompt": user_prompt,
        "tier": tier_classification.tier,
        "tier_reasoning": tier_classification.reasoning,
        "requires_council": tier_classification.requires_council,
        "requires_smes": tier_classification.requires_smes,
        "suggested_sme_count": tier_classification.suggested_sme_count,
        "session_id": session_id,
        "start_time": None,  # Will be set by pipeline
    }

    if additional_context:
        context.update(additional_context)

    return context


def estimate_pipeline_duration(tier: TierLevel) -> Dict[str, int]:
    """
    Estimate pipeline duration by tier.

    Args:
        tier: The tier level

    Returns:
        Dictionary with min, max, and estimated duration in seconds
    """
    estimates = {
        TierLevel.DIRECT: {"min": 10, "max": 30, "estimated": 15},
        TierLevel.STANDARD: {"min": 30, "max": 90, "estimated": 60},
        TierLevel.DEEP: {"min": 90, "max": 300, "estimated": 180},
        TierLevel.ADVERSARIAL: {"min": 180, "max": 600, "estimated": 360},
    }
    return estimates.get(tier, estimates[TierLevel.STANDARD])
