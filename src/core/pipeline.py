"""
Pipeline Orchestration Module

Implements the eight-phase execution pipeline for the Multi-Agent
Reasoning System.
"""

from enum import Enum
from typing import List, Optional, Dict, Any, Callable
from pydantic import BaseModel, Field
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
    current_phase: Optional[Phase] = None
    completed_phases: List[Phase] = Field(default_factory=list)
    tier_level: TierLevel = Field(default=TierLevel.STANDARD)
    revision_cycle: int = Field(default=0, ge=0)
    debate_rounds: int = Field(default=0, ge=0)
    total_cost_usd: float = Field(default=0.0, ge=0.0)
    total_tokens: int = Field(default=0, ge=0)
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    class Config:
        json_schema_extra = {
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
                "total_tokens": 12500
            }
        }


class ExecutionPipeline:
    """
    Orchestrates the eight-phase execution pipeline.

    Phase structure by tier:
    - Tier 1: Phase 5 + Phase 8 only
    - Tier 2: Phase 1, 3, (4 if needed), 5, 6, 8 (skips 2, 4, 7)
    - Tier 3-4: All phases (1-8) with full Council/SME participation
    """

    def __init__(
        self,
        tier_level: TierLevel = TierLevel.STANDARD,
        max_revisions: int = 2,
        max_debate_rounds: int = 2
    ):
        """
        Initialize the pipeline.

        Args:
            tier_level: The complexity tier (1-4)
            max_revisions: Maximum revision cycles
            max_debate_rounds: Maximum debate rounds
        """
        self.tier_level = tier_level
        self.max_revisions = max_revisions
        self.max_debate_rounds = max_debate_rounds
        self.state = PipelineState(tier_level=tier_level)
        self.phase_results: Dict[Phase, PhaseResult] = {}
        self.debate_protocol: Optional[DebateProtocol] = None

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
        Execute a single pipeline phase.

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

        # Get agents for this phase
        agents = self._get_agents_for_phase(phase)

        # Execute agents
        agent_results = []
        for agent_name in agents:
            result = agent_executor(
                agent_name=agent_name,
                phase=phase,
                context=context
            )
            agent_results.append(result)

        # Determine status
        status = self._determine_phase_status(agent_results)

        # Record result
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

    # ========================================================================
    # Pipeline Flow
    # ========================================================================

    def run_pipeline(
        self,
        agent_executor: Callable,
        initial_context: Dict[str, Any]
    ) -> PipelineState:
        """
        Run the complete pipeline based on tier level.

        Args:
            agent_executor: Function to execute agents
            initial_context: Initial execution context

        Returns:
            Final PipelineState
        """
        import time
        self.state.start_time = time.time()

        # Determine phases to run
        phases = self._get_phases_for_tier()

        # Run phases sequentially
        for phase in phases:
            # Execute phase
            result = self.execute_phase(phase, agent_executor, initial_context)

            # Check for errors
            if result.status == PhaseStatus.FAILED:
                # Handle failure based on phase
                if not self._handle_phase_failure(phase, result):
                    break

            # Special handling for Phase 6 (Review) - verdict matrix
            if phase == Phase.PHASE_6_REVIEW:
                action = self._evaluate_verdict_matrix(result)
                if action == MatrixAction.EXECUTOR_REVISE:
                    # Go to Phase 7
                    continue
                elif action in [MatrixAction.RESEARCHER_REVERIFY, MatrixAction.FULL_REGENERATION]:
                    # Loop back to earlier phase
                    self._handle_verdict_action(action, initial_context)
                    continue

            # Special handling for Phase 7 (Revision) - loop back
            if phase == Phase.PHASE_7_REVISION:
                if self.state.revision_cycle < self.max_revisions:
                    # Loop back to Phase 5 or 6
                    self.state.revision_cycle += 1
                    continue
                else:
                    # Max revisions reached - proceed to Phase 8
                    pass

        self.state.end_time = time.time()
        return self.state

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

    def _get_agents_for_phase(self, phase: Phase) -> List[str]:
        """Get the list of agents for a phase."""
        phase_agents = {
            Phase.PHASE_1_TASK_INTELLIGENCE: ["Task Analyst"],
            Phase.PHASE_2_COUNCIL_CONSULTATION: self._get_council_agents(),
            Phase.PHASE_3_PLANNING: ["Planner"],
            Phase.PHASE_4_RESEARCH: ["Researcher"],
            Phase.PHASE_5_SOLUTION_GENERATION: ["Executor"],
            Phase.PHASE_6_REVIEW: self._get_review_agents(),
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
        """Get review agents (can run in parallel)."""
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
                # Parse verdict from output
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
        """Handle a verdict matrix action."""
        logger = get_logger(__name__)

        if action == MatrixAction.RESEARCHER_REVERIFY:
            logger.info("Verdict action: RESEARCHER_REVERIFY - re-running research phase on flagged claims")
            # Collect flagged claims from the review phase results
            flagged_claims = self._extract_flagged_claims()
            context["flagged_claims"] = flagged_claims
            context["reverify_mode"] = True

            # Re-execute the research phase to gather new evidence
            if Phase.PHASE_4_RESEARCH in self.phase_results:
                # Remove previous research result so it can be re-run
                del self.phase_results[Phase.PHASE_4_RESEARCH]
                if Phase.PHASE_4_RESEARCH in self.state.completed_phases:
                    self.state.completed_phases.remove(Phase.PHASE_4_RESEARCH)

            # Re-run research phase with the agent_executor stored in context
            agent_executor = context.get("agent_executor")
            if agent_executor:
                research_result = self.execute_phase(
                    Phase.PHASE_4_RESEARCH, agent_executor, context
                )
                logger.info(
                    f"Research re-verification complete: status={research_result.status}"
                )
            else:
                logger.warning("No agent_executor in context; cannot re-run research phase")

        elif action == MatrixAction.FULL_REGENERATION:
            logger.info("Verdict action: FULL_REGENERATION - resetting to solution generation phase")
            # Incorporate feedback from the failed review into context
            review_result = self.phase_results.get(Phase.PHASE_6_REVIEW)
            if review_result:
                feedback = [
                    r.output for r in review_result.agent_results
                    if r.output and r.status != "error"
                ]
                context["regeneration_feedback"] = feedback
                context["regeneration_cycle"] = self.state.revision_cycle + 1

            # Remove previous solution and review results to allow re-execution
            for phase_to_reset in [Phase.PHASE_5_SOLUTION_GENERATION, Phase.PHASE_6_REVIEW]:
                if phase_to_reset in self.phase_results:
                    del self.phase_results[phase_to_reset]
                if phase_to_reset in self.state.completed_phases:
                    self.state.completed_phases.remove(phase_to_reset)

            # Re-run solution generation
            agent_executor = context.get("agent_executor")
            if agent_executor:
                gen_result = self.execute_phase(
                    Phase.PHASE_5_SOLUTION_GENERATION, agent_executor, context
                )
                logger.info(
                    f"Full regeneration complete: status={gen_result.status}"
                )
            else:
                logger.warning("No agent_executor in context; cannot regenerate solution")

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
                    # Collect flagged claims from verifier and critic outputs
                    claims = agent_result.output.get("flagged_claims", [])
                    if claims:
                        flagged.extend(claims)
                    # Also check for issues/challenges as flagged items
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

        # Add participants
        self.debate_protocol.add_participant("Executor")
        self.debate_protocol.add_participant("Critic")
        self.debate_protocol.add_participant("Verifier")

        # Add SMEs if available
        smes = context.get("active_smes", [])
        for sme in smes:
            self.debate_protocol.add_sme_participant(sme)

        return self.debate_protocol

    # ========================================================================
    # Utilities
    # ========================================================================

    def _determine_phase_status(self, agent_results: List[AgentResult]) -> PhaseStatus:
        """Determine phase status from agent results."""
        # Check for critical failures
        for result in agent_results:
            if result.status == "error":
                # Check if critical agent
                if result.agent_name in ["Verifier", "Domain Council Chair"]:
                    return PhaseStatus.FAILED

        # All others - complete if at least one succeeded
        for result in agent_results:
            if result.status == "success":
                return PhaseStatus.COMPLETE

        return PhaseStatus.FAILED

    def _extract_phase_output(self, agent_results: List[AgentResult]) -> Any:
        """Extract meaningful output from agent results."""
        outputs = [r.output for r in agent_results if r.output]
        return outputs[0] if outputs else None

    def _handle_phase_failure(
        self,
        phase: Phase,
        result: PhaseResult
    ) -> bool:
        """Handle a phase failure. Returns True if should continue."""
        # Non-critical phases can be skipped
        if phase in [Phase.PHASE_4_RESEARCH]:
            return True

        # Critical failures stop the pipeline
        return False

    def _invoke_quality_arbiter(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Invoke Quality Arbiter for dispute resolution.

        Collects critic and reviewer outputs, formulates the arbitration
        request, and invokes the arbiter (council chair) with the conflict data.

        Returns:
            The arbiter's decision dict, or None if invocation is not possible.
        """
        logger = get_logger(__name__)

        # Collect critic and verifier/reviewer outputs from the review phase
        review_result = self.phase_results.get(Phase.PHASE_6_REVIEW)
        critic_output = None
        verifier_output = None

        if review_result:
            for agent_result in review_result.agent_results:
                if agent_result.agent_name == "Critic":
                    critic_output = agent_result.output
                elif agent_result.agent_name == "Verifier":
                    verifier_output = agent_result.output

        # Formulate the arbitration request with full conflict data
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

        # Invoke arbiter via agent_executor if available
        agent_executor = context.get("agent_executor")
        if agent_executor:
            logger.info("Invoking Quality Arbiter with conflict data")
            arbiter_result = agent_executor(
                agent_name="Quality Arbiter",
                phase=Phase.PHASE_6_REVIEW,
                context={**context, "arbitration_request": arbitration_request},
            )
            logger.info(f"Quality Arbiter decision: status={arbiter_result.status}")

            # Store the arbiter's decision in context for downstream use
            context["arbiter_decision"] = arbiter_result.output
            context["arbiter_invoked"] = True
            return arbiter_result.output
        else:
            # Fallback: flag that arbiter is required for external handling
            logger.warning("No agent_executor in context; flagging arbiter requirement")
            context["require_arbiter"] = True
            context["arbitration_request"] = arbitration_request
            return None


class PipelineBuilder:
    """Builder for creating configured pipelines."""

    @staticmethod
    def for_tier(tier: TierLevel) -> ExecutionPipeline:
        """Create a pipeline for a specific tier."""
        return ExecutionPipeline(tier_level=tier)

    @staticmethod
    def from_classification(classification: TierClassification) -> ExecutionPipeline:
        """Create a pipeline from a TierClassification."""
        return ExecutionPipeline(tier_level=classification.tier)


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
