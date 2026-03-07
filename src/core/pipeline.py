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
    start_time: Optional[str] = None
    end_time: Optional[str] = None

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
        if self.tier_level >= TierLevel.DEEP:
            return ["Domain Council Chair"]
        elif self.tier_level == TierLevel.ADVERSARIAL:
            return [
                "Domain Council Chair",
                "Quality Arbiter",
                "Ethics & Safety Advisor"
            ]
        return []

    def _get_review_agents(self) -> List[str]:
        """Get review agents (can run in parallel)."""
        agents = ["Verifier", "Critic"]

        # Add Code Reviewer if code was generated
        # (This would be determined from context)
        # agents.append("Code Reviewer")

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
        if action == MatrixAction.RESEARCHER_REVERIFY:
            # Loop back to research
            pass
        elif action == MatrixAction.FULL_REGENERATION:
            # Loop back to solution generation
            pass
        elif action == MatrixAction.QUALITY_ARBITER:
            # Invoke Quality Arbiter
            self._invoke_quality_arbiter(context)

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

    def _invoke_quality_arbiter(self, context: Dict[str, Any]) -> None:
        """Invoke Quality Arbiter for dispute resolution."""
        # This would trigger the arbiter agent
        context["require_arbiter"] = True


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
