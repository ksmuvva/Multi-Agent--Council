"""
Verdict Matrix Module

Implements the verdict matrix logic for determining next actions
based on Verifier and Critic outcomes.
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class Verdict(str, Enum):
    """Pass/fail verdict from agents."""
    PASS = "PASS"
    FAIL = "FAIL"


class MatrixAction(str, Enum):
    """Actions triggered by verdict matrix."""
    PROCEED_TO_FORMATTER = "proceed_to_formatter"
    EXECUTOR_REVISE = "executor_revise"
    RESEARCHER_REVERIFY = "researcher_reverify"
    FULL_REGENERATION = "full_regeneration"
    QUALITY_ARBITER = "quality_arbiter"


class MatrixOutcome(BaseModel):
    """Result of verdict matrix evaluation."""
    verifier_verdict: Verdict = Field(..., description="Verifier's verdict")
    critic_verdict: Verdict = Field(..., description="Critic's verdict")
    action: MatrixAction = Field(..., description="Action to take")
    reason: str = Field(..., description="Reasoning for this action")
    revision_cycle: int = Field(..., ge=0, description="Current revision cycle")
    can_retry: bool = Field(..., description="Whether another retry is allowed")
    max_revisions: int = Field(default=2, description="Maximum revision cycles")

    class Config:
        json_schema_extra = {
            "example": {
                "verifier_verdict": "PASS",
                "critic_verdict": "FAIL",
                "action": "executor_revise",
                "reason": "Verifier passed but Critic found issues - Executor should revise",
                "revision_cycle": 1,
                "can_retry": True,
                "max_revisions": 2
            }
        }


# =============================================================================
# Verdict Matrix
# =============================================================================

# The verdict matrix defines the action based on Verifier × Critic results
VERDICT_MATRIX = {
    # (Verifier, Critic) -> Action
    (Verdict.PASS, Verdict.PASS): MatrixAction.PROCEED_TO_FORMATTER,
    (Verdict.PASS, Verdict.FAIL): MatrixAction.EXECUTOR_REVISE,
    (Verdict.FAIL, Verdict.PASS): MatrixAction.RESEARCHER_REVERIFY,
    (Verdict.FAIL, Verdict.FAIL): MatrixAction.FULL_REGENERATION,
}


# Alias for backwards-compatible imports
VerdictAction = MatrixAction


class VerdictMatrix:
    """
    Class-based wrapper for the verdict matrix.

    Provides an object-oriented interface for evaluating the verdict matrix.
    """

    def get_action(self, verifier_pass: bool, critic_pass: bool) -> MatrixAction:
        """Get the action for a given pair of verdicts."""
        v = Verdict.PASS if verifier_pass else Verdict.FAIL
        c = Verdict.PASS if critic_pass else Verdict.FAIL
        return VERDICT_MATRIX[(v, c)]


def evaluate_verdict_matrix(
    verifier_verdict: Verdict,
    critic_verdict: Verdict,
    revision_cycle: int = 0,
    max_revisions: int = 2,
    tier_level: int = 2
) -> MatrixOutcome:
    """
    Evaluate the verdict matrix to determine next action.

    Matrix logic:
    - PASS + PASS → Proceed to Formatter
    - PASS + FAIL → Executor revises (Phase 7)
    - FAIL + PASS → Researcher re-verifies
    - FAIL + FAIL → Full re-generation from Phase 5

    After 2 revision cycles, invoke Quality Arbiter on Tier 4.

    Args:
        verifier_verdict: The Verifier's verdict
        critic_verdict: The Critic's verdict
        revision_cycle: Current revision cycle (0-based)
        max_revisions: Maximum allowed revision cycles
        tier_level: Current tier level (for Quality Arbiter decision)

    Returns:
        MatrixOutcome with action and reasoning
    """
    # Get base action from matrix
    action = VERDICT_MATRIX.get((verifier_verdict, critic_verdict))

    # Check if we've exceeded revision cycles
    can_retry = revision_cycle < max_revisions

    # On Tier 4, after max revisions, invoke Quality Arbiter
    if not can_retry and tier_level >= 4:
        action = MatrixAction.QUALITY_ARBITER

    # Build reasoning
    reason_parts = []
    reason_parts.append(f"Verifier: {verifier_verdict.value}, Critic: {critic_verdict.value}")

    if action == MatrixAction.PROCEED_TO_FORMATTER:
        reason_parts.append("Both agents passed - ready for final formatting")
    elif action == MatrixAction.EXECUTOR_REVISE:
        reason_parts.append("Critic found issues - Executor should revise")
        if not can_retry:
            reason_parts.append(f"Revision limit ({max_revisions}) reached")
    elif action == MatrixAction.RESEARCHER_REVERIFY:
        reason_parts.append("Verifier failed claims - Researcher should re-verify")
    elif action == MatrixAction.FULL_REGENERATION:
        reason_parts.append("Both agents failed - full re-generation needed")
    elif action == MatrixAction.QUALITY_ARBITER:
        reason_parts.append("Revision limit exceeded - invoking Quality Arbiter")

    return MatrixOutcome(
        verifier_verdict=verifier_verdict,
        critic_verdict=critic_verdict,
        action=action,
        reason=". ".join(reason_parts),
        revision_cycle=revision_cycle,
        can_retry=can_retry,
        max_revisions=max_revisions
    )


def get_phase_for_action(action: MatrixAction) -> str:
    """
    Get the pipeline phase for a given matrix action.

    Args:
        action: The matrix action

    Returns:
        Description of the phase to execute
    """
    phase_map = {
        MatrixAction.PROCEED_TO_FORMATTER: "Phase 8: Final Review + Formatting",
        MatrixAction.EXECUTOR_REVISE: "Phase 7: Revision (Executor)",
        MatrixAction.RESEARCHER_REVERIFY: "Phase 4: Research (Re-verification)",
        MatrixAction.FULL_REGENERATION: "Phase 5: Solution Generation (Full redo)",
        MatrixAction.QUALITY_ARBITER: "Quality Arbiter Arbitration",
    }
    return phase_map.get(action, "Unknown phase")


def should_trigger_debate(
    verdict_outcome: MatrixOutcome,
    tier_level: int,
    debate_rounds: int = 0
) -> bool:
    """
    Determine if a debate should be triggered.

    Debates are triggered on:
    - Verifier/Critic disagreement (different verdicts)
    - All Tier 4 tasks
    - After 2 failed revision cycles

    Args:
        verdict_outcome: The verdict matrix outcome
        tier_level: Current tier level
        debate_rounds: Number of debate rounds already completed

    Returns:
        True if debate should be triggered
    """
    # Tier 4 always debates
    if tier_level >= 4:
        return True

    # Disagreement triggers debate
    if verdict_outcome.verifier_verdict != verdict_outcome.critic_verdict:
        return True

    # After max revisions on FAIL/FAIL, debate to break deadlock
    if (not verdict_outcome.can_retry and
        verdict_outcome.action == MatrixAction.FULL_REGENERATION):
        return True

    return False


class DebateConfig(BaseModel):
    """Configuration for a debate session."""
    max_rounds: int = Field(default=2, ge=1, le=5, description="Maximum debate rounds")
    current_round: int = Field(default=0, ge=0, description="Current debate round")
    consensus_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Agreement level for consensus"
    )
    participants: List[str] = Field(
        ...,
        description="Agents participating in debate"
    )
    smes_participating: List[str] = Field(
        default_factory=list,
        description="SME personas participating"
    )


def get_required_agents_for_phase(phase: str, tier: int) -> List[str]:
    """
    Get the list of agents required for a specific pipeline phase.

    Args:
        phase: The pipeline phase
        tier: The tier level

    Returns:
        List of required agent names
    """
    # Phase to agent mappings
    phase_agents = {
        "Phase 1": ["Task Analyst"],
        "Phase 2": ["Domain Council Chair"] if tier >= 3 else [],
        "Phase 3": ["Planner"],
        "Phase 4": ["Clarifier"],
        "Phase 5": ["Executor"],
        "Phase 6": ["Researcher", "Code Reviewer", "Verifier", "Critic"],
        "Phase 7": ["Executor"],
        "Phase 8": ["Reviewer", "Formatter"],
    }

    return phase_agents.get(phase, [])


def calculate_phase_cost_estimate(tier: int, phase: str) -> float:
    """
    Estimate the cost (in USD) for a phase at a given tier.

    Rough estimates based on model usage:
    - opus: ~$0.015/1K tokens (input), ~$0.075/1K tokens (output)
    - sonnet: ~$0.003/1K tokens (input), ~$0.015/1K tokens (output)

    Args:
        tier: The tier level
        phase: The pipeline phase

    Returns:
        Estimated cost in USD
    """
    # Base cost per agent interaction (very rough estimate)
    agent_cost = {
        "Orchestrator": 0.10,  # opus, more tokens
        "Verifier": 0.08,      # opus
        "Critic": 0.08,        # opus
        "Reviewer": 0.08,      # opus
        "Council": 0.10,       # opus
        "Analyst": 0.02,       # sonnet
        "Planner": 0.02,       # sonnet
        "Clarifier": 0.02,     # sonnet
        "Researcher": 0.03,    # sonnet + web search
        "Executor": 0.04,      # sonnet, more tokens
        "Code Reviewer": 0.02, # sonnet
        "Formatter": 0.02,     # sonnet
        "SME": 0.03,           # sonnet + skills
    }

    # Get agents for this phase
    agents = get_required_agents_for_phase(phase, tier)

    # Sum costs
    total = sum(agent_cost.get(agent, 0.02) for agent in agents)

    # Tier multiplier
    tier_multiplier = {1: 1.0, 2: 1.2, 3: 1.5, 4: 2.0}
    total *= tier_multiplier.get(tier, 1.0)

    return round(total, 4)
