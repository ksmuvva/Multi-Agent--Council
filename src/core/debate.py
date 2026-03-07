"""
Debate Protocol Module

Implements self-play debate protocol for resolving disagreements
and achieving consensus through multi-perspective reasoning.
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class ConsensusLevel(str, Enum):
    """Level of consensus achieved."""
    FULL = "full"  # ≥80% agreement
    MAJORITY = "majority"  # 50-79% agreement
    SPLIT = "split"  # <50% agreement


class DebateRound(BaseModel):
    """A single round of debate."""
    round_number: int = Field(..., ge=1, description="Round number")
    executor_position: str = Field(..., description="Executor's position")
    critic_challenges: List[str] = Field(..., description="Critic's challenges")
    verifier_checks: List[str] = Field(..., description="Verifier's fact checks")
    sme_arguments: Dict[str, str] = Field(
        default_factory=dict,
        description="SME contributions by persona"
    )
    consensus_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Agreement level (0-1)"
    )


class DebateOutcome(BaseModel):
    """Result of a debate session."""
    consensus_level: ConsensusLevel = Field(..., description="Level of consensus")
    consensus_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall consensus score (0-1)"
    )
    rounds_completed: int = Field(..., ge=0, description="Number of rounds completed")
    final_resolution: str = Field(..., description="Final resolution or decision")
    alternative_presented: bool = Field(
        default=False,
        description="Whether alternatives were presented to user"
    )
    arbiter_invoked: bool = Field(
        default=False,
        description="Whether Quality Arbiter was invoked"
    )
    summary: str = Field(..., description="Summary of the debate")


class DebateProtocol:
    """
    Manages self-play debate protocol for resolving disagreements.

    Protocol:
    1. Executor defends the solution
    2. Critic challenges with adversarial arguments
    3. Verifier fact-checks claims
    4. SMEs contribute domain-specific arguments
    5. Calculate consensus score
    6. Repeat if consensus not achieved (max 2 rounds)
    7. Quality Arbiter breaks deadlock if needed
    """

    def __init__(
        self,
        max_rounds: int = 2,
        consensus_threshold: float = 0.8,
        majority_threshold: float = 0.5
    ):
        """
        Initialize the debate protocol.

        Args:
            max_rounds: Maximum number of debate rounds
            consensus_threshold: Score for full consensus (default 0.8)
            majority_threshold: Score for majority (default 0.5)
        """
        self.max_rounds = max_rounds
        self.consensus_threshold = consensus_threshold
        self.majority_threshold = majority_threshold
        self.rounds: List[DebateRound] = []
        self.participants: List[str] = []
        self.sme_participants: List[str] = []

    def add_participant(self, agent_name: str) -> None:
        """Add an agent to the debate."""
        if agent_name not in self.participants:
            self.participants.append(agent_name)

    def add_sme_participant(self, sme_persona: str) -> None:
        """Add an SME persona to the debate."""
        if sme_persona not in self.sme_participants:
            self.sme_participants.append(sme_persona)

    def calculate_consensus(
        self,
        executor_agreement: float,
        critic_agreement: float,
        verifier_agreement: float,
        sme_agreements: Dict[str, float]
    ) -> float:
        """
        Calculate overall consensus score.

        Args:
            executor_agreement: Executor's agreement level (0-1)
            critic_agreement: Critic's agreement level (0-1)
            verifier_agreement: Verifier's agreement level (0-1)
            sme_agreements: SME agreement levels by persona (0-1)

        Returns:
            Overall consensus score (0-1)
        """
        # Weighted average of participants
        weights = {
            "executor": 0.25,
            "critic": 0.25,
            "verifier": 0.25,
        }
        sme_weight = 0.25 / max(1, len(sme_agreements))

        total = (
            weights["executor"] * executor_agreement +
            weights["critic"] * critic_agreement +
            weights["verifier"] * verifier_agreement +
            sum(sme_weight * agr for agr in sme_agreements.values())
        )

        return round(total, 2)

    def determine_consensus_level(self, score: float) -> ConsensusLevel:
        """
        Determine the consensus level from a score.

        Args:
            score: Consensus score (0-1)

        Returns:
            ConsensusLevel enum value
        """
        if score >= self.consensus_threshold:
            return ConsensusLevel.FULL
        elif score >= self.majority_threshold:
            return ConsensusLevel.MAJORITY
        else:
            return ConsensusLevel.SPLIT

    def should_continue_debate(self, current_score: float) -> bool:
        """
        Determine if debate should continue.

        Args:
            current_score: Current consensus score

        Returns:
            True if debate should continue
        """
        # Continue if we haven't reached max rounds and haven't achieved full consensus
        return (
            len(self.rounds) < self.max_rounds and
            current_score < self.consensus_threshold
        )

    def can_proceed(self, consensus_level: ConsensusLevel) -> bool:
        """
        Determine if the solution can proceed based on consensus.

        Args:
            consensus_level: The achieved consensus level

        Returns:
            True if the solution can proceed
        """
        return consensus_level in [ConsensusLevel.FULL, ConsensusLevel.MAJORITY]

    def needs_arbiter(self, consensus_level: ConsensusLevel, rounds: int) -> bool:
        """
        Determine if Quality Arbiter should be invoked.

        Args:
            consensus_level: The achieved consensus level
            rounds: Number of rounds completed

        Returns:
            True if arbiter should be invoked
        """
        # Arbiter for split consensus or after max rounds without full consensus
        return (
            consensus_level == ConsensusLevel.SPLIT or
            (rounds >= self.max_rounds and consensus_level != ConsensusLevel.FULL)
        )

    def conduct_round(
        self,
        executor_position: str,
        critic_challenges: List[str],
        verifier_checks: List[str],
        sme_arguments: Dict[str, str]
    ) -> DebateRound:
        """
        Conduct a single debate round.

        Args:
            executor_position: Executor's defense
            critic_challenges: Critic's challenges
            verifier_checks: Verifier's fact checks
            sme_arguments: SME contributions by persona

        Returns:
            DebateRound with results
        """
        # Simulate agreement scores (in real system, these would come from agents)
        executor_agreement = 0.9  # Executor generally agrees with their own solution
        critic_agreement = 0.4 if critic_challenges else 0.7
        verifier_agreement = 0.7 if verifier_checks else 0.8
        sme_agreements = {
            sme: 0.6 + (0.1 if args else 0.0)
            for sme, args in sme_arguments.items()
        }

        # Calculate consensus
        consensus = self.calculate_consensus(
            executor_agreement,
            critic_agreement,
            verifier_agreement,
            sme_agreements
        )

        round_num = len(self.rounds) + 1
        debate_round = DebateRound(
            round_number=round_num,
            executor_position=executor_position,
            critic_challenges=critic_challenges,
            verifier_checks=verifier_checks,
            sme_arguments=sme_arguments,
            consensus_score=consensus
        )

        self.rounds.append(debate_round)
        return debate_round

    def get_outcome(self) -> DebateOutcome:
        """
        Get the final outcome of the debate.

        Returns:
            DebateOutcome with final resolution
        """
        if not self.rounds:
            return DebateOutcome(
                consensus_level=ConsensusLevel.SPLIT,
                consensus_score=0.0,
                rounds_completed=0,
                final_resolution="No debate rounds conducted",
                summary="Debate was not conducted"
            )

        # Get latest consensus
        latest_round = self.rounds[-1]
        consensus = self.determine_consensus_level(latest_round.consensus_score)

        # Determine resolution
        if consensus == ConsensusLevel.FULL:
            resolution = f"Full consensus achieved ({latest_round.consensus_score:.0%}). Solution approved."
            alternative_presented = False
            arbiter_invoked = False
        elif consensus == ConsensusLevel.MAJORITY:
            resolution = f"Majority consensus achieved ({latest_round.consensus_score:.0%}). Solution approved with notes."
            alternative_presented = False
            arbiter_invoked = False
        else:  # SPLIT
            if self.needs_arbiter(consensus, len(self.rounds)):
                resolution = "Split consensus. Quality Arbiter invoked for final decision."
                arbiter_invoked = True
            else:
                resolution = "Split consensus. Presenting alternatives to user."
                arbiter_invoked = False
            alternative_presented = True

        return DebateOutcome(
            consensus_level=consensus,
            consensus_score=latest_round.consensus_score,
            rounds_completed=len(self.rounds),
            final_resolution=resolution,
            alternative_presented=alternative_presented,
            arbiter_invoked=arbiter_invoked,
            summary=self._generate_summary()
        )

    def _generate_summary(self) -> str:
        """Generate a summary of the debate."""
        if not self.rounds:
            return "No debate rounds conducted."

        parts = [f"Debate conducted over {len(self.rounds)} round(s)."]

        for round_data in self.rounds:
            level = self.determine_consensus_level(round_data.consensus_score)
            parts.append(
                f"Round {round_data.round_number}: "
                f"{level.value.upper()} consensus "
                f"({round_data.consensus_score:.0%})"
            )

        if self.sme_participants:
            parts.append(f"SME participants: {', '.join(self.sme_participants)}")

        return " ".join(parts)


def trigger_debate(
    verifier_verdict: str,
    critic_verdict: str,
    tier_level: int,
    available_smes: List[str] = None
) -> bool:
    """
    Determine if a debate should be triggered based on conditions.

    Triggers on:
    - Verifier/Critic disagreement
    - Tier 4 tasks
    - After 2 failed revision cycles

    Args:
        verifier_verdict: Verifier's verdict (PASS/FAIL)
        critic_verdict: Critic's verdict (PASS/FAIL)
        tier_level: Current tier level
        available_smes: Available SME personas

    Returns:
        True if debate should be triggered
    """
    # Disagreement triggers debate
    if verifier_verdict != critic_verdict:
        return True

    # Tier 4 always debates
    if tier_level >= 4:
        return True

    return False


def get_debate_participants(
    tier_level: int,
    available_smes: List[str] = None
) -> Dict[str, List[str]]:
    """
    Get the participants for a debate.

    Args:
        tier_level: Current tier level
        available_smes: Available SME personas

    Returns:
        Dictionary with 'agents' and 'smes' lists
    """
    participants = {
        "agents": ["Executor", "Critic", "Verifier"],
        "smes": []
    }

    # Add SMEs if available and tier >= 3
    if tier_level >= 3 and available_smes:
        participants["smes"] = available_smes[:3]  # Max 3 SMEs

    return participants
