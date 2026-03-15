"""
Debate Protocol Module

Implements self-play debate protocol for resolving disagreements
and achieving consensus through multi-perspective reasoning.
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

from src.utils.logging import get_logger
from src.utils.events import emit_system_message, emit_task_progress


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
        self.logger = get_logger("debate")

    def add_participant(self, agent_name: str) -> None:
        """Add an agent to the debate."""
        if agent_name not in self.participants:
            self.participants.append(agent_name)
            self.logger.debug("debate.participant_added", agent=agent_name)

    def add_sme_participant(self, sme_persona: str) -> None:
        """Add an SME persona to the debate."""
        if sme_persona not in self.sme_participants:
            self.sme_participants.append(sme_persona)
            self.logger.debug("debate.sme_participant_added", sme=sme_persona)

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
        round_num = len(self.rounds) + 1
        self.logger.info("debate.round_started",
                         round=round_num,
                         participants=self.participants,
                         sme_count=len(sme_arguments))

        # Calculate agreement scores by analyzing actual content from each agent
        executor_agreement = self._score_executor_agreement(
            executor_position, critic_challenges, verifier_checks
        )
        critic_agreement = self._score_critic_agreement(
            executor_position, critic_challenges
        )
        verifier_agreement = self._score_verifier_agreement(
            executor_position, verifier_checks
        )
        sme_agreements = {
            sme: self._score_sme_agreement(executor_position, args)
            for sme, args in sme_arguments.items()
        }

        self.logger.info("debate.agreement_scores",
                         round=round_num,
                         executor=executor_agreement,
                         critic=critic_agreement,
                         verifier=verifier_agreement,
                         sme_scores=sme_agreements)

        # Calculate consensus
        consensus = self.calculate_consensus(
            executor_agreement,
            critic_agreement,
            verifier_agreement,
            sme_agreements
        )

        consensus_level = self.determine_consensus_level(consensus)
        self.logger.info("debate.round_completed",
                         round=round_num,
                         consensus_score=consensus,
                         consensus_level=consensus_level,
                         critic_challenges_count=len(critic_challenges),
                         verifier_checks_count=len(verifier_checks))
        emit_system_message(
            f"Debate round {round_num}: {consensus_level} consensus ({consensus:.0%})")

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

    # ========================================================================
    # Agreement Scoring
    # ========================================================================

    @staticmethod
    def _text_overlap_score(text_a: str, text_b: str) -> float:
        """
        Calculate a normalized overlap score between two texts based on
        shared words (Jaccard similarity on word tokens).

        Returns a score between 0.0 and 1.0.
        """
        if not text_a or not text_b:
            return 0.0
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union)

    @staticmethod
    def _contains_disagreement_signals(text: str) -> float:
        """
        Detect disagreement language in text.

        Returns a penalty between 0.0 (no disagreement) and 0.5
        (strong disagreement signals).
        """
        if not text:
            return 0.0
        lower = text.lower()
        strong_signals = ["incorrect", "wrong", "invalid", "reject", "disagree", "fail"]
        mild_signals = ["however", "but", "concern", "issue", "unclear", "questionable"]
        penalty = 0.0
        for word in strong_signals:
            if word in lower:
                penalty += 0.1
        for word in mild_signals:
            if word in lower:
                penalty += 0.05
        return min(penalty, 0.5)

    def _score_executor_agreement(
        self,
        executor_position: str,
        critic_challenges: List[str],
        verifier_checks: List[str],
    ) -> float:
        """
        Score executor agreement: high baseline since executors defend
        their own solution, reduced by the volume and severity of
        challenges and verification issues raised.
        """
        base_score = 0.95
        # Reduce score based on number and content of challenges
        if critic_challenges:
            challenge_text = " ".join(critic_challenges)
            disagreement = self._contains_disagreement_signals(challenge_text)
            volume_penalty = min(len(critic_challenges) * 0.05, 0.2)
            base_score -= (disagreement + volume_penalty)
        if verifier_checks:
            check_text = " ".join(verifier_checks)
            disagreement = self._contains_disagreement_signals(check_text)
            base_score -= disagreement * 0.5
        return max(0.0, min(1.0, round(base_score, 2)))

    def _score_critic_agreement(
        self,
        executor_position: str,
        critic_challenges: List[str],
    ) -> float:
        """
        Score critic agreement: starts neutral and adjusts based on
        content overlap with the executor position and the severity
        of challenges raised.
        """
        if not critic_challenges:
            # No challenges means implicit agreement
            return 0.75

        challenge_text = " ".join(critic_challenges)
        # Higher text overlap with executor = more agreement
        overlap = self._text_overlap_score(executor_position, challenge_text)
        # Disagreement signals reduce agreement
        disagreement = self._contains_disagreement_signals(challenge_text)
        # Volume of challenges reduces agreement
        volume_factor = min(len(critic_challenges) * 0.08, 0.3)

        score = 0.5 + (overlap * 0.3) - disagreement - volume_factor
        return max(0.0, min(1.0, round(score, 2)))

    def _score_verifier_agreement(
        self,
        executor_position: str,
        verifier_checks: List[str],
    ) -> float:
        """
        Score verifier agreement: based on how well the executor
        position aligns with verification checks and whether checks
        indicate problems.
        """
        if not verifier_checks:
            # No checks filed means the verifier found no issues
            return 0.85

        check_text = " ".join(verifier_checks)
        overlap = self._text_overlap_score(executor_position, check_text)
        disagreement = self._contains_disagreement_signals(check_text)
        volume_factor = min(len(verifier_checks) * 0.06, 0.25)

        score = 0.6 + (overlap * 0.25) - disagreement - volume_factor
        return max(0.0, min(1.0, round(score, 2)))

    def _score_sme_agreement(
        self,
        executor_position: str,
        sme_argument: str,
    ) -> float:
        """
        Score SME agreement: based on semantic overlap between the
        SME's argument and the executor position, adjusted for
        disagreement signals.
        """
        if not sme_argument:
            # No argument means the SME abstained; treat as neutral
            return 0.5

        overlap = self._text_overlap_score(executor_position, sme_argument)
        disagreement = self._contains_disagreement_signals(sme_argument)

        score = 0.5 + (overlap * 0.4) - disagreement
        return max(0.0, min(1.0, round(score, 2)))

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

        outcome = DebateOutcome(
            consensus_level=consensus,
            consensus_score=latest_round.consensus_score,
            rounds_completed=len(self.rounds),
            final_resolution=resolution,
            alternative_presented=alternative_presented,
            arbiter_invoked=arbiter_invoked,
            summary=self._generate_summary()
        )
        self.logger.info("debate.outcome",
                         consensus_level=consensus,
                         consensus_score=latest_round.consensus_score,
                         rounds_completed=len(self.rounds),
                         arbiter_invoked=arbiter_invoked,
                         alternative_presented=alternative_presented)
        return outcome

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
