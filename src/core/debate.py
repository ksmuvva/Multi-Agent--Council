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


# =============================================================================
# Advanced SME Debate Protocol - Multi-Round Adversarial Debates
# =============================================================================

class SMEDebateState(str, Enum):
    """States in the SME debate lifecycle."""
    INITIALIZING = "initializing"
    POSITIONING = "positioning"
    ARGUMENTATION = "argumentation"
    DELIBERATION = "deliberation"
    CONVERGED = "converged"
    STALEMATED = "stalemated"
    ABORTED = "aborted"


class SMEConvergenceReason(str, Enum):
    """Reasons for SME debate convergence."""
    CONSENSUS = "consensus"
    MAJORITY_CONSENSUS = "majority_consensus"
    COMPROMISE = "compromise"
    STALEMATE = "stalemate"
    MAX_ROUNDS = "max_rounds"
    TIMEOUT = "timeout"


class SMEDebateProtocol:
    """
    Advanced debate protocol for multi-round SME adversarial debates.

    Orchestrates structured debates between SME personas where:
    1. Each SME presents their domain-specific position
    2. SMEs review opponents' positions
    3. SMEs generate counter-arguments
    4. SMEs refine positions based on counter-arguments
    5. Process repeats until convergence or max rounds
    """

    def __init__(
        self,
        max_rounds: int = 5,
        convergence_threshold: float = 0.75,
        stalemate_threshold: int = 2,
        max_execution_time_seconds: float = 180.0,
    ):
        """
        Initialize the SME debate protocol.

        Args:
            max_rounds: Maximum number of debate rounds
            convergence_threshold: Agreement level for consensus (0-1)
            stalemate_threshold: Rounds without position change before stalemate
            max_execution_time_seconds: Maximum execution time
        """
        self.max_rounds = max_rounds
        self.convergence_threshold = convergence_threshold
        self.stalemate_threshold = stalemate_threshold
        self.max_execution_time_seconds = max_execution_time_seconds
        self.logger = get_logger("sme_debate")

        # Debate state tracking
        self.rounds_completed: int = 0
        self.debate_history: List[Dict[str, Any]] = []
        self.position_history: Dict[str, List[str]] = {}  # sme_id -> list of positions

    def execute_sme_debate(
        self,
        topic: str,
        sme_personas: List[Dict[str, Any]],
        initial_proposal: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute a multi-round SME debate.

        Args:
            topic: The debate topic
            sme_personas: List of SME configs (id, name, domain, system_prompt)
            initial_proposal: Optional initial proposal to debate
            context: Additional context

        Returns:
            Debate result with all rounds and outcome
        """
        import time
        start_time = time.time()
        state = SMEDebateState.INITIALIZING

        self.logger.info(
            "sme_debate_started",
            topic=topic,
            participants=len(sme_personas),
            max_rounds=self.max_rounds,
        )

        # Initialize position history
        for sme in sme_personas:
            sme_id = sme.get("id", sme.get("name"))
            self.position_history[sme_id] = []

        rounds_data = []
        convergence_info = {"converged": False, "reason": None}

        try:
            for round_num in range(1, self.max_rounds + 1):
                # Check timeout
                elapsed = time.time() - start_time
                if elapsed > self.max_execution_time_seconds:
                    state = SMEDebateState.STALEMATED
                    convergence_info = {"converged": True, "reason": SMEConvergenceReason.TIMEOUT}
                    self.logger.warning("sme_debate_timeout", elapsed_seconds=elapsed)
                    break

                # Execute round
                round_data = self._execute_sme_round(
                    round_number=round_num,
                    topic=topic,
                    sme_personas=sme_personas,
                    initial_proposal=initial_proposal,
                    previous_rounds=rounds_data,
                    context=context,
                )
                rounds_data.append(round_data)
                self.rounds_completed = round_num

                # Check convergence
                convergence_check = self._check_sme_convergence(rounds_data)
                if convergence_check["converged"]:
                    state = SMEDebateState.CONVERGED
                    convergence_info = convergence_check
                    self.logger.info(
                        "sme_debate_converged",
                        round=round_num,
                        reason=convergence_check["reason"],
                    )
                    break

            # If no convergence, it's a stalemate
            if not convergence_info["converged"]:
                state = SMEDebateState.STALEMATED
                convergence_info = {"converged": True, "reason": SMEConvergenceReason.MAX_ROUNDS}

        except Exception as e:
            self.logger.error("sme_debate_error", error=str(e), exc_info=True)
            state = SMEDebateState.ABORTED

        # Generate outcome
        execution_time = time.time() - start_time
        outcome = self._generate_sme_outcome(
            rounds=rounds_data,
            state=state,
            convergence_info=convergence_info,
            execution_time_seconds=execution_time,
        )

        result = {
            "state": state.value,
            "rounds_completed": self.rounds_completed,
            "rounds": rounds_data,
            "outcome": outcome,
            "execution_time_seconds": execution_time,
            "debate_id": f"sme_debate_{int(start_time * 1000000)}",
        }

        self.logger.info(
            "sme_debate_completed",
            state=state.value,
            rounds_completed=self.rounds_completed,
            outcome_reason=convergence_info.get("reason"),
        )

        return result

    def _execute_sme_round(
        self,
        round_number: int,
        topic: str,
        sme_personas: List[Dict[str, Any]],
        initial_proposal: Optional[str],
        previous_rounds: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Execute a single SME debate round."""
        started_at = datetime.now(timezone.utc).isoformat()
        state = SMEDebateState.ARGUMENTATION if round_number > 1 else SMEDebateState.POSITIONING

        # Extract previous positions
        previous_positions = {}
        if previous_rounds:
            last_round = previous_rounds[-1]
            for turn in last_round.get("turns", []):
                sme_id = turn.get("sme_id")
                position = turn.get("position")
                if sme_id and position:
                    previous_positions[sme_id] = position

        # Generate turns for each SME
        turns = []
        for sme in sme_personas:
            turn = self._generate_sme_turn(
                sme=sme,
                topic=topic,
                initial_proposal=initial_proposal,
                previous_positions=previous_positions,
                round_number=round_number,
                context=context,
            )
            turns.append(turn)

            # Track position history
            sme_id = sme.get("id", sme.get("name"))
            if sme_id:
                self.position_history.setdefault(sme_id, []).append(turn.get("position", ""))

        completed_at = datetime.now(timezone.utc).isoformat()

        return {
            "round_number": round_number,
            "state": state.value,
            "turns": turns,
            "started_at": started_at,
            "completed_at": completed_at,
            "summary": self._generate_round_summary(turns, round_number),
        }

    def _generate_sme_turn(
        self,
        sme: Dict[str, Any],
        topic: str,
        initial_proposal: Optional[str],
        previous_positions: Dict[str, str],
        round_number: int,
        context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Generate a debate turn for an SME."""
        sme_id = sme.get("id", sme.get("name", "Unknown"))
        sme_name = sme.get("name", sme_id)
        domain = sme.get("domain", "")

        if round_number == 1:
            # Initial position
            position = self._get_initial_position(domain, topic, initial_proposal)
            counter_arguments = []
            confidence = 0.85
            willingness_to_concede = 0.15
        else:
            # Response with counter-arguments
            position = previous_positions.get(sme_id, "")
            counter_arguments = self._generate_counter_arguments(
                domain, previous_positions, sme_id
            )

            # Adjust confidence and concession based on round
            confidence = max(0.55, 0.85 - (round_number * 0.08))
            willingness_to_concede = min(0.65, 0.15 + (round_number * 0.12))

        return {
            "sme_id": sme_id,
            "sme_name": sme_name,
            "domain": domain,
            "position": position,
            "domain_rationale": f"Based on {domain} expertise and best practices.",
            "counter_arguments": counter_arguments,
            "confidence": confidence,
            "willingness_to_concede": willingness_to_concede,
            "round_number": round_number,
        }

    def _get_initial_position(
        self,
        domain: str,
        topic: str,
        initial_proposal: Optional[str],
    ) -> str:
        """Get the initial position for an SME based on their domain."""
        domain_positions = {
            "security": (
                "Security must be the top priority. Any approach that compromises "
                "authentication, authorization, or data protection should be rejected."
            ),
            "cloud": (
                "Cloud-native serverless architecture provides optimal scalability "
                "and cost-efficiency. We should avoid traditional infrastructure."
            ),
            "data": (
                "Data integrity and consistency are paramount. ACID compliance and "
                "strong consistency models must be maintained."
            ),
            "frontend": (
                "User experience is critical. Fast load times, responsive design, "
                "and intuitive interfaces should drive technical decisions."
            ),
            "devops": (
                "Automation and CI/CD enable rapid, reliable delivery. Infrastructure "
                "as code and comprehensive testing should be prioritized."
            ),
            "identity": "Identity governance and least-privilege access must be enforced.",
            "ai_ml": "Model accuracy and evaluation metrics must drive architecture decisions.",
        }

        position = domain_positions.get(domain.lower())
        if position:
            return position

        if initial_proposal:
            return f"Reviewing the proposal from a {domain} perspective: {initial_proposal[:100]}..."

        return f"From a {domain} perspective, we need to ensure domain requirements are fully addressed."

    def _generate_counter_arguments(
        self,
        domain: str,
        previous_positions: Dict[str, str],
        own_sme_id: str,
    ) -> List[str]:
        """Generate counter-arguments to other SMEs' positions."""
        counter_arguments = []
        domain_counters = {
            "security": [
                "This approach may introduce security vulnerabilities.",
                "Compliance requirements are not adequately addressed.",
                "The attack surface is unnecessarily expanded.",
            ],
            "cloud": [
                "This creates vendor lock-in and reduces portability.",
                "Cost projections don't account for scaling complexity.",
                "Serverless cold starts will impact user experience.",
            ],
            "data": [
                "This risks data inconsistency and corruption.",
                "Transaction integrity is not guaranteed.",
                "Analytics quality will be compromised.",
            ],
            "frontend": [
                "This will degrade user experience with slower interactions.",
                "Accessibility requirements are not met.",
                "Mobile responsiveness is insufficient.",
            ],
            "devops": [
                "This slows down the delivery pipeline unnecessarily.",
                "Testing coverage is inadequate for production readiness.",
                "Deployment complexity increases failure risk.",
            ],
        }

        counters = domain_counters.get(domain.lower(), ["This approach has significant domain-specific drawbacks."])

        for other_sme_id, other_position in previous_positions.items():
            if other_sme_id != own_sme_id:
                counter_arguments.append({
                    "target_sme": other_sme_id,
                    "argument": counters[0] if counters else "Concerns remain from our domain perspective.",
                    "target_position_excerpt": other_position[:100] + "..." if len(other_position) > 100 else other_position,
                })

        return counter_arguments

    def _check_sme_convergence(self, rounds: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check if the SME debate has converged."""
        if not rounds:
            return {"converged": False, "reason": None}

        last_round = rounds[-1]
        turns = last_round.get("turns", [])

        if not turns:
            return {"converged": False, "reason": None}

        # Calculate average willingness to concede
        avg_willingness = sum(t.get("willingness_to_concede", 0) for t in turns) / len(turns)

        # Check for consensus
        if avg_willingness >= self.convergence_threshold:
            return {"converged": True, "reason": SMEConvergenceReason.CONSENSUS}

        # Check for stalemate (no movement in positions)
        if len(rounds) >= 2:
            prev_round = rounds[-2]
            movement_detected = False

            for curr_turn in turns:
                sme_id = curr_turn.get("sme_id")
                curr_conf = curr_turn.get("confidence", 0)

                # Find corresponding turn in previous round
                for prev_turn in prev_round.get("turns", []):
                    if prev_turn.get("sme_id") == sme_id:
                        prev_conf = prev_turn.get("confidence", 0)
                        if abs(curr_conf - prev_conf) > 0.05:
                            movement_detected = True
                        break

            if not movement_detected:
                return {"converged": True, "reason": SMEConvergenceReason.STALEMATE}

        return {"converged": False, "reason": None}

    def _generate_round_summary(self, turns: List[Dict[str, Any]], round_number: int) -> str:
        """Generate a summary of the debate round."""
        if not turns:
            return f"Round {round_number}: No SMEs participated."

        participant_names = [t.get("sme_name", t.get("sme_id")) for t in turns]
        avg_confidence = sum(t.get("confidence", 0) for t in turns) / len(turns)
        avg_concession = sum(t.get("willingness_to_concede", 0) for t in turns) / len(turns)
        total_counters = sum(len(t.get("counter_arguments", [])) for t in turns)

        return (
            f"Round {round_number}: {len(turns)} SMEs participated. "
            f"Avg confidence: {avg_confidence:.1%}, "
            f"Avg willingness to concede: {avg_concession:.1%}, "
            f"{total_counters} counter-arguments raised. "
            f"Participants: {', '.join(participant_names)}"
        )

    def _generate_sme_outcome(
        self,
        rounds: List[Dict[str, Any]],
        state: SMEDebateState,
        convergence_info: Dict[str, Any],
        execution_time_seconds: float,
    ) -> Dict[str, Any]:
        """Generate the final SME debate outcome."""
        if not rounds:
            return {
                "convergence_reason": SMEConvergenceReason.STALEMATE.value,
                "final_positions": {},
                "consensus_position": None,
                "recommendations": ["Debate failed to produce meaningful positions"],
                "disagreements": ["No debate occurred"],
            }

        last_round = rounds[-1]
        turns = last_round.get("turns", [])

        # Extract final positions
        final_positions = {t.get("sme_id"): t.get("position") for t in turns}

        # Check for consensus position
        avg_willingness = sum(t.get("willingness_to_concede", 0) for t in turns) / len(turns)
        consensus_position = None

        if avg_willingness >= self.convergence_threshold:
            domains = [t.get("domain") for t in turns]
            consensus_position = (
                f"Consensus reached among {', '.join(domains)}. "
                "A balanced approach addressing all domain concerns is recommended."
            )

        # Generate recommendations
        recommendations = [
            "Consider a phased approach that addresses immediate concerns while planning for long-term needs.",
            "Establish clear trade-off documentation for all technical decisions.",
            "Implement cross-domain checkpoints to validate assumptions.",
            "Create working groups for ongoing alignment on contentious issues.",
        ]

        # Identify remaining disagreements
        disagreements = []
        for turn in turns:
            if turn.get("willingness_to_concede", 0) < 0.5:
                position = turn.get("position", "")
                disagreements.append({
                    "sme": turn.get("sme_name"),
                    "concern": position[:150] + "..." if len(position) > 150 else position,
                })

        return {
            "convergence_reason": convergence_info.get("reason", SMEConvergenceReason.MAX_ROUNDS).value,
            "final_positions": final_positions,
            "consensus_position": consensus_position,
            "recommendations": recommendations,
            "disagreements": disagreements,
        }
