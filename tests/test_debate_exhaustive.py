"""
Exhaustive Tests for Debate Protocol Module

Tests the DebateProtocol class, consensus calculation,
debate rounds, outcomes, and trigger functions.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from pydantic import ValidationError

from src.core.debate import (
    ConsensusLevel,
    DebateRound,
    DebateOutcome,
    DebateProtocol,
    trigger_debate,
    get_debate_participants,
)


# =============================================================================
# ConsensusLevel Enum Tests
# =============================================================================

class TestConsensusLevel:
    def test_values(self):
        assert ConsensusLevel.FULL == "full"
        assert ConsensusLevel.MAJORITY == "majority"
        assert ConsensusLevel.SPLIT == "split"

    def test_count(self):
        assert len(ConsensusLevel) == 3


# =============================================================================
# DebateRound Model Tests
# =============================================================================

class TestDebateRound:
    def test_valid_round(self):
        r = DebateRound(
            round_number=1,
            executor_position="Solution is correct",
            critic_challenges=["Missing edge case"],
            verifier_checks=["Claim verified"],
            consensus_score=0.7,
        )
        assert r.round_number == 1
        assert r.sme_arguments == {}

    def test_round_number_min(self):
        with pytest.raises(ValidationError):
            DebateRound(
                round_number=0, executor_position="", critic_challenges=[],
                verifier_checks=[], consensus_score=0.5,
            )

    def test_consensus_score_bounds(self):
        with pytest.raises(ValidationError):
            DebateRound(
                round_number=1, executor_position="", critic_challenges=[],
                verifier_checks=[], consensus_score=1.5,
            )
        with pytest.raises(ValidationError):
            DebateRound(
                round_number=1, executor_position="", critic_challenges=[],
                verifier_checks=[], consensus_score=-0.1,
            )

    def test_with_sme_arguments(self):
        r = DebateRound(
            round_number=1, executor_position="Pos", critic_challenges=["C1"],
            verifier_checks=["V1"], consensus_score=0.6,
            sme_arguments={"security": "Add rate limiting"},
        )
        assert "security" in r.sme_arguments


# =============================================================================
# DebateOutcome Model Tests
# =============================================================================

class TestDebateOutcome:
    def test_valid_outcome(self):
        o = DebateOutcome(
            consensus_level=ConsensusLevel.FULL,
            consensus_score=0.9,
            rounds_completed=2,
            final_resolution="Approved",
            summary="Debate completed",
        )
        assert o.alternative_presented is False
        assert o.arbiter_invoked is False

    def test_split_outcome(self):
        o = DebateOutcome(
            consensus_level=ConsensusLevel.SPLIT,
            consensus_score=0.3,
            rounds_completed=2,
            final_resolution="No agreement",
            alternative_presented=True,
            arbiter_invoked=True,
            summary="Deadlock",
        )
        assert o.arbiter_invoked is True


# =============================================================================
# DebateProtocol Tests
# =============================================================================

class TestDebateProtocol:
    def test_initialization(self):
        dp = DebateProtocol()
        assert dp.max_rounds == 2
        assert dp.consensus_threshold == 0.8
        assert dp.majority_threshold == 0.5
        assert dp.rounds == []
        assert dp.participants == []
        assert dp.sme_participants == []

    def test_custom_init(self):
        dp = DebateProtocol(max_rounds=3, consensus_threshold=0.9, majority_threshold=0.6)
        assert dp.max_rounds == 3
        assert dp.consensus_threshold == 0.9
        assert dp.majority_threshold == 0.6

    def test_add_participant(self):
        dp = DebateProtocol()
        dp.add_participant("Executor")
        dp.add_participant("Critic")
        assert len(dp.participants) == 2
        assert "Executor" in dp.participants

    def test_add_duplicate_participant(self):
        dp = DebateProtocol()
        dp.add_participant("Executor")
        dp.add_participant("Executor")
        assert len(dp.participants) == 1

    def test_add_sme_participant(self):
        dp = DebateProtocol()
        dp.add_sme_participant("cloud_architect")
        assert "cloud_architect" in dp.sme_participants

    def test_add_duplicate_sme(self):
        dp = DebateProtocol()
        dp.add_sme_participant("cloud_architect")
        dp.add_sme_participant("cloud_architect")
        assert len(dp.sme_participants) == 1

    # --------- Consensus Calculation ---------

    def test_calculate_consensus_all_agree(self):
        dp = DebateProtocol()
        score = dp.calculate_consensus(1.0, 1.0, 1.0, {})
        assert score == 0.75  # 0.25*1 + 0.25*1 + 0.25*1 + 0 (no SMEs = 0.25/1*0)

    def test_calculate_consensus_all_agree_with_sme(self):
        dp = DebateProtocol()
        score = dp.calculate_consensus(1.0, 1.0, 1.0, {"sme1": 1.0})
        assert score == 1.0

    def test_calculate_consensus_all_disagree(self):
        dp = DebateProtocol()
        score = dp.calculate_consensus(0.0, 0.0, 0.0, {})
        assert score == 0.0

    def test_calculate_consensus_mixed(self):
        dp = DebateProtocol()
        score = dp.calculate_consensus(0.9, 0.4, 0.7, {"sme1": 0.6})
        assert 0.0 <= score <= 1.0

    def test_calculate_consensus_multiple_smes(self):
        dp = DebateProtocol()
        score = dp.calculate_consensus(0.8, 0.8, 0.8, {"sme1": 0.8, "sme2": 0.8})
        assert 0.0 <= score <= 1.0

    # --------- Consensus Level ---------

    def test_determine_consensus_full(self):
        dp = DebateProtocol()
        assert dp.determine_consensus_level(0.8) == ConsensusLevel.FULL
        assert dp.determine_consensus_level(0.9) == ConsensusLevel.FULL
        assert dp.determine_consensus_level(1.0) == ConsensusLevel.FULL

    def test_determine_consensus_majority(self):
        dp = DebateProtocol()
        assert dp.determine_consensus_level(0.5) == ConsensusLevel.MAJORITY
        assert dp.determine_consensus_level(0.79) == ConsensusLevel.MAJORITY

    def test_determine_consensus_split(self):
        dp = DebateProtocol()
        assert dp.determine_consensus_level(0.0) == ConsensusLevel.SPLIT
        assert dp.determine_consensus_level(0.49) == ConsensusLevel.SPLIT

    def test_determine_consensus_boundary(self):
        dp = DebateProtocol(consensus_threshold=0.8, majority_threshold=0.5)
        assert dp.determine_consensus_level(0.8) == ConsensusLevel.FULL
        assert dp.determine_consensus_level(0.5) == ConsensusLevel.MAJORITY

    # --------- Should Continue ---------

    def test_should_continue_no_rounds(self):
        dp = DebateProtocol(max_rounds=2)
        assert dp.should_continue_debate(0.5) is True

    def test_should_continue_consensus_reached(self):
        dp = DebateProtocol(max_rounds=2)
        assert dp.should_continue_debate(0.9) is False

    def test_should_continue_max_rounds(self):
        dp = DebateProtocol(max_rounds=2)
        dp.rounds = [None, None]  # simulate 2 rounds
        assert dp.should_continue_debate(0.5) is False

    # --------- Can Proceed ---------

    def test_can_proceed_full(self):
        dp = DebateProtocol()
        assert dp.can_proceed(ConsensusLevel.FULL) is True

    def test_can_proceed_majority(self):
        dp = DebateProtocol()
        assert dp.can_proceed(ConsensusLevel.MAJORITY) is True

    def test_cannot_proceed_split(self):
        dp = DebateProtocol()
        assert dp.can_proceed(ConsensusLevel.SPLIT) is False

    # --------- Needs Arbiter ---------

    def test_needs_arbiter_split(self):
        dp = DebateProtocol()
        assert dp.needs_arbiter(ConsensusLevel.SPLIT, rounds=1) is True

    def test_needs_arbiter_max_rounds_majority(self):
        dp = DebateProtocol(max_rounds=2)
        assert dp.needs_arbiter(ConsensusLevel.MAJORITY, rounds=2) is True

    def test_no_arbiter_full_consensus(self):
        dp = DebateProtocol()
        assert dp.needs_arbiter(ConsensusLevel.FULL, rounds=2) is False

    def test_no_arbiter_majority_under_max(self):
        dp = DebateProtocol(max_rounds=3)
        assert dp.needs_arbiter(ConsensusLevel.MAJORITY, rounds=1) is False

    # --------- Conduct Round ---------

    def test_conduct_round(self):
        dp = DebateProtocol()
        result = dp.conduct_round(
            executor_position="Solution is valid",
            critic_challenges=["Missing error handling"],
            verifier_checks=["Verified claims"],
            sme_arguments={"security": "Add auth"},
        )
        assert isinstance(result, DebateRound)
        assert result.round_number == 1
        assert 0.0 <= result.consensus_score <= 1.0
        assert len(dp.rounds) == 1

    def test_conduct_multiple_rounds(self):
        dp = DebateProtocol()
        r1 = dp.conduct_round("Pos1", ["C1"], ["V1"], {})
        r2 = dp.conduct_round("Pos2", [], ["V2"], {"sme1": "arg"})
        assert r1.round_number == 1
        assert r2.round_number == 2
        assert len(dp.rounds) == 2

    def test_conduct_round_no_challenges(self):
        dp = DebateProtocol()
        result = dp.conduct_round("Position", [], [], {})
        assert result.consensus_score > 0

    def test_conduct_round_with_challenges_lowers_score(self):
        dp = DebateProtocol()
        r_no_challenges = dp.conduct_round("Pos", [], [], {})
        dp2 = DebateProtocol()
        r_with_challenges = dp2.conduct_round("Pos", ["C1", "C2"], [], {})
        assert r_no_challenges.consensus_score >= r_with_challenges.consensus_score

    # --------- Get Outcome ---------

    def test_get_outcome_no_rounds(self):
        dp = DebateProtocol()
        outcome = dp.get_outcome()
        assert outcome.consensus_level == ConsensusLevel.SPLIT
        assert outcome.consensus_score == 0.0
        assert outcome.rounds_completed == 0

    def test_get_outcome_after_round(self):
        dp = DebateProtocol()
        dp.conduct_round("Pos", [], ["V1"], {"sme1": "arg"})
        outcome = dp.get_outcome()
        assert isinstance(outcome, DebateOutcome)
        assert outcome.rounds_completed == 1

    def test_get_outcome_full_consensus(self):
        dp = DebateProtocol(consensus_threshold=0.5)
        dp.conduct_round("Pos", [], [], {"sme1": "support"})
        outcome = dp.get_outcome()
        assert "consensus" in outcome.final_resolution.lower() or "approved" in outcome.final_resolution.lower()

    def test_get_outcome_summary_content(self):
        dp = DebateProtocol()
        dp.add_sme_participant("security_analyst")
        dp.conduct_round("Pos", ["C1"], ["V1"], {"security_analyst": "Review"})
        outcome = dp.get_outcome()
        assert "round" in outcome.summary.lower()

    # --------- Generate Summary ---------

    def test_generate_summary_no_rounds(self):
        dp = DebateProtocol()
        assert "No debate rounds" in dp._generate_summary()

    def test_generate_summary_with_smes(self):
        dp = DebateProtocol()
        dp.add_sme_participant("cloud_architect")
        dp.conduct_round("Pos", ["C1"], ["V1"], {"cloud_architect": "Input"})
        summary = dp._generate_summary()
        assert "cloud_architect" in summary


# =============================================================================
# trigger_debate Function Tests
# =============================================================================

class TestTriggerDebate:
    def test_disagreement_triggers(self):
        assert trigger_debate("PASS", "FAIL", tier_level=2) is True

    def test_agreement_no_trigger_tier2(self):
        assert trigger_debate("PASS", "PASS", tier_level=2) is False

    def test_tier4_always_triggers(self):
        assert trigger_debate("PASS", "PASS", tier_level=4) is True

    def test_tier3_agreement_no_trigger(self):
        assert trigger_debate("PASS", "PASS", tier_level=3) is False

    def test_fail_fail_disagreement(self):
        # FAIL == FAIL, so no disagreement
        assert trigger_debate("FAIL", "FAIL", tier_level=2) is False

    def test_with_smes(self):
        assert trigger_debate("PASS", "FAIL", tier_level=3, available_smes=["sme1"]) is True


# =============================================================================
# get_debate_participants Tests
# =============================================================================

class TestGetDebateParticipants:
    def test_base_participants(self):
        result = get_debate_participants(tier_level=2)
        assert "Executor" in result["agents"]
        assert "Critic" in result["agents"]
        assert "Verifier" in result["agents"]
        assert result["smes"] == []

    def test_tier3_with_smes(self):
        result = get_debate_participants(tier_level=3, available_smes=["sme1", "sme2"])
        assert "sme1" in result["smes"]
        assert "sme2" in result["smes"]

    def test_tier2_no_smes_even_if_provided(self):
        result = get_debate_participants(tier_level=2, available_smes=["sme1"])
        assert result["smes"] == []

    def test_max_3_smes(self):
        result = get_debate_participants(tier_level=4, available_smes=["s1", "s2", "s3", "s4"])
        assert len(result["smes"]) == 3

    def test_no_smes_provided(self):
        result = get_debate_participants(tier_level=3, available_smes=None)
        assert result["smes"] == []
