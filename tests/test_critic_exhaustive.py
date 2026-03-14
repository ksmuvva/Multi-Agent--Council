"""
Exhaustive Tests for CriticAgent

Tests all methods of the CriticAgent including initialization,
attack vectors, conversion methods, assessment, and convenience functions.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import patch, mock_open

from src.agents.critic import CriticAgent, ArgumentAnalysis, create_critic
from src.schemas.critic import (
    CritiqueReport,
    Attack,
    AttackVector,
    LogicAttack,
    CompletenessAttack,
    QualityAttack,
    ContradictionScan,
    RedTeamArgument,
    SeverityLevel,
)


# =============================================================================
# Schema Tests
# =============================================================================

class TestAttackVectorEnum:
    """Tests for AttackVector enum."""

    def test_all_values(self):
        assert AttackVector.LOGIC == "logic"
        assert AttackVector.COMPLETENESS == "completeness"
        assert AttackVector.QUALITY == "quality"
        assert AttackVector.CONTRADICTION == "contradiction"
        assert AttackVector.RED_TEAM == "red_team"

    def test_count(self):
        assert len(AttackVector) == 5


class TestSeverityLevelEnum:
    """Tests for SeverityLevel enum."""

    def test_all_values(self):
        assert SeverityLevel.CRITICAL == "critical"
        assert SeverityLevel.HIGH == "high"
        assert SeverityLevel.MEDIUM == "medium"
        assert SeverityLevel.LOW == "low"

    def test_count(self):
        assert len(SeverityLevel) == 4


class TestAttackSchema:
    """Tests for Attack Pydantic model."""

    def test_minimal_attack(self):
        a = Attack(
            vector=AttackVector.LOGIC,
            target="Logic",
            finding="A finding",
            severity=SeverityLevel.HIGH,
            description="desc",
            scenario="scenario",
            suggestion="suggestion",
        )
        assert a.vector == AttackVector.LOGIC
        assert a.domain_specific is False
        assert a.sme_source is None

    def test_domain_specific_attack(self):
        a = Attack(
            vector=AttackVector.RED_TEAM,
            target="Domain",
            finding="domain finding",
            severity=SeverityLevel.HIGH,
            description="desc",
            scenario="scenario",
            suggestion="suggestion",
            domain_specific=True,
            sme_source="cloud_architect",
        )
        assert a.domain_specific is True
        assert a.sme_source == "cloud_architect"


class TestLogicAttackSchema:
    """Tests for LogicAttack model."""

    def test_creation(self):
        la = LogicAttack(
            invalid_arguments=["arg1"],
            fallacies_identified=["fallacy1"],
        )
        assert la.valid_arguments == []
        assert la.invalid_arguments == ["arg1"]

    def test_empty_lists(self):
        la = LogicAttack(invalid_arguments=[], fallacies_identified=[])
        assert la.invalid_arguments == []
        assert la.fallacies_identified == []


class TestCompletenessAttackSchema:
    """Tests for CompletenessAttack model."""

    def test_creation(self):
        ca = CompletenessAttack(covered=["a"], missing=["b"], assumptions=["c"])
        assert ca.covered == ["a"]
        assert ca.missing == ["b"]
        assert ca.assumptions == ["c"]


class TestQualityAttackSchema:
    """Tests for QualityAttack model."""

    def test_creation(self):
        qa = QualityAttack(weaknesses=["w1"], improvements=["i1"])
        assert qa.strengths == []
        assert qa.weaknesses == ["w1"]


class TestContradictionScanSchema:
    """Tests for ContradictionScan model."""

    def test_creation(self):
        cs = ContradictionScan(
            external_contradictions=["e1"],
            inconsistencies=["i1"],
        )
        assert cs.internal_contradictions == []
        assert cs.external_contradictions == ["e1"]


class TestRedTeamArgumentSchema:
    """Tests for RedTeamArgument model."""

    def test_creation(self):
        rta = RedTeamArgument(
            adversary_perspective="adv",
            attack_surface=["s1"],
            failure_modes=["f1"],
            worst_case_scenarios=["w1"],
        )
        assert rta.adversary_perspective == "adv"


class TestCritiqueReportSchema:
    """Tests for CritiqueReport model."""

    def test_creation(self):
        report = CritiqueReport(
            solution_summary="summary",
            attacks=[],
            logic_attack=LogicAttack(invalid_arguments=[], fallacies_identified=[]),
            completeness_attack=CompletenessAttack(covered=[], missing=[], assumptions=[]),
            quality_attack=QualityAttack(weaknesses=[], improvements=[]),
            contradiction_scan=ContradictionScan(
                external_contradictions=[], inconsistencies=[]
            ),
            red_team_argumentation=RedTeamArgument(
                adversary_perspective="p",
                attack_surface=[],
                failure_modes=[],
                worst_case_scenarios=[],
            ),
            overall_assessment="ok",
            critical_issues=[],
            recommended_revisions=[],
            would_approve=True,
        )
        assert report.would_approve is True


# =============================================================================
# CriticAgent.__init__ Tests
# =============================================================================

class TestCriticAgentInit:
    """Tests for CriticAgent initialization."""

    def test_defaults(self):
        with patch("builtins.open", side_effect=FileNotFoundError):
            agent = CriticAgent()
        assert agent.system_prompt_path == "config/agents/critic/CLAUDE.md"
        assert agent.model == "claude-3-5-opus-20240507"
        assert agent.max_turns == 30

    def test_custom_params(self):
        with patch("builtins.open", side_effect=FileNotFoundError):
            agent = CriticAgent(
                system_prompt_path="custom/path.md",
                model="claude-3-haiku",
                max_turns=10,
            )
        assert agent.system_prompt_path == "custom/path.md"
        assert agent.model == "claude-3-haiku"
        assert agent.max_turns == 10

    def test_system_prompt_loaded_from_file(self):
        prompt_content = "You are a test critic."
        with patch("builtins.open", mock_open(read_data=prompt_content)):
            agent = CriticAgent()
        assert agent.system_prompt == prompt_content

    def test_system_prompt_fallback_on_missing_file(self):
        with patch("builtins.open", side_effect=FileNotFoundError):
            agent = CriticAgent()
        assert "Critic" in agent.system_prompt
        assert "five vectors" in agent.system_prompt

    def test_fallacy_patterns_initialized(self):
        with patch("builtins.open", side_effect=FileNotFoundError):
            agent = CriticAgent()
        assert "ad_hominem" in agent.fallacy_patterns
        assert "straw_man" in agent.fallacy_patterns
        assert "slippery_slope" in agent.fallacy_patterns
        assert "appeal_to_authority" in agent.fallacy_patterns
        assert "circular_reasoning" in agent.fallacy_patterns
        assert len(agent.fallacy_patterns) == 5


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def critic():
    """Create a CriticAgent with mocked file loading."""
    with patch("builtins.open", side_effect=FileNotFoundError):
        return CriticAgent()


# =============================================================================
# _analyze_argument_structure Tests
# =============================================================================

class TestAnalyzeArgumentStructure:
    """Tests for _analyze_argument_structure."""

    def test_extracts_premises_and_conclusion(self, critic):
        solution = (
            "Python is a great language. "
            "It has many libraries. "
            "Therefore Python is suitable for data science."
        )
        analysis = critic._analyze_argument_structure(solution)
        assert isinstance(analysis, ArgumentAnalysis)
        assert len(analysis.premises) >= 1
        assert "therefore" in analysis.conclusion.lower() or "Python" in analysis.conclusion

    def test_conclusion_from_thus(self, critic):
        solution = "A is true. B is true. Thus the answer is clear enough."
        analysis = critic._analyze_argument_structure(solution)
        assert "thus" in analysis.conclusion.lower()

    def test_conclusion_from_so(self, critic):
        solution = "Data is valid. Process works. So the system is correct enough."
        analysis = critic._analyze_argument_structure(solution)
        assert "so" in analysis.conclusion.lower()

    def test_conclusion_from_consequently(self, critic):
        solution = "Input is clean. Logic is sound. Consequently the output is reliable enough."
        analysis = critic._analyze_argument_structure(solution)
        assert "consequently" in analysis.conclusion.lower()

    def test_no_conclusion(self, critic):
        solution = "Point one is valid. Point two is valid. Point three is valid."
        analysis = critic._analyze_argument_structure(solution)
        assert analysis.conclusion == ""

    def test_causal_reasoning(self, critic):
        solution = "We chose this approach because it is faster and more reliable."
        analysis = critic._analyze_argument_structure(solution)
        assert analysis.logical_structure == "causal reasoning"

    def test_causal_reasoning_since(self, critic):
        solution = "Since the data is available, we can proceed with analysis."
        analysis = critic._analyze_argument_structure(solution)
        assert analysis.logical_structure == "causal reasoning"

    def test_declarative_statements(self, critic):
        solution = "The system processes data. It returns results."
        analysis = critic._analyze_argument_structure(solution)
        assert analysis.logical_structure == "declarative statements"

    def test_short_sentences_filtered(self, critic):
        solution = "Yes. No. The system handles complex edge cases properly."
        analysis = critic._analyze_argument_structure(solution)
        # "Yes" and "No" are < 10 chars, should be filtered
        for p in analysis.premises:
            assert len(p) > 10

    def test_assumptions_empty(self, critic):
        solution = "A simple statement about the system."
        analysis = critic._analyze_argument_structure(solution)
        assert analysis.assumptions == []


# =============================================================================
# _logic_attack Tests
# =============================================================================

class TestLogicAttack:
    """Tests for _logic_attack."""

    def test_no_fallacies_no_issues(self, critic):
        solution = "The system processes data efficiently. It handles errors well."
        analysis = critic._analyze_argument_structure(solution)
        result = critic._logic_attack(solution, analysis)
        assert isinstance(result, LogicAttack)
        assert len(result.fallacies_identified) == 0

    def test_ad_hominem_detected(self, critic):
        solution = "He is a fool who knows nothing. The approach is wrong."
        analysis = critic._analyze_argument_structure(solution)
        result = critic._logic_attack(solution, analysis)
        found = any("ad hominem" in f.lower() for f in result.fallacies_identified)
        assert found

    def test_straw_man_detected(self, critic):
        solution = "This is just a bad idea that nobody should use."
        analysis = critic._analyze_argument_structure(solution)
        result = critic._logic_attack(solution, analysis)
        found = any("straw man" in f.lower() for f in result.fallacies_identified)
        assert found

    def test_slippery_slope_detected(self, critic):
        solution = "Using this feature will lead to chaos and inevitably cause failure."
        analysis = critic._analyze_argument_structure(solution)
        result = critic._logic_attack(solution, analysis)
        found = any("slippery slope" in f.lower() for f in result.fallacies_identified)
        assert found

    def test_appeal_to_authority_detected(self, critic):
        solution = "We should do it because the expert said it was the best way."
        analysis = critic._analyze_argument_structure(solution)
        result = critic._logic_attack(solution, analysis)
        found = any("appeal to authority" in f.lower() for f in result.fallacies_identified)
        assert found

    def test_circular_reasoning_detected(self, critic):
        solution = "We know it works therefore it is good because it works therefore we use it."
        analysis = critic._analyze_argument_structure(solution)
        result = critic._logic_attack(solution, analysis)
        found = any("circular reasoning" in f.lower() for f in result.fallacies_identified)
        assert found

    def test_conclusion_without_premises(self, critic):
        # Create an analysis with no premises but a conclusion
        analysis = ArgumentAnalysis(
            premises=[],
            conclusion="Therefore this is correct",
            logical_structure="declarative statements",
            completeness_score=0.5,
            quality_score=0.5,
            assumptions=[],
        )
        result = critic._logic_attack("Therefore this is correct", analysis)
        assert "Conclusion without supporting premises" in result.invalid_arguments

    def test_logical_contradictions_not_detected_due_to_loop_logic(self, critic):
        # Note: _has_logical_contradictions has a loop that breaks when i >= j,
        # which means it effectively never compares sentence pairs (j starts at 0,
        # i >= 0 is always True). So contradictions are never detected.
        solution = "The system is not reliable. The system is reliable and works well."
        analysis = critic._analyze_argument_structure(solution)
        result = critic._logic_attack(solution, analysis)
        assert "Internal logical contradictions detected" not in result.invalid_arguments

    def test_valid_arguments_empty_by_default(self, critic):
        solution = "Some normal text about the solution."
        analysis = critic._analyze_argument_structure(solution)
        result = critic._logic_attack(solution, analysis)
        assert result.valid_arguments == []


# =============================================================================
# _completeness_attack Tests
# =============================================================================

class TestCompletenessAttack:
    """Tests for _completeness_attack."""

    def test_all_aspects_covered(self, critic):
        solution = (
            "Handle errors with try/except. "
            "Consider edge cases at the boundary. "
            "Document all functions. "
            "Test and verify correctness. "
            "Add security with auth permissions. "
            "Optimize for performance and efficiency."
        )
        result = critic._completeness_attack(solution, "build something")
        assert "error_handling" in result.covered
        assert "edge_cases" in result.covered
        assert "documentation" in result.covered
        assert "testing" in result.covered
        assert "security" in result.covered
        assert "performance" in result.covered

    def test_missing_aspects(self, critic):
        solution = "A short solution without details."
        result = critic._completeness_attack(solution, "build a system")
        assert len(result.missing) > 0
        assert "error_handling" in result.missing
        assert "security" in result.missing

    @pytest.mark.parametrize("aspect,keyword", [
        ("error_handling", "error"),
        ("edge_cases", "edge case"),
        ("documentation", "document"),
        ("testing", "test"),
        ("security", "security"),
        ("performance", "performance"),
    ])
    def test_individual_coverage(self, critic, aspect, keyword):
        solution = f"We address {keyword} in this solution."
        result = critic._completeness_attack(solution, "request")
        assert aspect in result.covered

    def test_requirements_from_original_request(self, critic):
        solution = "A basic implementation."
        request = "Create a system with authentication and logging"
        result = critic._completeness_attack(solution, request)
        # "authentication" and "logging" should show up as missing
        missing_text = " ".join(result.missing).lower()
        assert "authentication" in missing_text or "logging" in missing_text

    def test_assumptions_empty(self, critic):
        result = critic._completeness_attack("some solution", "some request")
        assert result.assumptions == []


# =============================================================================
# _quality_attack Tests
# =============================================================================

class TestQualityAttack:
    """Tests for _quality_attack."""

    def test_strength_clear(self, critic):
        solution = "This is a clear and well-structured approach."
        analysis = critic._analyze_argument_structure(solution)
        result = critic._quality_attack(solution, analysis)
        assert any("clear" in s.lower() for s in result.strengths)

    def test_strength_examples(self, critic):
        solution = "Here is an example of how it works with a demonstration."
        analysis = critic._analyze_argument_structure(solution)
        result = critic._quality_attack(solution, analysis)
        assert any("example" in s.lower() for s in result.strengths)

    def test_strength_best_practice(self, critic):
        solution = "Following best practice we implement this pattern."
        analysis = critic._analyze_argument_structure(solution)
        result = critic._quality_attack(solution, analysis)
        assert any("best practice" in s.lower() for s in result.strengths)

    def test_weakness_short(self, critic):
        solution = "Do this.\nDo that.\nDone."
        analysis = critic._analyze_argument_structure(solution)
        result = critic._quality_attack(solution, analysis)
        assert any("short" in w.lower() for w in result.weaknesses)

    def test_weakness_vague(self, critic):
        solution = "The system could work and might be useful.\n" * 6
        analysis = critic._analyze_argument_structure(solution)
        result = critic._quality_attack(solution, analysis)
        assert any("vague" in w.lower() for w in result.weaknesses)

    def test_weakness_placeholders(self, critic):
        solution = "Step 1: TODO implement this.\n" * 6
        analysis = critic._analyze_argument_structure(solution)
        result = critic._quality_attack(solution, analysis)
        assert any("placeholder" in w.lower() for w in result.weaknesses)

    def test_improvements_generated_when_weaknesses(self, critic):
        solution = "Short.\nDone."
        analysis = critic._analyze_argument_structure(solution)
        result = critic._quality_attack(solution, analysis)
        assert len(result.improvements) > 0

    def test_improvement_error_handling(self, critic):
        solution = "A solution without any mention of failures.\n" * 6
        analysis = critic._analyze_argument_structure(solution)
        result = critic._quality_attack(solution, analysis)
        assert any("error" in i.lower() for i in result.improvements)

    def test_improvement_documentation(self, critic):
        solution = "Implementation code goes here.\n" * 6
        analysis = critic._analyze_argument_structure(solution)
        result = critic._quality_attack(solution, analysis)
        assert any("document" in i.lower() for i in result.improvements)


# =============================================================================
# _contradiction_scan Tests
# =============================================================================

class TestContradictionScan:
    """Tests for _contradiction_scan."""

    def test_no_contradictions(self, critic):
        solution = "The system works well. It processes data correctly."
        result = critic._contradiction_scan(solution)
        assert len(result.internal_contradictions) == 0
        assert len(result.external_contradictions) == 0

    @pytest.mark.parametrize("text,expected_desc", [
        ("Always use this pattern. But never apply it.", "Claims 'always' and 'never' for same thing"),
        ("Never do X. Always do X instead.", "Claims 'never' and 'always' for same thing"),
        ("All systems should do none of those things.", "Claims 'all' and 'none' together"),
        ("None of these matter. All are important.", "Claims 'none' and 'all' together"),
        ("This is impossible but also possible to achieve.", "Logical impossibility"),
        ("This is possible. It is also impossible to do.", "Logical impossibility"),
    ])
    def test_internal_contradiction_patterns(self, critic, text, expected_desc):
        result = critic._contradiction_scan(text)
        assert expected_desc in result.internal_contradictions

    def test_external_contradiction_prove_impossible_literal(self, critic):
        # The code checks `"prove.*impossible" in solution_lower` which is a
        # literal string match, not a regex. So only exact substring matches.
        solution = "We can prove.*impossible results with this method."
        result = critic._contradiction_scan(solution)
        assert len(result.external_contradictions) > 0

    def test_external_contradiction_no_match_without_literal(self, critic):
        # Without the literal "prove.*impossible" substring, no match occurs.
        solution = "We can prove that impossible results exist."
        result = critic._contradiction_scan(solution)
        assert len(result.external_contradictions) == 0

    def test_inconsistencies_populated_when_contradictions_exist(self, critic):
        solution = "Always use X. Never use X."
        result = critic._contradiction_scan(solution)
        assert len(result.inconsistencies) > 0
        assert "contradictory" in result.inconsistencies[0].lower()

    def test_no_inconsistencies_when_clean(self, critic):
        solution = "Step one. Step two. Step three."
        result = critic._contradiction_scan(solution)
        assert len(result.inconsistencies) == 0


# =============================================================================
# _red_team_argumentation Tests
# =============================================================================

class TestRedTeamArgumentation:
    """Tests for _red_team_argumentation."""

    def test_returns_red_team_argument(self, critic):
        result = critic._red_team_argumentation("A solution", "A request")
        assert isinstance(result, RedTeamArgument)

    def test_adversary_perspective(self, critic):
        result = critic._red_team_argumentation("solution text", "my request here")
        assert "adversary" in result.adversary_perspective.lower()
        assert "my request" in result.adversary_perspective

    @pytest.mark.parametrize("keyword,expected_fragment", [
        ("user input", "User input handling"),
        ("file", "File/database operations"),
        ("database", "File/database operations"),
        ("network", "Network calls"),
        ("api", "Network calls"),
        ("authentication", "Authentication"),
        ("auth", "Authentication"),
    ])
    def test_attack_surface_keywords(self, critic, keyword, expected_fragment):
        solution = f"The system uses {keyword} for processing."
        result = critic._red_team_argumentation(solution, "request")
        found = any(expected_fragment in s for s in result.attack_surface)
        assert found, f"Expected '{expected_fragment}' in attack surface for '{keyword}'"

    def test_data_without_encryption(self, critic):
        solution = "The system processes data and stores results."
        result = critic._red_team_argumentation(solution, "request")
        found = any("exposure" in s.lower() for s in result.attack_surface)
        assert found

    def test_no_attack_surface_fallback(self, critic):
        solution = "A simple statement."
        result = critic._red_team_argumentation(solution, "request")
        assert any("unverified" in s.lower() for s in result.attack_surface)

    def test_failure_modes_always_present(self, critic):
        result = critic._red_team_argumentation("any solution", "any request")
        assert len(result.failure_modes) >= 5

    def test_failure_mode_database(self, critic):
        solution = "The database stores all records."
        result = critic._red_team_argumentation(solution, "request")
        assert any("database" in m.lower() for m in result.failure_modes)

    def test_failure_mode_api(self, critic):
        solution = "We call the API for results."
        result = critic._red_team_argumentation(solution, "request")
        assert any("api" in m.lower() for m in result.failure_modes)

    def test_failure_mode_async(self, critic):
        solution = "The async operations run concurrently."
        result = critic._red_team_argumentation(solution, "request")
        assert any("race" in m.lower() or "async" in m.lower() for m in result.failure_modes)

    def test_worst_case_scenarios(self, critic):
        result = critic._red_team_argumentation("solution", "request")
        assert len(result.worst_case_scenarios) >= 2
        scenarios_text = " ".join(result.worst_case_scenarios).lower()
        assert "failure" in scenarios_text

    def test_worst_case_scenarios_domain_specific(self, critic):
        result = critic._red_team_argumentation(
            "deploy database api with async auth", "request")
        scenarios_text = " ".join(result.worst_case_scenarios).lower()
        # Should include domain-specific scenarios for database, api, async, auth
        assert len(result.worst_case_scenarios) >= 4


# =============================================================================
# _has_logical_contradictions Tests
# =============================================================================

class TestHasLogicalContradictions:
    """Tests for _has_logical_contradictions."""

    def test_no_contradictions(self, critic):
        assert critic._has_logical_contradictions("The sky is blue. Grass is green.") is False

    def test_not_x_vs_x_not_detected(self, critic):
        # The inner loop breaks immediately (i >= j when j=0) so no pairs compared
        solution = "The system is not reliable. The system is reliable and works."
        assert critic._has_logical_contradictions(solution) is False

    def test_single_sentence_no_contradiction(self, critic):
        assert critic._has_logical_contradictions("Everything works fine.") is False


# =============================================================================
# _extract_requirements Tests
# =============================================================================

class TestExtractRequirements:
    """Tests for _extract_requirements."""

    def test_significant_words(self, critic):
        result = critic._extract_requirements("Create a robust system with logging")
        # Words > 3 chars and not common
        assert "create" in result
        assert "robust" in result

    def test_filters_common_words(self, critic):
        result = critic._extract_requirements("this that with from have will would could")
        assert "this" not in result
        assert "that" not in result
        assert "with" not in result
        assert "from" not in result

    def test_filters_short_words(self, critic):
        result = critic._extract_requirements("a to is me do it")
        assert len(result) == 0

    def test_empty_request(self, critic):
        result = critic._extract_requirements("")
        assert result == []


# =============================================================================
# _assess_completeness_score Tests
# =============================================================================

class TestAssessCompletenessScore:
    """Tests for _assess_completeness_score."""

    def test_base_score(self, critic):
        score = critic._assess_completeness_score("short", [])
        assert score == 0.5

    def test_ten_lines_bonus(self, critic):
        solution = "\n".join([f"Line {i}" for i in range(11)])
        score = critic._assess_completeness_score(solution, [])
        assert score >= 0.7

    def test_twenty_lines_bonus(self, critic):
        solution = "\n".join([f"Line {i}" for i in range(21)])
        score = critic._assess_completeness_score(solution, [])
        # 0.5 base + 0.2 (>=10 lines) + 0.2 (>=20 lines) = 0.9
        assert score == pytest.approx(0.9, abs=0.01)

    def test_premises_bonus(self, critic):
        score = critic._assess_completeness_score("short", ["premise1", "premise2"])
        assert score == 0.6

    def test_capped_at_one(self, critic):
        solution = "\n".join([f"Line {i}" for i in range(30)])
        score = critic._assess_completeness_score(solution, ["p1", "p2", "p3"])
        assert score <= 1.0


# =============================================================================
# _assess_quality_score Tests
# =============================================================================

class TestAssessQualityScore:
    """Tests for _assess_quality_score."""

    def test_base_score(self, critic):
        score = critic._assess_quality_score("a plain statement")
        assert score == 0.5

    def test_example_bonus(self, critic):
        score = critic._assess_quality_score("Here is an example of usage")
        assert score >= 0.65

    def test_explain_bonus(self, critic):
        score = critic._assess_quality_score("Let me explain the approach")
        assert score >= 0.65

    def test_best_practice_bonus(self, critic):
        score = critic._assess_quality_score("This follows best practice guidelines")
        assert score >= 0.6

    def test_sequence_words_bonus(self, critic):
        score = critic._assess_quality_score("First do this, then do that, next proceed")
        # "first", "then", "next" all present but only one +0.1 bonus applies
        # since it's a single check: if any of first/next/then in text -> +0.1
        assert score >= 0.6

    def test_todo_penalty(self, critic):
        score = critic._assess_quality_score("todo: implement this feature")
        assert score <= 0.3

    def test_tbd_penalty(self, critic):
        score = critic._assess_quality_score("tbd: decide later what to use")
        assert score <= 0.3

    def test_clamped_to_zero(self, critic):
        score = critic._assess_quality_score("todo tbd quick")
        assert score >= 0.0

    def test_clamped_to_one(self, critic):
        score = critic._assess_quality_score(
            "example explain best practice first then next recommended"
        )
        assert score <= 1.0


# =============================================================================
# Conversion to Attack Lists Tests
# =============================================================================

class TestLogicAttackToList:
    """Tests for _logic_attack_to_list."""

    def test_invalid_arguments_create_high_severity(self, critic):
        la = LogicAttack(
            invalid_arguments=["arg1", "arg2"],
            fallacies_identified=[],
        )
        attacks = critic._logic_attack_to_list(la)
        assert len(attacks) == 2
        assert all(a.vector == AttackVector.LOGIC for a in attacks)
        assert all(a.severity == SeverityLevel.HIGH for a in attacks)

    def test_fallacies_create_medium_severity(self, critic):
        la = LogicAttack(
            invalid_arguments=[],
            fallacies_identified=["fallacy1"],
        )
        attacks = critic._logic_attack_to_list(la)
        assert len(attacks) == 1
        assert attacks[0].severity == SeverityLevel.MEDIUM
        assert "Fallacy:" in attacks[0].finding

    def test_empty_inputs(self, critic):
        la = LogicAttack(invalid_arguments=[], fallacies_identified=[])
        attacks = critic._logic_attack_to_list(la)
        assert attacks == []


class TestCompletenessAttackToList:
    """Tests for _completeness_attack_to_list."""

    def test_missing_create_medium_severity(self, critic):
        ca = CompletenessAttack(
            covered=[], missing=["missing1", "missing2"], assumptions=[]
        )
        attacks = critic._completeness_attack_to_list(ca)
        assert len(attacks) == 2
        assert all(a.vector == AttackVector.COMPLETENESS for a in attacks)
        assert all(a.severity == SeverityLevel.MEDIUM for a in attacks)

    def test_limits_to_five(self, critic):
        ca = CompletenessAttack(
            covered=[],
            missing=[f"missing{i}" for i in range(10)],
            assumptions=[],
        )
        attacks = critic._completeness_attack_to_list(ca)
        assert len(attacks) == 5

    def test_empty_missing(self, critic):
        ca = CompletenessAttack(covered=["a"], missing=[], assumptions=[])
        attacks = critic._completeness_attack_to_list(ca)
        assert attacks == []


class TestQualityAttackToList:
    """Tests for _quality_attack_to_list."""

    def test_weaknesses_create_low_severity(self, critic):
        qa = QualityAttack(weaknesses=["w1"], improvements=[])
        attacks = critic._quality_attack_to_list(qa)
        assert len(attacks) == 1
        assert attacks[0].severity == SeverityLevel.LOW
        assert attacks[0].finding == "w1"

    def test_improvements_create_low_severity(self, critic):
        qa = QualityAttack(weaknesses=[], improvements=["imp1"])
        attacks = critic._quality_attack_to_list(qa)
        assert len(attacks) == 1
        assert "Improvement needed:" in attacks[0].finding

    def test_combined(self, critic):
        qa = QualityAttack(weaknesses=["w1"], improvements=["i1"])
        attacks = critic._quality_attack_to_list(qa)
        assert len(attacks) == 2


class TestContradictionToList:
    """Tests for _contradiction_to_list."""

    def test_internal_contradictions_high_severity(self, critic):
        cs = ContradictionScan(
            internal_contradictions=["contra1"],
            external_contradictions=[],
            inconsistencies=[],
        )
        attacks = critic._contradiction_to_list(cs)
        assert len(attacks) == 1
        assert attacks[0].severity == SeverityLevel.HIGH
        assert attacks[0].vector == AttackVector.CONTRADICTION

    def test_external_contradictions_high_severity(self, critic):
        cs = ContradictionScan(
            internal_contradictions=[],
            external_contradictions=["ext1"],
            inconsistencies=[],
        )
        attacks = critic._contradiction_to_list(cs)
        assert len(attacks) == 1
        assert attacks[0].severity == SeverityLevel.HIGH

    def test_inconsistencies_medium_severity(self, critic):
        cs = ContradictionScan(
            internal_contradictions=[],
            external_contradictions=[],
            inconsistencies=["inc1"],
        )
        attacks = critic._contradiction_to_list(cs)
        assert len(attacks) == 1
        assert attacks[0].severity == SeverityLevel.MEDIUM

    def test_empty(self, critic):
        cs = ContradictionScan(
            internal_contradictions=[],
            external_contradictions=[],
            inconsistencies=[],
        )
        attacks = critic._contradiction_to_list(cs)
        assert attacks == []


class TestRedTeamToList:
    """Tests for _red_team_to_list."""

    def test_attack_surfaces_medium_severity(self, critic):
        rta = RedTeamArgument(
            adversary_perspective="adv",
            attack_surface=["surface1"],
            failure_modes=[],
            worst_case_scenarios=[],
        )
        attacks = critic._red_team_to_list(rta)
        assert len(attacks) == 1
        assert attacks[0].severity == SeverityLevel.MEDIUM
        assert attacks[0].vector == AttackVector.RED_TEAM

    def test_failure_modes_high_severity(self, critic):
        rta = RedTeamArgument(
            adversary_perspective="adv",
            attack_surface=[],
            failure_modes=["fail1"],
            worst_case_scenarios=[],
        )
        attacks = critic._red_team_to_list(rta)
        assert len(attacks) == 1
        assert attacks[0].severity == SeverityLevel.HIGH
        assert attacks[0].target == "Reliability"

    def test_combined(self, critic):
        rta = RedTeamArgument(
            adversary_perspective="adv",
            attack_surface=["s1", "s2"],
            failure_modes=["f1"],
            worst_case_scenarios=["w1"],
        )
        attacks = critic._red_team_to_list(rta)
        assert len(attacks) == 3  # 2 surfaces + 1 failure mode


class TestDomainAttacksToList:
    """Tests for _domain_attacks_to_list."""

    def test_basic_domain_attacks(self, critic):
        domain_attacks = ["SQL injection risk"]
        sme_inputs = {"security_analyst": "SQL injection risk in the query"}
        attacks = critic._domain_attacks_to_list(domain_attacks, sme_inputs)
        assert len(attacks) == 1
        assert attacks[0].domain_specific is True
        assert attacks[0].sme_source == "security_analyst"
        assert attacks[0].severity == SeverityLevel.HIGH

    def test_unknown_sme_source(self, critic):
        domain_attacks = ["missing encryption"]
        sme_inputs = {"other_sme": "different input"}
        attacks = critic._domain_attacks_to_list(domain_attacks, sme_inputs)
        assert attacks[0].sme_source == "Unknown"


# =============================================================================
# _generate_overall_assessment Tests
# =============================================================================

class TestGenerateOverallAssessment:
    """Tests for _generate_overall_assessment."""

    def test_critical_issues(self, critic):
        attacks = [
            Attack(
                vector=AttackVector.LOGIC, target="t", finding="f",
                severity=SeverityLevel.CRITICAL, description="d",
                scenario="s", suggestion="su",
            )
        ]
        analysis = ArgumentAnalysis([], "", "", 0.5, 0.5, [])
        result = critic._generate_overall_assessment(attacks, "sol", analysis)
        assert "critical" in result.lower()
        assert "1" in result

    def test_high_issues(self, critic):
        attacks = [
            Attack(
                vector=AttackVector.LOGIC, target="t", finding="f",
                severity=SeverityLevel.HIGH, description="d",
                scenario="s", suggestion="su",
            )
            for _ in range(4)
        ]
        analysis = ArgumentAnalysis([], "", "", 0.5, 0.5, [])
        result = critic._generate_overall_assessment(attacks, "sol", analysis)
        assert "high-priority" in result.lower()

    def test_any_issues(self, critic):
        attacks = [
            Attack(
                vector=AttackVector.LOGIC, target="t", finding="f",
                severity=SeverityLevel.LOW, description="d",
                scenario="s", suggestion="su",
            )
        ]
        analysis = ArgumentAnalysis([], "", "", 0.5, 0.5, [])
        result = critic._generate_overall_assessment(attacks, "sol", analysis)
        assert "issues" in result.lower()

    def test_no_issues(self, critic):
        analysis = ArgumentAnalysis([], "", "", 0.5, 0.5, [])
        result = critic._generate_overall_assessment([], "sol", analysis)
        assert "sound" in result.lower()


# =============================================================================
# _identify_critical_issues Tests
# =============================================================================

class TestIdentifyCriticalIssues:
    """Tests for _identify_critical_issues."""

    def test_filters_critical_only(self, critic):
        attacks = [
            Attack(
                vector=AttackVector.LOGIC, target="t", finding="critical finding",
                severity=SeverityLevel.CRITICAL, description="d",
                scenario="s", suggestion="su",
            ),
            Attack(
                vector=AttackVector.LOGIC, target="t", finding="high finding",
                severity=SeverityLevel.HIGH, description="d",
                scenario="s", suggestion="su",
            ),
        ]
        result = critic._identify_critical_issues(attacks)
        assert result == ["critical finding"]

    def test_empty_when_no_critical(self, critic):
        attacks = [
            Attack(
                vector=AttackVector.LOGIC, target="t", finding="f",
                severity=SeverityLevel.LOW, description="d",
                scenario="s", suggestion="su",
            ),
        ]
        result = critic._identify_critical_issues(attacks)
        assert result == []


# =============================================================================
# _generate_revisions Tests
# =============================================================================

class TestGenerateRevisions:
    """Tests for _generate_revisions."""

    def test_critical_issues_first(self, critic):
        attacks = [
            Attack(
                vector=AttackVector.LOGIC, target="t", finding="high issue",
                severity=SeverityLevel.HIGH, description="d",
                scenario="s", suggestion="su",
            ),
        ]
        critical_issues = ["critical problem"]
        result = critic._generate_revisions(attacks, critical_issues)
        assert result[0].startswith("CRITICAL:")

    def test_high_issues_after_critical(self, critic):
        attacks = [
            Attack(
                vector=AttackVector.LOGIC, target="t", finding="high issue",
                severity=SeverityLevel.HIGH, description="d",
                scenario="s", suggestion="su",
            ),
        ]
        result = critic._generate_revisions(attacks, [])
        assert any("HIGH:" in r for r in result)

    def test_suggested_improvements(self, critic):
        attacks = [
            Attack(
                vector=AttackVector.QUALITY, target="t", finding="f",
                severity=SeverityLevel.MEDIUM, description="d",
                scenario="s", suggestion="improve this",
            ),
        ]
        result = critic._generate_revisions(attacks, [])
        assert any("SUGGESTED:" in r for r in result)

    def test_priority_ordering(self, critic):
        attacks = [
            Attack(
                vector=AttackVector.LOGIC, target="t", finding="high",
                severity=SeverityLevel.HIGH, description="d",
                scenario="s", suggestion="su",
            ),
            Attack(
                vector=AttackVector.QUALITY, target="t", finding="low",
                severity=SeverityLevel.LOW, description="d",
                scenario="s", suggestion="improve",
            ),
        ]
        critical = ["crit issue"]
        result = critic._generate_revisions(attacks, critical)
        # CRITICAL first, then HIGH, then SUGGESTED
        assert result[0].startswith("CRITICAL:")
        assert result[1].startswith("HIGH:")
        assert result[2].startswith("SUGGESTED:")


# =============================================================================
# _would_approve_solution Tests
# =============================================================================

class TestWouldApproveSolution:
    """Tests for _would_approve_solution."""

    def test_fails_on_critical_issues(self, critic):
        attacks = []
        critical_issues = ["a critical problem"]
        assert critic._would_approve_solution(attacks, critical_issues) is False

    def test_fails_on_many_high_severity(self, critic):
        attacks = [
            Attack(
                vector=AttackVector.LOGIC, target="t", finding="f",
                severity=SeverityLevel.HIGH, description="d",
                scenario="s", suggestion="su",
            )
            for _ in range(6)
        ]
        assert critic._would_approve_solution(attacks, []) is False

    def test_passes_with_five_high_severity(self, critic):
        attacks = [
            Attack(
                vector=AttackVector.LOGIC, target="t", finding="f",
                severity=SeverityLevel.HIGH, description="d",
                scenario="s", suggestion="su",
            )
            for _ in range(5)
        ]
        assert critic._would_approve_solution(attacks, []) is True

    def test_passes_with_no_issues(self, critic):
        assert critic._would_approve_solution([], []) is True

    def test_passes_with_low_severity_only(self, critic):
        attacks = [
            Attack(
                vector=AttackVector.QUALITY, target="t", finding="f",
                severity=SeverityLevel.LOW, description="d",
                scenario="s", suggestion="su",
            )
            for _ in range(20)
        ]
        assert critic._would_approve_solution(attacks, []) is True


# =============================================================================
# critique() Full Method Tests
# =============================================================================

class TestCritiqueFullMethod:
    """Tests for the full critique() method."""

    def test_produces_critique_report(self, critic):
        report = critic.critique(
            solution="This is a good solution with proper error handling.",
            original_request="Build a system",
        )
        assert isinstance(report, CritiqueReport)

    def test_solution_summary_truncated(self, critic):
        long_solution = "x" * 200
        report = critic.critique(solution=long_solution, original_request="request")
        assert report.solution_summary.endswith("...")
        assert len(report.solution_summary) == 103  # 100 + "..."

    def test_solution_summary_short(self, critic):
        short_solution = "short"
        report = critic.critique(solution=short_solution, original_request="request")
        assert report.solution_summary == "short"

    def test_all_attack_fields_populated(self, critic):
        report = critic.critique(
            solution="A reasonable solution with testing and documentation.",
            original_request="Create a module",
        )
        assert report.logic_attack is not None
        assert report.completeness_attack is not None
        assert report.quality_attack is not None
        assert report.contradiction_scan is not None
        assert report.red_team_argumentation is not None

    def test_domain_attacks_included(self, critic):
        report = critic.critique(
            solution="A solution with security features.",
            original_request="Build a secure system",
            domain_attacks=["Missing encryption at rest"],
            sme_inputs={"security_analyst": "Missing encryption at rest"},
        )
        domain_found = any(a.domain_specific for a in report.attacks)
        assert domain_found

    def test_no_domain_attacks(self, critic):
        report = critic.critique(
            solution="A solution.", original_request="request"
        )
        domain_found = any(a.domain_specific for a in report.attacks)
        assert not domain_found

    def test_overall_assessment_populated(self, critic):
        report = critic.critique(solution="Solution text.", original_request="request")
        assert len(report.overall_assessment) > 0

    def test_would_approve_field(self, critic):
        report = critic.critique(
            solution="A thorough solution with testing, documentation, error handling, "
                     "security, and performance optimization.\n" * 10,
            original_request="request",
        )
        assert isinstance(report.would_approve, bool)


# =============================================================================
# create_critic() Convenience Function Tests
# =============================================================================

class TestCreateCritic:
    """Tests for create_critic convenience function."""

    def test_creates_default(self):
        with patch("builtins.open", side_effect=FileNotFoundError):
            agent = create_critic()
        assert isinstance(agent, CriticAgent)
        assert agent.model == "claude-3-5-opus-20240507"

    def test_custom_params(self):
        with patch("builtins.open", side_effect=FileNotFoundError):
            agent = create_critic(
                system_prompt_path="custom.md",
                model="claude-3-haiku",
            )
        assert agent.system_prompt_path == "custom.md"
        assert agent.model == "claude-3-haiku"
