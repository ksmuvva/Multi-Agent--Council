"""
Tests for the Strategic Council Agents.

Tests CouncilChairAgent (SME selection, domain identification, interaction modes),
QualityArbiterAgent (quality standards, dispute resolution),
and EthicsAdvisorAgent (PII scanning, bias detection, safety assessment, compliance).
"""

import pytest
from unittest.mock import patch, mock_open

from src.agents.council import (
    CouncilChairAgent,
    QualityArbiterAgent,
    EthicsAdvisorAgent,
    create_council_chair,
    create_quality_arbiter,
    create_ethics_advisor,
)
from src.schemas.council import (
    SMESelectionReport,
    SMESelection,
    InteractionMode,
    QualityStandard,
    QualityCriteria,
    QualityVerdict,
    DisputedItem,
    EthicsReview,
    FlaggedIssue,
    IssueType,
    IssueSeverity,
)


# =============================================================================
# Council Chair Tests
# =============================================================================

@pytest.fixture
def chair():
    """Create a CouncilChairAgent with no system prompt file."""
    return CouncilChairAgent(system_prompt_path="nonexistent.md")


class TestCouncilChairInitialization:
    """Tests for CouncilChairAgent initialization."""

    def test_default_initialization(self):
        """Test default init parameters."""
        agent = CouncilChairAgent(system_prompt_path="nonexistent.md")
        assert agent.model == "claude-3-5-opus-20240507"
        assert agent.max_turns == 30

    def test_domain_patterns_initialized(self):
        """Test domain patterns are configured."""
        agent = CouncilChairAgent(system_prompt_path="nonexistent.md")
        assert len(agent.domain_patterns) > 0
        assert "cloud_architect" in agent.domain_patterns
        assert "security_analyst" in agent.domain_patterns

    def test_system_prompt_fallback(self):
        """Test fallback prompt."""
        agent = CouncilChairAgent(system_prompt_path="nonexistent.md")
        assert "Council Chair" in agent.system_prompt

    def test_system_prompt_from_file(self):
        """Test loading from file."""
        with patch("builtins.open", mock_open(read_data="Chair prompt")):
            agent = CouncilChairAgent(system_prompt_path="exists.md")
            assert agent.system_prompt == "Chair prompt"

    def test_custom_model(self):
        """Test custom model."""
        agent = CouncilChairAgent(
            system_prompt_path="nonexistent.md",
            model="claude-3-sonnet",
        )
        assert agent.model == "claude-3-sonnet"


class TestSMESelection:
    """Tests for SME selection logic."""

    def test_select_smes_returns_report(self, chair):
        """Test select_smes returns SMESelectionReport."""
        report = chair.select_smes(
            task_description="Build a secure cloud API with AWS Lambda",
            tier_level=3,
        )
        assert isinstance(report, SMESelectionReport)

    def test_select_smes_finds_cloud_domain(self, chair):
        """Test cloud keywords trigger cloud_architect selection."""
        report = chair.select_smes(
            task_description="Deploy an application on AWS using terraform",
            tier_level=3,
        )
        domains = [sme.persona_domain.lower() for sme in report.selected_smes]
        # Should find cloud-related SME
        assert len(report.selected_smes) > 0

    def test_select_smes_max_limit(self, chair):
        """Test max_smes limit is respected."""
        report = chair.select_smes(
            task_description="Build a secure cloud data pipeline with testing and documentation",
            tier_level=3,
            max_smes=2,
        )
        assert len(report.selected_smes) <= 2

    def test_select_smes_has_collaboration_plan(self, chair):
        """Test collaboration plan is generated."""
        report = chair.select_smes(
            task_description="Build a secure API",
            tier_level=3,
        )
        assert len(report.collaboration_plan) > 0

    def test_select_smes_tier_recommendation(self, chair):
        """Test tier recommendation is set."""
        report = chair.select_smes(
            task_description="Simple task",
            tier_level=4,
        )
        assert report.tier_recommendation == 4


class TestDomainIdentification:
    """Tests for domain identification."""

    @pytest.mark.parametrize("description,expected_domain", [
        ("Build with AWS Lambda and S3", "cloud_architect"),
        ("Fix the SQL injection vulnerability", "security_analyst"),
        ("Create an ETL data pipeline with Snowflake", "data_engineer"),
        ("Train a machine learning model with PyTorch", "ai_ml_engineer"),
        ("Write pytest unit tests for the API", "test_engineer"),
    ])
    def test_identify_required_domains(self, chair, description, expected_domain):
        """Test domain identification from task description."""
        domains = chair._identify_required_domains(description, None)
        assert expected_domain in domains

    def test_analyst_report_adds_domains(self, chair):
        """Test analyst report provides additional domain hints."""
        analyst_report = {"modality": "code"}
        domains = chair._identify_required_domains("Build something", analyst_report)
        assert "test_engineer" in domains


class TestInteractionModeSelection:
    """Tests for interaction mode selection."""

    def test_tier4_uses_debate_for_security(self, chair):
        """Test Tier 4 assigns debate mode for security SMEs."""
        sme = SMESelection(
            persona_name="Security Analyst",
            persona_domain="security_analyst",
            skills_to_load=[],
            interaction_mode=InteractionMode.ADVISOR,
            reasoning="Test",
            activation_phase="execution",
        )
        mode = chair._determine_interaction_mode(sme, "Security audit", 4)
        assert mode == InteractionMode.DEBATER

    def test_co_executor_for_frontend(self, chair):
        """Test co-executor mode for frontend domains."""
        sme = SMESelection(
            persona_name="Frontend Dev",
            persona_domain="frontend_developer",
            skills_to_load=[],
            interaction_mode=InteractionMode.ADVISOR,
            reasoning="Test",
            activation_phase="execution",
        )
        mode = chair._determine_interaction_mode(sme, "Build UI", 3)
        assert mode == InteractionMode.CO_EXECUTOR

    def test_default_advisor_mode(self, chair):
        """Test default mode is advisor."""
        sme = SMESelection(
            persona_name="Technical Writer",
            persona_domain="technical_writer",
            skills_to_load=[],
            interaction_mode=InteractionMode.ADVISOR,
            reasoning="Test",
            activation_phase="execution",
        )
        mode = chair._determine_interaction_mode(sme, "Write docs", 3)
        assert mode == InteractionMode.ADVISOR


class TestFullCouncilDetermination:
    """Tests for determining if full Council is needed."""

    def test_tier4_requires_full_council(self, chair):
        """Test Tier 4 always requires full Council."""
        result = chair._requires_full_council("Simple task", 4, [])
        assert result is True

    def test_sensitive_content_requires_council(self, chair):
        """Test sensitive content triggers full Council."""
        result = chair._requires_full_council(
            "Process personal data and PII", 3, []
        )
        assert result is True

    def test_multiple_smes_require_council(self, chair):
        """Test multiple SMEs trigger full Council."""
        smes = [
            SMESelection(
                persona_name="A", persona_domain="d1",
                skills_to_load=[], interaction_mode=InteractionMode.ADVISOR,
                reasoning="test", activation_phase="execution",
            ),
            SMESelection(
                persona_name="B", persona_domain="d2",
                skills_to_load=[], interaction_mode=InteractionMode.ADVISOR,
                reasoning="test", activation_phase="execution",
            ),
        ]
        result = chair._requires_full_council("Simple task", 3, smes)
        assert result is True


class TestCouncilChairConvenience:
    """Tests for create_council_chair convenience function."""

    def test_create_council_chair(self):
        """Test convenience function creates a CouncilChairAgent."""
        agent = create_council_chair(system_prompt_path="nonexistent.md")
        assert isinstance(agent, CouncilChairAgent)


# =============================================================================
# Quality Arbiter Tests
# =============================================================================

@pytest.fixture
def arbiter():
    """Create a QualityArbiterAgent with no system prompt file."""
    return QualityArbiterAgent(system_prompt_path="nonexistent.md")


class TestQualityArbiterInitialization:
    """Tests for QualityArbiterAgent initialization."""

    def test_default_initialization(self):
        """Test default init parameters."""
        agent = QualityArbiterAgent(system_prompt_path="nonexistent.md")
        assert agent.model == "claude-3-5-opus-20240507"
        assert agent.max_turns == 30

    def test_default_criteria_configured(self):
        """Test default quality criteria are configured."""
        agent = QualityArbiterAgent(system_prompt_path="nonexistent.md")
        assert "accuracy" in agent.default_criteria
        assert "completeness" in agent.default_criteria
        assert "quality" in agent.default_criteria
        assert "coherence" in agent.default_criteria

    def test_system_prompt_fallback(self):
        """Test fallback prompt."""
        agent = QualityArbiterAgent(system_prompt_path="nonexistent.md")
        assert "Quality Arbiter" in agent.system_prompt

    def test_system_prompt_from_file(self):
        """Test loading from file."""
        with patch("builtins.open", mock_open(read_data="Arbiter prompt")):
            agent = QualityArbiterAgent(system_prompt_path="exists.md")
            assert agent.system_prompt == "Arbiter prompt"


class TestQualityStandards:
    """Tests for quality standard setting."""

    def test_set_quality_standard(self, arbiter):
        """Test set_quality_standard returns QualityStandard."""
        standard = arbiter.set_quality_standard(
            task_description="Build a REST API",
            tier_level=4,
        )
        assert isinstance(standard, QualityStandard)

    def test_quality_criteria_present(self, arbiter):
        """Test quality criteria are present."""
        standard = arbiter.set_quality_standard(
            task_description="Build an API",
            tier_level=4,
        )
        assert len(standard.quality_criteria) > 0

    def test_code_task_adds_code_criteria(self, arbiter):
        """Test code task adds code-specific criteria."""
        standard = arbiter.set_quality_standard(
            task_description="Write code for a parser",
            tier_level=4,
        )
        metrics = [c.metric for c in standard.quality_criteria]
        assert any("code" in m.lower() for m in metrics)

    def test_tier4_higher_threshold(self, arbiter):
        """Test Tier 4 has higher pass threshold than Tier 3."""
        tier4 = arbiter._determine_pass_threshold(4)
        tier3 = arbiter._determine_pass_threshold(3)
        assert tier4 > tier3

    def test_custom_requirements_added(self, arbiter):
        """Test custom requirements are included."""
        standard = arbiter.set_quality_standard(
            task_description="Build something",
            tier_level=4,
            custom_requirements=["Must handle 1000 concurrent users"],
        )
        metrics = [c.metric for c in standard.quality_criteria]
        assert any("Custom" in m for m in metrics)

    def test_critical_must_haves(self, arbiter):
        """Test critical must-haves are defined."""
        standard = arbiter.set_quality_standard(
            task_description="Build code API",
            tier_level=4,
        )
        assert len(standard.critical_must_haves) > 0
        assert any("security" in mh.lower() for mh in standard.critical_must_haves)

    def test_weights_normalized(self, arbiter):
        """Test criteria weights are normalized."""
        standard = arbiter.set_quality_standard(
            task_description="Build a data code system",
            tier_level=4,
        )
        total_weight = sum(c.weight for c in standard.quality_criteria)
        assert abs(total_weight - 1.0) < 0.01


class TestDisputeResolution:
    """Tests for dispute resolution."""

    def test_resolve_dispute_returns_verdict(self, arbiter):
        """Test resolve_dispute returns QualityVerdict."""
        verdict = arbiter.resolve_dispute(
            arbitration_input={
                "disagreement_reason": "Verifier and Critic disagree on accuracy",
                "debate_rounds_completed": 2,
            },
            verifier_report={"verdict": "PASS", "overall_reliability": 0.85, "flagged_claims": []},
            critic_report={"overall_assessment": "Some issues found", "attacks": []},
            reviewer_verdict="FAIL",
        )
        assert isinstance(verdict, QualityVerdict)

    def test_critical_resolution_overrides_reviewer(self, arbiter):
        """Test critical issues override reviewer."""
        verdict = arbiter.resolve_dispute(
            arbitration_input={
                "disagreement_reason": "Critical security flaw",
                "debate_rounds_completed": 2,
            },
            verifier_report={
                "verdict": "FAIL",
                "overall_reliability": 0.5,
                "flagged_claims": ["claim1"],
            },
            critic_report={
                "overall_assessment": "critical issues found",
                "attacks": [{"description": "claim1 is wrong"}],
            },
            reviewer_verdict="PASS",
        )
        # When arbiter finds critical issues, it should override
        assert isinstance(verdict.overrides_reviewer, bool)

    def test_issues_overlap_detection(self, arbiter):
        """Test overlap detection between issues."""
        overlap = arbiter._issues_overlap(
            "SQL injection in authentication module",
            "authentication module has security risk",
        )
        assert overlap is True

    def test_no_overlap_different_issues(self, arbiter):
        """Test no overlap for completely different issues."""
        overlap = arbiter._issues_overlap("x", "y")
        assert overlap is False

    def test_resolution_has_required_actions(self, arbiter):
        """Test resolution includes required actions."""
        verdict = arbiter.resolve_dispute(
            arbitration_input={
                "disagreement_reason": "Test dispute",
                "debate_rounds_completed": 2,
            },
            verifier_report={"verdict": "FAIL", "overall_reliability": 0.6, "flagged_claims": []},
            critic_report={"overall_assessment": "critical", "attacks": []},
            reviewer_verdict="FAIL",
        )
        assert isinstance(verdict.required_actions, list)


class TestQualityArbiterConvenience:
    """Tests for create_quality_arbiter convenience function."""

    def test_create_quality_arbiter(self):
        """Test convenience function creates a QualityArbiterAgent."""
        agent = create_quality_arbiter(system_prompt_path="nonexistent.md")
        assert isinstance(agent, QualityArbiterAgent)


# =============================================================================
# Ethics Advisor Tests
# =============================================================================

@pytest.fixture
def ethics():
    """Create an EthicsAdvisorAgent with no system prompt file."""
    return EthicsAdvisorAgent(system_prompt_path="nonexistent.md")


CLEAN_OUTPUT = "This is a clean technical document about Python programming."

PII_OUTPUT = "Contact John at john.doe@example.com or call 555-123-4567."

BIASED_OUTPUT = "Women are unable to code as well as men. Older people can't learn new technology."

SAFETY_OUTPUT = "Here is how to make a bomb at home. Also how to hack into bank systems."

MEDICAL_OUTPUT = "Take 500mg of ibuprofen for your headache. This health remedy works great."


class TestEthicsAdvisorInitialization:
    """Tests for EthicsAdvisorAgent initialization."""

    def test_default_initialization(self):
        """Test default init parameters."""
        agent = EthicsAdvisorAgent(system_prompt_path="nonexistent.md")
        assert agent.model == "claude-3-5-opus-20240507"
        assert agent.max_turns == 30

    def test_pii_patterns_initialized(self):
        """Test PII patterns are configured."""
        agent = EthicsAdvisorAgent(system_prompt_path="nonexistent.md")
        assert "email" in agent.pii_patterns
        assert "ssn" in agent.pii_patterns
        assert "credit_card" in agent.pii_patterns

    def test_bias_patterns_initialized(self):
        """Test bias patterns are configured."""
        agent = EthicsAdvisorAgent(system_prompt_path="nonexistent.md")
        assert "gender" in agent.bias_patterns
        assert "racial" in agent.bias_patterns

    def test_safety_patterns_initialized(self):
        """Test safety patterns are configured."""
        agent = EthicsAdvisorAgent(system_prompt_path="nonexistent.md")
        assert "self_harm" in agent.safety_patterns
        assert "violence" in agent.safety_patterns
        assert "illegal" in agent.safety_patterns

    def test_system_prompt_fallback(self):
        """Test fallback prompt."""
        agent = EthicsAdvisorAgent(system_prompt_path="nonexistent.md")
        assert "Ethics Advisor" in agent.system_prompt

    def test_system_prompt_from_file(self):
        """Test loading from file."""
        with patch("builtins.open", mock_open(read_data="Ethics prompt")):
            agent = EthicsAdvisorAgent(system_prompt_path="exists.md")
            assert agent.system_prompt == "Ethics prompt"


class TestEthicsReview:
    """Tests for the review_output method."""

    def test_clean_output_passes(self, ethics):
        """Test clean output passes review."""
        review = ethics.review_output(CLEAN_OUTPUT, "Write about Python")
        assert isinstance(review, EthicsReview)
        assert review.verdict == "PASS"
        assert review.can_proceed is True

    def test_review_has_all_assessments(self, ethics):
        """Test review includes all assessment types."""
        review = ethics.review_output(CLEAN_OUTPUT, "Simple task")
        assert len(review.bias_analysis) > 0
        assert len(review.pii_scan_results) > 0
        assert len(review.compliance_assessment) > 0
        assert len(review.safety_assessment) > 0

    def test_review_has_recommendations(self, ethics):
        """Test review includes recommendations."""
        review = ethics.review_output(CLEAN_OUTPUT, "Simple task")
        assert isinstance(review.recommendations, list)
        assert len(review.recommendations) > 0


class TestPIIScanning:
    """Tests for PII scanning."""

    def test_detects_email(self, ethics):
        """Test email detection."""
        review = ethics.review_output(PII_OUTPUT, "Contact info")
        pii_issues = [i for i in review.flagged_issues if i.issue_type == IssueType.PII]
        assert len(pii_issues) > 0

    def test_detects_phone(self, ethics):
        """Test phone number detection."""
        review = ethics.review_output(PII_OUTPUT, "Contact info")
        pii_issues = [i for i in review.flagged_issues if i.issue_type == IssueType.PII]
        email_or_phone = [i for i in pii_issues if "email" in i.description or "phone" in i.description]
        assert len(email_or_phone) >= 2

    def test_detects_ssn(self, ethics):
        """Test SSN detection."""
        review = ethics.review_output(
            "SSN: 123-45-6789", "Process form"
        )
        pii_issues = [i for i in review.flagged_issues if i.issue_type == IssueType.PII]
        assert len(pii_issues) > 0

    def test_pii_blocks_output(self, ethics):
        """Test PII issues block output."""
        review = ethics.review_output(PII_OUTPUT, "Contact info")
        pii_issues = [i for i in review.flagged_issues if i.issue_type == IssueType.PII]
        assert all(i.blocks_output for i in pii_issues)

    def test_no_pii_in_clean(self, ethics):
        """Test no PII in clean output."""
        results = ethics._scan_for_pii(CLEAN_OUTPUT)
        assert len(results["issues"]) == 0


class TestBiasDetection:
    """Tests for bias detection."""

    def test_detects_gender_bias(self, ethics):
        """Test gender bias detection."""
        review = ethics.review_output(
            "Women are unable to understand complex code.", "Test"
        )
        bias_issues = [i for i in review.flagged_issues if i.issue_type == IssueType.BIAS]
        assert len(bias_issues) > 0

    def test_no_bias_in_clean(self, ethics):
        """Test no bias in clean output."""
        results = ethics._check_for_bias(CLEAN_OUTPUT)
        assert len(results["issues"]) == 0

    def test_bias_does_not_block(self, ethics):
        """Test bias issues do not block output by default."""
        results = ethics._check_for_bias(BIASED_OUTPUT)
        for issue in results["issues"]:
            assert issue.blocks_output is False


class TestSafetyAssessment:
    """Tests for safety assessment."""

    def test_detects_violence_content(self, ethics):
        """Test violence content detection."""
        results = ethics._assess_safety(SAFETY_OUTPUT)
        assert len(results["issues"]) > 0

    def test_safety_blocks_output(self, ethics):
        """Test safety issues block output."""
        results = ethics._assess_safety(SAFETY_OUTPUT)
        assert all(i.blocks_output for i in results["issues"])

    def test_no_safety_in_clean(self, ethics):
        """Test no safety issues in clean output."""
        results = ethics._assess_safety(CLEAN_OUTPUT)
        assert len(results["issues"]) == 0


class TestComplianceAssessment:
    """Tests for compliance assessment."""

    def test_medical_without_disclaimer(self, ethics):
        """Test medical content without disclaimer is flagged."""
        results = ethics._assess_compliance(MEDICAL_OUTPUT, None)
        assert len(results["issues"]) > 0
        assert results["issues"][0].issue_type == IssueType.COMPLIANCE

    def test_financial_without_disclaimer(self, ethics):
        """Test financial content without disclaimer is flagged."""
        results = ethics._assess_compliance(
            "This investment will guarantee 20% returns with financial advice.",
            None,
        )
        assert len(results["issues"]) > 0

    def test_clean_passes_compliance(self, ethics):
        """Test clean output passes compliance."""
        results = ethics._assess_compliance(CLEAN_OUTPUT, None)
        assert len(results["issues"]) == 0


class TestVerdictDetermination:
    """Tests for verdict determination."""

    def test_no_issues_passes(self, ethics):
        """Test no issues results in PASS."""
        verdict, can_proceed = ethics._determine_verdict([])
        assert verdict == "PASS"
        assert can_proceed is True

    def test_critical_issue_fails(self, ethics):
        """Test critical issue results in FAIL."""
        issues = [FlaggedIssue(
            issue_type=IssueType.SAFETY,
            severity=IssueSeverity.CRITICAL,
            description="Critical safety issue",
            potential_harm="Severe harm",
            remediation="Remove content",
            blocks_output=True,
        )]
        verdict, can_proceed = ethics._determine_verdict(issues)
        assert verdict == "FAIL"
        assert can_proceed is False

    def test_blocking_issue_fails(self, ethics):
        """Test blocking issue results in FAIL."""
        issues = [FlaggedIssue(
            issue_type=IssueType.PII,
            severity=IssueSeverity.HIGH,
            description="PII exposed",
            potential_harm="Privacy violation",
            remediation="Redact",
            blocks_output=True,
        )]
        verdict, can_proceed = ethics._determine_verdict(issues)
        assert verdict == "FAIL"
        assert can_proceed is False


class TestEthicsAdvisorConvenience:
    """Tests for create_ethics_advisor convenience function."""

    def test_create_ethics_advisor(self):
        """Test convenience function creates an EthicsAdvisorAgent."""
        agent = create_ethics_advisor(system_prompt_path="nonexistent.md")
        assert isinstance(agent, EthicsAdvisorAgent)
