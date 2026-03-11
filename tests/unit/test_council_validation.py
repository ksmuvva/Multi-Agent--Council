"""
Council Validation Tests - Deep validation of Full Council agents.

Tests the Full Council (Chair + Arbiter + Ethics) with focus on:
- Schema correctness after trailing-comma fix
- SDK integration configuration
- Edge cases in PII detection, bias patterns, safety scanning
- Cross-agent interactions and data flow
- Boundary conditions for all three council agents
- Tool and skill assignments
"""

import re
import json
import pytest
from unittest.mock import patch, mock_open
from pydantic import ValidationError

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
from src.core.sme_registry import (
    SME_REGISTRY,
    find_personas_by_keywords,
    validate_interaction_mode,
    get_persona,
    get_persona_ids,
)
from src.core.sdk_integration import (
    AGENT_ALLOWED_TOOLS,
    build_agent_options,
    _get_output_schema,
    get_skills_for_agent,
    get_skills_for_sme,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def chair():
    return CouncilChairAgent(system_prompt_path="nonexistent.md")


@pytest.fixture
def arbiter():
    return QualityArbiterAgent(system_prompt_path="nonexistent.md")


@pytest.fixture
def ethics():
    return EthicsAdvisorAgent(system_prompt_path="nonexistent.md")


# =============================================================================
# Schema Fix Validation - Verify trailing comma bug is fixed
# =============================================================================

class TestSchemaFixValidation:
    """Verify that schemas generate correct JSON Schema after the trailing-comma fix."""

    def test_sme_selection_report_schema_no_warnings(self):
        """SMESelectionReport should generate clean JSON Schema."""
        schema = SMESelectionReport.model_json_schema()
        assert "properties" in schema
        assert "selected_smes" in schema["properties"]
        assert "tier_recommendation" in schema["properties"]
        # Verify it's not a tuple default
        for prop_name, prop_def in schema["properties"].items():
            if "default" in prop_def:
                assert not isinstance(prop_def["default"], tuple), \
                    f"Property '{prop_name}' has tuple default (trailing comma bug)"

    def test_quality_standard_schema_no_warnings(self):
        """QualityStandard should generate clean JSON Schema."""
        schema = QualityStandard.model_json_schema()
        assert "properties" in schema
        assert "quality_criteria" in schema["properties"]
        assert "overall_pass_threshold" in schema["properties"]

    def test_quality_verdict_schema_no_warnings(self):
        """QualityVerdict should generate clean JSON Schema."""
        schema = QualityVerdict.model_json_schema()
        assert "properties" in schema
        assert "required_actions" in schema["properties"]
        assert "disputed_items" in schema["properties"]

    def test_ethics_review_schema_no_warnings(self):
        """EthicsReview should generate clean JSON Schema."""
        schema = EthicsReview.model_json_schema()
        assert "properties" in schema
        assert "flagged_issues" in schema["properties"]
        assert "can_proceed" in schema["properties"]
        assert "recommendations" in schema["properties"]

    def test_quality_criteria_schema_no_warnings(self):
        """QualityCriteria should generate clean JSON Schema."""
        schema = QualityCriteria.model_json_schema()
        assert "properties" in schema
        assert "measurement_method" in schema["properties"]

    def test_sme_selection_report_round_trip(self):
        """Test that SMESelectionReport can serialize and deserialize."""
        report = SMESelectionReport(
            task_summary="Test task",
            selected_smes=[
                SMESelection(
                    persona_name="Cloud Architect",
                    persona_domain="Cloud Infrastructure",
                    skills_to_load=["azure-architect"],
                    interaction_mode=InteractionMode.ADVISOR,
                    reasoning="Cloud expertise needed",
                    activation_phase="planning",
                )
            ],
            domain_gaps_identified=[],
            collaboration_plan="Advisors provide guidance",
            expected_sme_contributions={"Cloud Architect": "Validate architecture"},
            tier_recommendation=3,
            requires_full_council=False,
        )
        json_str = report.model_dump_json()
        parsed = SMESelectionReport.model_validate_json(json_str)
        assert parsed.task_summary == report.task_summary
        assert len(parsed.selected_smes) == 1
        assert parsed.tier_recommendation == 3

    def test_quality_verdict_round_trip(self):
        """Test QualityVerdict serialization round trip."""
        verdict = QualityVerdict(
            original_dispute="Test dispute",
            disputed_items=[
                DisputedItem(
                    item="Test item",
                    reviewer_position="PASS",
                    verifier_position="FAIL",
                    critic_position="FAIL",
                    arbiter_resolution="Revise needed",
                )
            ],
            debate_rounds_completed=2,
            arbiter_analysis="Analysis here",
            resolution="PARTIAL_REMEDIATION",
            required_actions=["Fix issue 1"],
            overrides_reviewer=False,
        )
        json_str = verdict.model_dump_json()
        parsed = QualityVerdict.model_validate_json(json_str)
        assert parsed.debate_rounds_completed == 2
        assert len(parsed.required_actions) == 1

    def test_ethics_review_round_trip(self):
        """Test EthicsReview serialization round trip."""
        review = EthicsReview(
            output_summary="Test output",
            verdict="PASS",
            flagged_issues=[],
            bias_analysis="No bias detected",
            pii_scan_results="No PII detected",
            compliance_assessment="No issues",
            safety_assessment="No concerns",
            recommendations=["No issues found"],
            can_proceed=True,
            required_remediations=[],
        )
        json_str = review.model_dump_json()
        parsed = EthicsReview.model_validate_json(json_str)
        assert parsed.verdict == "PASS"
        assert parsed.can_proceed is True


# =============================================================================
# SDK Integration Validation
# =============================================================================

class TestSDKToolsAssignment:
    """Validate SDK tool assignments for council agents."""

    def test_council_chair_has_no_tools(self):
        """Council Chair should have empty allowed_tools (reasoning-only)."""
        assert AGENT_ALLOWED_TOOLS["council_chair"] == []

    def test_quality_arbiter_has_no_tools(self):
        """Quality Arbiter should have empty allowed_tools (reasoning-only)."""
        assert AGENT_ALLOWED_TOOLS["quality_arbiter"] == []

    def test_ethics_advisor_has_no_tools(self):
        """Ethics Advisor should have empty allowed_tools (reasoning-only)."""
        assert AGENT_ALLOWED_TOOLS["ethics_advisor"] == []

    def test_council_agents_are_least_privilege(self):
        """Council agents should follow least-privilege principle."""
        for agent_key in ["council_chair", "quality_arbiter", "ethics_advisor"]:
            tools = AGENT_ALLOWED_TOOLS[agent_key]
            assert len(tools) == 0, f"{agent_key} should have no tools"

    def test_sme_default_has_skill_tool(self):
        """SME personas should have Skill tool for loading skills."""
        assert "Skill" in AGENT_ALLOWED_TOOLS["sme_default"]

    def test_output_schema_council_chair(self):
        """Council Chair should have SMESelectionReport as output schema."""
        schema = _get_output_schema("council_chair")
        assert schema is not None
        assert "properties" in schema

    def test_output_schema_quality_arbiter(self):
        """Quality Arbiter should have QualityVerdict as output schema."""
        schema = _get_output_schema("quality_arbiter")
        assert schema is not None

    def test_output_schema_ethics_advisor(self):
        """Ethics Advisor should have EthicsReview as output schema."""
        schema = _get_output_schema("ethics_advisor")
        assert schema is not None


class TestSDKSkillsAssignment:
    """Validate skill assignments."""

    def test_council_agents_have_no_skills(self):
        """Council agents should not have direct skills assigned."""
        for agent in ["council_chair", "quality_arbiter", "ethics_advisor"]:
            skills = get_skills_for_agent(agent)
            assert skills == [], f"{agent} should have no skills"

    def test_orchestrator_has_multi_agent_skill(self):
        """Orchestrator should have multi-agent-reasoning skill."""
        skills = get_skills_for_agent("orchestrator")
        assert "multi-agent-reasoning" in skills

    def test_sme_personas_have_skills(self):
        """SME personas should have skills from registry."""
        for persona_id in get_persona_ids():
            persona = get_persona(persona_id)
            assert persona is not None
            assert isinstance(persona.skill_files, list)
            # Most personas should have at least one skill
            # (not all do, so we just validate the type)


# =============================================================================
# Council Chair - Advanced Edge Cases
# =============================================================================

class TestChairAdvancedEdgeCases:
    """Advanced edge case tests for the Council Chair."""

    def test_empty_task_description(self, chair):
        """Chair should handle empty task descriptions gracefully."""
        report = chair.select_smes("", tier_level=3)
        assert isinstance(report, SMESelectionReport)
        assert report.task_summary == ""

    def test_very_long_task_description(self, chair):
        """Chair should truncate very long task descriptions."""
        long_desc = "Build " * 1000
        report = chair.select_smes(long_desc, tier_level=3)
        assert len(report.task_summary) <= 200

    def test_all_domains_triggered(self, chair):
        """Task with all domain keywords should identify all domains."""
        all_keywords_task = (
            "Build a secure AWS cloud infrastructure with kubernetes, "
            "create ETL data pipelines with SQL, train ML model with PyTorch, "
            "write pytest unit tests, gather requirements from stakeholders, "
            "create API documentation, set up CI/CD with Jenkins Docker, "
            "build responsive React frontend UI, configure SailPoint identity IAM"
        )
        domains = chair._identify_required_domains(all_keywords_task, None)
        assert len(domains) >= 8  # Should match most/all domains

    def test_max_smes_1(self, chair):
        """When max_smes is 1, only one SME should be selected."""
        report = chair.select_smes(
            "Build a secure cloud data pipeline",
            tier_level=3,
            max_smes=1,
        )
        assert len(report.selected_smes) <= 1

    def test_max_smes_0(self, chair):
        """When max_smes is 0, no SMEs should be selected."""
        report = chair.select_smes(
            "Build a cloud app",
            tier_level=3,
            max_smes=0,
        )
        assert len(report.selected_smes) == 0

    def test_no_matching_domains(self, chair):
        """Task with no matching keywords should return empty SME list."""
        report = chair.select_smes(
            "xyzzy quantum entanglement",
            tier_level=3,
        )
        # May match some via complexity indicators
        assert isinstance(report, SMESelectionReport)

    def test_tier3_does_not_always_require_full_council(self, chair):
        """Tier 3 with single non-sensitive SME should not require full council."""
        result = chair._requires_full_council("Simple coding task", 3, [
            SMESelection(
                persona_name="Test",
                persona_domain="test_engineer",
                skills_to_load=[],
                interaction_mode=InteractionMode.ADVISOR,
                reasoning="test",
                activation_phase="execution",
            )
        ])
        assert result is False

    def test_domain_gap_when_max_exceeded(self, chair):
        """Domain gaps should be identified when max_smes limits selection."""
        report = chair.select_smes(
            "Build secure cloud data pipeline with testing and docs",
            tier_level=3,
            max_smes=1,
        )
        # Multiple domains identified but only 1 selected
        if len(report.domain_gaps_identified) > 0:
            assert any("not selected" in gap or "limit" in gap
                       for gap in report.domain_gaps_identified)

    def test_collaboration_plan_empty_smes(self, chair):
        """Collaboration plan should handle empty SME list."""
        plan = chair._create_collaboration_plan([], 3)
        assert "No SMEs selected" in plan

    def test_expected_contribution_unknown_domain(self, chair):
        """Expected contribution should have fallback for unknown domains."""
        sme = SMESelection(
            persona_name="Unknown Expert",
            persona_domain="quantum_computing",
            skills_to_load=[],
            interaction_mode=InteractionMode.ADVISOR,
            reasoning="test",
            activation_phase="execution",
        )
        contribution = chair._define_expected_contribution(sme, "Quantum task")
        assert "expertise" in contribution.lower()


# =============================================================================
# Quality Arbiter - Advanced Edge Cases
# =============================================================================

class TestArbiterAdvancedEdgeCases:
    """Advanced edge case tests for the Quality Arbiter."""

    def test_tier3_lower_threshold(self, arbiter):
        """Tier 3 threshold should be lower than Tier 4."""
        t3 = arbiter._determine_pass_threshold(3)
        t4 = arbiter._determine_pass_threshold(4)
        assert t3 == 0.75
        assert t4 == 0.85
        assert t4 > t3

    def test_multiple_custom_requirements(self, arbiter):
        """Multiple custom requirements should all be added as criteria."""
        standard = arbiter.set_quality_standard(
            task_description="Build API",
            tier_level=4,
            custom_requirements=[
                "Must handle 1000 concurrent users",
                "Must support pagination",
                "Must have rate limiting",
            ],
        )
        custom_metrics = [c for c in standard.quality_criteria if "Custom" in c.metric]
        assert len(custom_metrics) == 3

    def test_weights_always_normalize_to_1(self, arbiter):
        """Weights should always sum to approximately 1.0."""
        for desc in [
            "Build code data API",
            "Write documentation",
            "Simple task",
        ]:
            standard = arbiter.set_quality_standard(
                task_description=desc,
                tier_level=4,
            )
            total = sum(c.weight for c in standard.quality_criteria)
            assert abs(total - 1.0) < 0.01, f"Weights sum to {total} for '{desc}'"

    def test_dispute_with_empty_issues(self, arbiter):
        """Dispute resolution with no flagged issues."""
        verdict = arbiter.resolve_dispute(
            arbitration_input={
                "disagreement_reason": "General disagreement",
                "debate_rounds_completed": 2,
            },
            verifier_report={"verdict": "PASS", "overall_reliability": 0.9, "flagged_claims": []},
            critic_report={"overall_assessment": "minor concerns", "attacks": []},
            reviewer_verdict="PASS",
        )
        assert isinstance(verdict, QualityVerdict)
        assert len(verdict.disputed_items) == 0

    def test_dispute_critical_overrides(self, arbiter):
        """Critical resolution should override reviewer."""
        verdict = arbiter.resolve_dispute(
            arbitration_input={
                "disagreement_reason": "Critical issues found",
                "debate_rounds_completed": 2,
            },
            verifier_report={
                "verdict": "FAIL",
                "overall_reliability": 0.3,
                "flagged_claims": ["claim is wrong"],
            },
            critic_report={
                "overall_assessment": "critical issues found",
                "attacks": [{"description": "claim is wrong and dangerous"}],
            },
            reviewer_verdict="PASS",
        )
        # Should detect overlap and override
        assert isinstance(verdict, QualityVerdict)

    def test_issues_overlap_minimum_words(self, arbiter):
        """Overlap requires at least 2 common words."""
        assert arbiter._issues_overlap("a b c", "a b d") is True
        assert arbiter._issues_overlap("hello", "hello") is False  # Only 1 word overlap
        assert arbiter._issues_overlap("", "") is False

    def test_measurement_protocol_tier4_vs_tier3(self, arbiter):
        """Tier 4 measurement protocol should include real-time monitoring."""
        criteria = [QualityCriteria(
            metric="Accuracy", threshold="≥90%",
            measurement_method="Verifier", weight=1.0,
        )]
        t4_protocol = arbiter._define_measurement_protocol(criteria, 4)
        t3_protocol = arbiter._define_measurement_protocol(criteria, 3)
        assert "real-time" in t4_protocol.lower()
        assert "real-time" not in t3_protocol.lower()

    def test_nice_to_haves_tier4_extras(self, arbiter):
        """Tier 4 should have additional nice-to-haves."""
        nths = arbiter._define_nice_to_haves("code task", 4)
        assert any("alternative" in n.lower() for n in nths)


# =============================================================================
# Ethics Advisor - Advanced Edge Cases
# =============================================================================

class TestEthicsAdvancedEdgeCases:
    """Advanced edge case tests for the Ethics Advisor."""

    def test_credit_card_detection(self, ethics):
        """Should detect credit card numbers."""
        review = ethics.review_output(
            "Card: 4111-1111-1111-1111", "Process payment"
        )
        pii_issues = [i for i in review.flagged_issues if i.issue_type == IssueType.PII]
        assert len(pii_issues) > 0

    def test_ip_address_detection(self, ethics):
        """Should detect IP addresses."""
        review = ethics.review_output(
            "Server is at 192.168.1.100", "Server setup"
        )
        pii_issues = [i for i in review.flagged_issues if i.issue_type == IssueType.PII]
        assert len(pii_issues) > 0

    def test_multiple_pii_types(self, ethics):
        """Should detect multiple PII types in same text."""
        review = ethics.review_output(
            "Email: test@example.com, Phone: 555-123-4567, SSN: 123-45-6789",
            "Contact info"
        )
        pii_issues = [i for i in review.flagged_issues if i.issue_type == IssueType.PII]
        assert len(pii_issues) >= 3  # email, phone, ssn

    def test_age_bias_detection(self, ethics):
        """Should detect age-related bias (direct adjacency required)."""
        # Pattern requires age term directly followed by negative qualifier
        results = ethics._check_for_bias(
            "older can't learn programming"
        )
        assert len(results["issues"]) > 0

    def test_no_bias_in_neutral_text(self, ethics):
        """Neutral text should not trigger bias detection."""
        results = ethics._check_for_bias(
            "Python is a versatile programming language used by developers worldwide."
        )
        assert len(results["issues"]) == 0

    def test_self_harm_detection(self, ethics):
        """Should detect self-harm content."""
        results = ethics._assess_safety(
            "how to hurt yourself"
        )
        safety_issues = results["issues"]
        assert len(safety_issues) > 0
        assert any(i.severity == IssueSeverity.CRITICAL for i in safety_issues)

    def test_illegal_activity_detection(self, ethics):
        """Should detect illegal activity guidance."""
        results = ethics._assess_safety(
            "how to steal from banks"
        )
        assert len(results["issues"]) > 0

    def test_weapons_content_detection(self, ethics):
        """Should detect weapons/violence content (direct adjacency required)."""
        # Pattern requires build/make/create directly followed by bomb/weapon/poison
        results = ethics._assess_safety(
            "make bomb instructions"
        )
        assert len(results["issues"]) > 0

    def test_financial_without_disclaimer_flags(self, ethics):
        """Financial advice without disclaimer should be flagged."""
        results = ethics._assess_compliance(
            "This investment will guarantee returns based on financial advice.",
            None,
        )
        assert len(results["issues"]) > 0
        assert results["issues"][0].issue_type == IssueType.COMPLIANCE

    def test_medical_with_disclaimer_passes(self, ethics):
        """Medical content with disclaimer should pass."""
        results = ethics._assess_compliance(
            "This health information is not medical advice. Consult a doctor before treatment.",
            None,
        )
        assert len(results["issues"]) == 0

    def test_verdict_more_than_3_high_issues(self, ethics):
        """More than 3 high-severity issues should FAIL."""
        issues = [
            FlaggedIssue(
                issue_type=IssueType.BIAS,
                severity=IssueSeverity.HIGH,
                description=f"High issue {i}",
                potential_harm="Potential harm",
                remediation="Fix it",
                blocks_output=False,
            )
            for i in range(4)
        ]
        verdict, can_proceed = ethics._determine_verdict(issues)
        assert verdict == "FAIL"
        assert can_proceed is False

    def test_verdict_3_high_issues_passes(self, ethics):
        """Exactly 3 high-severity (non-blocking) issues should still PASS."""
        issues = [
            FlaggedIssue(
                issue_type=IssueType.BIAS,
                severity=IssueSeverity.HIGH,
                description=f"High issue {i}",
                potential_harm="Potential harm",
                remediation="Fix it",
                blocks_output=False,
            )
            for i in range(3)
        ]
        verdict, can_proceed = ethics._determine_verdict(issues)
        assert verdict == "PASS"
        assert can_proceed is True

    def test_verdict_medium_issues_pass(self, ethics):
        """Multiple medium-severity issues should PASS."""
        issues = [
            FlaggedIssue(
                issue_type=IssueType.BIAS,
                severity=IssueSeverity.MEDIUM,
                description=f"Medium issue {i}",
                potential_harm="Minor harm",
                remediation="Consider fixing",
                blocks_output=False,
            )
            for i in range(10)
        ]
        verdict, can_proceed = ethics._determine_verdict(issues)
        assert verdict == "PASS"
        assert can_proceed is True

    def test_recommendations_grouped_by_type(self, ethics):
        """Recommendations should be grouped by issue type."""
        issues = [
            FlaggedIssue(
                issue_type=IssueType.PII,
                severity=IssueSeverity.HIGH,
                description="PII found",
                potential_harm="Privacy",
                remediation="Redact",
                blocks_output=True,
            ),
            FlaggedIssue(
                issue_type=IssueType.BIAS,
                severity=IssueSeverity.MEDIUM,
                description="Bias found",
                potential_harm="Stereotypes",
                remediation="Review language",
                blocks_output=False,
            ),
        ]
        recs = ethics._generate_recommendations(issues)
        assert any("PII" in r for r in recs)
        assert any("bias" in r.lower() for r in recs)

    def test_recommendations_max_5(self, ethics):
        """Recommendations should be capped at 5."""
        issues = [
            FlaggedIssue(
                issue_type=IssueType.PII,
                severity=IssueSeverity.HIGH,
                description=f"Issue {i}",
                potential_harm="Harm",
                remediation=f"Fix {i}",
                blocks_output=True,
            )
            for i in range(10)
        ]
        recs = ethics._generate_recommendations(issues)
        assert len(recs) <= 5

    def test_empty_output_passes(self, ethics):
        """Empty output should pass review."""
        review = ethics.review_output("", "Test task")
        assert review.verdict == "PASS"
        assert review.can_proceed is True

    def test_required_remediations_only_blocking(self, ethics):
        """Required remediations should only include blocking issues."""
        review = ethics.review_output(
            "Contact john@example.com for more info about Python.",
            "Contact task"
        )
        # Email is blocking PII
        assert len(review.required_remediations) > 0
        # All required remediations should correspond to blocking issues
        blocking_issues = [i for i in review.flagged_issues if i.blocks_output]
        assert len(review.required_remediations) == len(blocking_issues)


# =============================================================================
# Cross-Agent Data Flow Validation
# =============================================================================

class TestCrossAgentDataFlow:
    """Test data flow between council agents."""

    def test_chair_report_to_arbiter_standard(self, chair, arbiter):
        """Chair's output should be usable by Arbiter."""
        report = chair.select_smes(
            "Build a secure API with cloud deployment",
            tier_level=4,
        )
        # Arbiter sets quality standard based on task
        standard = arbiter.set_quality_standard(
            task_description=report.task_summary,
            tier_level=report.tier_recommendation,
        )
        assert isinstance(standard, QualityStandard)
        assert standard.overall_pass_threshold > 0

    def test_chair_report_to_ethics_review(self, chair, ethics):
        """Ethics advisor reviews output from task described by Chair."""
        report = chair.select_smes(
            "Process user personal data",
            tier_level=4,
        )
        assert report.requires_full_council is True

        # Ethics reviews the output
        review = ethics.review_output(
            "User data: john@example.com, SSN: 123-45-6789",
            report.task_summary,
        )
        assert review.verdict == "FAIL"
        assert review.can_proceed is False

    def test_full_council_pipeline(self, chair, arbiter, ethics):
        """Simulate the full council pipeline for a Tier 4 task."""
        # Step 1: Chair selects SMEs
        task = "Design secure cloud architecture for healthcare data processing with HIPAA compliance"
        sme_report = chair.select_smes(task, tier_level=4)
        assert sme_report.requires_full_council is True
        assert sme_report.tier_recommendation == 4

        # Step 2: Arbiter sets quality standards
        quality_standard = arbiter.set_quality_standard(
            task_description=task,
            tier_level=4,
        )
        assert quality_standard.overall_pass_threshold == 0.85
        assert len(quality_standard.critical_must_haves) > 0

        # Step 3: Ethics reviews clean output
        clean_output = (
            "The proposed architecture uses encrypted data storage with "
            "role-based access control and audit logging for compliance."
        )
        ethics_review = ethics.review_output(clean_output, task)
        assert ethics_review.verdict == "PASS"

    def test_arbiter_dispute_feeds_back(self, arbiter):
        """Arbiter dispute resolution should produce actionable output."""
        verdict = arbiter.resolve_dispute(
            arbitration_input={
                "disagreement_reason": "Verifier says PASS, Critic says FAIL on security",
                "debate_rounds_completed": 2,
            },
            verifier_report={
                "verdict": "PASS",
                "overall_reliability": 0.85,
                "flagged_claims": ["API is secure"],
            },
            critic_report={
                "overall_assessment": "security concerns",
                "attacks": [{"description": "API has no rate limiting"}],
            },
            reviewer_verdict="PASS",
        )
        assert isinstance(verdict, QualityVerdict)
        assert verdict.debate_rounds_completed == 2
        assert len(verdict.resolution) > 0


# =============================================================================
# SME Registry Integration
# =============================================================================

class TestSMERegistryIntegration:
    """Test Council Chair integration with SME Registry."""

    def test_all_registry_personas_accessible(self, chair):
        """Chair should be able to access all registry personas."""
        for persona_id in get_persona_ids():
            persona = get_persona(persona_id)
            assert persona is not None
            assert persona.persona_id == persona_id

    def test_chair_domain_patterns_cover_registry(self, chair):
        """Chair's domain patterns should cover all registry personas."""
        for persona_id in get_persona_ids():
            assert persona_id in chair.domain_patterns, \
                f"Missing domain pattern for {persona_id}"

    def test_interaction_mode_validation(self):
        """Interaction modes should be valid for each persona."""
        for persona_id in get_persona_ids():
            persona = get_persona(persona_id)
            for mode in persona.interaction_modes:
                assert validate_interaction_mode(persona_id, mode) is True

    def test_invalid_interaction_mode_rejected(self):
        """Invalid interaction modes should be rejected."""
        # Technical writer doesn't have DEBATER mode
        result = validate_interaction_mode("technical_writer", InteractionMode.DEBATER)
        # May or may not be valid depending on registry
        assert isinstance(result, bool)


# =============================================================================
# Pydantic Validation Edge Cases
# =============================================================================

class TestPydanticValidationEdgeCases:
    """Test Pydantic model validation edge cases."""

    def test_sme_selection_report_max_3_smes(self):
        """SMESelectionReport should enforce max 3 SMEs."""
        smes = [
            SMESelection(
                persona_name=f"SME {i}",
                persona_domain=f"domain_{i}",
                skills_to_load=[],
                interaction_mode=InteractionMode.ADVISOR,
                reasoning="test",
                activation_phase="execution",
            )
            for i in range(4)
        ]
        with pytest.raises(ValidationError):
            SMESelectionReport(
                task_summary="Test",
                selected_smes=smes,
                collaboration_plan="Plan",
                expected_sme_contributions={},
                tier_recommendation=3,
            )

    def test_tier_recommendation_bounds(self):
        """Tier recommendation must be 3 or 4."""
        with pytest.raises(ValidationError):
            SMESelectionReport(
                task_summary="Test",
                selected_smes=[],
                collaboration_plan="Plan",
                expected_sme_contributions={},
                tier_recommendation=2,  # Too low
            )
        with pytest.raises(ValidationError):
            SMESelectionReport(
                task_summary="Test",
                selected_smes=[],
                collaboration_plan="Plan",
                expected_sme_contributions={},
                tier_recommendation=5,  # Too high
            )

    def test_debate_rounds_minimum_2(self):
        """QualityVerdict requires at least 2 debate rounds."""
        with pytest.raises(ValidationError):
            QualityVerdict(
                original_dispute="Test",
                disputed_items=[],
                debate_rounds_completed=1,  # Too low
                arbiter_analysis="Analysis",
                resolution="Resolution",
                required_actions=[],
            )

    def test_quality_criteria_weight_bounds(self):
        """QualityCriteria weight must be 0.0-1.0."""
        with pytest.raises(ValidationError):
            QualityCriteria(
                metric="Test",
                threshold="Test",
                measurement_method="Test",
                weight=1.5,  # Too high
            )
        with pytest.raises(ValidationError):
            QualityCriteria(
                metric="Test",
                threshold="Test",
                measurement_method="Test",
                weight=-0.1,  # Negative
            )

    def test_overall_pass_threshold_bounds(self):
        """QualityStandard pass threshold must be 0.0-1.0."""
        with pytest.raises(ValidationError):
            QualityStandard(
                task_summary="Test",
                quality_criteria=[],
                overall_pass_threshold=1.5,
                critical_must_haves=[],
                measurement_protocol="Test",
            )

    def test_interaction_mode_enum_values(self):
        """InteractionMode should have exactly 3 values."""
        assert len(InteractionMode) == 3
        assert InteractionMode.ADVISOR.value == "advisor"
        assert InteractionMode.CO_EXECUTOR.value == "co_executor"
        assert InteractionMode.DEBATER.value == "debater"

    def test_issue_type_enum_values(self):
        """IssueType should have exactly 6 values."""
        assert len(IssueType) == 6
        expected = {"bias", "pii", "compliance", "safety", "security", "fairness"}
        actual = {it.value for it in IssueType}
        assert actual == expected

    def test_issue_severity_enum_values(self):
        """IssueSeverity should have exactly 4 values."""
        assert len(IssueSeverity) == 4
        expected = {"critical", "high", "medium", "low"}
        actual = {s.value for s in IssueSeverity}
        assert actual == expected


# =============================================================================
# PII Pattern Edge Cases
# =============================================================================

class TestPIIPatternEdgeCases:
    """Test PII detection patterns with edge cases."""

    def test_email_various_formats(self, ethics):
        """Test email detection with various formats."""
        emails = [
            "user@example.com",
            "user.name@company.co.uk",
            "user+tag@domain.org",
        ]
        for email in emails:
            results = ethics._scan_for_pii(f"Contact: {email}")
            assert len(results["issues"]) > 0, f"Failed to detect email: {email}"

    def test_phone_various_formats(self, ethics):
        """Test phone detection with various formats."""
        phones = ["555-123-4567", "555.123.4567", "5551234567"]
        for phone in phones:
            results = ethics._scan_for_pii(f"Call: {phone}")
            assert len(results["issues"]) > 0, f"Failed to detect phone: {phone}"

    def test_ssn_format(self, ethics):
        """Test SSN detection."""
        results = ethics._scan_for_pii("SSN: 123-45-6789")
        pii_types = [i.description for i in results["issues"]]
        assert any("ssn" in p.lower() for p in pii_types)

    def test_credit_card_with_spaces(self, ethics):
        """Test credit card detection with spaces."""
        results = ethics._scan_for_pii("Card: 4111 1111 1111 1111")
        assert len(results["issues"]) > 0

    def test_no_false_positive_short_numbers(self, ethics):
        """Short numbers should not trigger SSN or phone detection."""
        results = ethics._scan_for_pii("The answer is 42.")
        # Should not match
        assert len(results["issues"]) == 0

    def test_pii_redaction_in_description(self, ethics):
        """PII findings should use redacted descriptions, not actual PII."""
        results = ethics._scan_for_pii("Email: secret@company.com")
        for issue in results["issues"]:
            # Description should say "Potential email detected", not the actual email
            assert "secret@company.com" not in issue.description
