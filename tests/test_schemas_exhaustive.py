"""
Exhaustive Tests for All Pydantic Schemas

Tests validation, serialization, edge cases, and defaults
for all schema models in the system.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from pydantic import ValidationError

# Analyst schemas
from src.schemas.analyst import (
    ModalityType, SeverityLevel as AnalystSeverity, MissingInfo,
    SubTask, TaskIntelligenceReport,
)

# Planner schemas
from src.schemas.planner import (
    StepStatus, AgentAssignment as PlannerAgentAssignment,
    ExecutionStep, ParallelGroup, ExecutionPlan,
)

# Clarifier schemas
from src.schemas.clarifier import (
    QuestionPriority, ImpactAssessment, ClarificationQuestion,
    ClarificationRequest,
)

# Researcher schemas
from src.schemas.researcher import (
    ConfidenceLevel, SourceReliability, Source, Finding,
    Conflict, KnowledgeGap, EvidenceBrief,
)

# Verifier schemas
from src.schemas.verifier import (
    VerificationStatus, FabricationRisk, Claim, ClaimBatch,
    VerificationReport,
)

# Critic schemas
from src.schemas.critic import (
    AttackVector, SeverityLevel as CriticSeverity, Attack,
    LogicAttack, CompletenessAttack, QualityAttack,
    ContradictionScan, RedTeamArgument, CritiqueReport,
)

# Reviewer schemas
from src.schemas.reviewer import (
    Verdict, CheckItem, Revision, QualityGateResults,
    ArbitrationInput, ReviewVerdict,
)

# Code reviewer schemas
from src.schemas.code_reviewer import (
    SeverityLevel as CodeSeverity, ReviewCategory, CodeFinding,
    SecurityScan, PerformanceAnalysis, StyleCompliance,
    CodeReviewReport,
)

# Council schemas
from src.schemas.council import (
    InteractionMode, SMESelection, SMESelectionReport,
    QualityCriteria, QualityStandard, DisputedItem,
    QualityVerdict, IssueType, IssueSeverity, FlaggedIssue,
    EthicsReview,
)

# SME schemas
from src.schemas.sme import (
    SMEInteractionMode, AdvisorReport, CoExecutorSection,
    CoExecutorReport, DebatePosition, DebaterReport,
    SMEAdvisoryReport,
)


# =============================================================================
# Analyst Schema Tests
# =============================================================================

class TestAnalystSchemas:
    def test_modality_type_values(self):
        assert ModalityType.TEXT == "text"
        assert ModalityType.CODE == "code"
        assert len(ModalityType) == 5

    def test_missing_info(self):
        mi = MissingInfo(
            requirement="Auth method", severity=AnalystSeverity.CRITICAL,
            impact="Affects security design",
        )
        assert mi.default_assumption is None

    def test_sub_task(self):
        st = SubTask(description="Implement API")
        assert st.dependencies == []
        assert st.estimated_complexity == "medium"

    def test_task_intelligence_report(self):
        report = TaskIntelligenceReport(
            literal_request="Build API",
            inferred_intent="Create REST endpoints",
            sub_tasks=[SubTask(description="Design models")],
            missing_info=[],
            assumptions=["Using Python"],
            modality=ModalityType.CODE,
            recommended_approach="Start with models",
        )
        assert report.escalation_needed is False
        assert report.suggested_tier == 2
        assert report.confidence == 0.8

    def test_suggested_tier_bounds(self):
        with pytest.raises(ValidationError):
            TaskIntelligenceReport(
                literal_request="X", inferred_intent="Y",
                sub_tasks=[], missing_info=[], assumptions=[],
                modality=ModalityType.TEXT, recommended_approach="Z",
                suggested_tier=5,
            )

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            TaskIntelligenceReport(
                literal_request="X", inferred_intent="Y",
                sub_tasks=[], missing_info=[], assumptions=[],
                modality=ModalityType.TEXT, recommended_approach="Z",
                confidence=1.5,
            )


# =============================================================================
# Planner Schema Tests
# =============================================================================

class TestPlannerSchemas:
    def test_step_status(self):
        assert StepStatus.PENDING == "pending"
        assert len(StepStatus) == 4

    def test_execution_step(self):
        step = ExecutionStep(
            step_number=1, description="Step 1",
            agent_assignments=[PlannerAgentAssignment(
                agent_name="Executor", role="coder", reason="Implementation"
            )],
        )
        assert step.status == StepStatus.PENDING
        assert step.can_parallelize is False

    def test_step_number_min(self):
        with pytest.raises(ValidationError):
            ExecutionStep(step_number=0, description="X", agent_assignments=[])

    def test_execution_plan(self):
        plan = ExecutionPlan(
            task_summary="Build API", total_steps=1,
            steps=[ExecutionStep(
                step_number=1, description="Step 1",
                agent_assignments=[PlannerAgentAssignment(
                    agent_name="E", role="R", reason="Reason"
                )],
            )],
            critical_path=[1],
        )
        assert plan.parallel_groups == []
        assert plan.risk_factors == []

    def test_parallel_group(self):
        pg = ParallelGroup(group_id="g1", steps=[2, 3], description="Parallel review")
        assert len(pg.steps) == 2


# =============================================================================
# Clarifier Schema Tests
# =============================================================================

class TestClarifierSchemas:
    def test_question_priority(self):
        assert QuestionPriority.CRITICAL == "critical"
        assert len(QuestionPriority) == 4

    def test_impact_assessment(self):
        ia = ImpactAssessment(
            quality_impact="May need revisions",
            risk_level="medium",
            potential_revisions=["Change ORM"],
        )
        assert ia.risk_level == "medium"

    def test_clarification_question(self):
        q = ClarificationQuestion(
            question="Which database?", priority=QuestionPriority.HIGH,
            reason="Affects schema", context="Data storage needed",
            default_answer="PostgreSQL",
            impact_if_unanswered=ImpactAssessment(
                quality_impact="Low", risk_level="low",
                potential_revisions=[],
            ),
        )
        assert q.answer_options is None

    def test_clarification_request(self):
        cr = ClarificationRequest(
            total_questions=0, questions=[],
            recommended_workflow="Skip",
            can_proceed_with_defaults=True,
            expected_quality_with_defaults="Good",
        )
        assert cr.can_proceed_with_defaults is True


# =============================================================================
# Researcher Schema Tests
# =============================================================================

class TestResearcherSchemas:
    def test_confidence_level(self):
        assert ConfidenceLevel.HIGH == "high"
        assert len(ConfidenceLevel) == 3

    def test_source_reliability(self):
        assert SourceReliability.UNKNOWN == "unknown"
        assert len(SourceReliability) == 4

    def test_source(self):
        s = Source(
            url="https://example.com", title="Example",
            reliability=SourceReliability.HIGH, access_date="2025-01-01",
        )
        assert s.excerpt is None

    def test_finding(self):
        f = Finding(
            claim="JWT is standard", confidence=ConfidenceLevel.HIGH,
            sources=[Source(
                url="https://auth0.com", title="Auth0",
                reliability=SourceReliability.HIGH, access_date="2025-01-01",
            )],
            context="Auth best practices",
        )
        assert f.caveats == []

    def test_evidence_brief(self):
        eb = EvidenceBrief(
            research_topic="Auth best practices",
            summary="JWT recommended",
            findings=[],
            overall_confidence=ConfidenceLevel.HIGH,
            recommended_approach="Use JWT",
        )
        assert eb.additional_research_needed is False
        assert eb.conflicts == []
        assert eb.gaps == []


# =============================================================================
# Verifier Schema Tests
# =============================================================================

class TestVerifierSchemas:
    def test_verification_status(self):
        assert VerificationStatus.VERIFIED == "verified"
        assert VerificationStatus.FABRICATED == "fabricated"
        assert len(VerificationStatus) == 4

    def test_fabrication_risk(self):
        assert FabricationRisk.LOW == "low"
        assert len(FabricationRisk) == 3

    def test_claim(self):
        c = Claim(
            claim_text="Python was released in 1991",
            confidence=10, fabrication_risk=FabricationRisk.LOW,
            verification_method="Web search",
            status=VerificationStatus.VERIFIED,
        )
        assert c.domain_verified is False
        assert c.source is None

    def test_claim_confidence_bounds(self):
        with pytest.raises(ValidationError):
            Claim(
                claim_text="X", confidence=0,
                fabrication_risk=FabricationRisk.LOW,
                verification_method="test",
                status=VerificationStatus.VERIFIED,
            )
        with pytest.raises(ValidationError):
            Claim(
                claim_text="X", confidence=11,
                fabrication_risk=FabricationRisk.LOW,
                verification_method="test",
                status=VerificationStatus.VERIFIED,
            )

    def test_claim_batch(self):
        cb = ClaimBatch(
            topic="Python history", claims=[], overall_reliability=0.9,
        )
        assert cb.overall_reliability == 0.9

    def test_verification_report(self):
        vr = VerificationReport(
            total_claims_checked=1, claims=[], verified_claims=1,
            unverified_claims=0, contradicted_claims=0, fabricated_claims=0,
            overall_reliability=0.95, verdict="PASS",
            flagged_claims=[], recommended_corrections=[],
            verification_summary="All verified",
        )
        assert vr.pass_threshold == 0.7


# =============================================================================
# Critic Schema Tests
# =============================================================================

class TestCriticSchemas:
    def test_attack_vector(self):
        assert AttackVector.LOGIC == "logic"
        assert AttackVector.RED_TEAM == "red_team"
        assert len(AttackVector) == 5

    def test_attack(self):
        a = Attack(
            vector=AttackVector.LOGIC, target="Auth flow",
            finding="Circular logic", severity=CriticSeverity.CRITICAL,
            description="Auth token validation is circular",
            scenario="Attacker exploits token refresh",
            suggestion="Simplify auth flow",
        )
        assert a.domain_specific is False
        assert a.sme_source is None

    def test_critique_report(self):
        cr = CritiqueReport(
            solution_summary="REST API",
            attacks=[],
            logic_attack=LogicAttack(invalid_arguments=[], fallacies_identified=[]),
            completeness_attack=CompletenessAttack(
                covered=["Auth"], missing=["Rate limiting"], assumptions=[],
            ),
            quality_attack=QualityAttack(weaknesses=["No tests"], improvements=["Add tests"]),
            contradiction_scan=ContradictionScan(
                external_contradictions=[], inconsistencies=[],
            ),
            red_team_argumentation=RedTeamArgument(
                adversary_perspective="Easy target",
                attack_surface=["All endpoints"],
                failure_modes=["DB exhaustion"],
                worst_case_scenarios=["Data breach"],
            ),
            overall_assessment="Needs work",
            critical_issues=["Missing auth"],
            recommended_revisions=["Add auth"],
            would_approve=False,
        )
        assert cr.would_approve is False


# =============================================================================
# Reviewer Schema Tests
# =============================================================================

class TestReviewerSchemas:
    def test_verdict(self):
        assert Verdict.PASS == "PASS"
        assert Verdict.FAIL == "FAIL"

    def test_check_item(self):
        ci = CheckItem(
            check_name="Completeness", passed=True,
            notes="All covered", severity_if_failed="high",
        )
        assert ci.passed is True

    def test_revision(self):
        r = Revision(
            category="Security", description="Add auth",
            reason="No authentication", priority="critical",
            specific_instructions="Add JWT middleware",
        )
        assert r.priority == "critical"

    def test_quality_gate_results(self):
        ci = CheckItem(check_name="Test", passed=True, notes="OK", severity_if_failed="low")
        qgr = QualityGateResults(
            completeness=ci, consistency=ci,
            verifier_signoff=ci, critic_findings_addressed=ci,
            readability=ci,
        )
        assert qgr.code_review_passed is None

    def test_review_verdict(self):
        ci = CheckItem(check_name="Test", passed=True, notes="OK", severity_if_failed="low")
        rv = ReviewVerdict(
            verdict=Verdict.PASS, confidence=0.9,
            quality_gate_results=QualityGateResults(
                completeness=ci, consistency=ci,
                verifier_signoff=ci, critic_findings_addressed=ci,
                readability=ci,
            ),
            reasons=["All checks passed"],
            can_revise=True, summary="Approved",
        )
        assert rv.arbitration_needed is False
        assert rv.revision_count == 0

    def test_arbitration_input(self):
        ai = ArbitrationInput(
            reviewer_verdict=Verdict.PASS, verifier_verdict=Verdict.FAIL,
            critic_verdict=Verdict.FAIL,
            disagreement_reason="Verifier and critic disagree",
            debate_rounds_completed=2,
        )
        assert ai.debate_rounds_completed == 2


# =============================================================================
# Code Reviewer Schema Tests
# =============================================================================

class TestCodeReviewerSchemas:
    def test_review_category(self):
        assert ReviewCategory.SECURITY == "security"
        assert ReviewCategory.PERFORMANCE == "performance"
        assert len(ReviewCategory) == 7

    def test_code_finding(self):
        cf = CodeFinding(
            severity=CodeSeverity.CRITICAL, category=ReviewCategory.SECURITY,
            file_path="api/auth.py", issue="SQL injection",
            recommendation="Use parameterized queries",
        )
        assert cf.line_number is None
        assert cf.references == []

    def test_security_scan(self):
        ss = SecurityScan(vulnerabilities_found=0)
        assert ss.sql_injection_risk is False
        assert ss.xss_risk is False

    def test_code_review_report(self):
        crr = CodeReviewReport(
            overall_assessment="Good quality",
            pass_fail=True,
            findings=[],
            security_scan=SecurityScan(vulnerabilities_found=0),
            performance_analysis=PerformanceAnalysis(),
            style_compliance=StyleCompliance(),
            error_handling_complete=True,
            test_coverage_assessment="Good coverage",
            recommended_actions=[],
        )
        assert crr.pass_fail is True


# =============================================================================
# Council Schema Tests
# =============================================================================

class TestCouncilSchemas:
    def test_interaction_mode(self):
        assert InteractionMode.ADVISOR == "advisor"
        assert len(InteractionMode) == 3

    def test_sme_selection(self):
        ss = SMESelection(
            persona_name="Cloud Architect", persona_domain="Cloud",
            skills_to_load=["azure"], interaction_mode=InteractionMode.ADVISOR,
            reasoning="Cloud expertise needed", activation_phase="Phase 2",
        )
        assert ss.persona_name == "Cloud Architect"

    def test_sme_selection_report(self):
        ssr = SMESelectionReport(
            task_summary="Architecture review",
            selected_smes=[],
            collaboration_plan="Parallel review",
            expected_sme_contributions={},
            tier_recommendation=3,
        )
        assert ssr.requires_full_council is False

    def test_tier_recommendation_bounds(self):
        with pytest.raises(ValidationError):
            SMESelectionReport(
                task_summary="X", selected_smes=[],
                collaboration_plan="X", expected_sme_contributions={},
                tier_recommendation=2,
            )

    def test_quality_criteria(self):
        qc = QualityCriteria(
            metric="Accuracy", threshold=">90%",
            measurement_method="Automated testing", weight=0.5,
        )
        assert qc.weight == 0.5

    def test_quality_standard(self):
        qs = QualityStandard(
            task_summary="Build API",
            quality_criteria=[],
            overall_pass_threshold=0.8,
            critical_must_haves=["Auth"],
            measurement_protocol="Auto + manual",
        )
        assert qs.nice_to_haves == []

    def test_quality_verdict(self):
        qv = QualityVerdict(
            original_dispute="Quality disagreement",
            disputed_items=[],
            debate_rounds_completed=2,
            arbiter_analysis="Arbiter analysis",
            resolution="Resolved in favor of reviewer",
            required_actions=["Fix issue"],
        )
        assert qv.overrides_reviewer is False

    def test_debate_rounds_min(self):
        with pytest.raises(ValidationError):
            QualityVerdict(
                original_dispute="X", disputed_items=[],
                debate_rounds_completed=1,
                arbiter_analysis="X", resolution="X",
                required_actions=[],
            )

    def test_issue_type(self):
        assert IssueType.BIAS == "bias"
        assert IssueType.PII == "pii"
        assert len(IssueType) == 6

    def test_flagged_issue(self):
        fi = FlaggedIssue(
            issue_type=IssueType.PII, severity=IssueSeverity.CRITICAL,
            description="PII exposed", potential_harm="Data breach",
            remediation="Remove PII", blocks_output=True,
        )
        assert fi.blocks_output is True

    def test_ethics_review(self):
        er = EthicsReview(
            output_summary="Review summary", verdict="PASS",
            flagged_issues=[], bias_analysis="No bias found",
            pii_scan_results="No PII", compliance_assessment="Compliant",
            safety_assessment="Safe", recommendations=[],
            can_proceed=True,
        )
        assert er.required_remediations == []


# =============================================================================
# SME Schema Tests
# =============================================================================

class TestSMESchemas:
    def test_sme_interaction_mode(self):
        assert SMEInteractionMode.ADVISOR == "advisor"
        assert len(SMEInteractionMode) == 3

    def test_advisor_report(self):
        ar = AdvisorReport(
            sme_persona="Security Analyst",
            reviewed_content="Auth flow",
            domain_corrections=["Add rate limiting"],
            missing_considerations=["Session timeout"],
            recommendations=["Use OWASP best practices"],
            confidence=0.9,
        )
        assert ar.confidence == 0.9

    def test_co_executor_section(self):
        s = CoExecutorSection(
            sme_persona="Data Engineer", section_title="Data Layer",
            content="Schema design...", domain_context="SQL expertise",
            integration_notes="Integrates with API layer",
        )
        assert s.section_title == "Data Layer"

    def test_co_executor_report(self):
        cr = CoExecutorReport(
            sme_persona="Data Engineer",
            contributed_sections=[],
            coordination_notes="Coordinated with Executor",
            domain_assumptions=["PostgreSQL"],
        )
        assert len(cr.contributed_sections) == 0

    def test_debate_position(self):
        dp = DebatePosition(
            sme_persona="Security Analyst", position="Need auth",
            domain_rationale="OWASP requirements",
            supporting_evidence=["OWASP Top 10"],
            confidence=0.85,
        )
        assert dp.confidence == 0.85

    def test_debater_report(self):
        dr = DebaterReport(
            sme_persona="Security Analyst", debate_round=1,
            position=DebatePosition(
                sme_persona="Security Analyst", position="Need auth",
                domain_rationale="OWASP", supporting_evidence=[],
                confidence=0.8,
            ),
            counter_arguments_addressed=["Performance concerns"],
            remaining_concerns=["Rate limiting"],
            willingness_to_concede=0.3,
        )
        assert dr.willingness_to_concede == 0.3

    def test_sme_advisory_report(self):
        sar = SMEAdvisoryReport(
            sme_persona="Cloud Architect",
            interaction_mode=SMEInteractionMode.ADVISOR,
            domain="Cloud Infrastructure",
            task_context="Architecture review",
            findings=["Need auto-scaling"],
            recommendations=["Use AKS"],
            confidence=0.85,
            skills_used=["azure-architect"],
        )
        assert sar.advisor_report is None
        assert sar.caveats == []

    def test_sme_advisory_confidence_bounds(self):
        with pytest.raises(ValidationError):
            SMEAdvisoryReport(
                sme_persona="X", interaction_mode=SMEInteractionMode.ADVISOR,
                domain="X", task_context="X", findings=[], recommendations=[],
                confidence=1.5, skills_used=[],
            )
