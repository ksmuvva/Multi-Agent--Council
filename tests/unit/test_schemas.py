"""
Unit Tests for Pydantic Schemas

Tests validation, serialization, and business logic for all schema models.
"""

import pytest
from pydantic import ValidationError
from datetime import datetime
from typing import List

from src.schemas.analyst import (
    TaskIntelligenceReport,
    SubTask,
    MissingInfo,
    ModalityType,
)
from src.schemas.planner import (
    ExecutionPlan,
    ExecutionStep,
    AgentAssignment,
    ParallelGroup,
)
from src.schemas.verifier import (
    VerificationReport,
    Claim,
    ClaimBatch,
    VerificationStatus,
    FabricationRisk,
)
from src.schemas.critic import (
    CritiqueReport,
    Attack,
    AttackVector,
    SeverityLevel,
    LogicAttack,
    CompletenessAttack,
    QualityAttack,
    ContradictionScan,
    RedTeamArgument,
)
from src.schemas.reviewer import (
    ReviewVerdict,
    QualityGateResults,
    CheckItem,
    Verdict,
    Revision,
)
from src.schemas.sme import (
    SMEAdvisoryReport,
    SMEInteractionMode,
)


# =============================================================================
# TaskIntelligenceReport Tests
# =============================================================================

class TestTaskIntelligenceReport:
    """Tests for TaskIntelligenceReport schema."""

    def test_valid_minimal_report(self):
        """Test creating a valid minimal report."""
        report = TaskIntelligenceReport(
            literal_request="Test request",
            inferred_intent="Test intent",
            sub_tasks=[],
            missing_info=[],
            assumptions=[],
            modality=ModalityType.TEXT,
            recommended_approach="Test approach",
        )
        assert report.literal_request == "Test request"
        assert report.modality == ModalityType.TEXT

    def test_valid_full_report(self):
        """Test creating a full report with all fields."""
        report = TaskIntelligenceReport(
            literal_request="Build a REST API",
            inferred_intent="Create backend endpoints",
            sub_tasks=[
                SubTask(description="Design models", dependencies=[]),
                SubTask(description="Implement endpoints", dependencies=["Design models"]),
            ],
            missing_info=[
                MissingInfo(
                    requirement="Auth method",
                    severity="important",
                    impact="Security depends on this",
                    default_assumption="JWT",
                ),
            ],
            assumptions=["Python/FastAPI", "PostgreSQL"],
            modality=ModalityType.CODE,
            recommended_approach="Start with models",
            escalation_needed=False,
            suggested_tier=2,
            confidence=0.9,
        )
        assert len(report.sub_tasks) == 2
        assert len(report.missing_info) == 1
        assert report.suggested_tier == 2

    def test_modality_types(self):
        """Test all modality type values."""
        for modality in ModalityType:
            report = TaskIntelligenceReport(
                literal_request="Test",
                inferred_intent="Test",
                sub_tasks=[],
                missing_info=[],
                assumptions=[],
                modality=modality,
                recommended_approach="Test",
            )
            assert report.modality == modality


# =============================================================================
# ExecutionPlan Tests
# =============================================================================

class TestExecutionPlan:
    """Tests for ExecutionPlan schema."""

    def _make_step(self, step_number=1, description="Execute task", agent_name="Executor"):
        """Helper to create an ExecutionStep."""
        return ExecutionStep(
            step_number=step_number,
            description=description,
            agent_assignments=[
                AgentAssignment(
                    agent_name=agent_name,
                    role="primary",
                    reason="Main executor",
                ),
            ],
            dependencies=[],
        )

    def test_valid_minimal_plan(self):
        """Test creating a valid minimal plan."""
        plan = ExecutionPlan(
            task_summary="Simple task",
            total_steps=1,
            steps=[self._make_step()],
            critical_path=[1],
        )
        assert len(plan.steps) == 1
        assert plan.total_steps == 1

    def test_valid_complex_plan(self):
        """Test creating a valid complex plan with dependencies."""
        step1 = self._make_step(1, "Research", "Researcher")
        step2 = ExecutionStep(
            step_number=2,
            description="Execute",
            agent_assignments=[
                AgentAssignment(agent_name="Executor", role="primary", reason="Implement"),
            ],
            dependencies=[1],
            can_parallelize=False,
        )
        plan = ExecutionPlan(
            task_summary="Complex task",
            total_steps=2,
            steps=[step1, step2],
            critical_path=[1, 2],
            parallel_groups=[],
            risk_factors=["Time estimation uncertainty"],
        )
        assert len(plan.steps) == 2
        assert plan.steps[1].dependencies == [1]

    def test_step_ordering(self):
        """Test that steps maintain their order."""
        steps = [self._make_step(i, f"Step {i}") for i in range(1, 6)]
        plan = ExecutionPlan(
            task_summary="Multi-step",
            total_steps=5,
            steps=steps,
            critical_path=[1, 2, 3, 4, 5],
        )
        assert plan.steps[0].step_number == 1
        assert plan.steps[-1].step_number == 5
        assert len(plan.steps) == 5


# =============================================================================
# VerificationReport Tests
# =============================================================================

class TestVerificationReport:
    """Tests for VerificationReport schema."""

    def _make_claim(self, text="Test claim", confidence=8, status=VerificationStatus.VERIFIED):
        """Helper to create a Claim."""
        return Claim(
            claim_text=text,
            confidence=confidence,
            fabrication_risk=FabricationRisk.LOW,
            source="Test source",
            verification_method="Test method",
            status=status,
        )

    def test_valid_verification_report(self):
        """Test creating a valid verification report."""
        report = VerificationReport(
            total_claims_checked=2,
            claims=[
                self._make_claim("Python is dynamically typed", 9, VerificationStatus.VERIFIED),
                self._make_claim("Java is slower than Python", 5, VerificationStatus.CONTRADICTED),
            ],
            verified_claims=1,
            unverified_claims=0,
            contradicted_claims=1,
            fabricated_claims=0,
            overall_reliability=0.85,
            verdict="PASS",
            flagged_claims=[],
            recommended_corrections=[],
            verification_summary="1 verified, 1 contradicted",
        )
        assert len(report.claims) == 2
        assert report.overall_reliability == 0.85

    def test_verification_status_values(self):
        """Test all VerificationStatus enum values."""
        valid_statuses = [
            VerificationStatus.VERIFIED,
            VerificationStatus.UNVERIFIED,
            VerificationStatus.CONTRADICTED,
            VerificationStatus.FABRICATED,
        ]
        for status in valid_statuses:
            claim = self._make_claim(status=status)
            assert claim.status == status

    def test_confidence_range(self):
        """Test confidence scores are in valid range (1-10)."""
        with pytest.raises(ValidationError):
            Claim(
                claim_text="Test",
                confidence=11,  # Invalid: > 10
                fabrication_risk=FabricationRisk.LOW,
                verification_method="Test",
                status=VerificationStatus.VERIFIED,
            )

        with pytest.raises(ValidationError):
            Claim(
                claim_text="Test",
                confidence=0,  # Invalid: < 1
                fabrication_risk=FabricationRisk.LOW,
                verification_method="Test",
                status=VerificationStatus.VERIFIED,
            )

    def test_reliability_range(self):
        """Test overall_reliability must be 0-1."""
        with pytest.raises(ValidationError):
            VerificationReport(
                total_claims_checked=0,
                claims=[],
                verified_claims=0,
                unverified_claims=0,
                contradicted_claims=0,
                fabricated_claims=0,
                overall_reliability=1.5,  # Invalid
                verdict="FAIL",
                flagged_claims=[],
                recommended_corrections=[],
                verification_summary="Test",
            )


# =============================================================================
# CritiqueReport Tests
# =============================================================================

class TestCritiqueReport:
    """Tests for CritiqueReport schema."""

    def _make_critique(self, would_approve=True):
        """Helper to create a CritiqueReport."""
        return CritiqueReport(
            solution_summary="Test solution",
            attacks=[
                Attack(
                    vector=AttackVector.LOGIC,
                    target="argument",
                    finding="Minor logical gap",
                    severity=SeverityLevel.LOW,
                    description="Small gap in logic",
                    scenario="Could cause confusion",
                    suggestion="Add more detail",
                ),
            ],
            logic_attack=LogicAttack(
                invalid_arguments=["Gap in reasoning"],
                fallacies_identified=["None"],
            ),
            completeness_attack=CompletenessAttack(
                covered=["Main points"],
                missing=["Edge cases"],
                assumptions=["User is technical"],
            ),
            quality_attack=QualityAttack(
                weaknesses=["Minor style issues"],
                improvements=["Add examples"],
            ),
            contradiction_scan=ContradictionScan(
                external_contradictions=[],
                inconsistencies=[],
            ),
            red_team_argumentation=RedTeamArgument(
                adversary_perspective="Low risk target",
                attack_surface=["Minimal"],
                failure_modes=["None critical"],
                worst_case_scenarios=["Minor data inconsistency"],
            ),
            overall_assessment="Solution is adequate",
            critical_issues=[],
            recommended_revisions=[],
            would_approve=would_approve,
        )

    def test_valid_critique_report(self):
        """Test creating a valid critique report."""
        report = self._make_critique()
        assert len(report.attacks) == 1
        assert report.attacks[0].severity == SeverityLevel.LOW
        assert report.would_approve is True

    def test_all_attack_vectors(self):
        """Test all defined attack vectors."""
        vectors = [
            AttackVector.LOGIC,
            AttackVector.COMPLETENESS,
            AttackVector.QUALITY,
            AttackVector.CONTRADICTION,
            AttackVector.RED_TEAM,
        ]
        for vector in vectors:
            attack = Attack(
                vector=vector,
                target="test",
                finding="test finding",
                severity=SeverityLevel.LOW,
                description="test",
                scenario="test",
                suggestion="test",
            )
            assert attack.vector == vector

    def test_finding_severity_levels(self):
        """Test all severity levels."""
        severities = [
            SeverityLevel.CRITICAL,
            SeverityLevel.HIGH,
            SeverityLevel.MEDIUM,
            SeverityLevel.LOW,
        ]
        for severity in severities:
            attack = Attack(
                vector=AttackVector.LOGIC,
                target="test",
                finding=f"Test {severity}",
                severity=severity,
                description="test",
                scenario="test",
                suggestion="test",
            )
            assert attack.severity == severity


# =============================================================================
# ReviewVerdict Tests
# =============================================================================

class TestReviewVerdict:
    """Tests for ReviewVerdict schema."""

    def _make_check(self, name="Test Check", passed=True):
        """Helper to create a CheckItem."""
        return CheckItem(
            check_name=name,
            passed=passed,
            notes="Test notes",
            severity_if_failed="medium",
        )

    def _make_quality_gates(self, all_pass=True):
        """Helper to create QualityGateResults."""
        return QualityGateResults(
            completeness=self._make_check("Completeness", all_pass),
            consistency=self._make_check("Consistency", all_pass),
            verifier_signoff=self._make_check("Verifier Sign-off", all_pass),
            critic_findings_addressed=self._make_check("Critic Findings", all_pass),
            readability=self._make_check("Readability", all_pass),
        )

    def test_valid_review_verdict(self):
        """Test creating a valid review verdict."""
        verdict = ReviewVerdict(
            verdict=Verdict.PASS,
            confidence=0.9,
            quality_gate_results=self._make_quality_gates(),
            reasons=["All checks passed"],
            can_revise=True,
            summary="Output passes all quality gates.",
        )
        assert verdict.verdict == Verdict.PASS
        assert verdict.confidence == 0.9

    def test_verdict_values(self):
        """Test all valid verdict values."""
        for v in Verdict:
            review = ReviewVerdict(
                verdict=v,
                confidence=0.5,
                quality_gate_results=self._make_quality_gates(),
                reasons=["Test"],
                can_revise=True,
                summary="Test",
            )
            assert review.verdict == v

    def test_fail_verdict_with_revisions(self):
        """Test FAIL verdict includes revision instructions."""
        verdict = ReviewVerdict(
            verdict=Verdict.FAIL,
            confidence=0.6,
            quality_gate_results=self._make_quality_gates(all_pass=False),
            reasons=["Completeness check failed"],
            revision_instructions=[
                Revision(
                    category="completeness",
                    description="Missing requirements",
                    reason="Not all requirements addressed",
                    priority="high",
                    specific_instructions="Add missing sections",
                ),
            ],
            revision_count=1,
            can_revise=True,
            summary="Output needs revision.",
        )
        assert verdict.verdict == Verdict.FAIL
        assert len(verdict.revision_instructions) == 1


# =============================================================================
# SMEAdvisoryReport Tests
# =============================================================================

class TestSMEAdvisoryReport:
    """Tests for SMEAdvisoryReport schema."""

    def test_valid_sme_advisory(self):
        """Test creating a valid SME advisory report."""
        advisory = SMEAdvisoryReport(
            sme_persona="cloud_architect",
            interaction_mode=SMEInteractionMode.ADVISOR,
            domain="Cloud Architecture",
            task_context="Reviewing cloud deployment",
            findings=["Use serverless for cost efficiency"],
            recommendations=["Implement auto-scaling"],
            confidence=0.9,
            skills_used=["architecture-design"],
        )
        assert advisory.sme_persona == "cloud_architect"
        assert advisory.interaction_mode == SMEInteractionMode.ADVISOR

    def test_all_interaction_modes(self):
        """Test all SME interaction modes."""
        modes = [
            SMEInteractionMode.ADVISOR,
            SMEInteractionMode.CO_EXECUTOR,
            SMEInteractionMode.DEBATER,
        ]
        for mode in modes:
            advisory = SMEAdvisoryReport(
                sme_persona="test_sme",
                interaction_mode=mode,
                domain="Test",
                task_context="Test",
                findings=[],
                recommendations=[],
                confidence=0.5,
                skills_used=[],
            )
            assert advisory.interaction_mode == mode


# =============================================================================
# Schema Serialization Tests
# =============================================================================

class TestSchemaSerialization:
    """Tests for JSON serialization of schemas."""

    def test_task_intelligence_report_serialization(self):
        """Test TaskIntelligenceReport serialization."""
        report = TaskIntelligenceReport(
            literal_request="Test request",
            inferred_intent="Test intent",
            sub_tasks=[SubTask(description="Task 1", dependencies=[])],
            missing_info=[],
            assumptions=["Assumption 1"],
            modality=ModalityType.TEXT,
            recommended_approach="Test approach",
        )
        data = report.model_dump()
        assert data["literal_request"] == "Test request"
        assert data["modality"] == "text"
        assert "sub_tasks" in data

        import json
        json_str = json.dumps(data, default=str)
        assert json_str is not None

    def test_verification_report_serialization(self):
        """Test VerificationReport serialization."""
        claim = Claim(
            claim_text="Test claim",
            confidence=8,
            fabrication_risk=FabricationRisk.LOW,
            source="Test source",
            verification_method="Test method",
            status=VerificationStatus.VERIFIED,
        )
        report = VerificationReport(
            total_claims_checked=1,
            claims=[claim],
            verified_claims=1,
            unverified_claims=0,
            contradicted_claims=0,
            fabricated_claims=0,
            overall_reliability=0.8,
            verdict="PASS",
            flagged_claims=[],
            recommended_corrections=[],
            verification_summary="1 claim verified",
        )
        data = report.model_dump()
        assert data["overall_reliability"] == 0.8
        assert "claims" in data
        assert data["claims"][0]["status"] == "verified"


# =============================================================================
# Edge Cases and Validation
# =============================================================================

class TestSchemaValidation:
    """Tests for edge cases and validation rules."""

    def test_empty_lists_allowed(self):
        """Test that empty lists are valid where expected."""
        report = TaskIntelligenceReport(
            literal_request="Test",
            inferred_intent="Test",
            sub_tasks=[],
            missing_info=[],
            assumptions=[],
            modality=ModalityType.TEXT,
            recommended_approach="Test",
        )
        assert len(report.sub_tasks) == 0
        assert len(report.missing_info) == 0

    def test_required_fields(self):
        """Test that required fields raise ValidationError when missing."""
        with pytest.raises(ValidationError):
            TaskIntelligenceReport(
                literal_request="Test",
            )

    def test_string_trimming(self):
        """Test that strings are properly handled."""
        report = TaskIntelligenceReport(
            literal_request="  Test request  ",
            inferred_intent="Test intent",
            sub_tasks=[],
            missing_info=[],
            assumptions=[],
            modality=ModalityType.TEXT,
            recommended_approach="Test",
        )
        # Pydantic v2 doesn't auto-trim by default
        assert "  Test request  " == report.literal_request
