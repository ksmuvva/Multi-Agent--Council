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
            literal_request="Write a hello world function",
            inferred_intent="User wants a simple hello world program",
            sub_tasks=[],
            missing_info=[],
            assumptions=["Language is Python"],
            modality=ModalityType.CODE,
            recommended_approach="Direct implementation",
        )

        assert report.literal_request == "Write a hello world function"
        assert report.modality == ModalityType.CODE
        assert report.escalation_needed is False

    def test_valid_full_report(self):
        """Test creating a valid full report with all fields."""
        report = TaskIntelligenceReport(
            literal_request="Build a full-stack app",
            inferred_intent="User wants a complete web application",
            sub_tasks=[
                SubTask(description="Frontend", dependencies=["Backend"]),
                SubTask(description="Backend", dependencies=[]),
            ],
            missing_info=[
                MissingInfo(
                    requirement="Tech stack",
                    severity="important",
                    impact="Cannot proceed without knowing framework",
                ),
            ],
            assumptions=["React for frontend", "Node.js for backend"],
            modality=ModalityType.DOCUMENT,
            recommended_approach="MVP first approach",
            escalation_needed=True,
        )

        assert len(report.sub_tasks) == 2
        assert len(report.missing_info) == 1
        assert report.escalation_needed is True

    def test_invalid_modality(self):
        """Test validation rejects invalid modality."""
        with pytest.raises(ValidationError):
            TaskIntelligenceReport(
                literal_request="Test",
                inferred_intent="Test",
                sub_tasks=[],
                missing_info=[],
                assumptions=[],
                modality="invalid_modality",  # Invalid
                recommended_approach="Test",
            )

    def test_sub_task_validation(self):
        """Test SubTask validation."""
        task = SubTask(
            description="Write code",
            dependencies=["Design"],
        )

        assert task.description == "Write code"
        assert task.dependencies == ["Design"]

    def test_missing_info_validation(self):
        """Test MissingInfo validation."""
        info = MissingInfo(
            requirement="API key",
            severity="critical",
            impact="Cannot make API calls",
        )

        assert info.requirement == "API key"
        assert info.severity == "critical"
        assert info.impact == "Cannot make API calls"


# =============================================================================
# ExecutionPlan Tests
# =============================================================================

class TestExecutionPlan:
    """Tests for ExecutionPlan schema."""

    def test_valid_minimal_plan(self):
        """Test creating a valid minimal plan."""
        plan = ExecutionPlan(
            task_summary="Execute a simple task",
            total_steps=1,
            steps=[
                ExecutionStep(
                    step_number=1,
                    description="Execute task",
                    agent_assignments=[
                        AgentAssignment(
                            agent_name="Executor",
                            role="Code generation",
                            reason="Primary executor",
                        )
                    ],
                    dependencies=[],
                ),
            ],
            critical_path=[1],
        )

        assert len(plan.steps) == 1
        assert plan.total_steps == 1

    def test_valid_complex_plan(self):
        """Test creating a valid complex plan with dependencies."""
        plan = ExecutionPlan(
            task_summary="Build a REST API",
            total_steps=2,
            steps=[
                ExecutionStep(
                    step_number=1,
                    description="Research",
                    agent_assignments=[
                        AgentAssignment(
                            agent_name="Researcher",
                            role="Evidence gathering",
                            reason="Need domain knowledge",
                        )
                    ],
                    dependencies=[],
                ),
                ExecutionStep(
                    step_number=2,
                    description="Execute",
                    agent_assignments=[
                        AgentAssignment(
                            agent_name="Executor",
                            role="Code generation",
                            reason="Create the implementation",
                        )
                    ],
                    dependencies=[1],
                ),
            ],
            critical_path=[1, 2],
            risk_factors=["Time estimation uncertainty"],
        )

        assert len(plan.steps) == 2
        assert plan.steps[1].dependencies == [1]

    def test_step_ordering(self):
        """Test that steps maintain their order."""
        plan = ExecutionPlan(
            task_summary="Multi-step task",
            total_steps=5,
            steps=[
                ExecutionStep(
                    step_number=i,
                    description=f"Step {i}",
                    agent_assignments=[
                        AgentAssignment(
                            agent_name="Agent",
                            role="Execute",
                            reason="Required",
                        )
                    ],
                    dependencies=[],
                )
                for i in range(1, 6)
            ],
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

    def test_valid_verification_report(self):
        """Test creating a valid verification report."""
        report = VerificationReport(
            total_claims_checked=2,
            claims=[
                Claim(
                    claim_text="Python is dynamically typed",
                    confidence=9,
                    fabrication_risk=FabricationRisk.LOW,
                    verification_method="Documentation check",
                    status=VerificationStatus.VERIFIED,
                    source="Language documentation",
                ),
                Claim(
                    claim_text="Java is slower than Python",
                    confidence=3,
                    fabrication_risk=FabricationRisk.HIGH,
                    verification_method="Benchmark review",
                    status=VerificationStatus.CONTRADICTED,
                    source="Benchmark tests",
                ),
            ],
            verified_claims=1,
            unverified_claims=0,
            contradicted_claims=1,
            fabricated_claims=0,
            overall_reliability=0.5,
            verdict="FAIL",
            flagged_claims=[],
            recommended_corrections=["Correct Java performance claim"],
            verification_summary="1 of 2 claims verified.",
        )

        assert len(report.claims) == 2
        assert report.overall_reliability == 0.5
        assert report.claims[0].status == VerificationStatus.VERIFIED

    def test_verification_status_values(self):
        """Test VerificationStatus enum values."""
        valid_statuses = [
            VerificationStatus.VERIFIED,
            VerificationStatus.UNVERIFIED,
            VerificationStatus.CONTRADICTED,
            VerificationStatus.FABRICATED,
        ]

        for status in valid_statuses:
            claim = Claim(
                claim_text="Test claim",
                confidence=5,
                fabrication_risk=FabricationRisk.LOW,
                verification_method="Test",
                status=status,
            )
            assert claim.status == status

    def test_confidence_range(self):
        """Test confidence scores are in valid range (1-10)."""
        with pytest.raises(ValidationError):
            Claim(
                claim_text="Test",
                confidence=15,  # Invalid: > 10
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


# =============================================================================
# CritiqueReport Tests
# =============================================================================

def _make_critique_report(**overrides) -> CritiqueReport:
    """Helper to create a CritiqueReport with defaults."""
    defaults = dict(
        solution_summary="Test solution",
        attacks=[],
        logic_attack=LogicAttack(
            invalid_arguments=[], fallacies_identified=[]
        ),
        completeness_attack=CompletenessAttack(
            covered=["Item 1"], missing=[], assumptions=[]
        ),
        quality_attack=QualityAttack(
            weaknesses=[], improvements=[]
        ),
        contradiction_scan=ContradictionScan(
            external_contradictions=[], inconsistencies=[]
        ),
        red_team_argumentation=RedTeamArgument(
            adversary_perspective="None identified",
            attack_surface=[], failure_modes=[], worst_case_scenarios=[]
        ),
        overall_assessment="Acceptable",
        critical_issues=[],
        recommended_revisions=[],
        would_approve=True,
    )
    defaults.update(overrides)
    return CritiqueReport(**defaults)


class TestCritiqueReport:
    """Tests for CritiqueReport schema."""

    def test_valid_critique_report(self):
        """Test creating a valid critique report."""
        report = _make_critique_report(
            attacks=[
                Attack(
                    vector=AttackVector.LOGIC,
                    target="Section 2",
                    finding="Minor logical gap",
                    severity=SeverityLevel.LOW,
                    description="Minor logical gap found in section 2",
                    scenario="Could lead to incorrect conclusions",
                    suggestion="Add more detail",
                ),
            ],
        )

        assert len(report.attacks) == 1
        assert report.attacks[0].severity == SeverityLevel.LOW

    def test_all_attack_vectors(self):
        """Test all defined attack vectors."""
        vectors = [
            AttackVector.LOGIC,
            AttackVector.COMPLETENESS,
            AttackVector.QUALITY,
            AttackVector.CONTRADICTION,
            AttackVector.RED_TEAM,
        ]

        attacks = [
            Attack(
                vector=v,
                target="Target",
                finding=f"Test {v}",
                severity=SeverityLevel.LOW,
                description=f"Test {v}",
                scenario="Scenario",
                suggestion="Fix",
            )
            for v in vectors
        ]

        report = _make_critique_report(attacks=attacks)

        assert len(report.attacks) == 5

    def test_finding_severity_levels(self):
        """Test all severity levels."""
        severities = [
            SeverityLevel.CRITICAL,
            SeverityLevel.HIGH,
            SeverityLevel.MEDIUM,
            SeverityLevel.LOW,
        ]

        attacks = [
            Attack(
                vector=AttackVector.LOGIC,
                target="Target",
                finding=f"Test {severity}",
                severity=severity,
                description=f"Test {severity}",
                scenario="Test scenario",
                suggestion="Fix it",
            )
            for severity in severities
        ]

        report = _make_critique_report(attacks=attacks)

        assert len(report.attacks) == 4


# =============================================================================
# ReviewVerdict Tests
# =============================================================================

def _make_check_item(name: str, passed: bool = True) -> CheckItem:
    """Helper to create a CheckItem."""
    return CheckItem(
        check_name=name,
        passed=passed,
        notes=f"{name} {'passed' if passed else 'failed'}",
        severity_if_failed="medium",
    )


def _make_quality_gates(all_pass: bool = True) -> QualityGateResults:
    """Helper to create QualityGateResults."""
    return QualityGateResults(
        completeness=_make_check_item("Completeness", all_pass),
        consistency=_make_check_item("Consistency", all_pass),
        verifier_signoff=_make_check_item("Verifier Sign-off", all_pass),
        critic_findings_addressed=_make_check_item("Critic Findings", all_pass),
        readability=_make_check_item("Readability", all_pass),
    )


class TestReviewVerdict:
    """Tests for ReviewVerdict schema."""

    def test_valid_review_verdict(self):
        """Test creating a valid review verdict."""
        verdict = ReviewVerdict(
            verdict=Verdict.PASS,
            confidence=0.9,
            quality_gate_results=_make_quality_gates(True),
            reasons=["All quality checks passed"],
            can_revise=True,
            summary="Quality is acceptable, proceed to formatting",
        )

        assert verdict.verdict == Verdict.PASS
        assert verdict.quality_gate_results.completeness.passed is True

    def test_verdict_values(self):
        """Test all valid verdict values."""
        valid_verdicts = [
            Verdict.PASS,
            Verdict.FAIL,
            Verdict.PASS_WITH_CAVEATS,
            Verdict.REVISE,
            Verdict.REJECT,
            Verdict.ESCALATE,
        ]

        for v in valid_verdicts:
            review = ReviewVerdict(
                verdict=v,
                confidence=0.5,
                quality_gate_results=_make_quality_gates(),
                reasons=["Test"],
                can_revise=True,
                summary="Test",
            )
            assert review.verdict == v


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
            task_context="Designing cloud infrastructure",
            findings=["Use serverless for cost efficiency"],
            recommendations=[
                "Use serverless for cost efficiency",
                "Implement auto-scaling",
            ],
            confidence=0.9,
            skills_used=["architecture-design"],
        )

        assert advisory.sme_persona == "cloud_architect"
        assert advisory.interaction_mode == SMEInteractionMode.ADVISOR
        assert len(advisory.recommendations) == 2

    def test_all_interaction_modes(self):
        """Test all interaction modes."""
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
                task_context="Test context",
                findings=["Finding"],
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

        # Test to_dict() method
        data = report.model_dump()

        assert data["literal_request"] == "Test request"
        assert data["modality"] == "text"
        assert "sub_tasks" in data

        # Test JSON serialization
        import json
        json_str = json.dumps(data, default=str)
        assert json_str is not None

    def test_verification_report_serialization(self):
        """Test VerificationReport serialization."""
        report = VerificationReport(
            total_claims_checked=1,
            claims=[
                Claim(
                    claim_text="Test claim",
                    confidence=8,
                    fabrication_risk=FabricationRisk.LOW,
                    verification_method="Manual check",
                    status=VerificationStatus.VERIFIED,
                    source="Test source",
                ),
            ],
            verified_claims=1,
            unverified_claims=0,
            contradicted_claims=0,
            fabricated_claims=0,
            overall_reliability=0.8,
            verdict="PASS",
            flagged_claims=[],
            recommended_corrections=[],
            verification_summary="All claims verified.",
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
                # Missing required fields
                literal_request="Test",
            )

    def test_string_trimming(self):
        """Test that strings are properly handled."""
        # Create with leading/trailing whitespace
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
