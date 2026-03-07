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
    PlanStep,
    Dependency,
)
from src.schemas.verifier import (
    VerificationReport,
    ClaimExtraction,
    ClaimStatus,
)
from src.schemas.critic import (
    CritiqueReport,
    Finding,
    AttackVector,
    SeverityLevel,
)
from src.schemas.reviewer import (
    ReviewVerdict,
    QualityGate,
    Verdict,
)
from src.schemas.sme import (
    SMEAdvisoryReport,
    AdvisoryType,
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
            steps=[
                PlanStep(
                    step=1,
                    description="Execute task",
                    agent="Executor",
                    dependencies=[],
                ),
            ],
            estimated_duration="5 minutes",
        )

        assert len(plan.steps) == 1
        assert plan.estimated_duration == "5 minutes"

    def test_valid_complex_plan(self):
        """Test creating a valid complex plan with dependencies."""
        plan = ExecutionPlan(
            steps=[
                PlanStep(
                    step=1,
                    description="Research",
                    agent="Researcher",
                    dependencies=[],
                ),
                PlanStep(
                    step=2,
                    description="Execute",
                    agent="Executor",
                    dependencies=["Research"],
                ),
            ],
            estimated_duration="10-15 minutes",
            dependencies=[
                Dependency(
                    from_step="Execute",
                    to_step="Research",
                    type="sequential",
                ),
            ],
            parallel_opportunities=[],
            risk_factors=["Time estimation uncertainty"],
        )

        assert len(plan.steps) == 2
        assert len(plan.dependencies) == 1
        assert plan.dependencies[0].type == "sequential"

    def test_step_ordering(self):
        """Test that steps maintain their order."""
        plan = ExecutionPlan(
            steps=[
                PlanStep(step=i, description=f"Step {i}", agent="Agent", dependencies=[])
                for i in range(1, 6)
            ],
            estimated_duration="5 minutes",
        )

        assert plan.steps[0].step == 1
        assert plan.steps[-1].step == 5
        assert len(plan.steps) == 5


# =============================================================================
# VerificationReport Tests
# =============================================================================

class TestVerificationReport:
    """Tests for VerificationReport schema."""

    def test_valid_verification_report(self):
        """Test creating a valid verification report."""
        report = VerificationReport(
            claims=[
                ClaimExtraction(
                    claim="Python is dynamically typed",
                    verification=ClaimStatus.VERIFIED,
                    confidence=0.95,
                    source="Language documentation",
                ),
                ClaimExtraction(
                    claim="Java is slower than Python",
                    verification=ClaimStatus.CONTRADICTED,
                    confidence=0.8,
                    source="Benchmark tests",
                ),
            ],
            factual_accuracy_score=0.85,
            hallucination_risk="low",
            recommendations=["Add more citations"],
        )

        assert len(report.claims) == 2
        assert report.factual_accuracy_score == 0.85
        assert report.claims[0].verification == ClaimStatus.VERIFIED

    def test_claim_status_validation(self):
        """Test ClaimStatus enum values."""
        valid_statuses = [
            ClaimStatus.VERIFIED,
            ClaimStatus.UNVERIFIED,
            ClaimStatus.CONTRADICTED,
            ClaimStatus.FABRICATED,
        ]

        for status in valid_statuses:
            claim = ClaimExtraction(
                claim="Test claim",
                verification=status,
                confidence=0.5,
            )
            assert claim.verification == status

    def test_confidence_range(self):
        """Test confidence scores are in valid range."""
        with pytest.raises(ValidationError):
            ClaimExtraction(
                claim="Test",
                verification=ClaimStatus.VERIFIED,
                confidence=1.5,  # Invalid: > 1.0
                source="Test",
            )

        with pytest.raises(ValidationError):
            ClaimExtraction(
                claim="Test",
                verification=ClaimStatus.VERIFIED,
                confidence=-0.1,  # Invalid: < 0
                source="Test",
            )


# =============================================================================
# CritiqueReport Tests
# =============================================================================

class TestCritiqueReport:
    """Tests for CritiqueReport schema."""

    def test_valid_critique_report(self):
        """Test creating a valid critique report."""
        report = CritiqueReport(
            attack_vectors_tested=[
                AttackVector.LOGIC,
                AttackVector.COMPLETENESS,
            ],
            findings=[
                Finding(
                    category=AttackVector.LOGIC,
                    severity=SeverityLevel.LOW,
                    description="Minor logical gap",
                    location="Section 2",
                    recommendation="Add more detail",
                ),
            ],
            overall_score=0.85,
            recommendations=["Expand section 2"],
        )

        assert len(report.attack_vectors_tested) == 2
        assert len(report.findings) == 1
        assert report.findings[0].severity == SeverityLevel.LOW

    def test_all_attack_vectors(self):
        """Test all defined attack vectors."""
        vectors = [
            AttackVector.LOGIC,
            AttackVector.COMPLETENESS,
            AttackVector.QUALITY,
            AttackVector.CONTRADICTION,
            AttackVector.RED_TEAM,
        ]

        report = CritiqueReport(
            attack_vectors_tested=vectors,
            findings=[],
            overall_score=1.0,
        )

        assert len(report.attack_vectors_tested) == 5

    def test_finding_severity_levels(self):
        """Test all severity levels."""
        severities = [
            SeverityLevel.CRITICAL,
            SeverityLevel.HIGH,
            SeverityLevel.MEDIUM,
            SeverityLevel.LOW,
            SeverityLevel.INFO,
        ]

        findings = [
            Finding(
                category=AttackVector.LOGIC,
                severity=severity,
                description=f"Test {severity}",
            )
            for severity in severities
        ]

        report = CritiqueReport(
            attack_vectors_tested=[AttackVector.LOGIC],
            findings=findings,
            overall_score=0.5,
        )

        assert len(report.findings) == 5


# =============================================================================
# ReviewVerdict Tests
# =============================================================================

class TestReviewVerdict:
    """Tests for ReviewVerdict schema."""

    def test_valid_review_verdict(self):
        """Test creating a valid review verdict."""
        verdict = ReviewVerdict(
            verdict=Verdict.PROCEED_TO_FORMATTER,
            quality_gates={
                "completeness": QualityGate.PASS,
                "consistency": QualityGate.PASS,
                "verifier_signoff": QualityGate.PASS,
                "critic_findings": QualityGate.PASS,
                "readability": QualityGate.PASS,
            },
            final_recommendation="Quality is acceptable, proceed to formatting",
        )

        assert verdict.verdict == Verdict.PROCEED_TO_FORMATTER
        assert verdict.quality_gates["completeness"] == QualityGate.PASS

    def test_verdict_matrix_values(self):
        """Test all valid verdict values."""
        valid_verdicts = [
            Verdict.PROCEED_TO_FORMATTER,
            Verdict.EXECUTOR_REVISE,
            Verdict.RESEARCHER_REVERIFY,
            Verdict.FULL_REGENERATION,
        ]

        for verdict in valid_verdicts:
            review = ReviewVerdict(
                verdict=verdict,
                quality_gates={},
                final_recommendation="Test",
            )
            assert review.verdict == verdict

    def test_quality_gate_values(self):
        """Test all quality gate values."""
        gate_values = [QualityGate.PASS, QualityGate.FAIL]

        for gate_value in gate_values:
            gates = {
                "test_gate": gate_value,
            }
            verdict = ReviewVerdict(
                verdict=Verdict.PROCEED_TO_FORMATTER,
                quality_gates=gates,
                final_recommendation="Test",
            )
            assert verdict.quality_gates["test_gate"] == gate_value


# =============================================================================
# SMEAdvisoryReport Tests
# =============================================================================

class TestSMEAdvisoryReport:
    """Tests for SMEAdvisoryReport schema."""

    def test_valid_sme_advisory(self):
        """Test creating a valid SME advisory report."""
        advisory = SMEAdvisoryReport(
            sme_persona="cloud_architect",
            advisory_type=AdvisoryType.ADVISOR,
            recommendations=[
                "Use serverless for cost efficiency",
                "Implement auto-scaling",
            ],
            domain_insights="Cloud architecture patterns for this use case",
            confidence=0.9,
        )

        assert advisory.sme_persona == "cloud_architect"
        assert advisory.advisory_type == AdvisoryType.ADVISOR
        assert len(advisory.recommendations) == 2

    def test_all_advisory_types(self):
        """Test all advisory types."""
        types = [
            AdvisoryType.ADVISOR,
            AdvisoryType.CO_EXECUTOR,
            AdvisoryType.DEBATER,
        ]

        for advisory_type in types:
            advisory = SMEAdvisoryReport(
                sme_persona="test_sme",
                advisory_type=advisory_type,
                recommendations=[],
                domain_insights="Test",
            )
            assert advisory.advisory_type == advisory_type


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
            claims=[
                ClaimExtraction(
                    claim="Test claim",
                    verification=ClaimStatus.VERIFIED,
                    confidence=0.8,
                    source="Test source",
                ),
            ],
            factual_accuracy_score=0.8,
            hallucination_risk="low",
            recommendations=[],
        )

        data = report.model_dump()

        assert data["factual_accuracy_score"] == 0.8
        assert "claims" in data
        assert data["claims"][0"]["verification"] == "verified"


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
