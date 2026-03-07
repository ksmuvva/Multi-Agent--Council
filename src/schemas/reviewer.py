"""
Reviewer Agent Schemas

Pydantic v2 models for the Final Reviewer subagent.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class Verdict(str, Enum):
    """Pass/fail verdict."""
    PASS = "PASS"
    FAIL = "FAIL"


class CheckItem(BaseModel):
    """A single quality check item."""
    check_name: str = Field(..., description="Name of the check")
    passed: bool = Field(..., description="Whether this check passed")
    notes: str = Field(..., description="Notes about this check")
    severity_if_failed: str = Field(
        ...,
        description="Severity if this check failed"
    )


class Revision(BaseModel):
    """A required revision."""
    category: str = Field(..., description="Category of the revision")
    description: str = Field(..., description="What needs to be revised")
    reason: str = Field(..., description="Why this revision is needed")
    priority: str = Field(..., description="Priority: critical/high/medium/low")
    specific_instructions: str = Field(
        ...,
        description="Specific instructions for the revision"
    )


class QualityGateResults(BaseModel):
    """Results from quality gate checks."""
    completeness: CheckItem = Field(..., description="Completeness check")
    consistency: CheckItem = Field(..., description="Consistency check")
    verifier_signoff: CheckItem = Field(..., description="Verifier sign-off")
    critic_findings_addressed: CheckItem = Field(
        ...,
        description="Critic findings addressed"
    )
    readability: CheckItem = Field(..., description="Readability check")
    code_review_passed: Optional[CheckItem] = Field(
        None,
        description="Code review (if applicable)"
    )


class ArbitrationInput(BaseModel):
    """Input for Quality Arbiter arbitration."""
    reviewer_verdict: Verdict = Field(..., description="Reviewer's verdict")
    verifier_verdict: Verdict = Field(..., description="Verifier's verdict")
    critic_verdict: Verdict = Field(..., description="Critic's verdict")
    disagreement_reason: str = Field(
        ...,
        description="Why there's disagreement"
    )
    debate_rounds_completed: int = Field(
        ...,
        ge=0,
        description="How many debate rounds were completed"
    )


class ReviewVerdict(BaseModel):
    """
    Structured output from the Reviewer subagent.

    Contains the final quality gate decision with pass/fail verdict,
    reasons, and revision instructions if failing.
    """
    verdict: Verdict = Field(..., description="Final PASS or FAIL verdict")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in this verdict (0-1)"
    )
    quality_gate_results: QualityGateResults = Field(
        ...,
        description="Results from all quality checks"
    )
    reasons: List[str] = Field(
        ...,
        description="Reasons supporting this verdict"
    )
    revision_instructions: List[Revision] = Field(
        default_factory=list,
        description="Specific revisions needed if FAIL"
    )
    revision_count: int = Field(
        default=0,
        ge=0,
        description="Number of revision cycles completed"
    )
    max_revisions: int = Field(
        default=2,
        description="Maximum allowed revision cycles"
    )
    can_revise: bool = Field(
        ...,
        description="Whether another revision cycle is allowed"
    )
    arbitration_needed: bool = Field(
        default=False,
        description="Whether Quality Arbiter arbitration is needed"
    )
    arbitration_input: Optional[ArbitrationInput] = Field(
        None,
        description="Input for arbitration if needed"
    )
    summary: str = Field(..., description="Human-readable summary of the review")
    tier_4_arbiter_involved: bool = Field(
        default=False,
        description="Whether Quality Arbiter was involved (Tier 4)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "verdict": "PASS",
                "confidence": 0.9,
                "quality_gate_results": {
                    "completeness": {
                        "check_name": "Completeness",
                        "passed": True,
                        "notes": "All requirements addressed",
                        "severity_if_failed": "high"
                    },
                    "consistency": {
                        "check_name": "Consistency",
                        "passed": True,
                        "notes": "All contributions consistent",
                        "severity_if_failed": "medium"
                    },
                    "verifier_signoff": {
                        "check_name": "Verifier Sign-off",
                        "passed": True,
                        "notes": "Verifier passed",
                        "severity_if_failed": "critical"
                    },
                    "critic_findings_addressed": {
                        "check_name": "Critic Findings",
                        "passed": True,
                        "notes": "All critic issues addressed",
                        "severity_if_failed": "high"
                    },
                    "readability": {
                        "check_name": "Readability",
                        "passed": True,
                        "notes": "Clear and well-structured",
                        "severity_if_failed": "low"
                    }
                },
                "reasons": [
                    "All quality checks passed",
                    "Verifier and Critic both satisfied",
                    "Meets all original requirements"
                ],
                "revision_instructions": [],
                "revision_count": 0,
                "can_revise": True,
                "summary": "Output passes all quality gates. Ready for formatting."
            }
        }
