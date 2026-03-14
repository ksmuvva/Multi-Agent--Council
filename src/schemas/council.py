"""
Council Agent Schemas

Pydantic v2 models for the Strategic Council agents:
- Domain Council Chair
- Quality Arbiter
- Ethics & Safety Advisor
"""

from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Dict, Any
from enum import Enum

from src.core.sme_registry import InteractionMode


# =============================================================================
# Domain Council Chair Schemas
# =============================================================================


class SMESelection(BaseModel):
    """A selected SME persona."""
    persona_name: str = Field(..., description="Name of the SME persona")
    persona_domain: str = Field(..., description="Domain of expertise")
    skills_to_load: List[str] = Field(
        ...,
        description="SKILL.md files this SME should load"
    )
    interaction_mode: InteractionMode = Field(
        ...,
        description="How this SME should interact"
    )
    reasoning: str = Field(..., description="Why this SME was selected")
    activation_phase: str = Field(
        ...,
        description="Which phase this SME should participate in"
    )


class SMESelectionReport(BaseModel):
    """
    Structured output from the Domain Council Chair.

    Contains selected SME personas with their configurations
    for a Tier 3-4 task.
    """
    task_summary: str = Field(..., description="Summary of the task")
    selected_smes: List[SMESelection] = Field(
        ...,
        max_length=3,
        description="Selected SME personas (max 3)"
    )
    domain_gaps_identified: List[str] = Field(
        default_factory=list,
        description="Domain gaps that couldn't be filled"
    )
    collaboration_plan: str = Field(
        ...,
        description="How SMEs should collaborate with operational agents"
    )
    expected_sme_contributions: Dict[str, str] = Field(
        ...,
        description="Expected contributions from each SME"
    )
    tier_recommendation: int = Field(
        ...,
        ge=3,
        le=4,
        description="Recommended tier (3 or 4)"
    )
    requires_full_council: bool = Field(
        default=False,
        description="Whether full Council (Arbiter + Ethics) is needed"
    )


# =============================================================================
# Quality Arbiter Schemas
# =============================================================================

class QualityCriteria(BaseModel):
    """A quality acceptance criterion."""
    metric: str = Field(..., description="What is being measured")
    threshold: str = Field(..., description="Pass/fail threshold")
    measurement_method: str = Field(
        ...,
        description="How this will be measured"
    )
    weight: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Weight in overall quality score"
    )


class QualityStandard(BaseModel):
    """
    Structured output from the Quality Arbiter (pre-execution).

    Defines quality acceptance criteria BEFORE execution begins.
    """
    task_summary: str = Field(..., description="Summary of the task")
    quality_criteria: List[QualityCriteria] = Field(
        ...,
        description="Quality acceptance criteria"
    )
    @model_validator(mode="after")
    def validate_criteria_weights(self) -> "QualityStandard":
        """Ensure quality criteria weights sum to approximately 1.0."""
        if self.quality_criteria:
            total_weight = sum(c.weight for c in self.quality_criteria)
            if not (0.95 <= total_weight <= 1.05):
                raise ValueError(
                    f"Quality criteria weights must sum to ~1.0, got {total_weight:.2f}"
                )
        return self

    overall_pass_threshold: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall score needed to pass"
    )
    critical_must_haves: List[str] = Field(
        ...,
        description="Non-negotiable requirements"
    )
    nice_to_haves: List[str] = Field(
        default_factory=list,
        description="Desirable but not required"
    )
    measurement_protocol: str = Field(
        ...,
        description="How quality will be measured"
    )


class DisputedItem(BaseModel):
    """A quality item in dispute."""
    item: str = Field(..., description="What is disputed")
    reviewer_position: str = Field(..., description="Reviewer's stance")
    verifier_position: str = Field(..., description="Verifier's stance")
    critic_position: str = Field(..., description="Critic's stance")
    arbiter_resolution: str = Field(..., description="Arbiter's resolution")


class QualityVerdict(BaseModel):
    """
    Structured output from the Quality Arbiter (dispute resolution).

    Binding resolution of quality disputes after 2 failed debate rounds.
    """
    original_dispute: str = Field(..., description="What the dispute was about")
    disputed_items: List[DisputedItem] = Field(
        ...,
        description="Items in dispute"
    )
    debate_rounds_completed: int = Field(
        ...,
        ge=2,
        description="Number of debate rounds completed"
    )
    arbiter_analysis: str = Field(..., description="Arbiter's analysis")
    resolution: str = Field(..., description="Final resolution (binding)")
    required_actions: List[str] = Field(
        ...,
        description="Actions required by resolution"
    )
    overrides_reviewer: bool = Field(
        default=False,
        description="Whether this overrides the Reviewer"
    )


# =============================================================================
# Ethics & Safety Advisor Schemas
# =============================================================================

class IssueType(str, Enum):
    """Types of ethics/safety issues."""
    BIAS = "bias"
    PII = "pii"
    COMPLIANCE = "compliance"
    SAFETY = "safety"
    SECURITY = "security"
    FAIRNESS = "fairness"


class IssueSeverity(str, Enum):
    """Severity levels for ethics/safety issues."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class FlaggedIssue(BaseModel):
    """An ethics or safety issue."""
    issue_type: IssueType = Field(..., description="Type of issue")
    severity: IssueSeverity = Field(..., description="Severity level")
    description: str = Field(..., description="What the issue is")
    location: Optional[str] = Field(None, description="Where in the output")
    potential_harm: str = Field(..., description="Potential harm if not addressed")
    remediation: str = Field(..., description="How to fix this")
    blocks_output: bool = Field(
        ...,
        description="Whether this blocks the output"
    )


class EthicsReview(BaseModel):
    """
    Structured output from the Ethics & Safety Advisor.

    Review of output for bias, PII, compliance risks, and safety concerns.
    """
    output_summary: str = Field(..., description="Summary of the output being reviewed")
    verdict: str = Field(..., description="PASS or FAIL")
    flagged_issues: List[FlaggedIssue] = Field(
        ...,
        description="Issues found during review"
    )
    bias_analysis: str = Field(..., description="Analysis of potential bias")
    pii_scan_results: str = Field(..., description="Results of PII scanning")
    compliance_assessment: str = Field(
        ...,
        description="Compliance risk assessment"
    )
    safety_assessment: str = Field(..., description="Safety risk assessment")
    recommendations: List[str] = Field(
        ...,
        description="Recommendations for improvement"
    )
    can_proceed: bool = Field(
        ...,
        description="Whether output can proceed as-is"
    )
    required_remediations: List[str] = Field(
        default_factory=list,
        description="Required fixes before output can proceed"
    )
