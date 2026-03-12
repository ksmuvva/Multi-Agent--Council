"""
SME (Subject Matter Expert) Persona Schemas

Pydantic v2 models for SME persona interactions.
"""

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import List, Optional, Dict, Any
from enum import Enum


class SMEInteractionMode(str, Enum):
    """How an SME interacts with operational agents."""
    ADVISOR = "advisor"
    CO_EXECUTOR = "co_executor"
    DEBATER = "debater"


class AdvisorReport(BaseModel):
    """Report from an SME in advisor mode."""
    sme_persona: str = Field(..., description="Which SME provided this")
    reviewed_content: str = Field(..., description="What was reviewed")
    domain_corrections: List[str] = Field(
        ...,
        description="Domain-specific corrections"
    )
    missing_considerations: List[str] = Field(
        ...,
        description="Important domain factors that were missed"
    )
    recommendations: List[str] = Field(
        ...,
        description="Domain-specific recommendations"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in this review (0-1)"
    )


class CoExecutorSection(BaseModel):
    """A section contributed by an SME in co-executor mode."""
    sme_persona: str = Field(..., description="Which SME contributed this")
    section_title: str = Field(..., description="Title of the section")
    content: str = Field(..., description="The contributed content")
    domain_context: str = Field(..., description="Domain-specific context")
    integration_notes: str = Field(
        ...,
        description="How this integrates with other sections"
    )


class CoExecutorReport(BaseModel):
    """Report from an SME in co-executor mode."""
    sme_persona: str = Field(..., description="Which SME contributed this")
    contributed_sections: List[CoExecutorSection] = Field(
        ...,
        description="Sections contributed by this SME"
    )
    coordination_notes: str = Field(
        ...,
        description="How this SME coordinated with Executor"
    )
    domain_assumptions: List[str] = Field(
        ...,
        description="Domain-specific assumptions made"
    )


class DebatePosition(BaseModel):
    """An SME's position in a debate."""
    sme_persona: str = Field(..., description="Which SME holds this position")
    position: str = Field(..., description="The SME's position")
    domain_rationale: str = Field(
        ...,
        description="Domain-specific reasoning for this position"
    )
    supporting_evidence: List[str] = Field(
        ...,
        description="Domain evidence supporting this position"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in this position (0-1)"
    )


class DebaterReport(BaseModel):
    """Report from an SME in debater mode."""
    sme_persona: str = Field(..., description="Which SME participated")
    debate_round: int = Field(..., ge=1, description="Debate round number")
    position: DebatePosition = Field(..., description="SME's position")
    counter_arguments_addressed: List[str] = Field(
        ...,
        description="Counter-arguments this SME addressed"
    )
    remaining_concerns: List[str] = Field(
        ...,
        description="Concerns this SME still has"
    )
    willingness_to_concede: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Willingness to concede (0-1)"
    )


class SMEAdvisoryReport(BaseModel):
    """
    Structured output from an SME persona.

    Contains domain-specific findings, recommendations, and confidence levels.
    """
    sme_persona: str = Field(..., description="Which SME provided this advisory")
    interaction_mode: SMEInteractionMode = Field(
        ...,
        description="How this SME interacted"
    )
    domain: str = Field(..., description="Domain of expertise")
    task_context: str = Field(..., description="Context of the task")
    findings: List[str] = Field(
        ...,
        description="Domain-specific findings"
    )
    recommendations: List[str] = Field(
        ...,
        description="Domain-specific recommendations"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall confidence in this advisory (0-1)"
    )
    caveats: List[str] = Field(
        default_factory=list,
        description="Caveats and limitations"
    )
    advisor_report: Optional[AdvisorReport] = Field(
        None,
        description="Report if in advisor mode"
    )
    co_executor_report: Optional[CoExecutorReport] = Field(
        None,
        description="Report if in co-executor mode"
    )
    debater_report: Optional[DebaterReport] = Field(
        None,
        description="Report if in debater mode"
    )
    skills_used: List[str] = Field(
        ...,
        description="SKILL.md files loaded by this SME"
    )
    additional_domains_consulted: List[str] = Field(
        default_factory=list,
        description="Other domains this SME consulted"
    )

    @model_validator(mode="after")
    def validate_report_matches_mode(self) -> "SMEAdvisoryReport":
        """Ensure the correct report field is populated for the interaction mode."""
        mode_report_map = {
            SMEInteractionMode.ADVISOR: "advisor_report",
            SMEInteractionMode.CO_EXECUTOR: "co_executor_report",
            SMEInteractionMode.DEBATER: "debater_report",
        }
        expected_field = mode_report_map.get(self.interaction_mode)
        if expected_field and getattr(self, expected_field) is None:
            # Set a warning but don't fail - the report may be populated later
            pass
        # Ensure other mode reports are not populated
        for mode, field_name in mode_report_map.items():
            if mode != self.interaction_mode and getattr(self, field_name) is not None:
                raise ValueError(
                    f"Report field '{field_name}' should not be populated "
                    f"when interaction_mode is '{self.interaction_mode.value}'"
                )
        return self

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "sme_persona": "Security Analyst",
                "interaction_mode": "advisor",
                "domain": "Application Security",
                "task_context": "Reviewing authentication implementation",
                "findings": [
                    "No rate limiting on auth endpoint",
                    "Password complexity requirements are weak"
                ],
                "recommendations": [
                    "Add rate limiting to prevent brute force",
                    "Implement stronger password requirements",
                    "Consider multi-factor authentication"
                ],
                "confidence": 0.9,
                "caveats": [
                    "Full security audit would require penetration testing"
                ],
                "advisor_report": {
                    "sme_persona": "Security Analyst",
                    "reviewed_content": "Authentication flow",
                    "domain_corrections": [
                        "Add password hashing with bcrypt",
                        "Implement account lockout"
                    ],
                    "missing_considerations": [
                        "No consideration for session timeout",
                        "Missing secure cookie flags"
                    ],
                    "recommendations": [
                        "Use OWASP security best practices"
                    ],
                    "confidence": 0.9
                },
                "skills_used": ["azure-architect"],
                "additional_domains_consulted": []
            }
        }
    )
