"""
Critic Agent Schemas

Pydantic v2 models for the Adversarial Critic subagent.
"""

from pydantic import BaseModel, ConfigDict, Field
from typing import List, Optional
from enum import Enum


class AttackVector(str, Enum):
    """Types of adversarial attacks."""
    LOGIC = "logic"
    COMPLETENESS = "completeness"
    QUALITY = "quality"
    CONTRADICTION = "contradiction"
    RED_TEAM = "red_team"


class SeverityLevel(str, Enum):
    """Severity levels for findings."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Attack(BaseModel):
    """An adversarial attack finding."""
    vector: AttackVector = Field(..., description="Type of attack")
    target: str = Field(..., description="What was attacked")
    finding: str = Field(..., description="What the attack revealed")
    severity: SeverityLevel = Field(..., description="Severity of the issue")
    description: str = Field(..., description="Full description of the issue")
    scenario: str = Field(..., description="Attack scenario - how this could be exploited")
    suggestion: str = Field(..., description="How to fix or mitigate")
    domain_specific: bool = Field(
        default=False,
        description="Whether this is a domain-specific attack from an SME"
    )
    sme_source: Optional[str] = Field(
        None,
        description="Which SME contributed this attack (if applicable)"
    )


class LogicAttack(BaseModel):
    """Logic attack results."""
    valid_arguments: List[str] = Field(
        default_factory=list,
        description="Arguments that are logically valid"
    )
    invalid_arguments: List[str] = Field(
        ...,
        description="Arguments with logical flaws"
    )
    fallacies_identified: List[str] = Field(
        ...,
        description="Logical fallacies found"
    )


class CompletenessAttack(BaseModel):
    """Completeness attack results."""
    covered: List[str] = Field(
        ...,
        description="What was covered"
    )
    missing: List[str] = Field(
        ...,
        description="What was missed"
    )
    assumptions: List[str] = Field(
        ...,
        description="Unstated assumptions that may be wrong"
    )


class QualityAttack(BaseModel):
    """Quality attack results."""
    strengths: List[str] = Field(
        default_factory=list,
        description="What's good"
    )
    weaknesses: List[str] = Field(
        ...,
        description="What's weak or could be better"
    )
    improvements: List[str] = Field(
        ...,
        description="Specific improvements"
    )


class ContradictionScan(BaseModel):
    """Contradiction scan results."""
    internal_contradictions: List[str] = Field(
        default_factory=list,
        description="Internal contradictions"
    )
    external_contradictions: List[str] = Field(
        ...,
        description="Contradictions with known facts"
    )
    inconsistencies: List[str] = Field(
        ...,
        description="Inconsistencies in the solution"
    )


class RedTeamArgument(BaseModel):
    """Red team argumentation results."""
    adversary_perspective: str = Field(
        ...,
        description="How an adversary would view this"
    )
    attack_surface: List[str] = Field(
        ...,
        description="Potential attack surfaces"
    )
    failure_modes: List[str] = Field(
        ...,
        description="How this could fail"
    )
    worst_case_scenarios: List[str] = Field(
        ...,
        description="Worst case outcomes"
    )


class CritiqueReport(BaseModel):
    """
    Structured output from the Critic subagent.

    Contains results from five adversarial attack vectors:
    logic, completeness, quality, contradiction, and red-team.
    """
    solution_summary: str = Field(..., description="Brief description of what was attacked")
    attacks: List[Attack] = Field(..., description="All attacks and findings")
    logic_attack: LogicAttack = Field(..., description="Logic attack results")
    completeness_attack: CompletenessAttack = Field(
        ...,
        description="Completeness attack results"
    )
    quality_attack: QualityAttack = Field(..., description="Quality attack results")
    contradiction_scan: ContradictionScan = Field(
        ...,
        description="Contradiction scan results"
    )
    red_team_argumentation: RedTeamArgument = Field(
        ...,
        description="Red team argumentation results"
    )
    overall_assessment: str = Field(..., description="Overall assessment of the solution")
    critical_issues: List[str] = Field(
        ...,
        description="Critical issues that must be addressed"
    )
    recommended_revisions: List[str] = Field(
        ...,
        description="Recommended revisions in priority order"
    )
    would_approve: bool = Field(..., description="Whether this passes the critique")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "solution_summary": "REST API for user management",
            "attacks": [
                {
                    "vector": "logic",
                    "finding": "Authentication flow has circular logic",
                    "severity": "critical",
                    "suggestion": "Simplify the auth flow"
                }
            ],
            "logic_attack": {
                "invalid_arguments": ["Auth token validation is circular"],
                "fallacies_identified": ["Begging the question in token refresh"]
            },
            "completeness_attack": {
                "missing": ["Rate limiting", "Input validation on all endpoints"]
            },
            "quality_attack": {
                "weaknesses": ["No error handling for database failures"],
                "improvements": ["Add comprehensive error handling"]
            },
            "contradiction_scan": {
                "inconsistencies": ["API returns 404 for auth failures"]
            },
            "red_team_argumentation": {
                "adversary_perspective": "No rate limiting = easy DDoS target",
                "attack_surface": ["All endpoints are unauthenticated"],
                "failure_modes": ["Database connection exhaustion"]
            },
            "overall_assessment": "Has potential but needs security hardening",
            "critical_issues": ["Missing authentication", "No rate limiting"],
            "recommended_revisions": ["Add authentication middleware", "Add rate limiting"],
            "would_approve": False
        }
    })
