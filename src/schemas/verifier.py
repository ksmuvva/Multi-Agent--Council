"""
Verifier Agent Schemas

Pydantic v2 models for the Verifier (Hallucination Guard) subagent.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class VerificationStatus(str, Enum):
    """Status of a claim verification."""
    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    CONTRADICTED = "contradicted"
    FABRICATED = "fabricated"


class FabricationRisk(str, Enum):
    """Risk levels for fabrication."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Claim(BaseModel):
    """A factual claim to be verified."""
    claim_text: str = Field(..., description="The exact claim being verified")
    confidence: int = Field(
        ...,
        ge=1,
        le=10,
        description="Confidence score (1-10)"
    )
    fabrication_risk: FabricationRisk = Field(
        ...,
        description="Risk that this is fabricated"
    )
    source: Optional[str] = Field(None, description="Source that supports this claim")
    verification_method: str = Field(
        ...,
        description="How this was verified"
    )
    status: VerificationStatus = Field(..., description="Verification status")
    correction: Optional[str] = Field(
        None,
        description="Suggested correction if needed"
    )
    domain_verified: bool = Field(
        default=False,
        description="Whether an SME verified this"
    )
    sme_verifier: Optional[str] = Field(
        None,
        description="Which SME verified this (if applicable)"
    )


class ClaimBatch(BaseModel):
    """A batch of related claims."""
    topic: str = Field(..., description="What these claims relate to")
    claims: List[Claim] = Field(..., description="Claims in this batch")
    overall_reliability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall reliability (0-1)"
    )


class VerificationReport(BaseModel):
    """
    Structured output from the Verifier subagent.

    Contains verification of all factual claims in proposed output,
    with confidence scores, fabrication risk assessment, and
    corrections for any issues found.
    """
    total_claims_checked: int = Field(
        ...,
        ge=0,
        description="Total number of claims verified"
    )
    claims: List[Claim] = Field(..., description="All claims with verification status")
    verified_claims: int = Field(..., description="Number of verified claims")
    unverified_claims: int = Field(..., description="Number of unverified claims")
    contradicted_claims: int = Field(..., description="Number of contradicted claims")
    fabricated_claims: int = Field(..., description="Number of likely fabricated claims")
    overall_reliability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall reliability score (0-1)"
    )
    pass_threshold: float = Field(
        default=0.7,
        description="Minimum reliability to pass"
    )
    verdict: str = Field(..., description="PASS or FAIL based on threshold")
    flagged_claims: List[Claim] = Field(
        ...,
        description="Claims that need correction (confidence < 7 or risk > LOW)"
    )
    recommended_corrections: List[str] = Field(
        ...,
        description="Specific corrections needed"
    )
    verification_summary: str = Field(
        ...,
        description="Human-readable summary of verification"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "total_claims_checked": 10,
                "claims": [
                    {
                        "claim_text": "Python was released in 1991",
                        "confidence": 10,
                        "fabrication_risk": "low",
                        "source": "Multiple sources",
                        "verification_method": "Web search + documentation",
                        "status": "verified"
                    },
                    {
                        "claim_text": "Guido van Rossum is American",
                        "confidence": 2,
                        "fabrication_risk": "high",
                        "source": None,
                        "verification_method": "Could not verify",
                        "status": "unverified",
                        "correction": "Guido van Rossum is Dutch"
                    }
                ],
                "verified_claims": 9,
                "unverified_claims": 1,
                "contradicted_claims": 0,
                "fabricated_claims": 0,
                "overall_reliability": 0.85,
                "verdict": "PASS",
                "flagged_claims": [],
                "recommended_corrections": [],
                "verification_summary": "9 out of 10 claims verified. One unverified claim corrected."
            }
        }
