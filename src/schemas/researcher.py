"""
Researcher Agent Schemas

Pydantic v2 models for the Researcher subagent.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class ConfidenceLevel(str, Enum):
    """Confidence levels for research findings."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class SourceReliability(str, Enum):
    """Reliability ratings for sources."""
    HIGH = "high"  # Official docs, academic sources
    MEDIUM = "medium"  # Reputable blogs, community resources
    LOW = "low"  # Forums, unverified sources
    UNKNOWN = "unknown"


class Source(BaseModel):
    """A research source."""
    url: str = Field(..., description="URL of the source")
    title: str = Field(..., description="Title of the source")
    reliability: SourceReliability = Field(
        ...,
        description="Reliability assessment"
    )
    access_date: str = Field(..., description="When this was accessed")
    excerpt: Optional[str] = Field(
        None,
        description="Relevant excerpt from the source"
    )


class Finding(BaseModel):
    """A research finding."""
    claim: str = Field(..., description="The finding or claim")
    confidence: ConfidenceLevel = Field(..., description="Confidence in this finding")
    sources: List[Source] = Field(..., description="Sources supporting this finding")
    context: str = Field(..., description="Additional context")
    caveats: List[str] = Field(
        default_factory=list,
        description="Any caveats or limitations"
    )


class Conflict(BaseModel):
    """A conflict between sources."""
    claim: str = Field(..., description="What the conflict is about")
    source_a: Source = Field(..., description="First source")
    source_b: Source = Field(..., description="Conflicting source")
    description: str = Field(..., description="Nature of the conflict")
    resolution_suggestion: str = Field(..., description="How to resolve")


class KnowledgeGap(BaseModel):
    """A gap in research - couldn't find information."""
    topic: str = Field(..., description="What we couldn't find")
    why_important: str = Field(..., description="Why this matters")
    searched_sources: List[str] = Field(
        ...,
        description="Where we looked"
    )
    suggested_approaches: List[str] = Field(
        ...,
        description="How to proceed despite the gap"
    )


class EvidenceBrief(BaseModel):
    """
    Structured output from the Researcher subagent.

    Contains gathered evidence from web sources with confidence
    levels, conflicting information, and knowledge gaps.
    """
    research_topic: str = Field(..., description="What was researched")
    summary: str = Field(..., description="Summary of findings")
    findings: List[Finding] = Field(..., description="All findings with confidence")
    conflicts: List[Conflict] = Field(
        default_factory=list,
        description="Conflicting information found"
    )
    gaps: List[KnowledgeGap] = Field(
        default_factory=list,
        description="Information that couldn't be found"
    )
    overall_confidence: ConfidenceLevel = Field(
        ...,
        description="Overall confidence in the research"
    )
    recommended_approach: str = Field(
        ...,
        description="Recommended approach based on findings"
    )
    additional_research_needed: bool = Field(
        default=False,
        description="Whether more research is needed"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "research_topic": "Best practices for REST API authentication",
                "summary": "JWT-based authentication is widely recommended",
                "findings": [
                    {
                        "claim": "JWT tokens are the industry standard",
                        "confidence": "high",
                        "sources": [
                            {
                                "url": "https://auth0.com/docs/secure/tokens",
                                "title": "Auth0 Documentation",
                                "reliability": "high"
                            }
                        ],
                        "context": "Multiple authoritative sources agree"
                    }
                ],
                "conflicts": [],
                "gaps": [],
                "overall_confidence": "high",
                "recommended_approach": "Implement JWT with refresh token rotation"
            }
        }
