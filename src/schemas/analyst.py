"""
Analyst Agent Schemas

Pydantic v2 models for the Task Analyst subagent.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class ModalityType(str, Enum):
    """Input/output modality types."""
    TEXT = "text"
    IMAGE = "image"
    CODE = "code"
    DOCUMENT = "document"
    DATA = "data"


class SeverityLevel(str, Enum):
    """Severity levels for missing information."""
    CRITICAL = "critical"
    IMPORTANT = "important"
    NICE_TO_HAVE = "nice_to_have"


class MissingInfo(BaseModel):
    """Missing information identified during task analysis."""
    requirement: str = Field(..., description="What information is missing")
    severity: SeverityLevel = Field(..., description="Impact severity")
    impact: str = Field(..., description="How this impacts the output")
    default_assumption: Optional[str] = Field(
        None,
        description="Sensible default if user doesn't provide"
    )


class SubTask(BaseModel):
    """A sub-task identified during task decomposition."""
    description: str = Field(..., description="What this sub-task accomplishes")
    dependencies: List[str] = Field(
        default_factory=list,
        description="Other sub-tasks this depends on"
    )
    estimated_complexity: str = Field(
        default="medium",
        description="Complexity: low/medium/high"
    )


class TaskIntelligenceReport(BaseModel):
    """
    Structured output from the Task Analyst subagent.

    Contains comprehensive analysis of the user's request including
    decomposition into sub-tasks, identification of missing information,
    and recommendations for how to proceed.
    """
    literal_request: str = Field(
        ...,
        description="The exact wording of the user's request"
    )
    inferred_intent: str = Field(
        ...,
        description="What the user actually wants to accomplish"
    )
    sub_tasks: List[SubTask] = Field(
        ...,
        description="Breakdown of the request into executable sub-tasks"
    )
    missing_info: List[MissingInfo] = Field(
        ...,
        description="Required/important information that's missing"
    )
    assumptions: List[str] = Field(
        ...,
        description="Assumptions made to proceed with the task"
    )
    modality: ModalityType = Field(
        ...,
        description="Detected input/output modality"
    )
    recommended_approach: str = Field(
        ...,
        description="Suggested approach for completing this task"
    )
    escalation_needed: bool = Field(
        default=False,
        description="Whether this requires higher-tier processing"
    )
    suggested_tier: int = Field(
        default=2,
        ge=1,
        le=4,
        description="Suggested complexity tier (1-4)"
    )
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confidence in this analysis (0-1)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "literal_request": "Write a REST API for user management",
                "inferred_intent": "Create a backend service for CRUD operations on user data",
                "sub_tasks": [
                    {
                        "description": "Design data models",
                        "dependencies": [],
                        "estimated_complexity": "medium"
                    },
                    {
                        "description": "Implement endpoints",
                        "dependencies": ["Design data models"],
                        "estimated_complexity": "high"
                    }
                ],
                "missing_info": [
                    {
                        "requirement": "Authentication method",
                        "severity": "important",
                        "impact": "Security design depends on this",
                        "default_assumption": "JWT-based authentication"
                    }
                ],
                "assumptions": [
                    "Using Python/FastAPI",
                    "PostgreSQL database",
                    "RESTful architecture"
                ],
                "modality": "code",
                "recommended_approach": "Start with data models, then implement endpoints",
                "escalation_needed": False,
                "suggested_tier": 2,
                "confidence": 0.9
            }
        }
