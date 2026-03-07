"""
Clarifier Agent Schemas

Pydantic v2 models for the Clarifier subagent.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class QuestionPriority(str, Enum):
    """Priority levels for clarification questions."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ImpactAssessment(BaseModel):
    """Assessment of impact if a question is not answered."""
    quality_impact: str = Field(..., description="How output quality is affected")
    risk_level: str = Field(..., description="Risk level: low/medium/high")
    potential_revisions: List[str] = Field(
        ...,
        description="Revisions that might be needed"
    )


class ClarificationQuestion(BaseModel):
    """A single clarification question for the user."""
    question: str = Field(..., description="The question to ask the user")
    priority: QuestionPriority = Field(..., description="Priority of this question")
    reason: str = Field(..., description="Why this question matters")
    context: str = Field(..., description="Context for the user")
    default_answer: str = Field(
        ...,
        description="Default assumption if user doesn't answer"
    )
    impact_if_unanswered: ImpactAssessment = Field(
        ...,
        description="What happens if this isn't answered"
    )
    answer_options: Optional[List[str]] = Field(
        None,
        description="Multiple choice options if applicable"
    )


class ClarificationRequest(BaseModel):
    """
    Structured output from the Clarifier subagent.

    Contains prioritized questions for the user to clarify
    missing or ambiguous requirements.
    """
    total_questions: int = Field(..., ge=0, description="Number of questions")
    questions: List[ClarificationQuestion] = Field(
        ...,
        description="Questions ranked by priority"
    )
    recommended_workflow: str = Field(
        ...,
        description="Suggested workflow for getting answers"
    )
    can_proceed_with_defaults: bool = Field(
        ...,
        description="Whether task can proceed with default assumptions"
    )
    expected_quality_with_defaults: str = Field(
        ...,
        description="Expected output quality if using all defaults"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "total_questions": 2,
                "questions": [
                    {
                        "question": "Which database system should be used?",
                        "priority": "high",
                        "reason": "Affects schema design and query syntax",
                        "context": "The API will need persistent data storage",
                        "default_answer": "PostgreSQL",
                        "impact_if_unanswered": {
                            "quality_impact": "May need to rewrite queries later",
                            "risk_level": "medium",
                            "potential_revisions": ["Change ORM configuration", "Update SQL syntax"]
                        },
                        "answer_options": ["PostgreSQL", "MySQL", "MongoDB", "SQLite"]
                    }
                ],
                "recommended_workflow": "Present critical questions first, allow user to skip",
                "can_proceed_with_defaults": True,
                "expected_quality_with_defaults": "Good, but may require revisions"
            }
        }
