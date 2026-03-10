"""
Planner Agent Schemas

Pydantic v2 models for the Planner subagent.
"""

from pydantic import BaseModel, ConfigDict, Field
from typing import List, Optional, Set
from enum import Enum


class StepStatus(str, Enum):
    """Status of a planning step."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    SKIPPED = "skipped"


class AgentAssignment(BaseModel):
    """Assignment of a step to an agent."""
    agent_name: str = Field(..., description="Name of the agent")
    role: str = Field(..., description="Role this agent plays in this step")
    reason: str = Field(..., description="Why this agent is assigned")


class ExecutionStep(BaseModel):
    """A single step in the execution plan."""
    step_number: int = Field(..., ge=1, description="Sequential step number")
    description: str = Field(..., description="What this step accomplishes")
    agent_assignments: List[AgentAssignment] = Field(
        ...,
        description="Agents assigned to this step"
    )
    dependencies: List[int] = Field(
        default_factory=list,
        description="Step numbers that must complete first"
    )
    can_parallelize: bool = Field(
        default=False,
        description="Can this step run in parallel with other steps?"
    )
    parallel_group_id: Optional[str] = Field(
        None,
        description="If parallel, ID of the parallel group"
    )
    estimated_complexity: str = Field(
        default="medium",
        description="Complexity: low/medium/high"
    )
    expected_outputs: List[str] = Field(
        default_factory=list,
        description="What this step should produce"
    )
    status: StepStatus = Field(
        default=StepStatus.PENDING,
        description="Current status of this step"
    )


class ParallelGroup(BaseModel):
    """A group of steps that can execute in parallel."""
    group_id: str = Field(..., description="Unique identifier for this group")
    steps: List[int] = Field(..., description="Step numbers in this group")
    description: str = Field(..., description="What this group accomplishes")


class ExecutionPlan(BaseModel):
    """
    Structured output from the Planner subagent.

    Contains a sequenced execution plan with agent assignments,
    dependencies, and parallel execution opportunities.
    """
    task_summary: str = Field(
        ...,
        description="Brief summary of what this plan accomplishes"
    )
    total_steps: int = Field(..., ge=1, description="Total number of steps")
    steps: List[ExecutionStep] = Field(
        ...,
        description="All steps in execution order"
    )
    parallel_groups: List[ParallelGroup] = Field(
        default_factory=list,
        description="Groups of steps that can run in parallel"
    )
    critical_path: List[int] = Field(
        ...,
        description="Step numbers on the critical path (longest path)"
    )
    estimated_duration_minutes: Optional[int] = Field(
        None,
        description="Estimated total execution time"
    )
    required_sme_personas: List[str] = Field(
        default_factory=list,
        description="SME personas that should participate"
    )
    risk_factors: List[str] = Field(
        default_factory=list,
        description="Potential risks or blockers"
    )
    contingency_plans: List[str] = Field(
        default_factory=list,
        description="Alternative approaches if primary plan fails"
    )

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "task_summary": "Create a REST API for user management",
            "total_steps": 4,
            "steps": [
                {
                    "step_number": 1,
                    "description": "Design data models",
                    "agent_assignments": [
                        {
                            "agent_name": "Data Architect SME",
                            "role": "Schema design",
                            "reason": "Domain expertise in data modeling"
                        }
                    ],
                    "dependencies": [],
                    "can_parallelize": False,
                    "estimated_complexity": "medium"
                },
                {
                    "step_number": 2,
                    "description": "Implement API endpoints",
                    "agent_assignments": [
                        {
                            "agent_name": "Executor",
                            "role": "Code generation",
                            "reason": "Create the implementation"
                        }
                    ],
                    "dependencies": [1],
                    "can_parallelize": False,
                    "estimated_complexity": "high"
                }
            ],
            "parallel_groups": [],
            "critical_path": [1, 2],
            "estimated_duration_minutes": 15
        }
    })
