"""
Code Reviewer Agent Schemas

Pydantic v2 models for the Code Reviewer subagent.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class SeverityLevel(str, Enum):
    """Severity levels for code review findings."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ReviewCategory(str, Enum):
    """Code review categories."""
    SECURITY = "security"
    PERFORMANCE = "performance"
    STYLE = "style"
    ERROR_HANDLING = "error_handling"
    TEST_COVERAGE = "test_coverage"
    DOCUMENTATION = "documentation"
    MAINTAINABILITY = "maintainability"


class CodeFinding(BaseModel):
    """A finding from code review."""
    severity: SeverityLevel = Field(..., description="Severity of the issue")
    category: ReviewCategory = Field(..., description="Category of the issue")
    file_path: str = Field(..., description="Path to the file")
    line_number: Optional[int] = Field(None, description="Line number if applicable")
    issue: str = Field(..., description="Description of the issue")
    recommendation: str = Field(..., description="How to fix it")
    code_snippet: Optional[str] = Field(
        None,
        description="Relevant code snippet"
    )
    fixed_snippet: Optional[str] = Field(
        None,
        description="Suggested fix"
    )
    references: List[str] = Field(
        default_factory=list,
        description="References for best practices"
    )


class SecurityScan(BaseModel):
    """Security-specific scan results."""
    vulnerabilities_found: int = Field(..., ge=0, description="Number of vulnerabilities")
    sql_injection_risk: bool = Field(default=False, description="SQL injection risks")
    xss_risk: bool = Field(default=False, description="XSS risks")
    auth_issues: List[str] = Field(
        default_factory=list,
        description="Authentication/authorization issues"
    )
    credential_exposure: List[str] = Field(
        default_factory=list,
        description="Potential credential exposures"
    )


class PerformanceAnalysis(BaseModel):
    """Performance-specific analysis."""
    n_plus_one_queries: List[str] = Field(
        default_factory=list,
        description="Potential N+1 query issues"
    )
    missing_indexes: List[str] = Field(
        default_factory=list,
        description="Suggested database indexes"
    )
    memory_leaks: List[str] = Field(
        default_factory=list,
        description="Potential memory leaks"
    )
    optimization_opportunities: List[str] = Field(
        default_factory=list,
        description="General optimization suggestions"
    )


class StyleCompliance(BaseModel):
    """Style and convention compliance."""
    pep8_compliant: bool = Field(default=True, description="PEP 8 compliance")
    naming_conventions: List[str] = Field(
        default_factory=list,
        description="Naming convention issues"
    )
    complexity_issues: List[str] = Field(
        default_factory=list,
        description="Complexity concerns"
    )
    consistency_issues: List[str] = Field(
        default_factory=list,
        description="Inconsistencies in code style"
    )


class CodeReviewReport(BaseModel):
    """
    Structured output from the Code Reviewer subagent.

    Contains comprehensive code review across five dimensions:
    security, performance, style, error handling, and test coverage.
    """
    overall_assessment: str = Field(..., description="Summary of code quality")
    pass_fail: bool = Field(..., description="Overall pass/fail")
    findings: List[CodeFinding] = Field(..., description="All findings ordered by severity")
    security_scan: SecurityScan = Field(..., description="Security-specific results")
    performance_analysis: PerformanceAnalysis = Field(
        ...,
        description="Performance-specific results"
    )
    style_compliance: StyleCompliance = Field(
        ...,
        description="Style and convention results"
    )
    error_handling_complete: bool = Field(
        ...,
        description="Whether error handling is complete"
    )
    test_coverage_assessment: str = Field(
        ...,
        description="Assessment of test coverage"
    )
    recommended_actions: List[str] = Field(
        ...,
        description="Prioritized recommended actions"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "overall_assessment": "Code is functional but has security concerns",
                "pass_fail": False,
                "findings": [
                    {
                        "severity": "critical",
                        "category": "security",
                        "file_path": "api/auth.py",
                        "line_number": 42,
                        "issue": "SQL injection vulnerability",
                        "recommendation": "Use parameterized queries"
                    }
                ],
                "security_scan": {
                    "vulnerabilities_found": 1,
                    "sql_injection_risk": True
                },
                "performance_analysis": {
                    "optimization_opportunities": ["Add caching for user lookups"]
                },
                "style_compliance": {
                    "pep8_compliant": True
                },
                "error_handling_complete": False,
                "test_coverage_assessment": "No tests found"
            }
        }
