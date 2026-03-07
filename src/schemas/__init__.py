"""
Pydantic Schemas for Multi-Agent Reasoning System

This package contains all Pydantic v2 models used for structured
agent output across the system.
"""

# =============================================================================
# Analyst Schemas
# =============================================================================
from .analyst import (
    ModalityType,
    SeverityLevel,
    MissingInfo,
    SubTask,
    TaskIntelligenceReport,
)

# =============================================================================
# Planner Schemas
# =============================================================================
from .planner import (
    StepStatus,
    AgentAssignment,
    ExecutionStep,
    ParallelGroup,
    ExecutionPlan,
)

# =============================================================================
# Clarifier Schemas
# =============================================================================
from .clarifier import (
    QuestionPriority,
    ImpactAssessment,
    ClarificationQuestion,
    ClarificationRequest,
)

# =============================================================================
# Researcher Schemas
# =============================================================================
from .researcher import (
    ConfidenceLevel,
    SourceReliability,
    Source,
    Finding,
    Conflict,
    KnowledgeGap,
    EvidenceBrief,
)

# =============================================================================
# Code Reviewer Schemas
# =============================================================================
from .code_reviewer import (
    SeverityLevel as CodeSeverityLevel,
    ReviewCategory,
    CodeFinding,
    SecurityScan,
    PerformanceAnalysis,
    StyleCompliance,
    CodeReviewReport,
)

# =============================================================================
# Verifier Schemas
# =============================================================================
from .verifier import (
    VerificationStatus,
    FabricationRisk,
    Claim,
    ClaimBatch,
    VerificationReport,
)

# =============================================================================
# Critic Schemas
# =============================================================================
from .critic import (
    AttackVector,
    SeverityLevel as CriticSeverityLevel,
    Attack,
    LogicAttack,
    CompletenessAttack,
    QualityAttack,
    ContradictionScan,
    RedTeamArgument,
    CritiqueReport,
)

# =============================================================================
# Reviewer Schemas
# =============================================================================
from .reviewer import (
    Verdict,
    CheckItem,
    Revision,
    QualityGateResults,
    ArbitrationInput,
    ReviewVerdict,
)

# =============================================================================
# Council Schemas
# =============================================================================
from .council import (
    InteractionMode as CouncilInteractionMode,
    SMESelection,
    SMESelectionReport,
    QualityCriteria,
    QualityStandard,
    DisputedItem,
    QualityVerdict,
    IssueType,
    IssueSeverity,
    FlaggedIssue,
    EthicsReview,
)

# =============================================================================
# SME Schemas
# =============================================================================
from .sme import (
    SMEInteractionMode,
    AdvisorReport,
    CoExecutorSection,
    CoExecutorReport,
    DebatePosition,
    DebaterReport,
    SMEAdvisoryReport,
)

# =============================================================================
# Public API
# =============================================================================
__all__ = [
    # Analyst
    "ModalityType",
    "SeverityLevel",
    "MissingInfo",
    "SubTask",
    "TaskIntelligenceReport",
    # Planner
    "StepStatus",
    "AgentAssignment",
    "ExecutionStep",
    "ParallelGroup",
    "ExecutionPlan",
    # Clarifier
    "QuestionPriority",
    "ImpactAssessment",
    "ClarificationQuestion",
    "ClarificationRequest",
    # Researcher
    "ConfidenceLevel",
    "SourceReliability",
    "Source",
    "Finding",
    "Conflict",
    "KnowledgeGap",
    "EvidenceBrief",
    # Code Reviewer
    "CodeSeverityLevel",
    "ReviewCategory",
    "CodeFinding",
    "SecurityScan",
    "PerformanceAnalysis",
    "StyleCompliance",
    "CodeReviewReport",
    # Verifier
    "VerificationStatus",
    "FabricationRisk",
    "Claim",
    "ClaimBatch",
    "VerificationReport",
    # Critic
    "AttackVector",
    "CriticSeverityLevel",
    "Attack",
    "LogicAttack",
    "CompletenessAttack",
    "QualityAttack",
    "ContradictionScan",
    "RedTeamArgument",
    "CritiqueReport",
    # Reviewer
    "Verdict",
    "CheckItem",
    "Revision",
    "QualityGateResults",
    "ArbitrationInput",
    "ReviewVerdict",
    # Council
    "CouncilInteractionMode",
    "SMESelection",
    "SMESelectionReport",
    "QualityCriteria",
    "QualityStandard",
    "DisputedItem",
    "QualityVerdict",
    "IssueType",
    "IssueSeverity",
    "FlaggedIssue",
    "EthicsReview",
    # SME
    "SMEInteractionMode",
    "AdvisorReport",
    "CoExecutorSection",
    "CoExecutorReport",
    "DebatePosition",
    "DebaterReport",
    "SMEAdvisoryReport",
]
