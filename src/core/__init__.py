"""
Core Infrastructure for Multi-Agent Reasoning System

This package contains the core infrastructure modules for
complexity classification, pipeline orchestration, verdict matrix,
and debate protocol.
"""

# =============================================================================
# Complexity Classification
# =============================================================================
from .complexity import (
    TierLevel,
    TierClassification,
    TIER_CONFIG,
    classify_complexity,
    should_escalate,
    get_escalated_tier,
    estimate_agent_count,
    get_active_agents,
    get_council_agents,
)

# =============================================================================
# Verdict Matrix
# =============================================================================
from .verdict import (
    Verdict,
    MatrixAction,
    MatrixOutcome,
    VERDICT_MATRIX,
    evaluate_verdict_matrix,
    get_phase_for_action,
    should_trigger_debate,
    DebateConfig,
    get_required_agents_for_phase,
    calculate_phase_cost_estimate,
)

# =============================================================================
# Debate Protocol
# =============================================================================
from .debate import (
    ConsensusLevel,
    DebateRound,
    DebateOutcome,
    DebateProtocol,
    trigger_debate,
    get_debate_participants,
)

# =============================================================================
# Pipeline Orchestration
# =============================================================================
from .pipeline import (
    Phase,
    PhaseStatus,
    AgentResult,
    PhaseResult,
    PipelineState,
    ExecutionPipeline,
    PipelineBuilder,
    create_execution_context,
    estimate_pipeline_duration,
)

# =============================================================================
# SME Registry
# =============================================================================
from .sme_registry import (
    InteractionMode as SMEInteractionMode,
    SMEPersona,
    SME_REGISTRY,
    get_persona,
    get_all_personas,
    find_personas_by_keywords,
    find_personas_by_domain,
    get_persona_ids,
    validate_interaction_mode,
    get_persona_for_display,
    get_registry_stats,
)

# =============================================================================
# Ensemble Patterns
# =============================================================================
from .ensemble import (
    EnsembleType,
    AgentRole,
    AgentAssignment,
    EnsembleConfig,
    EnsembleResult,
    EnsemblePattern,
    ArchitectureReviewBoard,
    CodeSprint,
    ResearchCouncil,
    DocumentAssembly,
    RequirementsWorkshop,
    ENSEMBLE_REGISTRY,
    get_ensemble,
    get_all_ensembles,
    suggest_ensemble,
    execute_ensemble,
    create_architecture_review,
    create_code_sprint,
    create_research_council,
    create_document_assembly,
    create_requirements_workshop,
)

__all__ = [
    # Complexity
    "TierLevel",
    "TierClassification",
    "TIER_CONFIG",
    "classify_complexity",
    "should_escalate",
    "get_escalated_tier",
    "estimate_agent_count",
    "get_active_agents",
    "get_council_agents",
    # Verdict
    "Verdict",
    "MatrixAction",
    "MatrixOutcome",
    "VERDICT_MATRIX",
    "evaluate_verdict_matrix",
    "get_phase_for_action",
    "should_trigger_debate",
    "DebateConfig",
    "get_required_agents_for_phase",
    "calculate_phase_cost_estimate",
    # Debate
    "ConsensusLevel",
    "DebateRound",
    "DebateOutcome",
    "DebateProtocol",
    "trigger_debate",
    "get_debate_participants",
    # Pipeline
    "Phase",
    "PhaseStatus",
    "AgentResult",
    "PhaseResult",
    "PipelineState",
    "ExecutionPipeline",
    "PipelineBuilder",
    "create_execution_context",
    "estimate_pipeline_duration",
    # SME Registry
    "SMEInteractionMode",
    "SMEPersona",
    "SME_REGISTRY",
    "get_persona",
    "get_all_personas",
    "find_personas_by_keywords",
    "find_personas_by_domain",
    "get_persona_ids",
    "validate_interaction_mode",
    "get_persona_for_display",
    "get_registry_stats",
    # Ensemble Patterns
    "EnsembleType",
    "AgentRole",
    "AgentAssignment",
    "EnsembleConfig",
    "EnsembleResult",
    "EnsemblePattern",
    "ArchitectureReviewBoard",
    "CodeSprint",
    "ResearchCouncil",
    "DocumentAssembly",
    "RequirementsWorkshop",
    "ENSEMBLE_REGISTRY",
    "get_ensemble",
    "get_all_ensembles",
    "suggest_ensemble",
    "execute_ensemble",
    "create_architecture_review",
    "create_code_sprint",
    "create_research_council",
    "create_document_assembly",
    "create_requirements_workshop",
]
