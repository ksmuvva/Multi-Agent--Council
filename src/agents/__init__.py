"""
Multi-Agent Reasoning System - Agent Package

Contains all agent implementations for the system.
"""

from .orchestrator import (
    OrchestratorAgent,
    AgentExecution,
    SessionState,
    create_orchestrator,
)

from .analyst import (
    AnalystAgent,
    create_analyst,
)

from .planner import (
    PlannerAgent,
    create_planner,
)

from .clarifier import (
    ClarifierAgent,
    create_clarifier,
)

from .researcher import (
    ResearcherAgent,
    create_researcher,
)

from .executor import (
    ExecutorAgent,
    create_executor,
)

from .code_reviewer import (
    CodeReviewerAgent,
    create_code_reviewer,
)

from .formatter import (
    FormatterAgent,
    create_formatter,
)

from .verifier import (
    VerifierAgent,
    create_verifier,
)

from .critic import (
    CriticAgent,
    create_critic,
)

from .reviewer import (
    ReviewerAgent,
    create_reviewer,
)

from .memory_curator import (
    MemoryCuratorAgent,
    create_memory_curator,
)

from .council import (
    CouncilChairAgent,
    QualityArbiterAgent,
    EthicsAdvisorAgent,
    create_council_chair,
    create_quality_arbiter,
    create_ethics_advisor,
)

from .sme_spawner import (
    SMESpawner,
    SpawnedSME,
    SpawnResult,
    create_sme_spawner,
)

__all__ = [
    # Orchestrator
    "OrchestratorAgent",
    "AgentExecution",
    "SessionState",
    "create_orchestrator",
    # Analysis & Planning
    "AnalystAgent",
    "create_analyst",
    "PlannerAgent",
    "create_planner",
    "ClarifierAgent",
    "create_clarifier",
    # Research & Execution
    "ResearcherAgent",
    "create_researcher",
    "ExecutorAgent",
    "create_executor",
    # Code Review & Format
    "CodeReviewerAgent",
    "create_code_reviewer",
    "FormatterAgent",
    "create_formatter",
    # Quality Assurance
    "VerifierAgent",
    "create_verifier",
    "CriticAgent",
    "create_critic",
    "ReviewerAgent",
    "create_reviewer",
    # Memory
    "MemoryCuratorAgent",
    "create_memory_curator",
    # Strategic Council
    "CouncilChairAgent",
    "QualityArbiterAgent",
    "EthicsAdvisorAgent",
    "create_council_chair",
    "create_quality_arbiter",
    "create_ethics_advisor",
    # SME Spawner
    "SMESpawner",
    "SpawnedSME",
    "SpawnResult",
    "create_sme_spawner",
]
