"""
Complexity Classification Module

Implements four-tier complexity classification for routing tasks
to appropriate agent configurations.
"""

from enum import IntEnum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from src.utils.logging import get_logger

_logger = get_logger("complexity")


class TierLevel(IntEnum):
    """
    Four-tier complexity levels for task classification.

    TIER 1 (Direct): Simple, straightforward tasks - 3 agents
    TIER 2 (Standard): Moderate complexity - 7 agents
    TIER 3 (Deep): Complex, domain-specific - 10-15 agents
    TIER 4 (Adversarial): High stakes, sensitive - 13-18 agents
    """
    DIRECT = 1
    STANDARD = 2
    DEEP = 3
    ADVERSARIAL = 4


class TierClassification(BaseModel):
    """
    Result of complexity classification.

    Contains the assigned tier, reasoning, and configuration
    for agent activation.
    """
    tier: TierLevel = Field(..., description="Assigned complexity tier (1-4)")
    reasoning: str = Field(..., description="Reasoning for this classification")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in this classification (0-1)"
    )
    estimated_agents: int = Field(..., description="Estimated number of agents")
    requires_council: bool = Field(..., description="Whether Council activation is needed")
    requires_smes: bool = Field(..., description="Whether SME personas are needed")
    suggested_sme_count: int = Field(
        default=0,
        ge=0,
        le=3,
        description="Suggested number of SMEs (0-3)"
    )
    escalation_risk: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Probability of mid-execution escalation (0-1)"
    )
    keywords_found: List[str] = Field(
        default_factory=list,
        description="Keywords that influenced this classification"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "tier": 3,
                "reasoning": "Complex task requiring domain expertise in security and cloud architecture",
                "confidence": 0.85,
                "estimated_agents": 12,
                "requires_council": True,
                "requires_smes": True,
                "suggested_sme_count": 2,
                "escalation_risk": 0.2,
                "keywords_found": ["security", "azure", "threat model"]
            }
        }


# =============================================================================
# Tier Configuration
# =============================================================================

TIER_CONFIG = {
    TierLevel.DIRECT: {
        "name": "Direct",
        "description": "Simple, straightforward tasks",
        "active_agents": ["Orchestrator", "Executor", "Formatter"],
        "agent_count": 3,
        "requires_council": False,
        "requires_smes": False,
        "max_sme_count": 0,
        "phases": ["Phase 5 (Solution Generation)", "Phase 8 (Formatting)"]
    },
    TierLevel.STANDARD: {
        "name": "Standard",
        "description": "Moderate complexity tasks",
        "active_agents": [
            "Orchestrator", "Analyst", "Planner", "Clarifier",
            "Executor", "Verifier", "Reviewer", "Formatter"
        ],
        "agent_count": 7,
        "requires_council": False,
        "requires_smes": False,
        "max_sme_count": 0,
        "phases": [
            "Phase 1 (Task Intelligence)",
            "Phase 3 (Planning)",
            "Phase 4 (Clarification if needed)",
            "Phase 5 (Solution Generation)",
            "Phase 6 (Verification)",
            "Phase 8 (Final Review + Formatting)"
        ]
    },
    TierLevel.DEEP: {
        "name": "Deep",
        "description": "Complex, domain-specific tasks",
        "active_agents": [
            "Orchestrator", "Analyst", "Planner", "Clarifier",
            "Researcher", "Executor", "Code Reviewer", "Formatter",
            "Verifier", "Critic", "Reviewer", "Memory Curator"
        ],
        "agent_count": 12,
        "requires_council": True,
        "council_agents": ["Domain Council Chair"],
        "requires_smes": True,
        "max_sme_count": 3,
        "phases": [
            "Phase 1 (Task Intelligence)",
            "Phase 2 (Council Consultation)",
            "Phase 3 (Planning)",
            "Phase 4 (Clarification if needed)",
            "Phase 5 (Solution Generation)",
            "Phase 6 (Verification with SME)",
            "Phase 8 (Final Review + Formatting)"
        ]
    },
    TierLevel.ADVERSARIAL: {
        "name": "Adversarial",
        "description": "High stakes, sensitive tasks",
        "active_agents": [
            "All operational agents"
        ],
        "agent_count": 18,
        "requires_council": True,
        "council_agents": [
            "Domain Council Chair",
            "Quality Arbiter",
            "Ethics & Safety Advisor"
        ],
        "requires_smes": True,
        "max_sme_count": 3,
        "phases": [
            "All 8 phases + Debate Protocol"
        ]
    },
}


# =============================================================================
# Complexity Classification
# =============================================================================

# Keywords and patterns that indicate higher complexity tiers
TIER_3_KEYWORDS = [
    # Domain-specific technical terms
    "threat model", "pentest", "compliance", "governance",
    "architecture", "design pattern", "system design",
    "data pipeline", "etl", "data warehouse",
    "machine learning", "ai", "rag", "llm",
    "security", "authentication", "authorization",
    "migration", "cloud native", "microservices",
    "test strategy", "test automation", "quality assurance",
    "requirements analysis", "gap analysis", "bpmn",
    "research", "investigate", "analyze multiple sources",
    "domain expert", "specialist knowledge"
]

TIER_4_KEYWORDS = [
    # High stakes and sensitive topics
    "security review", "security audit",
    "personal data", "pii", "gdpr", "hipaa",
    "financial", "banking", "payment",
    "medical", "health", "healthcare",
    "legal", "compliance", "regulatory",
    "government", "public sector",
    "safety", "risk assessment",
    "adversarial", "attack", "vulnerability",
    "critical", "production", "mission critical",
    "debate", "controversial", "multiple perspectives"
]

ESCALATION_KEYWORDS = [
    "complex", "complicated", "involved",
    "not sure", "uncertain", "may need",
    "depends on", "conditional",
    "multi-step", "multi-stage", "iterative"
]


def classify_complexity(
    user_prompt: str,
    analyst_report: Optional[Dict[str, Any]] = None
) -> TierClassification:
    """
    Classify the complexity of a user prompt into a tier level.

    Uses semantic analysis of the prompt content, considering:
    - Domain-specific keywords
    - Task complexity indicators
    - Stakes and sensitivity
    - Analyst recommendations (if available)

    Args:
        user_prompt: The user's request
        analyst_report: Optional TaskIntelligenceReport for additional context

    Returns:
        TierClassification with tier, reasoning, and configuration
    """
    prompt_lower = user_prompt.lower()
    keywords_found = []
    tier_score = 0
    reasoning_parts = []

    # Check for Tier 4 indicators (highest priority)
    tier_4_matches = [kw for kw in TIER_4_KEYWORDS if kw in prompt_lower]
    if tier_4_matches:
        tier_score = 4
        keywords_found.extend(tier_4_matches)
        reasoning_parts.append(
            f"Contains Tier 4 indicators: {', '.join(tier_4_matches[:3])}"
        )

    # Check for Tier 3 indicators
    tier_3_matches = [kw for kw in TIER_3_KEYWORDS if kw in prompt_lower]
    if tier_3_matches and tier_score < 3:
        tier_score = 3
        keywords_found.extend(tier_3_matches)
        reasoning_parts.append(
            f"Contains domain-specific keywords: {', '.join(tier_3_matches[:3])}"
        )

    # Check for escalation keywords
    escalation_matches = [kw for kw in ESCALATION_KEYWORDS if kw in prompt_lower]
    if escalation_matches:
        keywords_found.extend(escalation_matches)
        if tier_score < 2:
            tier_score = 2

    # Consider analyst recommendation if available
    suggested_tier = 2  # Default to Tier 2 if uncertain
    escalation_needed = False
    if analyst_report:
        suggested_tier = analyst_report.get("suggested_tier", 2)
        escalation_needed = analyst_report.get("escalation_needed", False)
        if suggested_tier > tier_score:
            tier_score = suggested_tier
            reasoning_parts.append(f"Analyst recommended Tier {suggested_tier}")

    # Ensure minimum tier of 1
    tier_score = max(1, tier_score)

    # Build reasoning
    if not reasoning_parts:
        if tier_score == 1:
            reasoning_parts.append("Simple, direct request with minimal complexity")
        else:
            reasoning_parts.append(f"Task complexity assessed at Tier {tier_score}")

    # Get configuration for this tier
    tier_enum = TierLevel(tier_score)
    config = TIER_CONFIG[tier_enum]

    # Calculate escalation risk
    escalation_risk = 0.1
    if escalation_matches:
        escalation_risk += 0.2
    if tier_score == 2:
        escalation_risk += 0.15

    classification = TierClassification(
        tier=tier_enum,
        reasoning=". ".join(reasoning_parts),
        confidence=0.8 if tier_score >= 3 else 0.7,
        estimated_agents=config["agent_count"],
        requires_council=config["requires_council"],
        requires_smes=config["requires_smes"],
        suggested_sme_count=config.get("max_sme_count", 0),
        escalation_risk=min(1.0, escalation_risk),
        keywords_found=keywords_found
    )
    _logger.info("complexity.classified",
                 tier=tier_enum,
                 confidence=classification.confidence,
                 estimated_agents=config["agent_count"],
                 requires_council=config["requires_council"],
                 requires_smes=config["requires_smes"],
                 keywords_found=keywords_found[:5],
                 reasoning=". ".join(reasoning_parts))
    return classification


def should_escalate(
    current_tier: TierLevel,
    subagent_feedback: Dict[str, Any]
) -> bool:
    """
    Determine if a task should be escalated to a higher tier.

    Args:
        current_tier: The current tier level
        subagent_feedback: Feedback from subagents indicating need for escalation

    Returns:
        True if escalation is recommended
    """
    # Check explicit escalation flag
    if subagent_feedback.get("escalation_needed", False):
        return True

    # Check for specific escalation indicators
    escalation_indicators = [
        "domain expertise required",
        "need specialist",
        "outside scope",
        "requires sme",
        "uncertain",
        "need verification"
    ]

    feedback_text = str(subagent_feedback).lower()
    for indicator in escalation_indicators:
        if indicator in feedback_text:
            return True

    return False


def get_escalated_tier(current_tier: TierLevel) -> TierLevel:
    """
    Get the next higher tier for escalation.

    Args:
        current_tier: The current tier level

    Returns:
        The next higher tier (capped at Tier 4)
    """
    return TierLevel(min(4, current_tier + 1))


def estimate_agent_count(tier: TierLevel, sme_count: int = 0) -> int:
    """
    Estimate the total number of agents for a tier.

    Args:
        tier: The tier level
        sme_count: Number of SME personas to activate

    Returns:
        Estimated total agent count
    """
    base_count = TIER_CONFIG[tier]["agent_count"]
    return base_count + sme_count


def get_active_agents(tier: TierLevel) -> List[str]:
    """
    Get the list of active agents for a tier.

    Args:
        tier: The tier level

    Returns:
        List of agent names
    """
    return TIER_CONFIG[tier]["active_agents"].copy()


def get_council_agents(tier: TierLevel) -> List[str]:
    """
    Get the list of Council agents for a tier.

    Args:
        tier: The tier level

    Returns:
        List of Council agent names (empty if Council not activated)
    """
    return TIER_CONFIG[tier].get("council_agents", [])
