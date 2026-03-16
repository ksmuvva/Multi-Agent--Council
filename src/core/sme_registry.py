"""
SME (Subject Matter Expert) Persona Registry

Defines all available SME personas with their configurations,
including trigger keywords, skills to load, and interaction modes.
"""

from dataclasses import dataclass
from typing import Any, List, Dict, Optional
from enum import Enum

from src.utils.logging import get_logger

logger = get_logger("sme_registry")


class InteractionMode(str, Enum):
    """SME interaction modes."""
    ADVISOR = "advisor"
    CO_EXECUTOR = "co_executor"
    DEBATER = "debater"


@dataclass
class SMEPersona:
    """
    Configuration for an SME (Subject Matter Expert) persona.

    Attributes:
        name: Display name of the persona
        persona_id: Unique identifier for the persona
        domain: Area of expertise
        trigger_keywords: Keywords that trigger this persona
        skill_files: SKILL.md files to load when this persona is spawned
        system_prompt_template: Path to the system prompt template
        interaction_modes: Allowed interaction modes for this persona
        default_model: Default model to use (usually sonnet for cost)
        description: Brief description of the persona
    """
    name: str
    persona_id: str
    domain: str
    trigger_keywords: List[str]
    skill_files: List[str]
    system_prompt_template: str
    interaction_modes: List[InteractionMode]
    default_model: str = "sonnet"
    description: str = ""


# =============================================================================
# SME Persona Registry
# =============================================================================

SME_REGISTRY: Dict[str, SMEPersona] = {
    "iam_architect": SMEPersona(
        name="IAM Architect",
        persona_id="iam_architect",
        domain="Identity and Access Management",
        trigger_keywords=[
            "sailpoint",
            "cyberark",
            "rbac",
            "identity",
            "azure ad",
            "okta",
            "authentication",
            "authorization",
            "privileged access",
            "iam",
            "identity governance",
            "lifecycle management"
        ],
        skill_files=["sailpoint-test-engineer", "azure-architect"],
        system_prompt_template="config/sme/iam_architect.md",
        interaction_modes=[InteractionMode.ADVISOR, InteractionMode.CO_EXECUTOR],
        description="Expert in identity governance, SailPoint, CyberArk, and Azure AD"
    ),

    "cloud_architect": SMEPersona(
        name="Cloud Architect",
        persona_id="cloud_architect",
        domain="Cloud Infrastructure Architecture",
        trigger_keywords=[
            "azure",
            "aws",
            "gcp",
            "cloud",
            "migration",
            "infrastructure",
            "serverless",
            "kubernetes",
            "docker",
            "container",
            "aks",
            "ecs",
            "lambda",
            "functions"
        ],
        skill_files=["azure-architect"],
        system_prompt_template="config/sme/cloud_architect.md",
        interaction_modes=[
            InteractionMode.ADVISOR,
            InteractionMode.CO_EXECUTOR,
            InteractionMode.DEBATER
        ],
        description="Expert in Azure, AWS, GCP cloud architecture and infrastructure"
    ),

    "security_analyst": SMEPersona(
        name="Security Analyst",
        persona_id="security_analyst",
        domain="Application & Infrastructure Security",
        trigger_keywords=[
            "threat model",
            "pentest",
            "security",
            "owasp",
            "vulnerability",
            "compliance",
            "secure coding",
            "encryption",
            "injection",
            "xss",
            "csrf",
            "security review"
        ],
        skill_files=["azure-architect"],
        system_prompt_template="config/sme/security_analyst.md",
        interaction_modes=[InteractionMode.ADVISOR, InteractionMode.DEBATER],
        description="Expert in threat modeling, OWASP, and secure development practices"
    ),

    "data_engineer": SMEPersona(
        name="Data Engineer",
        persona_id="data_engineer",
        domain="Data Engineering & Analytics",
        trigger_keywords=[
            "etl",
            "pipeline",
            "database",
            "schema",
            "sql",
            "data warehouse",
            "streaming",
            "batch",
            "data lake",
            "elt",
            "data modeling",
            "analytics"
        ],
        skill_files=["data-scientist"],
        system_prompt_template="config/sme/data_engineer.md",
        interaction_modes=[InteractionMode.ADVISOR, InteractionMode.CO_EXECUTOR],
        description="Expert in data pipelines, ETL/ELT, databases, and data modeling"
    ),

    "ai_ml_engineer": SMEPersona(
        name="AI/ML Engineer",
        persona_id="ai_ml_engineer",
        domain="Artificial Intelligence & Machine Learning",
        trigger_keywords=[
            "ml",
            "rag",
            "llm",
            "agent",
            "embeddings",
            "machine learning",
            "genai",
            "prompt engineering",
            "vector database",
            "fine-tuning",
            "inference",
            "model deployment"
        ],
        skill_files=["ai-engineer", "genai-system-design"],
        system_prompt_template="config/sme/ai_ml_engineer.md",
        interaction_modes=[
            InteractionMode.ADVISOR,
            InteractionMode.CO_EXECUTOR,
            InteractionMode.DEBATER
        ],
        description="Expert in GenAI, RAG systems, LLMs, and ML engineering"
    ),

    "test_engineer": SMEPersona(
        name="Test Engineer",
        persona_id="test_engineer",
        domain="Quality Assurance & Testing",
        trigger_keywords=[
            "test cases",
            "sit",
            "uat",
            "test plan",
            "automation",
            "pytest",
            "unit test",
            "integration test",
            "e2e test",
            "testing strategy",
            "quality assurance"
        ],
        skill_files=["sailpoint-test-engineer", "euroclear-test-cases"],
        system_prompt_template="config/sme/test_engineer.md",
        interaction_modes=[InteractionMode.ADVISOR, InteractionMode.CO_EXECUTOR],
        description="Expert in test strategy, test automation, and quality assurance"
    ),

    "business_analyst": SMEPersona(
        name="Business Analyst",
        persona_id="business_analyst",
        domain="Business Analysis & Requirements Engineering",
        trigger_keywords=[
            "requirements",
            "process",
            "gap analysis",
            "bpmn",
            "user stories",
            "acceptance criteria",
            "stakeholder",
            "business process",
            "workflow",
            "use cases"
        ],
        skill_files=["bpm-consultant", "vibe-requirements"],
        system_prompt_template="config/sme/business_analyst.md",
        interaction_modes=[InteractionMode.ADVISOR, InteractionMode.CO_EXECUTOR],
        description="Expert in requirements gathering, process analysis, and BPMN"
    ),

    "technical_writer": SMEPersona(
        name="Technical Writer",
        persona_id="technical_writer",
        domain="Technical Documentation & Communication",
        trigger_keywords=[
            "documentation",
            "tender",
            "report",
            "proposal",
            "user guide",
            "api docs",
            "readme",
            "technical writing",
            "documentation structure"
        ],
        skill_files=["human-like-writing", "tender-writing-expert"],
        system_prompt_template="config/sme/technical_writer.md",
        interaction_modes=[InteractionMode.ADVISOR, InteractionMode.CO_EXECUTOR],
        description="Expert in technical documentation, proposals, and communication"
    ),

    "devops_engineer": SMEPersona(
        name="DevOps Engineer",
        persona_id="devops_engineer",
        domain="DevOps & Infrastructure Automation",
        trigger_keywords=[
            "ci/cd",
            "docker",
            "kubernetes",
            "terraform",
            "deployment",
            "monitoring",
            "observability",
            "pipeline",
            "infrastructure as code",
            "continuous integration",
            "continuous deployment"
        ],
        skill_files=["azure-architect"],
        system_prompt_template="config/sme/devops_engineer.md",
        interaction_modes=[InteractionMode.ADVISOR, InteractionMode.CO_EXECUTOR],
        description="Expert in CI/CD, containers, Kubernetes, and infrastructure automation"
    ),

    "frontend_developer": SMEPersona(
        name="Frontend Developer",
        persona_id="frontend_developer",
        domain="Frontend Development & UI/UX",
        trigger_keywords=[
            "ui",
            "streamlit",
            "react",
            "dashboard",
            "web",
            "component",
            "responsive",
            "frontend",
            "user interface",
            "css",
            "javascript"
        ],
        skill_files=["frontend-design"],
        system_prompt_template="config/sme/frontend_developer.md",
        interaction_modes=[InteractionMode.ADVISOR, InteractionMode.CO_EXECUTOR],
        description="Expert in Streamlit, React, UI design, and frontend development"
    ),
}


# =============================================================================
# Registry Query Functions
# =============================================================================

def get_persona(persona_id: str) -> Optional[SMEPersona]:
    """
    Get an SME persona by ID.

    Args:
        persona_id: The unique identifier for the persona

    Returns:
        The SMEPersona if found, None otherwise
    """
    persona = SME_REGISTRY.get(persona_id)
    if persona:
        logger.debug("persona_found", persona_id=persona_id, domain=persona.domain)
    else:
        logger.warning("persona_not_found", persona_id=persona_id)
    return persona


def get_all_personas() -> Dict[str, SMEPersona]:
    """Get all registered SME personas."""
    return SME_REGISTRY.copy()


def find_personas_by_keywords(keywords: List[str]) -> List[SMEPersona]:
    """
    Find SME personas that match any of the given keywords.

    Args:
        keywords: List of keywords to search for

    Returns:
        List of matching SME personas, sorted by number of matches
    """
    logger.info("keyword_search_started", keywords=keywords, keyword_count=len(keywords))
    matches = []

    for persona_id, persona in SME_REGISTRY.items():
        match_count = 0
        matched_keywords = []

        for keyword in keywords:
            keyword_lower = keyword.lower()
            for trigger_keyword in persona.trigger_keywords:
                if keyword_lower in trigger_keyword.lower():
                    match_count += 1
                    matched_keywords.append(keyword)
                    break

        if match_count > 0:
            matches.append((persona, match_count, matched_keywords))
            logger.debug(
                "keyword_match",
                persona_id=persona.persona_id,
                match_count=match_count,
                matched_keywords=matched_keywords,
            )

    # Sort by match count (descending)
    matches.sort(key=lambda x: x[1], reverse=True)

    result = [persona for persona, _, _ in matches]
    logger.info(
        "keyword_search_completed",
        keywords=keywords,
        matches_found=len(result),
        matched_personas=[p.persona_id for p in result],
    )
    return result


def find_personas_by_domain(domain_keywords: List[str]) -> List[SMEPersona]:
    """
    Find SME personas by domain keywords.

    Args:
        domain_keywords: List of domain-related keywords

    Returns:
        List of matching SME personas
    """
    results = []

    for persona in SME_REGISTRY.values():
        domain_lower = persona.domain.lower()
        for keyword in domain_keywords:
            if keyword.lower() in domain_lower:
                results.append(persona)
                break

    return results


def get_persona_ids() -> List[str]:
    """Get all registered persona IDs."""
    return list(SME_REGISTRY.keys())


def validate_interaction_mode(
    persona_id: str,
    interaction_mode: InteractionMode
) -> bool:
    """
    Validate if an interaction mode is supported by a persona.

    Args:
        persona_id: The persona ID
        interaction_mode: The interaction mode to validate

    Returns:
        True if the mode is supported, False otherwise
    """
    persona = get_persona(persona_id)
    if persona is None:
        logger.warning("validate_mode_persona_missing", persona_id=persona_id)
        return False

    valid = interaction_mode in persona.interaction_modes
    if not valid:
        logger.warning(
            "interaction_mode_invalid",
            persona_id=persona_id,
            requested_mode=interaction_mode.value,
            allowed_modes=[m.value for m in persona.interaction_modes],
        )
    return valid


def get_persona_for_display(persona_id: str) -> Optional[Dict[str, Any]]:
    """
    Get persona information formatted for display in UI.

    Args:
        persona_id: The persona ID

    Returns:
        Dictionary with persona information for display, or None
    """
    persona = get_persona(persona_id)
    if persona is None:
        return None

    return {
        "id": persona.persona_id,
        "name": persona.name,
        "domain": persona.domain,
        "description": persona.description,
        "trigger_keywords": persona.trigger_keywords,
        "skill_files": persona.skill_files,
        "interaction_modes": [mode.value for mode in persona.interaction_modes],
        "default_model": persona.default_model
    }


# =============================================================================
# Registry Statistics
# =============================================================================

def get_registry_stats() -> Dict[str, Any]:
    """Get statistics about the SME registry."""
    return {
        "total_personas": len(SME_REGISTRY),
        "persona_ids": list(SME_REGISTRY.keys()),
        "domains": list(set(p.domain for p in SME_REGISTRY.values())),
        "total_trigger_keywords": sum(len(p.trigger_keywords) for p in SME_REGISTRY.values()),
        "available_skills": list(set(
            skill
            for persona in SME_REGISTRY.values()
            for skill in persona.skill_files
        ))
    }
