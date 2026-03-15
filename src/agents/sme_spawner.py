"""
SME Spawner - Dynamic Persona Spawning System

Creates and manages SME personas on-demand based on Council Chair selection.
Implements three interaction modes: Advisor, Co-executor, Debater.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum

from src.utils.logging import get_agent_logger, AgentLogContext
from src.utils.events import emit_agent_started, emit_agent_completed, emit_error

from src.schemas.sme import (
    SMEAdvisoryReport,
    SMEInteractionMode,
    AdvisorReport,
    CoExecutorReport,
    CoExecutorSection,
    DebaterReport,
    DebatePosition,
)

from src.schemas.council import (
    SMESelection,
    InteractionMode as CouncilInteractionMode,
)

from src.core.sme_registry import (
    SME_REGISTRY,
    get_persona,
    get_all_personas,
    find_personas_by_keywords,
    validate_interaction_mode,
    get_persona_for_display,
)


@dataclass
class SpawnedSME:
    """A spawned SME instance."""
    persona_id: str
    persona_name: str
    domain: str
    interaction_mode: SMEInteractionMode
    system_prompt: str
    skills_loaded: List[str]
    spawn_context: Dict[str, Any]


@dataclass
class SpawnResult:
    """Result of spawning SMEs."""
    spawned_smes: List[SpawnedSME]
    total_spawned: int
    interaction_modes_used: Set[SMEInteractionMode]
    spawn_metadata: Dict[str, Any]


class SMESpawner:
    """
    The SME Spawner dynamically creates SME personas.

    Key responsibilities:
    - Spawn SME personas based on Council Chair selection
    - Load appropriate SKILL.md files for each SME
    - Implement interaction modes (advisor/co-executor/debater)
    - Execute SME interactions in their assigned mode
    - Aggregate SME outputs into reports
    """

    def __init__(
        self,
        skills_dir: str = ".claude/skills",
        sme_templates_dir: str = "config/sme",
        model: str = "claude-3-5-sonnet-20241022",
    ):
        """
        Initialize the SME Spawner.

        Args:
            skills_dir: Directory containing SKILL.md files
            sme_templates_dir: Directory containing SME persona templates
            model: Default model for SME interactions
        """
        self.skills_dir = Path(skills_dir)
        self.sme_templates_dir = Path(sme_templates_dir)
        self.model = model
        self.logger = get_agent_logger("sme_spawner")
        self.logger.info("SMESpawner initialized", model=model)

        # Ensure directories exist
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self.sme_templates_dir.mkdir(parents=True, exist_ok=True)

    def spawn_from_selection(
        self,
        selections: List[SMESelection],
        task_context: str,
        execution_phase: str = "execution",
    ) -> SpawnResult:
        """
        Spawn SMEs based on Council Chair selection.

        Args:
            selections: SME selections from Council Chair
            task_context: Context of the current task
            execution_phase: Current execution phase

        Returns:
            SpawnResult with spawned SMEs
        """
        self.logger.info(
            "SME spawn started",
            selections_count=len(selections),
            task_context_preview=task_context[:100],
            execution_phase=execution_phase,
        )
        emit_agent_started("sme_spawner", phase="spawn")

        spawned_smes = []
        interaction_modes = set()

        for selection in selections:
            # Get persona from registry
            persona = get_persona(selection.persona_domain.lower().replace(" ", "_").replace("&", "and"))

            if persona is None:
                # Try persona_id directly
                persona = get_persona(selection.persona_name.lower().replace(" ", "_"))

            if persona is None:
                # Try finding by keywords
                matching = find_personas_by_keywords([selection.persona_domain])
                if matching:
                    persona = matching[0]

            if persona is None:
                self.logger.warning(
                    "Persona not found, skipping",
                    persona_name=selection.persona_name,
                    persona_domain=selection.persona_domain,
                )
                continue  # Skip if persona not found

            self.logger.debug(
                "Persona selected for spawn",
                persona_id=persona.persona_id,
                persona_name=persona.name,
                domain=persona.domain,
            )

            # Convert interaction mode
            interaction_mode = self._convert_interaction_mode(selection.interaction_mode)

            # Validate interaction mode
            if not validate_interaction_mode(persona.persona_id, interaction_mode):
                interaction_mode = SMEInteractionMode.ADVISOR  # Fallback

            # Load system prompt
            system_prompt = self._load_system_prompt(persona)

            # Load skills
            skills_loaded = self._load_skills(persona, selection.skills_to_load)
            self.logger.debug(
                "Skills loaded for SME",
                persona_id=persona.persona_id,
                skills_loaded=skills_loaded,
            )

            # Create spawned SME
            spawned = SpawnedSME(
                persona_id=persona.persona_id,
                persona_name=persona.name,
                domain=persona.domain,
                interaction_mode=interaction_mode,
                system_prompt=system_prompt,
                skills_loaded=skills_loaded,
                spawn_context={
                    "activation_phase": selection.activation_phase,
                    "reasoning": selection.reasoning,
                    "task_context": task_context,
                    "execution_phase": execution_phase,
                },
            )

            spawned_smes.append(spawned)
            interaction_modes.add(interaction_mode)
            self.logger.info(
                "SME spawned successfully",
                persona_id=persona.persona_id,
                persona_name=persona.name,
                interaction_mode=interaction_mode.value,
            )

        self.logger.info(
            "SME spawn completed",
            total_spawned=len(spawned_smes),
            interaction_modes=[m.value for m in interaction_modes],
        )
        emit_agent_completed("sme_spawner", output_summary=f"Spawned {len(spawned_smes)} SMEs")

        return SpawnResult(
            spawned_smes=spawned_smes,
            total_spawned=len(spawned_smes),
            interaction_modes_used=interaction_modes,
            spawn_metadata={
                "task_context": task_context,
                "execution_phase": execution_phase,
                "timestamp": self._get_timestamp(),
            },
        )

    def execute_sme_interaction(
        self,
        spawned_sme: SpawnedSME,
        content: str,
        interaction_type: str = "review",
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> SMEAdvisoryReport:
        """
        Execute an SME interaction in the assigned mode.

        Args:
            spawned_sme: The spawned SME to execute
            content: Content to review/process
            interaction_type: Type of interaction (review/execute/debate)
            additional_context: Additional context for the interaction

        Returns:
            SMEAdvisoryReport with mode-specific report
        """
        # Determine interaction mode
        mode = spawned_sme.interaction_mode
        self.logger.info(
            "SME interaction started",
            persona_name=spawned_sme.persona_name,
            interaction_mode=mode.value,
            interaction_type=interaction_type,
        )

        if mode == SMEInteractionMode.ADVISOR:
            return self._execute_advisor_mode(spawned_sme, content, additional_context)
        elif mode == SMEInteractionMode.CO_EXECUTOR:
            return self._execute_co_executor_mode(spawned_sme, content, additional_context)
        elif mode == SMEInteractionMode.DEBATER:
            return self._execute_debater_mode(spawned_sme, content, additional_context)
        else:
            # Default to advisor
            return self._execute_advisor_mode(spawned_sme, content, additional_context)

    # ========================================================================
    # Interaction Mode Implementations
    # ========================================================================

    def _execute_advisor_mode(
        self,
        spawned_sme: SpawnedSME,
        content: str,
        additional_context: Optional[Dict[str, Any]],
    ) -> SMEAdvisoryReport:
        """
        Execute SME in advisor mode - provides domain guidance.

        Advisor SMEs review content and provide recommendations
        without directly modifying output.
        """
        # Simulate domain analysis
        findings = self._analyze_domain_findings(spawned_sme, content)

        # Identify domain corrections
        domain_corrections = self._identify_domain_corrections(spawned_sme, content, findings)

        # Find missing considerations
        missing_considerations = self._identify_missing_considerations(spawned_sme, content)

        # Generate recommendations
        recommendations = self._generate_recommendations(spawned_sme, findings, domain_corrections)

        # Calculate confidence
        confidence = self._calculate_confidence(spawned_sme, findings)

        # Build advisor report
        advisor_report = AdvisorReport(
            sme_persona=spawned_sme.persona_name,
            reviewed_content=content[:100],
            domain_corrections=domain_corrections,
            missing_considerations=missing_considerations,
            recommendations=recommendations,
            confidence=confidence,
        )

        return SMEAdvisoryReport(
            sme_persona=spawned_sme.persona_name,
            interaction_mode=SMEInteractionMode.ADVISOR,
            domain=spawned_sme.domain,
            task_context=spawned_sme.spawn_context.get("task_context", ""),
            findings=findings,
            recommendations=recommendations,
            confidence=confidence,
            caveats=self._generate_caveats(spawned_sme),
            advisor_report=advisor_report,
            skills_used=spawned_sme.skills_loaded,
            additional_domains_consulted=[],
        )

    def _execute_co_executor_mode(
        self,
        spawned_sme: SpawnedSME,
        content: str,
        additional_context: Optional[Dict[str, Any]],
    ) -> SMEAdvisoryReport:
        """
        Execute SME in co-executor mode - contributes content directly.

        Co-executor SMEs work alongside Executor to produce
        domain-specific sections of the output.
        """
        # Determine what sections to contribute
        sections_to_contribute = self._determine_sections(spawned_sme, content)

        # Generate contributed sections
        contributed_sections = []
        for section_def in sections_to_contribute:
            section = self._generate_section(
                spawned_sme, section_def, content, additional_context
            )
            contributed_sections.append(section)

        # Coordination notes
        coordination_notes = self._generate_coordination_notes(
            spawned_sme, contributed_sections, content
        )

        # Domain assumptions
        domain_assumptions = self._identify_domain_assumptions(spawned_sme, content)

        # Build co-executor report
        co_executor_report = CoExecutorReport(
            sme_persona=spawned_sme.persona_name,
            contributed_sections=contributed_sections,
            coordination_notes=coordination_notes,
            domain_assumptions=domain_assumptions,
        )

        # Generate findings and recommendations from contributed content
        findings = [f"Contributed {len(contributed_sections)} section(s)"]
        recommendations = [coordination_notes]

        return SMEAdvisoryReport(
            sme_persona=spawned_sme.persona_name,
            interaction_mode=SMEInteractionMode.CO_EXECUTOR,
            domain=spawned_sme.domain,
            task_context=spawned_sme.spawn_context.get("task_context", ""),
            findings=findings,
            recommendations=recommendations,
            confidence=0.85,  # High confidence in own contributions
            caveats=[],
            co_executor_report=co_executor_report,
            skills_used=spawned_sme.skills_loaded,
            additional_domains_consulted=[],
        )

    def _execute_debater_mode(
        self,
        spawned_sme: SpawnedSME,
        content: str,
        additional_context: Optional[Dict[str, Any]],
    ) -> SMEAdvisoryReport:
        """
        Execute SME in debater mode - participates in adversarial debate.

        Debater SMEs argue their position, address counter-arguments,
        and indicate willingness to concede.
        """
        # Get debate round from context
        debate_round = additional_context.get("debate_round", 1) if additional_context else 1

        # Determine position based on domain perspective
        position = self._determine_debate_position(spawned_sme, content, additional_context)

        # Generate domain rationale
        domain_rationale = self._generate_domain_rationale(spawned_sme, position)

        # Gather supporting evidence
        supporting_evidence = self._gather_supporting_evidence(spawned_sme, position)

        # Calculate confidence in position
        confidence = self._calculate_position_confidence(spawned_sme, position)

        # Create debate position
        debate_position = DebatePosition(
            sme_persona=spawned_sme.persona_name,
            position=position,
            domain_rationale=domain_rationale,
            supporting_evidence=supporting_evidence,
            confidence=confidence,
        )

        # Address counter-arguments
        counter_arguments = additional_context.get("counter_arguments", []) if additional_context else []
        counter_arguments_addressed = self._address_counter_arguments(
            spawned_sme, debate_position, counter_arguments
        )

        # Identify remaining concerns
        remaining_concerns = self._identify_remaining_concerns(
            spawned_sme, debate_position, counter_arguments
        )

        # Calculate willingness to concede
        willingness_to_concede = self._calculate_concession_willingness(
            spawned_sme, debate_position, counter_arguments_addressed
        )

        # Build debater report
        debater_report = DebaterReport(
            sme_persona=spawned_sme.persona_name,
            debate_round=debate_round,
            position=debate_position,
            counter_arguments_addressed=counter_arguments_addressed,
            remaining_concerns=remaining_concerns,
            willingness_to_concede=willingness_to_concede,
        )

        # Generate findings from position
        findings = [
            f"Position: {position}",
            f"Confidence: {confidence:.1%}",
            f"Round {debate_round}"
        ]

        recommendations = []
        if willingness_to_concede > 0.7:
            recommendations.append("Willing to concede on minor points")
        elif willingness_to_concede < 0.3:
            recommendations.append("Strongly advocates for domain position")

        return SMEAdvisoryReport(
            sme_persona=spawned_sme.persona_name,
            interaction_mode=SMEInteractionMode.DEBATER,
            domain=spawned_sme.domain,
            task_context=spawned_sme.spawn_context.get("task_context", ""),
            findings=findings,
            recommendations=recommendations,
            confidence=confidence,
            caveats=self._generate_debate_caveats(spawned_sme),
            debater_report=debater_report,
            skills_used=spawned_sme.skills_loaded,
            additional_domains_consulted=[],
        )

    # ========================================================================
    # Helper Methods for Advisor Mode
    # ========================================================================

    def _analyze_domain_findings(
        self,
        spawned_sme: SpawnedSME,
        content: str,
    ) -> List[str]:
        """Analyze content for domain-specific findings."""
        findings = []

        content_lower = content.lower()

        # Domain-specific patterns
        domain_patterns = {
            "security": [
                (r"password\s*=\s*[\"']", "Hardcoded password"),
                (r"sql.*?\+", "SQL injection risk"),
                (r"eval\s*\(", "Code injection risk"),
            ],
            "cloud": [
                (r"hardcoded.*ip", "Hardcoded infrastructure"),
                (r"access.*key", "Potential credential exposure"),
                (r"region.*not.*specified", "Missing region specification"),
            ],
            "data": [
                (r"select\s*\*", "Potential over-fetching"),
                (r"n\+1", "N+1 query pattern"),
                (r"no.*index", "Missing database index"),
            ],
        }

        # Check patterns relevant to this SME's domain
        domain_lower = spawned_sme.domain.lower()

        for domain, patterns in domain_patterns.items():
            if domain in domain_lower:
                for pattern, description in patterns:
                    if re.search(pattern, content_lower):
                        findings.append(f"{domain.title()}: {description}")

        # If no specific findings, provide general observation
        if not findings:
            findings.append(f"Reviewed content for {spawned_sme.domain} considerations")

        return findings[:5]  # Limit to 5 findings

    def _identify_domain_corrections(
        self,
        spawned_sme: SpawnedSME,
        content: str,
        findings: List[str],
    ) -> List[str]:
        """Identify domain-specific corrections needed."""
        corrections = []

        # Map findings to corrections
        for finding in findings:
            if "password" in finding.lower():
                corrections.append("Use environment variables for credentials")
            elif "injection" in finding.lower():
                corrections.append("Use parameterized queries or input sanitization")
            elif "hardcoded" in finding.lower():
                corrections.append("Externalize configuration to secure storage")
            elif "over-fetching" in finding.lower():
                corrections.append("Specify exact columns needed")
            elif "n+1" in finding.lower():
                corrections.append("Use eager loading or batch queries")

        # Add domain-specific corrections if none found
        if not corrections:
            corrections.extend(self._get_default_corrections(spawned_sme))

        return corrections[:3]

    def _get_default_corrections(self, spawned_sme: SpawnedSME) -> List[str]:
        """Get default domain corrections."""
        domain_lower = spawned_sme.domain.lower()

        if "security" in domain_lower:
            return ["Follow OWASP security guidelines", "Implement defense in depth"]
        elif "cloud" in domain_lower:
            return ["Follow cloud provider best practices", "Use infrastructure as code"]
        elif "data" in domain_lower:
            return ["Normalize data schema", "Implement proper indexing"]
        elif "test" in domain_lower:
            return ["Increase test coverage", "Add edge case tests"]
        else:
            return ["Follow domain best practices", "Consult domain documentation"]

    def _identify_missing_considerations(
        self,
        spawned_sme: SpawnedSME,
        content: str,
    ) -> List[str]:
        """Identify domain considerations that are missing."""
        missing = []

        content_lower = content.lower()

        # Domain-specific missing considerations
        domain_considerations = {
            "security": [
                ("rate limiting", "No rate limiting mentioned"),
                ("encryption", "Encryption not specified"),
                ("audit", "No audit logging consideration"),
            ],
            "cloud": [
                ("scalability", "Scalability not addressed"),
                ("cost", "Cost optimization not considered"),
                ("disaster recovery", "No disaster recovery plan"),
            ],
            "data": [
                ("backup", "No backup strategy"),
                ("migration", "Data migration not addressed"),
                ("privacy", "Privacy considerations missing"),
            ],
        }

        domain_lower = spawned_sme.domain.lower()

        for domain, considerations in domain_considerations.items():
            if domain in domain_lower:
                for keyword, description in considerations:
                    if keyword not in content_lower:
                        missing.append(description)

        return missing[:3]

    def _generate_recommendations(
        self,
        spawned_sme: SpawnedSME,
        findings: List[str],
        corrections: List[str],
    ) -> List[str]:
        """Generate domain-specific recommendations."""
        recommendations = []

        # Add corrections as recommendations
        recommendations.extend(corrections)

        # Add domain-specific general recommendations
        domain_lower = spawned_sme.domain.lower()

        if "security" in domain_lower:
            recommendations.append("Conduct security review before deployment")
        elif "cloud" in domain_lower:
            recommendations.append("Review architecture for cost optimization")
        elif "data" in domain_lower:
            recommendations.append("Validate data model with stakeholders")
        elif "test" in domain_lower:
            recommendations.append("Achieve minimum 80% code coverage")

        return list(set(recommendations))[:5]  # Dedupe and limit

    def _calculate_confidence(
        self,
        spawned_sme: SpawnedSME,
        findings: List[str],
    ) -> float:
        """Calculate confidence in the advisory."""
        # Base confidence
        confidence = 0.8

        # Adjust based on findings
        if len(findings) > 0:
            confidence += 0.1  # Found something to review

        # SMEs are confident in their domain
        confidence = min(0.95, confidence)

        return confidence

    def _generate_caveats(self, spawned_sme: SpawnedSME) -> List[str]:
        """Generate caveats for the advisory."""
        caveats = []

        domain_lower = spawned_sme.domain.lower()

        if "security" in domain_lower:
            caveats.append("Full security audit requires penetration testing")
        elif "cloud" in domain_lower:
            caveats.append("Architecture depends on specific cloud provider")
        elif "data" in domain_lower:
            caveats.append("Data modeling may need iteration")
        elif "test" in domain_lower:
            caveats.append("Test coverage may need manual verification")

        # Add general caveat
        caveats.append("Review based on provided content only")

        return caveats

    # ========================================================================
    # Helper Methods for Co-Executor Mode
    # ========================================================================

    def _determine_sections(
        self,
        spawned_sme: SpawnedSME,
        content: str,
    ) -> List[Dict[str, str]]:
        """Determine what sections this SME should contribute."""
        sections = []

        domain_lower = spawned_sme.domain.lower()

        # Domain-specific section mappings
        if "security" in domain_lower:
            sections = [
                {"title": "Security Considerations", "type": "considerations"},
                {"title": "Authentication & Authorization", "type": "implementation"},
            ]
        elif "cloud" in domain_lower:
            sections = [
                {"title": "Infrastructure Architecture", "type": "architecture"},
                {"title": "Deployment Strategy", "type": "deployment"},
            ]
        elif "data" in domain_lower:
            sections = [
                {"title": "Data Architecture", "type": "architecture"},
                {"title": "Data Pipeline", "type": "implementation"},
            ]
        elif "frontend" in domain_lower:
            sections = [
                {"title": "User Interface", "type": "implementation"},
                {"title": "Component Structure", "type": "architecture"},
            ]
        elif "devops" in domain_lower:
            sections = [
                {"title": "CI/CD Pipeline", "type": "implementation"},
                {"title": "Monitoring & Observability", "type": "operations"},
            ]
        elif "documentation" in domain_lower or "technical writ" in domain_lower:
            sections = [
                {"title": "Documentation", "type": "documentation"},
                {"title": "Usage Guide", "type": "guide"},
            ]
        else:
            # Default section
            sections = [
                {"title": f"{spawned_sme.domain} Considerations", "type": "considerations"},
            ]

        return sections

    def _generate_section(
        self,
        spawned_sme: SpawnedSME,
        section_def: Dict[str, str],
        content: str,
        additional_context: Optional[Dict[str, Any]],
    ) -> CoExecutorSection:
        """Generate a specific section contribution."""
        title = section_def["title"]
        section_type = section_def["type"]

        # Generate content based on section type
        if section_type == "architecture":
            section_content = self._generate_architecture_section(spawned_sme, title)
        elif section_type == "implementation":
            section_content = self._generate_implementation_section(spawned_sme, title)
        elif section_type == "deployment":
            section_content = self._generate_deployment_section(spawned_sme, title)
        elif section_type == "documentation":
            section_content = self._generate_documentation_section(spawned_sme, title)
        elif section_type == "operations":
            section_content = self._generate_operations_section(spawned_sme, title)
        else:
            section_content = self._generate_generic_section(spawned_sme, title)

        return CoExecutorSection(
            sme_persona=spawned_sme.persona_name,
            section_title=title,
            content=section_content,
            domain_context=f"From {spawned_sme.domain} perspective",
            integration_notes=f"Integrate with main output",
        )

    def _generate_architecture_section(self, spawned_sme: SpawnedSME, title: str) -> str:
        """Generate an architecture section."""
        return f"""
## {title}

From a {spawned_sme.domain} perspective, the architecture should follow these principles:

1. **Separation of Concerns**: Components should have clear boundaries
2. **Scalability**: Design for horizontal scaling where appropriate
3. **Maintainability**: Use standard patterns and conventions

**Key Components:**
- Core service layer
- Data access layer
- Integration interfaces

**Recommendations:**
- Follow {spawned_sme.domain} best practices
- Consider future extensibility
- Document key architectural decisions
"""

    def _generate_implementation_section(self, spawned_sme: SpawnedSME, title: str) -> str:
        """Generate an implementation section."""
        return f"""
## {title}

**Implementation Approach for {spawned_sme.domain}:**

```python
# Example structure following {spawned_sme.domain} conventions
class {spawned_sme.domain.replace(' ', '').replace('&', 'And')}Handler:
    def __init__(self, config):
        self.config = config

    def process(self, input_data):
        # Process according to domain standards
        result = self._apply_domain_logic(input_data)
        return self._validate_result(result)

    def _apply_domain_logic(self, data):
        # Domain-specific implementation
        pass
```

**Key Considerations:**
- Error handling appropriate for {spawned_sme.domain}
- Logging for observability
- Performance optimization
"""

    def _generate_deployment_section(self, spawned_sme: SpawnedSME, title: str) -> str:
        """Generate a deployment section."""
        return f"""
## {title}

**Deployment Strategy:**

1. **Environment Configuration**
   - Development: Local development setup
   - Staging: Pre-production testing
   - Production: Live deployment

2. **Deployment Steps**
   ```bash
   # Prepare deployment
   npm run build  # or equivalent
   # Run tests
   npm test
   # Deploy
   npm run deploy
   ```

3. **Rollback Plan**
   - Maintain previous version
   - Automated rollback capability
   - Health check validation
"""

    def _generate_documentation_section(self, spawned_sme: SpawnedSME, title: str) -> str:
        """Generate a documentation section."""
        return f"""
## {title}

**Documentation Structure:**

### Overview
Brief description of the component and its purpose.

### API Reference
Details of public interfaces and their usage.

### Examples
Practical examples demonstrating common use cases.

### Troubleshooting
Common issues and their resolutions.

### Best Practices
Recommended patterns and approaches for {spawned_sme.domain}.
"""

    def _generate_operations_section(self, spawned_sme: SpawnedSME, title: str) -> str:
        """Generate an operations section."""
        return f"""
## {title}

**Monitoring & Observability:**

**Metrics to Track:**
- Performance metrics (latency, throughput)
- Error rates and types
- Resource utilization
- Business metrics specific to {spawned_sme.domain}

**Alerting:**
- Configure alerts for critical thresholds
- Escalation procedures
- On-call documentation

**Logging:**
- Structured logging format
- Log levels (DEBUG, INFO, WARNING, ERROR)
- Log aggregation and analysis
"""

    def _generate_generic_section(self, spawned_sme: SpawnedSME, title: str) -> str:
        """Generate a generic section."""
        return f"""
## {title}

**{spawned_sme.domain} Considerations:**

This section outlines key considerations from a {spawned_sme.domain} perspective.

**Key Points:**
- Follow {spawned_sme.domain} best practices
- Consider relevant standards and regulations
- Document all domain-specific decisions

**Recommendations:**
- Consult {spawned_sme.domain} documentation
- Review with relevant stakeholders
- Iterate based on feedback
"""

    def _generate_coordination_notes(
        self,
        spawned_sme: SpawnedSME,
        contributed_sections: List[CoExecutorSection],
        content: str,
    ) -> str:
        """Generate coordination notes for integrating contributed sections."""
        if not contributed_sections:
            return "No sections contributed - advisory role only"

        titles = [s.section_title for s in contributed_sections]

        return (
            f"Integrate the following sections: {', '.join(titles)}. "
            f"Ensure consistency with main content. "
            f"Review for overlapping information."
        )

    def _identify_domain_assumptions(
        self,
        spawned_sme: SpawnedSME,
        content: str,
    ) -> List[str]:
        """Identify domain-specific assumptions made."""
        assumptions = []

        domain_lower = spawned_sme.domain.lower()

        # Domain-specific assumptions
        if "cloud" in domain_lower:
            assumptions = [
                "Cloud provider is chosen",
                "Basic cloud infrastructure is available",
                "Team has cloud expertise",
            ]
        elif "security" in domain_lower:
            assumptions = [
                "Security requirements are defined",
                "Compliance standards are known",
                "Security review process exists",
            ]
        elif "data" in domain_lower:
            assumptions = [
                "Data sources are identified",
                "Data volume estimates are available",
                "Data quality is acceptable",
            ]
        else:
            assumptions = [
                f"{spawned_sme.domain} requirements are understood",
                f"Relevant {spawned_sme.domain} standards apply",
            ]

        return assumptions

    # ========================================================================
    # Helper Methods for Debater Mode
    # ========================================================================

    def _determine_debate_position(
        self,
        spawned_sme: SpawnedSME,
        content: str,
        additional_context: Optional[Dict[str, Any]],
    ) -> str:
        """Determine the SME's position in the debate."""
        # Get proposed position from content
        content_lower = content.lower()

        # Domain-specific default positions
        domain_positions = {
            "security": "Security should be the highest priority, even if it impacts performance or user experience",
            "cloud": "Cloud-native architecture with serverless components provides the best scalability and cost-efficiency",
            "data": "Data integrity and consistency must be maintained above all else, favoring ACID compliance",
            "frontend": "User experience should be prioritized, with fast load times and responsive design",
            "devops": "Automation and CI/CD should be prioritized for faster delivery and reliability",
        }

        domain_lower = spawned_sme.domain.lower()

        for domain, position in domain_positions.items():
            if domain in domain_lower:
                return position

        # Default position
        return f"Approach should prioritize {spawned_sme.domain} considerations"

    def _generate_domain_rationale(
        self,
        spawned_sme: SpawnedSME,
        position: str,
    ) -> str:
        """Generate domain-specific rationale for the position."""
        domain_lower = spawned_sme.domain.lower()

        rationales = {
            "security": "Security vulnerabilities can lead to data breaches, regulatory fines, and reputational damage. Prevention is far cheaper than remediation.",
            "cloud": "Cloud-native architectures provide automatic scaling, reduced operational overhead, and pay-per-use pricing. Serverless eliminates infrastructure management.",
            "data": "Data corruption or inconsistency can lead to incorrect business decisions and regulatory violations. ACID compliance ensures data integrity.",
            "frontend": "User experience directly impacts adoption and satisfaction. Slow or clunky interfaces drive users away regardless of backend quality.",
            "devops": "Automation reduces human error, accelerates delivery, and improves reliability. CI/CD enables rapid iteration and rollback capability.",
        }

        for domain, rationale in rationales.items():
            if domain in domain_lower:
                return rationale

        return f"Based on {spawned_sme.domain} expertise and best practices"

    def _gather_supporting_evidence(
        self,
        spawned_sme: SpawnedSME,
        position: str,
    ) -> List[str]:
        """Gather domain-specific evidence supporting the position."""
        domain_lower = spawned_sme.domain.lower()

        evidence = {
            "security": [
                "OWASP Top 10 highlights injection and authentication vulnerabilities as top risks",
                "IBM Cost of Data Breach Report 2023: Average breach cost is $4.45 million",
                "Verizon DBIR: 83% of breaches involved external actors, not insider threats",
            ],
            "cloud": [
                "AWS serverless growth: 300% YoY increase in Lambda usage",
                "Forrester: Cloud-native apps are 2.5x faster to develop and deploy",
                "Gartner: By 2025, 85% of enterprises will use cloud-native principles",
            ],
            "data": [
                "Gartner: Poor data quality costs organizations $15 million per year on average",
                "ACID compliance is required for financial transactions (SOX, Basel II)",
                "Data consistency issues caused 40% of system failures in 2023",
            ],
            "frontend": [
                "Google research: 53% of mobile users abandon sites that take >3 seconds to load",
                "Forrester: Every 1-second delay reduces conversions by 7%",
                "AWS: 88% of online users are less likely to return after poor UX",
            ],
            "devops": [
                "DORA 2023: High-performing teams deploy 208x more frequently with 7x lower failure rates",
                "GitHub: Teams with CI/CD ship code 2.5x faster",
                "Gartner: By 2025, 70% of organizations will use automated release pipelines",
            ],
        }

        for domain, domain_evidence in evidence.items():
            if domain in domain_lower:
                return domain_evidence

        return ["Industry best practices support this position"]

    def _calculate_position_confidence(
        self,
        spawned_sme: SpawnedSME,
        position: str,
    ) -> float:
        """Calculate confidence in the debate position."""
        # SMEs are generally confident in their domain positions
        return 0.85

    def _address_counter_arguments(
        self,
        spawned_sme: SpawnedSME,
        debate_position: DebatePosition,
        counter_arguments: List[str],
    ) -> List[str]:
        """Address counter-arguments raised by other debaters."""
        addressed = []

        for counter_arg in counter_arguments:
            # Generate responses to common counter-arguments
            if "cost" in counter_arg.lower():
                addressed.append(f"While cost is a factor, the long-term cost of not addressing {spawned_sme.domain} concerns is higher")
            elif "performance" in counter_arg.lower():
                addressed.append(f"Performance impact can be mitigated through proper architecture and optimization")
            elif "complexity" in counter_arg.lower():
                addressed.append(f"Complexity is justified by the risk mitigation provided")
            elif "time" in counter_arg.lower():
                addressed.append(f"Upfront investment saves significant time in remediation and rework")
            else:
                addressed.append(f"This concern is valid but can be addressed without compromising {spawned_sme.domain} principles")

        return addressed[:3]

    def _identify_remaining_concerns(
        self,
        spawned_sme: SpawnedSME,
        debate_position: DebatePosition,
        counter_arguments: List[str],
    ) -> List[str]:
        """Identify concerns the SME still has after debate."""
        concerns = []

        # Check if counter-arguments were compelling
        if len(counter_arguments) > 3:
            concerns.append("Multiple competing priorities require careful balancing")

        # Domain-specific concerns
        domain_lower = spawned_sme.domain.lower()

        if "security" in domain_lower:
            concerns.append("Security budgets are often cut to meet deadlines")
        elif "cloud" in domain_lower:
            concerns.append("Cloud cost overruns are common without proper governance")
        elif "data" in domain_lower:
            concerns.append("Data quality is often overestimated in planning")

        return concerns[:2]

    def _calculate_concession_willingness(
        self,
        spawned_sme: SpawnedSME,
        debate_position: DebatePosition,
        counter_arguments_addressed: List[str],
    ) -> float:
        """Calculate willingness to concede on the position."""
        # Start with willingness to concede on minor points
        willingness = 0.3

        # Increase if many counter-arguments were addressed
        willingness += len(counter_arguments_addressed) * 0.1

        # Cap at 0.7 (SMEs rarely fully concede on core domain positions)
        return min(0.7, willingness)

    def _generate_debate_caveats(self, spawned_sme: SpawnedSME) -> List[str]:
        """Generate caveats for the debate position."""
        return [
            "Position based on domain expertise and best practices",
            "Willing to consider alternative approaches if evidence supports them",
            "Recommend hybrid solution if pure approach has significant trade-offs",
        ]

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def _convert_interaction_mode(
        self,
        council_mode: CouncilInteractionMode,
    ) -> SMEInteractionMode:
        """Convert Council interaction mode to SME interaction mode."""
        mode_map = {
            CouncilInteractionMode.ADVISOR: SMEInteractionMode.ADVISOR,
            CouncilInteractionMode.CO_EXECUTOR: SMEInteractionMode.CO_EXECUTOR,
            CouncilInteractionMode.DEBATER: SMEInteractionMode.DEBATER,
        }
        return mode_map.get(council_mode, SMEInteractionMode.ADVISOR)

    def _load_system_prompt(self, persona) -> str:
        """Load system prompt for a persona."""
        template_path = Path(persona.system_prompt_template)

        if not template_path.exists():
            # Try in sme_templates_dir
            template_path = self.sme_templates_dir / f"{persona.persona_id}.md"

        if template_path.exists():
            try:
                with open(template_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                import logging
                logging.getLogger("sme_spawner").warning(f"Failed to load system prompt from {template_path}: {e}")

        # Generate default system prompt
        return f"""You are {persona.name}, an expert in {persona.domain}.

Your expertise includes:
- {persona.description}

Provide guidance and recommendations based on your domain expertise.
"""

    def _load_skills(
        self,
        persona,
        skills_to_load: List[str],
    ) -> List[str]:
        """Load SKILL.md files for a persona."""
        loaded = []

        for skill_name in persona.skill_files:
            skill_path = self.skills_dir / skill_name / "SKILL.md"

            if skill_path.exists():
                loaded.append(skill_name)

        # If specific skills requested, try to load those too
        if skills_to_load:
            for skill in skills_to_load:
                if skill not in loaded:
                    skill_path = self.skills_dir / skill / "SKILL.md"
                    if skill_path.exists():
                        loaded.append(skill)

        return loaded

    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()


# =============================================================================
# Convenience Functions
# =============================================================================

def create_sme_spawner(
    skills_dir: str = ".claude/skills",
    sme_templates_dir: str = "config/sme",
    model: str = "claude-3-5-sonnet-20241022",
) -> SMESpawner:
    """Create a configured SME Spawner."""
    return SMESpawner(
        skills_dir=skills_dir,
        sme_templates_dir=sme_templates_dir,
        model=model,
    )
