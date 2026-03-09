"""
Ensemble Patterns - Pre-configured Agent Workflows

Named ensemble patterns for common multi-agent scenarios:
- Architecture Review Board
- Code Sprint
- Research Council
- Document Assembly
- Requirements Workshop
"""

from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod


class EnsembleType(str, Enum):
    """Types of ensemble patterns."""
    ARCHITECTURE_REVIEW_BOARD = "architecture_review_board"
    CODE_SPRINT = "code_sprint"
    RESEARCH_COUNCIL = "research_council"
    DOCUMENT_ASSEMBLY = "document_assembly"
    REQUIREMENTS_WORKSHOP = "requirements_workshop"


class AgentRole(str, Enum):
    """Roles agents can play in an ensemble."""
    LEAD = "lead"
    REVIEWER = "reviewer"
    CONTRIBUTOR = "contributor"
    ADVISOR = "advisor"
    QUALITY_GATE = "quality_gate"
    OBSERVER = "observer"


@dataclass
class AgentAssignment:
    """An agent assignment in an ensemble."""
    agent_name: str
    role: AgentRole
    phase: str
    dependencies: List[str]  # Other agents this depends on
    parallel_with: List[str]  # Agents this can run in parallel with
    max_turns: int = 30
    model: str = "claude-3-5-sonnet-20241022"


@dataclass
class EnsembleConfig:
    """Configuration for an ensemble pattern."""
    name: str
    description: str
    ensemble_type: EnsembleType
    tier_level: int
    agent_assignments: List[AgentAssignment]
    required_smes: List[str]
    quality_gates: List[str]
    expected_output: str
    success_criteria: List[str]


@dataclass
class EnsembleResult:
    """Result from executing an ensemble pattern."""
    ensemble_type: EnsembleType
    success: bool
    outputs: Dict[str, Any]  # Agent outputs by agent name
    quality_gate_results: Dict[str, bool]
    total_turns: int
    execution_time_seconds: float
    recommendations: List[str]


class EnsemblePattern(ABC):
    """
    Base class for ensemble patterns.

    Each pattern defines a specific workflow optimized for a common task type.
    """

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def get_config(self) -> EnsembleConfig:
        """Get the ensemble configuration."""
        pass

    @abstractmethod
    def execute(
        self,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> EnsembleResult:
        """Execute the ensemble pattern."""
        pass

    def _validate_dependencies(self, config: EnsembleConfig) -> bool:
        """Validate that all dependencies are satisfied."""
        assigned_agents = {a.agent_name for a in config.agent_assignments}

        for assignment in config.agent_assignments:
            for dep in assignment.dependencies:
                if dep not in assigned_agents:
                    return False

        return True

    def _calculate_parallel_groups(
        self,
        config: EnsembleConfig,
    ) -> List[List[str]]:
        """Calculate groups of agents that can run in parallel."""
        groups = []
        assigned = set()

        # Find agents with no unassigned dependencies
        while len(assigned) < len(config.agent_assignments):
            ready = []

            for assignment in config.agent_assignments:
                if assignment.agent_name in assigned:
                    continue

                # Check if all dependencies are assigned
                deps_satisfied = all(
                    dep in assigned
                    for dep in assignment.dependencies
                )

                if deps_satisfied:
                    ready.append(assignment.agent_name)

            if not ready:
                # Circular dependency or error
                break

            # Group agents that can run in parallel
            parallel_group = []
            for agent_name in ready:
                assignment = next(
                    a for a in config.agent_assignments
                    if a.agent_name == agent_name
                )

                # Check if it can run with others in the ready group
                can_parallel = True
                for other_name in ready:
                    if other_name != agent_name:
                        if (other_name not in assignment.parallel_with and
                            agent_name not in next(
                                a for a in config.agent_assignments
                                if a.agent_name == other_name
                            ).parallel_with):
                            can_parallel = False
                            break

                if can_parallel:
                    parallel_group.append(agent_name)

            groups.append(parallel_group)
            assigned.update(parallel_group)

        return groups


# =============================================================================
# Architecture Review Board
# =============================================================================

class ArchitectureReviewBoard(EnsemblePattern):
    """
    Architecture Review Board ensemble pattern.

    Optimized for reviewing software/systems architecture for:
    - Structural soundness
    - Security considerations
    - Scalability concerns
    - Cost implications
    - Best practices adherence
    """

    def __init__(self):
        super().__init__(
            name="Architecture Review Board",
            description="Comprehensive architecture review with cloud, security, and data expertise"
        )

    def get_config(self) -> EnsembleConfig:
        """Get the ARB ensemble configuration."""
        return EnsembleConfig(
            name=self.name,
            description=self.description,
            ensemble_type=EnsembleType.ARCHITECTURE_REVIEW_BOARD,
            tier_level=3,  # Requires Council
            agent_assignments=[
                # Phase 1: Analysis
                AgentAssignment(
                    agent_name="Analyst",
                    role=AgentRole.LEAD,
                    phase="analysis",
                    dependencies=[],
                    parallel_with=[],
                    max_turns=20,
                ),
                # Phase 2: Domain Reviews (Parallel)
                AgentAssignment(
                    agent_name="Cloud Architect",
                    role=AgentRole.ADVISOR,
                    phase="domain_review",
                    dependencies=["Analyst"],
                    parallel_with=["Security Analyst", "Data Engineer"],
                    max_turns=25,
                ),
                AgentAssignment(
                    agent_name="Security Analyst",
                    role=AgentRole.ADVISOR,
                    phase="domain_review",
                    dependencies=["Analyst"],
                    parallel_with=["Cloud Architect", "Data Engineer"],
                    max_turns=25,
                ),
                AgentAssignment(
                    agent_name="Data Engineer",
                    role=AgentRole.ADVISOR,
                    phase="domain_review",
                    dependencies=["Analyst"],
                    parallel_with=["Cloud Architect", "Security Analyst"],
                    max_turns=25,
                ),
                # Phase 3: Synthesis
                AgentAssignment(
                    agent_name="Executor",
                    role=AgentRole.CONTRIBUTOR,
                    phase="synthesis",
                    dependencies=["Cloud Architect", "Security Analyst", "Data Engineer"],
                    parallel_with=[],
                    max_turns=30,
                ),
                # Phase 4: Quality Gates
                AgentAssignment(
                    agent_name="CodeReviewer",
                    role=AgentRole.QUALITY_GATE,
                    phase="quality",
                    dependencies=["Executor"],
                    parallel_with=["Verifier", "Critic"],
                    max_turns=25,
                ),
                AgentAssignment(
                    agent_name="Verifier",
                    role=AgentRole.QUALITY_GATE,
                    phase="quality",
                    dependencies=["Executor"],
                    parallel_with=["CodeReviewer", "Critic"],
                    max_turns=25,
                ),
                AgentAssignment(
                    agent_name="Critic",
                    role=AgentRole.QUALITY_GATE,
                    phase="quality",
                    dependencies=["Executor"],
                    parallel_with=["CodeReviewer", "Verifier"],
                    max_turns=25,
                ),
                # Phase 5: Final Review
                AgentAssignment(
                    agent_name="Reviewer",
                    role=AgentRole.QUALITY_GATE,
                    phase="final",
                    dependencies=["CodeReviewer", "Verifier", "Critic"],
                    parallel_with=[],
                    max_turns=20,
                ),
            ],
            required_smes=["cloud_architect", "security_analyst", "data_engineer"],
            quality_gates=["CodeReviewer", "Verifier", "Critic", "Reviewer"],
            expected_output="Comprehensive architecture review report with findings and recommendations",
            success_criteria=[
                "All three domain SMEs provide input",
                "Security vulnerabilities identified",
                "Scalability concerns addressed",
                "Cost implications analyzed",
                "Actionable recommendations provided",
            ],
        )

    def execute(
        self,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> EnsembleResult:
        """Execute the Architecture Review Board pattern."""
        config = self.get_config()

        # In a real implementation, this would:
        # 1. Spawn the required agents
        # 2. Execute them in the defined order
        # 3. Collect outputs
        # 4. Run quality gates
        # 5. Aggregate results

        # Simulated execution for now
        return EnsembleResult(
            ensemble_type=config.ensemble_type,
            success=True,
            outputs={
                "analyst": "Architecture requirements analyzed",
                "cloud_architect_sme": "Cloud infrastructure recommendations",
                "security_analyst_sme": "Security vulnerabilities identified",
                "data_engineer_sme": "Data pipeline considerations",
                "executor": "Synthesized architecture review",
                "code_reviewer": "Code quality assessment",
                "verifier": "Factual claims verified",
                "critic": "Critique completed",
                "reviewer": "Final review passed",
            },
            quality_gate_results={
                "code_reviewer": True,
                "verifier": True,
                "critic": True,
                "reviewer": True,
            },
            total_turns=225,
            execution_time_seconds=180.0,
            recommendations=[
                "Implement auto-scaling for production workloads",
                "Add WAF for public endpoints",
                "Use read replicas for reporting queries",
            ],
        )


# =============================================================================
# Code Sprint
# =============================================================================

class CodeSprint(EnsemblePattern):
    """
    Code Sprint ensemble pattern.

    Optimized for rapid code development with:
    - Quick task breakdown
    - Fast implementation
    - Concurrent code review
    - Automated testing guidance
    - Documentation generation
    """

    def __init__(self):
        super().__init__(
            name="Code Sprint",
            description="Rapid code development with parallel review and testing"
        )

    def get_config(self) -> EnsembleConfig:
        """Get the Code Sprint ensemble configuration."""
        return EnsembleConfig(
            name=self.name,
            description=self.description,
            ensemble_type=EnsembleType.CODE_SPRINT,
            tier_level=2,  # Standard tier
            agent_assignments=[
                # Phase 1: Quick Planning
                AgentAssignment(
                    agent_name="Planner",
                    role=AgentRole.LEAD,
                    phase="planning",
                    dependencies=[],
                    parallel_with=[],
                    max_turns=15,
                    model="claude-3-5-haiku-20250101",  # Faster model for planning
                ),
                # Phase 2: Implementation
                AgentAssignment(
                    agent_name="Executor",
                    role=AgentRole.CONTRIBUTOR,
                    phase="implementation",
                    dependencies=["Planner"],
                    parallel_with=[],
                    max_turns=40,
                ),
                # Phase 3: Parallel Quality Checks
                AgentAssignment(
                    agent_name="CodeReviewer",
                    role=AgentRole.REVIEWER,
                    phase="quality",
                    dependencies=["Executor"],
                    parallel_with=["Test Engineer SME"],
                    max_turns=25,
                ),
                AgentAssignment(
                    agent_name="Test Engineer",
                    role=AgentRole.ADVISOR,
                    phase="quality",
                    dependencies=["Executor"],
                    parallel_with=["CodeReviewer"],
                    max_turns=20,
                ),
                # Phase 4: Final Review & Format
                AgentAssignment(
                    agent_name="Verifier",
                    role=AgentRole.QUALITY_GATE,
                    phase="verification",
                    dependencies=["CodeReviewer"],
                    parallel_with=[],
                    max_turns=15,
                ),
                AgentAssignment(
                    agent_name="Formatter",
                    role=AgentRole.CONTRIBUTOR,
                    phase="formatting",
                    dependencies=["Verifier"],
                    parallel_with=[],
                    max_turns=10,
                ),
            ],
            required_smes=["test_engineer"],
            quality_gates=["CodeReviewer", "Verifier"],
            expected_output="Working code with tests and documentation",
            success_criteria=[
                "Code implements requirements",
                "No critical security issues",
                "Test cases provided",
                "Code is documented",
                "Code passes syntax validation",
            ],
        )

    def execute(
        self,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> EnsembleResult:
        """Execute the Code Sprint pattern."""
        config = self.get_config()

        return EnsembleResult(
            ensemble_type=config.ensemble_type,
            success=True,
            outputs={
                "planner": "Implementation plan created",
                "executor": "Code implemented",
                "code_reviewer": "Code reviewed - minor issues found",
                "test_engineer_sme": "Test strategy defined",
                "verifier": "Code verified",
                "formatter": "Code formatted with documentation",
            },
            quality_gate_results={
                "code_reviewer": True,
                "verifier": True,
            },
            total_turns=125,
            execution_time_seconds=90.0,
            recommendations=[
                "Add unit tests for edge cases",
                "Consider adding type hints",
                "Document error handling",
            ],
        )


# =============================================================================
# Research Council
# =============================================================================

class ResearchCouncil(EnsemblePattern):
    """
    Research Council ensemble pattern.

    Optimized for deep research tasks with:
    - Comprehensive evidence gathering
    - Source verification
    - Multiple perspective analysis
    - Conflict resolution
    - Synthesis of findings
    """

    def __init__(self):
        super().__init__(
            name="Research Council",
            description="Deep research with evidence gathering and verification"
        )

    def get_config(self) -> EnsembleConfig:
        """Get the Research Council ensemble configuration."""
        return EnsembleConfig(
            name=self.name,
            description=self.description,
            ensemble_type=EnsembleType.RESEARCH_COUNCIL,
            tier_level=4,  # Adversarial tier
            agent_assignments=[
                # Phase 1: Research Planning
                AgentAssignment(
                    agent_name="Analyst",
                    role=AgentRole.LEAD,
                    phase="planning",
                    dependencies=[],
                    parallel_with=[],
                    max_turns=20,
                ),
                # Phase 2: Parallel Research (Multiple Researchers)
                AgentAssignment(
                    agent_name="Researcher",
                    role=AgentRole.CONTRIBUTOR,
                    phase="research",
                    dependencies=["Analyst"],
                    parallel_with=[],
                    max_turns=40,
                ),
                # Phase 3: Domain Expert Analysis (Parallel)
                AgentAssignment(
                    agent_name="AI/ML Engineer SME",
                    role=AgentRole.ADVISOR,
                    phase="domain_analysis",
                    dependencies=["Researcher"],
                    parallel_with=["Data Engineer SME"],
                    max_turns=25,
                ),
                AgentAssignment(
                    agent_name="Data Engineer SME",
                    role=AgentRole.ADVISOR,
                    phase="domain_analysis",
                    dependencies=["Researcher"],
                    parallel_with=["AI/ML Engineer SME"],
                    max_turns=25,
                ),
                # Phase 4: Verification & Debate
                AgentAssignment(
                    agent_name="Verifier",
                    role=AgentRole.QUALITY_GATE,
                    phase="verification",
                    dependencies=["AI/ML Engineer SME", "Data Engineer SME"],
                    parallel_with=[],
                    max_turns=25,
                ),
                # Phase 5: Self-Play Debate
                AgentAssignment(
                    agent_name="Critic",
                    role=AgentRole.REVIEWER,
                    phase="debate",
                    dependencies=["Verifier"],
                    parallel_with=[],
                    max_turns=30,
                ),
                # Phase 6: Synthesis
                AgentAssignment(
                    agent_name="Executor",
                    role=AgentRole.CONTRIBUTOR,
                    phase="synthesis",
                    dependencies=["Critic"],
                    parallel_with=[],
                    max_turns=30,
                ),
                # Phase 7: Final Review with Ethics
                AgentAssignment(
                    agent_name="Ethics Advisor",
                    role=AgentRole.QUALITY_GATE,
                    phase="ethics_review",
                    dependencies=["Executor"],
                    parallel_with=["Reviewer"],
                    max_turns=20,
                ),
                AgentAssignment(
                    agent_name="Reviewer",
                    role=AgentRole.QUALITY_GATE,
                    phase="final_review",
                    dependencies=["Executor"],
                    parallel_with=["Ethics Advisor"],
                    max_turns=20,
                ),
            ],
            required_smes=["ai_ml_engineer", "data_engineer"],
            quality_gates=["Verifier", "Critic", "Ethics Advisor", "Reviewer"],
            expected_output="Comprehensive research brief with verified sources and ethical considerations",
            success_criteria=[
                "Multiple authoritative sources consulted",
                "Claims verified with sources",
                "Domain expertise incorporated",
                "Conflicts identified and resolved",
                "Ethical considerations addressed",
                "Synthesized findings with confidence levels",
            ],
        )

    def execute(
        self,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> EnsembleResult:
        """Execute the Research Council pattern."""
        config = self.get_config()

        return EnsembleResult(
            ensemble_type=config.ensemble_type,
            success=True,
            outputs={
                "analyst": "Research objectives defined",
                "researcher": "Evidence gathered from multiple sources",
                "ai_ml_engineer_sme": "AI/ML domain analysis provided",
                "data_engineer_sme": "Data engineering considerations",
                "verifier": "Claims verified against sources",
                "critic": "Potential biases identified",
                "executor": "Research findings synthesized",
                "ethics_advisor": "Ethical review passed",
                "reviewer": "Final review completed",
            },
            quality_gate_results={
                "verifier": True,
                "critic": True,
                "ethics_advisor": True,
                "reviewer": True,
            },
            total_turns=230,
            execution_time_seconds=300.0,
            recommendations=[
                "Include additional sources for controversial claims",
                "Consider alternative methodologies",
                "Document limitations of current research",
            ],
        )


# =============================================================================
# Document Assembly
# =============================================================================

class DocumentAssembly(EnsemblePattern):
    """
    Document Assembly ensemble pattern.

    Optimized for creating comprehensive documentation with:
    - Requirements gathering
    - Technical writing
    - Domain expertise
    - Review and refinement
    - Multiple output formats
    """

    def __init__(self):
        super().__init__(
            name="Document Assembly",
            description="Comprehensive documentation creation with review"
        )

    def get_config(self) -> EnsembleConfig:
        """Get the Document Assembly ensemble configuration."""
        return EnsembleConfig(
            name=self.name,
            description=self.description,
            ensemble_type=EnsembleType.DOCUMENT_ASSEMBLY,
            tier_level=2,  # Standard tier
            agent_assignments=[
                # Phase 1: Requirements Analysis
                AgentAssignment(
                    agent_name="Clarifier",
                    role=AgentRole.LEAD,
                    phase="clarification",
                    dependencies=[],
                    parallel_with=[],
                    max_turns=20,
                ),
                # Phase 2: Content Planning
                AgentAssignment(
                    agent_name="Planner",
                    role=AgentRole.CONTRIBUTOR,
                    phase="planning",
                    dependencies=["Clarifier"],
                    parallel_with=[],
                    max_turns=15,
                ),
                # Phase 3: Content Creation
                AgentAssignment(
                    agent_name="Technical Writer SME",
                    role=AgentRole.CONTRIBUTOR,
                    phase="content_creation",
                    dependencies=["Planner"],
                    parallel_with=["Executor"],
                    max_turns=35,
                ),
                AgentAssignment(
                    agent_name="Executor",
                    role=AgentRole.CONTRIBUTOR,
                    phase="content_creation",
                    dependencies=["Planner"],
                    parallel_with=["Technical Writer SME"],
                    max_turns=30,
                ),
                # Phase 4: Review
                AgentAssignment(
                    agent_name="Verifier",
                    role=AgentRole.REVIEWER,
                    phase="review",
                    dependencies=["Technical Writer SME", "Executor"],
                    parallel_with=[],
                    max_turns=20,
                ),
                # Phase 5: Formatting
                AgentAssignment(
                    agent_name="Formatter",
                    role=AgentRole.CONTRIBUTOR,
                    phase="formatting",
                    dependencies=["Verifier"],
                    parallel_with=[],
                    max_turns=15,
                ),
            ],
            required_smes=["technical_writer"],
            quality_gates=["Verifier"],
            expected_output="Professional documentation in requested format",
            success_criteria=[
                "All requirements addressed",
                "Clear and well-structured",
                "Appropriate technical depth",
                "Factually accurate",
                "Properly formatted",
            ],
        )

    def execute(
        self,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> EnsembleResult:
        """Execute the Document Assembly pattern."""
        config = self.get_config()

        return EnsembleResult(
            ensemble_type=config.ensemble_type,
            success=True,
            outputs={
                "clarifier": "Document requirements clarified",
                "planner": "Document structure planned",
                "technical_writer_sme": "Content authored",
                "executor": "Technical content added",
                "verifier": "Content verified",
                "formatter": "Document formatted",
            },
            quality_gate_results={
                "verifier": True,
            },
            total_turns=125,
            execution_time_seconds=120.0,
            recommendations=[
                "Add more diagrams for visual clarity",
                "Include code examples in formatted blocks",
                "Consider adding an FAQ section",
            ],
        )


# =============================================================================
# Requirements Workshop
# =============================================================================

class RequirementsWorkshop(EnsemblePattern):
    """
    Requirements Workshop ensemble pattern.

    Optimized for requirements engineering with:
    - Stakeholder input gathering
    - Requirement elicitation
    - Gap analysis
    - Prioritization
    - Acceptance criteria definition
    """

    def __init__(self):
        super().__init__(
            name="Requirements Workshop",
            description="Comprehensive requirements gathering and analysis"
        )

    def get_config(self) -> EnsembleConfig:
        """Get the Requirements Workshop ensemble configuration."""
        return EnsembleConfig(
            name=self.name,
            description=self.description,
            ensemble_type=EnsembleType.REQUIREMENTS_WORKSHOP,
            tier_level=2,  # Standard tier
            agent_assignments=[
                # Phase 1: Initial Analysis
                AgentAssignment(
                    agent_name="Analyst",
                    role=AgentRole.LEAD,
                    phase="analysis",
                    dependencies=[],
                    parallel_with=[],
                    max_turns=25,
                ),
                # Phase 2: Question Formulation
                AgentAssignment(
                    agent_name="Clarifier",
                    role=AgentRole.CONTRIBUTOR,
                    phase="clarification",
                    dependencies=["Analyst"],
                    parallel_with=[],
                    max_turns=20,
                ),
                # Phase 3: Domain Input
                AgentAssignment(
                    agent_name="Business Analyst SME",
                    role=AgentRole.ADVISOR,
                    phase="domain_input",
                    dependencies=["Clarifier"],
                    parallel_with=[],
                    max_turns=25,
                ),
                # Phase 4: Research
                AgentAssignment(
                    agent_name="Researcher",
                    role=AgentRole.CONTRIBUTOR,
                    phase="research",
                    dependencies=["Business Analyst SME"],
                    parallel_with=[],
                    max_turns=25,
                ),
                # Phase 5: Synthesis
                AgentAssignment(
                    agent_name="Executor",
                    role=AgentRole.CONTRIBUTOR,
                    phase="synthesis",
                    dependencies=["Researcher"],
                    parallel_with=[],
                    max_turns=30,
                ),
                # Phase 6: Verification
                AgentAssignment(
                    agent_name="Verifier",
                    role=AgentRole.QUALITY_GATE,
                    phase="verification",
                    dependencies=["Executor"],
                    parallel_with=[],
                    max_turns=20,
                ),
            ],
            required_smes=["business_analyst"],
            quality_gates=["Verifier"],
            expected_output="Complete requirements specification with acceptance criteria",
            success_criteria=[
                "All stakeholder concerns addressed",
                "Requirements are unambiguous",
                "Acceptance criteria defined",
                "Prioritization provided",
                "Dependencies identified",
            ],
        )

    def execute(
        self,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> EnsembleResult:
        """Execute the Requirements Workshop pattern."""
        config = self.get_config()

        return EnsembleResult(
            ensemble_type=config.ensemble_type,
            success=True,
            outputs={
                "analyst": "Requirements analyzed",
                "clarifier": "Clarifying questions formulated",
                "business_analyst_sme": "Business requirements gathered",
                "researcher": "Supporting research conducted",
                "executor": "Requirements specification synthesized",
                "verifier": "Requirements verified for completeness",
            },
            quality_gate_results={
                "verifier": True,
            },
            total_turns=145,
            execution_time_seconds=150.0,
            recommendations=[
                "Prioritize requirements by MoSCoW method",
                "Define non-functional requirements",
                "Add traceability matrix",
            ],
        )


# =============================================================================
# Ensemble Registry
# =============================================================================

ENSEMBLE_REGISTRY: Dict[EnsembleType, "EnsemblePattern"] = {
    EnsembleType.ARCHITECTURE_REVIEW_BOARD: ArchitectureReviewBoard(),
    EnsembleType.CODE_SPRINT: CodeSprint(),
    EnsembleType.RESEARCH_COUNCIL: ResearchCouncil(),
    EnsembleType.DOCUMENT_ASSEMBLY: DocumentAssembly(),
    EnsembleType.REQUIREMENTS_WORKSHOP: RequirementsWorkshop(),
}


def get_ensemble(ensemble_type: EnsembleType) -> Optional[EnsemblePattern]:
    """
    Get an ensemble pattern by type.

    Args:
        ensemble_type: The type of ensemble pattern

    Returns:
        The ensemble pattern instance, or None if not found
    """
    return ENSEMBLE_REGISTRY.get(ensemble_type)


def get_all_ensembles() -> Dict[EnsembleType, EnsemblePattern]:
    """Get all registered ensemble patterns."""
    return {
        ensemble_type: get_ensemble(ensemble_type)
        for ensemble_type in EnsembleType
    }


def suggest_ensemble(
    task_description: str,
    context: Optional[Dict[str, Any]] = None,
) -> Optional[EnsemblePattern]:
    """
    Suggest an appropriate ensemble pattern based on task description.

    Args:
        task_description: Description of the task
        context: Additional context (modality, tier, etc.)

    Returns:
        The suggested ensemble pattern, or None if no match
    """
    task_lower = task_description.lower()

    # Check for architecture keywords
    arch_keywords = ["architecture", "design", "structure", "component", "system"]
    if any(kw in task_lower for kw in arch_keywords):
        if any(kw in task_lower for kw in ["review", "assess", "evaluate"]):
            return get_ensemble(EnsembleType.ARCHITECTURE_REVIEW_BOARD)

    # Check for code keywords
    code_keywords = ["code", "implement", "function", "class", "develop"]
    if any(kw in task_lower for kw in code_keywords):
        if "sprint" in task_lower or "quick" in task_lower or "fast" in task_lower:
            return get_ensemble(EnsembleType.CODE_SPRINT)

    # Check for research keywords
    research_keywords = ["research", "investigate", "study", "analyze", "findings"]
    if any(kw in task_lower for kw in research_keywords):
        return get_ensemble(EnsembleType.RESEARCH_COUNCIL)

    # Check for documentation keywords
    doc_keywords = ["document", "documentation", "guide", "manual", "write"]
    if any(kw in task_lower for kw in doc_keywords):
        return get_ensemble(EnsembleType.DOCUMENT_ASSEMBLY)

    # Check for requirements keywords
    req_keywords = ["requirements", "specification", "user stories", "acceptance"]
    if any(kw in task_lower for kw in req_keywords):
        return get_ensemble(EnsembleType.REQUIREMENTS_WORKSHOP)

    # Default: Code Sprint for general tasks
    return get_ensemble(EnsembleType.CODE_SPRINT)


def execute_ensemble(
    ensemble_type: EnsembleType,
    input_data: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
) -> EnsembleResult:
    """
    Execute an ensemble pattern.

    Args:
        ensemble_type: The type of ensemble to execute
        input_data: Input data for the ensemble
        context: Additional context

    Returns:
        The ensemble execution result
    """
    ensemble = get_ensemble(ensemble_type)
    if ensemble is None:
        raise ValueError(f"Unknown ensemble type: {ensemble_type}")

    return ensemble.execute(input_data, context)


# =============================================================================
# Convenience Functions
# =============================================================================

def create_architecture_review() -> ArchitectureReviewBoard:
    """Create an Architecture Review Board ensemble."""
    return ArchitectureReviewBoard()


def create_code_sprint() -> CodeSprint:
    """Create a Code Sprint ensemble."""
    return CodeSprint()


def create_research_council() -> ResearchCouncil:
    """Create a Research Council ensemble."""
    return ResearchCouncil()


def create_document_assembly() -> DocumentAssembly:
    """Create a Document Assembly ensemble."""
    return DocumentAssembly()


def create_requirements_workshop() -> RequirementsWorkshop:
    """Create a Requirements Workshop ensemble."""
    return RequirementsWorkshop()
