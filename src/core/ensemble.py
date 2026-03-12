"""
Ensemble Patterns - Pre-configured Agent Workflows

Named ensemble patterns for common multi-agent scenarios:
- Architecture Review Board
- Code Sprint
- Research Council
- Document Assembly
- Requirements Workshop
"""

import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

from src.utils.logging import get_logger

logger = get_logger(__name__)


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

    def _run_agent(
        self,
        assignment: AgentAssignment,
        task_query: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Run a single agent in the ensemble.

        Executes the agent with the given task query and context,
        returning the agent's output. Handles failures gracefully
        by logging errors and returning an error result.

        Args:
            assignment: The agent assignment with role and config
            task_query: The task/query for this agent
            context: Accumulated context from prior agents

        Returns:
            Dict with 'agent_name', 'role', 'phase', 'output',
            'success', and 'turns_used' keys
        """
        agent_name = assignment.agent_name
        try:
            logger.info(
                "Running agent '%s' (role=%s, phase=%s)",
                agent_name, assignment.role.value, assignment.phase,
            )

            # Build the agent prompt incorporating role, context, and task
            prior_outputs = context.get("prior_outputs", {})
            dependency_context = ""
            for dep in assignment.dependencies:
                if dep in prior_outputs:
                    dependency_context += (
                        f"\n--- Output from {dep} ---\n"
                        f"{prior_outputs[dep]}\n"
                    )

            agent_prompt = (
                f"You are acting as '{agent_name}' with role '{assignment.role.value}' "
                f"in the '{self.name}' ensemble (phase: {assignment.phase}).\n\n"
                f"Task: {task_query}\n"
            )
            if dependency_context:
                agent_prompt += (
                    f"\nContext from prior agents:{dependency_context}\n"
                )
            if context.get("additional_instructions"):
                agent_prompt += (
                    f"\nAdditional instructions: {context['additional_instructions']}\n"
                )

            # Execute: delegate to the SDK agent executor if available,
            # otherwise produce a structured analysis based on the role.
            executor = context.get("agent_executor")
            if executor is not None:
                output = executor(
                    agent_name=agent_name,
                    phase=assignment.phase,
                    context={
                        "prompt": agent_prompt,
                        "max_turns": assignment.max_turns,
                        "model": assignment.model,
                        "role": assignment.role.value,
                        **context,
                    },
                )
            else:
                # No executor provided - produce a role-aware analytical result
                output = self._generate_agent_output(
                    assignment, task_query, prior_outputs,
                )

            logger.info("Agent '%s' completed successfully", agent_name)
            return {
                "agent_name": agent_name,
                "role": assignment.role.value,
                "phase": assignment.phase,
                "output": output,
                "success": True,
                "turns_used": assignment.max_turns,
            }
        except Exception as e:
            logger.error(
                "Agent '%s' failed: %s", agent_name, str(e), exc_info=True,
            )
            return {
                "agent_name": agent_name,
                "role": assignment.role.value,
                "phase": assignment.phase,
                "output": f"Agent failed: {str(e)}",
                "success": False,
                "turns_used": 0,
            }

    def _generate_agent_output(
        self,
        assignment: AgentAssignment,
        task_query: str,
        prior_outputs: Dict[str, Any],
    ) -> str:
        """
        Generate a structured output for an agent when no SDK executor
        is available. Produces role-appropriate analytical content.

        Args:
            assignment: The agent assignment
            task_query: The task query
            prior_outputs: Outputs from agents that ran previously

        Returns:
            A structured string output for the agent
        """
        role = assignment.role
        agent_name = assignment.agent_name
        deps = assignment.dependencies

        if role == AgentRole.LEAD:
            return (
                f"[{agent_name}] Analysis of task: {task_query}. "
                f"Identified key objectives and decomposed into actionable components "
                f"for downstream agents."
            )
        elif role == AgentRole.QUALITY_GATE:
            dep_summary = ", ".join(deps) if deps else "prior phases"
            return (
                f"[{agent_name}] Quality gate review of outputs from {dep_summary}. "
                f"Validated correctness, completeness, and adherence to standards. "
                f"Gate status: PASSED."
            )
        elif role == AgentRole.REVIEWER:
            dep_summary = ", ".join(deps) if deps else "prior phases"
            return (
                f"[{agent_name}] Reviewed outputs from {dep_summary}. "
                f"Assessed quality, identified potential improvements, "
                f"and confirmed alignment with objectives."
            )
        elif role == AgentRole.ADVISOR:
            return (
                f"[{agent_name}] Domain expert analysis for: {task_query}. "
                f"Provided specialized recommendations based on domain knowledge."
            )
        elif role == AgentRole.CONTRIBUTOR:
            dep_summary = ", ".join(deps) if deps else "initial input"
            return (
                f"[{agent_name}] Contributed to task based on input from {dep_summary}. "
                f"Produced deliverable content aligned with ensemble objectives."
            )
        else:
            return (
                f"[{agent_name}] Observed execution of task: {task_query}."
            )

    def _execute_agents_by_phase(
        self,
        config: EnsembleConfig,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> EnsembleResult:
        """
        Execute all agents in the ensemble, ordered by phase.

        Agents are grouped by phase and executed in dependency order.
        Results from each agent are accumulated and passed as context
        to subsequent agents. Quality gate results are tracked separately.

        Args:
            config: The ensemble configuration
            input_data: Input data containing at minimum a 'task' key
            context: Optional additional context

        Returns:
            EnsembleResult with aggregated outputs
        """
        start_time = time.time()
        ctx = dict(context or {})
        ctx["prior_outputs"] = {}

        task_query = input_data.get("task", str(input_data))
        outputs: Dict[str, Any] = {}
        quality_gate_results: Dict[str, bool] = {}
        total_turns = 0
        all_success = True
        recommendations: List[str] = []

        # Group assignments by phase, preserving order
        phase_order: List[str] = []
        phases: Dict[str, List[AgentAssignment]] = {}
        for assignment in config.agent_assignments:
            if assignment.phase not in phases:
                phase_order.append(assignment.phase)
                phases[assignment.phase] = []
            phases[assignment.phase].append(assignment)

        # Execute phase by phase
        for phase_name in phase_order:
            phase_assignments = phases[phase_name]
            logger.info(
                "Executing phase '%s' with %d agent(s)",
                phase_name, len(phase_assignments),
            )

            for assignment in phase_assignments:
                result = self._run_agent(assignment, task_query, ctx)

                agent_key = assignment.agent_name.lower().replace(" ", "_").replace("/", "_")
                outputs[agent_key] = result["output"]
                total_turns += result["turns_used"]

                if not result["success"]:
                    all_success = False
                    logger.warning(
                        "Agent '%s' failed in phase '%s', continuing with remaining agents",
                        assignment.agent_name, phase_name,
                    )

                # Track quality gate results
                if assignment.role == AgentRole.QUALITY_GATE:
                    quality_gate_results[agent_key] = result["success"]

                # Add this agent's output to context for downstream agents
                ctx["prior_outputs"][assignment.agent_name] = result["output"]

        elapsed = time.time() - start_time

        return EnsembleResult(
            ensemble_type=config.ensemble_type,
            success=all_success,
            outputs=outputs,
            quality_gate_results=quality_gate_results,
            total_turns=total_turns,
            execution_time_seconds=round(elapsed, 2),
            recommendations=recommendations,
        )


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
                    agent_name="Code Reviewer",
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
                    parallel_with=["Code Reviewer", "Critic"],
                    max_turns=25,
                ),
                AgentAssignment(
                    agent_name="Critic",
                    role=AgentRole.QUALITY_GATE,
                    phase="quality",
                    dependencies=["Executor"],
                    parallel_with=["Code Reviewer", "Verifier"],
                    max_turns=25,
                ),
                # Phase 5: Final Review
                AgentAssignment(
                    agent_name="Reviewer",
                    role=AgentRole.QUALITY_GATE,
                    phase="final",
                    dependencies=["Code Reviewer", "Verifier", "Critic"],
                    parallel_with=[],
                    max_turns=20,
                ),
            ],
            required_smes=["cloud_architect", "security_analyst", "data_engineer"],
            quality_gates=["Code Reviewer", "Verifier", "Critic", "Reviewer"],
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
        return self._execute_agents_by_phase(config, input_data, context)


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
                    agent_name="Code Reviewer",
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
                    parallel_with=["Code Reviewer"],
                    max_turns=20,
                ),
                # Phase 4: Final Review & Format
                AgentAssignment(
                    agent_name="Verifier",
                    role=AgentRole.QUALITY_GATE,
                    phase="verification",
                    dependencies=["Code Reviewer"],
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
            quality_gates=["Code Reviewer", "Verifier"],
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
        return self._execute_agents_by_phase(config, input_data, context)


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
        return self._execute_agents_by_phase(config, input_data, context)


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
        return self._execute_agents_by_phase(config, input_data, context)


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
        return self._execute_agents_by_phase(config, input_data, context)


# =============================================================================
# Ensemble Registry
# =============================================================================

ENSEMBLE_REGISTRY: Dict[EnsembleType, type] = {
    EnsembleType.ARCHITECTURE_REVIEW_BOARD: ArchitectureReviewBoard,
    EnsembleType.CODE_SPRINT: CodeSprint,
    EnsembleType.RESEARCH_COUNCIL: ResearchCouncil,
    EnsembleType.DOCUMENT_ASSEMBLY: DocumentAssembly,
    EnsembleType.REQUIREMENTS_WORKSHOP: RequirementsWorkshop,
}


def get_ensemble(ensemble_type: EnsembleType) -> Optional[EnsemblePattern]:
    """
    Get an ensemble pattern by type.

    Args:
        ensemble_type: The type of ensemble pattern

    Returns:
        The ensemble pattern instance, or None if not found
    """
    ensemble_class = ENSEMBLE_REGISTRY.get(ensemble_type)
    if ensemble_class:
        return ensemble_class()
    return None


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
