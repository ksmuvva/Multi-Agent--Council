"""
Planner Subagent

Creates sequenced execution plans with agent assignments,
dependencies, and parallelization opportunities.
"""

import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from src.core.react import ReactLoop

from src.schemas.planner import (
    ExecutionPlan,
    ExecutionStep,
    AgentAssignment,
    ParallelGroup,
    StepStatus,
)
from src.schemas.analyst import TaskIntelligenceReport
from src.utils.logging import get_agent_logger, AgentLogContext
from src.utils.events import emit_agent_started, emit_agent_completed, emit_error


class PlannerAgent:
    """
    The Planner creates sequenced execution plans.

    Key responsibilities:
    - Create numbered execution steps
    - Assign agents to each step
    - Define dependencies between steps
    - Identify parallel execution opportunities
    - Estimate complexity per step
    """

    def __init__(
        self,
        system_prompt_path: str = "config/agents/planner/CLAUDE.md",
        model: str = "claude-3-5-sonnet-20241022",
        max_turns: int = 30,
    ):
        """
        Initialize the Planner agent.

        Args:
            system_prompt_path: Path to system prompt file
            model: Model to use for planning
            max_turns: Maximum conversation turns
        """
        self.system_prompt_path = system_prompt_path
        self.model = model
        self.max_turns = max_turns
        self.system_prompt = self._load_system_prompt()
        self.logger = get_agent_logger("planner")

        # Agent capabilities mapping
        self.agent_capabilities = {
            "Task Analyst": ["analysis", "decomposition", "requirements"],
            "Planner": ["planning", "sequencing", "coordination"],
            "Clarifier": ["questions", "clarification", "requirements gathering"],
            "Researcher": ["research", "evidence", "documentation", "web search"],
            "Executor": ["generation", "creation", "implementation", "coding"],
            "Code Reviewer": ["review", "security", "performance", "style"],
            "Formatter": ["formatting", "presentation", "documentation"],
            "Verifier": ["verification", "fact-checking", "validation"],
            "Critic": ["critique", "adversarial", "quality assurance"],
            "Reviewer": ["review", "quality gate", "final approval"],
        }

    def create_plan(
        self,
        analyst_report: TaskIntelligenceReport,
        sme_selections: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
        mode: str = "react",
    ) -> ExecutionPlan:
        """
        Create an execution plan from the analyst's report.

        Args:
            analyst_report: The TaskIntelligenceReport from the Analyst
            sme_selections: Optional list of selected SME personas
            context: Additional context (tier, previous plans, etc.)

        Returns:
            ExecutionPlan with sequenced steps and agent assignments
        """
        self.logger.info(
            "planner_started",
            task_description_length=len(analyst_report.literal_request),
        )
        emit_agent_started("planner", phase="planning")

        try:
            if mode == "react":
                return self._react_create_plan(analyst_report, sme_selections, context)

            # Generate steps based on analyst report
            steps = self._generate_steps(analyst_report, sme_selections, context)

            # Determine approach based on analyst report
            approach = analyst_report.recommended_approach
            self.logger.debug("approach_selected", approach=approach)

            # Log step count after generation
            self.logger.debug("steps_generated", step_count=len(steps))

            # Identify parallel groups
            parallel_groups = self._identify_parallel_groups(steps)

            # Determine critical path
            critical_path = self._calculate_critical_path(steps)

            # Estimate duration
            estimated_duration = self._estimate_duration(steps, parallel_groups)

            # Determine required SMEs
            required_smes = sme_selections or self._determine_required_smes(analyst_report)

            # Identify risk factors
            risk_factors = self._identify_risks(analyst_report, steps)

            # Create contingency plans
            contingency_plans = self._create_contingency_plans(risk_factors)

            # Build summary
            task_summary = self._build_summary(analyst_report, len(steps))

            plan = ExecutionPlan(
                task_summary=task_summary,
                total_steps=len(steps),
                steps=steps,
                parallel_groups=parallel_groups,
                critical_path=critical_path,
                estimated_duration_minutes=estimated_duration,
                required_sme_personas=required_smes,
                risk_factors=risk_factors,
                contingency_plans=contingency_plans,
            )

            self.logger.info(
                "planner_completed",
                total_steps=len(steps),
                parallel_groups=len(parallel_groups),
                estimated_duration_minutes=estimated_duration,
                risk_count=len(risk_factors),
            )
            emit_agent_completed(
                "planner",
                output_summary=f"steps={len(steps)} parallel_groups={len(parallel_groups)} duration={estimated_duration}min",
            )

            return plan

        except Exception as e:
            self.logger.error("planner_failed", error=str(e), exc_info=True)
            emit_error("planner", error_message=str(e), error_type=type(e).__name__)
            raise

    # ========================================================================
    # ReAct Mode
    # ========================================================================

    def _react_create_plan(
        self,
        analyst_report: TaskIntelligenceReport,
        sme_selections: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> ExecutionPlan:
        """Run planning using ReAct loop."""
        react_instruction = (
            "You are the Planner. Create a step-by-step execution plan with agent "
            "assignments, parallelization groups, critical path, SME requirements, "
            "duration estimates, and risk factors. Return an ExecutionPlan JSON."
        )
        system_prompt = f"{self.system_prompt}\n\n{react_instruction}"

        task_input = f"Analyst report:\n{analyst_report.model_dump_json()}"
        if sme_selections:
            task_input += f"\n\nSME selections:\n{json.dumps(sme_selections)}"
        if context:
            task_input += f"\n\nContext:\n{json.dumps(context, default=str)}"

        loop = ReactLoop(
            agent_name="planner",
            system_prompt=system_prompt,
            allowed_tools=["Read", "Glob"],
            output_schema=ExecutionPlan,
            model=self.model,
            max_turns=self.max_turns,
        )

        result = loop.run(task_input)

        if result and "output" in result and isinstance(result["output"], ExecutionPlan):
            return result["output"]

        # Fallback to procedural logic
        return self.create_plan(analyst_report, sme_selections, context, mode="local")

    # ========================================================================
    # Plan Generation Methods
    # ========================================================================

    def _generate_steps(
        self,
        analyst_report: TaskIntelligenceReport,
        sme_selections: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ExecutionStep]:
        """Generate execution steps from the analyst report."""
        steps = []
        step_number = 1

        # Step 1: Task Intelligence (always first if not already done)
        # In most cases, Analyst has already run, so we start with planning

        # Step 2: Clarification (if needed)
        if analyst_report.missing_info:
            critical_missing = [
                m for m in analyst_report.missing_info
                if m.severity.value == "critical"
            ]
            if critical_missing:
                steps.append(ExecutionStep(
                    step_number=step_number,
                    description=f"Clarify missing requirements ({len(critical_missing)} critical)",
                    agent_assignments=[
                        AgentAssignment(
                            agent_name="Clarifier",
                            role="Formulate clarification questions",
                            reason="Critical information is missing"
                        )
                    ],
                    dependencies=[],
                    can_parallelize=False,
                    estimated_complexity="low",
                    expected_outputs=["User responses to clarification questions"],
                    status=StepStatus.PENDING,
                ))
                step_number += 1

        # Steps based on sub-tasks
        for i, sub_task in enumerate(analyst_report.sub_tasks, 1):
            # Determine agent for this task
            agents = self._assign_agents_to_task(sub_task, sme_selections)

            # Find dependencies
            dependencies = []
            if i > 0:
                dependencies.append(step_number - 1)

            steps.append(ExecutionStep(
                step_number=step_number,
                description=sub_task.description,
                agent_assignments=agents,
                dependencies=dependencies,
                can_parallelize=self._can_parallelize(sub_task, steps),
                estimated_complexity=sub_task.estimated_complexity,
                expected_outputs=[f"Completed: {sub_task.description}"],
                status=StepStatus.PENDING,
            ))
            step_number += 1

        # Add review steps based on tier/modality
        review_steps = self._add_review_steps(analyst_report, step_number, sme_selections)
        steps.extend(review_steps)

        return steps

    def _assign_agents_to_task(
        self,
        sub_task: "SubTask",
        sme_selections: Optional[List[str]] = None,
    ) -> List[AgentAssignment]:
        """Assign appropriate agents to a task."""
        description_lower = sub_task.description.lower()
        assignments = []

        # Determine primary agent based on task type
        if "implement" in description_lower or "generate" in description_lower or "create" in description_lower:
            assignments.append(AgentAssignment(
                agent_name="Executor",
                role="Generate solution",
                reason="Task involves creating or implementing"
            ))
        elif "analyze" in description_lower or "understand" in description_lower:
            assignments.append(AgentAssignment(
                agent_name="Task Analyst",
                role="Deep analysis",
                reason="Task requires analysis"
            ))
        elif "design" in description_lower:
            # Check if SME should be involved
            if sme_selections:
                # Dynamically select the matching SME from the provided selections
                architect_sme = next(
                    (s for s in sme_selections if "architect" in s.lower()),
                    None,
                )
                if architect_sme:
                    assignments.append(AgentAssignment(
                        agent_name=architect_sme,
                        role="Domain-specific design",
                        reason="SME has domain expertise",
                    ))
            assignments.append(AgentAssignment(
                agent_name="Planner",
                role="Create design",
                reason="Task involves design planning"
            ))
        elif "test" in description_lower:
            assignments.append(AgentAssignment(
                agent_name="Test Engineer SME",
                role="Test strategy",
                reason="Domain expertise in testing"
            ))
            assignments.append(AgentAssignment(
                agent_name="Executor",
                role="Generate tests",
                reason="Implement test code"
            ))
        else:
            # Default to Executor
            assignments.append(AgentAssignment(
                agent_name="Executor",
                role="Execute task",
                reason="Default execution agent"
            ))

        return assignments

    def _can_parallelize(
        self,
        sub_task: "SubTask",
        existing_steps: List[ExecutionStep]
    ) -> bool:
        """Determine if a task can run in parallel with others."""
        # Tasks with no dependencies on previous steps could potentially parallelize
        if not sub_task.dependencies:
            # Check if any existing steps also have no dependencies
            independent_steps = [s for s in existing_steps if not s.dependencies]
            return len(independent_steps) > 0
        return False

    def _add_review_steps(
        self,
        analyst_report: TaskIntelligenceReport,
        start_step: int,
        sme_selections: Optional[List[str]] = None,
    ) -> List[ExecutionStep]:
        """Add review and verification steps."""
        steps = []
        step_number = start_step

        modality = analyst_report.modality.value

        # Code-specific reviews
        if modality == "code":
            steps.append(ExecutionStep(
                step_number=step_number,
                description="Code Review (security, performance, style)",
                agent_assignments=[
                    AgentAssignment(
                        agent_name="Code Reviewer",
                        role="Review code quality",
                        reason="Code output requires review"
                    )
                ],
                dependencies=[step_number - 1] if step_number > 1 else [],
                can_parallelize=False,
                estimated_complexity="medium",
                expected_outputs=["CodeReviewReport"],
                status=StepStatus.PENDING,
            ))
            step_number += 1

        # Verification step (always included for Tier 2+)
        steps.append(ExecutionStep(
            step_number=step_number,
            description="Verify factual claims and detect hallucinations",
            agent_assignments=[
                AgentAssignment(
                    agent_name="Verifier",
                    role="Fact-check output",
                    reason="Ensure accuracy"
                )
            ],
            dependencies=[step_number - 1] if step_number > 1 else [],
            can_parallelize=False,
            estimated_complexity="medium",
            expected_outputs=["VerificationReport"],
            status=StepStatus.PENDING,
        ))
        step_number += 1

        # Critic step (for Tier 2+)
        steps.append(ExecutionStep(
            step_number=step_number,
            description="Adversarial critique and quality assessment",
            agent_assignments=[
                AgentAssignment(
                    agent_name="Critic",
                    role="Attack the solution",
                    reason="Find weaknesses and improvements"
                )
            ],
            dependencies=[step_number - 2] if step_number > 2 else [],  # Parallel with Verifier
            can_parallelize=True,
            parallel_group_id="review_group",
            estimated_complexity="medium",
            expected_outputs=["CritiqueReport"],
            status=StepStatus.PENDING,
        ))
        step_number += 1

        # Final review step
        steps.append(ExecutionStep(
            step_number=step_number,
            description="Final quality gate and approval",
            agent_assignments=[
                AgentAssignment(
                    agent_name="Reviewer",
                    role="Final quality check",
                    reason="Quality gate before output"
                )
            ],
            dependencies=[step_number - 1] if step_number > 1 else [],
            can_parallelize=False,
            estimated_complexity="low",
            expected_outputs=["ReviewVerdict"],
            status=StepStatus.PENDING,
        ))
        step_number += 1

        # Formatter step (always last)
        steps.append(ExecutionStep(
            step_number=step_number,
            description="Format and present output",
            agent_assignments=[
                AgentAssignment(
                    agent_name="Formatter",
                    role="Present results",
                    reason="User-facing output"
                )
            ],
            dependencies=[step_number - 1] if step_number > 1 else [],
            can_parallelize=False,
            estimated_complexity="low",
            expected_outputs=["Formatted output"],
            status=StepStatus.PENDING,
        ))

        return steps

    # ========================================================================
    # Analysis Methods
    # ========================================================================

    def _identify_parallel_groups(self, steps: List[ExecutionStep]) -> List[ParallelGroup]:
        """Identify groups of steps that can run in parallel."""
        groups = []
        group_map = {}

        for step in steps:
            if step.can_parallelize and step.parallel_group_id:
                if step.parallel_group_id not in group_map:
                    group_map[step.parallel_group_id] = {
                        "group_id": step.parallel_group_id,
                        "steps": [],
                        "description": f"Parallel: {step.description}",
                    }
                group_map[step.parallel_group_id]["steps"].append(step.step_number)

        for group_data in group_map.values():
            groups.append(ParallelGroup(**group_data))

        return groups

    def _calculate_critical_path(self, steps: List[ExecutionStep]) -> List[int]:
        """Calculate the critical path (longest dependency chain)."""
        if not steps:
            return []

        # Build dependency graph
        step_map = {s.step_number: s for s in steps}
        visited = set()
        path_lengths = {s.step_number: 0 for s in steps}

        # Visit steps in order
        for step in steps:
            self._visit_step(step.step_number, step_map, visited, path_lengths)

        # Find longest path
        max_length = max(path_lengths.values())
        critical_steps = [
            step_num for step_num, length in path_lengths.items()
            if length == max_length
        ]

        # Sort to maintain order
        critical_steps.sort()
        return critical_steps

    def _visit_step(
        self,
        step_num: int,
        step_map: Dict[int, ExecutionStep],
        visited: set,
        path_lengths: Dict[int, int],
    ) -> int:
        """Recursively visit steps to calculate path lengths."""
        if step_num in visited:
            return path_lengths[step_num]

        step = step_map.get(step_num)
        if not step:
            return 0

        visited.add(step_num)

        # Calculate max length through dependencies
        max_dep_length = 0
        for dep in step.dependencies:
            dep_length = self._visit_step(dep, step_map, visited, path_lengths)
            max_dep_length = max(max_dep_length, dep_length)

        path_lengths[step_num] = max_dep_length + 1
        return path_lengths[step_num]

    def _estimate_duration(
        self,
        steps: List[ExecutionStep],
        parallel_groups: List[ParallelGroup]
    ) -> Optional[int]:
        """Estimate total execution time in minutes."""
        # Rough estimates per complexity level
        complexity_minutes = {
            "low": 2,
            "medium": 5,
            "high": 10,
        }

        total = 0
        for step in steps:
            total += complexity_minutes.get(step.estimated_complexity, 5)

        # Subtract time saved by parallelization
        parallel_savings = 0
        for group in parallel_groups:
            if len(group.steps) > 1:
                # Assume 30% time savings for parallel steps
                group_duration = sum(
                    complexity_minutes.get(
                        next(s.estimated_complexity for s in steps if s.step_number == n),
                        5
                    )
                    for n in group.steps
                )
                parallel_savings += int(group_duration * 0.3)

        return max(1, total - parallel_savings)

    def _determine_required_smes(
        self,
        analyst_report: TaskIntelligenceReport
    ) -> List[str]:
        """Determine which SME personas are required."""
        required = []
        request_lower = analyst_report.literal_request.lower()

        # SME trigger keywords
        sme_triggers = {
            "IAM Architect": ["sailpoint", "cyberark", "identity", "rbac"],
            "Cloud Architect": ["azure", "aws", "cloud", "kubernetes"],
            "Security Analyst": ["security", "threat", "vulnerability"],
            "Data Engineer": ["database", "etl", "pipeline", "data"],
            "AI/ML Engineer": ["ml", "ai", "rag", "llm"],
            "Test Engineer": ["test", "testing", "qa"],
            "Business Analyst": ["requirements", "process", "workflow"],
            "Technical Writer": ["document", "docs", "readme"],
            "DevOps Engineer": ["deploy", "ci/cd", "docker"],
            "Frontend Developer": ["ui", "frontend", "streamlit"],
        }

        for sme, triggers in sme_triggers.items():
            if any(trigger in request_lower for trigger in triggers):
                required.append(sme)

        return required[:3]  # Max 3 SMEs

    def _identify_risks(
        self,
        analyst_report: TaskIntelligenceReport,
        steps: List[ExecutionStep]
    ) -> List[str]:
        """Identify potential risks and blockers."""
        risks = []

        # Check for critical missing info
        critical_missing = [
            m for m in analyst_report.missing_info
            if m.severity.value == "critical"
        ]
        if critical_missing:
            risks.append(
                f"Critical requirements missing: {', '.join(m.requirement for m in critical_missing)}"
            )

        # Check for complex dependencies
        complex_steps = [s for s in steps if s.estimated_complexity == "high"]
        if len(complex_steps) > 3:
            risks.append("Multiple high-complexity steps may extend execution time")

        # Check for escalation risk
        if analyst_report.escalation_needed:
            risks.append("May require escalation to higher tier during execution")

        return risks

    def _create_contingency_plans(self, risk_factors: List[str]) -> List[str]:
        """Create contingency plans for identified risks."""
        plans = []

        for risk in risk_factors:
            if "missing" in risk.lower():
                plans.append("If clarification cannot be obtained, proceed with documented assumptions")
            elif "complexity" in risk.lower():
                plans.append("Consider splitting complex steps or allocating more time")
            elif "escalation" in risk.lower():
                plans.append("Have Council and SME personas ready for activation")

        if not plans:
            plans.append("Standard execution with no special contingencies")

        return plans

    def _build_summary(
        self,
        analyst_report: TaskIntelligenceReport,
        total_steps: int
    ) -> str:
        """Build a task summary for the plan."""
        return (
            f"Execute {total_steps} steps to: {analyst_report.inferred_intent}. "
            f"Modality: {analyst_report.modality.value}. "
            f"Complexity: {len(analyst_report.sub_tasks)} sub-tasks identified."
        )

    def _load_system_prompt(self) -> str:
        """Load the system prompt from file."""
        try:
            with open(self.system_prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            return "You are the Planner. Create sequenced execution plans with agent assignments."


# =============================================================================
# Convenience Functions
# =============================================================================

def create_planner(
    system_prompt_path: str = "config/agents/planner/CLAUDE.md",
    model: str = "claude-3-5-sonnet-20241022",
) -> PlannerAgent:
    """Create a configured Planner agent."""
    return PlannerAgent(
        system_prompt_path=system_prompt_path,
        model=model,
    )
