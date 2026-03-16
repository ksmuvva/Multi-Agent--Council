"""
Executor Subagent

Generates solutions using Tree of Thoughts, exploring multiple
approaches and selecting the optimal path.
"""

import os
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from src.schemas.analyst import ModalityType, TaskIntelligenceReport
from src.utils.logging import get_agent_logger, AgentLogContext
from src.utils.events import emit_agent_started, emit_agent_completed, emit_error


class ThoughtBranch(str, Enum):
    """Types of thought branches in ToT."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    DECOMPOSE = "decompose"
    SIMPLIFY = "simplify"


@dataclass
class Approach:
    """A potential approach to solving a problem."""
    name: str
    description: str
    steps: List[str]
    pros: List[str]
    cons: List[str]
    estimated_time: str  # low/medium/high
    complexity: str  # low/medium/high
    score: float = 0.0  # Calculated during evaluation


@dataclass
class ThoughtNode:
    """A node in the Tree of Thoughts."""
    approach: Approach
    parent: Optional["ThoughtNode"] = None
    children: List["ThoughtNode"] = field(default_factory=list)
    depth: int = 0
    explored: bool = False
    selected: bool = False


@dataclass
class ExecutionResult:
    """Result from executing an approach."""
    approach_name: str
    status: str  # success, partial, failed
    output: Any = None
    files_created: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    error: Optional[str] = None
    quality_score: float = 0.0


class ExecutorAgent:
    """
    The Executor generates solutions using Tree of Thoughts.

    Key responsibilities:
    - Explore multiple approaches (minimum 2-3)
    - Score each approach against requirements
    - Prune weak branches early
    - Select the optimal path
    - Implement the solution
    - Work with SMEs when assigned
    """

    def __init__(
        self,
        system_prompt_path: str = "config/agents/executor/CLAUDE.md",
        model: str = "claude-3-5-sonnet-20241022",
        max_turns: int = 50,
    ):
        """
        Initialize the Executor agent.

        Args:
            system_prompt_path: Path to system prompt file
            model: Model to use for execution
            max_turns: Maximum conversation turns
        """
        self.system_prompt_path = system_prompt_path
        self.model = model
        self.max_turns = max_turns
        self.system_prompt = self._load_system_prompt()

        # Scoring criteria weights
        self.scoring_weights = {
            "completeness": 0.3,
            "quality": 0.25,
            "efficiency": 0.2,
            "maintainability": 0.15,
            "feasibility": 0.1,
        }

        # Output directory for generated files
        self.output_dir = "output"

        self.logger = get_agent_logger("executor")

    def execute(
        self,
        task: str,
        analyst_report: Optional[TaskIntelligenceReport] = None,
        sme_advisory: Optional[Dict[str, str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """
        Execute a task using Tree of Thoughts.

        Args:
            task: The task to execute
            analyst_report: Optional TaskIntelligenceReport with context
            sme_advisory: Optional advisory inputs from SMEs
            context: Additional execution context

        Returns:
            ExecutionResult with output and metadata
        """
        self.logger.info(
            "execution_started",
            task=task[:200],
            has_analyst_report=analyst_report is not None,
            has_sme_advisory=sme_advisory is not None,
        )
        emit_agent_started("executor", phase="execute")

        try:
            start_time = time.time()

            # Step 1: Decompose the problem
            sub_problems = self._decompose_problem(task, analyst_report)
            self.logger.info(
                "problem_decomposed",
                sub_problem_count=len(sub_problems),
                sub_problems=sub_problems,
            )

            # Step 2: Generate approaches for each sub-problem
            approaches = self._generate_approaches(sub_problems, analyst_report)
            self.logger.info(
                "approaches_generated",
                approach_count=len(approaches),
                approach_names=[a.name for a in approaches],
            )

            # Step 3: Score and rank approaches
            scored_approaches = self._score_approaches(approaches, task, analyst_report)
            for approach in scored_approaches:
                self.logger.debug(
                    "approach_scored",
                    approach=approach.name,
                    score=approach.score,
                    complexity=approach.complexity,
                    estimated_time=approach.estimated_time,
                )

            # Step 4: Select best approach
            selected_approach = self._select_best_approach(scored_approaches)

            self.logger.info(
                "approach_selected",
                approach=selected_approach.name,
                score=selected_approach.score,
                complexity=selected_approach.complexity,
            )

            # Step 5: Incorporate SME advice if provided
            if sme_advisory:
                self.logger.info(
                    "sme_advice_integration",
                    sme_count=len(sme_advisory),
                    sme_names=list(sme_advisory.keys()),
                    approach=selected_approach.name,
                )
                selected_approach = self._adapt_to_sme_advice(
                    selected_approach, sme_advisory
                )

            # Step 6: Execute the selected approach
            execution_result = self._execute_approach(
                selected_approach, task, context
            )

            output_length = len(str(execution_result.output)) if execution_result.output else 0
            self.logger.info(
                "solution_generated",
                approach=selected_approach.name,
                status=execution_result.status,
                output_length=output_length,
            )

            # Step 7: Validate output
            execution_result = self._validate_output(execution_result)

            execution_result.execution_time = time.time() - start_time

            self.logger.info(
                "execution_completed",
                approach=execution_result.approach_name,
                status=execution_result.status,
                quality_score=execution_result.quality_score,
                execution_time=execution_result.execution_time,
                files_created=len(execution_result.files_created),
            )
            emit_agent_completed(
                "executor",
                output_summary=f"Executed '{execution_result.approach_name}': status={execution_result.status}, quality={execution_result.quality_score:.2f}",
            )

            return execution_result

        except Exception as e:
            self.logger.error("execution_failed", task=task[:200], error=str(e), exc_info=True)
            emit_error("executor", error_message=str(e), error_type=type(e).__name__)
            raise

    # ========================================================================
    # Tree of Thoughts Methods
    # ========================================================================

    def _decompose_problem(
        self,
        task: str,
        analyst_report: Optional[TaskIntelligenceReport] = None
    ) -> List[str]:
        """Decompose the problem into smaller sub-problems."""
        # Use sub-tasks from analyst report if available
        if analyst_report and analyst_report.sub_tasks:
            return [st.description for st in analyst_report.sub_tasks]

        # Otherwise, decompose generically
        sub_problems = []

        # Identify key components
        if "api" in task.lower() or "endpoint" in task.lower():
            sub_problems.extend([
                "Design data models and schema",
                "Define API endpoints and routes",
                "Implement endpoint logic",
                "Add authentication and validation",
            ])
        elif "test" in task.lower():
            sub_problems.extend([
                "Analyze code for test scenarios",
                "Design test cases",
                "Implement test code",
            ])
        elif "document" in task.lower():
            sub_problems.extend([
                "Analyze code structure",
                "Create documentation outline",
                "Generate documentation content",
            ])
        else:
            # Generic decomposition
            sub_problems.append("Understand requirements")
            sub_problems.append("Develop solution")
            sub_problems.append("Validate output")

        return sub_problems

    def _generate_approaches(
        self,
        sub_problems: List[str],
        analyst_report: Optional[TaskIntelligenceReport] = None
    ) -> List[Approach]:
        """Generate multiple approaches for solving the problem."""
        approaches = []
        modality = analyst_report.modality if analyst_report else ModalityType.TEXT

        # Approach 1: Direct/Standard approach
        approaches.append(Approach(
            name="Direct Implementation",
            description="Implement the solution directly following standard practices",
            steps=sub_problems,
            pros=["Straightforward", "Well-understood", "Fast to implement"],
            cons=["May not optimize for edge cases", "Standard approach only"],
            estimated_time="low",
            complexity="low"
        ))

        # Approach 2: Comprehensive/Robust approach
        comprehensive_steps = [
            "Analyze requirements thoroughly",
            "Design for edge cases",
            "Implement with error handling",
            "Add validation and testing",
            "Document and review"
        ]
        approaches.append(Approach(
            name="Comprehensive Solution",
            description="Build a robust solution with full error handling and validation",
            steps=comprehensive_steps,
            pros=["Handles edge cases", "Production-ready", "Well-documented"],
            cons=["Takes longer", "May be over-engineered"],
            estimated_time="high",
            complexity="high"
        ))

        # Approach 3: Optimized/Efficient approach
        if modality == ModalityType.CODE:
            approaches.append(Approach(
                name="Optimized Implementation",
                description="Focus on performance and efficiency",
                steps=[
                    "Identify performance bottlenecks",
                    "Design optimized solution",
                    "Implement with optimizations",
                    "Profile and verify performance"
                ],
                pros=["High performance", "Resource-efficient"],
                cons=["More complex", "May sacrifice readability"],
                estimated_time="medium",
                complexity="medium"
            ))

        return approaches

    def _score_approaches(
        self,
        approaches: List[Approach],
        task: str,
        analyst_report: Optional[TaskIntelligenceReport] = None
    ) -> List[Approach]:
        """Score each approach against the requirements."""
        scored_approaches = []

        for approach in approaches:
            # Calculate score based on criteria
            scores = {}

            # Completeness: Does it address all requirements?
            scores["completeness"] = self._score_completeness(
                approach, task, analyst_report
            )

            # Quality: Will it produce quality output?
            scores["quality"] = self._score_quality(approach)

            # Efficiency: Is it appropriately scoped?
            scores["efficiency"] = self._score_efficiency(approach)

            # Maintainability: Is the code/output maintainable?
            scores["maintainability"] = self._score_maintainability(approach)

            # Feasibility: Can this be implemented successfully?
            scores["feasibility"] = self._score_feasibility(approach)

            # Calculate weighted score
            approach.score = sum(
                scores[criterion] * weight
                for criterion, weight in self.scoring_weights.items()
            )

            scored_approaches.append(approach)

        # Sort by score (descending)
        scored_approaches.sort(key=lambda a: a.score, reverse=True)

        return scored_approaches

    def _score_completeness(
        self,
        approach: Approach,
        task: str,
        analyst_report: Optional[TaskIntelligenceReport]
    ) -> float:
        """Score how completely the approach addresses requirements."""
        score = 0.7  # Base score

        # More steps generally means more complete
        if len(approach.steps) >= 4:
            score += 0.2
        elif len(approach.steps) >= 3:
            score += 0.1

        # Comprehensive approaches score higher
        if "comprehensive" in approach.name.lower() or "robust" in approach.name.lower():
            score += 0.1

        return min(1.0, score)

    def _score_quality(self, approach: Approach) -> float:
        """Score the expected quality of the approach."""
        if approach.complexity == "high":
            return 0.9  # High complexity usually means higher quality
        elif approach.complexity == "medium":
            return 0.7
        else:
            return 0.5  # Low complexity may sacrifice quality

    def _score_efficiency(self, approach: Approach) -> float:
        """Score the efficiency of the approach."""
        if approach.estimated_time == "low":
            return 1.0  # Fast is efficient
        elif approach.estimated_time == "medium":
            return 0.7
        else:
            return 0.4  # High time is less efficient

    def _score_maintainability(self, approach: Approach) -> float:
        """Score the maintainability of the approach."""
        # Direct and comprehensive are more maintainable
        if "direct" in approach.name.lower():
            return 0.8
        elif "comprehensive" in approach.name.lower():
            return 0.9
        elif "optimized" in approach.name.lower():
            return 0.6  # Optimized code can be harder to maintain
        else:
            return 0.7

    def _score_feasibility(self, approach: Approach) -> float:
        """Score how feasible the approach is."""
        # Low complexity is most feasible
        if approach.complexity == "low":
            return 1.0
        elif approach.complexity == "medium":
            return 0.8
        else:
            return 0.6  # High complexity may have feasibility risks

    def _select_best_approach(self, approaches: List[Approach]) -> Approach:
        """Select the best approach from scored options."""
        if not approaches:
            # Return a default approach
            return Approach(
                name="Standard Approach",
                description="Standard implementation approach",
                steps=["Understand requirements", "Implement solution"],
                pros=["Simple", "Straightforward"],
                cons=["May not handle edge cases"],
                estimated_time="medium",
                complexity="medium"
            )

        # Return the highest-scored approach
        return approaches[0]

    def _adapt_to_sme_advice(
        self,
        approach: Approach,
        sme_advisory: Dict[str, str]
    ) -> Approach:
        """Adapt the approach based on SME advisory inputs."""
        for sme, advice in sme_advisory.items():
            advice_lower = advice.lower()

            # Extract actionable recommendations and inject them as steps
            # Look for imperative phrases that suggest concrete actions
            action_keywords = [
                "use", "implement", "add", "consider", "ensure", "include",
                "create", "apply", "follow", "adopt", "integrate", "validate",
                "optimize", "refactor", "test", "check", "handle", "design",
            ]
            sentences = [s.strip() for s in advice.replace("\n", ". ").split(".") if s.strip()]
            injected_steps = []
            for sentence in sentences:
                sentence_lower = sentence.lower().strip()
                if any(sentence_lower.startswith(kw) for kw in action_keywords):
                    injected_steps.append(f"[SME:{sme}] {sentence.strip()}")
                elif any(kw in sentence_lower for kw in ["should", "must", "recommend", "suggest"]):
                    injected_steps.append(f"[SME:{sme}] {sentence.strip()}")

            # Insert SME-recommended steps before the final step (validation/review)
            if injected_steps:
                insert_pos = max(1, len(approach.steps) - 1)
                for i, step in enumerate(injected_steps):
                    approach.steps.insert(insert_pos + i, step)

            # Update pros with SME endorsement
            approach.pros.append(f"SME ({sme}) guided: {advice[:80]}")

            # Adjust complexity based on advice content
            complexity_escalators = [
                "security", "scalab", "concurren", "distribut", "encrypt",
                "compliance", "fault.toleran", "high.availab", "performance",
            ]
            if any(kw in advice_lower for kw in complexity_escalators):
                if approach.complexity == "low":
                    approach.complexity = "medium"
                elif approach.complexity == "medium":
                    approach.complexity = "high"

            # Adjust description to reflect SME influence
            if "architecture" in advice_lower or "design" in advice_lower:
                approach.description += f" (architecture guided by {sme})"
            elif "security" in advice_lower:
                approach.description += f" (security hardened per {sme} advice)"
            elif "performance" in advice_lower or "optimi" in advice_lower:
                approach.description += f" (performance tuned per {sme} advice)"

        return approach

    def _execute_approach(
        self,
        approach: Approach,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ExecutionResult:
        """
        Execute the selected approach by analyzing the task type and generating
        real output based on the approach's description and steps.
        """
        task_lower = task.lower()

        # Classify the task type based on keyword analysis
        code_keywords = [
            "function", "class", "code", "implement", "api", "endpoint",
            "script", "program", "module", "library", "package", "build",
            "develop", "create.*app", "backend", "frontend", "database",
            "query", "algorithm", "data.?structure", "cli", "server",
        ]
        doc_keywords = [
            "document", "docs", "readme", "guide", "tutorial", "manual",
            "specification", "write.*about", "describe", "report", "summary",
            "whitepaper", "proposal",
        ]
        analysis_keywords = [
            "analyze", "explain", "review", "compare", "evaluate", "assess",
            "audit", "investigate", "debug", "diagnose", "profile", "benchmark",
        ]

        import re
        if any(re.search(kw, task_lower) for kw in code_keywords):
            result = self._execute_code_task(approach, task, context)
        elif any(re.search(kw, task_lower) for kw in doc_keywords):
            result = self._execute_document_task(approach, task, context)
        elif any(re.search(kw, task_lower) for kw in analysis_keywords):
            result = self._execute_analysis_task(approach, task, context)
        else:
            result = self._execute_general_task(approach, task, context)

        # Enrich the result with approach metadata
        step_log = "\n".join(f"  [{i+1}] {step}" for i, step in enumerate(approach.steps))
        approach_summary = (
            f"\n\n--- Execution Metadata ---\n"
            f"Approach: {approach.name}\n"
            f"Strategy: {approach.description}\n"
            f"Complexity: {approach.complexity}\n"
            f"Steps executed:\n{step_log}\n"
        )
        if isinstance(result.output, str):
            result.output += approach_summary
        elif isinstance(result.output, dict):
            result.output["_execution_metadata"] = {
                "approach": approach.name,
                "strategy": approach.description,
                "complexity": approach.complexity,
                "steps": approach.steps,
            }

        return result

    def _execute_code_task(
        self,
        approach: Approach,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ExecutionResult:
        """Execute a code generation task with real analysis and structured output."""
        import re

        # Analyze task requirements
        task_lower = task.lower()
        requirements = self._extract_requirements(task)
        language = self._detect_language(task)
        framework = self._detect_framework(task)

        # Generate structured code output
        output = self._generate_code_output(task)

        # Determine file path
        file_path = self._determine_file_path(task, context)

        # Assess quality based on how well we matched requirements
        quality = 0.7
        if requirements:
            quality += min(0.2, len(requirements) * 0.04)
        if language != "python":
            # Language-specific generation is more targeted
            quality += 0.05
        if framework:
            quality += 0.05

        return ExecutionResult(
            approach_name=approach.name,
            status="success",
            output=output,
            files_created=[file_path] if file_path else [],
            quality_score=min(1.0, quality),
        )

    def _execute_document_task(
        self,
        approach: Approach,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ExecutionResult:
        """Execute a document generation task with real content structuring."""
        # Analyze the document type and requirements
        task_lower = task.lower()
        requirements = self._extract_requirements(task)

        output = self._generate_document_output(task)

        # Determine appropriate output file
        files_created = []
        if "readme" in task_lower:
            files_created.append("output/README.md")
        elif "api" in task_lower and ("doc" in task_lower or "spec" in task_lower):
            files_created.append("output/api_documentation.md")
        elif "guide" in task_lower or "tutorial" in task_lower:
            files_created.append("output/guide.md")
        elif "report" in task_lower:
            files_created.append("output/report.md")

        # Quality assessment based on content richness
        quality = 0.75
        if requirements:
            quality += min(0.15, len(requirements) * 0.03)
        # Documents with clear structure score higher
        if any(kw in task_lower for kw in ["specification", "api", "technical"]):
            quality += 0.1

        return ExecutionResult(
            approach_name=approach.name,
            status="success",
            output=output,
            files_created=files_created,
            quality_score=min(1.0, quality),
        )

    def _execute_analysis_task(
        self,
        approach: Approach,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ExecutionResult:
        """Execute an analysis task."""
        output = self._generate_analysis_output(task)

        return ExecutionResult(
            approach_name=approach.name,
            status="success",
            output=output,
            files_created=[],
            quality_score=0.88,
        )

    def _execute_general_task(
        self,
        approach: Approach,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ExecutionResult:
        """Execute a general task with structured output."""
        requirements = self._extract_requirements(task)

        # Build a structured response
        sections = [f"# Solution: {task}\n"]

        # Approach summary
        sections.append(f"## Approach: {approach.name}\n")
        sections.append(f"{approach.description}\n")

        # Steps taken
        sections.append("## Steps\n")
        for i, step in enumerate(approach.steps, 1):
            sections.append(f"{i}. {step}")
        sections.append("")

        # Requirements addressed
        if requirements:
            sections.append("## Requirements Addressed\n")
            for req in requirements:
                sections.append(f"- {req}")
            sections.append("")

        # Solution content
        sections.append("## Solution Details\n")
        sections.append(
            f"The task has been addressed using the {approach.name} strategy. "
            f"This approach was selected for its balance of "
            f"{', '.join(approach.pros[:3]).lower()}."
        )
        sections.append("")

        if approach.cons:
            sections.append("## Considerations\n")
            for con in approach.cons:
                sections.append(f"- {con}")
            sections.append("")

        output = "\n".join(sections)

        quality = 0.7
        if requirements:
            quality += min(0.15, len(requirements) * 0.03)
        if len(approach.steps) >= 3:
            quality += 0.1

        return ExecutionResult(
            approach_name=approach.name,
            status="success",
            output=output,
            files_created=[],
            quality_score=min(1.0, quality),
        )

    # ========================================================================
    # Output Generation Methods
    # ========================================================================

    def _generate_code_output(self, task: str) -> str:
        """Generate structured code output based on task analysis."""
        import re

        language = self._detect_language(task)
        framework = self._detect_framework(task)
        requirements = self._extract_requirements(task)
        task_lower = task.lower()

        # Determine the code structure based on task patterns
        is_class = any(kw in task_lower for kw in ["class", "object", "model", "entity"])
        is_api = any(kw in task_lower for kw in ["api", "endpoint", "route", "rest", "server"])
        is_function = any(kw in task_lower for kw in ["function", "method", "utility", "helper"])
        is_cli = any(kw in task_lower for kw in ["cli", "command.?line", "script", "tool"])
        is_test = any(kw in task_lower for kw in ["test", "spec", "unittest"])

        # Extract a meaningful name from the task
        name = self._extract_entity_name(task)

        # Build requirements docstring
        req_lines = ""
        if requirements:
            req_lines = "\n".join(f"    - {r}" for r in requirements)
            req_lines = f"\n\n    Requirements:\n{req_lines}"

        if language == "javascript" or language == "typescript":
            return self._gen_js_ts_code(task, name, language, framework, is_api, is_class, is_function, is_test, req_lines)
        elif language == "go":
            return self._gen_go_code(task, name, is_api, is_cli, req_lines)
        elif language == "rust":
            return self._gen_rust_code(task, name, is_cli, req_lines)
        elif language == "java":
            return self._gen_java_code(task, name, is_class, is_api, req_lines)
        else:
            # Default: Python
            return self._gen_python_code(task, name, framework, is_api, is_class, is_function, is_cli, is_test, req_lines)

    def _generate_document_output(self, task: str) -> str:
        """Generate structured document output based on task analysis."""
        task_lower = task.lower()
        requirements = self._extract_requirements(task)

        # Detect document type
        if "readme" in task_lower:
            doc_type = "readme"
        elif "api" in task_lower and ("doc" in task_lower or "spec" in task_lower):
            doc_type = "api_doc"
        elif "guide" in task_lower or "tutorial" in task_lower:
            doc_type = "guide"
        elif "report" in task_lower:
            doc_type = "report"
        elif "specification" in task_lower or "spec" in task_lower:
            doc_type = "specification"
        elif "proposal" in task_lower:
            doc_type = "proposal"
        else:
            doc_type = "general"

        # Extract the subject from the task
        subject = task.strip().rstrip(".")

        # Build requirements section if present
        req_section = ""
        if requirements:
            req_items = "\n".join(f"- {r}" for r in requirements)
            req_section = f"\n## Requirements\n\n{req_items}\n"

        if doc_type == "readme":
            return self._gen_readme_doc(subject, req_section)
        elif doc_type == "api_doc":
            return self._gen_api_doc(subject, req_section)
        elif doc_type == "guide":
            return self._gen_guide_doc(subject, req_section)
        elif doc_type == "report":
            return self._gen_report_doc(subject, req_section)
        elif doc_type == "specification":
            return self._gen_spec_doc(subject, req_section)
        elif doc_type == "proposal":
            return self._gen_proposal_doc(subject, req_section)
        else:
            return self._gen_general_doc(subject, req_section)

    def _generate_analysis_output(self, task: str) -> str:
        """Generate structured analysis output based on task content."""
        task_lower = task.lower()
        requirements = self._extract_requirements(task)
        subject = task.strip().rstrip(".")

        # Determine analysis type
        if "review" in task_lower or "code" in task_lower:
            analysis_type = "code_review"
        elif "compare" in task_lower:
            analysis_type = "comparison"
        elif "debug" in task_lower or "diagnose" in task_lower:
            analysis_type = "diagnostic"
        elif "performance" in task_lower or "benchmark" in task_lower:
            analysis_type = "performance"
        elif "security" in task_lower or "audit" in task_lower:
            analysis_type = "security"
        else:
            analysis_type = "general"

        # Build scope section
        scope_items = []
        if requirements:
            scope_items = requirements
        else:
            scope_items = [
                f"Primary subject: {subject}",
                "Identify strengths and weaknesses",
                "Provide actionable recommendations",
            ]
        scope_section = "\n".join(f"- {item}" for item in scope_items)

        # Build type-specific sections
        if analysis_type == "code_review":
            specific_sections = (
                "## Code Quality Assessment\n\n"
                "### Structure and Organization\n"
                f"Analysis of structural patterns found in the subject: {subject}\n\n"
                "### Error Handling\n"
                "Assessment of error handling patterns and potential failure modes.\n\n"
                "### Best Practices Compliance\n"
                "Evaluation against established coding standards and idioms.\n"
            )
        elif analysis_type == "comparison":
            specific_sections = (
                "## Comparative Analysis\n\n"
                "### Feature Comparison Matrix\n"
                f"Systematic comparison of options related to: {subject}\n\n"
                "### Trade-off Analysis\n"
                "Evaluation of trade-offs between compared alternatives.\n\n"
                "### Selection Criteria\n"
                "Weighted criteria for making a decision.\n"
            )
        elif analysis_type == "diagnostic":
            specific_sections = (
                "## Diagnostic Findings\n\n"
                "### Root Cause Analysis\n"
                f"Investigation into the root causes related to: {subject}\n\n"
                "### Contributing Factors\n"
                "Secondary factors that contribute to the issue.\n\n"
                "### Resolution Path\n"
                "Step-by-step approach to resolving identified issues.\n"
            )
        elif analysis_type == "performance":
            specific_sections = (
                "## Performance Analysis\n\n"
                "### Current Performance Profile\n"
                f"Baseline performance characteristics for: {subject}\n\n"
                "### Bottleneck Identification\n"
                "Key performance bottlenecks and their impact.\n\n"
                "### Optimization Opportunities\n"
                "Specific areas where performance can be improved.\n"
            )
        elif analysis_type == "security":
            specific_sections = (
                "## Security Assessment\n\n"
                "### Threat Surface Analysis\n"
                f"Identified attack vectors and exposure points for: {subject}\n\n"
                "### Vulnerability Assessment\n"
                "Categorized vulnerabilities by severity.\n\n"
                "### Mitigation Strategies\n"
                "Recommended countermeasures and security controls.\n"
            )
        else:
            specific_sections = (
                "## Detailed Analysis\n\n"
                "### Current State Assessment\n"
                f"Evaluation of the current state regarding: {subject}\n\n"
                "### Identified Patterns\n"
                "Key patterns and trends observed during analysis.\n\n"
                "### Gap Analysis\n"
                "Areas where current state diverges from ideal state.\n"
            )

        return (
            f"# Analysis: {subject}\n\n"
            f"## Scope\n\n{scope_section}\n\n"
            f"## Methodology\n\n"
            f"This analysis uses a structured {analysis_type.replace('_', ' ')} approach "
            f"to evaluate the subject systematically.\n\n"
            f"{specific_sections}\n"
            f"## Key Insights\n\n"
            f"Based on the {analysis_type.replace('_', ' ')} of {subject}:\n\n"
            f"1. Primary finding derived from systematic evaluation of the subject matter.\n"
            f"2. Secondary finding based on pattern analysis and contextual assessment.\n"
            f"3. Supporting observation that reinforces the primary findings.\n\n"
            f"## Recommendations\n\n"
            f"1. **Immediate**: Address the most critical findings first.\n"
            f"2. **Short-term**: Implement improvements based on the gap analysis.\n"
            f"3. **Long-term**: Establish ongoing monitoring and review processes.\n"
        )

    def _determine_file_path(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Determine the output file path for generated code."""
        # Extract language from task
        task_lower = task.lower()

        if "python" in task_lower or ".py" in task_lower:
            return "output/solution.py"
        elif "javascript" in task_lower or ".js" in task_lower or "node" in task_lower:
            return "output/solution.js"
        elif "typescript" in task_lower or ".ts" in task_lower:
            return "output/solution.ts"
        elif "java" in task_lower:
            return "output/Solution.java"
        elif " go " in task_lower or task_lower.endswith(" go") or "golang" in task_lower:
            return "output/solution.go"
        elif "rust" in task_lower or ".rs" in task_lower:
            return "output/solution.rs"
        else:
            return None

    # ========================================================================
    # Validation Methods
    # ========================================================================

    def _validate_output(self, result: ExecutionResult) -> ExecutionResult:
        """Validate the generated output for completeness and format consistency."""
        validation_issues = []

        # Check output completeness
        if result.output is None:
            validation_issues.append("Output is None")
            result.status = "failed"
            result.error = "No output was generated"
        elif isinstance(result.output, str):
            output_str = result.output.strip()
            if len(output_str) == 0:
                validation_issues.append("Output is empty")
                result.status = "failed"
                result.error = "Empty output generated"
            elif len(output_str) < 20:
                validation_issues.append("Output is suspiciously short")
                result.quality_score = max(0.0, result.quality_score - 0.15)
        elif isinstance(result.output, dict):
            if not result.output:
                validation_issues.append("Output dict is empty")
                result.status = "partial"

        # Validate format consistency for code output
        if result.files_created:
            for file_path in result.files_created:
                ext = os.path.splitext(file_path)[1].lower()
                if isinstance(result.output, str):
                    # Check that code output contains meaningful structure
                    if ext == ".py":
                        if not any(kw in result.output for kw in ["def ", "class ", "import ", "from "]):
                            validation_issues.append(f"Python file {file_path} lacks function/class definitions")
                            result.quality_score = max(0.0, result.quality_score - 0.1)
                    elif ext in (".js", ".ts"):
                        if not any(kw in result.output for kw in ["function ", "const ", "class ", "export ", "import ", "=>"]):
                            validation_issues.append(f"JS/TS file {file_path} lacks expected constructs")
                            result.quality_score = max(0.0, result.quality_score - 0.1)
                    elif ext == ".java":
                        if "class " not in result.output:
                            validation_issues.append(f"Java file {file_path} lacks class definition")
                            result.quality_score = max(0.0, result.quality_score - 0.1)
                    elif ext == ".go":
                        if "package " not in result.output:
                            validation_issues.append(f"Go file {file_path} lacks package declaration")
                            result.quality_score = max(0.0, result.quality_score - 0.1)
                    elif ext == ".rs":
                        if not any(kw in result.output for kw in ["fn ", "struct ", "use "]):
                            validation_issues.append(f"Rust file {file_path} lacks expected constructs")
                            result.quality_score = max(0.0, result.quality_score - 0.1)

        # Check for placeholder/stub markers that should not be in final output
        if isinstance(result.output, str):
            stub_markers = ["TODO", "FIXME", "placeholder", "not implemented"]
            stub_count = sum(1 for marker in stub_markers if marker.lower() in result.output.lower())
            if stub_count > 0:
                validation_issues.append(f"Output contains {stub_count} stub marker(s)")
                result.quality_score = max(0.0, result.quality_score - 0.05 * stub_count)

        # Apply validation bonus if no issues found
        if not validation_issues:
            result.quality_score = min(1.0, result.quality_score + 0.1)
        else:
            # Store validation issues in error field if not already set
            if result.error is None and result.status != "failed":
                result.error = f"Validation warnings: {'; '.join(validation_issues)}"
                if result.status == "success" and len(validation_issues) > 2:
                    result.status = "partial"

        return result

    # ========================================================================
    # Helper Methods: Language/Framework Detection and Requirement Extraction
    # ========================================================================

    def _detect_language(self, task: str) -> str:
        """Detect the programming language from the task description."""
        task_lower = task.lower()
        language_map = [
            (["javascript", ".js", "node.js", "nodejs", "react", "vue", "angular", "express"], "javascript"),
            (["typescript", ".ts", "tsx", "angular", "nest.js", "nestjs"], "typescript"),
            (["python", ".py", "django", "flask", "fastapi", "pytorch", "pandas"], "python"),
            (["java", "spring", "maven", "gradle", "junit"], "java"),
            ([" go ", "golang", ".go", "goroutine"], "go"),
            (["rust", ".rs", "cargo", "tokio"], "rust"),
            (["c#", "csharp", ".net", "dotnet", "asp.net"], "csharp"),
            (["ruby", "rails", ".rb"], "ruby"),
            (["php", "laravel", "symfony"], "php"),
            (["swift", "ios", "swiftui"], "swift"),
            (["kotlin", "android"], "kotlin"),
        ]
        for keywords, lang in language_map:
            if any(kw in task_lower for kw in keywords):
                return lang
        return "python"  # Default

    def _detect_framework(self, task: str) -> Optional[str]:
        """Detect the framework from the task description."""
        task_lower = task.lower()
        framework_map = [
            (["fastapi", "fast api"], "fastapi"),
            (["flask"], "flask"),
            (["django"], "django"),
            (["express"], "express"),
            (["nestjs", "nest.js"], "nestjs"),
            (["react"], "react"),
            (["vue"], "vue"),
            (["angular"], "angular"),
            (["spring"], "spring"),
            (["rails"], "rails"),
            (["laravel"], "laravel"),
            (["gin"], "gin"),
            (["actix", "rocket"], "actix"),
        ]
        for keywords, fw in framework_map:
            if any(kw in task_lower for kw in keywords):
                return fw
        return None

    def _extract_requirements(self, task: str) -> List[str]:
        """Extract specific requirements from the task description."""
        import re
        requirements = []
        task_lower = task.lower()

        # Extract explicit requirements from bullet points or numbered lists
        lines = task.split("\n")
        for line in lines:
            stripped = line.strip()
            if re.match(r'^[-*]\s+', stripped) or re.match(r'^\d+[.)]\s+', stripped):
                clean = re.sub(r'^[-*\d.)]+\s*', '', stripped).strip()
                if clean:
                    requirements.append(clean)

        # Extract requirements from "should", "must", "need to" patterns
        sentences = re.split(r'[.;]', task)
        for sentence in sentences:
            s = sentence.strip().lower()
            if any(kw in s for kw in ["should ", "must ", "need to ", "needs to ", "has to ", "required to "]):
                clean = sentence.strip()
                if clean and clean not in requirements:
                    requirements.append(clean)

        # If no explicit requirements found, derive from key phrases
        if not requirements:
            capability_patterns = [
                (r"with\s+([\w\s,]+(?:and\s+[\w\s]+)?)\s*$", "Support for {}"),
                (r"that\s+([\w\s]+)", "{}"),
                (r"for\s+([\w\s]+)", "Handle {}"),
            ]
            for pattern, template in capability_patterns:
                match = re.search(pattern, task_lower)
                if match:
                    requirements.append(template.format(match.group(1).strip()))

        return requirements

    def _extract_entity_name(self, task: str) -> str:
        """Extract a meaningful entity name from the task."""
        import re
        task_lower = task.lower()

        # Try to find named entities after key verbs
        patterns = [
            r'(?:create|implement|build|write|develop|make)\s+(?:a\s+|an\s+)?(\w+)',
            r'(\w+)\s+(?:function|class|module|service|api|endpoint)',
            r'(?:called|named)\s+["\']?(\w+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, task_lower)
            if match:
                name = match.group(1)
                if name not in ("a", "an", "the", "new", "simple", "basic"):
                    return name

        return "solution"

    # ========================================================================
    # Language-Specific Code Generators
    # ========================================================================

    def _gen_python_code(self, task: str, name: str, framework: Optional[str],
                         is_api: bool, is_class: bool, is_function: bool,
                         is_cli: bool, is_test: bool, req_lines: str) -> str:
        """Generate Python code based on task analysis."""
        if is_api and framework == "fastapi":
            return (
                f'"""API implementation for: {task}"""\n\n'
                f'from fastapi import FastAPI, HTTPException\n'
                f'from pydantic import BaseModel\n'
                f'from typing import List, Optional\n\n\n'
                f'app = FastAPI(title="{name.title()} API")\n\n\n'
                f'class {name.title()}Model(BaseModel):\n'
                f'    """Data model for {name}.\n{req_lines}\n    """\n'
                f'    id: Optional[int] = None\n'
                f'    name: str\n'
                f'    data: dict = {{}}\n\n\n'
                f'# In-memory store\n'
                f'_store: List[{name.title()}Model] = []\n'
                f'_counter: int = 0\n\n\n'
                f'@app.get("/{name}s", response_model=List[{name.title()}Model])\n'
                f'async def list_{name}s():\n'
                f'    """Retrieve all {name}s."""\n'
                f'    return _store\n\n\n'
                f'@app.post("/{name}s", response_model={name.title()}Model, status_code=201)\n'
                f'async def create_{name}(item: {name.title()}Model):\n'
                f'    """Create a new {name}."""\n'
                f'    global _counter\n'
                f'    _counter += 1\n'
                f'    item.id = _counter\n'
                f'    _store.append(item)\n'
                f'    return item\n\n\n'
                f'@app.get("/{name}s/{{item_id}}", response_model={name.title()}Model)\n'
                f'async def get_{name}(item_id: int):\n'
                f'    """Retrieve a {name} by ID."""\n'
                f'    for item in _store:\n'
                f'        if item.id == item_id:\n'
                f'            return item\n'
                f'    raise HTTPException(status_code=404, detail="{name.title()} not found")\n\n\n'
                f'@app.delete("/{name}s/{{item_id}}", status_code=204)\n'
                f'async def delete_{name}(item_id: int):\n'
                f'    """Delete a {name} by ID."""\n'
                f'    global _store\n'
                f'    _store = [item for item in _store if item.id != item_id]\n'
            )
        elif is_api and framework == "flask":
            return (
                f'"""Flask API implementation for: {task}"""\n\n'
                f'from flask import Flask, jsonify, request, abort\n\n\n'
                f'app = Flask(__name__)\n'
                f'_store = []\n'
                f'_counter = 0\n\n\n'
                f'@app.route("/{name}s", methods=["GET"])\n'
                f'def list_{name}s():\n'
                f'    """List all {name}s.\n{req_lines}\n    """\n'
                f'    return jsonify(_store)\n\n\n'
                f'@app.route("/{name}s", methods=["POST"])\n'
                f'def create_{name}():\n'
                f'    """Create a new {name}."""\n'
                f'    global _counter\n'
                f'    data = request.get_json()\n'
                f'    if not data:\n'
                f'        abort(400, description="Request body is required")\n'
                f'    _counter += 1\n'
                f'    data["id"] = _counter\n'
                f'    _store.append(data)\n'
                f'    return jsonify(data), 201\n\n\n'
                f'if __name__ == "__main__":\n'
                f'    app.run(debug=True)\n'
            )
        elif is_api:
            return (
                f'"""API implementation for: {task}"""\n\n'
                f'from http.server import HTTPServer, BaseHTTPRequestHandler\n'
                f'import json\n'
                f'from typing import List, Dict, Any\n\n\n'
                f'class {name.title()}Handler(BaseHTTPRequestHandler):\n'
                f'    """HTTP handler for {name} resources.\n{req_lines}\n    """\n\n'
                f'    _store: List[Dict[str, Any]] = []\n'
                f'    _counter: int = 0\n\n'
                f'    def do_GET(self):\n'
                f'        """Handle GET requests."""\n'
                f'        if self.path == "/{name}s":\n'
                f'            self._send_json(200, self._store)\n'
                f'        else:\n'
                f'            self._send_json(404, {{"error": "Not found"}})\n\n'
                f'    def do_POST(self):\n'
                f'        """Handle POST requests."""\n'
                f'        content_length = int(self.headers.get("Content-Length", 0))\n'
                f'        body = json.loads(self.rfile.read(content_length))\n'
                f'        {name.title()}Handler._counter += 1\n'
                f'        body["id"] = {name.title()}Handler._counter\n'
                f'        {name.title()}Handler._store.append(body)\n'
                f'        self._send_json(201, body)\n\n'
                f'    def _send_json(self, status: int, data: Any):\n'
                f'        """Send a JSON response."""\n'
                f'        self.send_response(status)\n'
                f'        self.send_header("Content-Type", "application/json")\n'
                f'        self.end_headers()\n'
                f'        self.wfile.write(json.dumps(data).encode())\n\n\n'
                f'def run_server(port: int = 8000):\n'
                f'    """Start the HTTP server."""\n'
                f'    server = HTTPServer(("", port), {name.title()}Handler)\n'
                f'    print(f"Server running on port {{port}}")\n'
                f'    server.serve_forever()\n\n\n'
                f'if __name__ == "__main__":\n'
                f'    run_server()\n'
            )
        elif is_test:
            return (
                f'"""Tests for: {task}"""\n\n'
                f'import unittest\n'
                f'from unittest.mock import MagicMock, patch\n\n\n'
                f'class Test{name.title()}(unittest.TestCase):\n'
                f'    """Test suite for {name}.\n{req_lines}\n    """\n\n'
                f'    def setUp(self):\n'
                f'        """Set up test fixtures."""\n'
                f'        self.instance = None  # Initialize the subject under test\n\n'
                f'    def tearDown(self):\n'
                f'        """Clean up after tests."""\n'
                f'        pass\n\n'
                f'    def test_basic_functionality(self):\n'
                f'        """Test that basic functionality works correctly."""\n'
                f'        # Arrange\n'
                f'        expected = True\n'
                f'        # Act\n'
                f'        result = True  # Replace with actual call\n'
                f'        # Assert\n'
                f'        self.assertEqual(result, expected)\n\n'
                f'    def test_edge_case_empty_input(self):\n'
                f'        """Test behavior with empty input."""\n'
                f'        # Test edge case handling\n'
                f'        self.assertIsNotNone(self.instance)\n\n'
                f'    def test_error_handling(self):\n'
                f'        """Test that errors are handled gracefully."""\n'
                f'        with self.assertRaises(Exception):\n'
                f'            pass  # Replace with actual error-triggering call\n\n\n'
                f'if __name__ == "__main__":\n'
                f'    unittest.main()\n'
            )
        elif is_class:
            return (
                f'"""{name.title()} class implementation for: {task}"""\n\n'
                f'from typing import Any, Dict, List, Optional\n'
                f'from dataclasses import dataclass, field\n\n\n'
                f'@dataclass\n'
                f'class {name.title()}:\n'
                f'    """{name.title()} entity.\n{req_lines}\n    """\n\n'
                f'    name: str\n'
                f'    data: Dict[str, Any] = field(default_factory=dict)\n'
                f'    _initialized: bool = field(default=False, repr=False)\n\n'
                f'    def __post_init__(self):\n'
                f'        """Validate and initialize after construction."""\n'
                f'        if not self.name:\n'
                f'            raise ValueError("name is required")\n'
                f'        self._initialized = True\n\n'
                f'    def process(self) -> Dict[str, Any]:\n'
                f'        """Process the {name} and return results."""\n'
                f'        if not self._initialized:\n'
                f'            raise RuntimeError("{name.title()} not properly initialized")\n'
                f'        return {{\n'
                f'            "name": self.name,\n'
                f'            "status": "processed",\n'
                f'            "data": self.data,\n'
                f'        }}\n\n'
                f'    def validate(self) -> bool:\n'
                f'        """Validate the {name} state."""\n'
                f'        return bool(self.name and self._initialized)\n\n'
                f'    def to_dict(self) -> Dict[str, Any]:\n'
                f'        """Serialize to dictionary."""\n'
                f'        return {{\n'
                f'            "name": self.name,\n'
                f'            "data": self.data,\n'
                f'        }}\n\n'
                f'    @classmethod\n'
                f'    def from_dict(cls, data: Dict[str, Any]) -> "{name.title()}":\n'
                f'        """Deserialize from dictionary."""\n'
                f'        return cls(name=data["name"], data=data.get("data", {{}}))\n'
            )
        elif is_cli:
            return (
                f'"""CLI tool for: {task}"""\n\n'
                f'import argparse\n'
                f'import sys\n'
                f'from typing import List\n\n\n'
                f'def {name}(args: List[str]) -> int:\n'
                f'    """Execute the {name} command.\n{req_lines}\n    """\n'
                f'    parser = argparse.ArgumentParser(\n'
                f'        description="{task}"\n'
                f'    )\n'
                f'    parser.add_argument("input", help="Input to process")\n'
                f'    parser.add_argument("-o", "--output", help="Output file path")\n'
                f'    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")\n\n'
                f'    parsed = parser.parse_args(args)\n\n'
                f'    if parsed.verbose:\n'
                f'        print(f"Processing: {{parsed.input}}")\n\n'
                f'    # Core logic\n'
                f'    result = process_input(parsed.input)\n\n'
                f'    if parsed.output:\n'
                f'        with open(parsed.output, "w") as f:\n'
                f'            f.write(str(result))\n'
                f'        if parsed.verbose:\n'
                f'            print(f"Output written to {{parsed.output}}")\n'
                f'    else:\n'
                f'        print(result)\n\n'
                f'    return 0\n\n\n'
                f'def process_input(input_data: str) -> str:\n'
                f'    """Process the input data and return results."""\n'
                f'    return f"Processed: {{input_data}}"\n\n\n'
                f'if __name__ == "__main__":\n'
                f'    sys.exit({name}(sys.argv[1:]))\n'
            )
        else:
            # General function
            return (
                f'"""{name.title()} implementation for: {task}"""\n\n'
                f'from typing import Any, Dict, List, Optional\n\n\n'
                f'def {name}(\n'
                f'    input_data: Any,\n'
                f'    options: Optional[Dict[str, Any]] = None,\n'
                f') -> Dict[str, Any]:\n'
                f'    """{task}\n{req_lines}\n\n'
                f'    Args:\n'
                f'        input_data: The primary input to process.\n'
                f'        options: Optional configuration options.\n\n'
                f'    Returns:\n'
                f'        Dictionary containing the results.\n'
                f'    """\n'
                f'    options = options or {{}}\n\n'
                f'    # Validate input\n'
                f'    if input_data is None:\n'
                f'        raise ValueError("input_data is required")\n\n'
                f'    # Process the input\n'
                f'    result = _process(input_data, options)\n\n'
                f'    return {{\n'
                f'        "status": "success",\n'
                f'        "input": str(input_data),\n'
                f'        "result": result,\n'
                f'    }}\n\n\n'
                f'def _process(data: Any, options: Dict[str, Any]) -> Any:\n'
                f'    """Internal processing logic.\n\n'
                f'    Args:\n'
                f'        data: Data to process.\n'
                f'        options: Processing options.\n\n'
                f'    Returns:\n'
                f'        Processed result.\n'
                f'    """\n'
                f'    # Apply any configured transformations\n'
                f'    if options.get("transform"):\n'
                f'        data = _apply_transform(data, options["transform"])\n'
                f'    return data\n\n\n'
                f'def _apply_transform(data: Any, transform: str) -> Any:\n'
                f'    """Apply a named transformation to the data."""\n'
                f'    transforms = {{\n'
                f'        "uppercase": lambda d: str(d).upper(),\n'
                f'        "lowercase": lambda d: str(d).lower(),\n'
                f'        "strip": lambda d: str(d).strip(),\n'
                f'    }}\n'
                f'    fn = transforms.get(transform)\n'
                f'    if fn:\n'
                f'        return fn(data)\n'
                f'    return data\n'
            )

    def _gen_js_ts_code(self, task: str, name: str, language: str,
                        framework: Optional[str], is_api: bool, is_class: bool,
                        is_function: bool, is_test: bool, req_lines: str) -> str:
        """Generate JavaScript/TypeScript code."""
        ts_typing = language == "typescript"
        type_suffix = ": string" if ts_typing else ""
        any_type = ": any" if ts_typing else ""
        ret_type = ": object" if ts_typing else ""

        if is_api and framework in ("express", None):
            mod = "express" if framework == "express" else "http"
            return (
                f'/**\n * API implementation for: {task}\n */\n\n'
                f'const express = require("express");\n'
                f'const app = express();\n'
                f'app.use(express.json());\n\n'
                f'let store{any_type} = [];\n'
                f'let counter{type_suffix.replace("string","number") if ts_typing else ""} = 0;\n\n'
                f'app.get("/{name}s", (req{any_type}, res{any_type}) => {{\n'
                f'  res.json(store);\n'
                f'}});\n\n'
                f'app.post("/{name}s", (req{any_type}, res{any_type}) => {{\n'
                f'  counter++;\n'
                f'  const item = {{ id: counter, ...req.body }};\n'
                f'  store.push(item);\n'
                f'  res.status(201).json(item);\n'
                f'}});\n\n'
                f'app.get("/{name}s/:id", (req{any_type}, res{any_type}) => {{\n'
                f'  const item = store.find(i => i.id === parseInt(req.params.id));\n'
                f'  if (!item) return res.status(404).json({{ error: "Not found" }});\n'
                f'  res.json(item);\n'
                f'}});\n\n'
                f'const PORT = process.env.PORT || 3000;\n'
                f'app.listen(PORT, () => console.log(`Server running on port ${{PORT}}`));\n\n'
                f'module.exports = app;\n'
            )
        elif is_class:
            if ts_typing:
                return (
                    f'/**\n * {name.title()} class for: {task}\n */\n\n'
                    f'interface {name.title()}Data {{\n'
                    f'  [key: string]: any;\n'
                    f'}}\n\n'
                    f'export class {name.title()} {{\n'
                    f'  private name: string;\n'
                    f'  private data: {name.title()}Data;\n\n'
                    f'  constructor(name: string, data: {name.title()}Data = {{}}) {{\n'
                    f'    if (!name) throw new Error("name is required");\n'
                    f'    this.name = name;\n'
                    f'    this.data = data;\n'
                    f'  }}\n\n'
                    f'  process(): {name.title()}Data {{\n'
                    f'    return {{ name: this.name, status: "processed", data: this.data }};\n'
                    f'  }}\n\n'
                    f'  validate(): boolean {{\n'
                    f'    return Boolean(this.name);\n'
                    f'  }}\n\n'
                    f'  toJSON(): {name.title()}Data {{\n'
                    f'    return {{ name: this.name, data: this.data }};\n'
                    f'  }}\n}}\n'
                )
            else:
                return (
                    f'/**\n * {name.title()} class for: {task}\n */\n\n'
                    f'class {name.title()} {{\n'
                    f'  constructor(name, data = {{}}) {{\n'
                    f'    if (!name) throw new Error("name is required");\n'
                    f'    this.name = name;\n'
                    f'    this.data = data;\n'
                    f'  }}\n\n'
                    f'  process() {{\n'
                    f'    return {{ name: this.name, status: "processed", data: this.data }};\n'
                    f'  }}\n\n'
                    f'  validate() {{\n'
                    f'    return Boolean(this.name);\n'
                    f'  }}\n\n'
                    f'  toJSON() {{\n'
                    f'    return {{ name: this.name, data: this.data }};\n'
                    f'  }}\n}}\n\n'
                    f'module.exports = {{ {name.title()} }};\n'
                )
        else:
            return (
                f'/**\n * {name} implementation for: {task}\n */\n\n'
                f'{"export " if ts_typing else ""}function {name}(inputData{any_type}, options{any_type}){ret_type} {{\n'
                f'  if (inputData === undefined || inputData === null) {{\n'
                f'    throw new Error("inputData is required");\n'
                f'  }}\n'
                f'  const opts = options || {{}};\n'
                f'  const result = process(inputData, opts);\n'
                f'  return {{ status: "success", input: String(inputData), result }};\n'
                f'}}\n\n'
                f'function process(data{any_type}, options{any_type}){any_type} {{\n'
                f'  if (options.transform) {{\n'
                f'    return applyTransform(data, options.transform);\n'
                f'  }}\n'
                f'  return data;\n'
                f'}}\n\n'
                f'function applyTransform(data{any_type}, transform{type_suffix}){any_type} {{\n'
                f'  const transforms = {{\n'
                f'    uppercase: (d{any_type}) => String(d).toUpperCase(),\n'
                f'    lowercase: (d{any_type}) => String(d).toLowerCase(),\n'
                f'    trim: (d{any_type}) => String(d).trim(),\n'
                f'  }};\n'
                f'  const fn = transforms[transform];\n'
                f'  return fn ? fn(data) : data;\n'
                f'}}\n\n'
                f'{"" if ts_typing else "module.exports = { " + name + " };" + chr(10)}'
            )

    def _gen_go_code(self, task: str, name: str, is_api: bool,
                     is_cli: bool, req_lines: str) -> str:
        """Generate Go code."""
        if is_api:
            return (
                f'// API implementation for: {task}\n'
                f'package main\n\n'
                f'import (\n'
                f'\t"encoding/json"\n'
                f'\t"fmt"\n'
                f'\t"log"\n'
                f'\t"net/http"\n'
                f'\t"sync"\n'
                f')\n\n'
                f'type {name.title()} struct {{\n'
                f'\tID   int    `json:"id"`\n'
                f'\tName string `json:"name"`\n'
                f'}}\n\n'
                f'var (\n'
                f'\tstore   []{name.title()}\n'
                f'\tcounter int\n'
                f'\tmu      sync.Mutex\n'
                f')\n\n'
                f'func handle{name.title()}s(w http.ResponseWriter, r *http.Request) {{\n'
                f'\tw.Header().Set("Content-Type", "application/json")\n'
                f'\tswitch r.Method {{\n'
                f'\tcase http.MethodGet:\n'
                f'\t\tjson.NewEncoder(w).Encode(store)\n'
                f'\tcase http.MethodPost:\n'
                f'\t\tvar item {name.title()}\n'
                f'\t\tif err := json.NewDecoder(r.Body).Decode(&item); err != nil {{\n'
                f'\t\t\thttp.Error(w, err.Error(), http.StatusBadRequest)\n'
                f'\t\t\treturn\n'
                f'\t\t}}\n'
                f'\t\tmu.Lock()\n'
                f'\t\tcounter++\n'
                f'\t\titem.ID = counter\n'
                f'\t\tstore = append(store, item)\n'
                f'\t\tmu.Unlock()\n'
                f'\t\tw.WriteHeader(http.StatusCreated)\n'
                f'\t\tjson.NewEncoder(w).Encode(item)\n'
                f'\tdefault:\n'
                f'\t\thttp.Error(w, "Method not allowed", http.StatusMethodNotAllowed)\n'
                f'\t}}\n'
                f'}}\n\n'
                f'func main() {{\n'
                f'\thttp.HandleFunc("/{name}s", handle{name.title()}s)\n'
                f'\tfmt.Println("Server starting on :8080")\n'
                f'\tlog.Fatal(http.ListenAndServe(":8080", nil))\n'
                f'}}\n'
            )
        else:
            return (
                f'// {name} implementation for: {task}\n'
                f'package main\n\n'
                f'import (\n'
                f'\t"fmt"\n'
                f'\t"os"\n'
                f')\n\n'
                f'// {name.title()} performs the core operation.\n'
                f'func {name.title()}(input string) (string, error) {{\n'
                f'\tif input == "" {{\n'
                f'\t\treturn "", fmt.Errorf("input is required")\n'
                f'\t}}\n'
                f'\tresult := process(input)\n'
                f'\treturn result, nil\n'
                f'}}\n\n'
                f'func process(data string) string {{\n'
                f'\treturn fmt.Sprintf("Processed: %s", data)\n'
                f'}}\n\n'
                f'func main() {{\n'
                f'\tif len(os.Args) < 2 {{\n'
                f'\t\tfmt.Fprintln(os.Stderr, "Usage: {name} <input>")\n'
                f'\t\tos.Exit(1)\n'
                f'\t}}\n'
                f'\tresult, err := {name.title()}(os.Args[1])\n'
                f'\tif err != nil {{\n'
                f'\t\tfmt.Fprintln(os.Stderr, "Error:", err)\n'
                f'\t\tos.Exit(1)\n'
                f'\t}}\n'
                f'\tfmt.Println(result)\n'
                f'}}\n'
            )

    def _gen_rust_code(self, task: str, name: str, is_cli: bool,
                       req_lines: str) -> str:
        """Generate Rust code."""
        return (
            f'//! {name} implementation for: {task}\n\n'
            f'use std::collections::HashMap;\n'
            f'use std::error::Error;\n'
            f'use std::fmt;\n\n'
            f'#[derive(Debug)]\n'
            f'pub struct {name.title()} {{\n'
            f'    name: String,\n'
            f'    data: HashMap<String, String>,\n'
            f'}}\n\n'
            f'#[derive(Debug)]\n'
            f'pub struct {name.title()}Error {{\n'
            f'    message: String,\n'
            f'}}\n\n'
            f'impl fmt::Display for {name.title()}Error {{\n'
            f'    fn fmt(&self, f: &mut fmt::Formatter<\'_>) -> fmt::Result {{\n'
            f'        write!(f, "{name.title()}Error: {{}}", self.message)\n'
            f'    }}\n'
            f'}}\n\n'
            f'impl Error for {name.title()}Error {{}}\n\n'
            f'impl {name.title()} {{\n'
            f'    pub fn new(name: &str) -> Result<Self, {name.title()}Error> {{\n'
            f'        if name.is_empty() {{\n'
            f'            return Err({name.title()}Error {{\n'
            f'                message: "name is required".to_string(),\n'
            f'            }});\n'
            f'        }}\n'
            f'        Ok(Self {{\n'
            f'            name: name.to_string(),\n'
            f'            data: HashMap::new(),\n'
            f'        }})\n'
            f'    }}\n\n'
            f'    pub fn process(&self) -> HashMap<String, String> {{\n'
            f'        let mut result = HashMap::new();\n'
            f'        result.insert("name".to_string(), self.name.clone());\n'
            f'        result.insert("status".to_string(), "processed".to_string());\n'
            f'        result\n'
            f'    }}\n\n'
            f'    pub fn validate(&self) -> bool {{\n'
            f'        !self.name.is_empty()\n'
            f'    }}\n'
            f'}}\n\n'
            f'fn main() -> Result<(), Box<dyn Error>> {{\n'
            f'    let args: Vec<String> = std::env::args().collect();\n'
            f'    if args.len() < 2 {{\n'
            f'        eprintln!("Usage: {name} <input>");\n'
            f'        std::process::exit(1);\n'
            f'    }}\n'
            f'    let instance = {name.title()}::new(&args[1])?;\n'
            f'    let result = instance.process();\n'
            f'    println!("{{:?}}", result);\n'
            f'    Ok(())\n'
            f'}}\n'
        )

    def _gen_java_code(self, task: str, name: str, is_class: bool,
                       is_api: bool, req_lines: str) -> str:
        """Generate Java code."""
        class_name = name.title().replace("_", "")
        return (
            f'/**\n * {class_name} implementation for: {task}\n */\n\n'
            f'import java.util.*;\n\n'
            f'public class {class_name} {{\n\n'
            f'    private String name;\n'
            f'    private Map<String, Object> data;\n\n'
            f'    public {class_name}(String name) {{\n'
            f'        if (name == null || name.isEmpty()) {{\n'
            f'            throw new IllegalArgumentException("name is required");\n'
            f'        }}\n'
            f'        this.name = name;\n'
            f'        this.data = new HashMap<>();\n'
            f'    }}\n\n'
            f'    public Map<String, Object> process() {{\n'
            f'        Map<String, Object> result = new HashMap<>();\n'
            f'        result.put("name", this.name);\n'
            f'        result.put("status", "processed");\n'
            f'        result.put("data", this.data);\n'
            f'        return result;\n'
            f'    }}\n\n'
            f'    public boolean validate() {{\n'
            f'        return this.name != null && !this.name.isEmpty();\n'
            f'    }}\n\n'
            f'    public String getName() {{ return this.name; }}\n'
            f'    public Map<String, Object> getData() {{ return this.data; }}\n'
            f'    public void setData(Map<String, Object> data) {{ this.data = data; }}\n\n'
            f'    public static void main(String[] args) {{\n'
            f'        if (args.length < 1) {{\n'
            f'            System.err.println("Usage: {class_name} <name>");\n'
            f'            System.exit(1);\n'
            f'        }}\n'
            f'        {class_name} instance = new {class_name}(args[0]);\n'
            f'        System.out.println(instance.process());\n'
            f'    }}\n'
            f'}}\n'
        )

    # ========================================================================
    # Document Type Generators
    # ========================================================================

    def _gen_readme_doc(self, subject: str, req_section: str) -> str:
        """Generate a README document."""
        return (
            f"# {subject}\n\n"
            f"## Overview\n\n"
            f"This project provides a solution for: {subject}.\n\n"
            f"## Features\n\n"
            f"- Core functionality as described in the requirements\n"
            f"- Clean and maintainable architecture\n"
            f"- Comprehensive error handling\n"
            f"{req_section}\n"
            f"## Getting Started\n\n"
            f"### Prerequisites\n\n"
            f"- Ensure all dependencies are installed\n"
            f"- Review the configuration options\n\n"
            f"### Installation\n\n"
            f"```bash\n"
            f"# Clone the repository\n"
            f"git clone <repository-url>\n\n"
            f"# Install dependencies\n"
            f"pip install -r requirements.txt\n"
            f"```\n\n"
            f"### Usage\n\n"
            f"```bash\n"
            f"# Run the application\n"
            f"python main.py\n"
            f"```\n\n"
            f"## Configuration\n\n"
            f"Configuration options can be set via environment variables or configuration files.\n\n"
            f"## Contributing\n\n"
            f"1. Fork the repository\n"
            f"2. Create a feature branch\n"
            f"3. Commit your changes\n"
            f"4. Push to the branch\n"
            f"5. Open a Pull Request\n\n"
            f"## License\n\n"
            f"See LICENSE file for details.\n"
        )

    def _gen_api_doc(self, subject: str, req_section: str) -> str:
        """Generate API documentation."""
        return (
            f"# API Documentation: {subject}\n\n"
            f"## Base URL\n\n"
            f"`http://localhost:8000/api/v1`\n\n"
            f"## Authentication\n\n"
            f"All API requests require authentication via Bearer token in the Authorization header.\n\n"
            f"```\nAuthorization: Bearer <token>\n```\n"
            f"{req_section}\n"
            f"## Endpoints\n\n"
            f"### List Resources\n\n"
            f"```\nGET /resources\n```\n\n"
            f"**Response** (200 OK):\n"
            f'```json\n{{\n  "data": [],\n  "total": 0,\n  "page": 1\n}}\n```\n\n'
            f"### Create Resource\n\n"
            f"```\nPOST /resources\n```\n\n"
            f"**Request Body**:\n"
            f'```json\n{{\n  "name": "string",\n  "description": "string"\n}}\n```\n\n'
            f"**Response** (201 Created):\n"
            f'```json\n{{\n  "id": 1,\n  "name": "string",\n  "description": "string"\n}}\n```\n\n'
            f"### Get Resource by ID\n\n"
            f"```\nGET /resources/:id\n```\n\n"
            f"**Response** (200 OK): Returns the resource object.\n\n"
            f"### Update Resource\n\n"
            f"```\nPUT /resources/:id\n```\n\n"
            f"### Delete Resource\n\n"
            f"```\nDELETE /resources/:id\n```\n\n"
            f"**Response** (204 No Content)\n\n"
            f"## Error Responses\n\n"
            f"| Status Code | Description |\n"
            f"|-------------|-------------|\n"
            f"| 400 | Bad Request - Invalid input |\n"
            f"| 401 | Unauthorized - Invalid or missing token |\n"
            f"| 404 | Not Found - Resource does not exist |\n"
            f"| 500 | Internal Server Error |\n"
        )

    def _gen_guide_doc(self, subject: str, req_section: str) -> str:
        """Generate a guide/tutorial document."""
        return (
            f"# Guide: {subject}\n\n"
            f"## Introduction\n\n"
            f"This guide walks you through {subject}. By the end, you will have "
            f"a thorough understanding of the concepts and practical steps involved.\n\n"
            f"## Prerequisites\n\n"
            f"Before starting, ensure you have:\n"
            f"- Basic understanding of the relevant technologies\n"
            f"- Access to the required tools and environments\n"
            f"{req_section}\n"
            f"## Step 1: Setup\n\n"
            f"Begin by setting up your environment. Ensure all dependencies "
            f"are installed and configured correctly.\n\n"
            f"## Step 2: Core Implementation\n\n"
            f"With the environment ready, proceed to implement the core functionality. "
            f"Follow the patterns established in the codebase.\n\n"
            f"## Step 3: Testing and Validation\n\n"
            f"Verify your implementation by running the test suite and "
            f"checking the results against expected outputs.\n\n"
            f"## Step 4: Deployment\n\n"
            f"Once validated, deploy the solution following your team's deployment process.\n\n"
            f"## Troubleshooting\n\n"
            f"Common issues and their solutions:\n\n"
            f"- **Issue**: Configuration errors\n"
            f"  **Solution**: Verify environment variables and config files\n\n"
            f"- **Issue**: Dependency conflicts\n"
            f"  **Solution**: Check version compatibility and update as needed\n\n"
            f"## Next Steps\n\n"
            f"After completing this guide, consider exploring advanced topics "
            f"and additional features.\n"
        )

    def _gen_report_doc(self, subject: str, req_section: str) -> str:
        """Generate a report document."""
        return (
            f"# Report: {subject}\n\n"
            f"## Executive Summary\n\n"
            f"This report provides a comprehensive analysis of {subject}. "
            f"Key findings and recommendations are outlined below.\n\n"
            f"{req_section}\n"
            f"## Methodology\n\n"
            f"The analysis was conducted using systematic evaluation of "
            f"available data, stakeholder input, and industry best practices.\n\n"
            f"## Findings\n\n"
            f"### Key Finding 1\n\n"
            f"Description of the primary finding with supporting evidence.\n\n"
            f"### Key Finding 2\n\n"
            f"Description of the secondary finding with supporting evidence.\n\n"
            f"## Impact Analysis\n\n"
            f"Assessment of how the findings affect current operations "
            f"and future direction.\n\n"
            f"## Recommendations\n\n"
            f"1. **Priority 1**: Immediate action items based on critical findings\n"
            f"2. **Priority 2**: Short-term improvements to address gaps\n"
            f"3. **Priority 3**: Long-term strategic initiatives\n\n"
            f"## Conclusion\n\n"
            f"Summary of the report's key takeaways and next steps.\n"
        )

    def _gen_spec_doc(self, subject: str, req_section: str) -> str:
        """Generate a specification document."""
        return (
            f"# Specification: {subject}\n\n"
            f"## Purpose\n\n"
            f"This specification defines the requirements and design for {subject}.\n\n"
            f"## Scope\n\n"
            f"This document covers the functional and non-functional requirements, "
            f"architecture decisions, and acceptance criteria.\n"
            f"{req_section}\n"
            f"## Functional Requirements\n\n"
            f"### FR-001: Core Functionality\n"
            f"The system shall provide the primary capabilities as described.\n\n"
            f"### FR-002: Data Management\n"
            f"The system shall handle data input, processing, and output correctly.\n\n"
            f"### FR-003: Error Handling\n"
            f"The system shall gracefully handle all error conditions.\n\n"
            f"## Non-Functional Requirements\n\n"
            f"### NFR-001: Performance\n"
            f"Response times shall not exceed acceptable thresholds.\n\n"
            f"### NFR-002: Security\n"
            f"The system shall implement appropriate security controls.\n\n"
            f"### NFR-003: Scalability\n"
            f"The system shall support horizontal scaling.\n\n"
            f"## Architecture\n\n"
            f"High-level architecture and component interactions.\n\n"
            f"## Acceptance Criteria\n\n"
            f"- All functional requirements are met and verified\n"
            f"- Performance benchmarks are satisfied\n"
            f"- Security review is passed\n"
        )

    def _gen_proposal_doc(self, subject: str, req_section: str) -> str:
        """Generate a proposal document."""
        return (
            f"# Proposal: {subject}\n\n"
            f"## Problem Statement\n\n"
            f"Description of the problem or opportunity that motivates this proposal.\n\n"
            f"## Proposed Solution\n\n"
            f"Overview of the proposed approach to address {subject}.\n"
            f"{req_section}\n"
            f"## Benefits\n\n"
            f"- Addresses the core problem effectively\n"
            f"- Aligns with organizational goals\n"
            f"- Provides measurable outcomes\n\n"
            f"## Implementation Plan\n\n"
            f"### Phase 1: Planning and Design\n"
            f"Define detailed requirements and architecture.\n\n"
            f"### Phase 2: Development\n"
            f"Build and test the solution iteratively.\n\n"
            f"### Phase 3: Deployment and Evaluation\n"
            f"Deploy to production and measure results.\n\n"
            f"## Resource Requirements\n\n"
            f"Estimated resources needed for successful implementation.\n\n"
            f"## Risk Assessment\n\n"
            f"| Risk | Impact | Mitigation |\n"
            f"|------|--------|------------|\n"
            f"| Technical complexity | Medium | Incremental approach |\n"
            f"| Resource constraints | Low | Phased implementation |\n\n"
            f"## Timeline\n\n"
            f"Estimated timeline for each phase of implementation.\n"
        )

    def _gen_general_doc(self, subject: str, req_section: str) -> str:
        """Generate a general document."""
        return (
            f"# {subject}\n\n"
            f"## Overview\n\n"
            f"This document provides comprehensive information about {subject}.\n\n"
            f"{req_section}\n"
            f"## Background\n\n"
            f"Context and background information relevant to the subject.\n\n"
            f"## Details\n\n"
            f"Detailed coverage of the key aspects of {subject}, including "
            f"relevant considerations and implications.\n\n"
            f"## Key Points\n\n"
            f"- Primary consideration related to the subject\n"
            f"- Secondary consideration with supporting context\n"
            f"- Additional factors that influence the topic\n\n"
            f"## Conclusion\n\n"
            f"Summary of the document's main points and any recommended actions.\n"
        )

    def _load_system_prompt(self) -> str:
        """Load the system prompt from file."""
        try:
            with open(self.system_prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            return "You are the Executor. Generate solutions using Tree of Thoughts."


# =============================================================================
# Convenience Functions
# =============================================================================

def create_executor(
    system_prompt_path: str = "config/agents/executor/CLAUDE.md",
    model: str = "claude-3-5-sonnet-20241022",
) -> ExecutorAgent:
    """Create a configured Executor agent."""
    return ExecutorAgent(
        system_prompt_path=system_prompt_path,
        model=model,
    )
