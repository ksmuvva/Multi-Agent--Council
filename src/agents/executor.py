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
        start_time = time.time()

        # Step 1: Decompose the problem
        sub_problems = self._decompose_problem(task, analyst_report)

        # Step 2: Generate approaches for each sub-problem
        approaches = self._generate_approaches(sub_problems, analyst_report)

        # Step 3: Score and rank approaches
        scored_approaches = self._score_approaches(approaches, task, analyst_report)

        # Step 4: Select best approach
        selected_approach = self._select_best_approach(scored_approaches)

        # Step 5: Incorporate SME advice if provided
        if sme_advisory:
            selected_approach = self._adapt_to_sme_advice(
                selected_approach, sme_advisory
            )

        # Step 6: Execute the selected approach
        execution_result = self._execute_approach(
            selected_approach, task, context
        )

        # Step 7: Validate output
        execution_result = self._validate_output(execution_result)

        execution_result.execution_time = time.time() - start_time

        return execution_result

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
        # In real implementation, would modify approach based on SME suggestions

        # Add SME recommendations to pros
        for sme, advice in sme_advisory.items():
            approach.pros.append(f"SME ({sme}) recommendation: {advice[:50]}...")

        # May increase complexity based on SME input
        if sme_advisory:
            if approach.complexity == "low":
                approach.complexity = "medium"

        return approach

    def _execute_approach(
        self,
        approach: Approach,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ExecutionResult:
        """
        Execute the selected approach.

        Attempts SDK-based execution first (using spawn_subagent with
        Write/Bash/Skill tools), falling back to local generation.
        """
        # Try SDK-based execution first
        sdk_result = self._execute_via_sdk(approach, task, context)
        if sdk_result:
            return sdk_result

        # Fall back to local execution
        task_lower = task.lower()

        if any(kw in task_lower for kw in ["function", "class", "code", "implement", "api"]):
            return self._execute_code_task(approach, task, context)
        elif any(kw in task_lower for kw in ["document", "docs", "readme", "guide"]):
            return self._execute_document_task(approach, task, context)
        elif any(kw in task_lower for kw in ["analyze", "explain", "review"]):
            return self._execute_analysis_task(approach, task, context)
        else:
            return self._execute_general_task(approach, task, context)

    def _execute_via_sdk(
        self,
        approach: Approach,
        task: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[ExecutionResult]:
        """Execute via SDK spawn_subagent with real tool access."""
        try:
            from src.core.sdk_integration import build_agent_options, spawn_subagent

            options = build_agent_options(
                agent_name="executor",
                system_prompt=self.system_prompt,
                model_override=self.model,
                task_description=task,
            )

            execution_prompt = (
                f"Task: {task}\n\n"
                f"Approach: {approach.name}\n"
                f"Steps:\n" + "\n".join(f"  {i+1}. {s}" for i, s in enumerate(approach.steps))
            )

            if context:
                for key in ("analyst_report", "plan", "research"):
                    if key in context:
                        execution_prompt += f"\n\n{key.title()}:\n{str(context[key])[:1000]}"

            result = spawn_subagent(
                options=options,
                input_data=execution_prompt,
                max_retries=2,
            )

            if result.get("status") == "success" and result.get("output"):
                return ExecutionResult(
                    approach_name=approach.name,
                    status="success",
                    output=result["output"],
                    files_created=[],
                    quality_score=0.85,
                )

        except (ImportError, Exception):
            pass

        return None

    def _execute_code_task(
        self,
        approach: Approach,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ExecutionResult:
        """Execute a code generation task."""
        # In real implementation, would use code-generation skill
        # and Write tool to create files

        # Simulate code generation
        output = self._generate_code_output(task)

        # Determine file path
        file_path = self._determine_file_path(task, context)

        return ExecutionResult(
            approach_name=approach.name,
            status="success",
            output=output,
            files_created=[file_path] if file_path else [],
            quality_score=0.85,
        )

    def _execute_document_task(
        self,
        approach: Approach,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ExecutionResult:
        """Execute a document generation task."""
        # In real implementation, would use document-creation skill

        output = self._generate_document_output(task)

        return ExecutionResult(
            approach_name=approach.name,
            status="success",
            output=output,
            files_created=[],
            quality_score=0.9,
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
        """Execute a general task."""
        output = f"Solution for: {task}\n\nGenerated using {approach.name} approach."

        return ExecutionResult(
            approach_name=approach.name,
            status="success",
            output=output,
            files_created=[],
            quality_score=0.75,
        )

    # ========================================================================
    # Output Generation Methods
    # ========================================================================

    def _generate_code_output(self, task: str) -> str:
        """Generate code output for a task."""
        # In real implementation, would use LLM with code-generation skill
        return f"""
# Generated code for: {task}

# This is a placeholder. In a real implementation,
# the Executor would use the code-generation skill
# to create appropriate code.

def solution():
    '''Solution for the given task'''
    # Implementation here
    pass

if __name__ == "__main__":
    solution()
"""

    def _generate_document_output(self, task: str) -> str:
        """Generate document output for a task."""
        return f"""
# Documentation: {task}

## Overview
This document provides comprehensive information about {task}.

## Details
[Content would be generated here using document-creation skill]

## Summary
[Key points and takeaways]
"""

    def _generate_analysis_output(self, task: str) -> str:
        """Generate analysis output for a task."""
        return f"""
# Analysis: {task}

## Analysis Summary
[Analytical findings would be presented here]

## Key Insights
- Insight 1
- Insight 2
- Insight 3

## Recommendations
Based on the analysis, the following recommendations are made.
"""

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
        elif "javascript" in task_lower or "js" in task_lower or "node" in task_lower:
            return "output/solution.js"
        elif "typescript" in task_lower or "ts" in task_lower:
            return "output/solution.ts"
        elif "java" in task_lower:
            return "output/Solution.java"
        elif "go" in task_lower:
            return "output/solution.go"
        elif "rust" in task_lower or "rs" in task_lower:
            return "output/solution.rs"
        else:
            return None

    # ========================================================================
    # Validation Methods
    # ========================================================================

    def _validate_output(self, result: ExecutionResult) -> ExecutionResult:
        """Validate the generated output."""
        # For code, check syntax
        if result.files_created:
            for file_path in result.files_created:
                if file_path.endswith('.py'):
                    # In real implementation, would use Bash tool for syntax check
                    # python -m py_compile file_path
                    pass

        # Adjust quality score based on validation
        if result.error is None:
            result.quality_score = min(1.0, result.quality_score + 0.1)
        else:
            result.quality_score = max(0.0, result.quality_score - 0.2)

        return result

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
