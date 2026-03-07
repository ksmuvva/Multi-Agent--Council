"""
Clarifier Subagent

Formulates precise questions when requirements are missing,
ranked by impact on output quality.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from src.schemas.clarifier import (
    ClarificationRequest,
    ClarificationQuestion,
    QuestionPriority,
    ImpactAssessment,
)
from src.schemas.analyst import TaskIntelligenceReport, MissingInfo, SeverityLevel


class ClarifierAgent:
    """
    The Clarifier formulates questions for missing requirements.

    Key responsibilities:
    - Rank questions by impact on output quality
    - Provide context for why each question matters
    - Suggest default assumptions
    - Assess impact if unanswered
    """

    def __init__(
        self,
        system_prompt_path: str = "config/agents/clarifier/CLAUDE.md",
        model: str = "claude-3-5-sonnet-20241022",
        max_turns: int = 30,
    ):
        """
        Initialize the Clarifier agent.

        Args:
            system_prompt_path: Path to system prompt file
            model: Model to use for clarification
            max_turns: Maximum conversation turns
        """
        self.system_prompt_path = system_prompt_path
        self.model = model
        self.max_turns = max_turns
        self.system_prompt = self._load_system_prompt()

        # Priority scoring factors
        self.priority_weights = {
            "quality_impact": 0.5,
            "reversibility": 0.3,
            "user_burden": 0.2,
        }

    def formulate_questions(
        self,
        analyst_report: TaskIntelligenceReport,
        context: Optional[Dict[str, Any]] = None,
        max_questions: int = 5,
    ) -> ClarificationRequest:
        """
        Formulate clarification questions for missing requirements.

        Args:
            analyst_report: The TaskIntelligenceReport with missing info
            context: Additional context (previous clarifications, user preferences)
            max_questions: Maximum number of questions to ask

        Returns:
            ClarificationRequest with ranked questions
        """
        if not analyst_report.missing_info:
            # No missing info - return empty request
            return ClarificationRequest(
                total_questions=0,
                questions=[],
                recommended_workflow="No clarifications needed - proceeding with task",
                can_proceed_with_defaults=True,
                expected_quality_with_defaults="High quality expected",
            )

        # Generate questions from missing info
        questions = self._generate_questions(
            analyst_report.missing_info,
            analyst_report.literal_request,
            context
        )

        # Rank questions by priority
        questions = self._rank_questions(questions)

        # Limit to max_questions
        questions = questions[:max_questions]

        # Determine workflow and quality assessment
        total_questions = len(questions)
        has_critical = any(q.priority == QuestionPriority.CRITICAL for q in questions)

        return ClarificationRequest(
            total_questions=total_questions,
            questions=questions,
            recommended_workflow=self._determine_workflow(questions, has_critical),
            can_proceed_with_defaults=self._can_proceed_with_defaults(questions),
            expected_quality_with_defaults=self._assess_quality_with_defaults(
                questions, analyst_report
            ),
        )

    # ========================================================================
    # Question Generation Methods
    # ========================================================================

    def _generate_questions(
        self,
        missing_info: List[MissingInfo],
        user_request: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ClarificationQuestion]:
        """Generate questions from missing information."""
        questions = []

        for info in missing_info:
            # Determine priority from severity
            priority = self._severity_to_priority(info.severity)

            # Build the question
            question_text = self._build_question_text(info.requirement, info.severity)

            # Build reason
            reason = self._build_reason(info)

            # Build context for user
            context_text = self._build_context(info, user_request)

            # Use default assumption
            default = info.default_assumption or "Will use standard best practices"

            # Assess impact if unanswered
            impact = self._assess_impact(info)

            # Determine answer options if applicable
            answer_options = self._get_answer_options(info.requirement)

            questions.append(ClarificationQuestion(
                question=question_text,
                priority=priority,
                reason=reason,
                context=context_text,
                default_answer=default,
                impact_if_unanswered=ImpactAssessment(
                    quality_impact=impact["quality"],
                    risk_level=impact["risk"],
                    potential_revisions=impact["revisions"],
                ),
                answer_options=answer_options,
            ))

        return questions

    def _build_question_text(self, requirement: str, severity: SeverityLevel) -> str:
        """Build the actual question text."""
        question_templates = {
            SeverityLevel.CRITICAL: "What {requirement} should be used?",
            SeverityLevel.IMPORTANT: "Which {requirement} do you prefer?",
            SeverityLevel.NICE_TO_HAVE: "Any preference for {requirement}?",
        }

        template = question_templates.get(severity, "What {requirement}?")
        return template.format(requirement=requirement.lower())

    def _build_reason(self, info: MissingInfo) -> str:
        """Build the reason for why this question matters."""
        return f"This affects: {info.impact}"

    def _build_context(self, info: MissingInfo, user_request: str) -> str:
        """Build context for the user."""
        return f"To complete your request, I need to know about {info.requirement}."

    def _assess_impact(self, info: MissingInfo) -> Dict[str, Any]:
        """Assess the impact if this question is not answered."""
        severity = info.severity.value

        if severity == "critical":
            return {
                "quality": "Severe impact - output may be unusable",
                "risk": "high",
                "revisions": [
                    "May need to completely redo the work",
                    "Output may not meet actual requirements"
                ]
            }
        elif severity == "important":
            return {
                "quality": "Moderate impact - output quality degraded",
                "risk": "medium",
                "revisions": [
                    "May require significant revisions",
                    "Some rework likely needed"
                ]
            }
        else:  # nice_to_have
            return {
                "quality": "Minor impact - output should be acceptable",
                "risk": "low",
                "revisions": [
                    "Minor enhancements may be needed",
                    "Default approach should work"
                ]
            }

    def _get_answer_options(self, requirement: str) -> Optional[List[str]]:
        """Get answer options for common questions."""
        options_map = {
            "authentication method": ["JWT", "OAuth 2.0", "API Key", "Session-based"],
            "database technology": ["PostgreSQL", "MySQL", "MongoDB", "SQLite", "None"],
            "deployment target": ["Docker", "Kubernetes", "Cloud (AWS/Azure/GCP)", "Local"],
            "programming language": ["Python", "JavaScript/TypeScript", "Java", "Go", "C++"],
            "testing framework": ["pytest", "unittest", "Jest", "JUnit", "None"],
            "documentation format": ["Markdown", "HTML", "PDF", "DocX"],
        }

        for key, options in options_map.items():
            if key.lower() in requirement.lower():
                return options

        return None

    # ========================================================================
    # Question Ranking Methods
    # ========================================================================

    def _rank_questions(
        self,
        questions: List[ClarificationQuestion]
    ) -> List[ClarificationQuestion]:
        """Rank questions by priority."""
        # First group by priority level
        critical = [q for q in questions if q.priority == QuestionPriority.CRITICAL]
        high = [q for q in questions if q.priority == QuestionPriority.HIGH]
        medium = [q for q in questions if q.priority == QuestionPriority.MEDIUM]
        low = [q for q in questions if q.priority == QuestionPriority.LOW]

        # Sort within each priority by quality impact
        def sort_by_impact(q):
            impact = q.impact_if_unanswered
            if impact.risk_level == "high":
                return 0
            elif impact.risk_level == "medium":
                return 1
            else:
                return 2

        critical.sort(key=sort_by_impact)
        high.sort(key=sort_by_impact)
        medium.sort(key=sort_by_impact)
        low.sort(key=sort_by_impact)

        # Combine: critical first, then high, medium, low
        return critical + high + medium + low

    def _severity_to_priority(self, severity: SeverityLevel) -> QuestionPriority:
        """Convert severity level to question priority."""
        mapping = {
            SeverityLevel.CRITICAL: QuestionPriority.CRITICAL,
            SeverityLevel.IMPORTANT: QuestionPriority.HIGH,
            SeverityLevel.NICE_TO_HAVE: QuestionPriority.MEDIUM,
        }
        return mapping.get(severity, QuestionPriority.LOW)

    # ========================================================================
    # Workflow & Quality Assessment Methods
    # ========================================================================

    def _determine_workflow(
        self,
        questions: List[ClarificationQuestion],
        has_critical: bool
    ) -> str:
        """Determine the recommended workflow for getting answers."""
        if has_critical:
            return (
                "Present critical questions first. "
                "Wait for answers before proceeding with execution."
            )
        elif len(questions) <= 3:
            return (
                "Present all questions together. "
                "Can proceed with defaults if user prefers."
            )
        else:
            return (
                "Present questions in priority groups. "
                "Allow user to skip non-critical questions."
            )

    def _can_proceed_with_defaults(self, questions: List[ClarificationQuestion]) -> bool:
        """Determine if task can proceed with default assumptions."""
        # Can proceed if no critical questions, or if defaults are well-defined
        has_critical = any(q.priority == QuestionPriority.CRITICAL for q in questions)
        all_have_defaults = all(q.default_answer for q in questions)

        return not has_critical or all_have_defaults

    def _assess_quality_with_defaults(
        self,
        questions: List[ClarificationQuestion],
        analyst_report: TaskIntelligenceReport
    ) -> str:
        """Assess expected output quality if using all defaults."""
        critical_count = sum(
            1 for q in questions
            if q.priority == QuestionPriority.CRITICAL
        )
        high_count = sum(
            1 for q in questions
            if q.priority == QuestionPriority.HIGH
        )

        if critical_count > 0:
            return "Low - critical questions unanswered may cause significant issues"
        elif high_count > 2:
            return "Medium - multiple important questions using defaults"
        elif len(questions) > 0:
            return "Good - defaults should work for most cases"
        else:
            return "High - no clarifications needed"

    def _load_system_prompt(self) -> str:
        """Load the system prompt from file."""
        try:
            with open(self.system_prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            return "You are the Clarifier. Formulate questions for missing requirements."


# =============================================================================
# Convenience Functions
# =============================================================================

def create_clarifier(
    system_prompt_path: str = "config/agents/clarifier/CLAUDE.md",
    model: str = "claude-3-5-sonnet-20241022",
) -> ClarifierAgent:
    """Create a configured Clarifier agent."""
    return ClarifierAgent(
        system_prompt_path=system_prompt_path,
        model=model,
    )
