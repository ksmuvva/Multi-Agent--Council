"""
Task Analyst Subagent

Decomposes user requests into structured requirements using
the TaskIntelligenceReport schema.
"""

import os
import re
from typing import List, Dict, Any, Optional
from pathlib import Path

from src.schemas.analyst import (
    TaskIntelligenceReport,
    SubTask,
    MissingInfo,
    SeverityLevel,
    ModalityType,
)


class AnalystAgent:
    """
    The Task Analyst decomposes user requests into structured requirements.

    Key responsibilities:
    - Extract literal request and inferred intent
    - Break down into sub-tasks
    - Identify missing information by severity
    - Detect input modality
    - Recommend an approach
    """

    def __init__(
        self,
        system_prompt_path: str = "config/agents/analyst/CLAUDE.md",
        model: str = "claude-3-5-sonnet-20241022",
        max_turns: int = 30,
    ):
        """
        Initialize the Analyst agent.

        Args:
            system_prompt_path: Path to system prompt file
            model: Model to use for analysis
            max_turns: Maximum conversation turns
        """
        self.system_prompt_path = system_prompt_path
        self.model = model
        self.max_turns = max_turns
        self.system_prompt = self._load_system_prompt()

        # Modality detection patterns
        self.modality_patterns = {
            ModalityType.CODE: [
                r'\.(py|js|ts|java|cpp|c|go|rs|rb|php)$',
                r'function|class|def |import |from ',
                r'write a function|create a class|implement',
            ],
            ModalityType.IMAGE: [
                r'\.(png|jpg|jpeg|gif|webp|svg)$',
                r'image|photo|diagram|screenshot',
            ],
            ModalityType.DOCUMENT: [
                r'\.(pdf|docx|doc|xlsx|pptx|txt)$',
                r'document|report|spreadsheet|presentation',
            ],
            ModalityType.DATA: [
                r'\.(json|yaml|yml|xml|csv|sql)$',
                r'data|dataset|database|query',
            ],
        }

    def analyze(
        self,
        user_request: str,
        context: Optional[Dict[str, Any]] = None,
        file_attachments: Optional[List[str]] = None,
    ) -> TaskIntelligenceReport:
        """
        Analyze a user request and produce a TaskIntelligenceReport.

        Args:
            user_request: The user's raw request
            context: Additional context (session, previous requests, etc.)
            file_attachments: Optional list of attached file paths

        Returns:
            TaskIntelligenceReport with complete analysis
        """
        # Enhance request with file info if provided
        enhanced_request = self._prepare_request(user_request, file_attachments)

        # Detect modality
        modality = self._detect_modality(enhanced_request, file_attachments)

        # Extract literal and inferred intent
        literal_request = self._extract_literal_request(user_request)
        inferred_intent = self._infer_intent(user_request, context)

        # Decompose into sub-tasks
        sub_tasks = self._decompose_tasks(user_request, inferred_intent, modality)

        # Identify missing information
        missing_info = self._identify_missing_info(user_request, sub_tasks, modality)

        # Generate assumptions
        assumptions = self._generate_assumptions(user_request, missing_info)

        # Determine recommended approach
        recommended_approach = self._recommend_approach(
            user_request, sub_tasks, modality, missing_info
        )

        # Determine tier suggestion
        suggested_tier = self._suggest_tier(
            user_request, sub_tasks, missing_info
        )

        # Check for escalation needs
        escalation_needed = self._check_escalation(user_request, sub_tasks)

        return TaskIntelligenceReport(
            literal_request=literal_request,
            inferred_intent=inferred_intent,
            sub_tasks=sub_tasks,
            missing_info=missing_info,
            assumptions=assumptions,
            modality=modality,
            recommended_approach=recommended_approach,
            suggested_tier=suggested_tier,
            escalation_needed=escalation_needed,
            confidence=self._calculate_confidence(sub_tasks, missing_info),
        )

    # ========================================================================
    # Analysis Methods
    # ========================================================================

    def _prepare_request(
        self,
        user_request: str,
        file_attachments: Optional[List[str]] = None
    ) -> str:
        """Prepare the request with file context."""
        if file_attachments:
            file_info = "\n".join([
                f"[File: {Path(f).name}]"
                for f in file_attachments
            ])
            return f"{user_request}\n\n{file_info}"
        return user_request

    def _detect_modality(
        self,
        request: str,
        file_attachments: Optional[List[str]] = None
    ) -> ModalityType:
        """Detect the input/output modality."""
        request_lower = request.lower()

        # Check file extensions first
        if file_attachments:
            for file_path in file_attachments:
                file_ext = Path(file_path).suffix.lower()
                if file_ext in ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.go', '.rs']:
                    return ModalityType.CODE
                elif file_ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']:
                    return ModalityType.IMAGE
                elif file_ext in ['.pdf', '.docx', '.doc', '.xlsx', '.pptx']:
                    return ModalityType.DOCUMENT
                elif file_ext in ['.json', '.yaml', '.yml', '.xml', '.csv']:
                    return ModalityType.DATA

        # Check patterns in request
        for modality, patterns in self.modality_patterns.items():
            for pattern in patterns:
                if re.search(pattern, request_lower):
                    return modality

        # Default to text
        return ModalityType.TEXT

    def _extract_literal_request(self, user_request: str) -> str:
        """Extract the exact literal request."""
        return user_request.strip()

    def _infer_intent(
        self,
        user_request: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Infer the actual intent behind the request.

        This interprets what the user actually wants,
        which may differ from the literal wording.
        """
        request_lower = user_request.lower()

        # Common intent patterns
        intent_patterns = {
            "create": "User wants to generate or create something new",
            "fix": "User wants to resolve a problem or bug",
            "explain": "User wants understanding or clarification",
            "improve": "User wants to enhance existing code/process",
            "analyze": "User wants detailed examination or insight",
            "convert": "User wants to transform one format to another",
            "compare": "User wants to understand differences",
            "implement": "User wants to build a specific feature/system",
        }

        # Check for intent keywords
        for keyword, intent in intent_patterns.items():
            if keyword in request_lower:
                return intent

        # Default: user wants help with the stated request
        return f"User wants assistance with: {user_request[:100]}..."

    def _decompose_tasks(
        self,
        user_request: str,
        intent: str,
        modality: ModalityType
    ) -> List[SubTask]:
        """
        Decompose the request into sub-tasks.

        Breaks down the work into manageable steps with dependencies.
        """
        request_lower = user_request.lower()
        sub_tasks = []

        # Task decomposition patterns
        if "api" in request_lower or "endpoint" in request_lower:
            sub_tasks.extend([
                SubTask(
                    description="Define data models and schema",
                    dependencies=[],
                    estimated_complexity="medium"
                ),
                SubTask(
                    description="Design API endpoints and routes",
                    dependencies=["Define data models and schema"],
                    estimated_complexity="high"
                ),
                SubTask(
                    description="Implement endpoint logic",
                    dependencies=["Design API endpoints and routes"],
                    estimated_complexity="high"
                ),
                SubTask(
                    description="Add authentication and validation",
                    dependencies=["Implement endpoint logic"],
                    estimated_complexity="medium"
                ),
            ])
        elif "test" in request_lower:
            sub_tasks.extend([
                SubTask(
                    description="Analyze code to identify test scenarios",
                    dependencies=[],
                    estimated_complexity="medium"
                ),
                SubTask(
                    description="Design test cases with coverage",
                    dependencies=["Analyze code to identify test scenarios"],
                    estimated_complexity="medium"
                ),
                SubTask(
                    description="Implement test code",
                    dependencies=["Design test cases with coverage"],
                    estimated_complexity="high"
                ),
            ])
        elif "document" in request_lower or "docs" in request_lower:
            sub_tasks.extend([
                SubTask(
                    description="Analyze code/features to document",
                    dependencies=[],
                    estimated_complexity="medium"
                ),
                SubTask(
                    description="Structure documentation outline",
                    dependencies=["Analyze code/features to document"],
                    estimated_complexity="low"
                ),
                SubTask(
                    description="Generate documentation content",
                    dependencies=["Structure documentation outline"],
                    estimated_complexity="medium"
                ),
            ])
        elif "bug" in request_lower or "fix" in request_lower or "error" in request_lower:
            sub_tasks.extend([
                SubTask(
                    description="Analyze the error/bug",
                    dependencies=[],
                    estimated_complexity="medium"
                ),
                SubTask(
                    description="Identify root cause",
                    dependencies=["Analyze the error/bug"],
                    estimated_complexity="high"
                ),
                SubTask(
                    description="Implement fix",
                    dependencies=["Identify root cause"],
                    estimated_complexity="high"
                ),
                SubTask(
                    description="Verify fix resolves issue",
                    dependencies=["Implement fix"],
                    estimated_complexity="medium"
                ),
            ])
        else:
            # Generic decomposition
            sub_tasks.append(SubTask(
                description=f"Understand and analyze requirements",
                dependencies=[],
                estimated_complexity="low"
            ))
            sub_tasks.append(SubTask(
                description=f"Develop solution for: {intent}",
                dependencies=["Understand and analyze requirements"],
                estimated_complexity="high"
            ))
            sub_tasks.append(SubTask(
                description="Review and validate output",
                dependencies=["Develop solution"],
                estimated_complexity="medium"
            ))

        return sub_tasks

    def _identify_missing_info(
        self,
        user_request: str,
        sub_tasks: List[SubTask],
        modality: ModalityType
    ) -> List[MissingInfo]:
        """
        Identify missing information categorized by severity.
        """
        request_lower = user_request.lower()
        missing_info = []

        # Check for common missing information
        # Critical missing info
        if "api" in request_lower and "auth" not in request_lower:
            missing_info.append(MissingInfo(
                requirement="Authentication method",
                severity=SeverityLevel.CRITICAL,
                impact="Security architecture depends on authentication approach",
                default_assumption="JWT-based authentication"
            ))

        if "database" in request_lower or "db" in request_lower:
            if "postgres" not in request_lower and "mysql" not in request_lower and "mongo" not in request_lower:
                missing_info.append(MissingInfo(
                    requirement="Database technology",
                    severity=SeverityLevel.CRITICAL,
                    impact="Schema design and queries depend on database choice",
                    default_assumption="PostgreSQL"
                ))

        # Important missing info
        if "deploy" in request_lower or "production" in request_lower:
            if "cloud" not in request_lower:
                missing_info.append(MissingInfo(
                    requirement="Deployment target/platform",
                    severity=SeverityLevel.IMPORTANT,
                    impact="Deployment configuration varies by platform",
                    default_assumption="Docker containers"
                ))

        # Nice-to-have missing info
        if "test" not in request_lower:
            missing_info.append(MissingInfo(
                requirement="Testing requirements",
                severity=SeverityLevel.NICE_TO_HAVE,
                impact="Could add test coverage for quality assurance",
                default_assumption="Unit tests for core functionality"
            ))

        return missing_info

    def _generate_assumptions(
        self,
        user_request: str,
        missing_info: List[MissingInfo]
    ) -> List[str]:
        """Generate assumptions to proceed with work."""
        assumptions = []

        # Add assumptions from missing info defaults
        for info in missing_info:
            if info.default_assumption:
                assumptions.append(
                    f"Assuming {info.requirement}: {info.default_assumption}"
                )

        # Add common assumptions
        assumptions.append("User has necessary permissions/access")
        assumptions.append("Standard best practices apply unless specified")
        assumptions.append("Code follows project conventions")

        return assumptions

    def _recommend_approach(
        self,
        user_request: str,
        sub_tasks: List[SubTask],
        modality: ModalityType,
        missing_info: List[MissingInfo]
    ) -> str:
        """Recommend the best approach for completing this task."""
        request_lower = user_request.lower()

        # Critical missing info affects approach
        critical_missing = [m for m in missing_info if m.severity == SeverityLevel.CRITICAL]
        if critical_missing:
            return (
                "FIRST: Clarify critical requirements with user. "
                "THEN: Proceed with standard execution."
            )

        # Modality-specific recommendations
        if modality == ModalityType.CODE:
            return "Use code-generation skills with Executor, validate with Code Reviewer"
        elif modality == ModalityType.DOCUMENT:
            return "Use document-creation skill with Formatter for final output"
        elif modality == ModalityType.DATA:
            return "Use data-analysis skills, validate with Verifier for accuracy"

        # Task-type specific recommendations
        if "api" in request_lower:
            return "Design schema first, then implement endpoints, add authentication last"
        elif "test" in request_lower:
            return "Analyze coverage gaps, design test cases, implement with test framework"
        elif "bug" in request_lower:
            return "Reproduce issue first, then trace to root cause, then implement fix"

        # Default recommendation
        return "Follow standard pipeline: analyze → plan → execute → verify → format"

    def _suggest_tier(
        self,
        user_request: str,
        sub_tasks: List[SubTask],
        missing_info: List[MissingInfo]
    ) -> int:
        """Suggest the appropriate complexity tier."""
        request_lower = user_request.lower()

        # Tier 4 indicators
        tier_4_keywords = ["security", "compliance", "adversarial", "attack", "critical"]
        if any(kw in request_lower for kw in tier_4_keywords):
            return 4

        # Tier 3 indicators
        tier_3_keywords = ["architecture", "design pattern", "domain expert", "specialist"]
        if any(kw in request_lower for kw in tier_3_keywords):
            return 3

        # Tier 2 indicators
        if len(sub_tasks) > 2 or len(missing_info) > 0:
            return 2

        # Default to Tier 1 for simple requests
        return 1

    def _check_escalation(self, user_request: str, sub_tasks: List[SubTask]) -> bool:
        """Check if escalation might be needed."""
        request_lower = user_request.lower()
        escalation_keywords = [
            "complex", "complicated", "uncertain", "may need",
            "depends on", "multiple factors"
        ]
        return any(kw in request_lower for kw in escalation_keywords)

    def _calculate_confidence(
        self,
        sub_tasks: List[SubTask],
        missing_info: List[MissingInfo]
    ) -> float:
        """Calculate confidence in the analysis."""
        # Start with base confidence
        confidence = 0.8

        # Reduce confidence for many missing items
        critical_missing = len([m for m in missing_info if m.severity == SeverityLevel.CRITICAL])
        confidence -= critical_missing * 0.1

        # Increase confidence for well-structured tasks
        if len(sub_tasks) >= 3:
            confidence += 0.1

        return max(0.0, min(1.0, confidence))

    def _load_system_prompt(self) -> str:
        """Load the system prompt from file."""
        try:
            with open(self.system_prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            return "You are the Task Analyst. Decompose user requests into structured requirements."


# =============================================================================
# Convenience Functions
# =============================================================================

def create_analyst(
    system_prompt_path: str = "config/agents/analyst/CLAUDE.md",
    model: str = "claude-3-5-sonnet-20241022",
) -> AnalystAgent:
    """Create a configured Analyst agent."""
    return AnalystAgent(
        system_prompt_path=system_prompt_path,
        model=model,
    )
