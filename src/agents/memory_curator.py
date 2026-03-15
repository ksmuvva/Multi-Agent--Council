"""
Memory Curator Subagent (Knowledge Extraction & Persistence)

Extracts and preserves knowledge from completed tasks for future reuse.
Writes knowledge files with YAML frontmatter to docs/knowledge/.
"""

import re
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from src.utils.logging import get_agent_logger, AgentLogContext
from src.utils.events import emit_agent_started, emit_agent_completed, emit_error

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


class KnowledgeCategory(str, Enum):
    """Categories of knowledge to capture."""
    ARCHITECTURAL_DECISION = "architectural_decision"
    CODE_PATTERN = "code_pattern"
    DOMAIN_INSIGHT = "domain_insight"
    TROUBLESHOOTING = "troubleshooting"
    BEST_PRACTICE = "best_practice"
    ANTI_PATTERN = "anti_pattern"
    LESSON_LEARNED = "lesson_learned"
    REFERENCE = "reference"


@dataclass
class KeyDecision:
    """A key decision made during task execution."""
    decision: str
    reasoning: str
    alternatives_considered: List[str]
    context: str


@dataclass
class Pattern:
    """A reusable pattern or approach."""
    name: str
    description: str
    when_to_use: str
    example: Optional[str]
    related_patterns: List[str]


@dataclass
class KnowledgeEntry:
    """A knowledge entry to persist."""
    topic: str
    category: KnowledgeCategory
    summary: str
    key_decisions: List[KeyDecision]
    patterns: List[Pattern]
    domain_insights: List[str]
    lessons_learned: List[str]
    references: List[str]
    related_topics: List[str]
    tags: List[str]


@dataclass
class ExtractionResult:
    """Result of knowledge extraction."""
    entries: List[KnowledgeEntry]
    total_extractions: int
    topics_created: List[str]
    extraction_metadata: Dict[str, Any]


class MemoryCuratorAgent:
    """
    The Memory Curator extracts and preserves knowledge.

    Key responsibilities:
    - Extract key decisions from completed tasks
    - Identify reusable patterns
    - Capture domain insights
    - Document lessons learned
    - Write knowledge files with YAML frontmatter
    - Enable future knowledge retrieval
    """

    def __init__(
        self,
        system_prompt_path: str = "config/agents/memory_curator/CLAUDE.md",
        model: str = "claude-3-5-sonnet-20241022",
        max_turns: int = 30,
        knowledge_dir: str = "docs/knowledge",
    ):
        """
        Initialize the Memory Curator agent.

        Args:
            system_prompt_path: Path to system prompt file
            model: Model to use for extraction
            max_turns: Maximum conversation turns
            knowledge_dir: Directory for knowledge files
        """
        self.system_prompt_path = system_prompt_path
        self.model = model
        self.max_turns = max_turns
        self.logger = get_agent_logger("memory_curator")
        self.system_prompt = self._load_system_prompt()
        self.knowledge_dir = Path(knowledge_dir)

        # Ensure knowledge directory exists
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info("MemoryCuratorAgent initialized", model=model, knowledge_dir=knowledge_dir)

        # Patterns that indicate important decisions
        self.decision_patterns = [
            r"(?:chose|choosing|selected|decided|opting|going with)\s+(\w+(?:\s+\w+)?)",
            r"(?:over|rather than|instead of|versus|vs\.)\s+(\w+(?:\s+\w+)?)",
            r"(?:why|because|reason|rationale)",
            r"(?:alternative|option|approach)",
        ]

        # Patterns that indicate reusable patterns
        self.pattern_indicators = [
            "pattern", "approach", "technique", "method",
            "strategy", "template", "idiom", "practice"
        ]

        # Patterns that indicate lessons learned
        self.lesson_patterns = [
            r"(?:learned|realized|discovered|found)",
            r"(?:worked well|successful|effective)",
            r"(?:didn't work|failed|mistake|issue|problem)",
            r"(?:avoid|prefer|recommend)",
        ]

    def extract_and_preserve(
        self,
        task_description: str,
        execution_context: Dict[str, Any],
        agent_outputs: Dict[str, Any],
        final_output: str,
        session_id: Optional[str] = None,
    ) -> ExtractionResult:
        """
        Extract knowledge from completed task and preserve it.

        Args:
            task_description: The original task/request
            execution_context: Context about task execution (tier, agents used, etc.)
            agent_outputs: Outputs from all agents involved
            final_output: The final output delivered to user
            session_id: Optional session ID for tracking

        Returns:
            ExtractionResult with all knowledge entries and metadata
        """
        self.logger.info(
            "Knowledge curation started",
            task_preview=task_description[:100],
            session_id=session_id,
            tier_level=execution_context.get("tier_level"),
            agents_involved=execution_context.get("agents_used", []),
        )
        emit_agent_started("memory_curator", phase="curation")

        # Step 1: Extract key decisions
        key_decisions = self._extract_key_decisions(
            task_description, agent_outputs, final_output
        )

        self.logger.debug("Key decisions extracted", count=len(key_decisions))

        # Step 2: Identify patterns
        patterns = self._identify_patterns(
            agent_outputs, final_output
        )
        self.logger.debug("Patterns identified", count=len(patterns))

        # Step 3: Capture domain insights
        domain_insights = self._capture_domain_insights(
            task_description, agent_outputs
        )
        self.logger.debug("Domain insights captured", count=len(domain_insights))

        # Step 4: Document lessons learned
        lessons_learned = self._document_lessons(
            execution_context, agent_outputs
        )
        self.logger.debug("Lessons documented", count=len(lessons_learned))

        # Step 5: Create knowledge entries
        entries = self._create_knowledge_entries(
            task_description,
            key_decisions,
            patterns,
            domain_insights,
            lessons_learned,
            execution_context,
        )

        # Step 6: Write knowledge files
        topics_created = []
        for entry in entries:
            topic_file = self._write_knowledge_file(entry)
            if topic_file:
                topics_created.append(topic_file)
                self.logger.debug("Knowledge file written", filename=topic_file, topic=entry.topic)
            else:
                self.logger.warning("Failed to write knowledge file", topic=entry.topic)

        self.logger.info(
            "Knowledge curation completed",
            entries_created=len(entries),
            topics_created=topics_created,
            total_extractions=sum(len(e.key_decisions) + len(e.patterns) + len(e.domain_insights) for e in entries),
        )
        emit_agent_completed("memory_curator", output_summary=f"Created {len(entries)} knowledge entries")

        # Step 7: Build metadata
        extraction_metadata = {
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tier_level": execution_context.get("tier_level"),
            "agents_involved": execution_context.get("agents_used", []),
            "entries_created": len(entries),
            "topics_created": topics_created,
        }

        return ExtractionResult(
            entries=entries,
            total_extractions=sum(len(e.key_decisions) + len(e.patterns) + len(e.domain_insights) for e in entries),
            topics_created=topics_created,
            extraction_metadata=extraction_metadata,
        )

    # ========================================================================
    # Knowledge Extraction
    # ========================================================================

    def _extract_key_decisions(
        self,
        task_description: str,
        agent_outputs: Dict[str, Any],
        final_output: str,
    ) -> List[KeyDecision]:
        """Extract key decisions from task execution."""
        decisions = []

        # Check Planner output for decisions
        if "planner" in agent_outputs:
            planner_output = str(agent_outputs["planner"])
            decisions.extend(self._parse_decisions_from_text(planner_output))

        # Check Executor output for implementation decisions
        if "executor" in agent_outputs:
            executor_output = str(agent_outputs["executor"])
            decisions.extend(self._parse_decisions_from_text(executor_output))

        # Check task description for constraints/requirements
        decisions.extend(self._parse_decisions_from_text(task_description))

        # Deduplicate by decision text
        seen = set()
        unique_decisions = []
        for decision in decisions:
            decision_key = decision.decision.lower().strip()
            if decision_key and decision_key not in seen:
                seen.add(decision_key)
                unique_decisions.append(decision)

        return unique_decisions[:10]  # Limit to top 10

    def _parse_decisions_from_text(self, text: str) -> List[KeyDecision]:
        """Parse decisions from text output."""
        decisions = []

        # Look for decision statements
        decision_sentences = re.finditer(
            r'(?:We|I|The system)?\s*(?:chose|decided|selected|opt(?:s|ed)|went with)\s+[^.!?]+[.!?]',
            text,
            re.IGNORECASE
        )

        for match in decision_sentences:
            sentence = match.group().strip()
            reasoning = self._extract_reasoning(sentence, text)
            alternatives = self._extract_alternatives(sentence, text)

            decisions.append(KeyDecision(
                decision=sentence,
                reasoning=reasoning,
                alternatives_considered=alternatives,
                context=self._get_context(sentence, text),
            ))

        return decisions

    def _extract_reasoning(self, decision: str, text: str) -> str:
        """Extract reasoning for a decision."""
        # Look for "because", "since", "due to", "as" near the decision
        reasoning_patterns = [
            r'(?:because|since|due to|as)\s+([^.!?]+)',
            r'(?:reason|rationale|why):\s*([^.!?]+)',
        ]

        for pattern in reasoning_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Check if reasoning is near the decision in text
                reasoning = match.group(1).strip()
                if reasoning and len(reasoning) > 5:
                    return reasoning

        return "Decision made based on task requirements"

    def _extract_alternatives(self, decision: str, text: str) -> List[str]:
        """Extract alternatives that were considered."""
        alternatives = []

        # Look for "instead of", "rather than", "versus", "vs"
        alt_patterns = [
            r'(?:instead of|rather than)\s+(\w+(?:\s+\w+)?)',
            r'(?:versus|vs\.)\s+(\w+(?:\s+\w+)?)',
            r'(?:alternative|option):\s*(\w+(?:\s+\w+)?)',
        ]

        for pattern in alt_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                alt = match.group(1).strip()
                if alt and alt not in alternatives:
                    alternatives.append(alt)

        return alternatives[:3]  # Limit to 3 alternatives

    def _get_context(self, decision: str, text: str) -> str:
        """Get context around a decision."""
        # Find the decision in text and get surrounding sentences
        decision_index = text.find(decision)
        if decision_index == -1:
            return ""

        # Get up to 100 chars before and after
        start = max(0, decision_index - 100)
        end = min(len(text), decision_index + len(decision) + 100)
        context = text[start:end].strip()

        # Clean up
        context = re.sub(r'\s+', ' ', context)
        return context[:200]  # Limit to 200 chars

    # ========================================================================
    # Pattern Identification
    # ========================================================================

    def _identify_patterns(
        self,
        agent_outputs: Dict[str, Any],
        final_output: str,
    ) -> List[Pattern]:
        """Identify reusable patterns from execution."""
        patterns = []

        # Check for code patterns in Executor output
        if "executor" in agent_outputs:
            executor_output = str(agent_outputs["executor"])
            patterns.extend(self._extract_code_patterns(executor_output))

        # Check for workflow patterns in Planner output
        if "planner" in agent_outputs:
            planner_output = str(agent_outputs["planner"])
            patterns.extend(self._extract_workflow_patterns(planner_output))

        # Check for architectural patterns
        for agent_name, output in agent_outputs.items():
            patterns.extend(self._extract_architectural_patterns(str(output)))

        # Deduplicate
        seen = set()
        unique_patterns = []
        for pattern in patterns:
            key = pattern.name.lower()
            if key and key not in seen:
                seen.add(key)
                unique_patterns.append(pattern)

        return unique_patterns[:8]  # Limit to top 8

    def _extract_code_patterns(self, text: str) -> List[Pattern]:
        """Extract code patterns from text."""
        patterns = []

        # Look for function/class definitions
        code_patterns = [
            (r'def\s+(\w+)', 'Function pattern'),
            (r'class\s+(\w+)', 'Class pattern'),
            (r'async\s+def\s+(\w+)', 'Async function pattern'),
        ]

        for pattern, description in code_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                name = match.group(1)
                patterns.append(Pattern(
                    name=name,
                    description=f"{description}: {name}",
                    when_to_use=self._infer_when_to_use(name, text),
                    example=None,  # Could extract actual code
                    related_patterns=[],
                ))

        return patterns

    def _extract_workflow_patterns(self, text: str) -> List[Pattern]:
        """Extract workflow/process patterns."""
        patterns = []

        # Look for step-by-step patterns
        step_matches = re.finditer(
            r'(?:step|phase|stage)\s+\d+:\s*([^.!?]+)',
            text,
            re.IGNORECASE
        )

        steps = [match.group(1).strip() for match in step_matches]

        if len(steps) >= 3:  # Only if it's a real pattern
            pattern_name = f"{len(steps)}-step workflow"
            patterns.append(Pattern(
                name=pattern_name,
                description=f"Process pattern with {len(steps)} steps",
                when_to_use="When following structured workflows",
                example=None,
                related_patterns=[],
            ))

        return patterns

    def _extract_architectural_patterns(self, text: str) -> List[Pattern]:
        """Extract architectural patterns."""
        patterns = []

        # Common architectural pattern keywords
        arch_keywords = {
            "mvc": "Model-View-Controller",
            "observer": "Observer Pattern",
            "singleton": "Singleton Pattern",
            "factory": "Factory Pattern",
            "repository": "Repository Pattern",
            "adapter": "Adapter Pattern",
            "decorator": "Decorator Pattern",
            "strategy": "Strategy Pattern",
        }

        text_lower = text.lower()

        for keyword, full_name in arch_keywords.items():
            if keyword in text_lower:
                patterns.append(Pattern(
                    name=full_name,
                    description=f"Architectural pattern: {full_name}",
                    when_to_use=self._infer_when_to_use_architectural(keyword),
                    example=None,
                    related_patterns=[],
                ))

        return patterns

    def _infer_when_to_use(self, name: str, context: str) -> str:
        """Infer when a pattern should be used."""
        # Simple heuristic based on name
        if "async" in name.lower():
            return "For asynchronous operations"
        elif "get" in name.lower():
            return "For retrieving data"
        elif "set" in name.lower() or "update" in name.lower():
            return "For modifying data"
        else:
            return "General purpose"

    def _infer_when_to_use_architectural(self, keyword: str) -> str:
        """Infer when to use an architectural pattern."""
        use_cases = {
            "mvc": "For separating concerns in UI applications",
            "observer": "For event-driven architectures",
            "singleton": "For single-instance components",
            "factory": "For object creation with flexibility",
            "repository": "For data access abstraction",
            "adapter": "For integrating incompatible interfaces",
            "decorator": "For adding behavior dynamically",
            "strategy": "For interchangeable algorithms",
        }
        return use_cases.get(keyword, "General architectural use")

    # ========================================================================
    # Domain Insights
    # ========================================================================

    def _capture_domain_insights(
        self,
        task_description: str,
        agent_outputs: Dict[str, Any],
    ) -> List[str]:
        """Capture domain-specific insights."""
        insights = []

        # Extract from Researcher if available
        if "researcher" in agent_outputs:
            researcher_output = str(agent_outputs["researcher"])
            insights.extend(self._extract_insights_from_research(researcher_output))

        # Extract from task description
        insights.extend(self._extract_insights_from_task(task_description))

        # Extract from Verifier (factual claims)
        if "verifier" in agent_outputs:
            verifier_output = str(agent_outputs["verifier"])
            insights.extend(self._extract_insights_from_verification(verifier_output))

        # Deduplicate and clean
        seen = set()
        unique_insights = []
        for insight in insights:
            cleaned = insight.strip()
            if cleaned and len(cleaned) > 10 and cleaned not in seen:
                seen.add(cleaned)
                unique_insights.append(cleaned)

        return unique_insights[:15]  # Limit to top 15

    def _extract_insights_from_research(self, text: str) -> List[str]:
        """Extract insights from research output."""
        insights = []

        # Look for findings and evidence
        finding_patterns = [
            r'(?:finding|discovered|found|identified):\s*([^.!?]+)',
            r'(?:evidence|research|study)\s+(?:shows?|indicates?|suggests?):\s*([^.!?]+)',
        ]

        for pattern in finding_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                insight = match.group(1).strip()
                if insight:
                    insights.append(insight)

        return insights

    def _extract_insights_from_task(self, text: str) -> List[str]:
        """Extract domain insights from task description."""
        insights = []

        # Look for domain-specific terms and requirements
        # This is a simplified approach - could use NLP in production

        # Extract quoted text (often domain concepts)
        quoted = re.findall(r'"([^"]+)"', text)
        insights.extend(quoted)

        # Extract statements about "is", "are", "should" (definitions)
        definition_patterns = [
            r'(\w+(?:\s+\w+)?)\s+is\s+(?:a|an|the)\s+([^.!?]+)',
            r'(\w+(?:\s+\w+)?)\s+should\s+([^.!?]+)',
        ]

        for pattern in definition_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                subject = match.group(1)
                predicate = match.group(2)
                insight = f"{subject} {predicate}"
                insights.append(insight)

        return insights

    def _extract_insights_from_verification(self, text: str) -> List[str]:
        """Extract insights from verification output."""
        insights = []

        # Look for verified claims
        verified_patterns = [
            r'(?:verified|confirmed):\s*([^.!?]+)',
            r'(?:claim|statement):\s*([^.!?]+)',
        ]

        for pattern in verified_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                insight = match.group(1).strip()
                if insight:
                    insights.append(f"Verified: {insight}")

        return insights

    # ========================================================================
    # Lessons Learned
    # ========================================================================

    def _document_lessons(
        self,
        execution_context: Dict[str, Any],
        agent_outputs: Dict[str, Any],
    ) -> List[str]:
        """Document lessons learned from execution."""
        lessons = []

        # Extract from Critic if available (what went wrong)
        if "critic" in agent_outputs:
            critic_output = str(agent_outputs["critic"])
            lessons.extend(self._extract_lessons_from_critic(critic_output))

        # Extract from Code Reviewer if applicable
        if "code_reviewer" in agent_outputs:
            reviewer_output = str(agent_outputs["code_reviewer"])
            lessons.extend(self._extract_lessons_from_code_review(reviewer_output))

        # Check execution context for issues
        if execution_context.get("revisions", 0) > 0:
            lessons.append(f"Required {execution_context['revisions']} revision(s) to achieve quality")

        # Check for escalations
        if execution_context.get("escalations", 0) > 0:
            lessons.append(f"Task was escalated {execution_context['escalations']} time(s)")

        # Check for SME involvement
        if execution_context.get("smes_involved"):
            for sme in execution_context["smes_involved"]:
                lessons.append(f"SME '{sme}' consultation was valuable")

        # Deduplicate
        seen = set()
        unique_lessons = []
        for lesson in lessons:
            cleaned = lesson.strip()
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                unique_lessons.append(cleaned)

        return unique_lessons[:10]  # Limit to top 10

    def _extract_lessons_from_critic(self, text: str) -> List[str]:
        """Extract lessons from critic output."""
        lessons = []

        # Look for issues and recommendations
        issue_patterns = [
            r'(?:issue|problem|weakness):\s*([^.!?]+)',
            r'(?:improvement|recommend|suggest):\s*([^.!?]+)',
        ]

        for pattern in issue_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                lesson = match.group(1).strip()
                if lesson:
                    lessons.append(f"Address: {lesson}")

        return lessons

    def _extract_lessons_from_code_review(self, text: str) -> List[str]:
        """Extract lessons from code review output."""
        lessons = []

        # Look for findings
        finding_patterns = [
            r'(?:security|performance|style)\s+(?:issue|concern):\s*([^.!?]+)',
            r'(?:recommend|suggest):\s*([^.!?]+)',
        ]

        for pattern in finding_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                lesson = match.group(1).strip()
                if lesson:
                    lessons.append(f"Code quality: {lesson}")

        return lessons

    # ========================================================================
    # Knowledge Entry Creation
    # ========================================================================

    def _create_knowledge_entries(
        self,
        task_description: str,
        key_decisions: List[KeyDecision],
        patterns: List[Pattern],
        domain_insights: List[str],
        lessons_learned: List[str],
        execution_context: Dict[str, Any],
    ) -> List[KnowledgeEntry]:
        """Create knowledge entries from extracted information."""
        entries = []

        # Infer category from context
        category = self._infer_category(task_description, execution_context)

        # Generate topic name from task
        topic = self._generate_topic(task_description, category)

        # Generate tags
        tags = self._generate_tags(task_description, execution_context)

        # Find related topics (simplified - could use semantic search)
        related_topics = self._find_related_topics(tags)

        # Create main entry
        entry = KnowledgeEntry(
            topic=topic,
            category=category,
            summary=self._generate_summary(task_description, key_decisions, domain_insights),
            key_decisions=key_decisions,
            patterns=patterns,
            domain_insights=domain_insights,
            lessons_learned=lessons_learned,
            references=[],
            related_topics=related_topics,
            tags=tags,
        )

        entries.append(entry)

        # Create additional entries if there are distinct subtopics
        if len(patterns) >= 3:
            # Create a separate patterns entry
            patterns_entry = KnowledgeEntry(
                topic=f"{topic}-patterns",
                category=KnowledgeCategory.CODE_PATTERN,
                summary=f"Reusable patterns identified during: {topic}",
                key_decisions=[],
                patterns=patterns,
                domain_insights=[],
                lessons_learned=[],
                references=[],
                related_topics=[topic],
                tags=tags + ["patterns", "reuse"],
            )
            entries.append(patterns_entry)

        return entries

    def _infer_category(
        self,
        task_description: str,
        context: Dict[str, Any],
    ) -> KnowledgeCategory:
        """Infer the category of the knowledge entry."""
        task_lower = task_description.lower()

        # Check for category indicators
        if any(word in task_lower for word in ["code", "function", "class", "implement", "program"]):
            return KnowledgeCategory.CODE_PATTERN
        elif any(word in task_lower for word in ["architecture", "design", "structure", "component"]):
            return KnowledgeCategory.ARCHITECTURAL_DECISION
        elif any(word in task_lower for word in ["fix", "bug", "error", "issue", "problem"]):
            return KnowledgeCategory.TROUBLESHOOTING
        elif any(word in task_lower for word in ["avoid", "don't", "mistake", "wrong"]):
            return KnowledgeCategory.ANTI_PATTERN
        else:
            return KnowledgeCategory.LESSON_LEARNED

    def _generate_topic(self, task_description: str, category: KnowledgeCategory) -> str:
        """Generate a topic name from task description."""
        # Extract key phrase from task
        words = task_description.split()

        # Take first few meaningful words
        meaningful_words = [
            w for w in words
            if len(w) > 3 and w.lower() not in
            ["the", "this", "that", "with", "from", "have", "been", "were"]
        ]

        if meaningful_words:
            # Join first 3-5 words and convert to kebab-case
            topic = "-".join(meaningful_words[:5]).lower()
            # Clean up
            topic = re.sub(r'[^\w-]', '', topic)
            return topic[:50]  # Limit length

        return f"task-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    def _generate_tags(
        self,
        task_description: str,
        context: Dict[str, Any],
    ) -> List[str]:
        """Generate tags for the knowledge entry."""
        tags = []

        # Add tier level
        tier = context.get("tier_level")
        if tier:
            tags.append(f"tier-{tier}")

        # Add agents involved
        agents = context.get("agents_used", [])
        for agent in agents[:3]:  # Limit to first 3
            tags.append(agent.lower().replace(" ", "-"))

        # Extract key terms from task
        task_words = re.findall(r'\b\w{4,}\b', task_description.lower())
        # Get most common words (simplified)
        from collections import Counter
        word_counts = Counter(task_words)
        for word, _ in word_counts.most_common(5):
            tags.append(word)

        # Deduplicate
        return list(set(tags))

    def _find_related_topics(self, tags: List[str]) -> List[str]:
        """Find related knowledge topics by tags."""
        # In a real implementation, this would search existing knowledge files
        # For now, return empty list
        return []

    def _generate_summary(
        self,
        task_description: str,
        key_decisions: List[KeyDecision],
        domain_insights: List[str],
    ) -> str:
        """Generate a summary for the knowledge entry."""
        summary_parts = []

        summary_parts.append(f"Task: {task_description[:100]}...")

        if key_decisions:
            summary_parts.append(f"{len(key_decisions)} key decision(s) made")

        if domain_insights:
            summary_parts.append(f"{len(domain_insights)} insight(s) captured")

        return ". ".join(summary_parts)

    # ========================================================================
    # File Writing
    # ========================================================================

    def _write_knowledge_file(self, entry: KnowledgeEntry) -> Optional[str]:
        """Write a knowledge entry to a markdown file with YAML frontmatter."""
        try:
            # Generate filename from topic
            filename = f"{entry.topic}.md"
            filepath = self.knowledge_dir / filename

            # Build YAML frontmatter
            frontmatter = {
                "topic": entry.topic,
                "category": entry.category.value,
                "date": datetime.now(timezone.utc).isoformat(),
                "tags": entry.tags,
                "related_topics": entry.related_topics,
            }

            # Build markdown content
            content_lines = []

            # YAML frontmatter
            content_lines.append("---")
            if HAS_YAML:
                content_lines.append(yaml.dump(frontmatter, default_flow_style=False).strip())
            else:
                # Fallback without yaml library
                for key, value in frontmatter.items():
                    if isinstance(value, list):
                        content_lines.append(f"{key}: {value}")
                    else:
                        content_lines.append(f"{key}: {value}")
            content_lines.append("---")
            content_lines.append("")

            # Title
            content_lines.append(f"# {entry.topic.title().replace('-', ' ')}")
            content_lines.append("")

            # Summary
            content_lines.append("## Summary")
            content_lines.append(entry.summary)
            content_lines.append("")

            # Key Decisions
            if entry.key_decisions:
                content_lines.append("## Key Decisions")
                for decision in entry.key_decisions:
                    content_lines.append(f"- **{decision.decision}**: {decision.reasoning}")
                    if decision.alternatives_considered:
                        content_lines.append(f"  - Alternatives: {', '.join(decision.alternatives_considered)}")
                content_lines.append("")

            # Patterns
            if entry.patterns:
                content_lines.append("## Patterns")
                for pattern in entry.patterns:
                    content_lines.append(f"### {pattern.name}")
                    content_lines.append(f"{pattern.description}")
                    content_lines.append(f"**When to use:** {pattern.when_to_use}")
                    if pattern.example:
                        content_lines.append(f"**Example:** {pattern.example}")
                    content_lines.append("")
                content_lines.append("")

            # Domain Insights
            if entry.domain_insights:
                content_lines.append("## Domain Knowledge")
                for insight in entry.domain_insights:
                    content_lines.append(f"- {insight}")
                content_lines.append("")

            # Lessons Learned
            if entry.lessons_learned:
                content_lines.append("## Lessons Learned")
                for lesson in entry.lessons_learned:
                    content_lines.append(f"- {lesson}")
                content_lines.append("")

            # References
            if entry.references:
                content_lines.append("## References")
                for ref in entry.references:
                    content_lines.append(f"- {ref}")
                content_lines.append("")

            # Write to file
            content = "\n".join(content_lines)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)

            return filename

        except Exception as e:
            self.logger.error("Failed to write knowledge file", error=str(e), topic=entry.topic)
            emit_error("memory_curator", error=str(e), context="write_knowledge_file")
            return None

    # ========================================================================
    # Knowledge Retrieval
    # ========================================================================

    def retrieve_knowledge(
        self,
        query: str,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant knowledge entries based on a query.

        Args:
            query: Search query
            limit: Maximum number of entries to return

        Returns:
            List of matching knowledge entries with metadata
        """
        self.logger.info("Knowledge retrieval started", query_preview=query[:100], limit=limit)

        results = []

        # Simple keyword matching (could be enhanced with embeddings)
        query_words = set(re.findall(r'\b\w{3,}\b', query.lower()))

        # Search all knowledge files
        for filepath in self.knowledge_dir.glob("*.md"):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()

                # Extract frontmatter and body
                frontmatter, body = self._parse_knowledge_file(content)

                # Calculate relevance score
                score = self._calculate_relevance(query_words, frontmatter, body)

                if score > 0:
                    results.append({
                        "topic": frontmatter.get("topic", filepath.stem),
                        "category": frontmatter.get("category"),
                        "tags": frontmatter.get("tags", []),
                        "score": score,
                        "filepath": str(filepath),
                        "summary": self._extract_summary_from_body(body),
                    })

            except Exception:
                continue  # Skip files that can't be read

        # Sort by score and limit
        results.sort(key=lambda x: x["score"], reverse=True)
        filtered = results[:limit]
        self.logger.info("Knowledge retrieval completed", total_matches=len(results), returned=len(filtered))
        return filtered

    def _parse_knowledge_file(self, content: str) -> tuple[Dict[str, Any], str]:
        """Parse YAML frontmatter and body from knowledge file."""
        # Check for frontmatter markers
        if not content.startswith("---"):
            return {}, content

        # Find end of frontmatter
        end_idx = content.find("\n---", 4)
        if end_idx == -1:
            return {}, content

        frontmatter_text = content[4:end_idx]
        body = content[end_idx + 5:]

        # Parse YAML
        frontmatter = {}
        if HAS_YAML:
            try:
                frontmatter = yaml.safe_load(frontmatter_text) or {}
            except yaml.YAMLError as e:
                self.logger.warning("Failed to parse YAML frontmatter", error=str(e))

        return frontmatter, body

    def _calculate_relevance(
        self,
        query_words: Set[str],
        frontmatter: Dict[str, Any],
        body: str,
    ) -> float:
        """Calculate relevance score for a knowledge entry."""
        score = 0.0

        # Check tags (highest weight)
        tags = [t.lower() for t in frontmatter.get("tags", [])]
        for tag in tags:
            if tag in query_words:
                score += 2.0

        # Check topic
        topic = frontmatter.get("topic", "").lower()
        for word in query_words:
            if word in topic:
                score += 1.5

        # Check body content
        body_lower = body.lower()
        for word in query_words:
            if word in body_lower:
                score += 0.5

        return score

    def _extract_summary_from_body(self, body: str) -> str:
        """Extract summary from knowledge file body."""
        # Look for Summary section
        summary_match = re.search(r'## Summary\s*\n(.*?)(?=\n##|\Z)', body, re.DOTALL)
        if summary_match:
            return summary_match.group(1).strip()[:200]

        # Fallback: first paragraph
        first_para = re.search(r'\n\n(.*?)(?=\n\n|\n##|\Z)', body, re.DOTALL)
        if first_para:
            return first_para.group(1).strip()[:200]

        return ""

    # ========================================================================
    # Knowledge Listing
    # ========================================================================

    def list_knowledge(self) -> List[Dict[str, Any]]:
        """List all knowledge entries."""
        entries = []

        for filepath in self.knowledge_dir.glob("*.md"):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()

                frontmatter, _ = self._parse_knowledge_file(content)

                entries.append({
                    "topic": frontmatter.get("topic", filepath.stem),
                    "category": frontmatter.get("category"),
                    "date": frontmatter.get("date"),
                    "tags": frontmatter.get("tags", []),
                    "filepath": str(filepath),
                })

            except Exception:
                continue

        # Sort by date (newest first)
        entries.sort(key=lambda x: x.get("date", ""), reverse=True)
        return entries

    def _load_system_prompt(self) -> str:
        """Load the system prompt from file."""
        try:
            with open(self.system_prompt_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return "You are the Memory Curator. Extract and preserve knowledge from completed tasks."


# =============================================================================
# Convenience Functions
# =============================================================================

def create_memory_curator(
    system_prompt_path: str = "config/agents/memory_curator/CLAUDE.md",
    model: str = "claude-3-5-sonnet-20241022",
    knowledge_dir: str = "docs/knowledge",
) -> MemoryCuratorAgent:
    """Create a configured Memory Curator agent."""
    return MemoryCuratorAgent(
        system_prompt_path=system_prompt_path,
        model=model,
        knowledge_dir=knowledge_dir,
    )
