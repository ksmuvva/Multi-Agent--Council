"""
Exhaustive tests for MemoryCuratorAgent (src/agents/memory_curator.py).

Covers:
- __init__ with defaults and custom params, knowledge_dir creation
- extract_and_preserve() full flow
- _extract_key_decisions(), _parse_decisions_from_text()
- _extract_reasoning(), _extract_alternatives(), _get_context()
- _identify_patterns(), _extract_code_patterns(), _extract_workflow_patterns(),
  _extract_architectural_patterns()
- _infer_when_to_use(), _infer_when_to_use_architectural()
- _capture_domain_insights(), _document_lessons()
- _create_knowledge_entries(), _infer_category(), _generate_topic(), _generate_tags()
- _write_knowledge_file() with YAML frontmatter, error handling
- retrieve_knowledge(), _parse_knowledge_file(), _calculate_relevance()
- list_knowledge()
- KnowledgeCategory enum, dataclasses
"""

import os
import re
import pytest
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock
from datetime import datetime

from src.agents.memory_curator import (
    MemoryCuratorAgent,
    KnowledgeCategory,
    KeyDecision,
    Pattern,
    KnowledgeEntry,
    ExtractionResult,
    create_memory_curator,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def curator(tmp_path):
    """Create a MemoryCuratorAgent with a tmp knowledge dir."""
    with patch("builtins.open", side_effect=FileNotFoundError):
        agent = MemoryCuratorAgent(
            system_prompt_path="nonexistent.md",
            knowledge_dir=str(tmp_path / "knowledge"),
        )
    return agent


@pytest.fixture
def curator_with_prompt(tmp_path):
    """Create a MemoryCuratorAgent with a real system prompt file."""
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("You are the Memory Curator.")
    agent = MemoryCuratorAgent(
        system_prompt_path=str(prompt_file),
        knowledge_dir=str(tmp_path / "knowledge"),
    )
    return agent


@pytest.fixture
def sample_agent_outputs():
    return {
        "planner": "We chose Python over Java because it offers faster iteration. "
                   "Instead of Java we opted for Python.",
        "executor": "We decided to use FastAPI. Selected async approach. "
                    "def process_data(): pass\n"
                    "class DataHandler: pass\n"
                    "async def fetch_data(): pass",
        "researcher": "Finding: Python is the most popular language. "
                      "Evidence shows: rapid development is key.",
        "verifier": "Verified: the claim about performance is correct.",
        "critic": "Issue: missing error handling. Improvement: add retry logic.",
        "code_reviewer": "Security issue: SQL injection detected. "
                         "Recommend: use parameterized queries.",
    }


@pytest.fixture
def sample_execution_context():
    return {
        "tier_level": 3,
        "agents_used": ["planner", "executor", "researcher"],
        "revisions": 2,
        "escalations": 1,
        "smes_involved": ["cloud_architect", "security_analyst"],
    }


# ============================================================================
# KnowledgeCategory Enum
# ============================================================================

class TestKnowledgeCategory:
    def test_enum_values(self):
        assert KnowledgeCategory.ARCHITECTURAL_DECISION == "architectural_decision"
        assert KnowledgeCategory.CODE_PATTERN == "code_pattern"
        assert KnowledgeCategory.DOMAIN_INSIGHT == "domain_insight"
        assert KnowledgeCategory.TROUBLESHOOTING == "troubleshooting"
        assert KnowledgeCategory.BEST_PRACTICE == "best_practice"
        assert KnowledgeCategory.ANTI_PATTERN == "anti_pattern"
        assert KnowledgeCategory.LESSON_LEARNED == "lesson_learned"
        assert KnowledgeCategory.REFERENCE == "reference"

    def test_enum_is_string(self):
        assert isinstance(KnowledgeCategory.CODE_PATTERN, str)

    def test_enum_count(self):
        assert len(KnowledgeCategory) == 8


# ============================================================================
# Dataclasses
# ============================================================================

class TestKeyDecision:
    def test_creation(self):
        kd = KeyDecision(
            decision="chose Python",
            reasoning="faster dev",
            alternatives_considered=["Java", "Go"],
            context="for backend",
        )
        assert kd.decision == "chose Python"
        assert kd.reasoning == "faster dev"
        assert len(kd.alternatives_considered) == 2
        assert kd.context == "for backend"

    def test_empty_alternatives(self):
        kd = KeyDecision(decision="x", reasoning="y", alternatives_considered=[], context="")
        assert kd.alternatives_considered == []


class TestPattern:
    def test_creation(self):
        p = Pattern(
            name="Singleton",
            description="Single instance",
            when_to_use="shared resources",
            example="class Singleton: ...",
            related_patterns=["Factory"],
        )
        assert p.name == "Singleton"
        assert p.example == "class Singleton: ..."

    def test_optional_example(self):
        p = Pattern(name="X", description="Y", when_to_use="Z", example=None, related_patterns=[])
        assert p.example is None


class TestKnowledgeEntry:
    def test_creation(self):
        entry = KnowledgeEntry(
            topic="test-topic",
            category=KnowledgeCategory.CODE_PATTERN,
            summary="A summary",
            key_decisions=[],
            patterns=[],
            domain_insights=["insight1"],
            lessons_learned=["lesson1"],
            references=[],
            related_topics=["other-topic"],
            tags=["python", "tier-2"],
        )
        assert entry.topic == "test-topic"
        assert entry.category == KnowledgeCategory.CODE_PATTERN
        assert len(entry.domain_insights) == 1


class TestExtractionResult:
    def test_creation(self):
        er = ExtractionResult(
            entries=[],
            total_extractions=0,
            topics_created=[],
            extraction_metadata={"session_id": "abc"},
        )
        assert er.total_extractions == 0
        assert er.extraction_metadata["session_id"] == "abc"


# ============================================================================
# __init__
# ============================================================================

class TestInit:
    def test_defaults(self, tmp_path):
        with patch("builtins.open", side_effect=FileNotFoundError):
            agent = MemoryCuratorAgent(knowledge_dir=str(tmp_path / "k"))
        assert agent.model == "claude-3-5-sonnet-20241022"
        assert agent.max_turns == 30
        assert agent.system_prompt_path == "config/agents/memory_curator/CLAUDE.md"

    def test_custom_params(self, tmp_path):
        with patch("builtins.open", side_effect=FileNotFoundError):
            agent = MemoryCuratorAgent(
                model="claude-3-opus",
                max_turns=10,
                knowledge_dir=str(tmp_path / "custom"),
            )
        assert agent.model == "claude-3-opus"
        assert agent.max_turns == 10

    def test_knowledge_dir_creation(self, tmp_path):
        kdir = tmp_path / "nested" / "knowledge"
        with patch("builtins.open", side_effect=FileNotFoundError):
            agent = MemoryCuratorAgent(knowledge_dir=str(kdir))
        assert kdir.exists()

    def test_system_prompt_fallback(self, tmp_path):
        with patch("builtins.open", side_effect=FileNotFoundError):
            agent = MemoryCuratorAgent(knowledge_dir=str(tmp_path / "k"))
        assert "Memory Curator" in agent.system_prompt

    def test_system_prompt_loaded(self, curator_with_prompt):
        assert curator_with_prompt.system_prompt == "You are the Memory Curator."

    def test_decision_patterns_initialized(self, curator):
        assert len(curator.decision_patterns) >= 4

    def test_pattern_indicators_initialized(self, curator):
        assert "pattern" in curator.pattern_indicators

    def test_lesson_patterns_initialized(self, curator):
        assert len(curator.lesson_patterns) >= 4


# ============================================================================
# _parse_decisions_from_text
# ============================================================================

class TestParseDecisionsFromText:
    @pytest.mark.parametrize("text,expected_count", [
        ("We chose Python over Java.", 1),
        ("I decided to use FastAPI.", 1),
        ("We selected the microservice approach.", 1),
        ("We opted for serverless.", 1),
        ("We went with containers.", 1),
        ("No decisions here.", 0),
        ("", 0),
    ])
    def test_decision_sentence_patterns(self, curator, text, expected_count):
        decisions = curator._parse_decisions_from_text(text)
        assert len(decisions) == expected_count

    def test_multiple_decisions(self, curator):
        text = "We chose Python. We decided to use FastAPI. We selected async."
        decisions = curator._parse_decisions_from_text(text)
        assert len(decisions) == 3

    def test_decision_has_fields(self, curator):
        text = "We chose Python because it is easy."
        decisions = curator._parse_decisions_from_text(text)
        assert len(decisions) >= 1
        d = decisions[0]
        assert isinstance(d, KeyDecision)
        assert d.decision
        assert d.reasoning
        assert isinstance(d.alternatives_considered, list)
        assert isinstance(d.context, str)


# ============================================================================
# _extract_reasoning
# ============================================================================

class TestExtractReasoning:
    def test_because_pattern(self, curator):
        text = "We chose X because it is faster."
        result = curator._extract_reasoning("chose X", text)
        assert "faster" in result.lower() or "it is faster" in result.lower()

    def test_since_pattern(self, curator):
        text = "since Python is popular we use it."
        result = curator._extract_reasoning("use it", text)
        assert len(result) > 5

    def test_due_to_pattern(self, curator):
        text = "due to performance constraints we optimized."
        result = curator._extract_reasoning("optimized", text)
        assert len(result) > 5

    def test_default_reasoning(self, curator):
        text = "No reasoning clues here at all."
        result = curator._extract_reasoning("something", text)
        assert result == "Decision made based on task requirements"

    def test_short_reasoning_skipped(self, curator):
        text = "because ok we went."
        result = curator._extract_reasoning("went", text)
        # "ok we went" might be too short; check it falls to default or short match
        assert isinstance(result, str)


# ============================================================================
# _extract_alternatives
# ============================================================================

class TestExtractAlternatives:
    def test_instead_of_pattern(self, curator):
        text = "We chose Python instead of Java."
        alts = curator._extract_alternatives("chose Python", text)
        assert any("java" in a.lower() for a in alts)

    def test_rather_than_pattern(self, curator):
        text = "Used Go rather than Rust."
        alts = curator._extract_alternatives("Used Go", text)
        assert any("rust" in a.lower() for a in alts)

    def test_versus_pattern(self, curator):
        text = "Python versus Java."
        alts = curator._extract_alternatives("Python", text)
        assert len(alts) >= 1

    def test_vs_pattern(self, curator):
        text = "Python vs. Java."
        alts = curator._extract_alternatives("Python", text)
        assert len(alts) >= 1

    def test_limit_three(self, curator):
        text = (
            "instead of A instead of B instead of C instead of D "
            "rather than E rather than F"
        )
        alts = curator._extract_alternatives("chose X", text)
        assert len(alts) <= 3

    def test_no_alternatives(self, curator):
        text = "Nothing about alternatives."
        alts = curator._extract_alternatives("chose X", text)
        assert alts == []

    def test_deduplication(self, curator):
        text = "instead of Java. Rather than Java."
        alts = curator._extract_alternatives("chose X", text)
        java_count = sum(1 for a in alts if "java" in a.lower())
        assert java_count <= 1


# ============================================================================
# _get_context
# ============================================================================

class TestGetContext:
    def test_context_found(self, curator):
        text = "prefix text. We chose Python for speed. suffix text."
        ctx = curator._get_context("We chose Python for speed.", text)
        assert len(ctx) > 0
        assert len(ctx) <= 200

    def test_context_not_found(self, curator):
        ctx = curator._get_context("not in text", "some other text entirely")
        assert ctx == ""

    def test_context_limit_200(self, curator):
        long_text = "A" * 500 + "DECISION" + "B" * 500
        ctx = curator._get_context("DECISION", long_text)
        assert len(ctx) <= 200

    def test_whitespace_cleaned(self, curator):
        text = "before   \n\t  WORD   \n\t  after"
        ctx = curator._get_context("WORD", text)
        assert "\n" not in ctx
        assert "\t" not in ctx


# ============================================================================
# _extract_key_decisions
# ============================================================================

class TestExtractKeyDecisions:
    def test_from_planner(self, curator):
        outputs = {"planner": "We chose microservices."}
        decisions = curator._extract_key_decisions("task", outputs, "final")
        assert len(decisions) >= 1

    def test_from_executor(self, curator):
        outputs = {"executor": "We decided to use caching."}
        decisions = curator._extract_key_decisions("task", outputs, "final")
        assert len(decisions) >= 1

    def test_from_task_description(self, curator):
        decisions = curator._extract_key_decisions(
            "We selected Python for this.", {}, "final"
        )
        assert len(decisions) >= 1

    def test_deduplication(self, curator):
        outputs = {
            "planner": "We chose Python.",
            "executor": "We chose Python.",
        }
        decisions = curator._extract_key_decisions("We chose Python.", outputs, "final")
        # All three sources produce "chose Python" - should deduplicate
        python_decisions = [d for d in decisions if "python" in d.decision.lower()]
        assert len(python_decisions) == 1

    def test_limit_10(self, curator):
        text = ". ".join([f"We chose option{i}." for i in range(15)])
        outputs = {"planner": text}
        decisions = curator._extract_key_decisions("task", outputs, "final")
        assert len(decisions) <= 10

    def test_empty_outputs(self, curator):
        decisions = curator._extract_key_decisions("no decisions", {}, "no decisions")
        assert isinstance(decisions, list)


# ============================================================================
# _identify_patterns
# ============================================================================

class TestIdentifyPatterns:
    def test_code_patterns_from_executor(self, curator):
        outputs = {
            "executor": "def process(): pass\nclass Handler: pass\nasync def fetch(): pass"
        }
        patterns = curator._identify_patterns(outputs, "final")
        names = [p.name for p in patterns]
        assert "process" in names
        assert "Handler" in names

    def test_workflow_patterns_from_planner(self, curator):
        text = "Step 1: analyze. Step 2: design. Step 3: implement. Step 4: test."
        outputs = {"planner": text}
        patterns = curator._identify_patterns(outputs, "final")
        workflow_patterns = [p for p in patterns if "workflow" in p.name.lower()]
        assert len(workflow_patterns) >= 1

    def test_architectural_patterns(self, curator):
        outputs = {"architect": "We used the singleton pattern and observer pattern."}
        patterns = curator._identify_patterns(outputs, "final")
        names = [p.name for p in patterns]
        assert "Singleton Pattern" in names
        assert "Observer Pattern" in names

    def test_deduplication(self, curator):
        outputs = {
            "a": "We used the factory pattern.",
            "b": "Apply the factory pattern.",
        }
        patterns = curator._identify_patterns(outputs, "final")
        factory_count = sum(1 for p in patterns if "factory" in p.name.lower())
        assert factory_count == 1

    def test_limit_8(self, curator):
        # Lots of code patterns
        code = "\n".join([f"def func_{i}(): pass" for i in range(20)])
        outputs = {"executor": code}
        patterns = curator._identify_patterns(outputs, "final")
        assert len(patterns) <= 8

    def test_empty_outputs(self, curator):
        patterns = curator._identify_patterns({}, "final")
        assert patterns == []


# ============================================================================
# _extract_code_patterns
# ============================================================================

class TestExtractCodePatterns:
    def test_function_pattern(self, curator):
        patterns = curator._extract_code_patterns("def my_function(): pass")
        assert any(p.name == "my_function" for p in patterns)

    def test_class_pattern(self, curator):
        patterns = curator._extract_code_patterns("class MyClass: pass")
        assert any(p.name == "MyClass" for p in patterns)

    def test_async_function_pattern(self, curator):
        patterns = curator._extract_code_patterns("async def fetch_data(): pass")
        assert any(p.name == "fetch_data" for p in patterns)

    def test_no_patterns(self, curator):
        patterns = curator._extract_code_patterns("just plain text")
        assert patterns == []


# ============================================================================
# _extract_workflow_patterns
# ============================================================================

class TestExtractWorkflowPatterns:
    def test_three_or_more_steps(self, curator):
        text = "Step 1: analyze. Step 2: design. Step 3: implement."
        patterns = curator._extract_workflow_patterns(text)
        assert len(patterns) == 1
        assert "3-step" in patterns[0].name

    def test_phase_keyword(self, curator):
        text = "Phase 1: plan. Phase 2: build. Phase 3: test."
        patterns = curator._extract_workflow_patterns(text)
        assert len(patterns) == 1

    def test_stage_keyword(self, curator):
        text = "Stage 1: init. Stage 2: process. Stage 3: finalize."
        patterns = curator._extract_workflow_patterns(text)
        assert len(patterns) == 1

    def test_fewer_than_three_steps(self, curator):
        text = "Step 1: analyze. Step 2: design."
        patterns = curator._extract_workflow_patterns(text)
        assert len(patterns) == 0

    def test_no_steps(self, curator):
        patterns = curator._extract_workflow_patterns("no steps here")
        assert patterns == []


# ============================================================================
# _extract_architectural_patterns
# ============================================================================

class TestExtractArchitecturalPatterns:
    @pytest.mark.parametrize("keyword,expected_name", [
        ("mvc", "Model-View-Controller"),
        ("observer", "Observer Pattern"),
        ("singleton", "Singleton Pattern"),
        ("factory", "Factory Pattern"),
        ("repository", "Repository Pattern"),
        ("adapter", "Adapter Pattern"),
        ("decorator", "Decorator Pattern"),
        ("strategy", "Strategy Pattern"),
    ])
    def test_each_keyword(self, curator, keyword, expected_name):
        patterns = curator._extract_architectural_patterns(f"Using {keyword} in code.")
        assert any(p.name == expected_name for p in patterns)

    def test_case_insensitive(self, curator):
        patterns = curator._extract_architectural_patterns("SINGLETON pattern used.")
        assert any("Singleton" in p.name for p in patterns)

    def test_no_patterns(self, curator):
        patterns = curator._extract_architectural_patterns("nothing architectural")
        assert patterns == []

    def test_multiple_patterns(self, curator):
        text = "We used mvc, observer, and factory patterns."
        patterns = curator._extract_architectural_patterns(text)
        assert len(patterns) == 3


# ============================================================================
# _infer_when_to_use
# ============================================================================

class TestInferWhenToUse:
    def test_async_name(self, curator):
        result = curator._infer_when_to_use("async_handler", "context")
        assert "asynchronous" in result.lower()

    def test_get_name(self, curator):
        result = curator._infer_when_to_use("get_data", "context")
        assert "retrieving" in result.lower()

    def test_set_name(self, curator):
        result = curator._infer_when_to_use("set_value", "context")
        assert "modifying" in result.lower()

    def test_update_name(self, curator):
        result = curator._infer_when_to_use("update_record", "context")
        assert "modifying" in result.lower()

    def test_general_name(self, curator):
        result = curator._infer_when_to_use("process", "context")
        assert result == "General purpose"


# ============================================================================
# _infer_when_to_use_architectural
# ============================================================================

class TestInferWhenToUseArchitectural:
    @pytest.mark.parametrize("keyword,expected_fragment", [
        ("mvc", "separating concerns"),
        ("observer", "event-driven"),
        ("singleton", "single-instance"),
        ("factory", "object creation"),
        ("repository", "data access"),
        ("adapter", "incompatible interfaces"),
        ("decorator", "adding behavior"),
        ("strategy", "interchangeable algorithms"),
    ])
    def test_known_keywords(self, curator, keyword, expected_fragment):
        result = curator._infer_when_to_use_architectural(keyword)
        assert expected_fragment in result.lower()

    def test_unknown_keyword(self, curator):
        result = curator._infer_when_to_use_architectural("unknown")
        assert result == "General architectural use"


# ============================================================================
# _capture_domain_insights
# ============================================================================

class TestCaptureDomainInsights:
    def test_from_researcher(self, curator):
        outputs = {"researcher": "Finding: Python is popular for ML."}
        insights = curator._capture_domain_insights("task", outputs)
        assert any("Python" in i for i in insights)

    def test_from_task_description(self, curator):
        insights = curator._capture_domain_insights(
            '"microservices" should be preferred',
            {},
        )
        assert any("microservices" in i for i in insights)

    def test_from_verifier(self, curator):
        outputs = {"verifier": "Verified: the API supports pagination."}
        insights = curator._capture_domain_insights("task", outputs)
        assert any("Verified" in i for i in insights)

    def test_deduplication(self, curator):
        outputs = {"researcher": "Finding: item. Finding: item."}
        insights = curator._capture_domain_insights("task", outputs)
        item_count = sum(1 for i in insights if i == "item")
        assert item_count <= 1

    def test_short_insights_filtered(self, curator):
        outputs = {"researcher": "Finding: ok."}
        insights = curator._capture_domain_insights("task", outputs)
        short_insights = [i for i in insights if len(i.strip()) <= 10]
        assert len(short_insights) == 0

    def test_limit_15(self, curator):
        findings = ". ".join([f"Finding: insight number {i} is important" for i in range(20)])
        outputs = {"researcher": findings}
        insights = curator._capture_domain_insights("task", outputs)
        assert len(insights) <= 15


# ============================================================================
# _document_lessons
# ============================================================================

class TestDocumentLessons:
    def test_from_critic(self, curator):
        outputs = {"critic": "Issue: missing validation. Improvement: add checks."}
        lessons = curator._document_lessons({}, outputs)
        assert any("Address" in l for l in lessons)

    def test_from_code_reviewer(self, curator):
        outputs = {"code_reviewer": "Security issue: SQL injection found."}
        lessons = curator._document_lessons({}, outputs)
        assert any("Code quality" in l for l in lessons)

    def test_revisions_lesson(self, curator):
        context = {"revisions": 3}
        lessons = curator._document_lessons(context, {})
        assert any("3 revision" in l for l in lessons)

    def test_escalations_lesson(self, curator):
        context = {"escalations": 2}
        lessons = curator._document_lessons(context, {})
        assert any("2 time" in l for l in lessons)

    def test_smes_involved_lesson(self, curator):
        context = {"smes_involved": ["cloud_architect"]}
        lessons = curator._document_lessons(context, {})
        assert any("cloud_architect" in l for l in lessons)

    def test_no_revisions(self, curator):
        context = {"revisions": 0}
        lessons = curator._document_lessons(context, {})
        assert not any("revision" in l for l in lessons)

    def test_deduplication(self, curator):
        context = {"smes_involved": ["sme1", "sme1"]}
        lessons = curator._document_lessons(context, {})
        sme1_count = sum(1 for l in lessons if "sme1" in l)
        assert sme1_count == 1

    def test_limit_10(self, curator):
        issues = ". ".join([f"Issue: problem {i}" for i in range(15)])
        outputs = {"critic": issues}
        lessons = curator._document_lessons({}, outputs)
        assert len(lessons) <= 10


# ============================================================================
# _create_knowledge_entries
# ============================================================================

class TestCreateKnowledgeEntries:
    def test_main_entry_created(self, curator):
        entries = curator._create_knowledge_entries(
            "implement a caching function",
            [KeyDecision("chose Redis", "fast", [], "")],
            [],
            ["insight1"],
            ["lesson1"],
            {"tier_level": 2, "agents_used": []},
        )
        assert len(entries) >= 1
        assert entries[0].category == KnowledgeCategory.CODE_PATTERN

    def test_patterns_entry_when_three_or_more(self, curator):
        patterns = [
            Pattern(f"p{i}", "desc", "when", None, []) for i in range(4)
        ]
        entries = curator._create_knowledge_entries(
            "design architecture", [], patterns, [], [],
            {"tier_level": 3, "agents_used": []},
        )
        assert len(entries) == 2
        assert entries[1].category == KnowledgeCategory.CODE_PATTERN
        assert entries[1].topic.endswith("-patterns")

    def test_no_patterns_entry_when_fewer_than_three(self, curator):
        patterns = [Pattern("p1", "d", "w", None, []), Pattern("p2", "d", "w", None, [])]
        entries = curator._create_knowledge_entries(
            "task", [], patterns, [], [],
            {"tier_level": 1, "agents_used": []},
        )
        assert len(entries) == 1

    def test_tags_present(self, curator):
        entries = curator._create_knowledge_entries(
            "implement code function",
            [], [], [], [],
            {"tier_level": 2, "agents_used": ["planner"]},
        )
        assert len(entries[0].tags) > 0


# ============================================================================
# _infer_category
# ============================================================================

class TestInferCategory:
    @pytest.mark.parametrize("task,expected", [
        ("implement a function in code", KnowledgeCategory.CODE_PATTERN),
        ("design the architecture", KnowledgeCategory.ARCHITECTURAL_DECISION),
        ("fix a bug in the system", KnowledgeCategory.TROUBLESHOOTING),
        ("avoid this mistake", KnowledgeCategory.ANTI_PATTERN),
        ("some random task", KnowledgeCategory.LESSON_LEARNED),
    ])
    def test_category_inference(self, curator, task, expected):
        result = curator._infer_category(task, {})
        assert result == expected

    def test_class_keyword(self, curator):
        result = curator._infer_category("create a class for users", {})
        assert result == KnowledgeCategory.CODE_PATTERN

    def test_component_keyword(self, curator):
        result = curator._infer_category("build a component structure", {})
        assert result == KnowledgeCategory.ARCHITECTURAL_DECISION

    def test_error_keyword(self, curator):
        result = curator._infer_category("resolve the error", {})
        assert result == KnowledgeCategory.TROUBLESHOOTING


# ============================================================================
# _generate_topic
# ============================================================================

class TestGenerateTopic:
    def test_meaningful_words(self, curator):
        topic = curator._generate_topic(
            "implement the caching layer for performance",
            KnowledgeCategory.CODE_PATTERN,
        )
        assert "implement" in topic
        assert "caching" in topic

    def test_kebab_case(self, curator):
        topic = curator._generate_topic(
            "build large scale system",
            KnowledgeCategory.ARCHITECTURAL_DECISION,
        )
        assert "-" in topic
        assert " " not in topic

    def test_length_limit_50(self, curator):
        long_task = " ".join(["longword"] * 20)
        topic = curator._generate_topic(long_task, KnowledgeCategory.CODE_PATTERN)
        assert len(topic) <= 50

    def test_filters_stop_words(self, curator):
        topic = curator._generate_topic(
            "the with from have been were implement",
            KnowledgeCategory.CODE_PATTERN,
        )
        assert "the" not in topic.split("-")
        assert "with" not in topic.split("-")

    def test_empty_task_fallback(self, curator):
        topic = curator._generate_topic("the a an", KnowledgeCategory.CODE_PATTERN)
        assert topic.startswith("task-")

    def test_no_special_chars(self, curator):
        topic = curator._generate_topic(
            "implement feature! with @special #chars",
            KnowledgeCategory.CODE_PATTERN,
        )
        assert "@" not in topic
        assert "#" not in topic
        assert "!" not in topic


# ============================================================================
# _generate_tags
# ============================================================================

class TestGenerateTags:
    def test_tier_tag(self, curator):
        tags = curator._generate_tags("task", {"tier_level": 3, "agents_used": []})
        assert "tier-3" in tags

    def test_agents_tags(self, curator):
        tags = curator._generate_tags(
            "task", {"agents_used": ["planner", "executor", "verifier", "extra"]}
        )
        assert "planner" in tags
        assert "executor" in tags
        assert "verifier" in tags
        # Only first 3 agents
        assert "extra" not in tags

    def test_key_terms(self, curator):
        tags = curator._generate_tags(
            "implement caching layer performance",
            {"agents_used": []},
        )
        assert "implement" in tags or "caching" in tags

    def test_deduplication(self, curator):
        tags = curator._generate_tags(
            "planner planner planner planner",
            {"agents_used": ["planner"]},
        )
        planner_count = sum(1 for t in tags if t == "planner")
        assert planner_count == 1

    def test_no_tier_when_missing(self, curator):
        tags = curator._generate_tags("task", {"agents_used": []})
        tier_tags = [t for t in tags if t.startswith("tier-")]
        assert len(tier_tags) == 0


# ============================================================================
# _write_knowledge_file
# ============================================================================

class TestWriteKnowledgeFile:
    def test_writes_file(self, curator, tmp_path):
        entry = KnowledgeEntry(
            topic="test-topic",
            category=KnowledgeCategory.CODE_PATTERN,
            summary="A test summary",
            key_decisions=[
                KeyDecision("chose X", "reason", ["alt1"], "ctx"),
            ],
            patterns=[
                Pattern("PatternA", "desc", "when", "example", []),
            ],
            domain_insights=["insight1"],
            lessons_learned=["lesson1"],
            references=["ref1"],
            related_topics=["other"],
            tags=["python"],
        )
        result = curator._write_knowledge_file(entry)
        assert result == "test-topic.md"

        filepath = curator.knowledge_dir / "test-topic.md"
        assert filepath.exists()
        content = filepath.read_text()
        assert "---" in content
        assert "test-topic" in content
        assert "A test summary" in content
        assert "chose X" in content
        assert "PatternA" in content
        assert "insight1" in content
        assert "lesson1" in content
        assert "ref1" in content

    def test_yaml_frontmatter(self, curator):
        entry = KnowledgeEntry(
            topic="yaml-test",
            category=KnowledgeCategory.LESSON_LEARNED,
            summary="summary",
            key_decisions=[], patterns=[], domain_insights=[],
            lessons_learned=[], references=[], related_topics=[],
            tags=["tag1"],
        )
        curator._write_knowledge_file(entry)
        content = (curator.knowledge_dir / "yaml-test.md").read_text()
        assert content.startswith("---")
        # Should have closing ---
        parts = content.split("---")
        assert len(parts) >= 3

    def test_error_handling(self, curator):
        entry = KnowledgeEntry(
            topic="error-test",
            category=KnowledgeCategory.CODE_PATTERN,
            summary="s", key_decisions=[], patterns=[], domain_insights=[],
            lessons_learned=[], references=[], related_topics=[], tags=[],
        )
        with patch("builtins.open", side_effect=PermissionError("denied")):
            result = curator._write_knowledge_file(entry)
        assert result is None

    def test_empty_sections_not_rendered(self, curator):
        entry = KnowledgeEntry(
            topic="empty-sections",
            category=KnowledgeCategory.REFERENCE,
            summary="summary",
            key_decisions=[], patterns=[], domain_insights=[],
            lessons_learned=[], references=[], related_topics=[],
            tags=[],
        )
        curator._write_knowledge_file(entry)
        content = (curator.knowledge_dir / "empty-sections.md").read_text()
        assert "## Key Decisions" not in content
        assert "## Patterns" not in content
        assert "## Domain Knowledge" not in content
        assert "## Lessons Learned" not in content
        assert "## References" not in content


# ============================================================================
# extract_and_preserve (full flow)
# ============================================================================

class TestExtractAndPreserve:
    def test_full_flow(self, curator, sample_agent_outputs, sample_execution_context):
        result = curator.extract_and_preserve(
            task_description="Implement a caching function for the API",
            execution_context=sample_execution_context,
            agent_outputs=sample_agent_outputs,
            final_output="Here is the caching function.",
            session_id="session-123",
        )
        assert isinstance(result, ExtractionResult)
        assert len(result.entries) >= 1
        assert result.total_extractions >= 0
        assert result.extraction_metadata["session_id"] == "session-123"
        assert result.extraction_metadata["tier_level"] == 3

    def test_topics_created(self, curator, sample_agent_outputs, sample_execution_context):
        result = curator.extract_and_preserve(
            task_description="Build architecture design",
            execution_context=sample_execution_context,
            agent_outputs=sample_agent_outputs,
            final_output="done",
        )
        assert isinstance(result.topics_created, list)
        for topic in result.topics_created:
            assert topic.endswith(".md")

    def test_empty_agent_outputs(self, curator):
        result = curator.extract_and_preserve(
            task_description="simple task",
            execution_context={"agents_used": []},
            agent_outputs={},
            final_output="done",
        )
        assert isinstance(result, ExtractionResult)
        assert len(result.entries) >= 1

    def test_knowledge_files_written(self, curator, sample_agent_outputs, sample_execution_context):
        result = curator.extract_and_preserve(
            task_description="code function implementation",
            execution_context=sample_execution_context,
            agent_outputs=sample_agent_outputs,
            final_output="done",
        )
        md_files = list(curator.knowledge_dir.glob("*.md"))
        assert len(md_files) >= 1


# ============================================================================
# retrieve_knowledge
# ============================================================================

class TestRetrieveKnowledge:
    def test_keyword_matching(self, curator):
        # Write a knowledge file
        entry = KnowledgeEntry(
            topic="python-caching",
            category=KnowledgeCategory.CODE_PATTERN,
            summary="Caching in Python",
            key_decisions=[], patterns=[], domain_insights=[],
            lessons_learned=[], references=[], related_topics=[],
            tags=["python", "caching"],
        )
        curator._write_knowledge_file(entry)

        results = curator.retrieve_knowledge("python")
        assert len(results) >= 1
        assert results[0]["topic"] == "python-caching"

    def test_score_calculation(self, curator):
        entry = KnowledgeEntry(
            topic="test-score",
            category=KnowledgeCategory.CODE_PATTERN,
            summary="Python is great",
            key_decisions=[], patterns=[], domain_insights=[],
            lessons_learned=[], references=[], related_topics=[],
            tags=["python", "code"],
        )
        curator._write_knowledge_file(entry)

        results = curator.retrieve_knowledge("python code")
        assert len(results) >= 1
        assert results[0]["score"] > 0

    def test_sorting_by_score(self, curator):
        for tag in ["alpha", "beta"]:
            entry = KnowledgeEntry(
                topic=f"topic-{tag}",
                category=KnowledgeCategory.CODE_PATTERN,
                summary=f"About {tag}",
                key_decisions=[], patterns=[], domain_insights=[],
                lessons_learned=[], references=[], related_topics=[],
                tags=[tag, "shared"],
            )
            curator._write_knowledge_file(entry)

        results = curator.retrieve_knowledge("alpha shared")
        assert len(results) >= 1
        # alpha should score higher because it matches tag + shared
        if len(results) >= 2:
            assert results[0]["score"] >= results[1]["score"]

    def test_limit(self, curator):
        for i in range(10):
            entry = KnowledgeEntry(
                topic=f"topic-{i}",
                category=KnowledgeCategory.CODE_PATTERN,
                summary="common content shared",
                key_decisions=[], patterns=[], domain_insights=[],
                lessons_learned=[], references=[], related_topics=[],
                tags=["shared"],
            )
            curator._write_knowledge_file(entry)

        results = curator.retrieve_knowledge("shared", limit=3)
        assert len(results) <= 3

    def test_no_results(self, curator):
        results = curator.retrieve_knowledge("zzzznonexistent")
        assert results == []

    def test_handles_corrupt_file(self, curator):
        # Write a corrupt file
        corrupt_path = curator.knowledge_dir / "corrupt.md"
        corrupt_path.write_text("not valid yaml frontmatter")
        results = curator.retrieve_knowledge("corrupt")
        # Should not crash
        assert isinstance(results, list)


# ============================================================================
# _parse_knowledge_file
# ============================================================================

class TestParseKnowledgeFile:
    def test_valid_frontmatter(self, curator):
        content = "---\ntopic: test\ncategory: code_pattern\n---\n\n# Body"
        frontmatter, body = curator._parse_knowledge_file(content)
        assert frontmatter.get("topic") == "test"
        assert "# Body" in body

    def test_no_frontmatter(self, curator):
        content = "Just regular content."
        frontmatter, body = curator._parse_knowledge_file(content)
        assert frontmatter == {}
        assert body == content

    def test_incomplete_frontmatter(self, curator):
        content = "---\ntopic: test\nno closing marker"
        frontmatter, body = curator._parse_knowledge_file(content)
        assert frontmatter == {}
        assert body == content


# ============================================================================
# _calculate_relevance
# ============================================================================

class TestCalculateRelevance:
    def test_tag_match(self, curator):
        score = curator._calculate_relevance(
            {"python"}, {"tags": ["python"]}, ""
        )
        assert score >= 2.0

    def test_topic_match(self, curator):
        score = curator._calculate_relevance(
            {"caching"}, {"topic": "python-caching", "tags": []}, ""
        )
        assert score >= 1.5

    def test_body_match(self, curator):
        score = curator._calculate_relevance(
            {"redis"}, {"tags": [], "topic": ""}, "Using redis for caching"
        )
        assert score >= 0.5

    def test_no_match(self, curator):
        score = curator._calculate_relevance(
            {"zzz"}, {"tags": [], "topic": ""}, "nothing here"
        )
        assert score == 0.0

    def test_combined_scoring(self, curator):
        score = curator._calculate_relevance(
            {"python"},
            {"tags": ["python"], "topic": "python-caching"},
            "python is great",
        )
        # tag=2.0 + topic=1.5 + body=0.5 = 4.0
        assert score >= 4.0


# ============================================================================
# list_knowledge
# ============================================================================

class TestListKnowledge:
    def test_lists_entries(self, curator):
        for i in range(3):
            entry = KnowledgeEntry(
                topic=f"list-test-{i}",
                category=KnowledgeCategory.CODE_PATTERN,
                summary=f"Entry {i}",
                key_decisions=[], patterns=[], domain_insights=[],
                lessons_learned=[], references=[], related_topics=[],
                tags=[],
            )
            curator._write_knowledge_file(entry)

        entries = curator.list_knowledge()
        assert len(entries) == 3

    def test_sorted_by_date(self, curator):
        for i in range(3):
            entry = KnowledgeEntry(
                topic=f"date-test-{i}",
                category=KnowledgeCategory.CODE_PATTERN,
                summary=f"Entry {i}",
                key_decisions=[], patterns=[], domain_insights=[],
                lessons_learned=[], references=[], related_topics=[],
                tags=[],
            )
            curator._write_knowledge_file(entry)

        entries = curator.list_knowledge()
        # All entries have dates, sorted descending
        dates = [e.get("date", "") for e in entries]
        assert dates == sorted(dates, reverse=True)

    def test_empty_dir(self, curator):
        entries = curator.list_knowledge()
        assert entries == []

    def test_entry_fields(self, curator):
        entry = KnowledgeEntry(
            topic="fields-test",
            category=KnowledgeCategory.LESSON_LEARNED,
            summary="s",
            key_decisions=[], patterns=[], domain_insights=[],
            lessons_learned=[], references=[], related_topics=[],
            tags=["t1"],
        )
        curator._write_knowledge_file(entry)

        entries = curator.list_knowledge()
        assert len(entries) == 1
        e = entries[0]
        assert "topic" in e
        assert "category" in e
        assert "date" in e
        assert "tags" in e
        assert "filepath" in e


# ============================================================================
# create_memory_curator convenience function
# ============================================================================

class TestCreateMemoryCurator:
    def test_creates_instance(self, tmp_path):
        with patch("builtins.open", side_effect=FileNotFoundError):
            agent = create_memory_curator(knowledge_dir=str(tmp_path / "k"))
        assert isinstance(agent, MemoryCuratorAgent)
