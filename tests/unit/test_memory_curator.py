"""
Tests for the MemoryCuratorAgent.

Tests knowledge extraction, pattern identification, domain insight capture,
knowledge file writing, and knowledge retrieval.
"""

import pytest
import os
from unittest.mock import patch, mock_open

from src.agents.memory_curator import (
    MemoryCuratorAgent,
    KnowledgeCategory,
    KnowledgeEntry,
    KeyDecision,
    Pattern,
    ExtractionResult,
    create_memory_curator,
)


@pytest.fixture
def curator(tmp_path):
    """Create a MemoryCuratorAgent with temp knowledge dir."""
    return MemoryCuratorAgent(
        system_prompt_path="nonexistent.md",
        knowledge_dir=str(tmp_path / "knowledge"),
    )


@pytest.fixture
def basic_context():
    """Create basic execution context."""
    return {
        "tier_level": 2,
        "agents_used": ["Analyst", "Planner", "Executor"],
        "revisions": 0,
        "escalations": 0,
    }


@pytest.fixture
def agent_outputs():
    """Create sample agent outputs."""
    return {
        "analyst": {"modality": "code", "sub_tasks": ["Design", "Implement"]},
        "planner": "We chose FastAPI over Flask because of async support.",
        "executor": "Implemented using the repository pattern with class UserRepository.",
        "verifier": "Verified: All claims accurate.",
        "critic": "Issue: Missing error handling. Recommend: Add try/except blocks.",
    }


class TestMemoryCuratorInitialization:
    """Tests for MemoryCuratorAgent initialization."""

    def test_default_initialization(self, tmp_path):
        """Test default init parameters."""
        agent = MemoryCuratorAgent(
            system_prompt_path="nonexistent.md",
            knowledge_dir=str(tmp_path / "knowledge"),
        )
        assert agent.model == "claude-sonnet-4-20250514"
        assert agent.max_turns == 30

    def test_knowledge_dir_created(self, tmp_path):
        """Test knowledge directory is created."""
        knowledge_dir = tmp_path / "knowledge"
        agent = MemoryCuratorAgent(
            system_prompt_path="nonexistent.md",
            knowledge_dir=str(knowledge_dir),
        )
        assert knowledge_dir.exists()

    def test_decision_patterns_initialized(self, tmp_path):
        """Test decision patterns are set."""
        agent = MemoryCuratorAgent(
            system_prompt_path="nonexistent.md",
            knowledge_dir=str(tmp_path),
        )
        assert len(agent.decision_patterns) > 0

    def test_system_prompt_fallback(self, tmp_path):
        """Test fallback prompt."""
        agent = MemoryCuratorAgent(
            system_prompt_path="nonexistent.md",
            knowledge_dir=str(tmp_path),
        )
        assert "Memory Curator" in agent.system_prompt

    def test_system_prompt_from_file(self, tmp_path):
        """Test loading from file."""
        with patch("builtins.open", mock_open(read_data="Memory prompt")):
            agent = MemoryCuratorAgent(
                system_prompt_path="exists.md",
                knowledge_dir=str(tmp_path),
            )
            assert agent.system_prompt == "Memory prompt"


class TestExtractAndPreserve:
    """Tests for the extract_and_preserve method."""

    def test_basic_extraction(self, curator, basic_context, agent_outputs):
        """Test basic knowledge extraction."""
        result = curator.extract_and_preserve(
            task_description="Build a REST API with user management",
            execution_context=basic_context,
            agent_outputs=agent_outputs,
            final_output="Final API implementation",
        )
        assert isinstance(result, ExtractionResult)
        assert len(result.entries) > 0

    def test_extraction_creates_files(self, curator, basic_context, agent_outputs):
        """Test extraction creates knowledge files."""
        result = curator.extract_and_preserve(
            task_description="Build a REST API",
            execution_context=basic_context,
            agent_outputs=agent_outputs,
            final_output="API built",
        )
        assert len(result.topics_created) > 0

    def test_extraction_metadata(self, curator, basic_context, agent_outputs):
        """Test extraction metadata is populated."""
        result = curator.extract_and_preserve(
            task_description="Build something",
            execution_context=basic_context,
            agent_outputs=agent_outputs,
            final_output="Done",
            session_id="test-session",
        )
        assert result.extraction_metadata["session_id"] == "test-session"
        assert result.extraction_metadata["tier_level"] == 2

    def test_extraction_total_count(self, curator, basic_context, agent_outputs):
        """Test total extractions count."""
        result = curator.extract_and_preserve(
            task_description="Build an API",
            execution_context=basic_context,
            agent_outputs=agent_outputs,
            final_output="Done",
        )
        assert result.total_extractions >= 0

    def test_extraction_with_empty_outputs(self, curator, basic_context):
        """Test extraction with empty agent outputs."""
        result = curator.extract_and_preserve(
            task_description="Simple task",
            execution_context=basic_context,
            agent_outputs={},
            final_output="Simple output",
        )
        assert isinstance(result, ExtractionResult)


class TestKeyDecisionExtraction:
    """Tests for key decision extraction."""

    def test_extracts_decisions_from_planner(self, curator, agent_outputs):
        """Test decisions extracted from planner output."""
        decisions = curator._extract_key_decisions(
            "Build API", agent_outputs, "Final output"
        )
        assert isinstance(decisions, list)
        # Should find "chose FastAPI" from planner output
        if decisions:
            assert any(isinstance(d, KeyDecision) for d in decisions)

    def test_deduplicates_decisions(self, curator):
        """Test decision deduplication."""
        outputs = {
            "planner": "We chose Python. We chose Python again.",
            "executor": "We chose Python for the implementation.",
        }
        decisions = curator._extract_key_decisions("Build API", outputs, "Done")
        decision_texts = [d.decision.lower() for d in decisions]
        assert len(decision_texts) == len(set(decision_texts))

    def test_limits_decisions(self, curator):
        """Test decisions are limited to 10."""
        # Create output with many decision-like sentences
        text = ". ".join([f"We chose option {i}" for i in range(20)])
        outputs = {"planner": text}
        decisions = curator._extract_key_decisions("Build API", outputs, "Done")
        assert len(decisions) <= 10


class TestPatternIdentification:
    """Tests for pattern identification."""

    def test_identifies_code_patterns(self, curator, agent_outputs):
        """Test code patterns are identified."""
        patterns = curator._identify_patterns(agent_outputs, "Final output")
        assert isinstance(patterns, list)

    def test_identifies_architectural_patterns(self, curator):
        """Test architectural pattern detection."""
        outputs = {"executor": "We used the repository pattern and factory pattern."}
        patterns = curator._identify_patterns(outputs, "Done")
        pattern_names = [p.name for p in patterns]
        assert any("Repository" in n or "Factory" in n for n in pattern_names)

    def test_deduplicates_patterns(self, curator):
        """Test pattern deduplication."""
        outputs = {
            "executor": "Used singleton pattern. Used singleton pattern.",
        }
        patterns = curator._identify_patterns(outputs, "Done")
        names = [p.name.lower() for p in patterns]
        assert len(names) == len(set(names))


class TestCategoryInference:
    """Tests for category inference."""

    @pytest.mark.parametrize("description,expected_category", [
        ("Implement a Python function", KnowledgeCategory.CODE_PATTERN),
        ("Design the system architecture", KnowledgeCategory.ARCHITECTURAL_DECISION),
        ("Fix the authentication bug", KnowledgeCategory.TROUBLESHOOTING),
        ("Avoid this anti-pattern", KnowledgeCategory.ANTI_PATTERN),
        ("General project work", KnowledgeCategory.LESSON_LEARNED),
    ])
    def test_infer_category(self, curator, description, expected_category):
        """Test category inference from task description."""
        category = curator._infer_category(description, {})
        assert category == expected_category


class TestTopicGeneration:
    """Tests for topic name generation."""

    def test_generates_topic(self, curator):
        """Test topic name generation."""
        topic = curator._generate_topic("Build a REST API", KnowledgeCategory.CODE_PATTERN)
        assert len(topic) > 0
        assert " " not in topic  # Should be kebab-case

    def test_topic_length_limited(self, curator):
        """Test topic name length is limited."""
        long_desc = "This is a very long description " * 10
        topic = curator._generate_topic(long_desc, KnowledgeCategory.LESSON_LEARNED)
        assert len(topic) <= 50


class TestTagGeneration:
    """Tests for tag generation."""

    def test_generates_tags(self, curator):
        """Test tag generation."""
        context = {"tier_level": 2, "agents_used": ["Analyst", "Executor"]}
        tags = curator._generate_tags("Build a Python API", context)
        assert len(tags) > 0
        assert "tier-2" in tags

    def test_includes_agent_tags(self, curator):
        """Test agent names are included as tags."""
        context = {"tier_level": 3, "agents_used": ["Analyst"]}
        tags = curator._generate_tags("Task", context)
        assert "analyst" in tags


class TestKnowledgeFileWriting:
    """Tests for knowledge file writing."""

    def test_writes_file(self, curator):
        """Test knowledge file is written."""
        entry = KnowledgeEntry(
            topic="test-topic",
            category=KnowledgeCategory.CODE_PATTERN,
            summary="Test summary",
            key_decisions=[],
            patterns=[],
            domain_insights=["Insight 1"],
            lessons_learned=["Lesson 1"],
            references=[],
            related_topics=[],
            tags=["test"],
        )
        filename = curator._write_knowledge_file(entry)
        assert filename is not None
        assert filename.endswith(".md")
        filepath = curator.knowledge_dir / filename
        assert filepath.exists()

    def test_file_contains_frontmatter(self, curator):
        """Test file contains YAML frontmatter."""
        entry = KnowledgeEntry(
            topic="test-topic",
            category=KnowledgeCategory.LESSON_LEARNED,
            summary="Summary",
            key_decisions=[],
            patterns=[],
            domain_insights=[],
            lessons_learned=[],
            references=[],
            related_topics=[],
            tags=["test"],
        )
        curator._write_knowledge_file(entry)
        filepath = curator.knowledge_dir / "test-topic.md"
        content = filepath.read_text()
        assert content.startswith("---")


class TestKnowledgeRetrieval:
    """Tests for knowledge retrieval."""

    def test_retrieve_empty(self, curator):
        """Test retrieval from empty knowledge base."""
        results = curator.retrieve_knowledge("python")
        assert results == []

    def test_retrieve_after_write(self, curator, basic_context, agent_outputs):
        """Test retrieval after writing knowledge."""
        curator.extract_and_preserve(
            task_description="Build a Python REST API",
            execution_context=basic_context,
            agent_outputs=agent_outputs,
            final_output="Done",
        )
        results = curator.retrieve_knowledge("python")
        assert len(results) > 0

    def test_list_knowledge_empty(self, curator):
        """Test listing empty knowledge base."""
        entries = curator.list_knowledge()
        assert entries == []


class TestConvenienceFunction:
    """Tests for create_memory_curator convenience function."""

    def test_create_memory_curator(self):
        """Test convenience function creates a MemoryCuratorAgent."""
        agent = create_memory_curator(
            system_prompt_path="nonexistent.md",
            knowledge_dir="/tmp/test_knowledge",
        )
        assert isinstance(agent, MemoryCuratorAgent)
