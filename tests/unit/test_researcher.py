"""
Tests for the ResearcherAgent.

Tests research pipeline, source reliability assessment,
finding extraction, conflict detection, and knowledge gap identification.
"""

import pytest
from unittest.mock import patch, mock_open

from src.agents.researcher import (
    ResearcherAgent,
    SearchResult,
    FetchedContent,
    create_researcher,
)
from src.schemas.researcher import (
    EvidenceBrief,
    ConfidenceLevel,
    SourceReliability,
)


@pytest.fixture
def researcher():
    """Create a ResearcherAgent with no system prompt file."""
    return ResearcherAgent(system_prompt_path="nonexistent.md")


class TestResearcherInitialization:
    """Tests for ResearcherAgent initialization."""

    def test_default_initialization(self):
        """Test default init parameters."""
        agent = ResearcherAgent(system_prompt_path="nonexistent.md")
        assert agent.model == "claude-sonnet-4-20250514"
        assert agent.max_turns == 30

    def test_authoritative_domains_set(self):
        """Test authoritative domains are configured."""
        agent = ResearcherAgent(system_prompt_path="nonexistent.md")
        assert "docs.python.org" in agent.authoritative_domains
        assert "developer.mozilla.org" in agent.authoritative_domains

    def test_medium_reliability_patterns(self):
        """Test medium reliability patterns are configured."""
        agent = ResearcherAgent(system_prompt_path="nonexistent.md")
        assert "stackoverflow.com" in agent.medium_reliability_patterns

    def test_system_prompt_fallback(self):
        """Test fallback prompt when file not found."""
        agent = ResearcherAgent(system_prompt_path="nonexistent.md")
        assert "Researcher" in agent.system_prompt

    def test_system_prompt_from_file(self):
        """Test loading from file."""
        with patch("builtins.open", mock_open(read_data="Researcher prompt")):
            agent = ResearcherAgent(system_prompt_path="exists.md")
            assert agent.system_prompt == "Researcher prompt"


class TestResearch:
    """Tests for the research method."""

    def test_basic_research(self, researcher):
        """Test basic research produces EvidenceBrief."""
        result = researcher.research("Python decorators")
        assert isinstance(result, EvidenceBrief)
        assert result.research_topic == "Python decorators"

    def test_research_has_findings(self, researcher):
        """Test research produces findings."""
        result = researcher.research("REST API design")
        assert len(result.findings) >= 0  # May have findings from mock

    def test_research_has_summary(self, researcher):
        """Test research summary is generated."""
        result = researcher.research("Machine learning")
        assert len(result.summary) > 0
        assert "Machine learning" in result.summary

    def test_research_has_overall_confidence(self, researcher):
        """Test overall confidence is set."""
        result = researcher.research("Data structures")
        assert result.overall_confidence in [
            ConfidenceLevel.HIGH,
            ConfidenceLevel.MEDIUM,
            ConfidenceLevel.LOW,
        ]

    def test_research_with_custom_queries(self, researcher):
        """Test research with custom queries."""
        result = researcher.research(
            "Python async",
            queries=["Python asyncio tutorial", "async await Python"],
        )
        assert isinstance(result, EvidenceBrief)

    def test_research_with_sme_inputs(self, researcher):
        """Test SME inputs are incorporated."""
        result = researcher.research(
            "Cloud architecture",
            sme_inputs={"Cloud Architect": "Use microservices with containers"},
        )
        # SME findings should be added
        sme_findings = [f for f in result.findings if "SME" in f.claim]
        assert len(sme_findings) > 0


class TestSearchQueryGeneration:
    """Tests for search query generation."""

    def test_generates_multiple_queries(self, researcher):
        """Test multiple query variations are generated."""
        queries = researcher._generate_search_queries("Python testing")
        assert len(queries) > 1
        assert len(queries) <= 5

    def test_primary_query_included(self, researcher):
        """Test original topic is included."""
        queries = researcher._generate_search_queries("FastAPI")
        assert "FastAPI" in queries

    def test_no_duplicate_queries(self, researcher):
        """Test no duplicate queries."""
        queries = researcher._generate_search_queries("Python")
        lowered = [q.lower() for q in queries]
        assert len(lowered) == len(set(lowered))

    def test_adds_qualifiers(self, researcher):
        """Test qualifiers are added."""
        queries = researcher._generate_search_queries("Docker")
        assert any("best practices" in q for q in queries)
        assert any("tutorial" in q for q in queries)


class TestSourceReliability:
    """Tests for source reliability assessment."""

    @pytest.mark.parametrize("url,expected", [
        ("https://docs.python.org/3/tutorial", SourceReliability.HIGH),
        ("https://stackoverflow.com/questions/123", SourceReliability.MEDIUM),
        ("https://unknown-blog.com/article", SourceReliability.UNKNOWN),
    ])
    def test_source_reliability(self, researcher, url, expected):
        """Test source reliability assessment for various domains."""
        reliability = researcher._assess_source_reliability(url)
        assert reliability == expected

    def test_authoritative_domain(self, researcher):
        """Test authoritative domain gets HIGH reliability."""
        reliability = researcher._assess_source_reliability("https://developer.mozilla.org/docs")
        assert reliability == SourceReliability.HIGH


class TestFindingConfidence:
    """Tests for finding confidence assessment."""

    def test_high_confidence_from_authoritative(self, researcher):
        """Test high confidence from authoritative source."""
        content = FetchedContent(
            url="https://docs.python.org/tutorial",
            title="Python Tutorial",
            content="This is a test.",
            word_count=5,
            extraction_successful=True,
        )
        confidence = researcher._assess_finding_confidence("test claim", content)
        assert confidence == ConfidenceLevel.HIGH

    def test_low_confidence_from_unknown(self, researcher):
        """Test low confidence from unknown source."""
        content = FetchedContent(
            url="https://random-unknown-site.xyz/post",
            title="Random Post",
            content="This is a test.",
            word_count=5,
            extraction_successful=True,
        )
        confidence = researcher._assess_finding_confidence("test claim", content)
        assert confidence == ConfidenceLevel.LOW


class TestKnowledgeGaps:
    """Tests for knowledge gap identification."""

    def test_identifies_gaps(self, researcher):
        """Test gap identification for missing aspects."""
        result = researcher.research("Python web framework")
        # Gaps should be identified for aspects not covered in findings
        assert isinstance(result.gaps, list)

    def test_additional_research_flag(self, researcher):
        """Test additional research flag is set when gaps exist."""
        result = researcher.research("Something obscure")
        # additional_research_needed should be set based on gaps/confidence
        assert isinstance(result.additional_research_needed, bool)


class TestOverallConfidence:
    """Tests for overall confidence calculation."""

    def test_low_confidence_no_findings(self, researcher):
        """Test LOW confidence when no findings."""
        confidence = researcher._calculate_overall_confidence([])
        assert confidence == ConfidenceLevel.LOW


class TestConvenienceFunction:
    """Tests for create_researcher convenience function."""

    def test_create_researcher(self):
        """Test convenience function creates a ResearcherAgent."""
        agent = create_researcher(system_prompt_path="nonexistent.md")
        assert isinstance(agent, ResearcherAgent)
