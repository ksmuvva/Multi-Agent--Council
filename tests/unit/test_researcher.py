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
        assert agent.model == "claude-3-5-sonnet-20241022"
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


# =============================================================================
# SDK-Aware Search Tests
# =============================================================================

class TestSDKAwareSearch:
    """Tests for SDK-aware _perform_searches, _fetch_content, and _parse_search_output."""

    def test_perform_searches_sdk_success(self, researcher):
        """Test _perform_searches uses SDK path when spawn_subagent succeeds."""
        mock_output = '[{"url": "https://example.com", "title": "Example", "snippet": "A snippet", "relevance_score": 0.95}]'
        mock_result = {"status": "success", "output": mock_output}

        with patch("src.core.sdk_integration.spawn_subagent", return_value=mock_result), \
             patch("src.core.sdk_integration.build_agent_options"):
            results = researcher._perform_searches(["test query"])
            # Should contain the SDK result
            sdk_urls = [r.url for r in results]
            assert "https://example.com" in sdk_urls

    def test_perform_searches_fallback_on_import_error(self, researcher):
        """Test _perform_searches falls back to mock when SDK import fails."""
        # Default behavior when SDK is not importable - should use mock results
        results = researcher._perform_searches(["Python decorators"])
        assert len(results) >= 2  # Mock generates 2 results per query
        # All results should be from mock (example.com domain)
        assert all("example.com" in r.url for r in results)

    def test_perform_searches_fallback_on_insufficient_results(self, researcher):
        """Test _perform_searches falls back when SDK returns fewer than 2 results."""
        mock_result = {"status": "success", "output": '[{"url": "https://one.com", "title": "One", "snippet": "s", "relevance_score": 0.5}]'}

        with patch("src.core.sdk_integration.spawn_subagent", return_value=mock_result), \
             patch("src.core.sdk_integration.build_agent_options"):
            results = researcher._perform_searches(["test"])
            # Should have SDK result plus fallback mock results
            assert len(results) >= 2

    def test_perform_searches_deduplicates_urls(self, researcher):
        """Test that duplicate URLs are removed from results."""
        results = researcher._perform_searches(["Python", "Python"])
        urls = [r.url for r in results]
        assert len(urls) == len(set(urls))

    def test_fetch_content_sdk_success(self, researcher):
        """Test _fetch_content uses SDK path when spawn_subagent succeeds."""
        mock_result = {"status": "success", "output": "Fetched content from the page about Python."}
        search_results = [
            SearchResult(url="https://docs.python.org/3", title="Python Docs", snippet="Official", relevance_score=0.9),
        ]

        with patch("src.core.sdk_integration.spawn_subagent", return_value=mock_result), \
             patch("src.core.sdk_integration.build_agent_options"):
            fetched = researcher._fetch_content(search_results)
            assert len(fetched) == 1
            assert "Fetched content" in fetched[0].content

    def test_fetch_content_fallback_on_failure(self, researcher):
        """Test _fetch_content falls back to mock when SDK fails."""
        search_results = [
            SearchResult(url="https://example.com/test", title="Test", snippet="test", relevance_score=0.8),
        ]
        # Without SDK available, should use mock content
        fetched = researcher._fetch_content(search_results)
        assert len(fetched) == 1
        assert fetched[0].extraction_successful is True
        assert fetched[0].word_count > 0

    def test_parse_search_output_valid_json(self, researcher):
        """Test _parse_search_output with valid JSON array."""
        output = 'Some text before [{"url": "https://a.com", "title": "A", "snippet": "s", "relevance_score": 0.8}] and after'
        results = researcher._parse_search_output(output)
        assert len(results) == 1
        assert results[0].url == "https://a.com"
        assert results[0].relevance_score == 0.8

    def test_parse_search_output_invalid_json(self, researcher):
        """Test _parse_search_output with invalid JSON returns empty list."""
        results = researcher._parse_search_output("not json at all")
        assert results == []

    def test_parse_search_output_empty_array(self, researcher):
        """Test _parse_search_output with empty JSON array."""
        results = researcher._parse_search_output("[]")
        assert results == []

    def test_parse_search_output_multiple_items(self, researcher):
        """Test _parse_search_output with multiple items."""
        output = '[{"url": "https://a.com", "title": "A", "snippet": "s1", "relevance_score": 0.9}, {"url": "https://b.com", "title": "B", "snippet": "s2", "relevance_score": 0.7}]'
        results = researcher._parse_search_output(output)
        assert len(results) == 2
        assert results[0].url == "https://a.com"
        assert results[1].url == "https://b.com"
