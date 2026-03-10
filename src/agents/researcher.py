"""
Researcher Subagent

Gathers evidence from external sources using WebSearch and WebFetch,
producing EvidenceBrief with confidence levels and source reliability.
"""

import re
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from urllib.parse import urlparse

from src.schemas.researcher import (
    EvidenceBrief,
    Source,
    Finding,
    Conflict,
    KnowledgeGap,
    ConfidenceLevel,
    SourceReliability,
)


@dataclass
class SearchResult:
    """A search result from WebSearch."""
    url: str
    title: str
    snippet: str
    relevance_score: float


@dataclass
class FetchedContent:
    """Content fetched from a URL."""
    url: str
    title: str
    content: str
    word_count: int
    extraction_successful: bool


class ResearcherAgent:
    """
    The Researcher gathers evidence from external sources.

    Key responsibilities:
    - Use WebSearch for broad queries
    - Use WebFetch for specific URLs
    - Evaluate source reliability
    - Synthesize findings by confidence level
    - Flag conflicting information
    - Identify knowledge gaps
    """

    def __init__(
        self,
        system_prompt_path: str = "config/agents/researcher/CLAUDE.md",
        model: str = "claude-3-5-sonnet-20241022",
        max_turns: int = 30,
    ):
        """
        Initialize the Researcher agent.

        Args:
            system_prompt_path: Path to system prompt file
            model: Model to use for research
            max_turns: Maximum conversation turns
        """
        self.system_prompt_path = system_prompt_path
        self.model = model
        self.max_turns = max_turns
        self.system_prompt = self._load_system_prompt()

        # Source reliability patterns
        self.authoritative_domains = {
            # Official documentation
            "docs.python.org": SourceReliability.HIGH,
            "developer.mozilla.org": SourceReliability.HIGH,
            "docs.oracle.com": SourceReliability.HIGH,
            "devdocs.io": SourceReliability.HIGH,
            # Tech giants
            "azure.microsoft.com": SourceReliability.HIGH,
            "cloud.google.com": SourceReliability.HIGH,
            "docs.aws.amazon.com": SourceReliability.HIGH,
            # Package repositories
            "pypi.org": SourceReliability.HIGH,
            "npmjs.com": SourceReliability.HIGH,
            "crates.io": SourceReliability.HIGH,
        }

        # Common knowledge domains (medium reliability)
        self.medium_reliability_patterns = [
            "stackoverflow.com",
            "github.com",
            "medium.com",
            "towardsdatascience.com",
            "realpython.com",
        ]

    def research(
        self,
        topic: str,
        queries: Optional[List[str]] = None,
        specific_urls: Optional[List[str]] = None,
        sme_inputs: Optional[Dict[str, str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> EvidenceBrief:
        """
        Research a topic and produce an EvidenceBrief.

        Args:
            topic: The topic to research
            queries: Optional specific search queries (generated if not provided)
            specific_urls: Optional specific URLs to fetch
            sme_inputs: Optional inputs from SME personas
            context: Additional context (tier, previous findings, etc.)

        Returns:
            EvidenceBrief with research findings
        """
        # Generate search queries if not provided
        if not queries:
            queries = self._generate_search_queries(topic)

        # Step 1: Search for information
        search_results = self._perform_searches(queries)

        # Step 2: Fetch content from promising results
        fetched_content = self._fetch_content(search_results[:5])

        # Step 3: Extract findings
        findings = self._extract_findings(fetched_content, topic)

        # Step 4: Identify conflicts
        conflicts = self._identify_conflicts(findings)

        # Step 5: Identify gaps
        gaps = self._identify_knowledge_gaps(topic, findings, queries)

        # Step 6: Incorporate SME inputs if provided
        if sme_inputs:
            findings = self._incorporate_sme_inputs(findings, sme_inputs)

        # Step 7: Determine overall confidence
        overall_confidence = self._calculate_overall_confidence(findings)

        # Step 8: Generate recommended approach
        recommended_approach = self._recommend_approach(
            findings, conflicts, gaps, overall_confidence
        )

        # Step 9: Build summary
        summary = self._build_summary(topic, findings, overall_confidence)

        # Build sources list
        sources = self._build_sources(fetched_content)

        return EvidenceBrief(
            research_topic=topic,
            summary=summary,
            findings=findings,
            conflicts=conflicts,
            gaps=gaps,
            overall_confidence=overall_confidence,
            recommended_approach=recommended_approach,
            additional_research_needed=len(gaps) > 0 or overall_confidence == ConfidenceLevel.LOW,
        )

    # ========================================================================
    # Search Methods
    # ========================================================================

    def _generate_search_queries(self, topic: str) -> List[str]:
        """Generate effective search queries for a topic."""
        queries = []

        # Primary query
        queries.append(topic)

        # Add qualifiers for better results
        query_variations = [
            f"{topic} best practices",
            f"{topic} tutorial",
            f"{topic} documentation",
            f"{topic} example",
            f"how to {topic.lower()}",
            f"{topic} guide",
        ]

        # Remove duplicates and limit
        seen = set()
        unique_queries = []
        for q in [topic] + query_variations:
            if q.lower() not in seen:
                seen.add(q.lower())
                unique_queries.append(q)

        return unique_queries[:5]

    def _perform_searches(self, queries: List[str]) -> List[SearchResult]:
        """
        Perform web searches for the queries.

        Uses the SDK spawn_subagent() with WebSearch tool when available,
        falling back to mock results for development/testing.
        """
        results = []

        try:
            from src.core.sdk_integration import build_agent_options, spawn_subagent

            # Spawn a research subagent with WebSearch capability
            options = build_agent_options(
                agent_name="researcher",
                system_prompt="You are a research assistant. For each query, search the web and return results as JSON with fields: url, title, snippet, relevance_score (0-1).",
                model_override=self.model,
            )

            search_prompt = "Search for the following queries and return results as a JSON array:\n"
            for i, q in enumerate(queries, 1):
                search_prompt += f"{i}. {q}\n"

            result = spawn_subagent(options=options, input_data=search_prompt, max_retries=1)

            if result.get("status") == "success" and result.get("output"):
                # Parse SDK response for search results
                output = result["output"]
                parsed = self._parse_search_output(output)
                if parsed:
                    results.extend(parsed)
        except ImportError:
            pass  # SDK not installed — fall back to mock results
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                "SDK search failed for researcher, falling back to mock: %s", e
            )

        # Fall back to mock results if SDK didn't return enough
        if len(results) < 2:
            for query in queries:
                mock_results = self._mock_search_results(query)
                results.extend(mock_results)

        # Deduplicate and rank by relevance
        seen_urls: Set[str] = set()
        unique_results = []
        for result in results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)

        unique_results.sort(key=lambda r: r.relevance_score, reverse=True)

        return unique_results

    def _parse_search_output(self, output: Any) -> List[SearchResult]:
        """Parse search results from SDK output."""
        import json
        results = []
        try:
            if isinstance(output, str):
                # Try to parse the entire output as JSON first
                try:
                    parsed = json.loads(output)
                except json.JSONDecodeError:
                    # Fall back to extracting the first JSON array
                    start = output.find('[')
                    end = output.rfind(']') + 1
                    if start < 0 or end <= start:
                        return results
                    parsed = json.loads(output[start:end])

                if not isinstance(parsed, list):
                    return results

                for item in parsed:
                    if not isinstance(item, dict):
                        continue
                    url = item.get("url", "")
                    # Skip items with empty URLs — they are not useful results
                    if not url:
                        continue
                    results.append(SearchResult(
                        url=url,
                        title=item.get("title", ""),
                        snippet=item.get("snippet", ""),
                        relevance_score=float(item.get("relevance_score", 0.5)),
                    ))
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
        return results

    def _mock_search_results(self, query: str) -> List[SearchResult]:
        """Generate mock search results for demonstration."""
        # In real implementation, this would call actual WebSearch
        return [
            SearchResult(
                url=f"https://example.com/docs/{query.replace(' ', '-')}",
                title=f"Documentation for {query}",
                snippet=f"This is the official documentation covering {query}...",
                relevance_score=0.9
            ),
            SearchResult(
                url=f"https://example.com/guide/{query.replace(' ', '-')}",
                title=f"Complete Guide to {query}",
                snippet=f"A comprehensive guide explaining {query} in detail...",
                relevance_score=0.8
            ),
        ]

    def _fetch_content(self, search_results: List[SearchResult]) -> List[FetchedContent]:
        """
        Fetch content from search results.

        Uses SDK spawn_subagent() with WebFetch tool when available,
        falling back to mock content for development/testing.
        """
        fetched = []

        for result in search_results:
            content = None

            try:
                from src.core.sdk_integration import build_agent_options, spawn_subagent

                options = build_agent_options(
                    agent_name="researcher",
                    system_prompt="Fetch the content from the given URL and return the main text content.",
                    model_override=self.model,
                )

                fetch_result = spawn_subagent(
                    options=options,
                    input_data=f"Fetch content from: {result.url}",
                    max_retries=0,
                )

                if fetch_result.get("status") == "success" and fetch_result.get("output"):
                    content = str(fetch_result["output"])
            except ImportError:
                pass  # SDK not installed — fall back to mock content
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    "SDK fetch failed for URL %s, falling back to mock: %s",
                    result.url, e,
                )

            # Fall back to mock
            if not content:
                content = self._mock_fetched_content(result.url)

            fetched.append(FetchedContent(
                url=result.url,
                title=result.title,
                content=content,
                word_count=len(content.split()),
                extraction_successful=True,
            ))

        return fetched

    def _mock_fetched_content(self, url: str) -> str:
        """Generate mock fetched content for demonstration."""
        return f"""
        This is the content from {url}.

        It contains detailed information about the topic, including:
        - Technical specifications
        - Best practices
        - Code examples
        - Common pitfalls and how to avoid them

        The content is well-structured and authoritative.
        """

    # ========================================================================
    # Analysis Methods
    # ========================================================================

    def _extract_findings(
        self,
        fetched_content: List[FetchedContent],
        topic: str
    ) -> List[Finding]:
        """Extract findings from fetched content."""
        findings = []

        for content in fetched_content:
            # Analyze content for key findings
            # In real implementation, this would use LLM to extract structured findings

            # Placeholder: extract key sentences/statements
            key_sentences = self._extract_key_sentences(content.content)

            for sentence in key_sentences:
                confidence = self._assess_finding_confidence(sentence, content)

                findings.append(Finding(
                    claim=sentence,
                    confidence=confidence,
                    sources=[
                        Source(
                            url=content.url,
                            title=content.title,
                            reliability=self._assess_source_reliability(content.url),
                            access_date="2024-03-07",  # In real impl: current date
                            excerpt=sentence[:100] + "..."
                        )
                    ],
                    context=f"From {content.title}",
                    caveats=[] if confidence == ConfidenceLevel.HIGH else ["Single source"],
                ))

        return findings

    def _extract_key_sentences(self, content: str) -> List[str]:
        """Extract key sentences from content."""
        # Split into sentences
        sentences = re.split(r'[.!?]+', content)

        # Filter for substantial sentences
        key_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 50 and len(sentence) < 200:
                key_sentences.append(sentence)

        return key_sentences[:5]  # Top 5 sentences

    def _assess_finding_confidence(
        self,
        claim: str,
        content: FetchedContent
    ) -> ConfidenceLevel:
        """Assess confidence level for a finding."""
        # High confidence if from authoritative source
        source_reliability = self._assess_source_reliability(content.url)

        if source_reliability == SourceReliability.HIGH:
            return ConfidenceLevel.HIGH
        elif source_reliability == SourceReliability.MEDIUM:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW

    def _assess_source_reliability(self, url: str) -> SourceReliability:
        """Assess the reliability of a source."""
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()

        # Check authoritative domains
        if domain in self.authoritative_domains:
            return self.authoritative_domains[domain]

        # Check medium reliability patterns
        for pattern in self.medium_reliability_patterns:
            if pattern in domain:
                return SourceReliability.MEDIUM

        # Default to unknown
        return SourceReliability.UNKNOWN

    def _build_sources(self, fetched_content: List[FetchedContent]) -> List[Source]:
        """Build a list of sources from fetched content."""
        sources = []
        for content in fetched_content:
            sources.append(Source(
                url=content.url,
                title=content.title,
                reliability=self._assess_source_reliability(content.url),
                access_date="2024-03-07",
                excerpt=content.content[:100] + "..."
            ))
        return sources

    def _identify_conflicts(self, findings: List[Finding]) -> List[Conflict]:
        """Identify conflicting information in findings."""
        conflicts = []

        # Group findings by similar claims
        claim_groups = self._group_similar_claims(findings)

        for group in claim_groups:
            if len(group) > 1:
                # Check if findings have different confidence levels
                confidences = [f.confidence for f in group]
                if len(set(confidences)) > 1:
                    # Find the highest and lowest confidence findings
                    sorted_findings = sorted(
                        group,
                        key=lambda f: f.confidence.value,
                        reverse=True
                    )
                    high_conf = sorted_findings[0]
                    low_conf = sorted_findings[-1]

                    if high_conf.sources and low_conf.sources:
                        conflicts.append(Conflict(
                            claim=high_conf.claim[:50] + "...",
                            source_a=high_conf.sources[0],
                            source_b=low_conf.sources[0],
                            description=f"Differing confidence levels: {high_conf.confidence.value} vs {low_conf.confidence.value}",
                            resolution_suggestion="Prefer higher confidence source"
                        ))

        return conflicts

    def _group_similar_claims(self, findings: List[Finding]) -> List[List[Finding]]:
        """Group findings with similar claims."""
        # Simple similarity based on keywords
        # In real implementation, would use embedding similarity

        groups = []
        used = set()

        for i, finding in enumerate(findings):
            if i in used:
                continue

            group = [finding]
            used.add(i)

            # Find similar findings
            claim_words = set(finding.claim.lower().split())

            for j, other in enumerate(findings):
                if j <= i or j in used:
                    continue

                other_words = set(other.claim.lower().split())
                overlap = len(claim_words & other_words)

                if overlap > 3:  # Significant overlap
                    group.append(other)
                    used.add(j)

            if len(group) > 1:
                groups.append(group)

        return groups

    def _identify_knowledge_gaps(
        self,
        topic: str,
        findings: List[Finding],
        queries: List[str]
    ) -> List[KnowledgeGap]:
        """Identify information that couldn't be found."""
        gaps = []

        # Check for aspects not covered in findings
        finding_claims = [f.claim.lower() for f in findings]

        # Common gap patterns
        gap_keywords = {
            "security": ["security", "authentication", "authorization", "vulnerability"],
            "performance": ["performance", "optimization", "scalability", "latency"],
            "error handling": ["error", "exception", "failure", "edge case"],
            "testing": ["test", "testing", "coverage", "unit test"],
        }

        for aspect, keywords in gap_keywords.items():
            aspect_covered = any(
                any(kw in claim for kw in keywords)
                for claim in finding_claims
            )

            if not aspect_covered:
                gaps.append(KnowledgeGap(
                    topic=f"{aspect} considerations for {topic}",
                    why_important=f"{aspect.capitalize()} is important for production-quality solutions",
                    searched_sources=[q for q in queries],
                    suggested_approaches=[
                        f"Consider {aspect} requirements",
                        f"Apply {aspect} best practices",
                        f"Test {aspect} scenarios"
                    ]
                ))

        return gaps

    # ========================================================================
    # SME Integration
    # ========================================================================

    def _incorporate_sme_inputs(
        self,
        findings: List[Finding],
        sme_inputs: Dict[str, str]
    ) -> List[Finding]:
        """Incorporate domain SME inputs into findings."""
        # In real implementation, would combine web research with SME expertise

        # Add SME inputs as high-confidence findings
        for sme, input_text in sme_inputs.items():
            findings.append(Finding(
                claim=f"SME ({sme}): {input_text[:100]}...",
                confidence=ConfidenceLevel.HIGH,
                sources=[
                    Source(
                        url=f"SME: {sme}",
                        title=f"Domain Expert: {sme}",
                        reliability=SourceReliability.HIGH,
                        access_date="2024-03-07",
                        excerpt=input_text[:100]
                    )
                ],
                context=f"Domain expertise from {sme}",
                caveats=["Based on SME domain knowledge"]
            ))

        return findings

    # ========================================================================
    # Assessment Methods
    # ========================================================================

    def _calculate_overall_confidence(
        self,
        findings: List[Finding]
    ) -> ConfidenceLevel:
        """Calculate overall confidence in the research."""
        if not findings:
            return ConfidenceLevel.LOW

        # Count confidence levels
        high_count = sum(1 for f in findings if f.confidence == ConfidenceLevel.HIGH)
        medium_count = sum(1 for f in findings if f.confidence == ConfidenceLevel.MEDIUM)
        low_count = sum(1 for f in findings if f.confidence == ConfidenceLevel.LOW)

        total = len(findings)

        # Determine overall confidence
        if high_count / total >= 0.7:
            return ConfidenceLevel.HIGH
        elif (high_count + medium_count) / total >= 0.7:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW

    def _recommend_approach(
        self,
        findings: List[Finding],
        conflicts: List[Conflict],
        gaps: List[KnowledgeGap],
        overall_confidence: ConfidenceLevel
    ) -> str:
        """Generate a recommended approach based on research."""
        if overall_confidence == ConfidenceLevel.HIGH and not conflicts:
            return "Proceed with high-confidence findings from authoritative sources"
        elif conflicts:
            return "Address conflicting sources by preferring higher-confidence findings"
        elif gaps:
            return f"Proceed with caution - {len(gaps)} knowledge gaps identified"
        elif overall_confidence == ConfidenceLevel.MEDIUM:
            return "Use findings but verify critical claims with additional sources"
        else:
            return "Limited information - recommend additional research or SME consultation"

    def _build_summary(
        self,
        topic: str,
        findings: List[Finding],
        overall_confidence: ConfidenceLevel
    ) -> str:
        """Build a summary of the research."""
        count = len(findings)
        return (
            f"Research on '{topic}' found {count} key findings "
            f"with {overall_confidence.value} confidence. "
        )

    def _load_system_prompt(self) -> str:
        """Load the system prompt from file."""
        try:
            with open(self.system_prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            return "You are the Researcher. Gather evidence from external sources."


# =============================================================================
# Convenience Functions
# =============================================================================

def create_researcher(
    system_prompt_path: str = "config/agents/researcher/CLAUDE.md",
    model: str = "claude-3-5-sonnet-20241022",
) -> ResearcherAgent:
    """Create a configured Researcher agent."""
    return ResearcherAgent(
        system_prompt_path=system_prompt_path,
        model=model,
    )
