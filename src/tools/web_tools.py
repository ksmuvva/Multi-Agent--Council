"""
Web Tools Module

Provides web search and content fetching capabilities for the Researcher agent.
This module implements real web search using available APIs and services.
"""

import httpx
import re
import json
from urllib.parse import quote
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class SearchResult:
    """A search result from web search."""
    url: str
    title: str
    snippet: str
    body: Optional[str] = None


def web_search_tool(
    query: str,
    max_results: int = 5,
    timeout: int = 10
) -> List[Dict[str, Any]]:
    """
    Perform a web search using available search APIs.

    This function attempts to use multiple search strategies in order:
    1. DuckDuckGo HTML API (free, no API key needed)
    2. Manual search suggestions (when search fails)

    Args:
        query: The search query
        max_results: Maximum number of results to return
        timeout: Request timeout in seconds

    Returns:
        List of search results as dictionaries with keys: url, title, snippet, body
        Returns suggested manual search links when web search is unavailable.
    """
    # Try DuckDuckGo HTML API (free, no API key required)
    try:
        results = _search_duckduckgo(query, max_results, timeout)
        if results:
            return results
    except Exception as e:
        # Log and continue to fallback
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Web search failed for query '{query[:50]}...': {e}")

    # Fallback: Return manual search suggestions (not fake results)
    return _generate_contextual_results(query, max_results)


def _search_duckduckgo(query: str, max_results: int, timeout: int) -> List[Dict[str, Any]]:
    """
    Search using DuckDuckGo HTML API.

    This is a free API that doesn't require authentication.
    It returns HTML results that we parse for search results.

    Args:
        query: The search query
        max_results: Maximum results to return
        timeout: Request timeout

    Returns:
        List of search results
    """
    try:
        import html
        from urllib.parse import urlencode, quote

        # Build DuckDuckGo search URL
        params = {
            'q': query,
            'kl': 'us-en'
        }
        url = f"https://html.duckduckgo.com/html/?{urlencode(params)}"

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        with httpx.Client(timeout=timeout) as client:
            response = client.get(url, headers=headers, follow_redirects=True)

            if response.status_code != 200:
                return []

            # Parse HTML to extract search results
            results = _parse_duckduckgo_results(response.text, max_results)
            return results

    except Exception as e:
        return []


def _parse_duckduckgo_results(html: str, max_results: int) -> List[Dict[str, Any]]:
    """
    Parse DuckDuckGo HTML results to extract search results.

    Args:
        html: The HTML response from DuckDuckGo
        max_results: Maximum results to extract

    Returns:
        List of parsed search results
    """
    results = []

    try:
        # DuckDuckGo HTML results have specific structure
        # Search results are in <a> tags with class="result__a"
        # Snippets are in <a> tags with class="result__snippet"

        # Pattern for finding result blocks
        result_pattern = r'<a[^>]*class="result__a[^"]*"[^>]*>(.*?)</a>.*?<a[^>]*class="result__snippet[^"]*"[^>]*>(.*?)</a>'

        matches = re.findall(result_pattern, html, re.DOTALL)

        for i, (title, snippet) in enumerate(matches[:max_results]):
            # Clean up HTML entities and tags
            title = re.sub(r'<[^>]+>', '', title)
            title = html.unescape(title)
            title = title.strip()

            snippet = re.sub(r'<[^>]+>', '', snippet)
            snippet = html.unescape(snippet)
            snippet = snippet.strip()

            # Extract URL from the result link
            # The actual URL is in the href attribute
            url_match = re.search(r'<a[^>]*href="([^"]+)"', html[html.find(title)-200:html.find(title)+200])
            if url_match:
                url = html.unescape(url_match.group(1))
                # DuckDuckGo redirects through their URL - get the actual URL
                if 'duckduckgo.com/l/?uddg=' in url:
                    actual_url = re.sub(r'^.*?uddg=([^&]+).*$', r'\1', url)
                    # Decode the URL (it's encoded)
                    import urllib.parse
                    try:
                        actual_url = urllib.parse.unquote(actual_url)
                    except:
                        actual_url = url
                    url = actual_url
            else:
                url = f"https://duckduckgo.com/?q={quote(title)}"

            if title:  # Only add if we have a title
                results.append({
                    'url': url,
                    'title': title,
                    'snippet': snippet or f"Search result for: {title}",
                    'body': snippet
                })

    except Exception as e:
        pass  # Return empty list on parse failure

    return results


def web_fetch_tool(
    url: str,
    prompt: str = "Extract the main content from this page",
    max_length: int = 5000
) -> str:
    """
    Fetch and extract content from a URL.

    Args:
        url: The URL to fetch
        prompt: What to extract from the page
        max_length: Maximum content length to return

    Returns:
        Extracted content as text
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        with httpx.Client(timeout=30) as client:
            response = client.get(url, headers=headers, follow_redirects=True)

            if response.status_code == 200:
                content_type = response.headers.get('content-type', '')

                # Handle different content types
                if 'html' in content_type:
                    return _extract_html_content(response.text, url, max_length)
                elif 'json' in content_type:
                    return _extract_json_content(response.text, max_length)
                else:
                    # Plain text or other
                    content = response.text
                    if len(content) > max_length:
                        content = content[:max_length] + "... [truncated]"
                    return content
            else:
                return f"Error: HTTP {response.status_code} when fetching {url}"

    except httpx.TimeoutError:
        return f"Error: Timeout while fetching {url}"
    except Exception as e:
        return f"Error: {str(e)} when fetching {url}"


def _extract_html_content(html: str, url: str, max_length: int) -> str:
    """
    Extract readable content from HTML.

    Args:
        html: The HTML content
        url: Source URL
        max_length: Maximum length

    Returns:
        Extracted text content
    """
    try:
        # Remove script and style tags
        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)

        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '\n', html)

        # Clean up whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)

        # Decode HTML entities
        import html as html_module
        text = html_module.unescape(text)

        text = text.strip()

        # Add source attribution
        result = f"Content from: {url}\n\n{text}"

        if len(result) > max_length:
            result = result[:max_length] + "\n\n... [Content truncated due to length]"

        return result

    except Exception:
        # If extraction fails, return a portion of the original HTML
        html = html[:max_length] + "..." if len(html) > max_length else html
        return f"Content from: {url}\n\n[Raw HTML - could not parse]\n\n{html}"


def _extract_json_content(json_text: str, max_length: int) -> str:
    """
    Extract content from JSON response.

    Args:
        json_text: The JSON content
        max_length: Maximum length

    Returns:
        Formatted JSON content
    """
    try:
        data = json.loads(json_text)
        formatted = json.dumps(data, indent=2)

        if len(formatted) > max_length:
            formatted = formatted[:max_length] + "\n\n... [Truncated]"

        return formatted
    except:
        # If not valid JSON, return as-is
        return json_text[:max_length] + "..." if len(json_text) > max_length else json_text


def _generate_contextual_results(query: str, max_results: int) -> List[Dict[str, Any]]:
    """
    Return search failure notice when real search is unavailable.

    Previously this generated fake results, but that was misleading.
    Now it returns a clear message that search failed, along with
    suggested manual search URLs.

    Args:
        query: The search query
        max_results: Maximum results (ignored, always returns suggestions)

    Returns:
        List with suggested manual search actions (not actual search results)
    """
    from urllib.parse import quote

    # Return helpful suggestions for manual searching instead of fake results
    return [
        {
            'url': f"https://duckduckgo.com/?q={quote(query)}",
            'title': f"Search for '{query}' on DuckDuckGo",
            'snippet': "⚠️ Web search is currently unavailable. Click here to search manually.",
            'body': "Web search service is not configured. Please search manually using this link.",
            '_suggested': True,  # Marker that this is a suggestion, not a real result
        },
        {
            'url': f"https://www.google.com/search?q={quote(query)}",
            'title': f"Search for '{query}' on Google",
            'snippet': "⚠️ Alternative: Search manually on Google.",
            'body': "Web search service is not configured. Please search manually using this link.",
            '_suggested': True,
        },
    ]
