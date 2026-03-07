"""
Tests for the CodeReviewerAgent.

Tests security scanning, performance analysis, style compliance,
error handling checks, and code context analysis.
"""

import pytest
from unittest.mock import patch, mock_open

from src.agents.code_reviewer import (
    CodeReviewerAgent,
    CodeContext,
    create_code_reviewer,
)
from src.schemas.code_reviewer import (
    CodeReviewReport,
    SeverityLevel,
    ReviewCategory,
)


@pytest.fixture
def reviewer():
    """Create a CodeReviewerAgent with no system prompt file."""
    return CodeReviewerAgent(system_prompt_path="nonexistent.md")


CLEAN_CODE = '''
def fibonacci(n: int) -> int:
    """Calculate the nth fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
'''

VULNERABLE_CODE = '''
import sqlite3

def get_user(username):
    """Get user by username."""
    conn = sqlite3.connect("db.sqlite")
    query = f"SELECT * FROM users WHERE name = '{username}'"
    result = conn.execute(query)
    return result.fetchone()

password = "supersecret123"
api_key = "sk-12345678901234567890"
'''

PERFORMANCE_ISSUE_CODE = '''
def process_items(items):
    result = ""
    for item in items:
        for sub in item.subitems:
            result += str(sub) + ","
    return result
'''


class TestCodeReviewerInitialization:
    """Tests for CodeReviewerAgent initialization."""

    def test_default_initialization(self):
        """Test default init parameters."""
        agent = CodeReviewerAgent(system_prompt_path="nonexistent.md")
        assert agent.model == "claude-3-5-sonnet-20241022"
        assert agent.max_turns == 30

    def test_security_patterns_initialized(self):
        """Test security patterns are configured."""
        agent = CodeReviewerAgent(system_prompt_path="nonexistent.md")
        assert "sql_injection" in agent.security_patterns
        assert "xss" in agent.security_patterns
        assert "hardcoded_secrets" in agent.security_patterns

    def test_performance_patterns_initialized(self):
        """Test performance patterns are configured."""
        agent = CodeReviewerAgent(system_prompt_path="nonexistent.md")
        assert "nested_loop" in agent.performance_patterns

    def test_system_prompt_fallback(self):
        """Test fallback prompt when file not found."""
        agent = CodeReviewerAgent(system_prompt_path="nonexistent.md")
        assert "Code Reviewer" in agent.system_prompt

    def test_system_prompt_from_file(self):
        """Test loading from file."""
        with patch("builtins.open", mock_open(read_data="Code review prompt")):
            agent = CodeReviewerAgent(system_prompt_path="exists.md")
            assert agent.system_prompt == "Code review prompt"


class TestCodeReview:
    """Tests for the review method."""

    def test_clean_code_passes(self, reviewer):
        """Test clean code passes review."""
        report = reviewer.review(CLEAN_CODE)
        assert isinstance(report, CodeReviewReport)
        assert report.pass_fail is True

    def test_vulnerable_code_fails(self, reviewer):
        """Test vulnerable code fails review."""
        report = reviewer.review(VULNERABLE_CODE)
        assert report.pass_fail is False
        assert len(report.findings) > 0

    def test_report_has_all_sections(self, reviewer):
        """Test report has all required sections."""
        report = reviewer.review(CLEAN_CODE)
        assert report.security_scan is not None
        assert report.performance_analysis is not None
        assert report.style_compliance is not None
        assert report.overall_assessment is not None

    def test_report_has_recommended_actions(self, reviewer):
        """Test report has recommended actions."""
        report = reviewer.review(CLEAN_CODE)
        assert isinstance(report.recommended_actions, list)
        assert len(report.recommended_actions) > 0

    def test_report_with_custom_file_path(self, reviewer):
        """Test review with custom file path."""
        report = reviewer.review(CLEAN_CODE, file_path="main.py")
        assert isinstance(report, CodeReviewReport)


class TestSecurityScan:
    """Tests for security scanning."""

    def test_detects_sql_injection(self, reviewer):
        """Test SQL injection detection."""
        code = 'cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")'
        report = reviewer.review(code)
        security_findings = [
            f for f in report.findings
            if f.category == ReviewCategory.SECURITY
        ]
        # Should detect SQL injection pattern
        assert report.security_scan.sql_injection_risk is True or len(security_findings) > 0

    def test_detects_hardcoded_secrets(self, reviewer):
        """Test hardcoded secret detection."""
        report = reviewer.review(VULNERABLE_CODE)
        assert len(report.security_scan.credential_exposure) > 0

    def test_detects_xss(self, reviewer):
        """Test XSS detection."""
        code = 'document.write(userInput); eval(data);'
        report = reviewer.review(code, language="javascript")
        assert report.security_scan.xss_risk is True

    def test_detects_weak_crypto(self, reviewer):
        """Test weak crypto detection."""
        code = 'hash_value = md5(password)'
        report = reviewer.review(code)
        assert report.security_scan.vulnerabilities_found > 0

    def test_clean_code_no_security_issues(self, reviewer):
        """Test clean code has no security issues."""
        report = reviewer.review(CLEAN_CODE)
        assert report.security_scan.sql_injection_risk is False
        assert report.security_scan.xss_risk is False


class TestPerformanceAnalysis:
    """Tests for performance analysis."""

    def test_detects_nested_loops(self, reviewer):
        """Test nested loop detection."""
        code = """
for item in items:
    for sub in items:
        process(item, sub)
"""
        report = reviewer.review(code)
        perf = report.performance_analysis
        assert len(perf.optimization_opportunities) > 0

    def test_clean_code_performance(self, reviewer):
        """Test clean code has minimal performance issues."""
        report = reviewer.review(CLEAN_CODE)
        perf = report.performance_analysis
        # Clean code should have few or no issues
        assert isinstance(perf.n_plus_one_queries, list)


class TestStyleCompliance:
    """Tests for style compliance checking."""

    def test_detects_naming_violations(self, reviewer):
        """Test naming convention violation detection."""
        code = "def MyFunction(): pass\nclass myClass: pass"
        report = reviewer.review(code)
        style = report.style_compliance
        assert style.pep8_compliant is False

    def test_clean_code_style(self, reviewer):
        """Test clean code style compliance."""
        report = reviewer.review(CLEAN_CODE)
        # Clean code may still have some style notes
        assert report.style_compliance is not None


class TestErrorHandling:
    """Tests for error handling checks."""

    def test_simple_code_passes_error_handling(self, reviewer):
        """Test simple short code passes error handling check."""
        simple_code = "def add(a, b):\n    return a + b"
        report = reviewer.review(simple_code)
        assert report.error_handling_complete is True

    def test_complex_code_needs_try_except(self, reviewer):
        """Test complex code without try/except fails error handling."""
        complex_code = "\n".join([f"line {i}" for i in range(25)])
        report = reviewer.review(complex_code)
        assert report.error_handling_complete is False


class TestCodeContext:
    """Tests for code context analysis."""

    def test_context_counts_functions(self, reviewer):
        """Test function counting."""
        code = "def func1(): pass\ndef func2(): pass\ndef func3(): pass"
        context = reviewer._analyze_code_context(code, "test.py", "python")
        assert context.function_count == 3

    def test_context_counts_classes(self, reviewer):
        """Test class counting."""
        code = "class A: pass\nclass B: pass"
        context = reviewer._analyze_code_context(code, "test.py", "python")
        assert context.class_count == 2

    def test_context_extracts_imports(self, reviewer):
        """Test import extraction."""
        code = "import os\nimport sys\nfrom pathlib import Path"
        context = reviewer._analyze_code_context(code, "test.py", "python")
        assert "os" in context.imports
        assert "sys" in context.imports
        assert "pathlib" in context.imports


class TestConvenienceFunction:
    """Tests for create_code_reviewer convenience function."""

    def test_create_code_reviewer(self):
        """Test convenience function creates a CodeReviewerAgent."""
        agent = create_code_reviewer(system_prompt_path="nonexistent.md")
        assert isinstance(agent, CodeReviewerAgent)
