"""
Code Reviewer Subagent

Performs comprehensive code review across five dimensions:
security, performance, style, error handling, and test coverage.
"""

import re
import ast
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from src.schemas.code_reviewer import (
    CodeReviewReport,
    CodeFinding,
    SecurityScan,
    PerformanceAnalysis,
    StyleCompliance,
    SeverityLevel,
    ReviewCategory,
)


@dataclass
class CodeContext:
    """Context information about code being reviewed."""
    file_path: str
    language: str
    line_count: int
    function_count: int
    class_count: int
    imports: List[str]
    has_tests: bool = False


class CodeReviewerAgent:
    """
    The Code Reviewer performs comprehensive code review.

    Key responsibilities:
    - Security vulnerability scanning
    - Performance anti-pattern detection
    - Style and convention compliance
    - Error handling completeness
    - Test coverage assessment
    """

    def __init__(
        self,
        system_prompt_path: str = "config/agents/code_reviewer/CLAUDE.md",
        model: str = "claude-sonnet-4-20250514",
        max_turns: int = 30,
    ):
        """
        Initialize the Code Reviewer agent.

        Args:
            system_prompt_path: Path to system prompt file
            model: Model to use for review
            max_turns: Maximum conversation turns
        """
        self.system_prompt_path = system_prompt_path
        self.model = model
        self.max_turns = max_turns
        self.system_prompt = self._load_system_prompt()

        # Security patterns to detect
        self.security_patterns = {
            "sql_injection": [
                (r'execute\(.+?%s', "SQL injection risk: use parameterized queries"),
                (r'query\(.+?format\(', "SQL injection risk: use parameterized queries"),
                (r'f"SELECT.*?\{.*?\}', "SQL injection risk: use parameterized queries"),
            ],
            "xss": [
                (r'innerHTML\s*=', "XSS vulnerability: sanitize user input"),
                (r'document\.write\s*\(', "XSS vulnerability: avoid document.write with user input"),
                (r'eval\s*\(', "XSS vulnerability: avoid eval with user input"),
            ],
            "hardcoded_secrets": [
                (r'(password|secret|key|token)\s*=\s*["\'][^"\']+["\']', "Hardcoded credential detected"),
                (r'api_key\s*=\s*["\'][^"\']+["\']', "Hardcoded API key detected"),
            ],
            "weak_crypto": [
                (r'md5\(', "Weak hashing: use stronger algorithm like SHA-256"),
                (r'sha1\(', "Weak hashing: consider SHA-256 or better"),
            ],
        }

        # Performance patterns
        self.performance_patterns = {
            "n_plus_one": [
                (r'for\s+\w+\s+in\s+.*?:\s*.*\.get\(', "N+1 query: move inside loop or use select_related/prefetch"),
            ],
            "nested_loop": [
                (r'for\s+\w+\s+in\s+.*?:\s*for\s+\w+\s+in\s+', "Nested loop: consider algorithm optimization"),
            ],
            "string_concatenation": [
                (r'\w+\s*\+=\s*\w+\s*\+', "String concatenation in loop: use list join or f-string"),
            ],
        }

        # Style patterns (Python-focused)
        self.style_patterns = {
            "naming_convention": [
                (r'def\s+[A-Z]', "Function name should be lowercase_with_underscores"),
                (r'class\s+[a-z]', "Class name should be CapWords"),
            ],
            "line_length": [
                (r'.{100,}', "Line exceeds 100 characters"),
            ],
        }

    def review(
        self,
        code: str,
        file_path: str = "solution.py",
        language: str = "python",
        context: Optional[Dict[str, Any]] = None,
    ) -> CodeReviewReport:
        """
        Perform comprehensive code review.

        Args:
            code: The code to review
            file_path: Path to the file being reviewed
            language: Programming language
            context: Additional context (task, requirements, etc.)

        Returns:
            CodeReviewReport with all findings
        """
        # Analyze code context
        code_context = self._analyze_code_context(code, file_path, language)

        # Run five dimensions of review
        security_scan = self._security_scan(code, code_context)
        performance_analysis = self._performance_analysis(code, code_context)
        style_compliance = self._style_compliance(code, code_context)

        # Error handling and test coverage
        error_handling_complete = self._check_error_handling(code, code_context)
        test_coverage_assessment = self._assess_test_coverage(code, code_context)

        # Aggregate all findings
        findings = (
            self._security_findings_to_list(security_scan, code_context) +
            self._performance_findings_to_list(performance_analysis, code_context) +
            self._style_findings_to_list(style_compliance, code_context)
        )

        # Determine overall assessment
        overall_assessment = self._generate_overall_assessment(
            findings, code_context
        )

        # Determine pass/fail
        has_critical = any(
            f.severity == SeverityLevel.CRITICAL for f in findings
        )
        pass_fail = not has_critical

        # Generate recommended actions
        recommended_actions = self._generate_recommendations(findings)

        return CodeReviewReport(
            overall_assessment=overall_assessment,
            pass_fail=pass_fail,
            findings=findings,
            security_scan=security_scan,
            performance_analysis=performance_analysis,
            style_compliance=style_compliance,
            error_handling_complete=error_handling_complete,
            test_coverage_assessment=test_coverage_assessment,
            recommended_actions=recommended_actions,
        )

    # ========================================================================
    # Security Scan
    # ========================================================================

    def _security_scan(
        self,
        code: str,
        context: CodeContext
    ) -> SecurityScan:
        """Scan for security vulnerabilities."""
        vulnerabilities_found = 0
        sql_injection_risk = False
        xss_risk = False
        auth_issues = []
        credential_exposure = []

        code_lower = code.lower()

        # Check each security pattern
        for vuln_type, patterns in self.security_patterns.items():
            for pattern, message in patterns:
                matches = re.finditer(pattern, code, re.IGNORECASE)
                for match in matches:
                    vulnerabilities_found += 1
                    line_num = code[:match.start()].count('\n') + 1

                    if vuln_type == "sql_injection":
                        sql_injection_risk = True
                    elif vuln_type == "xss":
                        xss_risk = True
                    elif vuln_type == "hardcoded_secrets":
                        credential_exposure.append(f"Line {line_num}: {message}")

        # Check for authentication/authorization issues
        if "login" in code_lower or "auth" in code_lower:
            if "verify" not in code_lower and "check" not in code_lower:
                auth_issues.append("No password verification detected")
            if "hash" not in code_lower and "bcrypt" not in code_lower:
                auth_issues.append("Passwords may not be properly hashed")

        return SecurityScan(
            vulnerabilities_found=vulnerabilities_found,
            sql_injection_risk=sql_injection_risk,
            xss_risk=xss_risk,
            auth_issues=auth_issues,
            credential_exposure=credential_exposure,
        )

    def _security_findings_to_list(
        self,
        scan: SecurityScan,
        context: CodeContext
    ) -> List[CodeFinding]:
        """Convert security scan to CodeFinding list."""
        findings = []

        # SQL Injection findings
        if scan.sql_injection_risk:
            findings.append(CodeFinding(
                severity=SeverityLevel.CRITICAL,
                category=ReviewCategory.SECURITY,
                file_path=context.file_path,
                line_number=None,  # Would be populated from regex
                issue="SQL injection vulnerability detected",
                recommendation="Use parameterized queries or prepared statements",
            ))

        # XSS findings
        if scan.xss_risk:
            findings.append(CodeFinding(
                severity=SeverityLevel.CRITICAL,
                category=ReviewCategory.SECURITY,
                file_path=context.file_path,
                issue="XSS vulnerability detected",
                recommendation="Sanitize user input and use safe DOM manipulation",
            ))

        # Auth issues
        for issue in scan.auth_issues:
            findings.append(CodeFinding(
                severity=SeverityLevel.HIGH,
                category=ReviewCategory.SECURITY,
                file_path=context.file_path,
                issue=f"Authentication issue: {issue}",
                recommendation="Implement proper authentication and verification",
            ))

        # Credential exposure
        for exposure in scan.credential_exposure:
            findings.append(CodeFinding(
                severity=SeverityLevel.CRITICAL,
                category=ReviewCategory.SECURITY,
                file_path=context.file_path,
                issue=f"Credential exposure: {exposure}",
                recommendation="Use environment variables or secret management",
            ))

        return findings

    # ========================================================================
    # Performance Analysis
    # ========================================================================

    def _performance_analysis(
        self,
        code: str,
        context: CodeContext
    ) -> PerformanceAnalysis:
        """Analyze performance characteristics."""
        n_plus_one_queries = []
        missing_indexes = []
        memory_leaks = []
        optimization_opportunities = []

        # Check performance patterns
        for perf_type, patterns in self.performance_patterns.items():
            for pattern, message in patterns:
                matches = re.finditer(pattern, code)
                for match in matches:
                    line_num = code[:match.start()].count('\n') + 1

                    if perf_type == "n_plus_one":
                        n_plus_one_queries.append(f"Line {line_num}: {message}")
                    elif perf_type == "nested_loop":
                        optimization_opportunities.append(f"Line {line_num}: {message}")

        # Check for common Python performance issues
        if context.language == "python":
            # Check for global imports
            imports_in_functions = re.findall(r'^\s*import ', code, re.MULTILINE)
            if len(imports_in_functions) > 0:
                optimization_opportunities.append(
                    f"Move imports to module level for better performance"
                )

            # Check for lack of type hints (affects mypy performance)
            has_type_hints = bool(re.search(r':\s*(int|str|bool|list|dict|float)', code))
            if not has_type_hints and context.function_count > 0:
                optimization_opportunities.append(
                    "Add type hints for better IDE support and mypy performance"
                )

        return PerformanceAnalysis(
            n_plus_one_queries=n_plus_one_queries,
            missing_indexes=missing_indexes,
            memory_leaks=memory_leaks,
            optimization_opportunities=optimization_opportunities,
        )

    def _performance_findings_to_list(
        self,
        analysis: PerformanceAnalysis,
        context: CodeContext
    ) -> List[CodeFinding]:
        """Convert performance analysis to CodeFinding list."""
        findings = []

        for query in analysis.n_plus_one_queries:
            findings.append(CodeFinding(
                severity=SeverityLevel.HIGH,
                category=ReviewCategory.PERFORMANCE,
                file_path=context.file_path,
                issue=f"N+1 query pattern: {query}",
                recommendation="Use select_related/prefetch or restructure query",
            ))

        for opt in analysis.optimization_opportunities:
            # Determine severity based on impact
            severity = SeverityLevel.MEDIUM
            if "nested loop" in opt.lower() or "memory" in opt.lower():
                severity = SeverityLevel.HIGH

            findings.append(CodeFinding(
                severity=severity,
                category=ReviewCategory.PERFORMANCE,
                file_path=context.file_path,
                issue=opt,
                recommendation="Apply suggested optimization",
            ))

        return findings

    # ========================================================================
    # Style Compliance
    # ========================================================================

    def _style_compliance(
        self,
        code: str,
        context: CodeContext
    ) -> StyleCompliance:
        """Check style and convention compliance."""
        pep8_compliant = True
        naming_conventions = []
        complexity_issues = []
        consistency_issues = []

        # Check style patterns
        for style_type, patterns in self.style_patterns.items():
            for pattern, message in patterns:
                matches = re.finditer(pattern, code)
                for match in matches:
                    line_num = code[:match.start()].count('\n') + 1

                    if style_type == "naming_convention":
                        naming_conventions.append(f"Line {line_num}: {message}")
                        pep8_compliant = False
                    elif style_type == "line_length":
                        # Check if line is actually over limit
                        line_start = code.rfind('\n', 0, match.start()) + 1
                        line_end = code.find('\n', match.start())
                        if line_end == -1:
                            line_end = len(code)
                        actual_line = code[line_start:line_end]
                        if len(actual_line) > 100:
                            consistency_issues.append(f"Line {line_num}: {message}")

        # Check for docstrings
        if context.function_count > 0:
            function_defs = re.findall(r'def\s+(\w+)', code)
            for func_name in function_defs:
                # Simple check: does the function have a docstring?
                pattern = rf'def\s+{func_name}\s*\(.*?\):\s*("""|\'\'\'|[^\n])'
                if not re.search(pattern, code):
                    naming_conventions.append(
                        f"Function '{func_name}': missing docstring"
                    )
                    pep8_compliant = False

        return StyleCompliance(
            pep8_compliant=pep8_compliant,
            naming_conventions=naming_conventions,
            complexity_issues=complexity_issues,
            consistency_issues=consistency_issues,
        )

    def _style_findings_to_list(
        self,
        style: StyleCompliance,
        context: CodeContext
    ) -> List[CodeFinding]:
        """Convert style compliance to CodeFinding list."""
        findings = []

        for issue in style.naming_conventions:
            findings.append(CodeFinding(
                severity=SeverityLevel.LOW,
                category=ReviewCategory.STYLE,
                file_path=context.file_path,
                issue=issue,
                recommendation="Follow PEP 8 naming conventions",
            ))

        for issue in style.consistency_issues:
            findings.append(CodeFinding(
                severity=SeverityLevel.LOW,
                category=ReviewCategory.STYLE,
                file_path=context.file_path,
                issue=issue,
                recommendation="Break long lines or use continuation",
            ))

        return findings

    # ========================================================================
    # Error Handling & Test Coverage
    # ========================================================================

    def _check_error_handling(
        self,
        code: str,
        context: CodeContext
    ) -> bool:
        """Check if error handling is complete."""
        code_lower = code.lower()

        # Check for try-except blocks
        has_try_except = bool(re.search(r'\btry\s*:', code))

        # Check for error raising
        has_raise = bool(re.search(r'\braise\s+', code))

        # Check for logging
        has_logging = bool(re.search(r'log\.|logger\.|print\(', code))

        # Consider handling complete if there's error handling OR it's simple code
        if context.line_count < 20:
            return True  # Simple code may not need complex error handling

        return has_try_except or has_raise

    def _assess_test_coverage(
        self,
        code: str,
        context: CodeContext
    ) -> str:
        """Assess test coverage."""
        if context.has_tests:
            return "Tests found - verify coverage and edge cases"

        # No tests found
        if context.function_count > 0:
            return f"No tests found - {context.function_count} functions need testing"
        elif context.class_count > 0:
            return f"No tests found - {context.class_count} classes need testing"
        else:
            return "No tests found - add unit tests for verification"

    # ========================================================================
    # Analysis & Report Generation
    # ========================================================================

    def _analyze_code_context(
        self,
        code: str,
        file_path: str,
        language: str
    ) -> CodeContext:
        """Analyze the code to extract context."""
        # Count lines
        line_count = len(code.split('\n'))

        # Count functions and classes (Python-specific)
        function_count = 0
        class_count = 0
        imports = []

        if language == "python":
            function_count = len(re.findall(r'\bdef\s+\w+', code))
            class_count = len(re.findall(r'\bclass\s+\w+', code))
            imports = re.findall(r'^\s*import\s+(\w+)', code, re.MULTILINE)
            imports.extend(re.findall(r'^\s*from\s+(\w+)', code, re.MULTILINE))

        return CodeContext(
            file_path=file_path,
            language=language,
            line_count=line_count,
            function_count=function_count,
            class_count=class_count,
            imports=imports,
        )

    def _generate_overall_assessment(
        self,
        findings: List[CodeFinding],
        context: CodeContext
    ) -> str:
        """Generate overall assessment."""
        critical_count = sum(1 for f in findings if f.severity == SeverityLevel.CRITICAL)
        high_count = sum(1 for f in findings if f.severity == SeverityLevel.HIGH)
        medium_count = sum(1 for f in findings if f.severity == SeverityLevel.MEDIUM)

        if critical_count > 0:
            return f"Code has {critical_count} critical issue(s) requiring immediate attention"
        elif high_count > 2:
            return f"Code has {high_count} high-priority issues that should be addressed"
        elif medium_count > 5:
            return f"Code has {medium_count} medium-priority style issues"
        else:
            return "Code is in good shape with minor improvements suggested"

    def _generate_recommendations(
        self,
        findings: List[CodeFinding]
    ) -> List[str]:
        """Generate prioritized recommended actions."""
        recommendations = []

        # Group by severity
        critical = [f for f in findings if f.severity == SeverityLevel.CRITICAL]
        high = [f for f in findings if f.severity == SeverityLevel.HIGH]
        medium_low = [f for f in findings if f.severity in [SeverityLevel.MEDIUM, SeverityLevel.LOW]]

        if critical:
            recommendations.append(
                f"URGENT: Address {len(critical)} critical security/stability issue(s)"
            )
        if high:
            recommendations.append(
                f"IMPORTANT: Fix {len(high)} high-priority issue(s)"
            )
        if medium_low:
            recommendations.append(
                f"RECOMMENDED: Address {len(medium_low)} style/improvement issue(s)"
            )

        if not recommendations:
            recommendations.append("No issues found - code is ready to merge")

        return recommendations

    def _load_system_prompt(self) -> str:
        """Load the system prompt from file."""
        try:
            with open(self.system_prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            return "You are the Code Reviewer. Review code for security, performance, style, error handling, and tests."


# =============================================================================
# Convenience Functions
# =============================================================================

def create_code_reviewer(
    system_prompt_path: str = "config/agents/code_reviewer/CLAUDE.md",
    model: str = "claude-sonnet-4-20250514",
) -> CodeReviewerAgent:
    """Create a configured Code Reviewer agent."""
    return CodeReviewerAgent(
        system_prompt_path=system_prompt_path,
        model=model,
    )
