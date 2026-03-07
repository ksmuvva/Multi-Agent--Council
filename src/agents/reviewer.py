"""
Reviewer Subagent (Final Quality Gate)

Serves as the final quality gate before output reaches the user.
Implements the verdict matrix logic and coordinates with Quality Arbiter.
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from src.schemas.reviewer import (
    ReviewVerdict,
    Verdict,
    CheckItem,
    Revision,
    QualityGateResults,
    ArbitrationInput,
)


@dataclass
class ReviewContext:
    """Context information for the review."""
    original_request: str
    agent_outputs: Dict[str, Any]
    revision_count: int
    max_revisions: int
    tier_level: int
    is_code_output: bool


class ReviewerAgent:
    """
    The Reviewer is the final quality gate.

    Key responsibilities:
    - Check completeness (addresses original request?)
    - Check consistency (all agent contributions consistent?)
    - Verify Verifier sign-off
    - Check Critic findings addressed
    - Check readability
    - Implement verdict matrix logic
    - Coordinate with Quality Arbiter on Tier 4
    """

    def __init__(
        self,
        system_prompt_path: str = "config/agents/reviewer/CLAUDE.md",
        model: str = "claude-3-5-opus-20240507",
        max_turns: int = 30,
    ):
        """
        Initialize the Reviewer agent.

        Args:
            system_prompt_path: Path to system prompt file
            model: Model to use for review (opus for accuracy)
            max_turns: Maximum conversation turns
        """
        self.system_prompt_path = system_prompt_path
        self.model = model
        self.max_turns = max_turns
        self.system_prompt = self._load_system_prompt()

        # Verdict matrix: (verifier_verdict, critic_verdict) -> Verdict
        self.verdict_matrix = {
            (Verdict.PASS, Verdict.PASS): Verdict.PASS,
            (Verdict.PASS, Verdict.FAIL): Verdict.PASS_WITH_CAVEATS,
            (Verdict.FAIL, Verdict.PASS): Verdict.REVISE,
            (Verdict.FAIL, Verdict.FAIL): Verdict.REJECT,
        }

        # Critical failure patterns that force FAIL
        self.critical_failure_patterns = {
            "security": [
                r"sql injection",
                r"xss vulnerability",
                r"hardcoded.*password",
                r"hardcoded.*secret",
                r"hardcoded.*key",
                r"api_key\s*=\s*[\"']",
            ],
            "hallucination": [
                r"\bfabricated\b",
                r"\bhallucination\b",
                r"\bmade up\b",
                r"\binvented\b",
                r"no source provided",
            ],
            "logic": [
                r"\bcontradiction\b",
                r"\binconsistent\b",
                r"\blogical fallacy\b",
                r"\binvalid argument\b",
            ],
        }

    def review(
        self,
        output: str,
        context: ReviewContext,
        verifier_report: Optional[Dict[str, Any]] = None,
        critic_report: Optional[Dict[str, Any]] = None,
        code_review_report: Optional[Dict[str, Any]] = None,
        quality_standard: Optional[Dict[str, Any]] = None,
    ) -> ReviewVerdict:
        """
        Perform final quality review on the output.

        Args:
            output: The content to review
            context: Review context with request and agent outputs
            verifier_report: Optional Verifier report
            critic_report: Optional Critic report
            code_review_report: Optional Code Reviewer report
            quality_standard: Optional QualityStandard from Council (Tier 3-4)

        Returns:
            ReviewVerdict with pass/fail decision and revision instructions
        """
        # Step 1: Run all quality gate checks
        quality_gates = self._run_quality_gates(
            output, context, verifier_report, critic_report, code_review_report
        )

        # Step 2: Check for critical failures
        critical_failures = self._check_critical_failures(
            output, verifier_report, critic_report, code_review_report
        )

        # Step 3: Determine verdict
        verdict = self._determine_verdict(
            quality_gates, critical_failures, context
        )

        # Step 4: Get verdicts from Verifier and Critic
        verifier_verdict = self._extract_verifier_verdict(verifier_report)
        critic_verdict = self._extract_critic_verdict(critic_report)

        # Step 5: Apply verdict matrix
        matrix_action = self._apply_verdict_matrix(verifier_verdict, critic_verdict)

        # Step 6: Generate reasons
        reasons = self._generate_reasons(
            verdict, quality_gates, critical_failures, matrix_action
        )

        # Step 7: Generate revision instructions for non-PASS verdicts
        revision_instructions = []
        if verdict in (Verdict.FAIL, Verdict.REVISE, Verdict.REJECT,
                       Verdict.PASS_WITH_CAVEATS, Verdict.ESCALATE):
            revision_instructions = self._generate_revision_instructions(
                quality_gates, critical_failures, matrix_action, context
            )

        # Step 8: Check if arbitration needed (Tier 4 disagreement)
        arbitration_needed, arbitration_input = self._check_arbitration_needed(
            verdict, verifier_verdict, critic_verdict, context
        )

        # Step 9: Generate summary
        summary = self._generate_summary(verdict, reasons, context)

        # Step 10: Determine if can revise
        can_revise = context.revision_count < context.max_revisions

        return ReviewVerdict(
            verdict=verdict,
            confidence=self._calculate_confidence(quality_gates, critical_failures),
            quality_gate_results=quality_gates,
            reasons=reasons,
            revision_instructions=revision_instructions,
            revision_count=context.revision_count,
            max_revisions=context.max_revisions,
            can_revise=can_revise,
            arbitration_needed=arbitration_needed,
            arbitration_input=arbitration_input,
            summary=summary,
            tier_4_arbiter_involved=(arbitration_input is not None),
        )

    # ========================================================================
    # Quality Gate Checks
    # ========================================================================

    def _run_quality_gates(
        self,
        output: str,
        context: ReviewContext,
        verifier_report: Optional[Dict[str, Any]],
        critic_report: Optional[Dict[str, Any]],
        code_review_report: Optional[Dict[str, Any]],
    ) -> QualityGateResults:
        """Run all quality gate checks."""
        return QualityGateResults(
            completeness=self._check_completeness(output, context),
            consistency=self._check_consistency(output, context),
            verifier_signoff=self._check_verifier_signoff(verifier_report),
            critic_findings_addressed=self._check_critic_findings(critic_report),
            readability=self._check_readability(output),
            code_review_passed=self._check_code_review(code_review_report)
            if context.is_code_output
            else None,
        )

    def _check_completeness(self, output: str, context: ReviewContext) -> CheckItem:
        """Check if output addresses the original request."""
        request_lower = context.original_request.lower()
        output_lower = output.lower()

        # Extract key requirements from request
        requirements = self._extract_requirements(context.original_request)

        # Check if each requirement is addressed
        missing_requirements = []
        for req in requirements:
            # Check if requirement keyword appears in output
            if req.lower() not in output_lower:
                # Check for synonyms or related terms
                if not self._is_requirement_addressed(req, output_lower):
                    missing_requirements.append(req)

        passed = len(missing_requirements) == 0
        severity = "critical" if missing_requirements else "high"

        if passed:
            notes = "All requirements addressed in output"
        else:
            notes = f"Missing requirements: {', '.join(missing_requirements[:3])}"

        return CheckItem(
            check_name="Completeness",
            passed=passed,
            notes=notes,
            severity_if_failed=severity,
        )

    def _extract_requirements(self, request: str) -> List[str]:
        """Extract key requirements from the request."""
        requirements = []

        # Action verbs indicating requirements
        action_patterns = [
            r"(?:create|write|build|generate|implement|design|develop)\s+(\w+(?:\s+\w+)?)",
            r"(?:include|add|ensure|provide)\s+(\w+(?:\s+\w+)?)",
            r"(?:must|should|shall)\s+(\w+(?:\s+\w+)?)",
        ]

        for pattern in action_patterns:
            matches = re.finditer(pattern, request, re.IGNORECASE)
            for match in matches:
                requirements.append(match.group(1))

        # Numbers and quantities
        number_matches = re.findall(r"\b\d+\b", request)
        requirements.extend([f"{n} items" for n in number_matches])

        # Specific terms in quotes
        quoted_terms = re.findall(r'"([^"]+)"', request)
        requirements.extend(quoted_terms)

        return list(set(requirements))  # Deduplicate

    def _is_requirement_addressed(self, requirement: str, output_lower: str) -> bool:
        """Check if a requirement is addressed in output using synonym matching."""
        # Check for direct match
        if requirement.lower() in output_lower:
            return True

        # Check for related terms
        synonym_map = {
            "test": ["testing", "tests", "test case", "unit test"],
            "documentation": ["docs", "documented", "readme", "comments"],
            "error": ["exception", "handling", "validation"],
            "security": ["secure", "authentication", "authorization"],
            "performance": ["optimize", "efficient", "fast"],
        }

        req_lower = requirement.lower()
        for key, synonyms in synonym_map.items():
            if key in req_lower:
                if any(syn in output_lower for syn in synonyms):
                    return True

        return False

    def _check_consistency(self, output: str, context: ReviewContext) -> CheckItem:
        """Check if all agent contributions are consistent."""
        inconsistencies = []

        # Check for contradictory statements
        contradiction_patterns = [
            (r"both (\w+) and (?:not|no) \1", "Self-contradiction"),
            (r"however,\s*\w+\s+is\s+(?:not|no)", "Possible contradiction"),
            (r"although\s+\w+\s+is\s+\w+,\s*\w+\s+is\s+not", "Contradiction"),
        ]

        for pattern, desc in contradiction_patterns:
            if re.search(pattern, output, re.IGNORECASE):
                inconsistencies.append(desc)

        # Check agent output consistency
        if context.agent_outputs:
            # Check if Analyst's modality matches output
            if "analyst" in context.agent_outputs:
                analyst_modality = context.agent_outputs["analyst"].get("modality", "")
                if analyst_modality == "code" and not self._looks_like_code(output):
                    inconsistencies.append("Analyst predicted code, output is not code")
                elif analyst_modality == "image" and "```" not in output:
                    inconsistencies.append("Analyst predicted image, output lacks diagram")

        passed = len(inconsistencies) == 0
        notes = (
            "All contributions are consistent"
            if passed
            else f"Inconsistencies: {', '.join(inconsistencies[:3])}"
        )

        return CheckItem(
            check_name="Consistency",
            passed=passed,
            notes=notes,
            severity_if_failed="medium",
        )

    def _looks_like_code(self, output: str) -> bool:
        """Check if output looks like code."""
        code_indicators = ["```", "def ", "function ", "class ", "import ", "return "]
        return any(indicator in output for indicator in code_indicators)

    def _check_verifier_signoff(
        self, verifier_report: Optional[Dict[str, Any]]
    ) -> CheckItem:
        """Check if Verifier passed."""
        if verifier_report is None:
            return CheckItem(
                check_name="Verifier Sign-off",
                passed=True,  # No verifier = assume pass for lower tiers
                notes="No verifier report (lower tier)",
                severity_if_failed="critical",
            )

        verdict = verifier_report.get("verdict", "PASS")
        reliability = verifier_report.get("overall_reliability", 0.7)

        passed = verdict == "PASS" and reliability >= 0.7
        notes = (
            f"Verifier {'passed' if passed else 'failed'} "
            f"(reliability: {reliability:.2f})"
        )

        return CheckItem(
            check_name="Verifier Sign-off",
            passed=passed,
            notes=notes,
            severity_if_failed="critical",
        )

    def _check_critic_findings(
        self, critic_report: Optional[Dict[str, Any]]
    ) -> CheckItem:
        """Check if Critic findings are addressed."""
        if critic_report is None:
            return CheckItem(
                check_name="Critic Findings",
                passed=True,  # No critic = assume pass for lower tiers
                notes="No critic report (lower tier)",
                severity_if_failed="high",
            )

        overall_assessment = critic_report.get("overall_assessment", "")

        # Check if critic flagged critical issues
        has_critical = "critical" in overall_assessment.lower()

        # Check number of attacks that failed
        attacks = critic_report.get("attacks", [])
        failed_attacks = [a for a in attacks if a.get("verdict") == "FAIL"]

        # Most critic findings should be addressed or mitigated
        passed = not has_critical and len(failed_attacks) <= 1

        notes = (
            f"Critic findings {'addressed' if passed else 'need attention'} "
            f"({len(failed_attacks)} failed attacks)"
        )

        return CheckItem(
            check_name="Critic Findings Addressed",
            passed=passed,
            notes=notes,
            severity_if_failed="high",
        )

    def _check_readability(self, output: str) -> CheckItem:
        """Check if output is clear and well-structured."""
        issues = []

        # Check sentence length
        sentences = re.split(r"[.!?]+", output)
        long_sentences = [
            s for s in sentences if len(s.split()) > 30
        ]  # More than 30 words
        if len(long_sentences) > len(sentences) * 0.2:  # More than 20%
            issues.append("Many sentences are too long")

        # Check for structure
        has_structure = any(
            pattern in output
            for pattern in ["## ", "1.", "```", "- ", "| ", "=== ", "# "]
        )
        if not has_structure and len(output) > 500:
            issues.append("Output lacks structure for long content")

        # Check for very short output (may be incomplete)
        if len(output.strip()) < 50:
            issues.append("Output is very short")

        passed = len(issues) == 0
        notes = (
            "Clear and well-structured"
            if passed
            else f"Readability issues: {', '.join(issues)}"
        )

        return CheckItem(
            check_name="Readability",
            passed=passed,
            notes=notes,
            severity_if_failed="low",
        )

    def _check_code_review(
        self, code_review_report: Optional[Dict[str, Any]]
    ) -> Optional[CheckItem]:
        """Check code review if applicable."""
        if code_review_report is None:
            return None

        pass_fail = code_review_report.get("pass_fail", True)
        findings = code_review_report.get("findings", [])

        # Check for critical security issues
        critical_findings = [
            f for f in findings if f.get("severity") == "CRITICAL"
        ]

        passed = pass_fail and len(critical_findings) == 0

        notes = (
            f"Code review {'passed' if passed else 'failed'} "
            f"({len(critical_findings)} critical issues)"
        )

        return CheckItem(
            check_name="Code Review",
            passed=passed,
            notes=notes,
            severity_if_failed="critical",
        )

    # ========================================================================
    # Critical Failure Detection
    # ========================================================================

    def _check_critical_failures(
        self,
        output: str,
        verifier_report: Optional[Dict[str, Any]],
        critic_report: Optional[Dict[str, Any]],
        code_review_report: Optional[Dict[str, Any]],
    ) -> Dict[str, List[str]]:
        """Check for critical failure patterns."""
        failures = {
            "security": [],
            "hallucination": [],
            "logic": [],
        }

        output_lower = output.lower()

        # Check each category
        for category, patterns in self.critical_failure_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, output_lower)
                for match in matches:
                    failures[category].append(match.group())

        # Check reports for additional failures
        if verifier_report:
            fabricated = verifier_report.get("fabricated_claims", 0)
            if fabricated > 0:
                failures["hallucination"].append(
                    f"{fabricated} potentially fabricated claims"
                )

        if code_review_report:
            findings = code_review_report.get("findings", [])
            for f in findings:
                if f.get("category") == "SECURITY" and f.get("severity") == "CRITICAL":
                    failures["security"].append(f.get("issue", ""))

        return failures

    # ========================================================================
    # Verdict Determination
    # ========================================================================

    def _determine_verdict(
        self,
        quality_gates: QualityGateResults,
        critical_failures: Dict[str, List[str]],
        context: ReviewContext,
    ) -> Verdict:
        """Determine the final verdict using the 5-verdict system (FR-031/FR-011)."""
        verifier_passed = quality_gates.verifier_signoff.passed
        critic_passed = quality_gates.critic_findings_addressed.passed

        # REJECT on critical security or hallucination issues
        if critical_failures.get("security") or critical_failures.get("hallucination"):
            return Verdict.REJECT

        # REJECT if code review failed with critical issues (for code output)
        if context.is_code_output and quality_gates.code_review_passed:
            if not quality_gates.code_review_passed.passed:
                return Verdict.REJECT

        # ESCALATE if tier 4 and there is disagreement between verifier and critic
        if context.tier_level >= 4 and verifier_passed != critic_passed:
            return Verdict.ESCALATE

        # Verifier failed but critic passed -> REVISE (fixable issues)
        if not verifier_passed and critic_passed:
            # If over max revisions, downgrade REVISE to PASS_WITH_CAVEATS
            if context.revision_count >= context.max_revisions:
                return Verdict.PASS_WITH_CAVEATS
            return Verdict.REVISE

        # Verifier passed but critic found issues -> PASS_WITH_CAVEATS
        if verifier_passed and not critic_passed:
            return Verdict.PASS_WITH_CAVEATS

        # Both failed -> REJECT
        if not verifier_passed and not critic_passed:
            return Verdict.REJECT

        # Both passed - check remaining gates
        if not quality_gates.completeness.passed:
            # If over max revisions, downgrade to PASS_WITH_CAVEATS
            if context.revision_count >= context.max_revisions:
                return Verdict.PASS_WITH_CAVEATS
            return Verdict.REVISE

        # Overall gate check
        gate_scores = [
            quality_gates.completeness.passed,
            quality_gates.consistency.passed,
            quality_gates.verifier_signoff.passed,
            quality_gates.critic_findings_addressed.passed,
            quality_gates.readability.passed,
        ]

        if context.is_code_output and quality_gates.code_review_passed:
            gate_scores.append(quality_gates.code_review_passed.passed)

        pass_ratio = sum(gate_scores) / len(gate_scores)

        if pass_ratio >= 0.8:
            return Verdict.PASS
        elif pass_ratio >= 0.6:
            return Verdict.PASS_WITH_CAVEATS
        else:
            # If over max revisions, accept with caveats
            if context.revision_count >= context.max_revisions:
                return Verdict.PASS_WITH_CAVEATS
            return Verdict.REVISE

    def _extract_verifier_verdict(
        self, verifier_report: Optional[Dict[str, Any]]
    ) -> Verdict:
        """Extract verdict from Verifier report."""
        if verifier_report is None:
            return Verdict.PASS

        verdict_str = verifier_report.get("verdict", "PASS")
        return Verdict.PASS if verdict_str == "PASS" else Verdict.FAIL

    def _extract_critic_verdict(
        self, critic_report: Optional[Dict[str, Any]]
    ) -> Verdict:
        """Extract verdict from Critic report."""
        if critic_report is None:
            return Verdict.PASS

        overall = critic_report.get("overall_assessment", "").lower()

        # Critic is harsh - FAIL only if critical issues
        if "critical" in overall:
            return Verdict.FAIL
        return Verdict.PASS

    def _apply_verdict_matrix(
        self, verifier_verdict: Verdict, critic_verdict: Verdict
    ) -> Verdict:
        """Apply the verdict matrix to determine action."""
        return self.verdict_matrix.get(
            (verifier_verdict, critic_verdict), Verdict.REVISE
        )

    # ========================================================================
    # Report Generation
    # ========================================================================

    def _generate_reasons(
        self,
        verdict: Verdict,
        quality_gates: QualityGateResults,
        critical_failures: Dict[str, List[str]],
        matrix_action: Verdict,
    ) -> List[str]:
        """Generate reasons supporting the verdict."""
        reasons = []

        all_gates = [
            ("Completeness", quality_gates.completeness),
            ("Consistency", quality_gates.consistency),
            ("Verifier", quality_gates.verifier_signoff),
            ("Critic", quality_gates.critic_findings_addressed),
            ("Readability", quality_gates.readability),
        ]

        passed_gates = [name for name, gate in all_gates if gate.passed]
        failed_gates = [
            f"{gate.check_name}: {gate.notes}"
            for _, gate in all_gates
            if not gate.passed
        ]

        if verdict == Verdict.PASS:
            reasons.append("All quality gates passed")
            reasons.append(f"Passed: {', '.join(passed_gates)}")
            if matrix_action == Verdict.PASS:
                reasons.append("Verdict matrix: Proceed to formatting")

        elif verdict == Verdict.PASS_WITH_CAVEATS:
            reasons.append("Output accepted with caveats")
            if passed_gates:
                reasons.append(f"Passed: {', '.join(passed_gates)}")
            if failed_gates:
                reasons.extend(failed_gates)
            reasons.append(f"Verdict matrix action: {matrix_action.value}")

        elif verdict == Verdict.REVISE:
            reasons.append("Output requires revision")
            reasons.extend(failed_gates)
            reasons.append(f"Verdict matrix action: {matrix_action.value}")

        elif verdict == Verdict.REJECT:
            reasons.append("Output rejected due to critical issues")
            reasons.extend(failed_gates)
            for category, failures in critical_failures.items():
                if failures:
                    reasons.append(
                        f"{category.capitalize()} issues: {', '.join(failures[:2])}"
                    )
            reasons.append(f"Verdict matrix action: {matrix_action.value}")

        elif verdict == Verdict.ESCALATE:
            reasons.append("Escalating to Quality Arbiter due to Tier 4 disagreement")
            reasons.extend(failed_gates)
            reasons.append(f"Verdict matrix action: {matrix_action.value}")

        else:
            # Backward compat for FAIL
            reasons.extend(failed_gates)
            for category, failures in critical_failures.items():
                if failures:
                    reasons.append(
                        f"{category.capitalize()} issues: {', '.join(failures[:2])}"
                    )
            reasons.append(f"Verdict matrix action: {matrix_action.value}")

        return reasons

    def _generate_revision_instructions(
        self,
        quality_gates: QualityGateResults,
        critical_failures: Dict[str, List[str]],
        matrix_action: Verdict,
        context: ReviewContext,
    ) -> List[Revision]:
        """Generate specific revision instructions based on verdict type."""
        revisions = []

        # Map matrix verdict to revision category
        action_categories = {
            Verdict.PASS_WITH_CAVEATS: "Content Revision",
            Verdict.REVISE: "Verification",
            Verdict.REJECT: "Complete Regeneration",
        }

        category = action_categories.get(matrix_action, "Quality Improvement")

        # Generate revisions for failed gates
        for gate in [
            quality_gates.completeness,
            quality_gates.consistency,
            quality_gates.critic_findings_addressed,
        ]:
            if not gate.passed:
                priority = (
                    "critical"
                    if gate.severity_if_failed == "critical"
                    else "high"
                    if gate.severity_if_failed == "high"
                    else "medium"
                )

                revisions.append(
                    Revision(
                        category=category,
                        description=f"Address {gate.check_name} issue",
                        reason=gate.notes,
                        priority=priority,
                        specific_instructions=f"Fix the following: {gate.notes}",
                    )
                )

        # Add critical failure revisions
        for fail_category, failures in critical_failures.items():
            for failure in failures[:3]:  # Limit to 3 per category
                revisions.append(
                    Revision(
                        category="Critical Issue",
                        description=f"{fail_category.capitalize()} problem",
                        reason=f"Detected: {failure}",
                        priority="critical",
                        specific_instructions=f"Remove or fix: {failure}",
                    )
                )

        # Add verdict-specific instructions
        if matrix_action == Verdict.PASS_WITH_CAVEATS:
            revisions.append(
                Revision(
                    category="Content Revision",
                    description="Revise output based on Critic findings",
                    reason="Critic identified issues but Verifier passed",
                    priority="high",
                    specific_instructions="Address all Critic-identified problems",
                )
            )
        elif matrix_action == Verdict.REVISE:
            revisions.append(
                Revision(
                    category="Verification",
                    description="Re-verify claims and sources",
                    reason="Verifier found issues but Critic passed",
                    priority="high",
                    specific_instructions="Verify all factual claims with sources",
                )
            )
        elif matrix_action == Verdict.REJECT:
            revisions.append(
                Revision(
                    category="Complete Regeneration",
                    description="Start over with full regeneration",
                    reason="Both Verifier and Critic failed",
                    priority="critical",
                    specific_instructions="Regenerate from Planner phase",
                )
            )
        elif matrix_action == Verdict.ESCALATE:
            revisions.append(
                Revision(
                    category="Escalation",
                    description="Escalate to Quality Arbiter for resolution",
                    reason="Tier 4 disagreement between Verifier and Critic",
                    priority="critical",
                    specific_instructions="Quality Arbiter must arbitrate before proceeding",
                )
            )

        return revisions[:5]  # Limit to 5 revisions

    def _check_arbitration_needed(
        self,
        verdict: Verdict,
        verifier_verdict: Verdict,
        critic_verdict: Verdict,
        context: ReviewContext,
    ) -> tuple[bool, Optional[ArbitrationInput]]:
        """Check if Quality Arbiter arbitration is needed (Tier 4)."""
        # Only Tier 4 gets arbitration
        if context.tier_level < 4:
            return False, None

        # Arbitration needed if there's disagreement
        if verifier_verdict != critic_verdict:
            return True, ArbitrationInput(
                reviewer_verdict=verdict,
                verifier_verdict=verifier_verdict,
                critic_verdict=critic_verdict,
                disagreement_reason=f"Verifier said {verifier_verdict.value}, Critic said {critic_verdict.value}",
                debate_rounds_completed=0,
            )

        return False, None

    def _calculate_confidence(
        self,
        quality_gates: QualityGateResults,
        critical_failures: Dict[str, List[str]],
    ) -> float:
        """Calculate confidence in the verdict."""
        # Start with base confidence
        confidence = 0.5

        # Adjust for passed gates
        gates = [
            quality_gates.completeness,
            quality_gates.consistency,
            quality_gates.verifier_signoff,
            quality_gates.critic_findings_addressed,
            quality_gates.readability,
        ]

        passed_ratio = sum(1 for g in gates if g.passed) / len(gates)
        confidence += passed_ratio * 0.4

        # Reduce for critical failures
        for failures in critical_failures.values():
            if failures:
                confidence -= 0.1 * len(failures)

        # Clamp to 0-1
        return max(0.0, min(1.0, confidence))

    def _generate_summary(self, verdict: Verdict, reasons: List[str], context: ReviewContext) -> str:
        """Generate human-readable summary."""
        if verdict == Verdict.PASS:
            summary = "Output passes all quality gates"
            if context.revision_count > 0:
                summary += f" after {context.revision_count} revision(s)"
            summary += ". Ready for formatting."
        elif verdict == Verdict.PASS_WITH_CAVEATS:
            summary = "Output accepted with caveats"
            if context.revision_count > 0:
                summary += f" after {context.revision_count} revision(s)"
            summary += ". Minor issues noted but output is usable."
        elif verdict == Verdict.REVISE:
            summary = "Output requires revision"
            if context.revision_count >= context.max_revisions:
                summary += " (max revisions reached - accepting best effort)"
            else:
                summary += f" ({context.revision_count + 1}/{context.max_revisions})"
        elif verdict == Verdict.REJECT:
            summary = "Output rejected due to critical issues. Full regeneration required."
        elif verdict == Verdict.ESCALATE:
            summary = "Escalating to Quality Arbiter for Tier 4 arbitration."
        else:
            # Backward compat for FAIL
            summary = "Output requires revision"
            if context.revision_count >= context.max_revisions:
                summary += " (max revisions reached - accepting best effort)"
            else:
                summary += f" ({context.revision_count + 1}/{context.max_revisions})"

        if reasons:
            primary_reason = reasons[0]
            summary += f" Primary reason: {primary_reason}"

        return summary

    def _load_system_prompt(self) -> str:
        """Load the system prompt from file."""
        try:
            with open(self.system_prompt_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return "You are the Reviewer. Final quality gate before output reaches the user."


# =============================================================================
# Convenience Functions
# =============================================================================

def create_reviewer(
    system_prompt_path: str = "config/agents/reviewer/CLAUDE.md",
    model: str = "claude-3-5-opus-20240507",
) -> ReviewerAgent:
    """Create a configured Reviewer agent."""
    return ReviewerAgent(
        system_prompt_path=system_prompt_path,
        model=model,
    )
