"""
Critic Subagent (Adversarial Critic)

Attacks proposed solutions through five vectors:
logic, completeness, quality, contradiction, and red-team.
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from src.schemas.critic import (
    CritiqueReport,
    Attack,
    AttackVector,
    LogicAttack,
    CompletenessAttack,
    QualityAttack,
    ContradictionScan,
    RedTeamArgument,
    SeverityLevel,
)


@dataclass
class ArgumentAnalysis:
    """Analysis of an argument or solution."""
    premises: List[str]
    conclusion: str
    logical_structure: str
    completeness_score: float
    quality_score: float
    assumptions: List[str]


class CriticAgent:
    """
    The Critic applies adversarial thinking to find weaknesses.

    Key responsibilities:
    - Logic attack: Check argument validity
    - Completeness attack: Find missing elements
    - Quality attack: Assess quality level
    - Contradiction scan: Find inconsistencies
    - Red-team argumentation: Think like an adversary
    """

    def __init__(
        self,
        system_prompt_path: str = "config/agents/critic/CLAUDE.md",
        model: str = "claude-3-5-opus-20240507",
        max_turns: int = 30,
    ):
        """
        Initialize the Critic agent.

        Args:
            system_prompt_path: Path to system prompt file
            model: Model to use for critique (opus for thoroughness)
            max_turns: Maximum conversation turns
        """
        self.system_prompt_path = system_prompt_path
        self.model = model
        self.max_turns = max_turns
        self.system_prompt = self._load_system_prompt()

        # Logic fallacy patterns
        self.fallacy_patterns = {
            "ad_hominem": r"^(?:He|She|They)\s+(?:is|are)\s+(?:a |an |)",
            "straw_man": r"^(?:This is|It's |It is)",
            "slippery_slope": r"will.*lead to.*inevitably",
            "appeal_to_authority": r"because.*?(?:expert|authority|doctor)",
            "circular_reasoning": r"therefore.*because.*?therefore",
        }

    def critique(
        self,
        solution: str,
        original_request: str,
        domain_attacks: Optional[List[str]] = None,
        sme_inputs: Optional[Dict[str, str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> CritiqueReport:
        """
        Critique a proposed solution through five attack vectors.

        Args:
            solution: The proposed solution to critique
            original_request: The original user request
            domain_attacks: Optional domain-specific attacks from SMEs
            sme_inputs: Optional inputs from SME personas
            context: Additional critique context

        Returns:
            CritiqueReport with all attacks and findings
        """
        # Analyze the argument structure
        argument_analysis = self._analyze_argument_structure(solution)

        # Execute five attack vectors
        logic_attack = self._logic_attack(solution, argument_analysis)
        completeness_attack = self._completeness_attack(solution, original_request)
        quality_attack = self._quality_attack(solution, argument_analysis)
        contradiction_scan = self._contradiction_scan(solution)
        red_team_argument = self._red_team_argumentation(solution, original_request)

        # Combine all attacks
        attacks = (
            self._logic_attack_to_list(logic_attack) +
            self._completeness_attack_to_list(completeness_attack) +
            self._quality_attack_to_list(quality_attack) +
            self._contradiction_to_list(contradiction_scan) +
            self._red_team_to_list(red_team_argument)
        )

        # Add domain-specific attacks if provided
        if domain_attacks:
            domain_attacks_list = self._domain_attacks_to_list(
                domain_attacks, sme_inputs
            )
            attacks.extend(domain_attacks_list)

        # Determine overall assessment
        overall_assessment = self._generate_overall_assessment(
            attacks, solution, argument_analysis
        )

        # Identify critical issues
        critical_issues = self._identify_critical_issues(attacks)

        # Generate recommended revisions
        recommended_revisions = self._generate_revisions(attacks, critical_issues)

        # Determine if the solution passes critique
        would_approve = self._would_approve_solution(
            attacks, critical_issues
        )

        return CritiqueReport(
            solution_summary=solution[:100] + "..." if len(solution) > 100 else solution,
            attacks=attacks,
            logic_attack=logic_attack,
            completeness_attack=completeness_attack,
            quality_attack=quality_attack,
            contradiction_scan=contradiction_scan,
            red_team_argumentation=red_team_argument,
            overall_assessment=overall_assessment,
            critical_issues=critical_issues,
            recommended_revisions=recommended_revisions,
            would_approve=would_approve,
        )

    # ========================================================================
    # Attack Vectors
    # ========================================================================

    def _analyze_argument_structure(self, solution: str) -> ArgumentAnalysis:
        """Analyze the structure of the argument/solution."""
        # Extract sentences as potential premises/conclusions
        sentences = re.split(r'[.!?]+', solution)

        premises = []
        conclusion = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:
                # Look for conclusion indicators
                if any(word in sentence.lower() for word in ["therefore", "thus", "so", "consequently"]):
                    conclusion = sentence
                else:
                    premises.append(sentence)

        # Determine logical structure
        if "because" in solution.lower() or "since" in solution.lower():
            logical_structure = "causal reasoning"
        elif "if.*then" in solution.lower() or "when.*then" in solution.lower():
            logical_structure = "conditional reasoning"
        else:
            logical_structure = "declarative statements"

        # Calculate scores
        completeness_score = self._assess_completeness_score(solution, premises)
        quality_score = self._assess_quality_score(solution)

        return ArgumentAnalysis(
            premises=premises,
            conclusion=conclusion,
            logical_structure=logical_structure,
            completeness_score=completeness_score,
            quality_score=quality_score,
            assumptions=[],
        )

    def _logic_attack(self, solution: str, analysis: ArgumentAnalysis) -> LogicAttack:
        """Apply logic attack - check argument validity."""
        invalid_arguments = []
        fallacies_identified = []

        # Check for formal fallacies
        for fallacy, pattern in self.fallacy_patterns.items():
            matches = re.finditer(pattern, solution, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                line_num = solution[:match.start()].count('\n') + 1
                fallacies_identified.append(
                    f"Line {line_num}: {fallacy.replace('_', ' ')} detected"
                )

        # Check argument structure
        if not analysis.premises and analysis.conclusion:
            invalid_arguments.append(
                "Conclusion without supporting premises"
            )

        # Check for logical consistency
        if self._has_logical_contradictions(solution):
            invalid_arguments.append("Internal logical contradictions detected")

        return LogicAttack(
            valid_arguments=[],
            invalid_arguments=invalid_arguments,
            fallacies_identified=fallacies_identified,
        )

    def _completeness_attack(self, solution: str, original_request: str) -> CompletenessAttack:
        """Apply completeness attack - find what's missing."""
        covered = []
        missing = []
        assumptions = []

        solution_lower = solution.lower()

        # Check for common missing elements
        completeness_checks = {
            "error_handling": ["error", "exception", "try", "catch", "handle"],
            "edge_cases": ["edge case", "boundary", "limit", "maximum", "minimum"],
            "documentation": ["document", "comment", "explain", "describe"],
            "testing": ["test", "verify", "validate", "check"],
            "security": ["security", "auth", "permission", "authorize"],
            "performance": ["performance", "optimize", "efficient", "fast"],
        }

        for aspect, keywords in completeness_checks.items():
            if any(kw in solution_lower for kw in keywords):
                covered.append(aspect)
            else:
                missing.append(aspect)

        # Check against original request requirements
        request_requirements = self._extract_requirements(original_request)
        for requirement in request_requirements:
            if requirement.lower() not in solution_lower:
                missing.append(f"Required: {requirement}")

        return CompletenessAttack(
            covered=covered,
            missing=missing,
            assumptions=assumptions,
        )

    def _quality_attack(self, solution: str, analysis: ArgumentAnalysis) -> QualityAttack:
        """Apply quality attack - assess quality level."""
        strengths = []
        weaknesses = []
        improvements = []

        solution_lower = solution.lower()

        # Quality indicators
        if "clear" in solution_lower or "well-structured" in solution_lower:
            strengths.append("Clear and well-structured presentation")
        if "example" in solution_lower or "demonstrat" in solution_lower:
            strengths.append("Includes examples or demonstrations")
        if "best practice" in solution_lower:
            strengths.append("Follows best practices")

        # Weakness indicators
        if len(solution.split('\n')) < 5:
            weaknesses.append("Very short - likely lacks detail")
        if "could" in solution_lower or "might" in solution_lower:
            weaknesses.append("Vague language - not definitive")
        if "todo" in solution_lower or "tbd" in solution_lower:
            weaknesses.append("Contains placeholders or incomplete sections")

        # Improvements
        if weaknesses:
            improvements.append("Add more specific details and examples")
        if "error" not in solution_lower:
            improvements.append("Include error handling considerations")
        if "document" not in solution_lower:
            improvements.append("Add documentation")

        return QualityAttack(
            strengths=strengths,
            weaknesses=weaknesses,
            improvements=improvements,
        )

    def _contradiction_scan(self, solution: str) -> ContradictionScan:
        """Apply contradiction scan - find inconsistencies."""
        internal_contradictions = []
        external_contradictions = []
        inconsistencies = []

        solution_lower = solution.lower()

        # Check for direct contradictions
        contradiction_patterns = [
            (r'always.*never', "Claims 'always' and 'never' for same thing"),
            (r'all\s+.*\s+none', "Claims 'all' and 'none' together"),
            (r'impossible.*possible', "Logical impossibility"),
        ]

        for pattern, description in contradiction_patterns:
            if re.search(pattern, solution_lower, re.DOTALL):
                internal_contradictions.append(description)

        # Check for external contradictions with common knowledge
        if "prove.*impossible" in solution_lower:
            external_contradictions.append("Claims to prove the impossible")

        # General inconsistency check
        if internal_contradictions or external_contradictions:
            inconsistencies.append("Solution contains contradictory statements")

        return ContradictionScan(
            internal_contradictions=internal_contradictions,
            external_contradictions=external_contradictions,
            inconsistencies=inconsistencies,
        )

    def _red_team_argumentation(self, solution: str, original_request: str) -> RedTeamArgument:
        """Apply red-team argumentation - think like an adversary."""
        adversary_perspective = self._imagine_adversary_perspective(solution, original_request)

        attack_surface = self._identify_attack_surface(solution)

        failure_modes = self._imagine_failure_modes(solution)

        worst_case_scenarios = self._imagine_worst_cases(solution)

        return RedTeamArgument(
            adversary_perspective=adversary_perspective,
            attack_surface=attack_surface,
            failure_modes=failure_modes,
            worst_case_scenarios=worst_case_scenarios,
        )

    # ========================================================================
    # Attack Helper Methods
    # ========================================================================

    def _has_logical_contradictions(self, solution: str) -> bool:
        """Check if solution has internal logical contradictions."""
        # Simple check for "X and not X" patterns
        solution_lower = solution.lower()

        # Split into sentences for comparison
        sentences = re.split(r'[.!?]+', solution_lower)

        for i, sentence in enumerate(sentences):
            for j, other in enumerate(sentences):
                if i >= j:
                    break

                # Check for "X and not X" pattern
                if "not " in sentence:
                    # Extract key term after "not"
                    not_match = re.search(r'not\s+(\w+)', sentence)
                    if not_match:
                        term = not_match.group(1)
                        if term in other:
                            return True

        return False

    def _extract_requirements(self, request: str) -> List[str]:
        """Extract requirements from the original request."""
        # Simple extraction of key phrases
        words = request.lower().split()

        # Filter to significant words
        significant = [w for w in words if len(w) > 3]

        # Remove common words
        common_words = {"this", "that", "with", "from", "have", "will", "would", "could"}
        requirements = [w for w in significant if w not in common_words]

        return requirements

    def _assess_completeness_score(self, solution: str, premises: List[str]) -> float:
        """Assess how complete the solution is."""
        score = 0.5  # Base score

        # More content = more complete
        line_count = len(solution.split('\n'))
        if line_count >= 10:
            score += 0.2
        if line_count >= 20:
            score += 0.2

        # Has supporting premises
        if len(premises) >= 2:
            score += 0.1

        return min(1.0, score)

    def _assess_quality_score(self, solution: str) -> float:
        """Assess the overall quality of the solution."""
        score = 0.5  # Base score

        solution_lower = solution.lower()

        # Quality indicators
        if "example" in solution_lower:
            score += 0.15
        if "explain" in solution_lower or "describe" in solution_lower:
            score += 0.15
        if "best practice" in solution_lower or "recommended" in solution_lower:
            score += 0.1
        if "first" in solution_lower or "next" in solution_lower or "then" in solution_lower:
            score += 0.1

        # Quality anti-patterns
        if "todo" in solution_lower or "tbd" in solution_lower:
            score -= 0.2
        if "quick" in solution_lower and "fix" not in solution_lower:
            score -= 0.1

        return max(0.0, min(1.0, score))

    def _imagine_adversary_perspective(self, solution: str, request: str) -> str:
        """Imagine how an adversary would view this solution."""
        return (
            f"An adversary would note that this solution makes claims about '{request[:50]}...' "
            f"without independent verification. It assumes ideal conditions "
            f"and may not account for malicious actors or edge cases."
        )

    def _identify_attack_surface(self, solution: str) -> List[str]:
        """Identify potential attack surfaces in the solution."""
        attack_surface = []

        solution_lower = solution.lower()

        # Common attack surfaces
        if "user input" in solution_lower:
            attack_surface.append("User input handling - injection risks")
        if "file" in solution_lower or "database" in solution_lower:
            attack_surface.append("File/database operations - path traversal, injection")
        if "network" in solution_lower or "api" in solution_lower:
            attack_surface.append("Network calls - MITM, DoS")
        if "authentication" in solution_lower or "auth" in solution_lower:
            attack_surface.append("Authentication - credential theft")
        if "encryption" not in solution_lower and "data" in solution_lower:
            attack_surface.append("Data at rest - exposure risk")

        if not attack_surface:
            attack_surface.append("Solution claims - all assertions are unverified")

        return attack_surface

    def _imagine_failure_modes(self, solution: str) -> List[str]:
        """Imagine how this solution could fail."""
        failure_modes = [
            "Assumptions prove invalid",
            "External dependencies unavailable",
            "Performance degrades under load",
            "Edge cases not handled",
            "Unexpected input formats",
        ]

        # Add specific failure modes based on content
        solution_lower = solution.lower()
        if "database" in solution_lower:
            failure_modes.append("Database connection fails")
        if "api" in solution_lower:
            failure_modes.append("Third-party API unavailable")
        if "async" in solution_lower:
            failure_modes.append("Race conditions in async operations")

        return failure_modes

    def _imagine_worst_cases(self, solution: str) -> List[str]:
        """Imagine worst case scenarios."""
        return [
            "Complete system failure",
            "Data corruption or loss",
            "Security breach",
            "Service disruption",
            "Incorrect results leading to wrong decisions",
        ]

    # ========================================================================
    # Conversion to Attack Lists
    # ========================================================================

    def _logic_attack_to_list(self, attack: LogicAttack) -> List[Attack]:
        """Convert LogicAttack to Attack list."""
        attacks = []

        for invalid_arg in attack.invalid_arguments:
            attacks.append(Attack(
                vector=AttackVector.LOGIC,
                target="Logic",
                finding=invalid_arg,
                severity=SeverityLevel.HIGH,
                description="Logical flaw detected",
                scenario="This invalid argument undermines the solution's logic",
                suggestion="Fix logical structure or provide valid premises",
            ))

        for fallacy in attack.fallacies_identified:
            attacks.append(Attack(
                vector=AttackVector.LOGIC,
                target="Logic",
                finding=f"Fallacy: {fallacy}",
                severity=SeverityLevel.MEDIUM,
                description="Logical fallacy detected",
                scenario="This fallacy weakens the argument",
                suggestion="Remove fallacy and restructure argument",
            ))

        return attacks

    def _completeness_attack_to_list(self, attack: CompletenessAttack) -> List[Attack]:
        """Convert CompletenessAttack to Attack list."""
        attacks = []

        for missing in attack.missing[:5]:  # Limit to top 5
            attacks.append(Attack(
                vector=AttackVector.COMPLETENESS,
                target="Completeness",
                finding=f"Missing element: {missing}",
                severity=SeverityLevel.MEDIUM,
                description="Required component not addressed",
                scenario="Solution incomplete without this element",
                suggestion=f"Add {missing} considerations",
            ))

        return attacks

    def _quality_attack_to_list(self, attack: QualityAttack) -> List[Attack]:
        """Convert QualityAttack to Attack list."""
        attacks = []

        for weakness in attack.weaknesses:
            attacks.append(Attack(
                vector=AttackVector.QUALITY,
                target="Quality",
                finding=weakness,
                severity=SeverityLevel.LOW,
                description="Quality issue identified",
                scenario="Affects overall solution quality",
                suggestion="Address to improve quality",
            ))

        for improvement in attack.improvements:
            attacks.append(Attack(
                vector=AttackVector.QUALITY,
                target="Quality",
                finding=f"Improvement needed: {improvement}",
                severity=SeverityLevel.LOW,
                description="Suggested improvement",
                scenario="Implementing this would enhance quality",
                suggestion=improvement,
            ))

        return attacks

    def _contradiction_to_list(self, scan: ContradictionScan) -> List[Attack]:
        """Convert ContradictionScan to Attack list."""
        attacks = []

        for internal in scan.internal_contradictions:
            attacks.append(Attack(
                vector=AttackVector.CONTRADICTION,
                target="Consistency",
                finding=f"Internal contradiction: {internal}",
                severity=SeverityLevel.HIGH,
                description="Solution contradicts itself",
                scenario="Creates confusion and ambiguity",
                suggestion="Resolve contradiction by fixing one side",
            ))

        for external in scan.external_contradictions:
            attacks.append(Attack(
                vector=AttackVector.CONTRADICTION,
                target="Consistency",
                finding=f"External contradiction: {external}",
                severity=SeverityLevel.HIGH,
                description="Solution contradicts known facts",
                scenario="Undermines solution credibility",
                suggestion="Verify and correct the contradiction",
            ))

        for inconsistency in scan.inconsistencies:
            attacks.append(Attack(
                vector=AttackVector.CONTRADICTION,
                target="Consistency",
                finding=inconsistency,
                severity=SeverityLevel.MEDIUM,
                description="Inconsistency detected",
                scenario="Creates logical problems",
                suggestion="Review and align content",
            ))

        return attacks

    def _red_team_to_list(self, argument: RedTeamArgument) -> List[Attack]:
        """Convert RedTeamArgument to Attack list."""
        attacks = []

        for surface in argument.attack_surface:
            attacks.append(Attack(
                vector=AttackVector.RED_TEAM,
                target=surface,
                finding=f"Attack surface: {surface}",
                severity=SeverityLevel.MEDIUM,
                description="Potential vulnerability identified",
                scenario=argument.adversary_perspective,
                suggestion=f"Secure {surface} against threats",
            ))

        for mode in argument.failure_modes:
            attacks.append(Attack(
                vector=AttackVector.RED_TEAM,
                target="Reliability",
                finding=f"Failure mode: {mode}",
                severity=SeverityLevel.HIGH,
                description="Potential failure scenario",
                scenario="When this occurs, solution fails",
                suggestion=f"Add handling for: {mode}",
            ))

        return attacks

    def _domain_attacks_to_list(
        self,
        domain_attacks: List[str],
        sme_inputs: Optional[Dict[str, str]]
    ) -> List[Attack]:
        """Convert domain attacks to Attack list."""
        attacks = []

        for attack in domain_attacks:
            sme = next(
                (sme for sme, inputs in sme_inputs.items() if attack in inputs),
                "Unknown"
            )

            attacks.append(Attack(
                vector=AttackVector.RED_TEAM,
                target="Domain",
                finding=f"Domain-specific attack: {attack}",
                severity=SeverityLevel.HIGH,
                description=f"SME ({sme}) identified this domain issue",
                scenario="Could cause domain-specific failures",
                suggestion=attack,
                domain_specific=True,
                sme_source=sme,
            ))

        return attacks

    # ========================================================================
    # Assessment & Recommendations
    # ========================================================================

    def _generate_overall_assessment(
        self,
        attacks: List[Attack],
        solution: str,
        analysis: ArgumentAnalysis
    ) -> str:
        """Generate overall assessment of the solution."""
        critical_count = sum(1 for a in attacks if a.severity == SeverityLevel.CRITICAL)
        high_count = sum(1 for a in attacks if a.severity == SeverityLevel.HIGH)

        total_attacks = len(attacks)

        if critical_count > 0:
            return f"Solution has {critical_count} critical flaw(s) that must be addressed"
        elif high_count > 3:
            return f"Solution has {high_count} high-priority issues requiring attention"
        elif total_attacks > 0:
            return f"Solution has {total_attacks} issues that should be considered"
        else:
            return "Solution appears sound with minor suggestions for improvement"

    def _identify_critical_issues(self, attacks: List[Attack]) -> List[str]:
        """Identify critical issues from all attacks."""
        return [
            a.finding for a in attacks
            if a.severity == SeverityLevel.CRITICAL
        ]

    def _generate_revisions(
        self,
        attacks: List[Attack],
        critical_issues: List[str]
    ) -> List[str]:
        """Generate recommended revisions in priority order."""
        revisions = []

        # Critical issues first
        for issue in critical_issues:
            revisions.append(f"CRITICAL: {issue}")

        # High severity issues
        high_issues = [
            a.finding for a in attacks
            if a.severity == SeverityLevel.HIGH
        ]
        for issue in high_issues[:5]:
            revisions.append(f"HIGH: {issue}")

        # Medium/low severity items
        other_improvements = [
            a.suggestion for a in attacks
            if a.severity in [SeverityLevel.MEDIUM, SeverityLevel.LOW]
        ]
        for improvement in other_improvements[:5]:
            revisions.append(f"SUGGESTED: {improvement}")

        return revisions

    def _would_approve_solution(
        self,
        attacks: List[Attack],
        critical_issues: List[str]
    ) -> bool:
        """Determine if the solution passes the critique."""
        # Fail if critical issues
        if critical_issues:
            return False

        # Fail if many high-severity issues
        high_count = sum(1 for a in attacks if a.severity == SeverityLevel.HIGH)
        if high_count > 5:
            return False

        # Pass otherwise
        return True

    def _load_system_prompt(self) -> str:
        """Load the system prompt from file."""
        try:
            with open(self.system_prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            return "You are the Critic. Attack solutions through five vectors."


# =============================================================================
# Convenience Functions
# =============================================================================

def create_critic(
    system_prompt_path: str = "config/agents/critic/CLAUDE.md",
    model: str = "claude-3-5-opus-20240507",
) -> CriticAgent:
    """Create a configured Critic agent."""
    return CriticAgent(
        system_prompt_path=system_prompt_path,
        model=model,
    )
