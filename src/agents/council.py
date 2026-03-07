"""
Strategic Council Agents

Three governance agents for Tier 3-4 tasks:
- Domain Council Chair: SME selection and collaboration
- Quality Arbiter: Quality standards and dispute resolution
- Ethics & Safety Advisor: Bias, PII, compliance, safety review
"""

import re
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum

from src.schemas.council import (
    SMESelectionReport,
    SMESelection,
    InteractionMode,
    QualityStandard,
    QualityCriteria,
    QualityVerdict,
    DisputedItem,
    EthicsReview,
    FlaggedIssue,
    IssueType,
    IssueSeverity,
)

from src.core.sme_registry import (
    SME_REGISTRY,
    find_personas_by_keywords,
    validate_interaction_mode,
)


# =============================================================================
# Domain Council Chair
# =============================================================================

class CouncilChairAgent:
    """
    The Domain Council Chair selects SME personas for Tier 3-4 tasks.

    Key responsibilities:
    - Analyze task domain requirements
    - Select up to 3 SME personas from registry
    - Specify skills for each SME
    - Define interaction modes
    - Plan SME collaboration
    """

    def __init__(
        self,
        system_prompt_path: str = "config/agents/council/CLAUDE.md",
        model: str = "claude-3-5-opus-20240507",
        max_turns: int = 30,
    ):
        """
        Initialize the Council Chair agent.

        Args:
            system_prompt_path: Path to system prompt file
            model: Model to use (opus for complex selection)
            max_turns: Maximum conversation turns
        """
        self.system_prompt_path = system_prompt_path
        self.model = model
        self.max_turns = max_turns
        self.system_prompt = self._load_system_prompt()

        # Domain keyword patterns for SME selection
        self.domain_patterns = {
            "iam_architect": [
                "sailpoint", "cyberark", "rbac", "identity", "azure ad",
                "okta", "authentication", "authorization", "iam", "ldap"
            ],
            "cloud_architect": [
                "aws", "azure", "gcp", "cloud", "serverless", "lambda",
                "ec2", "s3", "kubernetes", "eks", "aks", "terraform"
            ],
            "security_analyst": [
                "security", "vulnerability", "penetration", "threat",
                "exploit", "attack", "defense", "firewall", "waf"
            ],
            "data_engineer": [
                "data", "etl", "pipeline", "sql", "database", "warehouse",
                "snowflake", "databricks", "spark", "kafka", "airflow"
            ],
            "ai_ml_engineer": [
                "machine learning", "ml", "ai", "model", "training",
                "tensorflow", "pytorch", "scikit", "nlp", "llm", "prompt"
            ],
            "test_engineer": [
                "test", "testing", "qa", "pytest", "selenium", "cypress",
                "unit test", "integration test", "e2e", "mock"
            ],
            "business_analyst": [
                "requirement", "user story", "acceptance", "stakeholder",
                "workflow", "process", "business", "analysis"
            ],
            "technical_writer": [
                "documentation", "docs", "readme", "guide", "manual",
                "api doc", "tutorial", "explain"
            ],
            "devops_engineer": [
                "devops", "ci/cd", "deployment", "jenkins", "github actions",
                "docker", "container", "infrastructure", "monitoring"
            ],
            "frontend_developer": [
                "frontend", "ui", "react", "vue", "angular", "css",
                "javascript", "typescript", "component", "responsive"
            ],
        }

    def select_smes(
        self,
        task_description: str,
        analyst_report: Optional[Dict[str, Any]] = None,
        tier_level: int = 3,
        max_smes: int = 3,
    ) -> SMESelectionReport:
        """
        Select SME personas for the task.

        Args:
            task_description: The task description
            analyst_report: Optional Analyst report for additional context
            tier_level: Current tier level (3 or 4)
            max_smes: Maximum SMEs to select

        Returns:
            SMESelectionReport with selected SMEs and collaboration plan
        """
        # Step 1: Identify required domains
        required_domains = self._identify_required_domains(
            task_description, analyst_report
        )

        # Step 2: Select SME personas for each domain
        selected_smes = self._select_smes_for_domains(
            required_domains, tier_level, max_smes
        )

        # Step 3: Define interaction modes
        for sme in selected_smes:
            if sme.interaction_mode == InteractionMode.ADVISOR:
                sme.interaction_mode = self._determine_interaction_mode(
                    sme, task_description, tier_level
                )

        # Step 4: Identify domain gaps
        domain_gaps = self._identify_domain_gaps(
            required_domains, selected_smes
        )

        # Step 5: Create collaboration plan
        collaboration_plan = self._create_collaboration_plan(
            selected_smes, tier_level
        )

        # Step 6: Define expected contributions
        expected_contributions = {
            sme.persona_name: self._define_expected_contribution(sme, task_description)
            for sme in selected_smes
        }

        # Step 7: Determine if full Council is needed
        requires_full_council = self._requires_full_council(
            task_description, tier_level, selected_smes
        )

        return SMESelectionReport(
            task_summary=task_description[:200],
            selected_smes=selected_smes,
            domain_gaps_identified=domain_gaps,
            collaboration_plan=collaboration_plan,
            expected_sme_contributions=expected_contributions,
            tier_recommendation=tier_level,
            requires_full_council=requires_full_council,
        )

    def _identify_required_domains(
        self,
        task_description: str,
        analyst_report: Optional[Dict[str, Any]],
    ) -> List[str]:
        """Identify which domains are required."""
        domains = set()
        task_lower = task_description.lower()

        # Check against domain patterns
        for persona_name, keywords in self.domain_patterns.items():
            for keyword in keywords:
                if keyword in task_lower:
                    domains.add(persona_name)
                    break

        # Check analyst report for additional hints
        if analyst_report:
            modality = analyst_report.get("modality", "").lower()
            if "code" in modality:
                domains.add("test_engineer")  # Code needs testing
            if "image" in modality:
                domains.add("frontend_developer")  # Images often mean UI/diagrams

        # Check for complexity indicators
        complexity_indicators = {
            "architecture": "cloud_architect",
            "design": "technical_writer",
            "data": "data_engineer",
            "model": "ai_ml_engineer",
            "deploy": "devops_engineer",
        }

        for indicator, domain in complexity_indicators.items():
            if indicator in task_lower:
                domains.add(domain)

        return list(domains)

    def _select_smes_for_domains(
        self,
        required_domains: List[str],
        tier_level: int,
        max_smes: int,
    ) -> List[SMESelection]:
        """Select SME personas for the required domains."""
        selections = []

        for domain_name in required_domains[:max_smes]:
            if domain_name in SME_REGISTRY:
                persona = SME_REGISTRY[domain_name]

                # Determine activation phase
                activation_phase = self._determine_activation_phase(
                    domain_name, tier_level
                )

                # Get skills for this SME
                skills_to_load = persona.skill_files or []

                selections.append(SMESelection(
                    persona_name=persona.name,
                    persona_domain=persona.domain,
                    skills_to_load=skills_to_load,
                    interaction_mode=InteractionMode.ADVISOR,  # Will be refined
                    reasoning=f"Domain expertise required for {persona.domain}",
                    activation_phase=activation_phase,
                ))

        return selections

    def _determine_activation_phase(
        self,
        domain_name: str,
        tier_level: int,
    ) -> str:
        """Determine which phase the SME should activate in."""
        # Early phases for certain domains
        early_phase_domains = {
            "business_analyst": "clarification",
            "technical_writer": "planning",
            "cloud_architect": "planning",
        }

        if domain_name in early_phase_domains:
            return early_phase_domains[domain_name]

        # Default: participate in execution
        return "execution"

    def _determine_interaction_mode(
        self,
        sme: SMESelection,
        task_description: str,
        tier_level: int,
    ) -> InteractionMode:
        """Determine the interaction mode for an SME."""
        # Tier 4 uses debate mode for some SMEs
        if tier_level == 4:
            debate_domains = {
                "security_analyst",
                "ai_ml_engineer",
                "cloud_architect",
            }
            if sme.persona_domain.lower() in [d.lower() for d in debate_domains]:
                return InteractionMode.DEBATER

        # Co-execution for hands-on domains
        co_executor_domains = {
            "frontend_developer",
            "devops_engineer",
            "data_engineer",
        }
        if sme.persona_domain.lower() in [d.lower() for d in co_executor_domains]:
            return InteractionMode.CO_EXECUTOR

        # Default: advisor
        return InteractionMode.ADVISOR

    def _identify_domain_gaps(
        self,
        required_domains: List[str],
        selected_smes: List[SMESelection],
    ) -> List[str]:
        """Identify domain gaps that couldn't be filled."""
        selected_domains = {sme.persona_domain.lower() for sme in selected_smes}

        gaps = []
        for domain in required_domains:
            if domain not in SME_REGISTRY:
                # Domain not in registry
                gaps.append(f"Domain '{domain}' not available in registry")
            elif domain not in selected_domains:
                # Available but not selected (probably due to max_smes limit)
                gaps.append(f"Domain '{domain}' available but not selected (limit reached)")

        return gaps

    def _create_collaboration_plan(
        self,
        selected_smes: List[SMESelection],
        tier_level: int,
    ) -> str:
        """Create a plan for how SMEs should collaborate."""
        if not selected_smes:
            return "No SMEs selected - operational agents will handle task"

        # Group SMEs by interaction mode
        advisors = [s for s in selected_smes if s.interaction_mode == InteractionMode.ADVISOR]
        co_executors = [s for s in selected_smes if s.interaction_mode == InteractionMode.CO_EXECUTOR]
        debaters = [s for s in selected_smes if s.interaction_mode == InteractionMode.DEBATER]

        plan_parts = []

        if advisors:
            advisor_names = ", ".join([a.persona_name for a in advisors])
            plan_parts.append(f"**Advisors** ({advisor_names}) will provide guidance during their assigned phases")

        if co_executors:
            co_names = ", ".join([c.persona_name for c in co_executors])
            plan_parts.append(f"**Co-executors** ({co_names}) will work alongside Executor to produce output")

        if debaters:
            debate_names = ", ".join([d.persona_name for d in debaters])
            plan_parts.append(f"**Debaters** ({debate_names}) will participate in self-play debate for Tier {tier_level}")

        # Add collaboration protocol
        plan_parts.append("\n**Collaboration Protocol:**")
        plan_parts.append("- SMEs will load their designated skills")
        plan_parts.append("- Executor will incorporate SME input into final output")
        plan_parts.append("- Disagreements between SMEs will be resolved by Quality Arbiter")

        return "\n".join(plan_parts)

    def _define_expected_contribution(
        self,
        sme: SMESelection,
        task_description: str,
    ) -> str:
        """Define expected contribution from an SME."""
        domain = sme.persona_domain.lower()

        contributions = {
            "iam_architect": "Review identity and access management requirements",
            "cloud_architect": "Validate architectural decisions and cloud best practices",
            "security_analyst": "Identify security vulnerabilities and recommend mitigations",
            "data_engineer": "Design data structures and pipeline architecture",
            "ai_ml_engineer": "Guide model selection and implementation approach",
            "test_engineer": "Define testing strategy and coverage requirements",
            "business_analyst": "Ensure requirements are properly captured and prioritized",
            "technical_writer": "Ensure output is clearly documented and explainable",
            "devops_engineer": "Validate deployment and infrastructure considerations",
            "frontend_developer": "Implement or review frontend components",
        }

        return contributions.get(domain, f"Provide {sme.persona_domain} expertise")

    def _requires_full_council(
        self,
        task_description: str,
        tier_level: int,
        selected_smes: List[SMESelection],
    ) -> bool:
        """Determine if full Council (Arbiter + Ethics) is needed."""
        # Tier 4 always requires full Council
        if tier_level == 4:
            return True

        # Check for sensitive content
        sensitive_keywords = [
            "personal data", "pii", "medical", "health", "financial",
            "credit card", "social security", "children", "vulnerable"
        ]
        task_lower = task_description.lower()
        if any(kw in task_lower for kw in sensitive_keywords):
            return True

        # Check for multiple SMEs (complex collaboration)
        if len(selected_smes) >= 2:
            return True

        return False

    def _load_system_prompt(self) -> str:
        """Load the system prompt from file."""
        try:
            with open(self.system_prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            return "You are the Council Chair. Select SME personas for Tier 3-4 tasks."


# =============================================================================
# Quality Arbiter
# =============================================================================

class QualityArbiterAgent:
    """
    The Quality Arbiter sets quality standards and resolves disputes.

    Key responsibilities:
    - Set quality acceptance criteria BEFORE execution (Tier 4)
    - Act as final tiebreaker for quality disputes
    - Resolve disagreements between Verifier and Critic
    - Provide binding QualityVerdict after 2 failed debate rounds
    """

    def __init__(
        self,
        system_prompt_path: str = "config/agents/council/CLAUDE.md",
        model: str = "claude-3-5-opus-20240507",
        max_turns: int = 30,
    ):
        """
        Initialize the Quality Arbiter agent.

        Args:
            system_prompt_path: Path to system prompt file
            model: Model to use (opus for quality decisions)
            max_turns: Maximum conversation turns
        """
        self.system_prompt_path = system_prompt_path
        self.model = model
        self.max_turns = max_turns
        self.system_prompt = self._load_system_prompt()

        # Default quality criteria templates
        self.default_criteria = {
            "accuracy": {
                "metric": "Factual accuracy",
                "threshold": "≥90% of claims verified",
                "measurement_method": "Verifier claim validation",
                "weight": 0.3,
            },
            "completeness": {
                "metric": "Requirement coverage",
                "threshold": "100% of critical requirements addressed",
                "measurement_method": "Analyst requirement checklist",
                "weight": 0.25,
            },
            "quality": {
                "metric": "Output quality",
                "threshold": "No critical issues from Critic",
                "measurement_method": "Critic attack vector analysis",
                "weight": 0.25,
            },
            "coherence": {
                "metric": "Logical consistency",
                "threshold": "No internal contradictions",
                "measurement_method": "Reviewer consistency check",
                "weight": 0.2,
            },
        }

    def set_quality_standard(
        self,
        task_description: str,
        analyst_report: Optional[Dict[str, Any]] = None,
        tier_level: int = 4,
        custom_requirements: Optional[List[str]] = None,
    ) -> QualityStandard:
        """
        Set quality acceptance criteria before execution.

        Args:
            task_description: The task description
            analyst_report: Optional Analyst report
            tier_level: Current tier level
            custom_requirements: Optional custom quality requirements

        Returns:
            QualityStandard with acceptance criteria
        """
        # Step 1: Build quality criteria
        quality_criteria = self._build_quality_criteria(
            task_description, analyst_report, tier_level, custom_requirements
        )

        # Step 2: Set pass threshold
        pass_threshold = self._determine_pass_threshold(tier_level)

        # Step 3: Define critical must-haves
        critical_must_haves = self._define_critical_must_haves(
            task_description, analyst_report
        )

        # Step 4: Define nice-to-haves
        nice_to_haves = self._define_nice_to_haves(
            task_description, tier_level
        )

        # Step 5: Define measurement protocol
        measurement_protocol = self._define_measurement_protocol(
            quality_criteria, tier_level
        )

        return QualityStandard(
            task_summary=task_description[:200],
            quality_criteria=quality_criteria,
            overall_pass_threshold=pass_threshold,
            critical_must_haves=critical_must_haves,
            nice_to_haves=nice_to_haves,
            measurement_protocol=measurement_protocol,
        )

    def resolve_dispute(
        self,
        arbitration_input: Dict[str, Any],
        verifier_report: Dict[str, Any],
        critic_report: Dict[str, Any],
        reviewer_verdict: str,
    ) -> QualityVerdict:
        """
        Resolve a quality dispute between agents.

        Args:
            arbitration_input: Input from Reviewer with disagreement details
            verifier_report: Verifier's report
            critic_report: Critic's report
            reviewer_verdict: Reviewer's verdict

        Returns:
            QualityVerdict with binding resolution
        """
        # Step 1: Analyze the dispute
        disputed_items = self._analyze_dispute(
            verifier_report, critic_report, reviewer_verdict
        )

        # Step 2: Perform arbiter analysis
        arbiter_analysis = self._perform_arbiter_analysis(
            disputed_items, verifier_report, critic_report
        )

        # Step 3: Determine resolution
        resolution = self._determine_resolution(
            arbiter_analysis, disputed_items
        )

        # Step 4: Define required actions
        required_actions = self._define_required_actions(
            resolution, disputed_items
        )

        # Step 5: Check if overriding Reviewer
        overrides_reviewer = self._should_override_reviewer(
            resolution, reviewer_verdict
        )

        return QualityVerdict(
            original_dispute=arbitration_input.get("disagreement_reason", ""),
            disputed_items=disputed_items,
            debate_rounds_completed=arbitration_input.get("debate_rounds_completed", 2),
            arbiter_analysis=arbiter_analysis,
            resolution=resolution,
            required_actions=required_actions,
            overrides_reviewer=overrides_reviewer,
        )

    def _build_quality_criteria(
        self,
        task_description: str,
        analyst_report: Optional[Dict[str, Any]],
        tier_level: int,
        custom_requirements: Optional[List[str]],
    ) -> List[QualityCriteria]:
        """Build quality criteria list."""
        criteria = []

        # Start with default criteria
        for name, template in self.default_criteria.items():
            criteria.append(QualityCriteria(**template))

        # Add task-specific criteria
        task_lower = task_description.lower()

        if "code" in task_lower:
            criteria.append(QualityCriteria(
                metric="Code quality",
                threshold="No critical security issues",
                measurement_method="Code Reviewer security scan",
                weight=0.15,
            ))

        if "data" in task_lower:
            criteria.append(QualityCriteria(
                metric="Data integrity",
                threshold="No data corruption or loss",
                measurement_method="Verifier validation",
                weight=0.15,
            ))

        # Add custom requirements if provided
        if custom_requirements:
            for i, req in enumerate(custom_requirements):
                criteria.append(QualityCriteria(
                    metric=f"Custom requirement {i+1}",
                    threshold=req,
                    measurement_method="Manual review",
                    weight=0.1,
                ))

        # Normalize weights
        total_weight = sum(c.weight for c in criteria)
        if total_weight > 0:
            for c in criteria:
                c.weight = c.weight / total_weight

        return criteria

    def _determine_pass_threshold(self, tier_level: int) -> float:
        """Determine overall pass threshold."""
        if tier_level == 4:
            return 0.85  # Higher standard for adversarial tasks
        return 0.75  # Standard for Tier 3

    def _define_critical_must_haves(
        self,
        task_description: str,
        analyst_report: Optional[Dict[str, Any]],
    ) -> List[str]:
        """Define non-negotiable requirements."""
        must_haves = []

        # Security is always critical
        must_haves.append("No critical security vulnerabilities")

        # Check for domain-specific critical requirements
        task_lower = task_description.lower()

        if "code" in task_lower:
            must_haves.append("Code must be syntactically valid")
            must_haves.append("Code must not contain hardcoded secrets")

        if "data" in task_lower or "api" in task_lower:
            must_haves.append("No PII or sensitive data exposure")

        # Check analyst report for critical missing info
        if analyst_report:
            missing_critical = [
                m.get("requirement", "")
                for m in analyst_report.get("missing_info", [])
                if m.get("severity") == "critical"
            ]
            must_haves.extend(missing_critical)

        return must_haves

    def _define_nice_to_haves(
        self,
        task_description: str,
        tier_level: int,
    ) -> List[str]:
        """Define desirable but not required features."""
        nice_to_haves = []

        task_lower = task_description.lower()

        if "code" in task_lower:
            nice_to_haves.append("Type hints included")
            nice_to_haves.append("Docstrings for functions")
            nice_to_haves.append("Unit tests included")

        if "documentation" in task_lower or "explain" in task_lower:
            nice_to_haves.append("Examples provided")
            nice_to_haves.append("Visual aids (diagrams, tables)")

        if tier_level == 4:
            nice_to_haves.append("Multiple approaches considered")
            nice_to_haves.append("Alternative solutions discussed")

        return nice_to_haves

    def _define_measurement_protocol(
        self,
        quality_criteria: List[QualityCriteria],
        tier_level: int,
    ) -> str:
        """Define how quality will be measured."""
        protocol_parts = []

        protocol_parts.append("**Quality Measurement Protocol:**")
        protocol_parts.append("\n1. **Pre-execution**: QualityStandard defines acceptance criteria")

        if tier_level == 4:
            protocol_parts.append("2. **During execution**: Real-time quality monitoring")
            protocol_parts.append("3. **Post-execution**: Multi-agent quality assessment")
        else:
            protocol_parts.append("2. **Post-execution**: Agent quality assessment")

        protocol_parts.append("\n**Scoring:**")
        for criterion in quality_criteria:
            protocol_parts.append(
                f"- {criterion.metric}: {criterion.threshold} "
                f"(weight: {criterion.weight:.1%})"
            )

        protocol_parts.append(f"\n**Pass Threshold:** {sum(c.weight for c in quality_criteria):.1%}")

        return "\n".join(protocol_parts)

    def _analyze_dispute(
        self,
        verifier_report: Dict[str, Any],
        critic_report: Dict[str, Any],
        reviewer_verdict: str,
    ) -> List[DisputedItem]:
        """Analyze the dispute to identify specific disputed items."""
        disputed = []

        # Get verdicts
        verifier_verdict = verifier_report.get("verdict", "PASS")
        critic_verdict_str = critic_report.get("overall_assessment", "")

        # Check for specific disagreement points
        verifier_issues = verifier_report.get("flagged_claims", [])
        critic_issues = [
            attack.get("description", "")
            for attack in critic_report.get("attacks", [])
        ]

        # Find overlapping issues (both flagged but different conclusions)
        for v_issue in verifier_issues:
            for c_issue in critic_issues:
                if self._issues_overlap(v_issue, c_issue):
                    disputed.append(DisputedItem(
                        item=f"Overlap: {v_issue[:50]}...",
                        reviewer_position=reviewer_verdict,
                        verifier_position=verifier_verdict,
                        critic_position="FAIL" if "critical" in critic_verdict_str else "PASS",
                        arbiter_resolution="",  # To be filled
                    ))

        return disputed

    def _issues_overlap(self, issue1: str, issue2: str) -> bool:
        """Check if two issues overlap semantically."""
        # Simple word overlap check
        words1 = set(issue1.lower().split())
        words2 = set(issue2.lower().split())

        overlap = words1 & words2
        return len(overlap) >= 2

    def _perform_arbiter_analysis(
        self,
        disputed_items: List[DisputedItem],
        verifier_report: Dict[str, Any],
        critic_report: Dict[str, Any],
    ) -> str:
        """Perform the arbiter's analysis."""
        analysis_parts = []

        analysis_parts.append("**Arbiter Analysis:**\n")

        # Weigh Verifier's position
        verifier_reliability = verifier_report.get("overall_reliability", 0.7)
        analysis_parts.append(
            f"**Verifier Assessment:** Reliability {verifier_reliability:.1%}"
        )

        # Weigh Critic's position
        critic_assessment = critic_report.get("overall_assessment", "")
        analysis_parts.append(f"**Critic Assessment:** {critic_assessment}")

        # Analyze disputed items
        if disputed_items:
            analysis_parts.append(f"\n**Disputed Items:** {len(disputed_items)}")
            for i, item in enumerate(disputed_items[:3], 1):
                analysis_parts.append(f"{i}. {item.item}")

        # Make determination
        analysis_parts.append("\n**Determination:**")

        if verifier_reliability >= 0.8 and "critical" not in critic_assessment.lower():
            analysis_parts.append("Verifier's high reliability outweighs Critic's concerns")
        elif "critical" in critic_assessment.lower():
            analysis_parts.append("Critic identified critical issues requiring resolution")
        else:
            analysis_parts.append("Both positions have merit - partial remediation required")

        return "\n".join(analysis_parts)

    def _determine_resolution(
        self,
        arbiter_analysis: str,
        disputed_items: List[DisputedItem],
    ) -> str:
        """Determine the final resolution."""
        if "critical" in arbiter_analysis.lower():
            return "CRITICAL_ISSUES - Executor must revise before proceeding"
        elif "reliability" in arbiter_analysis.lower():
            return "VERIFIER_PREVAILS - Output may proceed with minor clarifications"
        else:
            return "PARTIAL_REMEDIATION - Address specific disputed items"

    def _define_required_actions(
        self,
        resolution: str,
        disputed_items: List[DisputedItem],
    ) -> List[str]:
        """Define required actions from resolution."""
        actions = []

        if "CRITICAL" in resolution:
            actions.append("Executor to revise output addressing all critical issues")
            actions.append("Verifier to re-check revised output")
            actions.append("Critic to re-evaluate revised output")

        elif "VERIFIER_PREVAILS" in resolution:
            actions.append("Add clarifications for disputed items")
            actions.append("Document why Critic's concerns were addressed")

        elif "PARTIAL" in resolution:
            for item in disputed_items[:3]:
                actions.append(f"Address: {item.item}")

        return actions

    def _should_override_reviewer(
        self,
        resolution: str,
        reviewer_verdict: str,
    ) -> bool:
        """Determine if Arbiter should override Reviewer."""
        # Override if critical issues found
        if "CRITICAL" in resolution:
            return True

        # Override if reviewer passed but arbiter found issues
        if reviewer_verdict == "PASS" and "REMED" in resolution:
            return True

        return False

    def _load_system_prompt(self) -> str:
        """Load the system prompt from file."""
        try:
            with open(self.system_prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            return "You are the Quality Arbiter. Set quality standards and resolve disputes."


# =============================================================================
# Ethics & Safety Advisor
# =============================================================================

class EthicsAdvisorAgent:
    """
    The Ethics & Safety Advisor reviews for bias, PII, compliance, safety.

    Key responsibilities:
    - Review output for potential bias
    - Scan for PII exposure
    - Assess compliance risks
    - Identify safety concerns
    - Provide binding PASS/FAIL verdict
    """

    def __init__(
        self,
        system_prompt_path: str = "config/agents/council/CLAUDE.md",
        model: str = "claude-3-5-opus-20240507",
        max_turns: int = 30,
    ):
        """
        Initialize the Ethics Advisor agent.

        Args:
            system_prompt_path: Path to system prompt file
            model: Model to use (opus for sensitive reviews)
            max_turns: Maximum conversation turns
        """
        self.system_prompt_path = system_prompt_path
        self.model = model
        self.max_turns = max_turns
        self.system_prompt = self._load_system_prompt()

        # PII patterns
        self.pii_patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
            "credit_card": r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',
            "ip_address": r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
            "api_key": r'\b[A-Za-z0-9]{32,}\b',  # Long alphanumeric strings
        }

        # Bias indicators
        self.bias_patterns = {
            "gender": [
                r'\b(he|she|him|her|his|hers)\s+(always|never|obviously|clearly)',
                r'\b(men|women|males|females)\s+(are|can\'t|shouldn\'t|unable to)',
            ],
            "racial": [
                r'\b(all|every|none|no)\s+\w+(?:s|people|folks)\s+(are|do|think)',
            ],
            "age": [
                r'\b(older|younger|elderly|boomers|gen z)\s+(can\'t|unable to|bad at)',
            ],
        }

        # Safety concern patterns
        self.safety_patterns = {
            "self_harm": [
                r'\b(kill|hurt|harm)\s+(myself|yourself)\b',
                r'\bsuicide\s+(method|ways|how)\b',
            ],
            "violence": [
                r'\b(how\s+to\s+(kill|murder|attack|harm))\b',
                r'\b(build|make|create)\s+(bomb|weapon|poison)\b',
            ],
            "illegal": [
                r'\b(how\s+to\s+(steal|rob|cheat|hack))\b',
                r'\b(bypass|avoid)\s+(security|payment|auth)\b',
            ],
        }

    def review_output(
        self,
        output: str,
        task_description: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> EthicsReview:
        """
        Review output for ethics and safety concerns.

        Args:
            output: The content to review
            task_description: Original task description
            context: Additional context (tier, agents used, etc.)

        Returns:
            EthicsReview with findings and verdict
        """
        # Step 1: Scan for PII
        pii_results = self._scan_for_pii(output)

        # Step 2: Check for bias
        bias_results = self._check_for_bias(output)

        # Step 3: Assess safety concerns
        safety_results = self._assess_safety(output)

        # Step 4: Compliance assessment
        compliance_results = self._assess_compliance(output, context)

        # Step 5: Compile flagged issues
        flagged_issues = (
            pii_results["issues"] +
            bias_results["issues"] +
            safety_results["issues"] +
            compliance_results["issues"]
        )

        # Step 6: Determine verdict
        verdict, can_proceed = self._determine_verdict(flagged_issues)

        # Step 7: Generate recommendations
        recommendations = self._generate_recommendations(flagged_issues)

        # Step 8: Build assessment strings
        bias_analysis = self._build_bias_analysis(bias_results)
        pii_scan_results = self._build_pii_analysis(pii_results)
        compliance_assessment = self._build_compliance_analysis(compliance_results)
        safety_assessment = self._build_safety_analysis(safety_results)

        # Step 9: Required remediations
        required_remediations = [
            issue.remediation
            for issue in flagged_issues
            if issue.blocks_output
        ]

        return EthicsReview(
            output_summary=output[:100],
            verdict=verdict,
            flagged_issues=flagged_issues,
            bias_analysis=bias_analysis,
            pii_scan_results=pii_scan_results,
            compliance_assessment=compliance_assessment,
            safety_assessment=safety_assessment,
            recommendations=recommendations,
            can_proceed=can_proceed,
            required_remediations=required_remediations,
        )

    def _scan_for_pii(self, output: str) -> Dict[str, Any]:
        """Scan output for PII."""
        issues = []
        findings = []

        for pii_type, pattern in self.pii_patterns.items():
            matches = re.finditer(pattern, output)
            for match in matches:
                found_text = match.group()
                # Don't include actual PII in output - redact it
                redacted = f"[{pii_type.upper()}_REDACTED]"

                issues.append(FlaggedIssue(
                    issue_type=IssueType.PII,
                    severity=IssueSeverity.HIGH,
                    description=f"Potential {pii_type} detected",
                    location=f"Position {match.start()}-{match.end()}",
                    potential_harm=f"Exposure of {pii_type} could violate privacy regulations",
                    remediation=f"Redact or remove: {redacted}",
                    blocks_output=True,
                ))

                findings.append(f"{pii_type}: {redacted}")

        return {"issues": issues, "findings": findings}

    def _check_for_bias(self, output: str) -> Dict[str, Any]:
        """Check output for potential bias."""
        issues = []
        findings = []

        output_lower = output.lower()

        for bias_type, patterns in self.bias_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, output_lower)
                for match in matches:
                    issues.append(FlaggedIssue(
                        issue_type=IssueType.BIAS,
                        severity=IssueSeverity.MEDIUM,
                        description=f"Potential {bias_type} bias detected",
                        location=f"Position {match.start()}-{match.end()}",
                        potential_harm="Perpetuation of harmful stereotypes",
                        remediation=f"Review language for {bias_type} bias",
                        blocks_output=False,
                    ))

                    findings.append(f"{bias_type} bias: {match.group()[:30]}...")

        return {"issues": issues, "findings": findings}

    def _assess_safety(self, output: str) -> Dict[str, Any]:
        """Assess safety concerns."""
        issues = []
        findings = []

        output_lower = output.lower()

        for safety_type, patterns in self.safety_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, output_lower)
                for match in matches:
                    # Determine severity based on type
                    severity = IssueSeverity.CRITICAL if safety_type == "self_harm" else IssueSeverity.HIGH

                    issues.append(FlaggedIssue(
                        issue_type=IssueType.SAFETY,
                        severity=severity,
                        description=f"Potential {safety_type} content",
                        location=f"Position {match.start()}-{match.end()}",
                        potential_harm=f"{safety_type.replace('_', ' ').title()} risk",
                        remediation="Remove or refuse to generate this content",
                        blocks_output=True,
                    ))

                    findings.append(f"{safety_type}: {match.group()[:30]}...")

        return {"issues": issues, "findings": findings}

    def _assess_compliance(
        self,
        output: str,
        context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Assess compliance risks."""
        issues = []
        findings = []

        output_lower = output.lower()

        # Check for medical advice disclaimers
        if "medical" in output_lower or "health" in output_lower:
            if "consult" not in output_lower and "doctor" not in output_lower:
                issues.append(FlaggedIssue(
                    issue_type=IssueType.COMPLIANCE,
                    severity=IssueSeverity.HIGH,
                    description="Medical content without disclaimer",
                    location="N/A",
                    potential_harm="Liability for practicing medicine without license",
                    remediation="Add disclaimer: Not medical advice, consult a professional",
                    blocks_output=True,
                ))
                findings.append("Medical content lacks disclaimer")

        # Check for financial advice
        if "investment" in output_lower or "financial advice" in output_lower:
            if "consult" not in output_lower and "advisor" not in output_lower:
                issues.append(FlaggedIssue(
                    issue_type=IssueType.COMPLIANCE,
                    severity=IssueSeverity.MEDIUM,
                    description="Financial content without disclaimer",
                    location="N/A",
                    potential_harm="Regulatory liability",
                    remediation="Add disclaimer: Not financial advice",
                    blocks_output=False,
                ))
                findings.append("Financial content lacks disclaimer")

        return {"issues": issues, "findings": findings}

    def _determine_verdict(self, flagged_issues: List[FlaggedIssue]) -> tuple[str, bool]:
        """Determine the overall verdict."""
        # FAIL if any critical issues or blocking issues
        for issue in flagged_issues:
            if issue.severity == IssueSeverity.CRITICAL:
                return "FAIL", False
            if issue.blocks_output:
                return "FAIL", False

        # PASS if no issues
        if not flagged_issues:
            return "PASS", True

        # FAIL if more than 3 high-severity issues
        high_count = sum(1 for i in flagged_issues if i.severity == IssueSeverity.HIGH)
        if high_count > 3:
            return "FAIL", False

        # Otherwise PASS with notes
        return "PASS", True

    def _generate_recommendations(self, flagged_issues: List[FlaggedIssue]) -> List[str]:
        """Generate recommendations from flagged issues."""
        recommendations = []

        if not flagged_issues:
            recommendations.append("No ethics or safety concerns detected")
            return recommendations

        # Group by type
        by_type: Dict[str, List[FlaggedIssue]] = {}
        for issue in flagged_issues:
            if issue.issue_type.value not in by_type:
                by_type[issue.issue_type.value] = []
            by_type[issue.issue_type.value].append(issue)

        # Generate recommendations per type
        for issue_type, issues in by_type.items():
            if issue_type == "pii":
                recommendations.append("Implement PII redaction for all personal data")
            elif issue_type == "bias":
                recommendations.append("Review language for unconscious bias")
            elif issue_type == "safety":
                recommendations.append("Remove content that could cause harm")
            elif issue_type == "compliance":
                recommendations.append("Add appropriate disclaimers for regulated content")

        # Count issues
        total_issues = len(flagged_issues)
        blocking = sum(1 for i in flagged_issues if i.blocks_output)
        recommendations.append(
            f"Address {blocking} blocking issue(s) out of {total_issues} total"
        )

        return recommendations[:5]

    def _build_bias_analysis(self, bias_results: Dict[str, Any]) -> str:
        """Build bias analysis string."""
        if not bias_results["findings"]:
            return "No significant bias detected"

        return f"Found {len(bias_results['findings'])} potential bias issue(s): " + \
               ", ".join(bias_results["findings"][:3])

    def _build_pii_analysis(self, pii_results: Dict[str, Any]) -> str:
        """Build PII analysis string."""
        if not pii_results["findings"]:
            return "No PII detected"

        return f"Found {len(pii_results['findings'])} potential PII instance(s): " + \
               ", ".join(pii_results["findings"][:3])

    def _build_compliance_analysis(self, compliance_results: Dict[str, Any]) -> str:
        """Build compliance analysis string."""
        if not compliance_results["findings"]:
            return "No compliance concerns"

        return "Compliance issues: " + ", ".join(compliance_results["findings"])

    def _build_safety_analysis(self, safety_results: Dict[str, Any]) -> str:
        """Build safety analysis string."""
        if not safety_results["findings"]:
            return "No safety concerns"

        severity_counts = {}
        for issue in safety_results["issues"]:
            severity_counts[issue.severity.value] = \
                severity_counts.get(issue.severity.value, 0) + 1

        return f"Safety concerns: {', '.join(f'{k}: {v}' for k, v in severity_counts.items())}"

    def _load_system_prompt(self) -> str:
        """Load the system prompt from file."""
        try:
            with open(self.system_prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            return "You are the Ethics Advisor. Review for bias, PII, compliance, and safety."


# =============================================================================
# Convenience Functions
# =============================================================================

def create_council_chair(
    system_prompt_path: str = "config/agents/council/CLAUDE.md",
    model: str = "claude-3-5-opus-20240507",
) -> CouncilChairAgent:
    """Create a configured Council Chair agent."""
    return CouncilChairAgent(
        system_prompt_path=system_prompt_path,
        model=model,
    )


def create_quality_arbiter(
    system_prompt_path: str = "config/agents/council/CLAUDE.md",
    model: str = "claude-3-5-opus-20240507",
) -> QualityArbiterAgent:
    """Create a configured Quality Arbiter agent."""
    return QualityArbiterAgent(
        system_prompt_path=system_prompt_path,
        model=model,
    )


def create_ethics_advisor(
    system_prompt_path: str = "config/agents/council/CLAUDE.md",
    model: str = "claude-3-5-opus-20240507",
) -> EthicsAdvisorAgent:
    """Create a configured Ethics Advisor agent."""
    return EthicsAdvisorAgent(
        system_prompt_path=system_prompt_path,
        model=model,
    )
