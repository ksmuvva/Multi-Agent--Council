"""
Verifier Subagent (Hallucination Guard)

Scans every factual claim in proposed output for verification,
confidence scoring, and fabrication risk assessment.
"""

import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from urllib.parse import urlparse

from src.schemas.verifier import (
    VerificationReport,
    Claim,
    ClaimBatch,
    VerificationStatus,
    FabricationRisk,
)


@dataclass
class ClaimExtraction:
    """Result of extracting claims from content."""
    claims: List[str]
    claim_locations: List[Dict[str, int]]  # {"claim": str, "start": int, "end": int}
    total_claims: int


class VerifierAgent:
    """
    The Verifier detects factual errors and hallucinations.

    Key responsibilities:
    - Extract factual claims from output
    - Trace each claim to a source
    - Assign confidence (1-10 scale)
    - Rate fabrication risk
    - Flag claims needing correction
    - Work with SMEs for domain verification
    """

    def __init__(
        self,
        system_prompt_path: str = "config/agents/verifier/CLAUDE.md",
        model: str = "claude-3-5-opus-20240507",
        max_turns: int = 30,
    ):
        """
        Initialize the Verifier agent.

        Args:
            system_prompt_path: Path to system prompt file
            model: Model to use for verification (opus for accuracy)
            max_turns: Maximum conversation turns
        """
        self.system_prompt_path = system_prompt_path
        self.model = model
        self.max_turns = max_turns
        self.system_prompt = self._load_system_prompt()

        # Tool executor for WebSearch verification (set by orchestrator when available)
        self._tool_executor = None

        # Patterns that indicate factual claims
        self.claim_patterns = [
            r'\b\d{4}\b',  # Years
            r'\b\d{1,3}(?:,\d{3})*\b',  # Numbers with commas
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,4}',  # Dates
            r'\b\d+\s*(?:percent|%|degrees?|°)\b',  # Measurements
            r'\b(?:https?://|www\.)\S+\b',  # URLs
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Proper nouns (potential)
        ]

        # Keywords that indicate claims requiring verification
        self.claim_keywords = [
            "according to", "based on", "research shows",
            "studies have", "it is known", "typically",
            "usually", "generally", "approximately",
            "estimated", "around", "about",
        ]

    def verify(
        self,
        content: str,
        sources: Optional[List[str]] = None,
        sme_verifications: Optional[Dict[str, str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> VerificationReport:
        """
        Verify factual claims in the content.

        Args:
            content: The content to verify
            sources: Optional list of sources/references used
            sme_verifications: Optional dict of SME verifications
            context: Additional verification context

        Returns:
            VerificationReport with all claims and their verification status
        """
        # Step 1: Extract claims
        extraction = self._extract_claims(content)

        # Step 2: Verify each claim
        verified_claims = []
        for claim in extraction.claims:
            claim_verification = self._verify_claim(
                claim,
                content,
                sources,
                sme_verifications
            )
            verified_claims.append(claim_verification)

        # Step 3: Count by status
        verified_count = sum(
            1 for c in verified_claims
            if c.status == VerificationStatus.VERIFIED
        )
        unverified_count = sum(
            1 for c in verified_claims
            if c.status == VerificationStatus.UNVERIFIED
        )
        contradicted_count = sum(
            1 for c in verified_claims
            if c.status == VerificationStatus.CONTRADICTED
        )
        fabricated_count = sum(
            1 for c in verified_claims
            if c.status == VerificationStatus.FABRICATED
        )

        # Step 4: Calculate overall reliability
        total_claims = len(verified_claims)
        overall_reliability = verified_count / total_claims if total_claims > 0 else 0.5

        # Step 5: Get flagged claims (need correction)
        flagged_claims = [
            c for c in verified_claims
            if c.confidence < 7 or c.fabrication_risk != FabricationRisk.LOW
        ]

        # Step 6: Generate recommended corrections
        recommended_corrections = self._generate_corrections(flagged_claims)

        # Step 7: Build verification summary
        verification_summary = self._build_summary(
            total_claims, verified_count, unverified_count,
            contradicted_count, fabricated_count, overall_reliability
        )

        return VerificationReport(
            total_claims_checked=total_claims,
            claims=verified_claims,
            verified_claims=verified_count,
            unverified_claims=unverified_count,
            contradicted_claims=contradicted_count,
            fabricated_claims=fabricated_count,
            overall_reliability=round(overall_reliability, 2),
            pass_threshold=0.7,
            verdict="PASS" if overall_reliability >= 0.7 else "FAIL",
            flagged_claims=flagged_claims,
            recommended_corrections=recommended_corrections,
            verification_summary=verification_summary,
        )

    # ========================================================================
    # Claim Extraction
    # ========================================================================

    def _extract_claims(self, content: str) -> ClaimExtraction:
        """Extract factual claims from content."""
        claims = []
        claim_locations = []

        # Split content into sentences
        sentences = re.split(r'[.!?]+', content)

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # Too short to be a claim
                continue

            # Check if sentence contains claim indicators
            if self._is_claim(sentence):
                claims.append(sentence)
                claim_locations.append({
                    "claim": sentence,
                    "start": content.find(sentence) if sentence in content else 0,
                    "end": content.find(sentence) + len(sentence) if sentence in content else len(sentence),
                })

        return ClaimExtraction(
            claims=claims,
            claim_locations=claim_locations,
            total_claims=len(claims),
        )

    def _is_claim(self, sentence: str) -> bool:
        """Check if a sentence contains a factual claim."""
        sentence_lower = sentence.lower()

        # Check for claim keywords
        for keyword in self.claim_keywords:
            if keyword in sentence_lower:
                return True

        # Check for claim patterns
        for pattern in self.claim_patterns:
            if re.search(pattern, sentence):
                return True

        # Check for specific factual indicators
        if any(word in sentence_lower for word in ["is", "are", "was", "were", "will"]):
            # Has a verb - could be a claim
            return True

        return False

    # ========================================================================
    # Claim Verification
    # ========================================================================

    def _verify_claim(
        self,
        claim: str,
        content: str,
        sources: Optional[List[str]] = None,
        sme_verifications: Optional[Dict[str, str]] = None
    ) -> Claim:
        """
        Verify a single claim.

        Returns a Claim with verification status, confidence, and risk.
        """
        # Check SME verification first (highest confidence)
        if sme_verifications:
            for sme, verification in sme_verifications.items():
                if self._claim_matches_domain(claim, sme):
                    return Claim(
                        claim_text=claim,
                        confidence=10,  # SME verified = max confidence
                        fabrication_risk=FabricationRisk.LOW,
                        source=f"SME: {sme}",
                        verification_method="Domain expert verification",
                        status=VerificationStatus.VERIFIED,
                        domain_verified=True,
                        sme_verifier=sme,
                    )

        # Check against provided sources
        if sources:
            source_verification = self._verify_against_sources(claim, sources)
            if source_verification["found"]:
                return Claim(
                    claim_text=claim,
                    confidence=source_verification["confidence"],
                    fabrication_risk=source_verification["risk"],
                    source=source_verification["source"],
                    verification_method="Source cross-reference",
                    status=source_verification["status"],
                )

        # Perform verification based on claim type
        verification_result = self._verify_claim_by_type(claim, content)

        return Claim(
            claim_text=claim,
            confidence=verification_result["confidence"],
            fabrication_risk=verification_result["risk"],
            source=verification_result.get("source"),
            verification_method=verification_result["method"],
            status=verification_result["status"],
        )

    def _claim_matches_domain(self, claim: str, sme_domain: str) -> bool:
        """Check if a claim relates to an SME's domain."""
        claim_lower = claim.lower()
        domain_lower = sme_domain.lower()

        # Simple keyword matching
        domain_keywords = domain_lower.split()
        return any(kw in claim_lower for kw in domain_keywords)

    def _verify_against_sources(
        self,
        claim: str,
        sources: List[str]
    ) -> Dict[str, Any]:
        """Verify claim against provided sources."""
        claim_words = set(claim.lower().split())

        for source in sources:
            source_lower = source.lower()
            # Check if claim appears in source
            if any(word in source_lower for word in claim_words if len(word) > 3):
                # Found matching content
                return {
                    "found": True,
                    "confidence": 9,  # Found in source = high confidence
                    "risk": FabricationRisk.LOW,
                    "source": source,
                    "status": VerificationStatus.VERIFIED,
                }

        return {"found": False}

    def _verify_claim_by_type(
        self,
        claim: str,
        content: str
    ) -> Dict[str, Any]:
        """Verify a claim based on its type."""
        claim_lower = claim.lower()

        # Date/Time claims
        if re.search(r'\b\d{4}\b', claim):
            return self._verify_date_claim(claim)

        # URL claims
        if re.search(r'https?://', claim_lower):
            return self._verify_url_claim(claim)

        # Measurement claims
        if re.search(r'\d+\s*%|\d+\s*degrees?', claim_lower):
            return self._verify_measurement_claim(claim)

        # General factual claims
        return self._verify_general_claim(claim, content)

    def _verify_date_claim(self, claim: str) -> Dict[str, Any]:
        """Verify a claim containing a date."""
        # Extract year
        year_match = re.search(r'\b(\d{4})\b', claim)
        if year_match:
            year = int(year_match.group(1))
            current_year = datetime.now().year

            # Plausibility check
            if year > current_year + 1:
                return {
                    "confidence": 2,
                    "risk": FabricationRisk.HIGH,
                    "method": "Date plausibility check",
                    "status": VerificationStatus.UNVERIFIED,
                    "correction": f"Year {year} is in the future",
                }
            elif year < 1900:
                return {
                    "confidence": 5,
                    "risk": FabricationRisk.MEDIUM,
                    "method": "Date plausibility check",
                    "status": VerificationStatus.UNVERIFIED,
                }

        # Default verification for dates
        return {
            "confidence": 6,
            "risk": FabricationRisk.MEDIUM,
            "method": "Date extraction (unverified)",
            "status": VerificationStatus.UNVERIFIED,
        }

    def _verify_url_claim(self, claim: str) -> Dict[str, Any]:
        """Verify a claim containing a URL."""
        # Extract URL
        url_match = re.search(r'(https?://[^\s]+)', claim)
        if url_match:
            url = url_match.group(1)

            # Basic URL validation
            if not url.startswith(('http://', 'https://')):
                return {
                    "confidence": 1,
                    "risk": FabricationRisk.HIGH,
                    "method": "URL format validation",
                    "status": VerificationStatus.UNVERIFIED,
                    "correction": "Invalid URL format",
                }

            # Check domain
            parsed = urlparse(url)
            if not parsed.netloc:
                return {
                    "confidence": 3,
                    "risk": FabricationRisk.HIGH,
                    "method": "URL structure check",
                    "status": VerificationStatus.UNVERIFIED,
                    "correction": "Malformed URL",
                }

        # Default for URLs
        return {
            "confidence": 7,
            "risk": FabricationRisk.MEDIUM,
            "method": "URL extraction",
            "status": VerificationStatus.UNVERIFIED,
        }

    def _verify_measurement_claim(self, claim: str) -> Dict[str, Any]:
        """Verify a claim containing measurements."""
        # Check for obviously wrong values
        if "100%" in claim and "0%" in claim:
            return {
                "confidence": 2,
                "risk": FabricationRisk.HIGH,
                "method": "Logical consistency check",
                "status": VerificationStatus.CONTRADICTED,
                "correction": "Cannot be both 100% and 0%",
            }

        if re.search(r'\d{3,}\s*%', claim):  # Over 100%
            return {
                "confidence": 3,
                "risk": FabricationRisk.HIGH,
                "method": "Range validation",
                "status": VerificationStatus.UNVERIFIED,
                "correction": "Percentage exceeds 100%",
            }

        return {
            "confidence": 6,
            "risk": FabricationRisk.MEDIUM,
            "method": "Measurement extraction",
            "status": VerificationStatus.UNVERIFIED,
        }

    def _verify_general_claim(self, claim: str, content: str) -> Dict[str, Any]:
        """Verify a general factual claim.

        Attempts WebSearch verification when a tool executor is available.
        Falls back to heuristic pattern analysis for offline operation.
        """
        claim_lower = claim.lower()

        # Attempt real WebSearch verification if tool executor is available
        if hasattr(self, '_tool_executor') and self._tool_executor:
            try:
                raw = self._tool_executor("WebSearch", {"query": claim, "max_results": 3})
                if isinstance(raw, list) and raw:
                    # Check if any result corroborates the claim
                    snippets = " ".join(r.get("snippet", "") for r in raw).lower()
                    claim_words = set(claim_lower.split())
                    snippet_words = set(snippets.split())
                    overlap = len(claim_words & snippet_words) / max(len(claim_words), 1)

                    if overlap > 0.5:
                        return {
                            "confidence": 8,
                            "risk": FabricationRisk.LOW,
                            "method": "WebSearch corroboration",
                            "status": VerificationStatus.VERIFIED,
                        }
                    elif overlap > 0.2:
                        return {
                            "confidence": 5,
                            "risk": FabricationRisk.MEDIUM,
                            "method": "WebSearch partial match",
                            "status": VerificationStatus.UNVERIFIED,
                        }
                    else:
                        return {
                            "confidence": 3,
                            "risk": FabricationRisk.HIGH,
                            "method": "WebSearch no corroboration",
                            "status": VerificationStatus.UNVERIFIED,
                        }
            except Exception:
                pass  # Fall through to heuristic analysis

        # Heuristic: check for common hallucination patterns
        hallucination_patterns = [
            ("demonstrated", "demonstrates"),
            ("obviously", "obvious"),
            ("clearly", "clear"),
            ("everyone knows", "everyone knows"),
            ("it is well known", "well-known"),
        ]

        for pattern, _marker in hallucination_patterns:
            if pattern in claim_lower:
                return {
                    "confidence": 4,
                    "risk": FabricationRisk.MEDIUM,
                    "method": "Hallucination pattern detection",
                    "status": VerificationStatus.UNVERIFIED,
                }

        # Check if the claim is supported by the content itself
        content_lower = content.lower() if content else ""
        claim_words = set(claim_lower.split()) - {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "for"}
        if content_lower:
            content_words = set(content_lower.split())
            overlap = len(claim_words & content_words) / max(len(claim_words), 1)
            if overlap > 0.6:
                return {
                    "confidence": 7,
                    "risk": FabricationRisk.LOW,
                    "method": "Content cross-reference",
                    "status": VerificationStatus.UNVERIFIED,
                }

        return {
            "confidence": 5,
            "risk": FabricationRisk.MEDIUM,
            "method": "General claim analysis",
            "status": VerificationStatus.UNVERIFIED,
        }

    # ========================================================================
    # Report Generation
    # ========================================================================

    def _generate_corrections(self, flagged_claims: List[Claim]) -> List[str]:
        """Generate recommended corrections for flagged claims."""
        corrections = []

        for claim in flagged_claims[:5]:  # Limit to top 5
            correction = claim.correction

            if not correction:
                if claim.status == VerificationStatus.FABRICATED:
                    correction = f"Remove or provide source for: {claim.claim_text[:50]}..."
                elif claim.status == VerificationStatus.UNVERIFIED:
                    correction = f"Add verification for: {claim.claim_text[:50]}..."
                elif claim.status == VerificationStatus.CONTRADICTED:
                    correction = f"Resolve contradiction: {claim.claim_text[:50]}..."

            if correction:
                corrections.append(correction)

        return corrections

    def _build_summary(
        self,
        total: int,
        verified: int,
        unverified: int,
        contradicted: int,
        fabricated: int,
        reliability: float
    ) -> str:
        """Build a verification summary."""
        parts = [
            f"Verified {total} claims: {verified} verified, "
            f"{unverified} unverified, {contradicted} contradicted, "
            f"{fabricated} potentially fabricated."
        ]

        if reliability >= 0.8:
            parts.append(" High confidence in output.")
        elif reliability >= 0.6:
            parts.append(" Medium confidence - some claims need verification.")
        else:
            parts.append(" Low confidence - significant verification concerns.")

        return "".join(parts)

    def _load_system_prompt(self) -> str:
        """Load the system prompt from file."""
        try:
            with open(self.system_prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            return "You are the Verifier. Detect factual errors and hallucinations."


# =============================================================================
# Convenience Functions
# =============================================================================

def create_verifier(
    system_prompt_path: str = "config/agents/verifier/CLAUDE.md",
    model: str = "claude-3-5-opus-20240507",
) -> VerifierAgent:
    """Create a configured Verifier agent."""
    return VerifierAgent(
        system_prompt_path=system_prompt_path,
        model=model,
    )
