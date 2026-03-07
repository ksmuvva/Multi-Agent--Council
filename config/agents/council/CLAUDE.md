# Council Agents Configuration

This directory contains system prompts for the three Strategic Council agents.

## Domain Council Chair

Activated on Tier 3-4 tasks. Analyzes domain expertise needs and selects SME personas.

### Role
- Receives TaskIntelligenceReport from Analyst
- Identifies which domains require expertise
- Selects up to 3 SME personas from registry
- Specifies which skills each SME should load
- Defines interaction mode for each SME (advisor/co-executor/debater)

### Output
SMESelectionReport with:
- Selected SME personas (max 3)
- Skills for each SME
- Interaction mode per SME
- Reasoning for selection

---

## Quality Arbiter

Activated on Tier 4 tasks or as tiebreaker after 2 failed debate rounds.

### Role
- Sets quality acceptance criteria BEFORE execution (Tier 4)
- Acts as final tiebreaker for quality disputes
- Defines measurable pass/fail criteria
- Resolves disagreements between Verifier and Critic

### Output
- Pre-execution: QualityStandard with criteria
- Dispute resolution: QualityVerdict with binding decision

---

## Ethics & Safety Advisor

Activated on Tier 4 tasks or tasks involving sensitive content.

### Role
- Reviews final output for bias
- Checks for PII exposure
- Identifies compliance risks
- Flags safety concerns

### Output
EthicsReview with:
- Pass/fail verdict
- Flagged issues with severity
- Recommended remediations
- Block output until addressed if critical
