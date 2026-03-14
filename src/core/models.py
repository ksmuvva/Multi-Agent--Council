"""
Model Constants - Single Source of Truth for Claude Model IDs

All model references throughout the codebase should use these constants
rather than hardcoding model ID strings.
"""

# =============================================================================
# Current Claude Model IDs (updated 2025)
# =============================================================================

MODEL_HAIKU = "claude-haiku-4-5-20251001"
MODEL_SONNET = "claude-sonnet-4-20250514"
MODEL_OPUS = "claude-opus-4-20250514"

# =============================================================================
# Logical Aliases
# =============================================================================

MODEL_FAST = MODEL_HAIKU       # Low cost, fast responses
MODEL_BALANCED = MODEL_SONNET  # Good balance of quality and cost
MODEL_POWERFUL = MODEL_OPUS    # Highest capability

# =============================================================================
# Per-Tier Default Models
# =============================================================================

TIER_MODELS = {
    1: MODEL_FAST,      # Direct tier — speed matters
    2: MODEL_BALANCED,  # Standard tier
    3: MODEL_POWERFUL,  # Deep tier — quality matters
    4: MODEL_POWERFUL,  # Adversarial tier — needs best model
}
