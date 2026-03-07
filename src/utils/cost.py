"""
Cost Tracking - Token Usage and Budget Management

Tracks token usage, costs, and enforces budget limits for agent operations.
Provides per-session, daily, and cumulative cost tracking.
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from collections import defaultdict

from .logging import get_logger, log_cost


# =============================================================================
# Pricing Configuration
# =============================================================================

class ModelPricing(str, Enum):
    """Available Claude models with pricing."""
    HAIKU = "claude-3-5-haiku-20250101"
    SONNET = "claude-3-5-sonnet-20241022"
    OPUS = "claude-3-5-opus-20240507"


# USD per 1M tokens (as of 2025)
MODEL_COSTS = {
    ModelPricing.HAIKU: {"input": 0.25, "output": 1.25},
    ModelPricing.SONNET: {"input": 3.0, "output": 15.0},
    ModelPricing.OPUS: {"input": 15.0, "output": 75.0},
}


# Token costs per agent turn (estimates)
AGENT_TOKEN_COSTS = {
    "Analyst": {"input": 1000, "output": 800},
    "Planner": {"input": 800, "output": 600},
    "Clarifier": {"input": 600, "output": 400},
    "Researcher": {"input": 2000, "output": 1200},  # May use WebSearch
    "Executor": {"input": 1500, "output": 1000},
    "CodeReviewer": {"input": 1200, "output": 800},
    "Formatter": {"input": 500, "output": 300},
    "Verifier": {"input": 1000, "output": 700},
    "Critic": {"input": 1200, "output": 800},
    "Reviewer": {"input": 800, "output": 600},
    "MemoryCurator": {"input": 400, "output": 300},
    "CouncilChair": {"input": 600, "output": 400},
    "QualityArbiter": {"input": 600, "output": 400},
    "EthicsAdvisor": {"input": 800, "output": 500},
}


# =============================================================================
# Cost Tracking Data Structures
# =============================================================================

@dataclass
class TokenUsage:
    """Token usage for a single operation."""
    input_tokens: int
    output_tokens: int
    total_tokens: int

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


@dataclass
class OperationCost:
    """Cost tracking for a single operation."""
    operation_id: str
    agent_name: str
    model: ModelPricing
    token_usage: TokenUsage
    cost_usd: float
    timestamp: datetime
    phase: str = ""
    tier: int = 2


@dataclass
class BudgetState:
    """Current budget state."""
    max_budget_usd: float
    spent_usd: float
    remaining_usd: float

    @property
    def utilization_pct(self) -> float:
        """Budget utilization as percentage."""
        if self.max_budget_usd <= 0:
            return 0.0
        return (self.spent_usd / self.max_budget_usd) * 100

    @property
    def is_exceeded(self) -> bool:
        """Check if budget is exceeded."""
        return self.spent_usd > self.max_budget_usd

    @property
    def is_warning(self) -> bool:
        """Check if budget is at warning level (>80%)."""
        return self.utilization_pct > 80


@dataclass
class SessionCosts:
    """Cost tracking for a session."""
    session_id: str
    start_time: datetime
    operations: List[OperationCost] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        """Total tokens used in session."""
        return sum(op.token_usage.total_tokens for op in self.operations)

    @property
    def total_cost(self) -> float:
        """Total cost in USD."""
        return sum(op.cost_usd for op in self.operations)

    @property
    def input_tokens(self) -> int:
        """Total input tokens."""
        return sum(op.token_usage.input_tokens for op in self.operations)

    @property
    def output_tokens(self) -> int:
        """Total output tokens."""
        return sum(op.token_usage.output_tokens for op in self.operations)

    @property
    def duration(self) -> timedelta:
        """Session duration."""
        return datetime.now() - self.start_time

    def get_costs_by_agent(self) -> Dict[str, float]:
        """Get costs broken down by agent."""
        costs = defaultdict(float)
        for op in self.operations:
            costs[op.agent_name] += op.cost_usd
        return dict(costs)

    def get_costs_by_model(self) -> Dict[str, float]:
        """Get costs broken down by model."""
        costs = defaultdict(float)
        for op in self.operations:
            costs[op.model.value] += op.cost_usd
        return dict(costs)


# =============================================================================
# Budget Enforcement Callbacks
# =============================================================================

class BudgetExceededError(Exception):
    """Raised when budget is exceeded."""
    def __init__(self, spent: float, budget: float):
        self.spent = spent
        self.budget = budget
        super().__init__(f"Budget exceeded: ${spent:.4f} / ${budget:.2f}")


class BudgetWarning(Exception):
    """Raised when budget is at warning level (>80%)."""
    def __init__(self, spent: float, budget: float, pct: float):
        self.spent = spent
        self.budget = budget
        self.pct = pct
        super().__init__(f"Budget warning: {pct:.1f}% used (${spent:.2f} / ${budget:.2f})")


# =============================================================================
# Cost Tracker
# =============================================================================

class CostTracker:
    """
    Tracks costs and enforces budget limits.

    Thread-safe singleton that tracks costs across operations.
    """

    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._sessions: Dict[str, SessionCosts] = {}
        self._daily_costs: Dict[str, float] = {}  # date string -> cost
        self._callbacks: List[Callable] = []
        self._enforcement_enabled = True
        self._logger = get_logger("cost_tracker")

        self._initialized = True

    def register_callback(self, callback: Callable[[BudgetState], None]) -> None:
        """
        Register a callback to be called on budget state changes.

        Args:
            callback: Function that takes BudgetState
        """
        self._callbacks.append(callback)

    def _notify_callbacks(self, state: BudgetState) -> None:
        """Notify all registered callbacks of budget state change."""
        for callback in self._callbacks:
            try:
                callback(state)
            except Exception as e:
                self._logger.error("Failed to notify callback", error=str(e))

    def create_session(
        self,
        session_id: str,
        max_budget_usd: float = 10.0,
    ) -> SessionCosts:
        """
        Create a new cost tracking session.

        Args:
            session_id: Unique session identifier
            max_budget_usd: Maximum budget for this session

        Returns:
            SessionCosts instance
        """
        with self._lock:
            session = SessionCosts(
                session_id=session_id,
                start_time=datetime.now(),
            )
            self._sessions[session_id] = session

            self._logger.info(
                "session_created",
                session_id=session_id,
                max_budget=max_budget_usd,
            )

            return session

    def track_operation(
        self,
        session_id: str,
        agent_name: str,
        model: ModelPricing,
        input_tokens: int,
        output_tokens: int,
        phase: str = "",
        tier: int = 2,
    ) -> OperationCost:
        """
        Track a single operation's cost.

        Args:
            session_id: Session identifier
            agent_name: Name of the agent
            model: Model used
            input_tokens: Input tokens consumed
            output_tokens: Output tokens consumed
            phase: Phase in which operation occurred
            tier: Tier level

        Returns:
            OperationCost instance

        Raises:
            BudgetExceededError: If budget is exceeded and enforcement enabled
            BudgetWarning: If budget is at warning level
        """
        with self._lock:
            session = self._sessions.get(session_id)

            if not session:
                # Auto-create session
                session = self.create_session(session_id)

            # Calculate cost
            pricing = MODEL_COSTS[model]
            cost_usd = (
                (input_tokens / 1_000_000) * pricing["input"] +
                (output_tokens / 1_000_000) * pricing["output"]
            )

            # Create operation record
            operation = OperationCost(
                operation_id=f"op_{session_id}_{int(time.time() * 1000000)}",
                agent_name=agent_name,
                model=model,
                token_usage=TokenUsage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=input_tokens + output_tokens,
                ),
                cost_usd=cost_usd,
                timestamp=datetime.now(),
                phase=phase,
                tier=tier,
            )

            session.operations.append(operation)

            # Update daily costs
            date_str = datetime.now().strftime("%Y-%m-%d")
            self._daily_costs[date_str] = self._daily_costs.get(date_str, 0) + cost_usd

            # Log the cost
            log_cost(
                tokens=operation.token_usage.total_tokens,
                cost_usd=cost_usd,
                model=model.value,
                operation=f"{agent_name}:{phase}",
            )

            # Check budget
            self._check_budget(session_id)

            return operation

    def get_session_state(self, session_id: str) -> Optional[SessionCosts]:
        """Get the cost state for a session."""
        return self._sessions.get(session_id)

    def get_budget_state(
        self,
        session_id: str,
        max_budget_usd: float = 10.0,
    ) -> BudgetState:
        """
        Get the current budget state for a session.

        Args:
            session_id: Session identifier
            max_budget_usd: Budget limit to check against

        Returns:
            BudgetState instance
        """
        session = self._sessions.get(session_id)

        spent = session.total_cost if session else 0.0

        return BudgetState(
            max_budget_usd=max_budget_usd,
            spent_usd=spent,
            remaining_usd=max(max_budget_usd - spent, 0),
        )

    def _check_budget(self, session_id: str) -> None:
        """
        Check budget and raise exceptions if needed.

        Args:
            session_id: Session to check

        Raises:
            BudgetExceededError: If budget exceeded
            BudgetWarning: If budget at warning level
        """
        if not self._enforcement_enabled:
            return

        session = self._sessions.get(session_id)
        if not session:
            return

        # Get max budget from session state or default
        # In real implementation, this would come from session config
        max_budget = 10.0  # Default

        state = self.get_budget_state(session_id, max_budget)

        # Notify callbacks
        self._notify_callbacks(state)

        # Check warning level
        if state.is_warning:
            self._logger.warning(
                "budget_warning",
                spent=state.spent_usd,
                budget=state.max_budget_usd,
                utilization=state.utilization_pct,
            )

        # Check exceeded
        if state.is_exceeded:
            self._logger.error(
                "budget_exceeded",
                spent=state.spent_usd,
                budget=state.max_budget_usd,
            )
            raise BudgetExceededError(state.spent_usd, state.max_budget_usd)

    def set_enforcement(self, enabled: bool) -> None:
        """Enable or disable budget enforcement."""
        self._enforcement_enabled = enabled
        self._logger.info("enforcement_changed", enabled=enabled)

    def get_daily_cost(self, date: Optional[datetime] = None) -> float:
        """
        Get total cost for a day.

        Args:
            date: Date to check (default: today)

        Returns:
            Total cost in USD
        """
        if date is None:
            date = datetime.now()

        date_str = date.strftime("%Y-%m-%d")
        return self._daily_costs.get(date_str, 0.0)

    def get_weekly_cost(self) -> Dict[str, float]:
        """
        Get cost breakdown for the past week.

        Returns:
            Dictionary mapping date strings to costs
        """
        weekly = {}

        for i in range(7):
            date = datetime.now() - timedelta(days=i)
            date_str = date.strftime("%Y-%m-%d")
            weekly[date_str] = self.get_daily_cost(date)

        return weekly

    def estimate_cost(
        self,
        agents: List[tuple[str, int]],
        tier: int = 2,
    ) -> Dict[str, Any]:
        """
        Estimate cost for a proposed operation.

        Args:
            agents: List of (agent_name, turns) tuples
            tier: Tier level (affects model selection)

        Returns:
            Cost estimate dictionary
        """
        # Model selection by tier
        tier_models = {
            1: ModelPricing.HAIKU,
            2: ModelPricing.SONNET,
            3: ModelPricing.OPUS,
            4: ModelPricing.OPUS,
        }

        model = tier_models.get(tier, ModelPricing.SONNET)

        # Calculate costs
        total_input = 0
        total_output = 0
        total_cost = 0.0
        agent_breakdown = []

        for agent_name, turns in agents:
            token_costs = AGENT_TOKEN_COSTS.get(agent_name, {"input": 1000, "output": 800})

            input_tokens = token_costs["input"] * turns
            output_tokens = token_costs["output"] * turns

            pricing = MODEL_COSTS[model]
            cost = (
                (input_tokens / 1_000_000) * pricing["input"] +
                (output_tokens / 1_000_000) * pricing["output"]
            )

            total_input += input_tokens
            total_output += output_tokens
            total_cost += cost

            agent_breakdown.append({
                "agent": agent_name,
                "turns": turns,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": round(cost, 4),
            })

        return {
            "tier": tier,
            "model": model.value,
            "total_tokens": total_input + total_output,
            "total_cost_usd": round(total_cost, 4),
            "agent_breakdown": agent_breakdown,
        }

    def clear_session(self, session_id: str) -> None:
        """Clear a session's cost tracking."""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                self._logger.info("session_cleared", session_id=session_id)

    def clear_all_sessions(self) -> None:
        """Clear all session cost tracking."""
        with self._lock:
            self._sessions.clear()
            self._logger.info("all_sessions_cleared")


# =============================================================================
# Global Instance
# =============================================================================

def get_cost_tracker() -> CostTracker:
    """Get the global cost tracker instance."""
    return CostTracker()


# =============================================================================
# Cost Context Managers
# =============================================================================

class CostLimit:
    """Context manager for cost-limited operations."""

    def __init__(
        self,
        session_id: str,
        max_budget_usd: float,
        tracker: Optional[CostTracker] = None,
    ):
        self.session_id = session_id
        self.max_budget_usd = max_budget_usd
        self.tracker = tracker or get_cost_tracker()

    def __enter__(self):
        self.tracker.create_session(self.session_id, self.max_budget_usd)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Session remains for inspection
        pass


# =============================================================================
# Utility Functions
# =============================================================================

def calculate_tokens_from_text(text: str) -> int:
    """
    Estimate token count from text (rough approximation).

    Args:
        text: Text to count tokens for

    Returns:
        Estimated token count
    """
    # Rough estimate: ~4 characters per token for English text
    return len(text) // 4


def calculate_max_turns_for_budget(
    budget_usd: float,
    agent_name: str,
    model: ModelPricing = ModelPricing.SONNET,
) -> int:
    """
    Calculate maximum turns for an agent given a budget.

    Args:
        budget_usd: Available budget
        agent_name: Agent to run
        model: Model to use

    Returns:
        Maximum number of turns possible
    """
    token_costs = AGENT_TOKEN_COSTS.get(agent_name, {"input": 1000, "output": 800})
    pricing = MODEL_COSTS[model]

    cost_per_turn = (
        (token_costs["input"] / 1_000_000) * pricing["input"] +
        (token_costs["output"] / 1_000_000) * pricing["output"]
    )

    if cost_per_turn <= 0:
        return 999  # Effectively unlimited

    return int(budget_usd / cost_per_turn)
