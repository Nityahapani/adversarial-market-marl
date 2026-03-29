"""
Global market state container — shared across all agents each step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from adversarial_market.environment.order import Fill, Side


@dataclass
class AgentState:
    """Per-agent mutable state."""

    agent_id: int
    inventory: int = 0  # current position in lots (signed)
    cash: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    n_orders_submitted: int = 0
    n_fills: int = 0
    total_filled_qty: int = 0
    avg_fill_price: float = 0.0
    _fill_value: float = 0.0  # running sum for avg fill computation

    def update_fill(self, price: float, qty: int, side: Side) -> None:
        signed_qty = qty if side == Side.BID else -qty
        self.inventory += signed_qty
        cash_delta = -price * qty if side == Side.BID else price * qty
        self.cash += cash_delta
        self.n_fills += 1
        self.total_filled_qty += qty
        self._fill_value += price * qty
        self.avg_fill_price = self._fill_value / self.total_filled_qty

    def mark_to_market(self, mid_price: float) -> None:
        self.unrealized_pnl = self.inventory * mid_price + self.cash

    def total_pnl(self) -> float:
        return self.realized_pnl + self.unrealized_pnl


@dataclass
class MarketState:
    """
    Complete market state snapshot passed to agents each step.

    Contains:
      - LOB snapshot (normalised numpy array)
      - Recent fills (order flow visible to market maker)
      - Per-agent state (inventory, PnL, etc.)
      - Mid price, spread, volatility estimates
      - Step counter and episode info
    """

    step: int = 0
    max_steps: int = 390

    # Price state
    mid_price: float = 100.0
    best_bid: Optional[float] = None
    best_ask: Optional[float] = None
    spread: Optional[float] = None
    last_trade_price: Optional[float] = None
    realized_volatility: float = 0.0

    # LOB snapshot: shape (n_price_levels * 4,) normalised
    lob_snapshot: np.ndarray = field(default_factory=lambda: np.zeros(80, dtype=np.float32))

    # Order flow: list of recent fills (agent id, side, price, qty, timestamp)
    recent_fills: List[Fill] = field(default_factory=list)

    # Per-agent state
    agent_states: Dict[int, AgentState] = field(default_factory=dict)

    # Market maker belief about execution agent
    mm_belief: float = 0.5  # b_t ∈ [0,1]: P(agent=informed | F_t)

    # Execution agent remaining inventory
    exec_remaining_inventory: int = 0
    exec_initial_inventory: int = 0

    # Price history for volatility calculation
    _mid_price_history: List[float] = field(default_factory=list)

    def update_price_history(self) -> None:
        self._mid_price_history.append(self.mid_price)
        if len(self._mid_price_history) > 100:
            self._mid_price_history.pop(0)
        if len(self._mid_price_history) >= 2:
            log_returns = np.diff(np.log(self._mid_price_history))
            self.realized_volatility = float(np.std(log_returns)) * np.sqrt(390)

    @property
    def time_remaining_frac(self) -> float:
        return 1.0 - self.step / max(self.max_steps, 1)

    @property
    def exec_completion_frac(self) -> float:
        if self.exec_initial_inventory == 0:
            return 1.0
        return 1.0 - self.exec_remaining_inventory / self.exec_initial_inventory

    def to_array(self) -> np.ndarray:
        """Compact numpy representation of global market state (for critic)."""
        return np.concatenate(
            [
                self.lob_snapshot,
                [
                    self.mid_price / 100.0,
                    self.spread / self.mid_price if self.spread else 0.0,
                    self.realized_volatility,
                    self.time_remaining_frac,
                    self.exec_completion_frac,
                    self.mm_belief,
                ],
            ]
        ).astype(np.float32)
