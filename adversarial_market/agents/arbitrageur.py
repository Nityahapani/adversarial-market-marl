"""
Latency Arbitrageur Agent — standalone wrapper.

The arbitrageur exploits temporal asymmetry between:
  1. Information arrival in the order flow
  2. The market maker's belief update delay
  3. Consequent quote staleness

It acts on the fastest timescale of the three agents, placing it as the
mechanism that prevents the market maker from adopting trivially safe
static strategies (always-wide spread). By sniping stale quotes, the
arbitrageur forces the market maker to continuously solve the inference
problem rather than hiding behind a permanent wide spread.

Theoretical role: introduces the second dimension of competition (time),
ensuring the equilibrium is a genuine three-way saddle point rather than
a degenerate two-agent game.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from adversarial_market.agents.base_agent import BaseAgent
from adversarial_market.networks.actor_critic import ArbitrageActor


class ArbitrageAgent(BaseAgent):
    """
    Latency arbitrageur.

    Exploits quote staleness and belief-update lag to extract profit
    from the market maker. Its presence prevents degenerate equilibria.

    Action interpretation:
        a > +0.33  → lift best ask (buy, expecting upward correction)
        a < -0.33  → hit best bid  (sell, expecting downward correction)
        otherwise  → do nothing

    Args:
        config:  agents.arbitrageur config section.
        net_cfg: networks config section.
        obs_dim: Actor observation dimension.
        device:  Torch device string.
    """

    AGENT_ID = 2
    ACTION_THRESHOLD = 0.33

    def __init__(
        self,
        config: Dict[str, Any],
        net_cfg: Dict[str, Any],
        obs_dim: int,
        device: str = "cpu",
    ) -> None:
        super().__init__(
            agent_id=self.AGENT_ID,
            name="arbitrageur",
            config=config,
            device=device,
        )
        self.obs_dim = obs_dim
        self.max_position: int = config["max_position_lots"]
        self.hold_period: int = config["hold_period_steps"]
        self.delta: float = config["delta_belief_lag"]
        self.latency_advantage_ms: float = config["latency_advantage_ms"]

        # Actor network
        arb_cfg = net_cfg["arb_actor"]
        self.actor = ArbitrageActor(
            obs_dim=obs_dim,
            hidden_dims=arb_cfg["hidden_dims"],
            activation=arb_cfg["activation"],
        )
        self.register_network(self.actor)

        # Episode state
        self._position: int = 0
        self._pnl: float = 0.0
        self._n_snipes: int = 0
        self._snipe_profits: List[float] = []
        self._belief_lag_history: List[float] = []

    # ── BaseAgent interface ───────────────────────────────────────────────

    def build_networks(self) -> None:
        pass

    def act(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, float, float]:
        obs_t = self.to_tensor(obs).unsqueeze(0)
        with torch.no_grad():
            action, log_prob, entropy = self.actor(obs_t)
        return (
            action.squeeze(0).cpu().numpy(),
            float(log_prob.item()),
            float(entropy.item()),
        )

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        return {}

    # ── Staleness signal ──────────────────────────────────────────────────

    @staticmethod
    def compute_quote_staleness(
        best_bid: Optional[float],
        best_ask: Optional[float],
        mid_price: float,
        tick_size: float,
        fair_spread_ticks: float = 2.0,
    ) -> float:
        """
        Compute a staleness score ∈ [0, 5] for current MM quotes.

        Staleness = how much wider the current spread is versus
        the fair (competitive) spread. Wider than fair → MM has not
        yet adjusted quotes to new information → sniping opportunity.

        Args:
            best_bid:         Current best bid price (or None).
            best_ask:         Current best ask price (or None).
            mid_price:        Current mid-point price.
            tick_size:        Minimum price increment.
            fair_spread_ticks: Expected competitive spread in ticks.

        Returns:
            Staleness score clipped to [0, 5].
        """
        if best_bid is None or best_ask is None:
            return 0.0
        current_spread = best_ask - best_bid
        fair_spread = fair_spread_ticks * tick_size
        staleness = (current_spread - fair_spread) / max(fair_spread, 1e-8)
        return float(np.clip(staleness, 0.0, 5.0))

    def should_trade(self, action_scalar: float) -> str:
        """Interpret scalar action as a trading decision."""
        if action_scalar > self.ACTION_THRESHOLD:
            return "buy"
        elif action_scalar < -self.ACTION_THRESHOLD:
            return "sell"
        return "hold"

    # ── Episode bookkeeping ───────────────────────────────────────────────

    def reset_episode(self) -> None:
        self._position = 0
        self._pnl = 0.0
        self._n_snipes = 0
        self._snipe_profits = []
        self._belief_lag_history = []

    def record_snipe(self, profit: float, belief_lag: float) -> None:
        self._position = 0  # flat immediately after snipe (hold_period=1)
        self._pnl += profit
        self._n_snipes += 1
        self._snipe_profits.append(profit)
        self._belief_lag_history.append(belief_lag)

    @property
    def episode_pnl(self) -> float:
        return self._pnl

    @property
    def n_snipes(self) -> int:
        return self._n_snipes

    @property
    def avg_snipe_profit(self) -> float:
        if not self._snipe_profits:
            return 0.0
        return float(np.mean(self._snipe_profits))

    @property
    def avg_belief_lag_exploited(self) -> float:
        if not self._belief_lag_history:
            return 0.0
        return float(np.mean(self._belief_lag_history))

    def __repr__(self) -> str:
        return (
            f"ArbitrageAgent("
            f"latency={self.latency_advantage_ms}ms, "
            f"max_pos={self.max_position}, "
            f"snipes={self._n_snipes}, "
            f"pnl={self._pnl:.4f})"
        )
