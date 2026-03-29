"""
Market Maker Agent — standalone wrapper.

Wraps the MarketMakerActor and BeliefTransformer into a self-contained
agent that provides liquidity while performing real-time sequential
inference on incoming order flow.

The market maker is simultaneously:
  1. A liquidity provider (profit motive via spread capture)
  2. A Bayesian classifier (infer informed vs noise flow)
  3. An adversary (penalise execution agent for detectable patterns)

Its belief b_t is the central observable that determines how much
information the execution agent has leaked.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List, Tuple

import numpy as np
import torch

from adversarial_market.agents.base_agent import BaseAgent
from adversarial_market.networks.actor_critic import MarketMakerActor
from adversarial_market.networks.belief_transformer import BeliefTransformer

# Flow event feature dimension: (side, price_rel, size_rel, order_type, time_frac)
FLOW_DIM = 5


class MarketMakerAgent(BaseAgent):
    """
    Adaptive market maker with embedded sequential belief inference.

    Maintains a rolling window of order flow observations and passes
    them through the BeliefTransformer to compute b_t = P(informed | F_t).

    Args:
        config:   agents.market_maker config section.
        net_cfg:  networks config section.
        obs_dim:  Actor observation dimension.
        device:   Torch device string.
    """

    AGENT_ID = 1

    def __init__(
        self,
        config: Dict[str, Any],
        net_cfg: Dict[str, Any],
        obs_dim: int,
        device: str = "cpu",
    ) -> None:
        super().__init__(
            agent_id=self.AGENT_ID,
            name="market_maker",
            config=config,
            device=device,
        )
        self.obs_dim = obs_dim
        self.max_spread_ticks: int = config["max_spread_ticks"]
        self.max_inventory: int = config["max_inventory_lots"]
        self.max_quote_size: int = config["max_quote_size_lots"]
        self.obs_history: int = config["observation_history"]
        self.alpha: float = config["alpha_adverse_selection"]
        self.beta: float = config["beta_entropy_reg"]
        self.gamma: float = config["gamma_belief_accuracy"]

        # Actor network
        mm_cfg = net_cfg["mm_actor"]
        self.actor = MarketMakerActor(
            obs_dim=obs_dim,
            hidden_dims=mm_cfg["hidden_dims"],
            activation=mm_cfg["activation"],
        )
        self.register_network(self.actor)

        # Belief transformer
        bt_cfg = net_cfg["belief_transformer"]
        self.belief_transformer = BeliefTransformer(
            input_dim=FLOW_DIM,
            d_model=bt_cfg["d_model"],
            n_heads=bt_cfg["n_heads"],
            n_layers=bt_cfg["n_layers"],
            d_ff=bt_cfg["d_ff"],
            dropout=bt_cfg["dropout"],
            max_seq_len=bt_cfg["max_seq_len"],
        )
        self.register_network(self.belief_transformer)

        # Rolling flow window for belief updating
        self._flow_window: Deque[np.ndarray] = deque(maxlen=bt_cfg["max_seq_len"])
        self._belief: float = 0.5
        self._belief_history: List[float] = []

        # Episode tracking
        self._pnl: float = 0.0
        self._adverse_selection_total: float = 0.0
        self._n_fills: int = 0

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

    # ── Belief management ─────────────────────────────────────────────────

    def observe_flow_event(
        self,
        side: float,
        price_rel: float,
        size_rel: float,
        order_type: float,
        time_frac: float,
    ) -> None:
        """Add a new order flow event to the rolling window."""
        feat = np.array([side, price_rel, size_rel, order_type, time_frac], dtype=np.float32)
        self._flow_window.append(feat)

    @torch.no_grad()
    def update_belief(self) -> float:
        """
        Recompute b_t from the current flow window.
        Returns the updated belief scalar.
        """
        if len(self._flow_window) == 0:
            return self._belief

        seq = np.array(list(self._flow_window), dtype=np.float32)
        seq_t = torch.as_tensor(seq, dtype=torch.float32, device=self.device).unsqueeze(0)
        self._belief = self.belief_transformer.update_belief(seq_t)
        self._belief_history.append(self._belief)
        return self._belief

    @property
    def belief(self) -> float:
        return self._belief

    @property
    def belief_history(self) -> List[float]:
        return self._belief_history.copy()

    def implied_spread_multiplier(self) -> float:
        """
        Returns a multiplier in [1, 3] that widens the spread
        proportionally to the current belief b_t.

        Rational MM: spread ∝ adverse selection risk ∝ P(informed).
        Derived from Glosten-Milgrom (1985):
            spread = 2 * b_t * E[|v - p| | informed] / (1 - b_t + ε)
        We approximate with a simple linear scale.
        """
        return 1.0 + 2.0 * self._belief

    # ── Episode bookkeeping ───────────────────────────────────────────────

    def reset_episode(self) -> None:
        self._flow_window.clear()
        self._belief = 0.5
        self._belief_history = []
        self._pnl = 0.0
        self._adverse_selection_total = 0.0
        self._n_fills = 0

    def record_fill(self, pnl_delta: float, adverse_sel: float) -> None:
        self._pnl += pnl_delta
        self._adverse_selection_total += adverse_sel
        self._n_fills += 1

    @property
    def episode_pnl(self) -> float:
        return self._pnl

    @property
    def episode_adverse_selection(self) -> float:
        return self._adverse_selection_total
