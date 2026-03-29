"""
Execution Agent — standalone agent wrapper.

Wraps the ExecutionActor policy network and MINE estimator into a
self-contained agent object that can be used for evaluation, simulation,
and analysis independently of the full training loop.

The execution agent represents an institutional trader with a large
directional position to complete. It encodes its intent into order flow
while minimising statistical detectability.

Design pattern: thin wrapper — all heavy computation lives in the
network modules. The agent class handles observation processing,
action clipping, and episode-level bookkeeping.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from adversarial_market.agents.base_agent import BaseAgent
from adversarial_market.networks.actor_critic import ExecutionActor
from adversarial_market.networks.mine_estimator import MINEEstimator, PredictabilityPenalty


class ExecutionAgent(BaseAgent):
    """
    Institutional execution agent.

    Maintains:
        - Actor policy π_E(a | o_E)
        - MINE estimator for leakage measurement
        - Predictability penalty network
        - Episode-level tracking (inventory, fills, IS)

    Args:
        config:   Full agent config dict (agents.execution section).
        net_cfg:  Network architecture config dict (networks section).
        device:   Torch device string.
    """

    AGENT_ID = 0

    def __init__(
        self,
        config: Dict[str, Any],
        net_cfg: Dict[str, Any],
        obs_dim: int,
        device: str = "cpu",
    ) -> None:
        super().__init__(
            agent_id=self.AGENT_ID,
            name="execution",
            config=config,
            device=device,
        )
        self.obs_dim = obs_dim
        self.inventory: int = config["inventory_lots"]
        self.horizon: int = config["horizon"]
        self.max_order_size: int = config["max_order_size"]
        self.lambda_leakage: float = config["lambda_leakage"]
        self.mu_predictability: float = config["mu_predictability"]
        self.terminal_penalty: float = config["terminal_inventory_penalty"]

        # Networks
        actor_cfg = net_cfg["execution_actor"]
        self.actor = ExecutionActor(
            obs_dim=obs_dim,
            hidden_dims=actor_cfg["hidden_dims"],
            activation=actor_cfg["activation"],
        )
        self.register_network(self.actor)

        mine_cfg = net_cfg["mine"]
        self.mine = MINEEstimator(
            z_dim=1,
            f_dim=4,
            hidden_dims=tuple(mine_cfg["hidden_dims"]),
            ema_decay=mine_cfg["ema_decay"],
        )
        self.register_network(self.mine)

        self.pred_penalty = PredictabilityPenalty(flow_dim=4)
        self.register_network(self.pred_penalty)

        # Episode state
        self._remaining: int = self.inventory
        self._arrival_price: Optional[float] = None
        self._fill_prices: List[float] = []
        self._fill_qtys: List[int] = []
        self._flow_buffer: List[np.ndarray] = []

    # ── BaseAgent interface ───────────────────────────────────────────────

    def build_networks(self) -> None:
        # Networks built in __init__
        pass

    def act(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, float, float]:
        obs_t = self.to_tensor(obs).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                # Return mean action (mode of Beta = (α-1)/(α+β-2) for α,β>1)
                action, log_prob, entropy = self.actor(obs_t)
            else:
                action, log_prob, entropy = self.actor(obs_t)
        return (
            action.squeeze(0).cpu().numpy(),
            float(log_prob.item()),
            float(entropy.item()),
        )

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        # PPO update is handled by PPOUpdate in training/ppo_update.py
        # This method exists for interface compliance
        return {}

    # ── Episode bookkeeping ───────────────────────────────────────────────

    def reset_episode(self, arrival_price: float) -> None:
        self._remaining = self.inventory
        self._arrival_price = arrival_price
        self._fill_prices = []
        self._fill_qtys = []
        self._flow_buffer = []

    def record_fill(self, price: float, qty: int) -> None:
        self._remaining -= qty
        self._fill_prices.append(price)
        self._fill_qtys.append(qty)

    def record_flow(self, flow_feature: np.ndarray) -> None:
        self._flow_buffer.append(flow_feature)

    @property
    def remaining_inventory(self) -> int:
        return self._remaining

    @property
    def completion_rate(self) -> float:
        return 1.0 - self._remaining / max(self.inventory, 1)

    @property
    def implementation_shortfall(self) -> float:
        if not self._fill_prices or self._arrival_price is None:
            return 0.0
        total_qty = sum(self._fill_qtys)
        if total_qty == 0:
            return 0.0
        vwap = sum(p * q for p, q in zip(self._fill_prices, self._fill_qtys)) / total_qty
        return float(vwap - self._arrival_price)

    def get_flow_tensor(self) -> Optional[torch.Tensor]:
        if not self._flow_buffer:
            return None
        return torch.as_tensor(np.array(self._flow_buffer), dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def estimate_leakage(self) -> float:
        """Compute current MI leakage estimate from flow buffer."""
        ft = self.get_flow_tensor()
        if ft is None or len(ft) < 4:
            return 0.0
        z = torch.ones(len(ft), 1, device=self.device)
        return float(self.mine.estimate_only(z, ft))
