"""
Policy evaluation harness.

Runs trained policies in evaluation mode (no gradient updates),
collects episode metrics, and produces summary statistics for
analysis of the detectability phase transition.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from adversarial_market.environment.lob_env import LOBEnvironment
from adversarial_market.evaluation.metrics import (
    EpisodeMetrics,
    belief_accuracy,
    compute_summary_metrics,
    flow_entropy,
    implementation_shortfall,
    kl_divergence_flows,
)
from adversarial_market.networks.actor_critic import (
    ArbitrageActor,
    ExecutionActor,
    MarketMakerActor,
)
from adversarial_market.networks.belief_transformer import BeliefTransformer
from adversarial_market.networks.mine_estimator import MINEEstimator


class Evaluator:
    """
    Runs n_episodes of evaluation using provided trained policies.

    Args:
        config:    Full configuration dict.
        device:    Torch device string.
    """

    def __init__(self, config: Dict[str, Any], device: str = "cpu") -> None:
        self.cfg = config
        self.device = torch.device(device)
        self.env = LOBEnvironment(config)

    def evaluate(
        self,
        exec_actor: ExecutionActor,
        mm_actor: MarketMakerActor,
        arb_actor: ArbitrageActor,
        belief_transformer: BeliefTransformer,
        mine: MINEEstimator,
        n_episodes: int = 10,
        noise_only_episodes: int = 5,
        seed: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Run evaluation and return summary metrics dict.

        Also runs noise-only episodes to compute KL divergence between
        informed and noise flow distributions.
        """
        exec_actor.eval()
        mm_actor.eval()
        arb_actor.eval()
        belief_transformer.eval()
        mine.eval()

        informed_flows_all = []
        noise_flows_all = []
        episodes: List[EpisodeMetrics] = []

        # ── Informed episodes (execution agent present) ───────────────────
        for ep in range(n_episodes):
            ep_seed = seed + ep if seed is not None else None
            metrics, flow_feats = self._run_episode(
                exec_actor,
                mm_actor,
                arb_actor,
                belief_transformer,
                mine,
                seed=ep_seed,
                exec_present=True,
            )
            episodes.append(metrics)
            informed_flows_all.extend(flow_feats)

        # ── Noise-only episodes (no execution agent) ──────────────────────
        for ep in range(noise_only_episodes):
            ep_seed = seed + n_episodes + ep if seed is not None else None
            _, flow_feats = self._run_episode(
                exec_actor,
                mm_actor,
                arb_actor,
                belief_transformer,
                mine,
                seed=ep_seed,
                exec_present=False,
            )
            noise_flows_all.extend(flow_feats)

        # ── KL divergence between informed and noise flow ─────────────────
        if informed_flows_all and noise_flows_all:
            inf_arr = np.array(informed_flows_all)
            noise_arr = np.array(noise_flows_all)
            kl = kl_divergence_flows(inf_arr, noise_arr)
            for ep_m in episodes:
                ep_m.kl_informed_noise = kl

        summary = compute_summary_metrics(episodes)

        exec_actor.train()
        mm_actor.train()
        arb_actor.train()
        belief_transformer.train()
        mine.train()

        return summary

    def _run_episode(
        self,
        exec_actor: ExecutionActor,
        mm_actor: MarketMakerActor,
        arb_actor: ArbitrageActor,
        belief_transformer: BeliefTransformer,
        mine: MINEEstimator,
        seed: Optional[int],
        exec_present: bool,
    ) -> Tuple[EpisodeMetrics, List[List[float]]]:
        """Run a single evaluation episode."""
        obs, _ = self.env.reset(seed=seed)
        assert self.env.state is not None
        m = EpisodeMetrics()
        m.arrival_price = self.env.state.mid_price

        beliefs_list = []
        true_labels = []
        flow_feats = []

        done = False
        while not done:
            with torch.no_grad():
                exec_obs = torch.as_tensor(
                    obs["execution"], dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                mm_obs = torch.as_tensor(
                    obs["market_maker"], dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                arb_obs = torch.as_tensor(
                    obs["arbitrageur"], dtype=torch.float32, device=self.device
                ).unsqueeze(0)

                if exec_present:
                    exec_action, _, _ = exec_actor(exec_obs)
                else:
                    exec_action = torch.zeros(1, 3, device=self.device)

                mm_action, _, _ = mm_actor(mm_obs)
                arb_action, _, _ = arb_actor(arb_obs)

            actions = {
                "execution": exec_action.squeeze(0).cpu().numpy(),
                "market_maker": mm_action.squeeze(0).cpu().numpy(),
                "arbitrageur": arb_action.squeeze(0).cpu().numpy(),
            }

            obs, rewards, terminated, truncated, info = self.env.step(actions)
            done = any(terminated.values())

            # Collect belief trajectory
            b = self.env.state.mm_belief if self.env.state else 0.5
            beliefs_list.append(b)
            true_labels.append(1.0 if exec_present else 0.0)

            m.spread_trajectory.append(self.env.state.spread or 0.0)
            m.mm_belief_trajectory.append(b)
            m.price_trajectory.append(self.env.state.mid_price)

            m.mm_pnl += rewards["market_maker"]
            m.arb_pnl += rewards["arbitrageur"]

        # Compute exec metrics using AgentState tracked fields
        exec_state = self.env.state.agent_states.get(0)
        if exec_state and exec_state.total_filled_qty > 0:
            m.avg_fill_price = exec_state.avg_fill_price or 0.0
            m.total_filled = exec_state.total_filled_qty
            m.implementation_shortfall = implementation_shortfall(
                m.arrival_price,
                [m.avg_fill_price],
                [m.total_filled],
            )

        m.remaining_inventory = self.env.state.exec_remaining_inventory
        init_inv = self.env.state.exec_initial_inventory
        m.completion_rate = 1.0 - m.remaining_inventory / max(init_inv, 1)

        # Belief accuracy
        if beliefs_list:
            m.mm_belief_accuracy = belief_accuracy(np.array(beliefs_list), np.array(true_labels))

        m.avg_spread = float(np.mean(m.spread_trajectory)) if m.spread_trajectory else 0.0
        m.realized_volatility = self.env.state.realized_volatility if self.env.state else 0.0

        # Flow features for KL computation
        flow_buf = self.env.get_flow_buffer()
        if len(flow_buf) > 0:
            flow_feats = flow_buf.tolist()
            m.flow_entropy = flow_entropy(flow_buf)

            # MI estimate
            if len(flow_buf) >= 4:
                ft = torch.as_tensor(flow_buf, dtype=torch.float32, device=self.device)
                z = torch.ones(len(ft), 1, device=self.device) * float(exec_present)
                m.mi_estimate = mine.estimate_only(z, ft)

        return m, flow_feats
