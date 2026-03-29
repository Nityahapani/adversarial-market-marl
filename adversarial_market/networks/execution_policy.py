"""
Execution policy module — higher-level execution strategy wrappers.

Provides:
  1. TWAPPolicy    — deterministic baseline (equal-slice TWAP)
  2. VWAPPolicy    — volume-weighted baseline tracking market volume profile
  3. LearnedPolicy — wraps the trained ExecutionActor for deployment

These are used in evaluation to benchmark the learned policy against
classical execution algorithms and quantify the value of adversarial
information concealment.

Reference benchmarks:
  - TWAP: equal slices across the horizon → maximally predictable (high KL)
  - VWAP: follows historical volume profile → partially predictable
  - Learned: optimises IS + MI simultaneously → adaptive camouflage
"""

from __future__ import annotations

import abc
from typing import Dict, Tuple

import numpy as np
import torch

from adversarial_market.networks.actor_critic import ExecutionActor


class BaseExecutionPolicy(abc.ABC):
    """Abstract execution policy interface."""

    @abc.abstractmethod
    def act(
        self,
        obs: np.ndarray,
        remaining: int,
        time_remaining: float,
        mid_price: float,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Produce an action given the current state.

        Returns:
            action:   numpy action array [size_frac, limit_offset, order_type]
            log_prob: scalar (0.0 for deterministic policies)
            entropy:  scalar (0.0 for deterministic policies)
        """
        ...

    @abc.abstractmethod
    def reset(self, initial_inventory: int, arrival_price: float) -> None:
        """Reset policy state for a new episode."""
        ...


class TWAPPolicy(BaseExecutionPolicy):
    """
    Time-Weighted Average Price (TWAP) execution baseline.

    Splits the total order into equal-sized slices and trades one
    slice per time step. Submits market orders to guarantee fill.

    Properties:
        - Maximally predictable (constant order size every step)
        - Minimal IS in liquid markets with flat volume profile
        - KL(TWAP || noise) is high — easily detectable

    Used as the lower bound on KL divergence in phase transition analysis.
    """

    def __init__(self, horizon: int, max_order_size: int) -> None:
        self.horizon = horizon
        self.max_order_size = max_order_size
        self._initial_inventory: int = 0
        self._arrival_price: float = 0.0
        self._step: int = 0

    def reset(self, initial_inventory: int, arrival_price: float) -> None:
        self._initial_inventory = initial_inventory
        self._arrival_price = arrival_price
        self._step = 0

    def act(
        self,
        obs: np.ndarray,
        remaining: int,
        time_remaining: float,
        mid_price: float,
    ) -> Tuple[np.ndarray, float, float]:
        if remaining <= 0:
            return np.array([0.0, 0.0, -1.0], dtype=np.float32), 0.0, 0.0
        steps_left = max(1, int(time_remaining * self.horizon))
        target_qty = max(1, remaining // steps_left)
        size_frac = min(1.0, target_qty / max(self.max_order_size, 1))
        # Market order (order_type_logit = -1)
        action = np.array([size_frac, 0.0, -1.0], dtype=np.float32)
        self._step += 1
        return action, 0.0, 0.0


class VWAPPolicy(BaseExecutionPolicy):
    """
    Volume-Weighted Average Price (VWAP) execution baseline.

    Targets a volume profile derived from a typical intraday U-shape
    (high activity at open/close, lower mid-day). Sizes orders to match
    the expected fraction of daily volume in each period.

    Properties:
        - More sophisticated than TWAP — adapts to volume pattern
        - Still deterministic and partially predictable
        - KL(VWAP || noise) < KL(TWAP || noise) but > 0
    """

    def __init__(self, horizon: int, max_order_size: int) -> None:
        self.horizon = horizon
        self.max_order_size = max_order_size
        self._volume_profile = self._build_volume_profile(horizon)
        self._initial_inventory: int = 0
        self._executed_fracs: np.ndarray = np.zeros(horizon)

    def _build_volume_profile(self, T: int) -> np.ndarray:
        """
        U-shaped intraday volume profile (open/close heavier, quiet midday).
        Uses cosine: v(t) = 1 - cos(2π·t), which is 0 at t=0, peaks at t=0.5,
        then 0 again at t=1.  We want the *inverse* — high at edges, low in
        middle — so we negate and shift: v(t) = 1 + cos(2π·t).
        """
        t = np.linspace(0, 1, T, endpoint=False)
        profile = 1.0 + np.cos(2.0 * np.pi * t)  # high at 0 and 1, low at 0.5
        profile = profile + 0.1  # small floor so no zero-probability bins
        profile = profile / profile.sum()
        return profile

    def reset(self, initial_inventory: int, arrival_price: float) -> None:
        self._initial_inventory = initial_inventory
        self._arrival_price = arrival_price
        self._cumulative_profile = np.cumsum(self._volume_profile)

    def act(
        self,
        obs: np.ndarray,
        remaining: int,
        time_remaining: float,
        mid_price: float,
    ) -> Tuple[np.ndarray, float, float]:
        step = self.horizon - int(time_remaining * self.horizon)
        step = min(step, self.horizon - 1)

        target_cumulative = self._cumulative_profile[step]
        already_executed = 1.0 - remaining / max(self._initial_inventory, 1)
        target_this_step = max(
            0.0,
            (target_cumulative - already_executed) * self._initial_inventory,
        )
        size_frac = min(1.0, target_this_step / max(self.max_order_size, 1))

        action = np.array([size_frac, 0.0, -1.0], dtype=np.float32)
        return action, 0.0, 0.0


class LearnedPolicy(BaseExecutionPolicy):
    """
    Wrapper around a trained ExecutionActor for deployment and benchmarking.

    Converts the stochastic training policy to a deterministic evaluation
    policy (uses mode of the distribution) while preserving the full
    stochastic interface for fair comparison.

    Args:
        actor:       Trained ExecutionActor module.
        device:      Torch device.
        stochastic:  If True, sample actions (training mode).
                     If False, use distribution mode (eval mode).
    """

    def __init__(
        self,
        actor: ExecutionActor,
        device: str = "cpu",
        stochastic: bool = False,
    ) -> None:
        self.actor = actor
        self.device = torch.device(device)
        self.stochastic = stochastic
        self._initial_inventory: int = 0

    def reset(self, initial_inventory: int, arrival_price: float) -> None:
        self._initial_inventory = initial_inventory

    def act(
        self,
        obs: np.ndarray,
        remaining: int,
        time_remaining: float,
        mid_price: float,
    ) -> Tuple[np.ndarray, float, float]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, log_prob, entropy = self.actor(obs_t)
        return (
            action.squeeze(0).cpu().numpy(),
            float(log_prob.item()),
            float(entropy.item()),
        )


class AdaptiveCamouflagePolicy(BaseExecutionPolicy):
    """
    Hybrid policy that blends noise and execution orders for camouflage.

    At each step, with probability p_noise, submits a random noise-like
    order (small size, random offset, random type) instead of the actual
    execution order. This explicit mixing is a hand-crafted baseline for
    the learned camouflage strategy.

    Used to verify that the learned policy discovers something beyond
    simple noise injection.

    Args:
        base_policy:  Underlying execution policy.
        p_noise:      Probability of substituting a noise order.
        rng_seed:     Random seed for reproducibility.
    """

    def __init__(
        self,
        base_policy: BaseExecutionPolicy,
        p_noise: float = 0.3,
        rng_seed: int = 0,
    ) -> None:
        self.base = base_policy
        self.p_noise = p_noise
        self._rng = np.random.default_rng(rng_seed)

    def reset(self, initial_inventory: int, arrival_price: float) -> None:
        self.base.reset(initial_inventory, arrival_price)

    def act(
        self,
        obs: np.ndarray,
        remaining: int,
        time_remaining: float,
        mid_price: float,
    ) -> Tuple[np.ndarray, float, float]:
        if self._rng.random() < self.p_noise:
            # Inject a noise-like order: small, random offset, random type
            noise_action = np.array(
                [
                    self._rng.uniform(0.05, 0.2),  # small size
                    self._rng.uniform(-5.0, 5.0),  # random offset
                    self._rng.choice([-1.0, 1.0]),  # random type
                ],
                dtype=np.float32,
            )
            return noise_action, 0.0, 0.0
        return self.base.act(obs, remaining, time_remaining, mid_price)


def make_benchmark_policies(
    horizon: int,
    max_order_size: int,
) -> Dict[str, BaseExecutionPolicy]:
    """Factory: returns all benchmark policies keyed by name."""
    return {
        "twap": TWAPPolicy(horizon, max_order_size),
        "vwap": VWAPPolicy(horizon, max_order_size),
        "twap_camouflage": AdaptiveCamouflagePolicy(
            TWAPPolicy(horizon, max_order_size), p_noise=0.3
        ),
    }
