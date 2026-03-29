"""
Rollout buffer for MAPPO — stores transitions for all agents.

Implements Generalized Advantage Estimation (GAE) for computing
returns and advantages used in the PPO policy gradient update.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Generator, List

import numpy as np
import torch


@dataclass
class AgentBuffer:
    """Per-agent rollout storage."""

    obs: List[np.ndarray] = field(default_factory=list)
    actions: List[np.ndarray] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    entropies: List[float] = field(default_factory=list)
    # Computed after rollout collection by compute_returns_and_advantages
    advantages: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    returns: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))

    def clear(self) -> None:
        self.obs.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
        self.entropies.clear()

    def __len__(self) -> int:
        return len(self.rewards)


class RolloutBuffer:
    """
    Stores multi-agent rollout transitions and computes GAE advantages.

    Usage:
        buffer = RolloutBuffer(rollout_length=2048, n_envs=8, gamma=0.99, gae_lambda=0.95)
        # ... collect transitions ...
        buffer.add(obs_dict, action_dict, logp_dict, reward_dict, value_dict, done_dict)
        buffer.compute_returns_and_advantages(last_values_dict)
        for batch in buffer.get_minibatches(minibatch_size=256):
            # PPO update
    """

    AGENT_KEYS = ("execution", "market_maker", "arbitrageur")

    def __init__(
        self,
        rollout_length: int,
        n_envs: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: str = "cpu",
    ) -> None:
        self.rollout_length = rollout_length
        self.n_envs = n_envs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        self._buffers: Dict[str, AgentBuffer] = {k: AgentBuffer() for k in self.AGENT_KEYS}
        # Global state for critic
        self._global_states: list = []
        self._ptr = 0

    def add(
        self,
        obs: Dict[str, np.ndarray],
        global_state: np.ndarray,
        actions: Dict[str, np.ndarray],
        log_probs: Dict[str, float],
        rewards: Dict[str, float],
        values: Dict[str, float],
        dones: Dict[str, bool],
        entropies: Dict[str, float],
    ) -> None:
        """Add one timestep of transitions (all envs batched)."""
        for key in self.AGENT_KEYS:
            buf = self._buffers[key]
            buf.obs.append(obs[key])
            buf.actions.append(actions[key])
            buf.log_probs.append(log_probs[key])
            buf.rewards.append(rewards[key])
            buf.values.append(values[key])
            buf.dones.append(dones[key])
            buf.entropies.append(entropies[key])
        self._global_states.append(global_state)
        self._ptr += 1

    def compute_returns_and_advantages(self, last_values: Dict[str, float]) -> None:
        """
        Compute GAE-λ advantages and discounted returns for each agent.

        GAE: A_t = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}
        where δ_t = r_t + γ * V(s_{t+1}) * (1-d_t) - V(s_t)
        """
        for key in self.AGENT_KEYS:
            buf = self._buffers[key]
            T = len(buf.rewards)
            advantages = np.zeros(T, dtype=np.float32)
            last_gae = 0.0
            last_val = last_values[key]

            for t in reversed(range(T)):
                next_val = last_val if t == T - 1 else buf.values[t + 1]
                done = float(buf.dones[t])
                delta = buf.rewards[t] + self.gamma * next_val * (1.0 - done) - buf.values[t]
                last_gae = delta + self.gamma * self.gae_lambda * (1.0 - done) * last_gae
                advantages[t] = last_gae

            # Normalise advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            returns = advantages + np.array(buf.values, dtype=np.float32)

            # Store back
            buf.advantages = advantages
            buf.returns = returns

    def get_minibatches(
        self, minibatch_size: int
    ) -> Generator[Dict[str, Dict[str, torch.Tensor]], None, None]:
        """
        Yield random minibatches of transitions for PPO updates.
        Each minibatch contains tensors for all agents.
        """
        T = self._ptr
        indices = np.random.permutation(T)

        for start in range(0, T, minibatch_size):
            idx = indices[start : start + minibatch_size]
            batch: Dict[str, Dict[str, torch.Tensor]] = {}

            for key in self.AGENT_KEYS:
                buf = self._buffers[key]
                batch[key] = {
                    "obs": self._to_tensor(np.array(buf.obs)[idx]),
                    "actions": self._to_tensor(np.array(buf.actions)[idx]),
                    "log_probs_old": self._to_tensor(np.array(buf.log_probs)[idx]),
                    "advantages": self._to_tensor(buf.advantages[idx]),
                    "returns": self._to_tensor(buf.returns[idx]),
                    "values_old": self._to_tensor(np.array(buf.values)[idx]),
                }

            batch["global"] = {"states": self._to_tensor(np.array(self._global_states)[idx])}
            yield batch

    def clear(self) -> None:
        for buf in self._buffers.values():
            buf.clear()
        self._global_states.clear()
        self._ptr = 0

    def is_full(self) -> bool:
        return self._ptr >= self.rollout_length

    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(arr, dtype=torch.float32, device=self.device)
