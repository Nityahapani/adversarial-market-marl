"""
Abstract base class for all agents in the adversarial market framework.

All agents share a common interface:
  - observe(state) → agent-specific observation vector
  - act(obs) → action, log_prob, entropy
  - update(batch) → loss metrics dict
  - reset() → clear episode-local state

The base class enforces this contract and provides shared utilities
(device management, gradient freezing, parameter counting).
"""

from __future__ import annotations

import abc
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn


class BaseAgent(abc.ABC):
    """
    Abstract base class for execution agent, market maker, and arbitrageur.

    Subclasses implement:
        build_networks() → construct policy and auxiliary networks
        observe(state)   → extract own observation from global MarketState
        act(obs)         → sample action from policy; return (action, log_prob, entropy)
        update(batch)    → run gradient update; return metrics dict

    Attributes:
        agent_id:  Integer ID (0=exec, 1=mm, 2=arb).
        name:      Human-readable agent name.
        device:    Torch device for network tensors.
        config:    Agent-specific config sub-dict.
    """

    def __init__(
        self,
        agent_id: int,
        name: str,
        config: Dict[str, Any],
        device: str = "cpu",
    ) -> None:
        self.agent_id = agent_id
        self.name = name
        self.config = config
        self.device = torch.device(device)
        self._networks: List[nn.Module] = []
        self._step = 0

    # ── Abstract interface ────────────────────────────────────────────────

    @abc.abstractmethod
    def build_networks(self) -> None:
        """Construct and register all neural network modules."""
        ...

    @abc.abstractmethod
    def act(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Sample an action from the current policy.

        Args:
            obs:           Observation vector (numpy, already on CPU).
            deterministic: If True, return mode rather than sample (for eval).

        Returns:
            action:   Numpy action array.
            log_prob: Log probability of the sampled action (scalar float).
            entropy:  Policy entropy at this observation (scalar float).
        """
        ...

    @abc.abstractmethod
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform a gradient update step on the given minibatch.

        Args:
            batch: Dict containing tensors for obs, actions, advantages, returns, etc.

        Returns:
            Dict of scalar metric values for logging.
        """
        ...

    # ── Shared utilities ──────────────────────────────────────────────────

    def register_network(self, net: nn.Module) -> nn.Module:
        """Register a network module so it is tracked for freezing/parameter counting."""
        net = net.to(self.device)
        self._networks.append(net)
        return net

    def freeze(self) -> None:
        """Freeze all registered network parameters (no gradient updates)."""
        for net in self._networks:
            for param in net.parameters():
                param.requires_grad_(False)

    def unfreeze(self) -> None:
        """Unfreeze all registered network parameters."""
        for net in self._networks:
            for param in net.parameters():
                param.requires_grad_(True)

    def train_mode(self) -> None:
        for net in self._networks:
            net.train()

    def eval_mode(self) -> None:
        for net in self._networks:
            net.eval()

    def parameter_count(self) -> int:
        """Total number of trainable parameters across all registered networks."""
        return sum(p.numel() for net in self._networks for p in net.parameters() if p.requires_grad)

    def to_tensor(
        self,
        arr: np.ndarray,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        return torch.as_tensor(arr, dtype=dtype, device=self.device)

    def save_state(self, path: str) -> None:
        """Save all network state dicts to a single file."""
        state: Dict[str, Any] = {
            f"network_{i}": net.state_dict() for i, net in enumerate(self._networks)
        }
        state["step"] = self._step
        torch.save(state, path)

    def load_state(self, path: str) -> None:
        """Load network state dicts from file."""
        state = torch.load(path, map_location=self.device)
        for i, net in enumerate(self._networks):
            key = f"network_{i}"
            if key in state:
                net.load_state_dict(state[key])
        self._step = int(state.get("step", 0))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"id={self.agent_id}, "
            f"name={self.name}, "
            f"params={self.parameter_count():,})"
        )
