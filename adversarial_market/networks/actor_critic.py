"""
Actor-critic networks for MAPPO.

Architecture:
    - Shared centralized critic V(s): takes global state + all agent obs
    - Agent-specific actors π_i(a | o_i): each takes only own observation

The shared critic sees the full global state during training (CTDE).
At execution time, each agent uses only its own policy (decentralized).
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta, Normal


def build_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    activation: str = "tanh",
    use_layer_norm: bool = False,
) -> nn.Sequential:
    """Utility: build a multi-layer perceptron."""
    act_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU}[activation]
    layers: List[nn.Module] = []
    in_dim = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(in_dim, h))
        if use_layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(act_fn())
        in_dim = h
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


class SharedCritic(nn.Module):
    """
    Centralized value function V(s) for MAPPO.

    Input: concatenation of global market state + all agent observations.
    Output: scalar value estimate.
    """

    def __init__(
        self,
        global_state_dim: int,
        hidden_dims: Sequence[int] = (512, 512, 256),
        activation: str = "tanh",
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.net = build_mlp(global_state_dim, hidden_dims, 1, activation, use_layer_norm)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        """Returns value estimate, shape (B,)."""
        return self.net(global_state).squeeze(-1)


class ExecutionActor(nn.Module):
    """
    Execution agent stochastic policy π_E(a | o_E).

    Action: [size_frac, limit_offset, order_type_logit]
    Uses Beta distribution for bounded continuous actions (size_frac, limit_offset_norm)
    and Bernoulli for discrete order type.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        self.backbone = build_mlp(obs_dim, hidden_dims, hidden_dims[-1], activation)
        # Beta distribution parameters for size_frac and limit_offset_norm
        self.alpha_head = nn.Linear(hidden_dims[-1], 2)
        self.beta_head = nn.Linear(hidden_dims[-1], 2)
        # Bernoulli logit for order type
        self.order_type_head = nn.Linear(hidden_dims[-1], 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            action: (B, 3) sampled action
            log_prob: (B,) log probability
            entropy: (B,) policy entropy
        """
        h = self.backbone(obs)
        alpha = F.softplus(self.alpha_head(h)) + 1.0  # ensure > 1 for unimodal
        beta = F.softplus(self.beta_head(h)) + 1.0
        type_logit = self.order_type_head(h)

        beta_dist = Beta(alpha, beta)
        type_dist = torch.distributions.Bernoulli(logits=type_logit.squeeze(-1))

        size_frac_norm, offset_norm = beta_dist.rsample().split(1, dim=-1)
        order_type = type_dist.sample().unsqueeze(-1)

        # Map [0,1] → actual ranges
        size_frac = size_frac_norm  # [0,1]
        limit_offset = offset_norm * 10.0 - 5.0  # [-5, 5]

        action = torch.cat([size_frac, limit_offset, order_type], dim=-1)

        log_prob = beta_dist.log_prob(torch.cat([size_frac_norm, offset_norm], dim=-1)).sum(
            -1
        ) + type_dist.log_prob(order_type.squeeze(-1))
        entropy = beta_dist.entropy().sum(-1) + type_dist.entropy()

        return action, log_prob, entropy

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log prob and entropy of given actions (for PPO update)."""
        h = self.backbone(obs)
        alpha = F.softplus(self.alpha_head(h)) + 1.0
        beta_ = F.softplus(self.beta_head(h)) + 1.0
        type_logit = self.order_type_head(h)

        beta_dist = Beta(alpha, beta_)
        type_dist = torch.distributions.Bernoulli(logits=type_logit.squeeze(-1))

        # Recover normalised values — clamp strictly into Beta support (0, 1)
        _eps = 1e-6
        size_frac = actions[:, 0:1].clamp(_eps, 1.0 - _eps)
        offset_norm = ((actions[:, 1:2] + 5.0) / 10.0).clamp(_eps, 1.0 - _eps)
        order_type = actions[:, 2]

        log_prob = beta_dist.log_prob(torch.cat([size_frac, offset_norm], dim=-1)).sum(
            -1
        ) + type_dist.log_prob(order_type)
        entropy = beta_dist.entropy().sum(-1) + type_dist.entropy()
        return log_prob, entropy


class MarketMakerActor(nn.Module):
    """
    Market maker stochastic policy π_M(a | o_M, b_t).

    Action: [bid_offset, ask_offset, bid_size_frac, ask_size_frac]
    The belief b_t is appended to the observation before passing through
    the network, so spread decisions are directly conditioned on inference.

    Uses Normal distribution with learned log_std.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        activation: str = "relu",
        log_std_init: float = -0.5,
    ) -> None:
        super().__init__()
        self.action_dim = 4
        self.backbone = build_mlp(obs_dim, hidden_dims, hidden_dims[-1], activation)
        self.mean_head = nn.Linear(hidden_dims[-1], self.action_dim)
        self.log_std = nn.Parameter(torch.full((self.action_dim,), log_std_init))
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.backbone(obs)
        mean = self.mean_head(h)
        std = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return action, log_prob, entropy

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(obs)
        mean = self.mean_head(h)
        std = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy


class ArbitrageActor(nn.Module):
    """
    Arbitrageur stochastic policy π_A(a | o_A).

    Action: scalar ∈ [-1, 1] — direction and intensity of aggression.
    Uses tanh-squashed Normal distribution.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dims: Sequence[int] = (128, 128),
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.backbone = build_mlp(obs_dim, hidden_dims, hidden_dims[-1], activation)
        self.mean_head = nn.Linear(hidden_dims[-1], 1)
        self.log_std = nn.Parameter(torch.tensor(-1.0))
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.backbone(obs)
        mean = self.mean_head(h).squeeze(-1)
        std = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        raw = dist.rsample()
        action = torch.tanh(raw)
        # Log prob with tanh correction
        log_prob = dist.log_prob(raw) - torch.log(1 - action.pow(2) + 1e-6)
        entropy = dist.entropy()
        return action.unsqueeze(-1), log_prob, entropy

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(obs)
        mean = self.mean_head(h).squeeze(-1)
        std = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        raw = torch.atanh(actions.squeeze(-1).clamp(-0.999, 0.999))
        log_prob = dist.log_prob(raw) - torch.log(1 - actions.squeeze(-1).pow(2) + 1e-6)
        entropy = dist.entropy()
        return log_prob, entropy
