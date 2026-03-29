"""
MINE: Mutual Information Neural Estimator.

Estimates I(X; Y) using the Donsker-Varadhan lower bound:
    I(X; Y) >= E_{p(x,y)}[T(x,y)] - log(E_{p(x)p(y)}[e^{T(x,y)}])

where T: X × Y → R is a learned statistics network.

For the execution agent, we estimate:
    MI(z ; F_t)
where z ∈ {0,1} is the agent's latent type (informed) and F_t is the
observable order flow feature vector.

References:
    Belghazi et al. (2018). MINE: Mutual Information Neural Estimation.
    https://arxiv.org/abs/1801.04062
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MINENetwork(nn.Module):
    """
    Statistics network T(z, f) mapping (type, flow) → scalar.

    Inputs:
        z: agent type embedding, shape (B, z_dim)
        f: flow feature vector, shape (B, f_dim)
    Output:
        scalar T(z, f), shape (B,)
    """

    def __init__(
        self,
        z_dim: int,
        f_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.f_dim = f_dim

        act_fn = {"relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU}[activation]

        layers = []
        in_dim = z_dim + f_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), act_fn()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

        # Xavier initialisation is essential for stable MINE training
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, f], dim=-1)
        return self.net(x).squeeze(-1)


class MINEEstimator(nn.Module):
    """
    Full MINE wrapper with exponential moving average baseline for
    variance reduction (MINE-f variant from Belghazi et al.).

    Usage:
        estimator = MINEEstimator(z_dim=1, f_dim=4)
        mi_lower_bound, loss = estimator.compute(z_samples, f_samples)
        loss.backward()  # update the statistics network

    The MI lower bound is used as a penalty term in the execution
    agent's reward:
        r_E += -lambda_leakage * mi_lower_bound
    """

    def __init__(
        self,
        z_dim: int,
        f_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        activation: str = "relu",
        ema_decay: float = 0.99,
    ) -> None:
        super().__init__()
        self.net = MINENetwork(z_dim, f_dim, hidden_dims, activation)
        self.ema_decay = ema_decay
        # EMA of the exponential term (variance reduction baseline)
        self.register_buffer("ema_et", torch.tensor(1.0))
        self.ema_et: torch.Tensor

    def compute(
        self,
        z: torch.Tensor,
        f: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate MI(z; f) and return (mi_estimate, loss).

        Args:
            z: shape (B, z_dim) — joint samples (z_i, f_i drawn together)
            f: shape (B, f_dim) — marginal shuffle applied inside

        Returns:
            mi_estimate: scalar lower bound on I(z; f)
            loss: negative DV bound (minimise to tighten the estimate)
        """
        batch_size = z.shape[0]

        # Joint term: E_{p(z,f)}[T(z,f)]
        t_joint = self.net(z, f)

        # Marginal term: shuffle f to break z-f dependence
        shuffle_idx = torch.randperm(batch_size, device=z.device)
        f_marginal = f[shuffle_idx]
        t_marginal = self.net(z, f_marginal)

        # DV bound: E[T_joint] - log(E[exp(T_marginal)])
        # Use EMA baseline for variance reduction (MINE-f)
        et = torch.exp(t_marginal)
        ema_et = self.ema_decay * self.ema_et + (1 - self.ema_decay) * et.mean().detach()
        self.ema_et = ema_et

        mi_estimate = t_joint.mean() - torch.log(ema_et + 1e-8)
        # Loss = negative bound (we maximise the bound via minimising loss)
        loss = -(t_joint.mean() - et.mean() / (ema_et.detach() + 1e-8))

        return mi_estimate, loss

    @torch.no_grad()
    def estimate_only(self, z: torch.Tensor, f: torch.Tensor) -> float:
        """Fast inference-only MI estimate (no gradient tracking)."""
        mi, _ = self.compute(z, f)
        return float(mi.item())


class PredictabilityPenalty(nn.Module):
    """
    Measures predictability of the execution agent's order flow.

    Trains a small sequence model to predict the next order feature
    from recent history. High predictability → the agent's pattern
    is detectable → penalise.

    Implements: -H(F_{t+1} | F_{t-k:t}) — higher entropy = less predictable.
    We proxy this as the negative log-likelihood of a learned predictor.
    """

    def __init__(self, flow_dim: int = 4, hidden_dim: int = 64) -> None:
        super().__init__()
        self.gru = nn.GRU(flow_dim, hidden_dim, batch_first=True, num_layers=1)
        self.head = nn.Linear(hidden_dim, flow_dim)

    def forward(self, flow_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            flow_seq: (B, T, flow_dim) sequence of order flow features

        Returns:
            predictability: scalar (lower = less predictable)
            loss: MSE prediction loss (train to tighten the measure)
        """
        if flow_seq.shape[1] < 2:
            zero = torch.tensor(0.0, device=flow_seq.device)
            return zero, zero

        x = flow_seq[:, :-1, :]  # input: all but last
        y = flow_seq[:, 1:, :]  # target: all but first

        h, _ = self.gru(x)
        pred = self.head(h)
        loss = F.mse_loss(pred, y)

        # Predictability ∝ how well the model predicts (low loss = high predictability)
        predictability = -loss  # negate: use as penalty when high
        return predictability, loss
