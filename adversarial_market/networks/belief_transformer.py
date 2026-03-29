"""
Belief Transformer — sequential inference model for the market maker.

Learns a probabilistic mapping:
    p(z = informed | F_t) = b_t ∈ [0, 1]

where F_t is the history of observed order flow events up to step t.

Architecture:
    - Input: sequence of (side, price_rel, size_rel, timestamp) order events
    - Positional encoding: sinusoidal
    - Transformer encoder: multi-head self-attention + FFN
    - Output head: MLP → sigmoid → scalar belief b_t

The belief b_t is:
    - Integrated into the market maker's action (spread/depth decisions)
    - Used to compute the adverse selection proxy
    - Tracked as a key metric for the detectability phase transition
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al. 2017)."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)
        self.pe: torch.Tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model)"""
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class OrderFlowEmbedding(nn.Module):
    """
    Projects raw order flow features to d_model-dimensional tokens.

    Input features per event (5 dimensions):
        0: side          ∈ {-1, 0, 1}    (ASK, unknown, BID)
        1: price_rel     ∈ R             (price / mid - 1)
        2: size_rel      ∈ [0, 1]        (size / max_size)
        3: order_type    ∈ {0, 1}        (MARKET, LIMIT)
        4: time_frac     ∈ [0, 1]        (step / horizon)
    """

    def __init__(self, input_dim: int = 5, d_model: int = 128) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class BeliefTransformer(nn.Module):
    """
    Transformer encoder that processes order flow history and
    outputs a belief scalar b_t = P(flow is informed | F_t).

    Args:
        input_dim:   Dimension of each order flow event feature vector.
        d_model:     Transformer hidden dimension.
        n_heads:     Number of attention heads.
        n_layers:    Number of encoder layers.
        d_ff:        Feed-forward sublayer hidden dimension.
        dropout:     Dropout probability.
        max_seq_len: Maximum sequence length (history window).
    """

    def __init__(
        self,
        input_dim: int = 5,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 200,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.embedding = OrderFlowEmbedding(input_dim, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_seq_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN for training stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Belief output head
        self.belief_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

        # Latent representation head (for adversarial use by exec agent proxy)
        self.repr_head = nn.Linear(d_model, d_model // 4)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        flow_seq: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            flow_seq:              (B, T, input_dim) order flow event sequence
            src_key_padding_mask:  (B, T) bool mask — True = padding position

        Returns:
            belief:    (B,) scalar in [0, 1] — P(informed | F_t)
            repr:      (B, d_model//4) latent representation of the flow
        """
        # Embed + positional encoding
        x = self.embedding(flow_seq)  # (B, T, d_model)
        x = self.pos_enc(x)  # (B, T, d_model)

        # Transformer encoding
        h = self.encoder(x, src_key_padding_mask=src_key_padding_mask)  # (B, T, d_model)

        # Aggregate: use the last valid token (causal belief update)
        if src_key_padding_mask is not None:
            # Find last non-padding position per batch
            lengths = (~src_key_padding_mask).sum(dim=1) - 1  # (B,)
            lengths = lengths.clamp(min=0)
            idx = lengths.unsqueeze(1).unsqueeze(2).expand(-1, 1, self.d_model)
            agg = h.gather(1, idx).squeeze(1)  # (B, d_model)
        else:
            agg = h[:, -1, :]  # last position

        belief = self.belief_head(agg).squeeze(-1)  # (B,)
        repr_ = self.repr_head(agg)  # (B, d_model//4)
        return belief, repr_

    def compute_belief_loss(
        self,
        flow_seq: torch.Tensor,
        labels: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        entropy_reg_weight: float = 0.05,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Supervised belief loss for market maker training.

        Labels = 1 if the flow in this episode came from the execution
        agent (informed), 0 if pure noise.

        Loss = BCE(belief, label) - beta * H(belief)
        The entropy regularisation prevents overconfident beliefs.

        Args:
            flow_seq:          (B, T, input_dim)
            labels:            (B,) binary 0/1
            src_key_padding_mask: (B, T)
            entropy_reg_weight: beta in the reward formula

        Returns:
            loss: scalar training loss
            belief: (B,) predicted beliefs
        """
        belief, _ = self.forward(flow_seq, src_key_padding_mask)

        # Binary cross-entropy
        bce_loss = F.binary_cross_entropy(belief, labels.float())

        # Entropy regularisation: H(b) = -b*log(b) - (1-b)*log(1-b)
        eps = 1e-7
        entropy = -(belief * torch.log(belief + eps) + (1 - belief) * torch.log(1 - belief + eps))
        # Subtract entropy to encourage calibration (not overconfidence)
        loss = bce_loss - entropy_reg_weight * entropy.mean()

        return loss, belief

    @torch.no_grad()
    def update_belief(
        self,
        flow_seq: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> float:
        """Online belief update — inference only."""
        self.eval()
        belief, _ = self.forward(flow_seq, src_key_padding_mask)
        self.train()
        return float(belief.mean().item())
