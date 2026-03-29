"""
Evaluation metrics for the adversarial market MARL framework.

Key metrics:
  - Implementation shortfall (IS): execution cost vs arrival price
  - KL divergence: D_KL(informed_flow || noise_flow) — detectability measure
  - Market maker belief accuracy: how accurately MM classifies informed flow
  - Spread dynamics: how spread responds to perceived toxicity
  - Leakage score: mutual information estimate at evaluation time
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
from scipy import stats


@dataclass
class EpisodeMetrics:
    """Metrics collected over one evaluation episode."""

    # Execution agent
    implementation_shortfall: float = 0.0
    arrival_price: float = 0.0
    avg_fill_price: float = 0.0
    total_filled: int = 0
    remaining_inventory: int = 0
    completion_rate: float = 0.0
    n_orders: int = 0
    market_order_fraction: float = 0.0
    order_sizes: List[int] = field(default_factory=list)

    # Information leakage
    mi_estimate: float = 0.0
    kl_informed_noise: float = 0.0
    flow_entropy: float = 0.0

    # Market maker
    mm_pnl: float = 0.0
    mm_adverse_selection: float = 0.0
    mm_belief_accuracy: float = 0.0
    mm_belief_trajectory: List[float] = field(default_factory=list)
    spread_trajectory: List[float] = field(default_factory=list)
    mm_inventory_trajectory: List[int] = field(default_factory=list)

    # Arbitrageur
    arb_pnl: float = 0.0
    arb_n_trades: int = 0

    # Market quality
    realized_volatility: float = 0.0
    avg_spread: float = 0.0
    total_volume: int = 0
    price_trajectory: List[float] = field(default_factory=list)


def implementation_shortfall(
    arrival_price: float,
    fill_prices: List[float],
    fill_quantities: List[int],
    side: int = 1,  # 1 = buy, -1 = sell
) -> float:
    """
    Implementation shortfall = signed cost above arrival price.

    For a buy order: IS = sum(fill_price * qty) / sum(qty) - arrival_price
    Higher IS = worse execution.
    """
    total_qty = sum(fill_quantities)
    if total_qty == 0:
        return 0.0
    vwap = sum(p * q for p, q in zip(fill_prices, fill_quantities)) / total_qty
    return side * (vwap - arrival_price)


def kl_divergence_flows(
    informed_flows: np.ndarray,
    noise_flows: np.ndarray,
    n_bins: int = 20,
) -> float:
    """
    Estimate D_KL(P_informed || P_noise) over order flow features.

    Uses histogram-based KL divergence on each feature dimension,
    then averages (simplification; for full multivariate KL use kernel methods).

    Args:
        informed_flows: (N, F) feature matrix from execution agent episodes
        noise_flows:    (M, F) feature matrix from noise-trader-only episodes

    Returns:
        Scalar KL divergence estimate. Near 0 = indistinguishable flows.
    """
    if len(informed_flows) < 10 or len(noise_flows) < 10:
        return float("nan")

    n_features = informed_flows.shape[1]
    kl_total = 0.0

    for f in range(n_features):
        x = informed_flows[:, f]
        y = noise_flows[:, f]
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        if lo == hi:
            continue
        bins = np.linspace(lo, hi, n_bins + 1)

        p, _ = np.histogram(x, bins=bins, density=True)
        q, _ = np.histogram(y, bins=bins, density=True)

        # Smooth with small epsilon to avoid log(0)
        p = p + 1e-8
        q = q + 1e-8
        p /= p.sum()
        q /= q.sum()

        kl = float(stats.entropy(p, q))  # type: ignore[arg-type]
        kl_total += kl

    return kl_total / n_features


def belief_accuracy(
    beliefs: np.ndarray,
    true_labels: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """
    Classification accuracy of market maker's belief b_t.

    Args:
        beliefs:     (T,) predicted probabilities in [0,1]
        true_labels: (T,) binary 0/1 ground truth
        threshold:   Decision boundary

    Returns:
        Fraction of correctly classified steps.
    """
    predictions = (beliefs > threshold).astype(int)
    return float((predictions == true_labels.astype(int)).mean())


def belief_brier_score(beliefs: np.ndarray, true_labels: np.ndarray) -> float:
    """Brier score: mean squared error of probabilistic predictions."""
    return float(np.mean((beliefs - true_labels) ** 2))


def flow_entropy(flow_features: np.ndarray, n_bins: int = 10) -> float:
    """
    Empirical entropy of order flow distribution.

    Higher entropy = more uniform / less predictable flow pattern.
    This is the quantity the execution agent tries to maximise.
    """
    if len(flow_features) < 10:
        return 0.0
    n_features = flow_features.shape[1]
    h_total = 0.0
    for f in range(n_features):
        x = flow_features[:, f]
        counts, _ = np.histogram(x, bins=n_bins)
        counts = counts + 1e-8
        probs = counts / counts.sum()
        h_total += float(stats.entropy(probs))
    return h_total / n_features


def adverse_selection_cost(
    fill_prices: List[float],
    post_fill_prices: List[float],
    fill_quantities: List[int],
    side: int = 1,  # MM side: +1 = sold to buyer, -1 = bought from seller
) -> float:
    """
    Adverse selection = price movement against MM position after fill.

    For each MM fill at price p, measure p_t+k - p where k is a
    short lookback. Positive AS = MM was adversely selected.
    """
    if not fill_prices:
        return 0.0
    total_qty = sum(fill_quantities)
    cost = sum(
        side * (post - fill) * qty
        for fill, post, qty in zip(fill_prices, post_fill_prices, fill_quantities)
    )
    return cost / max(total_qty, 1)


def twap_benchmark(prices: List[float]) -> float:
    """Time-weighted average price over the episode."""
    return float(np.mean(prices)) if prices else 0.0


def vwap_benchmark(prices: List[float], volumes: List[int]) -> float:
    """Volume-weighted average price over the episode."""
    total_vol = sum(volumes)
    if total_vol == 0:
        return 0.0
    return sum(p * v for p, v in zip(prices, volumes)) / total_vol


def spread_toxicity_correlation(
    spreads: List[float],
    beliefs: List[float],
) -> float:
    """
    Pearson correlation between MM spread and belief b_t.

    A well-trained MM should widen spreads when belief is high
    (more informed flow suspected). Positive correlation = correct behaviour.
    """
    if len(spreads) < 5 or len(beliefs) < 5:
        return float("nan")
    n = min(len(spreads), len(beliefs))
    result = stats.pearsonr(spreads[:n], beliefs[:n])
    return float(result[0])


def compute_summary_metrics(episodes: List[EpisodeMetrics]) -> Dict[str, float]:
    """Aggregate metrics across multiple evaluation episodes."""
    if not episodes:
        return {}

    def _mean(key: str) -> float:
        vals = [getattr(e, key) for e in episodes]
        finite = [v for v in vals if np.isfinite(v)]
        return float(np.mean(finite)) if finite else float("nan")

    def _std(key: str) -> float:
        vals = [getattr(e, key) for e in episodes]
        finite = [v for v in vals if np.isfinite(v)]
        return float(np.std(finite)) if len(finite) > 1 else 0.0

    return {
        "eval/exec/implementation_shortfall_mean": _mean("implementation_shortfall"),
        "eval/exec/implementation_shortfall_std": _std("implementation_shortfall"),
        "eval/exec/completion_rate_mean": _mean("completion_rate"),
        "eval/exec/mi_estimate_mean": _mean("mi_estimate"),
        "eval/exec/kl_divergence_mean": _mean("kl_informed_noise"),
        "eval/exec/flow_entropy_mean": _mean("flow_entropy"),
        "eval/mm/pnl_mean": _mean("mm_pnl"),
        "eval/mm/adverse_selection_mean": _mean("mm_adverse_selection"),
        "eval/mm/belief_accuracy_mean": _mean("mm_belief_accuracy"),
        "eval/mm/avg_spread_mean": _mean("avg_spread"),
        "eval/arb/pnl_mean": _mean("arb_pnl"),
        "eval/market/realized_vol_mean": _mean("realized_volatility"),
        "eval/market/total_volume_mean": _mean("total_volume"),
    }
