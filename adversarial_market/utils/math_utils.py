"""
Mathematical utilities for information-theoretic computations.

All functions are numpy-based for use outside the training loop.
Torch-based equivalents live in the MINE estimator module.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def compute_kl_divergence(
    p: np.ndarray,
    q: np.ndarray,
    epsilon: float = 1e-8,
) -> float:
    """
    Discrete KL divergence D_KL(P || Q).

    Args:
        p: probability distribution (will be normalised)
        q: probability distribution (will be normalised)
        epsilon: smoothing constant to avoid log(0)

    Returns:
        Scalar KL divergence in nats.
    """
    p = np.asarray(p, dtype=np.float64) + epsilon
    q = np.asarray(q, dtype=np.float64) + epsilon
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def compute_entropy(p: np.ndarray, epsilon: float = 1e-8) -> float:
    """Shannon entropy H(P) in nats."""
    p = np.asarray(p, dtype=np.float64) + epsilon
    p /= p.sum()
    return float(-np.sum(p * np.log(p)))


def empirical_entropy(
    samples: np.ndarray,
    n_bins: int = 20,
) -> float:
    """
    Estimate H(X) from samples using histogram density estimation.

    Args:
        samples: 1-D array of observations
        n_bins:  number of histogram bins

    Returns:
        Entropy estimate in nats.
    """
    counts, _ = np.histogram(samples, bins=n_bins)
    return compute_entropy(counts.astype(float))


def mutual_information_histogram(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int = 20,
) -> float:
    """
    Estimate I(X; Y) = H(X) + H(Y) - H(X, Y) via joint histogram.

    Args:
        x: (N,) samples
        y: (N,) samples (same length as x)
        n_bins: bins per dimension

    Returns:
        MI estimate in nats.
    """
    assert len(x) == len(y), "x and y must have equal length"
    joint_counts, _, _ = np.histogram2d(x, y, bins=n_bins)
    joint_counts = joint_counts.astype(float)

    hxy = compute_entropy(joint_counts.ravel())
    hx = empirical_entropy(x, n_bins)
    hy = empirical_entropy(y, n_bins)
    return max(0.0, hx + hy - hxy)


def jensen_shannon_divergence(
    p: np.ndarray,
    q: np.ndarray,
    epsilon: float = 1e-8,
) -> float:
    """
    Jensen-Shannon divergence (symmetric, bounded in [0, ln2]).

    JSD(P||Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
    where M = 0.5 * (P + Q)
    """
    p = np.asarray(p, dtype=np.float64) + epsilon
    q = np.asarray(q, dtype=np.float64) + epsilon
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    return 0.5 * float(stats.entropy(p, m)) + 0.5 * float(stats.entropy(q, m))


def total_variation_distance(p: np.ndarray, q: np.ndarray) -> float:
    """TV distance: 0.5 * sum |p_i - q_i|."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p /= p.sum()
    q /= q.sum()
    return 0.5 * float(np.sum(np.abs(p - q)))


def gae_lambda_returns(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute GAE-λ advantages and discounted returns.

    Args:
        rewards: (T,) reward sequence
        values:  (T+1,) value estimates (last entry = bootstrap value)
        dones:   (T,) episode termination flags
        gamma:   discount factor
        lam:     GAE lambda

    Returns:
        advantages: (T,) normalised GAE advantages
        returns:    (T,) discounted returns
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0

    for t in reversed(range(T)):
        next_val = values[t + 1] if t < T - 1 else values[T]
        delta = rewards[t] + gamma * next_val * (1.0 - dones[t]) - values[t]
        last_gae = delta + gamma * lam * (1.0 - dones[t]) * last_gae
        advantages[t] = last_gae

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    returns = advantages + values[:T]
    return advantages, returns


def implementation_shortfall_bps(
    arrival_price: float,
    avg_fill_price: float,
    side: int = 1,
) -> float:
    """
    Implementation shortfall in basis points.
    side=1 (buy): positive IS = paid more than arrival price (bad).
    side=-1 (sell): positive IS = received less than arrival price (bad).
    """
    if arrival_price == 0:
        return 0.0
    return side * (avg_fill_price - arrival_price) / arrival_price * 10_000


def ewma(series: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Exponentially weighted moving average."""
    out = np.empty_like(series, dtype=np.float64)
    out[0] = series[0]
    for i in range(1, len(series)):
        out[i] = alpha * series[i] + (1 - alpha) * out[i - 1]
    return out


def realized_volatility(
    prices: np.ndarray,
    annualize: bool = True,
    periods_per_year: int = 390,
) -> float:
    """
    Realised volatility from price series (close-to-close log returns).

    Args:
        prices:           Price series (length N).
        annualize:        If True, annualise by sqrt(periods_per_year).
        periods_per_year: Trading periods per year (default: 390 minutes).

    Returns:
        Volatility estimate (annualised if requested).
    """
    if len(prices) < 2:
        return 0.0
    log_returns = np.diff(np.log(np.asarray(prices, dtype=np.float64)))
    vol = float(np.std(log_returns))
    if annualize:
        vol *= np.sqrt(periods_per_year)
    return vol
