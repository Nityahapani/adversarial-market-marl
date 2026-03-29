"""
Visualization utilities for emergent market dynamics.

Produces publication-quality plots of:
  - Detectability phase transition (KL vs λ)
  - Belief trajectory over an episode
  - Spread dynamics vs perceived toxicity
  - Order flow entropy over training
  - Implementation shortfall vs leakage penalty
  - Price trajectory with agent activity overlay
"""

from __future__ import annotations

from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

# ── Style ──────────────────────────────────────────────────────────────────

COLORS = {
    "exec": "#534AB7",  # purple
    "mm": "#0F6E56",  # teal
    "arb": "#993C1D",  # coral
    "noise": "#888780",  # gray
    "phase": "#E24B4A",  # red for phase boundary
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.5,
        "lines.linewidth": 1.5,
        "figure.dpi": 150,
    }
)


# ── Phase transition plot ──────────────────────────────────────────────────


def plot_phase_transition(
    lambda_values: List[float],
    kl_divergences: List[float],
    impl_shortfalls: List[float],
    belief_accuracies: List[float],
    kl_stds: Optional[List[float]] = None,
    is_stds: Optional[List[float]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    The central result: KL divergence and IS as functions of λ.

    Shows the detectability phase transition — the critical λ* where
    informed flow becomes statistically indistinguishable from noise.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    lam = np.array(lambda_values)

    # Left: KL divergence
    ax = axes[0]
    ax.plot(lam, kl_divergences, color=COLORS["exec"], label="D_KL(informed || noise)")
    if kl_stds:
        ax.fill_between(
            lam,
            np.array(kl_divergences) - np.array(kl_stds),
            np.array(kl_divergences) + np.array(kl_stds),
            alpha=0.15,
            color=COLORS["exec"],
        )

    # Find and mark phase transition (where KL drops below threshold)
    kl_arr = np.array(kl_divergences)
    threshold = 0.05 * kl_arr.max()
    transition_idx = np.argmax(kl_arr < threshold)
    if 0 < transition_idx < len(lam) - 1:
        ax.axvline(
            lam[transition_idx],
            color=COLORS["phase"],
            linestyle="--",
            linewidth=1.0,
            label=f"λ* ≈ {lam[transition_idx]:.2f}",
        )

    ax.set_xlabel("Obfuscation penalty λ")
    ax.set_ylabel("KL divergence")
    ax.set_title("Detectability vs obfuscation intensity")
    ax.legend(frameon=False, fontsize=9)
    ax.set_ylim(bottom=0)

    # Right: Implementation shortfall
    ax = axes[1]
    ax.plot(lam, impl_shortfalls, color=COLORS["mm"], label="Implementation shortfall")
    if is_stds:
        ax.fill_between(
            lam,
            np.array(impl_shortfalls) - np.array(is_stds),
            np.array(impl_shortfalls) + np.array(is_stds),
            alpha=0.15,
            color=COLORS["mm"],
        )
    ax2 = ax.twinx()
    ax2.plot(lam, belief_accuracies, color=COLORS["arb"], linestyle=":", label="MM belief accuracy")
    ax2.set_ylabel("MM belief accuracy", color=COLORS["arb"])
    ax2.tick_params(axis="y", labelcolor=COLORS["arb"])
    ax2.set_ylim(0, 1)

    if 0 < transition_idx < len(lam) - 1:
        ax.axvline(lam[transition_idx], color=COLORS["phase"], linestyle="--", linewidth=1.0)

    ax.set_xlabel("Obfuscation penalty λ")
    ax.set_ylabel("Implementation shortfall")
    ax.set_title("Execution cost and inference accuracy vs λ")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, frameon=False, fontsize=9)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ── Episode belief trajectory ──────────────────────────────────────────────


def plot_episode_belief(
    beliefs: List[float],
    spreads: List[float],
    exec_order_steps: List[int],
    prices: List[float],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot market maker belief b_t, spread, and price over an episode.
    Marks steps where execution agent submitted orders.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    steps = np.arange(len(beliefs))

    # Price
    axes[0].plot(steps, prices, color=COLORS["noise"], linewidth=0.8)
    for s in exec_order_steps:
        axes[0].axvline(s, color=COLORS["exec"], alpha=0.3, linewidth=0.5)
    axes[0].set_ylabel("Mid price")
    axes[0].set_title("Episode dynamics")

    # Belief
    axes[1].plot(steps, beliefs, color=COLORS["mm"])
    axes[1].axhline(0.5, color=COLORS["noise"], linestyle=":", linewidth=0.8)
    axes[1].fill_between(
        steps,
        0.5,
        beliefs,
        where=np.array(beliefs) > 0.5,
        alpha=0.2,
        color=COLORS["mm"],
        label="b_t > 0.5",
    )
    axes[1].set_ylabel("MM belief b_t")
    axes[1].set_ylim(0, 1)
    axes[1].legend(frameon=False, fontsize=9)

    # Spread
    axes[2].plot(steps, spreads, color=COLORS["arb"])
    axes[2].set_ylabel("Bid-ask spread")
    axes[2].set_xlabel("Environment step")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ── Training curves ────────────────────────────────────────────────────────


def plot_training_curves(
    timesteps: List[int],
    exec_rewards: List[float],
    mm_rewards: List[float],
    arb_rewards: List[float],
    mi_estimates: List[float],
    phase_switches: Optional[List[int]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot per-agent rewards and MI estimate over training."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 6))
    ts = np.array(timesteps)

    def smooth(x: list, w: int = 20) -> list:
        if len(x) >= w:
            return list(np.convolve(x, np.ones(w) / w, mode="valid"))
        return x

    for ax, vals, label, color in zip(
        axes.flat,
        [exec_rewards, mm_rewards, arb_rewards, mi_estimates],
        ["Execution reward", "Market maker reward", "Arbitrageur reward", "MI estimate I(z;F_t)"],
        [COLORS["exec"], COLORS["mm"], COLORS["arb"], COLORS["noise"]],
    ):
        ax.plot(ts[: len(smooth(vals))], smooth(vals), color=color)
        if phase_switches:
            for ps in phase_switches:
                ax.axvline(ps, color="#D3D1C7", linewidth=0.4, zorder=0)
        ax.set_ylabel(label)
        ax.set_xlabel("Timesteps")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ── Order flow heatmap ─────────────────────────────────────────────────────


def plot_flow_heatmap(
    informed_flows: np.ndarray,
    noise_flows: np.ndarray,
    feature_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Side-by-side distribution comparison of informed vs noise order flow.
    Each feature is shown as a KDE/histogram row.
    """
    if feature_names is None:
        feature_names = ["size_frac", "limit_offset", "order_type", "time_frac"]

    n_features = min(len(feature_names), informed_flows.shape[1], noise_flows.shape[1])
    fig, axes = plt.subplots(1, n_features, figsize=(4 * n_features, 3.5))
    if n_features == 1:
        axes = [axes]

    for i, (ax, name) in enumerate(zip(axes, feature_names[:n_features])):
        lo = min(informed_flows[:, i].min(), noise_flows[:, i].min())
        hi = max(informed_flows[:, i].max(), noise_flows[:, i].max())
        bins = np.linspace(lo, hi, 30)

        ax.hist(
            informed_flows[:, i],
            bins=bins,
            density=True,
            alpha=0.6,
            color=COLORS["exec"],
            label="Informed",
        )
        ax.hist(
            noise_flows[:, i],
            bins=bins,
            density=True,
            alpha=0.6,
            color=COLORS["noise"],
            label="Noise",
        )
        ax.set_title(name)
        ax.legend(frameon=False, fontsize=8)
        ax.set_ylabel("Density" if i == 0 else "")

    fig.suptitle("Order flow distribution: informed vs noise", y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig
