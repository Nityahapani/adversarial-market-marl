"""
Unit tests for evaluation metrics.
"""

import numpy as np
import pytest

from adversarial_market.evaluation.metrics import (
    adverse_selection_cost,
    belief_accuracy,
    belief_brier_score,
    flow_entropy,
    implementation_shortfall,
    kl_divergence_flows,
    spread_toxicity_correlation,
)
from adversarial_market.utils.math_utils import (
    compute_entropy,
    compute_kl_divergence,
    implementation_shortfall_bps,
    jensen_shannon_divergence,
    realized_volatility,
    total_variation_distance,
)


class TestImplementationShortfall:
    def test_zero_shortfall(self):
        """Fill at exact arrival price → IS = 0."""
        assert implementation_shortfall(100.0, [100.0], [10]) == pytest.approx(0.0)

    def test_positive_shortfall_for_buy(self):
        """Buy above arrival price → positive IS."""
        is_ = implementation_shortfall(100.0, [101.0], [10], side=1)
        assert is_ > 0

    def test_negative_shortfall_is_good_for_buy(self):
        """Buy below arrival price → negative IS (beneficial)."""
        is_ = implementation_shortfall(100.0, [99.0], [10], side=1)
        assert is_ < 0

    def test_volume_weighted_average(self):
        """IS uses VWAP of fills, not simple average."""
        is_ = implementation_shortfall(100.0, [101.0, 99.0], [9, 1], side=1)
        # VWAP = (101*9 + 99*1) / 10 = 100.8; IS = 100.8 - 100.0 = 0.8
        assert is_ == pytest.approx(0.8, rel=1e-4)

    def test_empty_fills_returns_zero(self):
        assert implementation_shortfall(100.0, [], []) == pytest.approx(0.0)


class TestKLDivergenceFlows:
    def test_identical_distributions_near_zero(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, (500, 4))
        y = rng.normal(0, 1, (500, 4))
        kl = kl_divergence_flows(x, y)
        assert kl < 0.5  # small but may not be exactly 0 due to sampling

    def test_different_distributions_positive(self):
        rng = np.random.default_rng(1)
        x = rng.normal(0, 1, (500, 4))
        y = rng.normal(5, 1, (500, 4))  # very different mean
        kl = kl_divergence_flows(x, y)
        assert kl > 1.0  # clearly distinguishable

    def test_insufficient_samples_returns_nan(self):
        x = np.random.randn(5, 4)
        y = np.random.randn(5, 4)
        kl = kl_divergence_flows(x, y)
        assert np.isnan(kl)


class TestBeliefAccuracy:
    def test_perfect_accuracy(self):
        beliefs = np.array([0.9, 0.8, 0.1, 0.2])
        labels = np.array([1, 1, 0, 0])
        assert belief_accuracy(beliefs, labels) == pytest.approx(1.0)

    def test_zero_accuracy(self):
        beliefs = np.array([0.1, 0.1, 0.9, 0.9])
        labels = np.array([1, 1, 0, 0])
        assert belief_accuracy(beliefs, labels) == pytest.approx(0.0)

    def test_random_belief_near_fifty_percent(self):
        rng = np.random.default_rng(42)
        beliefs = rng.uniform(0, 1, 1000)
        labels = rng.integers(0, 2, 1000)
        acc = belief_accuracy(beliefs, labels)
        assert 0.4 < acc < 0.6

    def test_brier_score_perfect(self):
        beliefs = np.array([1.0, 1.0, 0.0, 0.0])
        labels = np.array([1, 1, 0, 0])
        assert belief_brier_score(beliefs, labels) == pytest.approx(0.0)


class TestFlowEntropy:
    def test_uniform_flow_high_entropy(self):
        rng = np.random.default_rng(0)
        # Uniform: high entropy
        uniform = rng.uniform(0, 1, (500, 4))
        h_uniform = flow_entropy(uniform)
        # Concentrated: low entropy
        concentrated = np.ones((500, 4)) * 0.5 + rng.normal(0, 0.01, (500, 4))
        h_concentrated = flow_entropy(concentrated)
        assert h_uniform > h_concentrated

    def test_insufficient_data(self):
        x = np.random.randn(5, 4)
        assert flow_entropy(x) == pytest.approx(0.0)


class TestMathUtils:
    def test_kl_zero_for_identical(self):
        p = np.array([0.25, 0.25, 0.25, 0.25])
        kl = compute_kl_divergence(p, p.copy())
        assert kl < 1e-6

    def test_kl_positive_for_different(self):
        p = np.array([0.7, 0.3])
        q = np.array([0.3, 0.7])
        kl = compute_kl_divergence(p, q)
        assert kl > 0

    def test_entropy_uniform(self):
        p = np.ones(4) / 4
        h = compute_entropy(p)
        assert h == pytest.approx(np.log(4), rel=1e-4)

    def test_entropy_deterministic_near_zero(self):
        p = np.array([1.0, 0.0, 0.0])
        h = compute_entropy(p)
        assert h < 0.01

    def test_jsd_symmetric(self):
        p = np.array([0.6, 0.4])
        q = np.array([0.3, 0.7])
        assert jensen_shannon_divergence(p, q) == pytest.approx(
            jensen_shannon_divergence(q, p), rel=1e-4
        )

    def test_jsd_bounded(self):
        p = np.array([1.0, 0.0])
        q = np.array([0.0, 1.0])
        jsd = jensen_shannon_divergence(p, q)
        assert 0 <= jsd <= np.log(2) + 1e-6

    def test_tv_distance_bounded(self):
        p = np.array([0.9, 0.1])
        q = np.array([0.1, 0.9])
        tv = total_variation_distance(p, q)
        assert 0 <= tv <= 1.0

    def test_is_bps(self):
        is_bps = implementation_shortfall_bps(100.0, 100.1, side=1)
        assert is_bps == pytest.approx(10.0, rel=1e-4)

    def test_realized_vol_constant_prices(self):
        prices = [100.0] * 50
        vol = realized_volatility(prices)
        assert vol == pytest.approx(0.0, abs=1e-10)

    def test_realized_vol_positive_for_varying(self):
        rng = np.random.default_rng(0)
        prices = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 100)))
        vol = realized_volatility(prices.tolist())
        assert vol > 0


class TestAdverseSelection:
    def test_no_fills_returns_zero(self):
        assert adverse_selection_cost([], [], []) == pytest.approx(0.0)

    def test_price_moves_against_mm(self):
        """MM sells at 100.1, price moves up to 100.5 → adverse selection."""
        cost = adverse_selection_cost([100.1], [100.5], [10], side=1)
        assert cost > 0


class TestSpreadToxicityCorrelation:
    def test_positive_correlation(self):
        """Spread should increase with belief."""
        beliefs = np.linspace(0.1, 0.9, 50)
        spreads = 0.02 + 0.1 * beliefs + np.random.default_rng(0).normal(0, 0.005, 50)
        corr = spread_toxicity_correlation(spreads.tolist(), beliefs.tolist())
        assert corr > 0.8

    def test_short_series_returns_nan(self):
        corr = spread_toxicity_correlation([0.02, 0.03], [0.4, 0.6])
        assert np.isnan(corr)
