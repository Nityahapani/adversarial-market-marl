"""
Unit tests for execution policy baselines and the learned policy wrapper.
"""

import numpy as np
import pytest

from adversarial_market.networks.execution_policy import (
    AdaptiveCamouflagePolicy,
    LearnedPolicy,
    TWAPPolicy,
    VWAPPolicy,
    make_benchmark_policies,
)

HORIZON = 50
MAX_SIZE = 10


@pytest.fixture
def twap():
    p = TWAPPolicy(horizon=HORIZON, max_order_size=MAX_SIZE)
    p.reset(initial_inventory=100, arrival_price=100.0)
    return p


@pytest.fixture
def vwap():
    p = VWAPPolicy(horizon=HORIZON, max_order_size=MAX_SIZE)
    p.reset(initial_inventory=100, arrival_price=100.0)
    return p


# ── TWAPPolicy ─────────────────────────────────────────────────────────────


class TestTWAPPolicy:
    def test_action_shape(self, twap):
        obs = np.random.randn(80).astype(np.float32)
        action, lp, ent = twap.act(obs, remaining=100, time_remaining=1.0, mid_price=100.0)
        assert action.shape == (3,)
        assert lp == pytest.approx(0.0)
        assert ent == pytest.approx(0.0)

    def test_action_is_market_order(self, twap):
        obs = np.zeros(80, dtype=np.float32)
        action, _, _ = twap.act(obs, remaining=100, time_remaining=1.0, mid_price=100.0)
        assert action[2] < 0  # order_type_logit <= 0 → market

    def test_size_frac_in_range(self, twap):
        obs = np.zeros(80, dtype=np.float32)
        for t in np.linspace(1.0, 0.1, 10):
            remaining = int(t * 100)
            action, _, _ = twap.act(obs, remaining=remaining, time_remaining=t, mid_price=100.0)
            assert 0.0 <= action[0] <= 1.0

    def test_equal_slices_at_uniform_pace(self, twap):
        """TWAP should produce approximately equal-size orders each step."""
        obs = np.zeros(80, dtype=np.float32)
        sizes = []
        remaining = 100
        for step in range(10):
            t_rem = 1.0 - step / HORIZON
            action, _, _ = twap.act(obs, remaining=remaining, time_remaining=t_rem, mid_price=100.0)
            size = int(action[0] * MAX_SIZE)
            sizes.append(size)
            remaining = max(0, remaining - size)
        # Sizes should not vary wildly
        assert np.std(sizes) < 5

    def test_handles_zero_remaining(self, twap):
        obs = np.zeros(80, dtype=np.float32)
        action, _, _ = twap.act(obs, remaining=0, time_remaining=0.5, mid_price=100.0)
        assert action[0] == pytest.approx(0.0)


# ── VWAPPolicy ─────────────────────────────────────────────────────────────


class TestVWAPPolicy:
    def test_action_shape(self, vwap):
        obs = np.zeros(80, dtype=np.float32)
        action, lp, ent = vwap.act(obs, remaining=100, time_remaining=1.0, mid_price=100.0)
        assert action.shape == (3,)

    def test_volume_profile_sums_to_one(self, vwap):
        assert vwap._volume_profile.sum() == pytest.approx(1.0, rel=1e-5)

    def test_u_shaped_profile(self, vwap):
        """Volume should be higher at start and end than at midday."""
        profile = vwap._volume_profile
        n = len(profile)
        start = profile[:5].mean()
        mid = profile[n // 2 - 2 : n // 2 + 2].mean()
        end = profile[-5:].mean()
        assert start > mid
        assert end > mid

    def test_size_frac_in_range(self, vwap):
        obs = np.zeros(80, dtype=np.float32)
        for step in range(HORIZON):
            action, _, _ = vwap.act(
                obs, remaining=100, time_remaining=1.0 - step / HORIZON, mid_price=100.0
            )
            assert 0.0 <= action[0] <= 1.0


# ── AdaptiveCamouflagePolicy ───────────────────────────────────────────────


class TestAdaptiveCamouflagePolicy:
    def test_noise_injection_rate(self):
        """With p_noise=0.5, approximately 50% of actions should be noise."""
        base = TWAPPolicy(horizon=HORIZON, max_order_size=MAX_SIZE)
        base.reset(100, 100.0)
        cam = AdaptiveCamouflagePolicy(base, p_noise=0.5, rng_seed=42)
        cam.reset(100, 100.0)
        obs = np.zeros(80, dtype=np.float32)

        # TWAP always emits size_frac ≈ constant; noise emits random
        sizes = []
        for _ in range(200):
            action, _, _ = cam.act(obs, remaining=100, time_remaining=0.5, mid_price=100.0)
            sizes.append(action[0])

        # With noise injection, variance should be higher than pure TWAP
        assert np.std(sizes) > 0.05

    def test_zero_noise_equals_base(self):
        base = TWAPPolicy(horizon=HORIZON, max_order_size=MAX_SIZE)
        base.reset(100, 100.0)
        cam = AdaptiveCamouflagePolicy(base, p_noise=0.0)
        cam.reset(100, 100.0)
        obs = np.zeros(80, dtype=np.float32)

        action_cam, _, _ = cam.act(obs, remaining=50, time_remaining=0.5, mid_price=100.0)
        base.reset(100, 100.0)
        action_base, _, _ = base.act(obs, remaining=50, time_remaining=0.5, mid_price=100.0)
        assert np.allclose(action_cam, action_base)


# ── LearnedPolicy ──────────────────────────────────────────────────────────


class TestLearnedPolicy:
    def test_act_returns_correct_shapes(self, debug_config):
        from adversarial_market.environment.lob_env import LOBEnvironment
        from adversarial_market.networks.actor_critic import ExecutionActor

        env = LOBEnvironment(debug_config)
        obs_dim = env.observation_space["execution"].shape[0]
        cfg = debug_config["networks"]["execution_actor"]
        actor = ExecutionActor(obs_dim=obs_dim, hidden_dims=cfg["hidden_dims"])
        policy = LearnedPolicy(actor)
        policy.reset(100, 100.0)
        obs = np.random.randn(obs_dim).astype(np.float32)
        action, lp, ent = policy.act(obs, remaining=50, time_remaining=0.5, mid_price=100.0)
        assert action.shape == (3,)


# ── Factory ────────────────────────────────────────────────────────────────


class TestMakeBenchmarkPolicies:
    def test_returns_all_policies(self):
        policies = make_benchmark_policies(horizon=50, max_order_size=10)
        assert "twap" in policies
        assert "vwap" in policies
        assert "twap_camouflage" in policies

    def test_all_policies_act(self):
        policies = make_benchmark_policies(horizon=50, max_order_size=10)
        obs = np.zeros(80, dtype=np.float32)
        for name, policy in policies.items():
            policy.reset(100, 100.0)
            action, _, _ = policy.act(obs, remaining=50, time_remaining=0.5, mid_price=100.0)
            assert action.shape == (3,), f"Wrong shape for {name}"
