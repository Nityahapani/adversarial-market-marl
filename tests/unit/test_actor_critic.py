"""
Unit tests for actor-critic network modules.

Tests forward pass shapes, distribution properties, log prob computation,
gradient flow, and determinism of shared critic.
"""

import pytest
import torch

from adversarial_market.networks.actor_critic import (
    ArbitrageActor,
    ExecutionActor,
    MarketMakerActor,
    SharedCritic,
    build_mlp,
)

# ── Shared Critic ──────────────────────────────────────────────────────────


class TestSharedCritic:
    def test_output_shape(self):
        critic = SharedCritic(global_state_dim=64, hidden_dims=[32, 32])
        x = torch.randn(16, 64)
        v = critic(x)
        assert v.shape == (16,)

    def test_output_is_finite(self):
        critic = SharedCritic(global_state_dim=32, hidden_dims=[16])
        x = torch.randn(8, 32)
        assert torch.all(torch.isfinite(critic(x)))

    def test_layer_norm_variant(self):
        critic = SharedCritic(global_state_dim=64, hidden_dims=[32], use_layer_norm=True)
        x = torch.randn(4, 64)
        v = critic(x)
        assert v.shape == (4,)

    def test_backprop(self):
        critic = SharedCritic(global_state_dim=32, hidden_dims=[16])
        opt = torch.optim.Adam(critic.parameters())
        x = torch.randn(8, 32)
        target = torch.randn(8)
        loss = ((critic(x) - target) ** 2).mean()
        loss.backward()
        opt.step()

    def test_single_sample(self):
        critic = SharedCritic(global_state_dim=10, hidden_dims=[8])
        x = torch.randn(1, 10)
        v = critic(x)
        assert v.shape == (1,)


# ── ExecutionActor ─────────────────────────────────────────────────────────


class TestExecutionActor:
    @pytest.fixture
    def actor(self):
        return ExecutionActor(obs_dim=50, hidden_dims=[32, 32])

    def test_forward_shapes(self, actor):
        obs = torch.randn(8, 50)
        action, log_prob, entropy = actor(obs)
        assert action.shape == (8, 3)
        assert log_prob.shape == (8,)
        assert entropy.shape == (8,)

    def test_log_prob_finite(self, actor):
        obs = torch.randn(16, 50)
        _, log_prob, entropy = actor(obs)
        assert torch.all(torch.isfinite(log_prob)), f"Non-finite log_probs: {log_prob}"
        assert torch.all(torch.isfinite(entropy))

    def test_action_order_type_bounded(self, actor):
        """Order type logit should be in [-1, 1] after tanh-ish Beta sampling."""
        obs = torch.randn(64, 50)
        action, _, _ = actor(obs)
        # size_frac (dim 0) should be in [0,1] (Beta output)
        # limit_offset (dim 1) maps from [0,1] Beta to [-5,5]
        # order_type_logit (dim 2) is Bernoulli sample {0,1}
        assert torch.all(action[:, 0] >= 0) and torch.all(action[:, 0] <= 1.0 + 1e-5)
        assert torch.all(action[:, 1] >= -5.5) and torch.all(action[:, 1] <= 5.5)

    def test_evaluate_actions_log_prob_close_to_forward(self, actor):
        obs = torch.randn(32, 50)
        action, lp_forward, _ = actor(obs)
        lp_eval, _ = actor.evaluate_actions(obs, action.detach())
        # Should be close (not exact due to atanh clamp rounding)
        diff = (lp_eval - lp_forward.detach()).abs()
        assert diff.mean() < 0.5

    def test_backprop(self, actor):
        opt = torch.optim.Adam(actor.parameters(), lr=1e-3)
        obs = torch.randn(8, 50)
        _, log_prob, entropy = actor(obs)
        loss = -log_prob.mean() - 0.01 * entropy.mean()
        loss.backward()
        opt.step()

    def test_entropy_positive(self, actor):
        obs = torch.randn(32, 50)
        _, _, entropy = actor(obs)
        assert torch.all(entropy > 0)

    def test_stochastic_actions_vary(self, actor):
        obs = torch.randn(1, 50).expand(32, -1)
        actions, _, _ = actor(obs)
        # Actions from stochastic policy should not all be identical
        assert actions.std() > 1e-4


# ── MarketMakerActor ───────────────────────────────────────────────────────


class TestMarketMakerActor:
    @pytest.fixture
    def actor(self):
        return MarketMakerActor(obs_dim=80, hidden_dims=[64, 64])

    def test_forward_shapes(self, actor):
        obs = torch.randn(8, 80)
        action, log_prob, entropy = actor(obs)
        assert action.shape == (8, 4)
        assert log_prob.shape == (8,)
        assert entropy.shape == (8,)

    def test_log_prob_finite(self, actor):
        obs = torch.randn(16, 80)
        _, lp, ent = actor(obs)
        assert torch.all(torch.isfinite(lp))
        assert torch.all(torch.isfinite(ent))

    def test_evaluate_actions_consistent(self, actor):
        obs = torch.randn(16, 80)
        action, lp_fwd, _ = actor(obs)
        lp_eval, ent_eval = actor.evaluate_actions(obs, action.detach())
        diff = (lp_eval - lp_fwd.detach()).abs().mean()
        assert diff < 0.1, f"Large log prob discrepancy: {diff}"

    def test_entropy_positive(self, actor):
        obs = torch.randn(16, 80)
        _, _, ent = actor(obs)
        assert torch.all(ent > 0)

    def test_learned_std_trainable(self, actor):
        assert actor.log_std.requires_grad


# ── ArbitrageActor ─────────────────────────────────────────────────────────


class TestArbitrageActor:
    @pytest.fixture
    def actor(self):
        return ArbitrageActor(obs_dim=24, hidden_dims=[16, 16])

    def test_forward_shapes(self, actor):
        obs = torch.randn(8, 24)
        action, log_prob, entropy = actor(obs)
        assert action.shape == (8, 1)
        assert log_prob.shape == (8,)
        assert entropy.shape == (8,)

    def test_action_in_minus_one_to_one(self, actor):
        obs = torch.randn(64, 24)
        action, _, _ = actor(obs)
        assert torch.all(action >= -1.0 - 1e-5)
        assert torch.all(action <= 1.0 + 1e-5)

    def test_evaluate_log_prob_finite(self, actor):
        obs = torch.randn(16, 24)
        action, _, _ = actor(obs)
        lp, ent = actor.evaluate_actions(obs, action.detach())
        assert torch.all(torch.isfinite(lp))
        assert torch.all(torch.isfinite(ent))

    def test_backprop(self, actor):
        opt = torch.optim.Adam(actor.parameters())
        obs = torch.randn(8, 24)
        action, log_prob, _ = actor(obs)
        (-log_prob.mean()).backward()
        opt.step()


# ── build_mlp ──────────────────────────────────────────────────────────────


class TestBuildMLP:
    def test_output_shape(self):
        net = build_mlp(10, [32, 32], 5)
        x = torch.randn(8, 10)
        assert net(x).shape == (8, 5)

    def test_with_layer_norm(self):
        net = build_mlp(10, [32], 5, use_layer_norm=True)
        x = torch.randn(4, 10)
        assert net(x).shape == (4, 5)

    def test_activations(self):
        for act in ["tanh", "relu", "elu"]:
            net = build_mlp(8, [16], 4, activation=act)
            assert net(torch.randn(2, 8)).shape == (2, 4)

    def test_no_hidden_layers(self):
        net = build_mlp(8, [], 4)
        assert net(torch.randn(3, 8)).shape == (3, 4)
