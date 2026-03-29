"""
Unit tests for the rollout buffer and PPO update logic.
"""

import numpy as np
import pytest
import torch

from adversarial_market.networks.actor_critic import ExecutionActor, SharedCritic
from adversarial_market.training.ppo_update import PPOUpdate
from adversarial_market.training.rollout_buffer import RolloutBuffer

# ── RolloutBuffer ──────────────────────────────────────────────────────────


class TestRolloutBuffer:
    OBS_DIM = 20
    GLOBAL_DIM = 60
    ACT_DIM = 3
    ROLLOUT_LEN = 64
    N_ENVS = 1

    @pytest.fixture
    def buffer(self):
        return RolloutBuffer(
            rollout_length=self.ROLLOUT_LEN,
            n_envs=self.N_ENVS,
            gamma=0.99,
            gae_lambda=0.95,
            device="cpu",
        )

    def _make_transition(self):
        obs = {
            k: np.random.randn(self.OBS_DIM).astype(np.float32)
            for k in ("execution", "market_maker", "arbitrageur")
        }
        gs = np.random.randn(self.GLOBAL_DIM).astype(np.float32)
        actions = {
            k: np.random.randn(self.ACT_DIM).astype(np.float32)
            for k in ("execution", "market_maker", "arbitrageur")
        }
        log_probs = {
            k: float(np.random.randn()) for k in ("execution", "market_maker", "arbitrageur")
        }
        rewards = {
            k: float(np.random.randn()) for k in ("execution", "market_maker", "arbitrageur")
        }
        values = {k: float(np.random.randn()) for k in ("execution", "market_maker", "arbitrageur")}
        dones = {k: False for k in ("execution", "market_maker", "arbitrageur")}
        entropies = {
            k: float(abs(np.random.randn())) for k in ("execution", "market_maker", "arbitrageur")
        }
        return obs, gs, actions, log_probs, rewards, values, dones, entropies

    def test_add_increments_pointer(self, buffer):
        assert buffer._ptr == 0
        obs, gs, a, lp, r, v, d, ent = self._make_transition()
        buffer.add(obs, gs, a, lp, r, v, d, ent)
        assert buffer._ptr == 1

    def test_is_full_after_rollout_length(self, buffer):
        for _ in range(self.ROLLOUT_LEN):
            buffer.add(*self._make_transition())
        assert buffer.is_full()

    def test_is_not_full_before(self, buffer):
        for _ in range(self.ROLLOUT_LEN - 1):
            buffer.add(*self._make_transition())
        assert not buffer.is_full()

    def test_compute_returns_runs(self, buffer):
        for _ in range(self.ROLLOUT_LEN):
            buffer.add(*self._make_transition())
        buffer.compute_returns_and_advantages(
            {"execution": 0.0, "market_maker": 0.0, "arbitrageur": 0.0}
        )
        # Should have _advantages and _returns on each agent buffer
        for key in buffer.AGENT_KEYS:
            buf = buffer._buffers[key]
            assert hasattr(buf, "advantages")
            assert hasattr(buf, "returns")

    def test_advantages_normalised(self, buffer):
        for _ in range(self.ROLLOUT_LEN):
            buffer.add(*self._make_transition())
        buffer.compute_returns_and_advantages(
            {"execution": 0.5, "market_maker": 0.3, "arbitrageur": 0.2}
        )
        adv = buffer._buffers["execution"].advantages
        assert abs(adv.mean()) < 0.1
        assert abs(adv.std() - 1.0) < 0.1

    def test_minibatch_generator_yields_correct_keys(self, buffer):
        for _ in range(self.ROLLOUT_LEN):
            buffer.add(*self._make_transition())
        buffer.compute_returns_and_advantages(
            {"execution": 0.0, "market_maker": 0.0, "arbitrageur": 0.0}
        )
        batch_count = 0
        for batch in buffer.get_minibatches(minibatch_size=16):
            for key in buffer.AGENT_KEYS:
                assert key in batch
                assert "obs" in batch[key]
                assert "advantages" in batch[key]
                assert "returns" in batch[key]
            assert "global" in batch
            batch_count += 1
        assert batch_count == self.ROLLOUT_LEN // 16

    def test_clear_resets_pointer(self, buffer):
        for _ in range(self.ROLLOUT_LEN):
            buffer.add(*self._make_transition())
        buffer.clear()
        assert buffer._ptr == 0
        assert not buffer.is_full()

    def test_minibatch_tensors_on_correct_device(self, buffer):
        for _ in range(self.ROLLOUT_LEN):
            buffer.add(*self._make_transition())
        buffer.compute_returns_and_advantages({k: 0.0 for k in buffer.AGENT_KEYS})
        for batch in buffer.get_minibatches(16):
            for key in buffer.AGENT_KEYS:
                assert batch[key]["obs"].device == torch.device("cpu")
            break


# ── PPOUpdate ──────────────────────────────────────────────────────────────


class TestPPOUpdate:
    OBS_DIM = 32
    GLOBAL_DIM = 96
    BATCH_SIZE = 16

    @pytest.fixture
    def actor_critic_ppo(self):
        actor = ExecutionActor(obs_dim=self.OBS_DIM, hidden_dims=[32])
        critic = SharedCritic(global_state_dim=self.GLOBAL_DIM, hidden_dims=[32])
        actor_opt = torch.optim.Adam(actor.parameters(), lr=1e-3)
        critic_opt = torch.optim.Adam(critic.parameters(), lr=1e-3)
        ppo = PPOUpdate(
            actor=actor,
            critic=critic,
            actor_optimizer=actor_opt,
            critic_optimizer=critic_opt,
            clip_range=0.2,
            value_coef=0.5,
            entropy_coef=0.01,
            max_grad_norm=0.5,
        )
        return actor, critic, ppo

    def _make_batch(self, actor, obs_dim, global_dim, B):
        """Generate a valid minibatch by sampling from the actor."""
        obs = torch.randn(B, obs_dim)
        global_states = torch.randn(B, global_dim)
        with torch.no_grad():
            actions, log_probs_old, _ = actor(obs)
        advantages = torch.randn(B)
        returns = torch.randn(B)
        values_old = torch.randn(B)
        return {
            "obs": obs,
            "global_states": global_states,
            "actions": actions.detach(),
            "log_probs_old": log_probs_old.detach(),
            "advantages": advantages,
            "returns": returns,
            "values_old": values_old,
        }

    def test_update_returns_metrics(self, actor_critic_ppo):
        actor, critic, ppo = actor_critic_ppo
        b = self._make_batch(actor, self.OBS_DIM, self.GLOBAL_DIM, self.BATCH_SIZE)
        metrics = ppo.update(**b)
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "entropy" in metrics
        assert "approx_kl" in metrics

    def test_metrics_are_finite(self, actor_critic_ppo):
        actor, critic, ppo = actor_critic_ppo
        b = self._make_batch(actor, self.OBS_DIM, self.GLOBAL_DIM, self.BATCH_SIZE)
        metrics = ppo.update(**b)
        for k, v in metrics.items():
            assert np.isfinite(v), f"Non-finite metric {k}: {v}"

    def test_actor_parameters_change(self, actor_critic_ppo):
        actor, critic, ppo = actor_critic_ppo
        params_before = [p.data.clone() for p in actor.parameters()]
        b = self._make_batch(actor, self.OBS_DIM, self.GLOBAL_DIM, self.BATCH_SIZE)
        ppo.update(**b)
        params_after = [p.data for p in actor.parameters()]
        changed = any(not torch.equal(p1, p2) for p1, p2 in zip(params_before, params_after))
        assert changed, "Actor parameters did not change after update"

    def test_vf_clipping_enabled(self):
        actor = ExecutionActor(obs_dim=16, hidden_dims=[16])
        critic = SharedCritic(global_state_dim=32, hidden_dims=[16])
        ppo = PPOUpdate(
            actor=actor,
            critic=critic,
            actor_optimizer=torch.optim.Adam(actor.parameters()),
            critic_optimizer=torch.optim.Adam(critic.parameters()),
            clip_range=0.2,
            clip_range_vf=0.2,
        )
        b = self._make_batch(actor, 16, 32, 8)
        metrics = ppo.update(**b)
        assert "value_loss" in metrics

    def test_extra_reward_injected(self, actor_critic_ppo):
        actor, critic, ppo = actor_critic_ppo
        b = self._make_batch(actor, self.OBS_DIM, self.GLOBAL_DIM, self.BATCH_SIZE)
        extra = torch.randn(self.BATCH_SIZE) * 0.1
        metrics = ppo.update(**b, extra_reward=extra)
        assert "policy_loss" in metrics

    def test_should_stop_early_with_large_kl(self, actor_critic_ppo):
        _, _, ppo = actor_critic_ppo
        ppo.target_kl = 0.01
        assert ppo.should_stop_early(0.02) is True
        assert ppo.should_stop_early(0.001) is False

    def test_no_target_kl_never_stops(self, actor_critic_ppo):
        _, _, ppo = actor_critic_ppo
        ppo.target_kl = None
        assert ppo.should_stop_early(100.0) is False
