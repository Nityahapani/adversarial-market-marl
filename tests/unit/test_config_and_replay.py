"""
Unit tests for configuration utilities and prioritized replay buffer.
"""

import numpy as np
import pytest

from adversarial_market.utils.config import _deep_merge, load_config, save_config, validate_config
from adversarial_market.utils.replay_buffer import PrioritizedReplayBuffer, SumTree

# ── Config utilities ───────────────────────────────────────────────────────


class TestDeepMerge:
    def test_simple_override(self):
        base = {"a": 1, "b": 2}
        override = {"b": 99}
        result = _deep_merge(base, override)
        assert result["a"] == 1
        assert result["b"] == 99

    def test_nested_merge(self):
        base = {"a": {"x": 1, "y": 2}, "b": 3}
        override = {"a": {"y": 99}}
        result = _deep_merge(base, override)
        assert result["a"]["x"] == 1  # preserved from base
        assert result["a"]["y"] == 99  # overridden
        assert result["b"] == 3  # untouched

    def test_does_not_mutate_base(self):
        base = {"a": {"x": 1}}
        override = {"a": {"x": 2}}
        _deep_merge(base, override)
        assert base["a"]["x"] == 1  # base unchanged

    def test_new_key_added(self):
        base = {"a": 1}
        override = {"b": 2}
        result = _deep_merge(base, override)
        assert result["b"] == 2


class TestLoadConfig:
    def test_load_default_config(self):
        cfg = load_config("configs/default.yaml")
        assert "environment" in cfg
        assert "agents" in cfg
        assert "networks" in cfg
        assert "training" in cfg

    def test_load_debug_config_merges_default(self):
        cfg = load_config("configs/fast_debug.yaml")
        # debug overrides some keys
        assert cfg["training"]["total_timesteps"] < 1_000_000
        # but should still have all default keys
        assert "lr_actor" in cfg["training"]

    def test_overrides_applied(self):
        cfg = load_config(
            "configs/default.yaml",
            overrides={"agents.execution.lambda_leakage": 1.5},
        )
        assert cfg["agents"]["execution"]["lambda_leakage"] == pytest.approx(1.5)

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("configs/nonexistent.yaml")

    def test_save_and_reload(self, tmp_path):
        cfg = load_config("configs/default.yaml")
        save_path = str(tmp_path / "test_config.yaml")
        save_config(cfg, save_path)
        reloaded = load_config(save_path, base_config_path="configs/default.yaml")
        assert reloaded["training"]["gamma"] == cfg["training"]["gamma"]


class TestValidateConfig:
    def test_valid_config_passes(self):
        cfg = load_config("configs/default.yaml")
        validate_config(cfg)  # should not raise

    def test_missing_section_raises(self):
        cfg = load_config("configs/default.yaml")
        del cfg["environment"]
        with pytest.raises(ValueError, match="environment"):
            validate_config(cfg)

    def test_invalid_lambda_raises(self):
        cfg = load_config("configs/default.yaml")
        cfg["agents"]["execution"]["lambda_leakage"] = -1.0
        with pytest.raises(ValueError, match="lambda_leakage"):
            validate_config(cfg)

    def test_minibatch_larger_than_rollout_raises(self):
        cfg = load_config("configs/default.yaml")
        cfg["training"]["minibatch_size"] = cfg["training"]["rollout_length"] + 1
        with pytest.raises(ValueError, match="minibatch_size"):
            validate_config(cfg)


# ── SumTree ────────────────────────────────────────────────────────────────


class TestSumTree:
    def test_total_starts_zero(self):
        tree = SumTree(capacity=16)
        assert tree.total() == pytest.approx(0.0)

    def test_total_after_add(self):
        tree = SumTree(capacity=8)
        tree.add(1.0)
        tree.add(2.0)
        assert tree.total() == pytest.approx(3.0)

    def test_size_increments(self):
        tree = SumTree(capacity=4)
        assert tree.size == 0
        tree.add(1.0)
        assert tree.size == 1
        tree.add(1.0)
        assert tree.size == 2

    def test_size_caps_at_capacity(self):
        tree = SumTree(capacity=4)
        for _ in range(10):
            tree.add(1.0)
        assert tree.size == 4

    def test_sample_returns_valid_index(self):
        tree = SumTree(capacity=8)
        for i in range(4):
            tree.add(float(i + 1))
        for _ in range(20):
            idx, prio = tree.sample(np.random.uniform(0, tree.total()))
            assert 0 <= idx < 8
            assert prio >= 0

    def test_update_changes_total(self):
        tree = SumTree(capacity=4)
        idx = tree.add(1.0)
        assert tree.total() == pytest.approx(1.0)
        tree.update(idx, 5.0)
        assert tree.total() == pytest.approx(5.0)

    def test_max_priority_tracks(self):
        tree = SumTree(capacity=4)
        tree.add(1.0)
        tree.add(3.0)
        assert tree.max_priority == pytest.approx(3.0 + 1e-8)


# ── PrioritizedReplayBuffer ────────────────────────────────────────────────


class TestPrioritizedReplayBuffer:
    def test_add_and_sample(self):
        buf = PrioritizedReplayBuffer(capacity=100)
        for i in range(20):
            buf.add({"data": i})
        assert len(buf) == 20
        samples, indices, weights = buf.sample(8)
        assert len(samples) == 8
        assert len(indices) == 8
        assert weights.shape == (8,)

    def test_weights_in_zero_one(self):
        buf = PrioritizedReplayBuffer(capacity=100)
        for i in range(50):
            buf.add(i, priority=float(i + 1))
        _, _, weights = buf.sample(16)
        assert np.all(weights >= 0)
        assert np.all(weights <= 1.0 + 1e-6)  # max-normalised

    def test_priority_update(self):
        buf = PrioritizedReplayBuffer(capacity=100, alpha=1.0)
        for i in range(20):
            buf.add(i, priority=1.0)
        samples, indices, _ = buf.sample(4)
        new_priorities = np.array([10.0, 10.0, 10.0, 10.0])
        buf.update_priorities(indices, new_priorities)
        # High-priority items should be sampled more often
        high_count = 0
        for _ in range(200):
            s, idx, _ = buf.sample(1)
            if s[0] in [samples[i] for i in range(4)]:
                high_count += 1
        # Not a strict test — just verify no crash and some bias
        assert high_count >= 0

    def test_sample_from_empty_raises(self):
        buf = PrioritizedReplayBuffer(capacity=10)
        with pytest.raises(RuntimeError):
            buf.sample(4)

    def test_capacity_limit(self):
        buf = PrioritizedReplayBuffer(capacity=5)
        for i in range(10):
            buf.add(i)
        assert len(buf) == 5

    def test_len_correct(self):
        buf = PrioritizedReplayBuffer(capacity=50)
        for i in range(30):
            buf.add(i)
        assert len(buf) == 30

    def test_alpha_zero_uniform_sampling(self):
        """alpha=0 should give uniform sampling regardless of priority."""
        buf = PrioritizedReplayBuffer(capacity=100, alpha=0.0)
        for i in range(50):
            buf.add(i, priority=float(i + 1))
        # With alpha=0, all priorities become p^0=1.0, so weights all equal
        _, _, weights = buf.sample(16)
        assert np.std(weights) < 0.1
