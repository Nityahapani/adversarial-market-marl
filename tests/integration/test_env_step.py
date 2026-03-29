"""
Integration tests for the LOB environment step cycle.

Tests that the environment resets and steps without error,
that observations have correct shapes, rewards are finite,
and multi-episode dynamics are consistent.
"""

import numpy as np
import pytest

from adversarial_market.environment.lob_env import LOBEnvironment
from adversarial_market.utils.config import load_config


@pytest.fixture
def debug_config():
    return load_config("configs/fast_debug.yaml")


@pytest.fixture
def env(debug_config):
    return LOBEnvironment(debug_config)


class TestEnvironmentReset:
    def test_reset_returns_obs_and_info(self, env):
        obs, info = env.reset(seed=42)
        assert isinstance(obs, dict)
        assert isinstance(info, dict)

    def test_obs_keys(self, env):
        obs, _ = env.reset()
        assert set(obs.keys()) == {"execution", "market_maker", "arbitrageur"}

    def test_obs_shapes(self, env):
        obs, _ = env.reset()
        for key, space in env.observation_space.items():
            assert obs[key].shape == space.shape, f"Shape mismatch for {key}"
        assert obs["execution"].dtype == np.float32
        assert obs["market_maker"].dtype == np.float32
        assert obs["arbitrageur"].dtype == np.float32

    def test_initial_mid_price(self, env, debug_config):
        env.reset()
        expected = debug_config["environment"]["initial_mid_price"]
        assert env.state.mid_price == pytest.approx(expected, rel=0.05)

    def test_initial_exec_inventory(self, env, debug_config):
        env.reset()
        expected = debug_config["agents"]["execution"]["inventory_lots"]
        assert env.state.exec_remaining_inventory == expected

    def test_seeded_reset_reproducible(self, env):
        obs1, _ = env.reset(seed=7)
        obs2, _ = env.reset(seed=7)
        for key in obs1:
            assert np.allclose(obs1[key], obs2[key])


class TestEnvironmentStep:
    def _zero_actions(self, env):
        return {
            "execution": np.zeros(3, dtype=np.float32),
            "market_maker": np.array([-1.0, 1.0, 0.5, 0.5], dtype=np.float32),
            "arbitrageur": np.zeros(1, dtype=np.float32),
        }

    def test_step_returns_correct_keys(self, env):
        env.reset()
        obs, rewards, terminated, truncated, info = env.step(self._zero_actions(env))
        for key in ("execution", "market_maker", "arbitrageur"):
            assert key in obs
            assert key in rewards
            assert key in terminated
            assert key in truncated

    def test_rewards_are_finite(self, env):
        env.reset()
        for _ in range(5):
            _, rewards, _, _, _ = env.step(self._zero_actions(env))
        for key, r in rewards.items():
            assert np.isfinite(r), f"Non-finite reward for {key}: {r}"

    def test_obs_values_are_finite(self, env):
        env.reset()
        obs, _, _, _, _ = env.step(self._zero_actions(env))
        for key, arr in obs.items():
            assert np.all(np.isfinite(arr)), f"Non-finite obs for {key}"

    def test_step_counter_increments(self, env):
        env.reset()
        assert env._step == 0
        env.step(self._zero_actions(env))
        assert env._step == 1

    def test_episode_terminates(self, env, debug_config):
        env.reset()
        max_steps = debug_config["environment"]["max_steps_per_episode"]
        terminated_count = 0
        for _ in range(max_steps + 5):
            _, _, terminated, _, _ = env.step(self._zero_actions(env))
            if any(terminated.values()):
                terminated_count += 1
                break
        assert terminated_count >= 1

    def test_inventory_decreases_with_market_buys(self, env):
        env.reset(seed=0)
        initial_inv = env.state.exec_remaining_inventory
        # Submit aggressive market buy
        actions = {
            "execution": np.array([1.0, 0.0, -1.0], dtype=np.float32),  # market order
            "market_maker": np.array([-1.0, 1.0, 0.5, 0.5], dtype=np.float32),
            "arbitrageur": np.zeros(1, dtype=np.float32),
        }
        for _ in range(5):
            env.step(actions)
        # Inventory should not have increased
        assert env.state.exec_remaining_inventory <= initial_inv


class TestFlowBuffer:
    def test_flow_buffer_populated_after_steps(self, env):
        env.reset()
        actions = {
            "execution": np.array([0.5, 0.0, 1.0], dtype=np.float32),
            "market_maker": np.array([-1.0, 1.0, 0.5, 0.5], dtype=np.float32),
            "arbitrageur": np.zeros(1, dtype=np.float32),
        }
        for _ in range(10):
            env.step(actions)
        buf = env.get_flow_buffer()
        assert len(buf) > 0
        assert buf.shape[1] == 4  # 4 flow features

    def test_flow_buffer_shape(self, env):
        env.reset()
        buf = env.get_flow_buffer()
        assert buf.ndim == 2
        assert buf.shape[1] == 4


class TestMultiEpisode:
    def test_multiple_episodes_no_crash(self, env):
        for ep in range(3):
            env.reset(seed=ep)
            done = False
            step = 0
            while not done and step < 30:
                actions = {
                    "execution": np.array([0.1, 0.0, -1.0], dtype=np.float32),
                    "market_maker": np.array([-1.0, 1.0, 0.3, 0.3], dtype=np.float32),
                    "arbitrageur": np.zeros(1, dtype=np.float32),
                }
                _, _, terminated, _, _ = env.step(actions)
                done = any(terminated.values())
                step += 1

    def test_reset_clears_flow_buffer(self, env):
        env.reset()
        actions = {
            "execution": np.array([0.5, 0.0, 1.0], dtype=np.float32),
            "market_maker": np.array([-1.0, 1.0, 0.5, 0.5], dtype=np.float32),
            "arbitrageur": np.zeros(1, dtype=np.float32),
        }
        for _ in range(5):
            env.step(actions)
        assert len(env.get_flow_buffer()) > 0

        env.reset()
        # After reset, buffer should be empty
        assert len(env.get_flow_buffer()) == 0
