"""
Pytest configuration and shared fixtures.

All fixtures here are available to every test in the suite without
explicit import. Scope is kept minimal (function-scoped by default)
to prevent state leakage between tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Ensure project root is on path regardless of where pytest is invoked from
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Config fixtures ────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def default_config():
    """Full default config (loaded once per session)."""
    from adversarial_market.utils.config import load_config

    return load_config("configs/default.yaml")


@pytest.fixture(scope="session")
def debug_config():
    """Fast debug config for integration tests."""
    from adversarial_market.utils.config import load_config

    return load_config("configs/fast_debug.yaml")


@pytest.fixture
def tiny_config(debug_config):
    """Minimal config override for unit tests — tiny networks, 10-step episodes."""
    import copy

    cfg = copy.deepcopy(debug_config)
    cfg["environment"]["max_steps_per_episode"] = 10
    cfg["agents"]["execution"]["inventory_lots"] = 5
    cfg["agents"]["execution"]["horizon"] = 10
    cfg["training"]["rollout_length"] = 10
    cfg["training"]["minibatch_size"] = 5
    for key in cfg["networks"]:
        if isinstance(cfg["networks"][key], dict):
            if "hidden_dims" in cfg["networks"][key]:
                cfg["networks"][key]["hidden_dims"] = [16]
            if "d_model" in cfg["networks"][key]:
                cfg["networks"][key]["d_model"] = 16
                cfg["networks"][key]["n_heads"] = 2
                cfg["networks"][key]["n_layers"] = 1
                cfg["networks"][key]["d_ff"] = 32
    return cfg


# ── Environment fixtures ───────────────────────────────────────────────────


@pytest.fixture
def env(debug_config):
    """Fresh LOBEnvironment instance for each test."""
    from adversarial_market.environment.lob_env import LOBEnvironment

    return LOBEnvironment(debug_config)


@pytest.fixture
def env_reset(env):
    """LOBEnvironment with a fresh episode started (seed=42)."""
    obs, _ = env.reset(seed=42)
    return env, obs


# ── Network fixtures ───────────────────────────────────────────────────────


@pytest.fixture
def exec_actor(debug_config):
    from adversarial_market.environment.lob_env import LOBEnvironment
    from adversarial_market.networks.actor_critic import ExecutionActor

    tmp_env = LOBEnvironment(debug_config)
    obs_dim = tmp_env.observation_space["execution"].shape[0]
    cfg = debug_config["networks"]["execution_actor"]
    return ExecutionActor(obs_dim=obs_dim, hidden_dims=cfg["hidden_dims"])


@pytest.fixture
def mm_actor(debug_config):
    from adversarial_market.environment.lob_env import LOBEnvironment
    from adversarial_market.networks.actor_critic import MarketMakerActor

    tmp_env = LOBEnvironment(debug_config)
    obs_dim = tmp_env.observation_space["market_maker"].shape[0]
    cfg = debug_config["networks"]["mm_actor"]
    return MarketMakerActor(obs_dim=obs_dim, hidden_dims=cfg["hidden_dims"])


@pytest.fixture
def arb_actor(debug_config):
    from adversarial_market.environment.lob_env import LOBEnvironment
    from adversarial_market.networks.actor_critic import ArbitrageActor

    tmp_env = LOBEnvironment(debug_config)
    obs_dim = tmp_env.observation_space["arbitrageur"].shape[0]
    cfg = debug_config["networks"]["arb_actor"]
    return ArbitrageActor(obs_dim=obs_dim, hidden_dims=cfg["hidden_dims"])


@pytest.fixture
def shared_critic(debug_config):
    from adversarial_market.environment.lob_env import LOBEnvironment
    from adversarial_market.networks.actor_critic import SharedCritic

    tmp_env = LOBEnvironment(debug_config)
    obs_spaces = tmp_env.observation_space
    global_dim = sum(obs_spaces[k].shape[0] for k in obs_spaces)
    cfg = debug_config["networks"]["shared_critic"]
    return SharedCritic(
        global_state_dim=global_dim,
        hidden_dims=cfg["hidden_dims"],
        use_layer_norm=cfg["use_layer_norm"],
    )


@pytest.fixture
def belief_transformer(debug_config):
    from adversarial_market.networks.belief_transformer import BeliefTransformer

    cfg = debug_config["networks"]["belief_transformer"]
    return BeliefTransformer(
        input_dim=5,
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        d_ff=cfg["d_ff"],
        dropout=0.0,
        max_seq_len=cfg["max_seq_len"],
    )


@pytest.fixture
def mine_estimator(debug_config):
    from adversarial_market.networks.mine_estimator import MINEEstimator

    cfg = debug_config["networks"]["mine"]
    return MINEEstimator(
        z_dim=1,
        f_dim=4,
        hidden_dims=tuple(cfg["hidden_dims"]),
        ema_decay=cfg["ema_decay"],
    )


# ── Data fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def random_lob_snapshot():
    """Random normalised LOB snapshot array, shape (80,)."""
    rng = np.random.default_rng(0)
    snap = rng.uniform(-0.01, 0.01, 80).astype(np.float32)
    snap[1::2] = np.abs(snap[1::2])  # sizes must be non-negative
    snap[1::2] /= snap[1::2].max() + 1e-8
    return snap


@pytest.fixture
def random_flow_sequence():
    """Random order flow feature sequence, shape (32, 5)."""
    rng = np.random.default_rng(1)
    return rng.standard_normal((32, 5)).astype(np.float32)


# ── Reproducibility helpers ────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def set_random_seeds():
    """Set all RNG seeds before each test for reproducibility."""
    np.random.seed(42)
    torch.manual_seed(42)
    yield
    # No teardown needed


# ── Markers ───────────────────────────────────────────────────────────────


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow (skip with -m 'not slow')")
    config.addinivalue_line("markers", "gpu: mark test as requiring CUDA")
    config.addinivalue_line("markers", "integration: mark as integration test")
