"""
Unit tests for agent wrapper classes.
"""

import numpy as np
import pytest
import torch

from adversarial_market.agents.arbitrageur import ArbitrageAgent
from adversarial_market.agents.base_agent import BaseAgent
from adversarial_market.agents.execution_agent import ExecutionAgent
from adversarial_market.agents.market_maker import MarketMakerAgent

# ── Concrete BaseAgent subclass for testing ────────────────────────────────


class DummyAgent(BaseAgent):
    def build_networks(self):
        pass

    def act(self, obs, deterministic=False):
        return np.zeros(3, dtype=np.float32), 0.0, 1.0

    def update(self, batch):
        return {}


class TestBaseAgent:
    def test_repr(self):
        agent = DummyAgent(0, "dummy", {})
        r = repr(agent)
        assert "DummyAgent" in r
        assert "dummy" in r

    def test_freeze_unfreeze(self):
        import torch.nn as nn

        agent = DummyAgent(0, "test", {})
        net = agent.register_network(nn.Linear(4, 2))
        agent.freeze()
        assert not net.weight.requires_grad
        agent.unfreeze()
        assert net.weight.requires_grad

    def test_parameter_count(self):
        import torch.nn as nn

        agent = DummyAgent(0, "test", {})
        agent.register_network(nn.Linear(4, 2))  # 4*2 + 2 = 10 params
        assert agent.parameter_count() == 10

    def test_to_tensor(self):
        agent = DummyAgent(0, "test", {})
        arr = np.array([1.0, 2.0, 3.0])
        t = agent.to_tensor(arr)
        assert t.dtype == torch.float32
        assert t.shape == (3,)

    def test_save_load_roundtrip(self, tmp_path):
        import torch.nn as nn

        agent = DummyAgent(0, "test", {})
        net = agent.register_network(nn.Linear(4, 2))
        agent._step = 42
        save_path = str(tmp_path / "agent.pt")
        agent.save_state(save_path)

        # Create new agent, load state
        agent2 = DummyAgent(0, "test", {})
        net2 = agent2.register_network(nn.Linear(4, 2))
        agent2.load_state(save_path)
        assert agent2._step == 42
        assert torch.equal(net.weight.data, net2.weight.data)


# ── ExecutionAgent ─────────────────────────────────────────────────────────


@pytest.fixture
def exec_agent(debug_config):
    from adversarial_market.environment.lob_env import LOBEnvironment

    env = LOBEnvironment(debug_config)
    obs_dim = env.observation_space["execution"].shape[0]
    return ExecutionAgent(
        config=debug_config["agents"]["execution"],
        net_cfg=debug_config["networks"],
        obs_dim=obs_dim,
    )


class TestExecutionAgent:
    def test_instantiation(self, exec_agent):
        assert exec_agent.agent_id == 0
        assert exec_agent.name == "execution"

    def test_act_returns_correct_shapes(self, exec_agent):
        obs = np.random.randn(exec_agent.obs_dim).astype(np.float32)
        action, lp, ent = exec_agent.act(obs)
        assert action.shape == (3,)
        assert isinstance(lp, float)
        assert isinstance(ent, float)

    def test_episode_reset(self, exec_agent):
        exec_agent.reset_episode(arrival_price=100.0)
        assert exec_agent.remaining_inventory == exec_agent.inventory
        assert exec_agent.completion_rate == pytest.approx(0.0)

    def test_record_fill_reduces_inventory(self, exec_agent):
        exec_agent.reset_episode(100.0)
        initial = exec_agent.remaining_inventory
        exec_agent.record_fill(price=100.1, qty=5)
        assert exec_agent.remaining_inventory == initial - 5

    def test_implementation_shortfall_above_arrival(self, exec_agent):
        exec_agent.reset_episode(100.0)
        exec_agent.record_fill(price=101.0, qty=10)
        is_ = exec_agent.implementation_shortfall
        assert is_ > 0  # bought above arrival price

    def test_flow_tensor_none_when_empty(self, exec_agent):
        exec_agent.reset_episode(100.0)
        assert exec_agent.get_flow_tensor() is None

    def test_flow_tensor_after_records(self, exec_agent):
        exec_agent.reset_episode(100.0)
        for _ in range(5):
            exec_agent.record_flow(np.random.randn(4).astype(np.float32))
        ft = exec_agent.get_flow_tensor()
        assert ft is not None
        assert ft.shape == (5, 4)

    def test_parameter_count_positive(self, exec_agent):
        assert exec_agent.parameter_count() > 0


# ── MarketMakerAgent ───────────────────────────────────────────────────────


@pytest.fixture
def mm_agent(debug_config):
    from adversarial_market.environment.lob_env import LOBEnvironment

    env = LOBEnvironment(debug_config)
    obs_dim = env.observation_space["market_maker"].shape[0]
    return MarketMakerAgent(
        config=debug_config["agents"]["market_maker"],
        net_cfg=debug_config["networks"],
        obs_dim=obs_dim,
    )


class TestMarketMakerAgent:
    def test_instantiation(self, mm_agent):
        assert mm_agent.agent_id == 1
        assert mm_agent.name == "market_maker"

    def test_initial_belief_is_half(self, mm_agent):
        assert mm_agent.belief == pytest.approx(0.5)

    def test_observe_flow_event_adds_to_window(self, mm_agent):
        mm_agent.observe_flow_event(1.0, 0.001, 0.5, 1.0, 0.1)
        assert len(mm_agent._flow_window) == 1

    def test_update_belief_returns_float_in_range(self, mm_agent):
        for _ in range(5):
            mm_agent.observe_flow_event(1.0, 0.001, 0.5, 1.0, 0.1)
        b = mm_agent.update_belief()
        assert isinstance(b, float)
        assert 0.0 <= b <= 1.0

    def test_belief_history_appends(self, mm_agent):
        for _ in range(3):
            mm_agent.observe_flow_event(1.0, 0.0, 0.5, 0.0, 0.1)
        mm_agent.update_belief()
        mm_agent.update_belief()
        assert len(mm_agent.belief_history) == 2

    def test_implied_spread_multiplier_range(self, mm_agent):
        m = mm_agent.implied_spread_multiplier()
        assert 1.0 <= m <= 3.0

    def test_spread_multiplier_higher_with_higher_belief(self, mm_agent):
        mm_agent._belief = 0.1
        low = mm_agent.implied_spread_multiplier()
        mm_agent._belief = 0.9
        high = mm_agent.implied_spread_multiplier()
        assert high > low

    def test_reset_clears_state(self, mm_agent):
        mm_agent.observe_flow_event(1.0, 0.0, 0.5, 0.0, 0.1)
        mm_agent._pnl = 42.0
        mm_agent.reset_episode()
        assert len(mm_agent._flow_window) == 0
        assert mm_agent._pnl == 0.0
        assert mm_agent.belief == pytest.approx(0.5)


# ── ArbitrageAgent ─────────────────────────────────────────────────────────


@pytest.fixture
def arb_agent(debug_config):
    from adversarial_market.environment.lob_env import LOBEnvironment

    env = LOBEnvironment(debug_config)
    obs_dim = env.observation_space["arbitrageur"].shape[0]
    return ArbitrageAgent(
        config=debug_config["agents"]["arbitrageur"],
        net_cfg=debug_config["networks"],
        obs_dim=obs_dim,
    )


class TestArbitrageAgent:
    def test_instantiation(self, arb_agent):
        assert arb_agent.agent_id == 2
        assert arb_agent.name == "arbitrageur"

    def test_act_returns_correct_shapes(self, arb_agent):
        obs = np.random.randn(arb_agent.obs_dim).astype(np.float32)
        action, lp, ent = arb_agent.act(obs)
        assert action.shape == (1,)
        assert -1.0 <= action[0] <= 1.0

    def test_should_trade_buy(self, arb_agent):
        assert arb_agent.should_trade(0.5) == "buy"

    def test_should_trade_sell(self, arb_agent):
        assert arb_agent.should_trade(-0.5) == "sell"

    def test_should_trade_hold(self, arb_agent):
        assert arb_agent.should_trade(0.1) == "hold"

    def test_compute_quote_staleness_zero_when_fair(self, _):
        stale = ArbitrageAgent.compute_quote_staleness(
            best_bid=99.99,
            best_ask=100.01,
            mid_price=100.0,
            tick_size=0.01,
            fair_spread_ticks=2.0,
        )
        assert stale == pytest.approx(0.0)

    def test_compute_quote_staleness_positive_when_wide(self, _):
        stale = ArbitrageAgent.compute_quote_staleness(
            best_bid=99.90,
            best_ask=100.10,
            mid_price=100.0,
            tick_size=0.01,
            fair_spread_ticks=2.0,
        )
        assert stale > 0

    def test_record_snipe_updates_pnl(self, arb_agent):
        arb_agent.record_snipe(profit=0.05, belief_lag=0.3)
        assert arb_agent.episode_pnl == pytest.approx(0.05)
        assert arb_agent.n_snipes == 1

    def test_reset_clears_episode_state(self, arb_agent):
        arb_agent.record_snipe(profit=0.1, belief_lag=0.2)
        arb_agent.reset_episode()
        assert arb_agent.episode_pnl == 0.0
        assert arb_agent.n_snipes == 0

    @pytest.fixture
    def _(self):
        return None  # placeholder for static method tests that don't need self
