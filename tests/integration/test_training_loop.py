"""
Integration test — training loop smoke test.

Verifies that the full MAPPO training loop runs end-to-end for a small
number of steps without errors, NaN losses, or shape mismatches.
"""

import pytest


@pytest.fixture
def debug_config():
    from adversarial_market.utils.config import load_config

    return load_config("configs/fast_debug.yaml")


class TestTrainingSmoke:
    def test_trainer_initializes(self, debug_config):
        from adversarial_market.training.trainer import MARLTrainer

        trainer = MARLTrainer(debug_config)
        assert trainer.exec_actor is not None
        assert trainer.mm_actor is not None
        assert trainer.arb_actor is not None
        assert trainer.critic is not None
        assert trainer.belief_transformer is not None
        assert trainer.mine is not None

    def test_action_collection_shapes(self, debug_config):

        from adversarial_market.training.trainer import MARLTrainer

        trainer = MARLTrainer(debug_config)
        obs, _ = trainer.env.reset(seed=0)

        actions, log_probs, values, entropies = trainer._get_actions(obs)

        assert "execution" in actions
        assert "market_maker" in actions
        assert "arbitrageur" in actions

        exec_act = actions["execution"]
        assert exec_act.shape == (3,)

        mm_act = actions["market_maker"]
        assert mm_act.shape == (4,)

        arb_act = actions["arbitrageur"]
        assert arb_act.shape == (1,)

    def test_global_state_shape(self, debug_config):

        from adversarial_market.training.trainer import MARLTrainer

        trainer = MARLTrainer(debug_config)
        obs, _ = trainer.env.reset()
        gs = trainer._build_global_state(obs)
        expected_dim = (
            trainer.env.observation_space["execution"].shape[0]
            + trainer.env.observation_space["market_maker"].shape[0]
            + trainer.env.observation_space["arbitrageur"].shape[0]
        )
        assert gs.shape == (expected_dim,)

    def test_rollout_buffer_fills(self, debug_config):

        from adversarial_market.training.trainer import MARLTrainer

        cfg = dict(debug_config)
        cfg["training"]["rollout_length"] = 16
        trainer = MARLTrainer(cfg)
        obs, _ = trainer.env.reset()

        for _ in range(16):
            actions, log_probs, values, entropies = trainer._get_actions(obs)
            next_obs, rewards, terminated, _, _ = trainer.env.step(actions)
            gs = trainer._build_global_state(obs)
            trainer.buffer.add(
                obs={k: v for k, v in obs.items()},
                global_state=gs,
                actions={k: v for k, v in actions.items()},
                log_probs={k: float(v) for k, v in log_probs.items()},
                rewards=rewards,
                values={k: 0.0 for k in rewards},
                dones=terminated,
                entropies={k: float(v) for k, v in entropies.items()},
            )
            if any(terminated.values()):
                obs, _ = trainer.env.reset()
            else:
                obs = next_obs

        assert trainer.buffer.is_full()

    def test_ppo_update_runs_without_nan(self, debug_config):
        """PPO update should not produce NaN losses."""

        from adversarial_market.training.trainer import MARLTrainer

        cfg = dict(debug_config)
        cfg["training"]["rollout_length"] = 32
        cfg["training"]["minibatch_size"] = 16
        cfg["training"]["n_epochs"] = 2
        trainer = MARLTrainer(cfg)

        obs, _ = trainer.env.reset(seed=1)
        for _ in range(32):
            actions, log_probs, values, entropies = trainer._get_actions(obs)
            next_obs, rewards, terminated, _, _ = trainer.env.step(actions)
            gs = trainer._build_global_state(obs)
            trainer.buffer.add(
                obs={k: v for k, v in obs.items()},
                global_state=gs,
                actions={k: v for k, v in actions.items()},
                log_probs={k: float(v) for k, v in log_probs.items()},
                rewards=rewards,
                values={k: 0.0 for k in rewards},
                dones=terminated,
                entropies={k: float(v) for k, v in entropies.items()},
            )
            if any(terminated.values()):
                obs, _ = trainer.env.reset()
            else:
                obs = next_obs

        trainer.buffer.compute_returns_and_advantages(
            {"execution": 0.0, "market_maker": 0.0, "arbitrageur": 0.0}
        )
        metrics = trainer._run_ppo_updates()

        for k, v in metrics.items():
            assert v == v, f"NaN detected in metric: {k}"  # nan != nan


class TestAlternatingOptimizer:
    def test_phase_switches(self):
        from adversarial_market.training.alternating_opt import AlternatingOptimizer, Phase

        opt = AlternatingOptimizer(exec_phase_steps=10, mm_arb_phase_steps=10)
        assert opt.current_phase == Phase.EXECUTION

        opt.step(10)
        assert opt.current_phase == Phase.MM_ARB

        opt.step(10)
        assert opt.current_phase == Phase.EXECUTION

    def test_active_agents_correct(self):
        from adversarial_market.training.alternating_opt import AlternatingOptimizer

        opt = AlternatingOptimizer(exec_phase_steps=100, mm_arb_phase_steps=100)
        assert opt.active_agents == ["execution"]
        opt.step(100)
        assert set(opt.active_agents) == {"market_maker", "arbitrageur"}

    def test_state_dict_roundtrip(self):
        from adversarial_market.training.alternating_opt import AlternatingOptimizer

        opt = AlternatingOptimizer(exec_phase_steps=50, mm_arb_phase_steps=50)
        opt.step(50)
        d = opt.state_dict()
        opt2 = AlternatingOptimizer(exec_phase_steps=50, mm_arb_phase_steps=50)
        opt2.load_state_dict(d)
        assert opt2.current_phase == opt.current_phase
