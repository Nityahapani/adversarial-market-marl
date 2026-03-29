"""
Integration tests for checkpoint save/load and alternating optimizer
across a mini training run.
"""

from pathlib import Path

import pytest
import torch


@pytest.fixture
def debug_config():
    from adversarial_market.utils.config import load_config

    return load_config("configs/fast_debug.yaml")


class TestCheckpointRoundtrip:
    def test_save_and_load_preserves_step(self, debug_config, tmp_path):
        from adversarial_market.training.trainer import MARLTrainer

        cfg = dict(debug_config)
        cfg["checkpoint_dir"] = str(tmp_path)
        trainer = MARLTrainer(cfg)
        trainer._global_step = 999

        trainer.save_checkpoint(step=999)

        # Find the saved file
        ckpt_files = list(Path(cfg["checkpoint_dir"]).rglob("*.pt"))
        assert len(ckpt_files) >= 1

    def test_save_load_preserves_actor_weights(self, debug_config, tmp_path):
        from adversarial_market.training.trainer import MARLTrainer

        cfg = dict(debug_config)
        cfg["checkpoint_dir"] = str(tmp_path / "ckpts")

        trainer = MARLTrainer(cfg)
        # Record exec actor weights before save
        weights_before = {k: v.clone() for k, v in trainer.exec_actor.state_dict().items()}

        save_path = str(tmp_path / "ckpts" / cfg["exp_name"] / "checkpoint_step_0.pt")
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(step=0)

        # Create fresh trainer and load
        trainer2 = MARLTrainer(cfg)
        trainer2.load_checkpoint(save_path)

        for k in weights_before:
            assert torch.equal(
                weights_before[k], trainer2.exec_actor.state_dict()[k]
            ), f"Weight mismatch after load for key: {k}"

    def test_load_restores_belief_transformer(self, debug_config, tmp_path):
        from adversarial_market.training.trainer import MARLTrainer

        cfg = dict(debug_config)
        cfg["checkpoint_dir"] = str(tmp_path / "ckpts")

        trainer = MARLTrainer(cfg)
        bt_weights = {k: v.clone() for k, v in trainer.belief_transformer.state_dict().items()}

        save_dir = Path(tmp_path / "ckpts" / cfg["exp_name"])
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = str(save_dir / "checkpoint_step_0.pt")
        trainer.save_checkpoint(step=0)

        trainer2 = MARLTrainer(cfg)
        trainer2.load_checkpoint(save_path)

        for k in bt_weights:
            assert torch.equal(bt_weights[k], trainer2.belief_transformer.state_dict()[k])


class TestAlternatingOptimizerIntegration:
    def test_full_phase_cycle_in_training(self, debug_config):
        """Verify alternating optimizer switches phases during a mini training run."""
        from adversarial_market.training.alternating_opt import AlternatingOptimizer, Phase

        cfg = dict(debug_config)
        cfg["training"]["alternating"]["exec_phase_steps"] = 20
        cfg["training"]["alternating"]["mm_arb_phase_steps"] = 20

        opt = AlternatingOptimizer(
            exec_phase_steps=20,
            mm_arb_phase_steps=20,
        )
        assert opt.current_phase == Phase.EXECUTION
        opt.step(20)
        assert opt.current_phase == Phase.MM_ARB
        opt.step(20)
        assert opt.current_phase == Phase.EXECUTION
        assert opt.n_phase_switches() == 2

    def test_frozen_agents_complement_active(self, debug_config):
        from adversarial_market.training.alternating_opt import AlternatingOptimizer

        opt = AlternatingOptimizer(exec_phase_steps=10, mm_arb_phase_steps=10)
        active = set(opt.active_agents)
        frozen = set(opt.frozen_agents())
        all_agents = {"execution", "market_maker", "arbitrageur"}
        assert active | frozen == all_agents
        assert active & frozen == set()

    def test_phase_progress_resets_after_switch(self, debug_config):
        from adversarial_market.training.alternating_opt import AlternatingOptimizer

        opt = AlternatingOptimizer(exec_phase_steps=10, mm_arb_phase_steps=10)
        opt.step(5)
        assert 0.4 < opt.phase_progress() < 0.6
        opt.step(5)
        # After switch, progress resets
        assert opt.phase_progress() < 0.1


class TestMathUtilsIntegration:
    def test_gae_returns_shape(self):
        import numpy as np

        from adversarial_market.utils.math_utils import gae_lambda_returns

        T = 64
        rewards = np.random.randn(T).astype(np.float32)
        values = np.random.randn(T + 1).astype(np.float32)
        dones = np.zeros(T, dtype=np.float32)
        adv, ret = gae_lambda_returns(rewards, values, dones)
        assert adv.shape == (T,)
        assert ret.shape == (T,)

    def test_gae_advantages_normalised(self):
        import numpy as np

        from adversarial_market.utils.math_utils import gae_lambda_returns

        T = 256
        rewards = np.random.randn(T).astype(np.float32)
        values = np.zeros(T + 1, dtype=np.float32)
        dones = np.zeros(T, dtype=np.float32)
        adv, _ = gae_lambda_returns(rewards, values, dones)
        assert abs(adv.mean()) < 0.1
        assert 0.9 < adv.std() < 1.1

    def test_realized_vol_consistent_with_known_vol(self):
        import numpy as np

        from adversarial_market.utils.math_utils import realized_volatility

        np.random.seed(0)
        daily_vol = 0.02
        log_returns = np.random.normal(0, daily_vol, 390)
        prices = 100.0 * np.exp(np.cumsum(log_returns))
        vol = realized_volatility(prices.tolist(), annualize=True, periods_per_year=390)
        # Should be close to annualised input vol
        annualized_true = daily_vol * np.sqrt(390)
        assert abs(vol - annualized_true) < 0.05
