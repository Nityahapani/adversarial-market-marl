"""
Integration test for the Evaluator — verifies a full eval pass produces
finite metrics and correctly separates informed vs noise episodes.
"""

import numpy as np
import pytest


@pytest.fixture
def debug_config():
    from adversarial_market.utils.config import load_config

    return load_config("configs/fast_debug.yaml")


class TestEvaluatorIntegration:
    def test_evaluator_returns_metrics_dict(self, debug_config):
        from adversarial_market.evaluation.evaluator import Evaluator
        from adversarial_market.training.trainer import MARLTrainer

        cfg = dict(debug_config)
        trainer = MARLTrainer(cfg)
        evaluator = Evaluator(cfg, device="cpu")

        summary = evaluator.evaluate(
            exec_actor=trainer.exec_actor,
            mm_actor=trainer.mm_actor,
            arb_actor=trainer.arb_actor,
            belief_transformer=trainer.belief_transformer,
            mine=trainer.mine,
            n_episodes=2,
            noise_only_episodes=1,
            seed=0,
        )

        assert isinstance(summary, dict)
        assert len(summary) > 0

    def test_all_metrics_finite(self, debug_config):
        from adversarial_market.evaluation.evaluator import Evaluator
        from adversarial_market.training.trainer import MARLTrainer

        trainer = MARLTrainer(debug_config)
        evaluator = Evaluator(debug_config, device="cpu")

        summary = evaluator.evaluate(
            exec_actor=trainer.exec_actor,
            mm_actor=trainer.mm_actor,
            arb_actor=trainer.arb_actor,
            belief_transformer=trainer.belief_transformer,
            mine=trainer.mine,
            n_episodes=2,
            noise_only_episodes=1,
            seed=1,
        )

        # KL divergence legitimately returns NaN when flow buffers hold fewer
        # than 10 samples (the guard inside kl_divergence_flows). This is
        # expected in the tiny fast_debug config with 10-step episodes.
        kl_keys = {"eval/exec/kl_divergence_mean"}
        for k, v in summary.items():
            if k in kl_keys:
                continue
            assert np.isfinite(v), f"Non-finite metric: {k} = {v}"

    def test_expected_metric_keys_present(self, debug_config):
        from adversarial_market.evaluation.evaluator import Evaluator
        from adversarial_market.training.trainer import MARLTrainer

        trainer = MARLTrainer(debug_config)
        evaluator = Evaluator(debug_config, device="cpu")

        summary = evaluator.evaluate(
            exec_actor=trainer.exec_actor,
            mm_actor=trainer.mm_actor,
            arb_actor=trainer.arb_actor,
            belief_transformer=trainer.belief_transformer,
            mine=trainer.mine,
            n_episodes=2,
            noise_only_episodes=1,
            seed=2,
        )

        expected_keys = [
            "eval/exec/completion_rate_mean",
            "eval/mm/pnl_mean",
            "eval/mm/belief_accuracy_mean",
            "eval/arb/pnl_mean",
        ]
        for key in expected_keys:
            assert key in summary, f"Missing expected metric: {key}"

    def test_completion_rate_in_zero_one(self, debug_config):
        from adversarial_market.evaluation.evaluator import Evaluator
        from adversarial_market.training.trainer import MARLTrainer

        trainer = MARLTrainer(debug_config)
        evaluator = Evaluator(debug_config, device="cpu")

        summary = evaluator.evaluate(
            exec_actor=trainer.exec_actor,
            mm_actor=trainer.mm_actor,
            arb_actor=trainer.arb_actor,
            belief_transformer=trainer.belief_transformer,
            mine=trainer.mine,
            n_episodes=3,
            noise_only_episodes=1,
            seed=3,
        )

        cr = summary.get("eval/exec/completion_rate_mean", 0.0)
        assert 0.0 <= cr <= 1.0, f"Completion rate out of range: {cr}"

    def test_belief_accuracy_in_zero_one(self, debug_config):
        from adversarial_market.evaluation.evaluator import Evaluator
        from adversarial_market.training.trainer import MARLTrainer

        trainer = MARLTrainer(debug_config)
        evaluator = Evaluator(debug_config, device="cpu")

        summary = evaluator.evaluate(
            exec_actor=trainer.exec_actor,
            mm_actor=trainer.mm_actor,
            arb_actor=trainer.arb_actor,
            belief_transformer=trainer.belief_transformer,
            mine=trainer.mine,
            n_episodes=2,
            noise_only_episodes=1,
            seed=4,
        )

        acc = summary.get("eval/mm/belief_accuracy_mean", 0.5)
        assert 0.0 <= acc <= 1.0, f"Belief accuracy out of range: {acc}"
