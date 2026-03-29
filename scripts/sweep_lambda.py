"""
Lambda sweep script — generates the detectability phase transition curve.

Trains one model per lambda value (or loads existing checkpoints) and
evaluates KL divergence, implementation shortfall, and MM belief accuracy
across the sweep. Produces the central Figure 1 of the paper.

Usage:
    python scripts/sweep_lambda.py --config configs/ablation_lambda.yaml
    python scripts/sweep_lambda.py \\
        --lambda-min 0.0 --lambda-max 2.0 --n-steps 10 \\
        --seeds 3 --output results/phase_transition/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep lambda to characterise the detectability phase transition.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="configs/ablation_lambda.yaml")
    parser.add_argument("--lambda-min", type=float, default=0.0)
    parser.add_argument("--lambda-max", type=float, default=2.0)
    parser.add_argument(
        "--n-steps", type=int, default=10, help="Number of lambda values to evaluate."
    )
    parser.add_argument(
        "--seeds", type=int, default=3, help="Number of random seeds per lambda value."
    )
    parser.add_argument("--output", type=str, default="results/phase_transition/")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n-eval-episodes", type=int, default=10)
    parser.add_argument("--plot", action="store_true", default=True)
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help=("If set, load existing checkpoints from this directory " "instead of retraining."),
    )
    return parser.parse_args()


def run_single_lambda(
    lambda_val: float,
    seed: int,
    config: dict,
    device: str,
    n_eval_episodes: int,
    output_dir: Path,  # noqa: ARG001 — reserved for future per-run output
) -> Dict[str, float]:
    """Train and evaluate a single (lambda, seed) configuration."""
    from adversarial_market.evaluation.evaluator import Evaluator
    from adversarial_market.training.trainer import MARLTrainer

    cfg = {**config}
    cfg["agents"]["execution"]["lambda_leakage"] = lambda_val
    cfg["seed"] = seed
    cfg["device"] = device
    cfg["exp_name"] = f"lambda_{lambda_val:.3f}_seed_{seed}"
    cfg["training"]["total_timesteps"] = min(
        cfg["training"].get("total_timesteps", 3_000_000),
        3_000_000,
    )

    trainer = MARLTrainer(cfg)
    trainer.train()

    evaluator = Evaluator(cfg, device=device)
    summary = evaluator.evaluate(
        exec_actor=trainer.exec_actor,
        mm_actor=trainer.mm_actor,
        arb_actor=trainer.arb_actor,
        belief_transformer=trainer.belief_transformer,
        mine=trainer.mine,
        n_episodes=n_eval_episodes,
        noise_only_episodes=n_eval_episodes // 2,
        seed=seed + 1000,
    )
    return summary


def main() -> None:
    import numpy as np

    args = parse_args()
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from adversarial_market.evaluation.visualizer import plot_phase_transition
    from adversarial_market.utils.config import load_config

    config = load_config(args.config)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    lambda_values = np.linspace(args.lambda_min, args.lambda_max, args.n_steps).tolist()
    seeds = list(range(args.seeds))

    print("\nPhase transition sweep")
    print(f"lambda values: {[f'{lv:.2f}' for lv in lambda_values]}")
    print(f"Seeds per lambda: {args.seeds}")
    print(f"Total runs: {len(lambda_values) * args.seeds}")
    print(f"Output: {output_dir}\n")

    results: Dict[float, List[Dict]] = {lam: [] for lam in lambda_values}

    for lam in lambda_values:
        print(f"\n{'─' * 40}")
        print(f"  lambda = {lam:.3f}")
        for seed in seeds:
            print(f"    seed {seed}...", end=" ", flush=True)
            try:
                metrics = run_single_lambda(
                    lambda_val=lam,
                    seed=seed,
                    config=config,
                    device=args.device,
                    n_eval_episodes=args.n_eval_episodes,
                    output_dir=output_dir,
                )
                results[lam].append(metrics)
                kl = metrics.get("eval/exec/kl_divergence_mean", float("nan"))
                shortfall = metrics.get("eval/exec/implementation_shortfall_mean", float("nan"))
                print(f"KL={kl:.3f}  IS={shortfall:.4f}")
            except Exception as exc:
                print(f"FAILED: {exc}")
                results[lam].append({})

    kl_means: List[float] = []
    kl_stds: List[float] = []
    is_means: List[float] = []
    is_stds: List[float] = []
    acc_means: List[float] = []

    for lam in lambda_values:
        seed_results = [r for r in results[lam] if r]

        def _agg(key: str):
            vals = [r.get(key, float("nan")) for r in seed_results]
            finite = [v for v in vals if not np.isnan(v)]
            mean = float(np.mean(finite)) if finite else float("nan")
            std = float(np.std(finite)) if finite else 0.0
            return mean, std

        km, ks = _agg("eval/exec/kl_divergence_mean")
        im, istd = _agg("eval/exec/implementation_shortfall_mean")
        am, _ = _agg("eval/mm/belief_accuracy_mean")

        kl_means.append(km)
        kl_stds.append(ks)
        is_means.append(im)
        is_stds.append(istd)
        acc_means.append(am)

    agg = {
        "lambda_values": lambda_values,
        "kl_divergence_mean": kl_means,
        "kl_divergence_std": kl_stds,
        "implementation_shortfall_mean": is_means,
        "implementation_shortfall_std": is_stds,
        "belief_accuracy_mean": acc_means,
    }
    results_path = output_dir / "sweep_results.json"
    with open(results_path, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"\nAggregated results saved to {results_path}")

    if args.plot:
        plot_phase_transition(
            lambda_values=lambda_values,
            kl_divergences=kl_means,
            impl_shortfalls=is_means,
            belief_accuracies=acc_means,
            kl_stds=kl_stds,
            is_stds=is_stds,
            save_path=str(output_dir / "phase_transition.pdf"),
        )
        print(f"Phase transition plot saved to {output_dir / 'phase_transition.pdf'}")

    print("\nSweep complete.")


if __name__ == "__main__":
    main()
