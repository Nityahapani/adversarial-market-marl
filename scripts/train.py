"""
Training entry point.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/fast_debug.yaml --exp-name debug_run
    python scripts/train.py --config configs/default.yaml \
        --override agents.execution.lambda_leakage=0.8
    python scripts/train.py --resume checkpoints/my_run/checkpoint_step_100000.pt
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the adversarial market MARL system.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml", help="Path to YAML config file."
    )
    parser.add_argument(
        "--exp-name", type=str, default=None, help="Experiment name (overrides config value)."
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from."
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device override: 'cpu' or 'cuda'."
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed override.")
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Dot-path config overrides, e.g. agents.execution.lambda_leakage=0.5",
    )
    return parser.parse_args()


def parse_overrides(override_list: list[str]) -> dict:
    """Parse list of KEY=VALUE strings into a dict."""
    overrides = {}
    for item in override_list:
        if "=" not in item:
            raise ValueError(f"Invalid override format '{item}'. Expected KEY=VALUE.")
        key, _, raw_val = item.partition("=")
        # Try to parse as int, float, bool, else keep as string
        for typ in (int, float):
            try:
                val = typ(raw_val)
                break
            except ValueError:
                pass
        else:
            if raw_val.lower() == "true":
                val = True
            elif raw_val.lower() == "false":
                val = False
            else:
                val = raw_val
        overrides[key] = val
    return overrides


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()

    # Add project root to path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from adversarial_market.training.trainer import MARLTrainer
    from adversarial_market.utils.config import load_config, validate_config

    overrides = parse_overrides(args.override)
    if args.device:
        overrides["device"] = args.device
    if args.seed is not None:
        overrides["seed"] = args.seed

    config = load_config(args.config, overrides=overrides if overrides else None)

    if args.exp_name:
        config["exp_name"] = args.exp_name

    validate_config(config)
    set_seed(config.get("seed", 42))

    sep = "=" * 60
    print(f"\n{sep}")
    print("  Adversarial Market MARL Training")
    print(f"  Experiment: {config['exp_name']}")
    print(f"  Device: {config['device']}")
    lam = config["agents"]["execution"]["lambda_leakage"]
    print(f"  lambda (leakage penalty): {lam}")
    total_ts = config["training"]["total_timesteps"]
    print(f"  Total timesteps: {total_ts:,}")
    print(f"{sep}\n")

    trainer = MARLTrainer(config)

    if args.resume:
        step = trainer.load_checkpoint(args.resume)
        trainer._global_step = step
        print(f"Resumed from step {step:,}")

    trainer.train()
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
