"""
Minimal working example.

Runs one episode of the adversarial LOB environment with randomly
initialised (untrained) policies. Completes in ~30 seconds on CPU.

Demonstrates:
  - Environment reset and step cycle
  - All three agent action spaces
  - Market maker belief b_t updating each step
  - MINE-based leakage estimate from the execution agent's flow buffer

Usage:
    python examples/minimal_example.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# Allow running from repo root without installing
sys.path.insert(0, str(Path(__file__).parent.parent))

from adversarial_market.environment.lob_env import LOBEnvironment  # noqa: E402
from adversarial_market.networks.actor_critic import (  # noqa: E402
    ArbitrageActor,
    ExecutionActor,
    MarketMakerActor,
)
from adversarial_market.networks.mine_estimator import MINEEstimator  # noqa: E402
from adversarial_market.utils.config import load_config  # noqa: E402


def main() -> None:
    # ── Setup ──────────────────────────────────────────────────────────────
    config = load_config("configs/fast_debug.yaml")
    env = LOBEnvironment(config)

    obs_spaces = env.observation_space
    exec_actor = ExecutionActor(
        obs_dim=obs_spaces["execution"].shape[0],  # type: ignore[index]
        hidden_dims=[32],
    )
    mm_actor = MarketMakerActor(
        obs_dim=obs_spaces["market_maker"].shape[0],  # type: ignore[index]
        hidden_dims=[32],
    )
    arb_actor = ArbitrageActor(
        obs_dim=obs_spaces["arbitrageur"].shape[0],  # type: ignore[index]
        hidden_dims=[32],
    )
    mine = MINEEstimator(z_dim=1, f_dim=4, hidden_dims=(32, 32))

    # ── Episode ────────────────────────────────────────────────────────────
    observations, _ = env.reset(seed=42)
    done = False
    step = 0

    print("\nAdversarial LOB — one episode with random policies")
    print(f"{'Step':>4}  {'Mid price':>10}  {'MM belief b_t':>14}  {'Exec remaining':>15}")
    print("─" * 52)

    while not done:
        with torch.no_grad():
            exec_obs = torch.as_tensor(observations["execution"], dtype=torch.float32).unsqueeze(0)
            mm_obs = torch.as_tensor(observations["market_maker"], dtype=torch.float32).unsqueeze(0)
            arb_obs = torch.as_tensor(observations["arbitrageur"], dtype=torch.float32).unsqueeze(0)

            exec_action, _, _ = exec_actor(exec_obs)
            mm_action, _, _ = mm_actor(mm_obs)
            arb_action, _, _ = arb_actor(arb_obs)

        actions = {
            "execution": exec_action.squeeze(0).numpy(),
            "market_maker": mm_action.squeeze(0).numpy(),
            "arbitrageur": arb_action.squeeze(0).numpy(),
        }

        observations, rewards, terminated, _, _ = env.step(actions)
        done = any(terminated.values())
        step += 1

        mid = env.state.mid_price
        belief = env.state.mm_belief
        remaining = env.state.exec_remaining_inventory
        print(f"{step:>4}  {mid:>10.4f}  {belief:>14.4f}  {remaining:>15d}")

    # ── Leakage estimate ───────────────────────────────────────────────────
    print("\nEpisode complete.")
    flow_buf = env.get_flow_buffer()
    if len(flow_buf) > 0:
        z = torch.ones(len(flow_buf), 1)
        ft = torch.as_tensor(flow_buf, dtype=torch.float32)
        mi = mine.estimate_only(z, ft)
        print(f"Flow events recorded : {len(flow_buf)}")
        print(f"MI leakage estimate  : {mi:.4f}  (lower = better camouflage)")
        print("  (untrained agents — MI will decrease as lambda_leakage increases during training)")
    else:
        print("No execution agent flow recorded (inventory was 0 or no fills).")

    print("\nTo run full training:")
    print("  python scripts/train.py --config configs/fast_debug.yaml --exp-name debug")


if __name__ == "__main__":
    main()
