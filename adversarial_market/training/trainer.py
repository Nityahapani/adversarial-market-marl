"""
Main MARL training loop — MAPPO with centralized critic and alternating optimization.

Training loop:
  1. Collect rollouts from all agents (decentralized execution)
  2. Compute MINE-based MI estimate for execution agent reward shaping
  3. Update belief transformer from collected flow sequences
  4. Run PPO updates for active agents (alternating schedule)
  5. Log metrics and checkpoint

The MINE estimator is updated concurrently with the execution agent
during Phase A. The belief transformer is updated concurrently with
the market maker during Phase B.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from adversarial_market.environment.lob_env import LOBEnvironment
from adversarial_market.networks.actor_critic import (
    ArbitrageActor,
    ExecutionActor,
    MarketMakerActor,
    SharedCritic,
)
from adversarial_market.networks.belief_transformer import BeliefTransformer
from adversarial_market.networks.mine_estimator import MINEEstimator, PredictabilityPenalty
from adversarial_market.training.alternating_opt import AlternatingOptimizer
from adversarial_market.training.ppo_update import PPOUpdate
from adversarial_market.training.rollout_buffer import RolloutBuffer
from adversarial_market.utils.logger import TrainingLogger


class MARLTrainer:
    """
    End-to-end MAPPO trainer for the adversarial market MARL system.

    Components:
        - LOBEnvironment: single-agent step interface (multi-agent coordinated here)
        - ExecutionActor, MarketMakerActor, ArbitrageActor: agent policies
        - SharedCritic: centralized value function
        - BeliefTransformer: MM's sequential inference model
        - MINEEstimator: MI lower bound for execution agent reward shaping
        - PredictabilityPenalty: flow pattern detection penalty
        - AlternatingOptimizer: phase scheduler
        - RolloutBuffer: GAE-λ rollout storage
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.cfg = config
        self.device = torch.device(config.get("device", "cpu"))
        self.exp_name = config.get("exp_name", "run")

        # Build environment
        self.env = LOBEnvironment(config)
        obs_spaces = self.env.observation_space

        exec_obs_dim = int(obs_spaces["execution"].shape[0])  # type: ignore[index]
        mm_obs_dim = int(obs_spaces["market_maker"].shape[0])  # type: ignore[index]
        arb_obs_dim = int(obs_spaces["arbitrageur"].shape[0])  # type: ignore[index]

        # Global state dimension for critic
        # = exec_obs + mm_obs + arb_obs + market features
        global_state_dim = exec_obs_dim + mm_obs_dim + arb_obs_dim

        net_cfg = config["networks"]
        train_cfg = config["training"]

        # ── Networks ──────────────────────────────────────────────────────
        self.exec_actor = ExecutionActor(
            obs_dim=exec_obs_dim,
            hidden_dims=net_cfg["execution_actor"]["hidden_dims"],
            activation=net_cfg["execution_actor"]["activation"],
        ).to(self.device)

        self.mm_actor = MarketMakerActor(
            obs_dim=mm_obs_dim,
            hidden_dims=net_cfg["mm_actor"]["hidden_dims"],
            activation=net_cfg["mm_actor"]["activation"],
        ).to(self.device)

        self.arb_actor = ArbitrageActor(
            obs_dim=arb_obs_dim,
            hidden_dims=net_cfg["arb_actor"]["hidden_dims"],
            activation=net_cfg["arb_actor"]["activation"],
        ).to(self.device)

        self.critic = SharedCritic(
            global_state_dim=global_state_dim,
            hidden_dims=net_cfg["shared_critic"]["hidden_dims"],
            activation=net_cfg["shared_critic"]["activation"],
            use_layer_norm=net_cfg["shared_critic"]["use_layer_norm"],
        ).to(self.device)

        bt_cfg = net_cfg["belief_transformer"]
        self.belief_transformer = BeliefTransformer(
            input_dim=5,
            d_model=bt_cfg["d_model"],
            n_heads=bt_cfg["n_heads"],
            n_layers=bt_cfg["n_layers"],
            d_ff=bt_cfg["d_ff"],
            dropout=bt_cfg["dropout"],
            max_seq_len=bt_cfg["max_seq_len"],
        ).to(self.device)

        mine_cfg = net_cfg["mine"]
        self.mine = MINEEstimator(
            z_dim=1,  # binary type: informed (1) vs noise (0)
            f_dim=4,  # flow feature: (size_frac, offset, order_type, time_frac)
            hidden_dims=tuple(mine_cfg["hidden_dims"]),
            ema_decay=mine_cfg["ema_decay"],
        ).to(self.device)

        self.pred_penalty = PredictabilityPenalty(flow_dim=4).to(self.device)

        # ── Optimizers ────────────────────────────────────────────────────
        lr_a = train_cfg["lr_actor"]
        lr_c = train_cfg["lr_critic"]
        lr_m = train_cfg["lr_mine"]
        lr_b = train_cfg["lr_belief"]
        wd = train_cfg["weight_decay"]

        self.exec_opt = torch.optim.Adam(self.exec_actor.parameters(), lr=lr_a, weight_decay=wd)
        self.mm_opt = torch.optim.Adam(self.mm_actor.parameters(), lr=lr_a, weight_decay=wd)
        self.arb_opt = torch.optim.Adam(self.arb_actor.parameters(), lr=lr_a, weight_decay=wd)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr_c, weight_decay=wd)
        self.mine_opt = torch.optim.Adam(self.mine.parameters(), lr=lr_m, weight_decay=wd)
        self.belief_opt = torch.optim.Adam(
            list(self.belief_transformer.parameters()) + list(self.pred_penalty.parameters()),
            lr=lr_b,
            weight_decay=wd,
        )

        # ── PPO updaters ──────────────────────────────────────────────────
        _shared_ppo_kwargs = dict(
            critic=self.critic,
            critic_optimizer=self.critic_opt,
            clip_range=train_cfg["clip_range"],
            clip_range_vf=train_cfg.get("clip_range_vf"),
            value_coef=train_cfg["value_coef"],
            entropy_coef=train_cfg["entropy_coef"],
            max_grad_norm=train_cfg["max_grad_norm"],
        )
        self.exec_ppo = PPOUpdate(
            actor=self.exec_actor,
            actor_optimizer=self.exec_opt,
            **_shared_ppo_kwargs,
        )
        self.mm_ppo = PPOUpdate(
            actor=self.mm_actor,
            actor_optimizer=self.mm_opt,
            **_shared_ppo_kwargs,
        )
        self.arb_ppo = PPOUpdate(
            actor=self.arb_actor,
            actor_optimizer=self.arb_opt,
            **_shared_ppo_kwargs,
        )

        # ── Training infrastructure ───────────────────────────────────────
        alt_cfg = train_cfg["alternating"]
        self.alt_opt = AlternatingOptimizer(
            exec_phase_steps=alt_cfg["exec_phase_steps"],
            mm_arb_phase_steps=alt_cfg["mm_arb_phase_steps"],
        )

        self.buffer = RolloutBuffer(
            rollout_length=train_cfg["rollout_length"],
            n_envs=1,
            gamma=train_cfg["gamma"],
            gae_lambda=train_cfg["gae_lambda"],
            device=str(self.device),
        )

        self.logger = TrainingLogger(
            exp_name=self.exp_name,
            log_dir=config.get("log_dir", "runs/"),
            use_tensorboard=config["logging"]["use_tensorboard"],
            use_wandb=config["logging"]["use_wandb"],
        )

        self.total_timesteps = train_cfg["total_timesteps"]
        self.rollout_length = train_cfg["rollout_length"]
        self.n_epochs = train_cfg["n_epochs"]
        self.minibatch_size = train_cfg["minibatch_size"]
        self.lambda_leakage = config["agents"]["execution"]["lambda_leakage"]
        self.mu_predictability = config["agents"]["execution"]["mu_predictability"]

        self._global_step = 0
        self._episode = 0
        self._checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints/")) / self.exp_name
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ── Main training loop ─────────────────────────────────────────────────

    def train(self) -> None:
        """Run the full training loop."""
        obs, _ = self.env.reset()
        episode_rewards = {k: 0.0 for k in ("execution", "market_maker", "arbitrageur")}
        start_time = time.time()

        while self._global_step < self.total_timesteps:
            # ── Collect rollout ───────────────────────────────────────────
            self.buffer.clear()
            for _ in range(self.rollout_length):
                actions, log_probs, values, entropies = self._get_actions(obs)

                _np_actions = {
                    k: (v.numpy() if hasattr(v, "numpy") else v)  # type: ignore[union-attr]
                    for k, v in actions.items()
                }
                next_obs, rewards, terminated, truncated, info = self.env.step(
                    _np_actions  # type: ignore[arg-type]
                )
                done = any(terminated.values())

                # Inject MI penalty into execution reward
                mi_penalty = self._compute_mi_penalty()
                rewards["execution"] -= self.lambda_leakage * mi_penalty

                global_state = self._build_global_state(obs)
                self.buffer.add(
                    obs=obs,
                    global_state=global_state,
                    actions={k: v.cpu().numpy() for k, v in actions.items()},
                    log_probs={k: float(v) for k, v in log_probs.items()},
                    rewards=rewards,
                    values={k: float(v) for k, v in values.items()},
                    dones=terminated,
                    entropies={k: float(v) for k, v in entropies.items()},
                )

                for k in episode_rewards:
                    episode_rewards[k] += rewards[k]

                self._global_step += 1
                self.alt_opt.step(1)

                if done:
                    obs, _ = self.env.reset()
                    self._episode += 1
                    self.logger.log_episode(self._episode, episode_rewards, self._global_step)
                    episode_rewards = {k: 0.0 for k in episode_rewards}
                else:
                    obs = next_obs

            # ── Compute returns and advantages ────────────────────────────
            with torch.no_grad():
                last_global = torch.as_tensor(
                    self._build_global_state(obs), dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                last_values = {
                    "execution": float(self.critic(last_global).item()),
                    "market_maker": float(self.critic(last_global).item()),
                    "arbitrageur": float(self.critic(last_global).item()),
                }
            self.buffer.compute_returns_and_advantages(last_values)

            # ── Update belief transformer ─────────────────────────────────
            belief_metrics = self._update_belief_transformer()

            # ── PPO updates ───────────────────────────────────────────────
            update_metrics = self._run_ppo_updates()

            # ── Advance MINE ──────────────────────────────────────────────
            mine_metrics = self._update_mine()

            # ── Logging ───────────────────────────────────────────────────
            metrics = {**update_metrics, **belief_metrics, **mine_metrics}
            metrics["fps"] = int(self._global_step / (time.time() - start_time + 1e-8))
            metrics["phase"] = int(self.alt_opt.current_phase)
            self.logger.log_train(self._global_step, metrics)

            # ── Checkpoint ────────────────────────────────────────────────
            n_rollouts = self._global_step // self.rollout_length
            if n_rollouts % self.cfg.get("checkpoint_interval", 100) == 0:
                self.save_checkpoint(self._global_step)

        self.logger.close()
        self.save_checkpoint(self._global_step, final=True)

    # ── Action collection ──────────────────────────────────────────────────

    @torch.no_grad()
    def _get_actions(self, obs: Dict[str, np.ndarray]) -> Tuple[
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
    ]:
        exec_obs = torch.as_tensor(
            obs["execution"], dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        mm_obs = torch.as_tensor(
            obs["market_maker"], dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        arb_obs = torch.as_tensor(
            obs["arbitrageur"], dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        global_state = torch.as_tensor(
            self._build_global_state(obs), dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        exec_action, exec_logp, exec_ent = self.exec_actor(exec_obs)
        mm_action, mm_logp, mm_ent = self.mm_actor(mm_obs)
        arb_action, arb_logp, arb_ent = self.arb_actor(arb_obs)
        value = self.critic(global_state)

        return (
            {
                "execution": exec_action.squeeze(0).cpu().numpy(),
                "market_maker": mm_action.squeeze(0).cpu().numpy(),
                "arbitrageur": arb_action.squeeze(0).cpu().numpy(),
            },
            {
                "execution": exec_logp.squeeze(0),
                "market_maker": mm_logp.squeeze(0),
                "arbitrageur": arb_logp.squeeze(0),
            },
            {
                "execution": value.squeeze(0),
                "market_maker": value.squeeze(0),
                "arbitrageur": value.squeeze(0),
            },
            {
                "execution": exec_ent.squeeze(0),
                "market_maker": mm_ent.squeeze(0),
                "arbitrageur": arb_ent.squeeze(0),
            },
        )

    # ── PPO update orchestration ───────────────────────────────────────────

    def _run_ppo_updates(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        active = self.alt_opt.active_agents

        for epoch in range(self.n_epochs):
            for batch in self.buffer.get_minibatches(self.minibatch_size):
                g = batch["global"]["states"]

                if "execution" in active:
                    m = self.exec_ppo.update(
                        obs=batch["execution"]["obs"],
                        global_states=g,
                        actions=batch["execution"]["actions"],
                        log_probs_old=batch["execution"]["log_probs_old"],
                        advantages=batch["execution"]["advantages"],
                        returns=batch["execution"]["returns"],
                        values_old=batch["execution"]["values_old"],
                    )
                    for k, v in m.items():
                        metrics[f"exec/{k}"] = v

                if "market_maker" in active:
                    m = self.mm_ppo.update(
                        obs=batch["market_maker"]["obs"],
                        global_states=g,
                        actions=batch["market_maker"]["actions"],
                        log_probs_old=batch["market_maker"]["log_probs_old"],
                        advantages=batch["market_maker"]["advantages"],
                        returns=batch["market_maker"]["returns"],
                        values_old=batch["market_maker"]["values_old"],
                    )
                    for k, v in m.items():
                        metrics[f"mm/{k}"] = v

                if "arbitrageur" in active:
                    m = self.arb_ppo.update(
                        obs=batch["arbitrageur"]["obs"],
                        global_states=g,
                        actions=batch["arbitrageur"]["actions"],
                        log_probs_old=batch["arbitrageur"]["log_probs_old"],
                        advantages=batch["arbitrageur"]["advantages"],
                        returns=batch["arbitrageur"]["returns"],
                        values_old=batch["arbitrageur"]["values_old"],
                    )
                    for k, v in m.items():
                        metrics[f"arb/{k}"] = v

        return metrics

    def _update_belief_transformer(self) -> Dict[str, float]:
        """Update belief transformer using collected flow sequences."""
        flow_buf = self.env.get_flow_buffer()
        if len(flow_buf) < 10:
            return {}

        flow_tensor = torch.as_tensor(flow_buf, dtype=torch.float32, device=self.device)
        # Treat all flows as from informed trader (label=1) for supervised training
        # In production, mix with pure noise trajectories (label=0)
        labels = torch.ones(1, device=self.device)
        seq = flow_tensor.unsqueeze(0)  # (1, T, 4) — add 5th dim (order_type) if available

        # Pad to 5-dim if needed
        if seq.shape[-1] == 4:
            ot = torch.zeros(*seq.shape[:-1], 1, device=self.device)
            seq = torch.cat([seq, ot], dim=-1)

        loss, belief = self.belief_transformer.compute_belief_loss(
            seq,
            labels,
            entropy_reg_weight=self.cfg["agents"]["market_maker"]["beta_entropy_reg"],
        )

        self.belief_opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.belief_transformer.parameters(), 0.5)
        self.belief_opt.step()

        # Update belief in environment state
        b = float(belief.mean().item())
        if self.env.state:
            self.env.state.mm_belief = b

        return {"belief/loss": float(loss.item()), "belief/b_t": b}

    def _update_mine(self) -> Dict[str, float]:
        """Update MINE estimator from recent execution flow."""
        flow_buf = self.env.get_flow_buffer()
        if len(flow_buf) < 16:
            return {}

        flow_t = torch.as_tensor(flow_buf[-64:], dtype=torch.float32, device=self.device)
        # z = 1 (informed) for all execution agent flows
        z = torch.ones(len(flow_t), 1, device=self.device)

        mi_est, mine_loss = self.mine.compute(z, flow_t)
        self.mine_opt.zero_grad()
        mine_loss.backward()
        self.mine_opt.step()

        return {
            "mine/mi_estimate": float(mi_est.item()),
            "mine/loss": float(mine_loss.item()),
        }

    def _compute_mi_penalty(self) -> float:
        """Return current MI estimate to inject into execution reward."""
        flow_buf = self.env.get_flow_buffer()
        if len(flow_buf) < 4:
            return 0.0
        flow_t = torch.as_tensor(flow_buf[-32:], dtype=torch.float32, device=self.device)
        z = torch.ones(len(flow_t), 1, device=self.device)
        return self.mine.estimate_only(z, flow_t)

    # ── Utility ────────────────────────────────────────────────────────────

    def _build_global_state(self, obs: Dict[str, np.ndarray]) -> np.ndarray:  # type: ignore[return]
        return np.concatenate([obs["execution"], obs["market_maker"], obs["arbitrageur"]])

    # ── Checkpointing ──────────────────────────────────────────────────────

    def save_checkpoint(self, step: int, final: bool = False) -> None:
        tag = "final" if final else f"step_{step}"
        path = self._checkpoint_dir / f"checkpoint_{tag}.pt"
        torch.save(
            {
                "step": step,
                "exec_actor": self.exec_actor.state_dict(),
                "mm_actor": self.mm_actor.state_dict(),
                "arb_actor": self.arb_actor.state_dict(),
                "critic": self.critic.state_dict(),
                "belief_transformer": self.belief_transformer.state_dict(),
                "mine": self.mine.state_dict(),
                "alt_opt": self.alt_opt.state_dict(),
                "exec_opt": self.exec_opt.state_dict(),
                "mm_opt": self.mm_opt.state_dict(),
                "arb_opt": self.arb_opt.state_dict(),
            },
            path,
        )

    def load_checkpoint(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device)
        self.exec_actor.load_state_dict(ckpt["exec_actor"])
        self.mm_actor.load_state_dict(ckpt["mm_actor"])
        self.arb_actor.load_state_dict(ckpt["arb_actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.belief_transformer.load_state_dict(ckpt["belief_transformer"])
        self.mine.load_state_dict(ckpt["mine"])
        self.alt_opt.load_state_dict(ckpt["alt_opt"])
        return int(ckpt["step"])
