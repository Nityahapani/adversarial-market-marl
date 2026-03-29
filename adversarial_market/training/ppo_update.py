"""
PPO policy gradient update for a single agent.

Implements the clipped surrogate objective with value function loss
and entropy bonus. Supports optional value function clipping.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PPOUpdate:
    """
    Single-agent PPO update step.

    Clipped objective:
        L_CLIP = E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)]
        r_t = π_new(a|s) / π_old(a|s)

    Total loss:
        L = -L_CLIP + c_v * L_VF - c_e * H(π)
    """

    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        critic_optimizer: torch.optim.Optimizer,
        clip_range: float = 0.2,
        clip_range_vf: Optional[float] = None,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        target_kl: Optional[float] = 0.015,
    ) -> None:
        self.actor = actor
        self.critic = critic
        self.actor_opt = actor_optimizer
        self.critic_opt = critic_optimizer
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl

    def update(
        self,
        obs: torch.Tensor,
        global_states: torch.Tensor,
        actions: torch.Tensor,
        log_probs_old: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        values_old: torch.Tensor,
        extra_reward: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Perform one PPO minibatch update.

        Args:
            obs:            Agent-specific observations, (B, obs_dim)
            global_states:  Global state for critic, (B, global_dim)
            actions:        Taken actions, (B, action_dim)
            log_probs_old:  Log probs under old policy, (B,)
            advantages:     GAE advantages, (B,)
            returns:        Discounted returns, (B,)
            values_old:     Old value estimates, (B,)
            extra_reward:   Optional additional scalar reward (e.g., -MI penalty)

        Returns:
            Dictionary of scalar metrics for logging.
        """
        if extra_reward is not None:
            advantages = advantages + extra_reward

        # ── Actor update ──────────────────────────────────────────────────
        log_probs_new, entropy = self.actor.evaluate_actions(obs, actions)  # type: ignore[operator]
        ratio = torch.exp(log_probs_new - log_probs_old)

        # Approximate KL for early stopping
        approx_kl = ((ratio - 1) - (log_probs_new - log_probs_old)).mean()

        # Clipped surrogate loss
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        entropy_loss = -entropy.mean()

        actor_loss = policy_loss + self.entropy_coef * entropy_loss

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_opt.step()

        # ── Critic update ─────────────────────────────────────────────────
        values_new = self.critic(global_states)

        if self.clip_range_vf is not None:
            values_clipped = values_old + torch.clamp(
                values_new - values_old, -self.clip_range_vf, self.clip_range_vf
            )
            vf_loss = torch.max(
                F.mse_loss(values_new, returns),
                F.mse_loss(values_clipped, returns),
            )
        else:
            vf_loss = F.mse_loss(values_new, returns)

        critic_loss = self.value_coef * vf_loss

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_opt.step()

        return {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(vf_loss.item()),
            "entropy": float(-entropy_loss.item()),
            "approx_kl": float(approx_kl.item()),
            "ratio_mean": float(ratio.mean().item()),
            "ratio_max": float(ratio.max().item()),
        }

    def should_stop_early(self, approx_kl: float) -> bool:
        if self.target_kl is None:
            return False
        return approx_kl > 1.5 * self.target_kl
