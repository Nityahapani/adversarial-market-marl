"""
Alternating optimization scheduler for adversarial MARL training.

Alternating optimization is critical for stability in adversarial settings.
Rather than updating all agents simultaneously (which causes non-stationarity),
we alternate between:
  Phase A: Train execution agent (MM + Arb frozen)
  Phase B: Train market maker + arbitrageur (Exec frozen)

This prevents the gradient signals from collapsing due to rapidly shifting
opponent strategies, and allows each side to converge locally before
the opponent adapts.

Reference:
    Goodfellow et al. (2014). Generative Adversarial Networks.
    (Same alternating logic applied to the MARL setting.)
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any, Dict, List


class Phase(IntEnum):
    EXECUTION = 0  # train execution agent
    MM_ARB = 1  # train market maker + arbitrageur


class AlternatingOptimizer:
    """
    Controls which agents are active (receiving gradient updates) each step.

    Args:
        exec_phase_steps:   Number of environment steps per execution phase.
        mm_arb_phase_steps: Number of environment steps per MM+Arb phase.
        start_phase:        Which phase to start in.
    """

    def __init__(
        self,
        exec_phase_steps: int = 1000,
        mm_arb_phase_steps: int = 1000,
        start_phase: Phase = Phase.EXECUTION,
    ) -> None:
        self.exec_phase_steps = exec_phase_steps
        self.mm_arb_phase_steps = mm_arb_phase_steps
        self.current_phase = start_phase
        self._steps_in_phase = 0
        self._phase_history: List[Phase] = []
        self._total_steps = 0

    def step(self, n_steps: int = 1) -> None:
        """Advance the scheduler by n_steps and switch phase if needed."""
        self._steps_in_phase += n_steps
        self._total_steps += n_steps
        phase_len = (
            self.exec_phase_steps
            if self.current_phase == Phase.EXECUTION
            else self.mm_arb_phase_steps
        )
        if self._steps_in_phase >= phase_len:
            self._phase_history.append(self.current_phase)
            self.current_phase = (
                Phase.MM_ARB if self.current_phase == Phase.EXECUTION else Phase.EXECUTION
            )
            self._steps_in_phase = 0

    @property
    def active_agents(self) -> List[str]:
        """Return the list of agent names that should be updated this phase."""
        if self.current_phase == Phase.EXECUTION:
            return ["execution"]
        else:
            return ["market_maker", "arbitrageur"]

    def is_active(self, agent_name: str) -> bool:
        return agent_name in self.active_agents

    def frozen_agents(self) -> List[str]:
        all_agents = ["execution", "market_maker", "arbitrageur"]
        return [a for a in all_agents if a not in self.active_agents]

    def phase_progress(self) -> float:
        """Fraction of current phase completed."""
        phase_len = (
            self.exec_phase_steps
            if self.current_phase == Phase.EXECUTION
            else self.mm_arb_phase_steps
        )
        return self._steps_in_phase / max(phase_len, 1)

    def n_phase_switches(self) -> int:
        return len(self._phase_history)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "current_phase": int(self.current_phase),
            "steps_in_phase": self._steps_in_phase,
            "total_steps": self._total_steps,
            "phase_history_len": len(self._phase_history),
        }

    def load_state_dict(self, d: Dict[str, Any]) -> None:
        self.current_phase = Phase(d["current_phase"])
        self._steps_in_phase = d["steps_in_phase"]
        self._total_steps = d["total_steps"]

    def __repr__(self) -> str:
        return (
            f"AlternatingOptimizer("
            f"phase={self.current_phase.name}, "
            f"progress={self.phase_progress():.1%}, "
            f"switches={self.n_phase_switches()}, "
            f"active={self.active_agents})"
        )
