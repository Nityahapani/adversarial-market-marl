"""
Structured training logger supporting TensorBoard and optional Weights & Biases.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from rich.console import Console
from rich.table import Table


class TrainingLogger:
    """
    Unified logger for training metrics.

    Writes to:
      - TensorBoard SummaryWriter (if enabled)
      - Weights & Biases run (if enabled)
      - Rich console (always)
    """

    def __init__(
        self,
        exp_name: str,
        log_dir: str = "runs/",
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.exp_name = exp_name
        self.console = Console()
        self._tb_writer = None
        self._wandb_run = None

        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                tb_path = Path(log_dir) / exp_name
                self._tb_writer = SummaryWriter(str(tb_path))
            except ImportError:
                self.console.print("[yellow]TensorBoard not available — skipping.[/yellow]")

        if use_wandb:
            try:
                import wandb  # noqa: F401

                self._wandb_run = wandb.init(
                    project=wandb_project or "adversarial-market-marl",
                    name=exp_name,
                    config=config,
                    reinit=True,
                )
            except ImportError:
                self.console.print("[yellow]wandb not available — skipping.[/yellow]")

    def log_train(self, step: int, metrics: Dict[str, Any]) -> None:
        """Log training step metrics."""
        if self._tb_writer:
            for key, val in metrics.items():
                if isinstance(val, (int, float)):
                    self._tb_writer.add_scalar(key, val, step)

        if self._wandb_run:
            import wandb  # noqa: F401

            self._wandb_run.log({**metrics, "step": step})

        # Console: print a subset of key metrics
        if step % 5000 == 0:
            self._print_metrics(step, metrics)

    def log_episode(self, episode: int, rewards: Dict[str, float], step: int) -> None:
        """Log per-episode aggregate rewards."""
        flat = {f"episode/{k}_reward": v for k, v in rewards.items()}
        flat["episode/episode"] = episode

        if self._tb_writer:
            for key, val in flat.items():
                if isinstance(val, (int, float)):
                    self._tb_writer.add_scalar(key, val, step)

        if self._wandb_run:
            import wandb  # noqa: F401

            self._wandb_run.log({**flat, "step": step})

    def log_eval(self, step: int, metrics: Dict[str, float]) -> None:
        """Log evaluation metrics."""
        self.log_train(step, metrics)
        self._print_eval(step, metrics)

    def _print_metrics(self, step: int, metrics: Dict[str, Any]) -> None:
        table = Table(title=f"Step {step:,}", show_header=True, header_style="bold")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="white")
        for k, v in sorted(metrics.items()):
            if isinstance(v, float):
                table.add_row(k, f"{v:.4f}")
            elif isinstance(v, int):
                table.add_row(k, str(v))
        self.console.print(table)

    def _print_eval(self, step: int, metrics: Dict[str, float]) -> None:
        self.console.rule(f"[bold]Evaluation @ step {step:,}[/bold]")
        for k, v in sorted(metrics.items()):
            self.console.print(f"  {k}: {v:.4f}")

    def close(self) -> None:
        if self._tb_writer:
            self._tb_writer.close()
        if self._wandb_run:
            self._wandb_run.finish()
