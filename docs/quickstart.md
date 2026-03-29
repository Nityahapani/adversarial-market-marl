# Quick Start

## Command-line training

```bash
# Fast smoke test — CPU, ~2 minutes
python scripts/train.py --config configs/fast_debug.yaml --exp-name debug

# Full run — GPU recommended
python scripts/train.py --config configs/default.yaml --exp-name my_run

# Override hyperparameters inline
python scripts/train.py \
    --config configs/default.yaml \
    --exp-name lambda_search \
    --override agents.execution.lambda_leakage=0.75

# Resume from checkpoint
python scripts/train.py \
    --config configs/default.yaml \
    --resume checkpoints/my_run/checkpoint_step_5000000.pt
```

## Python API

```python
from adversarial_market.training.trainer import MARLTrainer
from adversarial_market.utils.config import load_config

config = load_config(
    "configs/default.yaml",
    overrides={"agents.execution.lambda_leakage": 0.5}
)
trainer = MARLTrainer(config)
trainer.train()
```

## Evaluation

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/my_run/checkpoint_final.pt \
    --n-episodes 20 \
    --output results/my_run/ \
    --plot
```

## Phase transition sweep

```bash
python scripts/sweep_lambda.py \
    --lambda-min 0.0 \
    --lambda-max 2.0 \
    --n-steps 20 \
    --seeds 3 \
    --output results/phase_transition/ \
    --plot
```

## Monitor with TensorBoard

```bash
tensorboard --logdir runs/
```
