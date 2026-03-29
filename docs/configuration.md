# Configuration

All hyperparameters live in YAML files under `configs/`. Files inherit from
`default.yaml` and override only the keys they change — no duplication.

## Config files

| File | Purpose |
|------|---------|
| `configs/default.yaml` | Full production config with all parameters documented |
| `configs/fast_debug.yaml` | Smoke-test config — tiny networks, 10-step episodes, CPU |
| `configs/ablation_lambda.yaml` | λ sweep config for phase transition analysis |

## Key parameters

```yaml
agents:
  execution:
    lambda_leakage: 0.5      # λ — MI penalty weight (the main experimental variable)
    mu_predictability: 0.1   # μ — flow predictability penalty weight
    inventory_lots: 100      # Total position to execute per episode
    horizon: 390             # Steps per episode

  market_maker:
    beta_entropy_reg: 0.05   # Prevents overconfident beliefs
    alpha_adverse_selection: 1.0
    gamma_belief_accuracy: 0.2

networks:
  belief_transformer:
    d_model: 128
    n_heads: 4
    n_layers: 4

training:
  total_timesteps: 10_000_000
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  alternating:
    exec_phase_steps: 1000
    mm_arb_phase_steps: 1000
```

## CLI overrides

Any dot-path key can be overridden from the command line:

```bash
python scripts/train.py \
    --config configs/default.yaml \
    --override agents.execution.lambda_leakage=1.0 \
    --override training.total_timesteps=5000000 \
    --override device=cpu
```
