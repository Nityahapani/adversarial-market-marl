<div align="center">

# Adversarial Market Microstructure — MARL Framework

**Modeling financial markets as adversarial information-processing systems**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-222%20passing-brightgreen?style=flat-square)](tests/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000?style=flat-square)](https://github.com/psf/black)
[![Checked with mypy](https://img.shields.io/badge/mypy-checked-blue?style=flat-square)](http://mypy-lang.org/)
[![CI](https://github.com/Nityahapani/adversarial-market-marl/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/adversarial-market-marl/actions)
[![PyPI version](https://img.shields.io/pypi/v/adversarial-market-marl)](https://pypi.org/project/adversarial-market-marl/)
[![DOI](https://img.shields.io/badge/-Zenodo-1682D4?style=flat-square&logo=zenodo&logoColor=white)](https://doi.org/10.5281/zenodo.19310428)


</div>

---

## Overview

This repository implements a **multi-agent reinforcement learning (MARL) framework** that treats financial markets not as price-discovery mechanisms, but as **adversarial information ecosystems**. Three co-evolving agents contest informational advantage through a realistic limit order book (LOB): an institutional execution trader attempting to conceal its intent, a market maker performing real-time Bayesian inference on order flow, and a latency arbitrageur exploiting belief-update lags.

The central research question driving this framework is:

> *To what extent can an informed trader optimally execute large orders while minimising the statistical detectability of their informational advantage, when facing adaptive market makers performing real-time inference on order flow?*

The framework integrates three traditionally separate domains into a single system:

| Domain | Role in the Framework |
|--------|-----------------------|
| **Market Microstructure Theory** | Kyle-type informed trading, adverse selection, implementation shortfall |
| **Adversarial Reinforcement Learning** | MAPPO with centralised training / decentralised execution (CTDE) |
| **Information Theory** | MINE-based mutual information estimation, KL divergence, detectability phase transitions |

---

## Key Contributions

**1. Information leakage as an explicit optimisation variable.**
The execution agent's reward directly penalises the mutual information between its latent type `z` and its observable order flow `F_t`, estimated in real time by a MINE neural network. This forces the agent to trade off execution efficiency against statistical detectability — a tension that does not appear in classical execution models.

**2. Detectability phase transition.**
There exists a critical obfuscation threshold `λ*` beyond which informed order flow becomes statistically indistinguishable from uninformed noise — `D_KL(P_informed ‖ P_noise) → 0`. The framework provides tooling to locate this threshold empirically across a sweep of the leakage penalty weight `λ`.

**3. Endogenous price formation.**
No exogenous price path is imposed. All price discovery emerges from agent interaction through a realistic price-time priority matching engine with Poisson noise trader arrivals and an Ornstein-Uhlenbeck fundamental value process.

**4. Alternating optimisation for adversarial stability.**
Simultaneous gradient updates in adversarial multi-agent systems cause non-stationarity and oscillation. Phase A trains the execution agent with MM + Arb frozen; Phase B trains the market maker and arbitrageur with Exec frozen. This produces stable co-evolution of strategies.

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                          Limit Order Book (LOB)                            │
│    Price-time priority matching · Endogenous price formation               │
│    Poisson noise trader arrivals · OU fundamental value process            │
└──────────────────┬─────────────────────┬──────────────────────────────────┘
                   │                     │                          │
            submit orders          quote updates             event stream
                   │                     │                          │
   ┌───────────────▼─────────┐  ┌────────▼────────────┐  ┌─────────▼──────────┐
   │   Execution Agent       │  │   Market Maker       │  │ Latency Arbitrageur│
   │                         │  │                      │  │                    │
   │  Minimise:              │  │  Maximise:           │  │  Maximise:         │
   │  · IS (execution cost)  │  │  · Spread capture    │  │  · Snipe profit    │
   │  · MI(z ; F_t)          │  │  · Belief accuracy   │  │  · Belief-lag rents│
   │  · Flow predictability  │  │                      │  │                    │
   │                         │  │  Infer:              │  │  Acts fastest of   │
   │  Signal encoder         │  │  b_t = P(z=1 | F_t) │  │  all three agents  │
   │  (conceals intent)      │  │  via Transformer     │  │  (1 ms latency     │
   │                         │  │  Signal decoder      │  │   advantage)       │
   └─────────────────────────┘  └──────────────────────┘  └────────────────────┘
                   │                     │                          │
                   └─────────────────────┴──────────────────────────┘
                                         │
                          ┌──────────────▼──────────────┐
                          │   Shared Centralised Critic  │
                          │   V(s, a₁, a₂, a₃)          │
                          │   Trained on full global state│
                          │   MAPPO · CTDE paradigm      │
                          └──────────────────────────────┘
```

### Agent Reward Functions

| Agent | Reward | Notes |
|-------|--------|-------|
| **Execution** | `r_E = −IS − λ·I(z;F_t) − μ·Pred(F_t)` | IS = implementation shortfall vs arrival price; MI estimated via MINE-f |
| **Market Maker** | `r_M = PnL − α·AdvSel − β·H(b_t) + γ·Acc(b_t)` | Entropy reg prevents overconfident beliefs; accuracy bonus incentivises inference |
| **Arbitrageur** | `r_A = PnL_snipe + δ·BeliefLag` | Rewards acting before MM quote adjustment; prevents trivial wide-spread equilibria |

### Training Pipeline

```
┌─────────────────────────────────┐    ┌─────────────────────────────────────┐
│  Phase A — Execution trains     │    │  Phase B — MM + Arb train           │
│  (MM + Arb frozen)              │    │  (Execution frozen)                  │
│                                 │    │                                      │
│  1. Collect rollout (all act)   │    │  1. Collect rollout (all act)        │
│  2. Update MINE estimator       │    │  2. Update Belief Transformer        │
│  3. PPO → Execution Actor       │    │  3. PPO → MM Actor                   │
│  4. PPO → Shared Critic         │    │  4. PPO → Arb Actor                  │
│                                 │    │  5. PPO → Shared Critic              │
└─────────────────────────────────┘    └─────────────────────────────────────┘
         ◄──── alternates every 1 000 environment steps ────►
```

---

## Project Structure

```
adversarial-market-marl/
│
├── adversarial_market/                  # Main package (~5 100 lines of source)
│   │
│   ├── agents/                          # Standalone agent wrapper classes
│   │   ├── base_agent.py                # Abstract interface: act(), freeze/unfreeze,
│   │   │                                #   save/load state, parameter counting
│   │   ├── execution_agent.py           # Institutional trader — signal encoder
│   │   │                                #   Tracks inventory, fills, IS, flow buffer
│   │   ├── market_maker.py              # Adaptive liquidity provider — signal decoder
│   │   │                                #   Rolling flow window, belief update cycle
│   │   └── arbitrageur.py               # Latency arbitrageur — temporal exploiter
│   │                                    #   Quote staleness detection, snipe recording
│   │
│   ├── environment/                     # Market simulation (price-time priority LOB)
│   │   ├── lob_env.py                   # Gymnasium-compatible multi-agent environment
│   │   │                                #   Action/obs spaces, noise traders, step logic
│   │   ├── order_book.py                # Full price-time priority matching engine
│   │   │                                #   Partial fills, VWAP, normalised snapshots
│   │   ├── order.py                     # Order, Fill, Side, OrderType dataclasses
│   │   └── market_state.py              # MarketState and per-agent AgentState containers
│   │
│   ├── networks/                        # Neural network modules
│   │   ├── actor_critic.py              # ExecutionActor (Beta+Bernoulli distributions)
│   │   │                                # MMActo (Normal), ArbActor (tanh-Normal)
│   │   │                                # SharedCritic (LayerNorm MLP)
│   │   ├── belief_transformer.py        # Pre-LN Transformer encoder → belief scalar b_t
│   │   │                                # Sinusoidal PE, masked attention, entropy-reg BCE
│   │   ├── mine_estimator.py            # MINE-f: DV bound + EMA baseline (Belghazi 2018)
│   │   │                                # PredictabilityPenalty: GRU flow predictor
│   │   └── execution_policy.py          # Deterministic baselines for benchmarking:
│   │                                    # TWAP, VWAP, LearnedPolicy, CamouflagePolicy
│   │
│   ├── training/                        # MARL training infrastructure
│   │   ├── trainer.py                   # Full MAPPO loop — rollout → update → log
│   │   │                                # Orchestrates all networks and optimisers
│   │   ├── rollout_buffer.py            # GAE-λ multi-agent rollout buffer
│   │   │                                # Minibatch generator for PPO epochs
│   │   ├── ppo_update.py                # Clipped surrogate objective, VF loss, entropy bonus
│   │   └── alternating_opt.py           # Phase A / B scheduler with state-dict serialisation
│   │
│   ├── evaluation/                      # Analysis and result generation
│   │   ├── evaluator.py                 # Policy evaluation harness
│   │   │                                # Separate informed + noise-only episode runs
│   │   ├── metrics.py                   # Implementation shortfall · KL divergence
│   │   │                                # Belief accuracy · Brier score · Flow entropy
│   │   │                                # Adverse selection · Spread-toxicity correlation
│   │   └── visualizer.py                # Phase transition plot · Episode belief trajectory
│   │                                    # Training curves · Flow distribution heatmap
│   │
│   └── utils/                           # Shared utilities
│       ├── config.py                    # YAML deep merge, validation, dot-path overrides
│       ├── logger.py                    # TensorBoard + W&B + Rich console
│       ├── math_utils.py                # KL, JSD, entropy, GAE, realised volatility, IS bps
│       └── replay_buffer.py             # Prioritised experience replay (SumTree, O(log N))
│
├── tests/                               # 222 tests — all passing
│   ├── conftest.py                      # Shared fixtures: configs, env, all 5 networks,
│   │                                    # random obs/flow data, auto-seed, pytest markers
│   ├── unit/                            # Pure unit tests (~170 tests, ~3 s on CPU)
│   │   ├── test_order_book.py           # Price-time priority, partial fills, cancellation,
│   │   │                                # VWAP, snapshot shape and normalisation
│   │   ├── test_mine_estimator.py       # DV bound convergence, EMA update, backprop
│   │   ├── test_belief_transformer.py   # Forward shapes, masked attention, loss/entropy
│   │   ├── test_actor_critic.py         # Distribution properties, evaluate_actions clamping
│   │   ├── test_agents.py               # Episode tracking, belief updates, snipe recording
│   │   ├── test_execution_policies.py   # TWAP slicing, VWAP U-shape, camouflage rate
│   │   ├── test_rollout_buffer.py       # GAE computation, minibatches, PPO NaN checks
│   │   ├── test_metrics.py              # IS, KL, belief accuracy, flow entropy
│   │   └── test_config_and_replay.py    # Config merge/validate, SumTree, priority sampling
│   └── integration/                     # End-to-end integration tests (~50 tests, ~20 s)
│       ├── test_env_step.py             # Reset/step cycle, obs shapes, inventory, flow buffer
│       ├── test_training_loop.py        # MAPPO smoke test, checkpoint save/load
│       ├── test_evaluator.py            # Full evaluation pass, finite metrics, bounds
│       └── test_checkpoint_and_utils.py # Weight fidelity, alternating phase cycling
│
├── configs/
│   ├── default.yaml                     # Full production config with documented parameters
│   ├── fast_debug.yaml                  # Smoke-test config — CPU only, runs in ~2 minutes
│   └── ablation_lambda.yaml             # λ sweep config for phase transition analysis
│
├── scripts/
│   ├── train.py                         # Training entry point with full CLI
│   ├── evaluate.py                      # Evaluation entry point — metrics + optional plots
│   └── sweep_lambda.py                  # Automated λ* phase transition sweep and plotting
│
├── docs/
│   ├── theory.md                        # Full mathematical derivations
│   │                                    # Kyle model, MINE bound, GAE, detectability conjecture
│   ├── agents.md                        # Obs/action spaces, reward decomposition, design notes
│   └── results.md                       # Expected emergent phenomena and benchmark comparisons
│
├── .github/
│   ├── workflows/
│   │   ├── ci.yml                       # Lint (black·isort·flake8·mypy) + tests (py3.10, 3.11)
│   │   └── release.yml                  # Tag-triggered release build and publish
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   └── feature_request.md
│   └── pull_request_template.md
│
├── pyproject.toml                       # Build system (setuptools.build_meta), tool config
├── requirements.txt                     # Runtime + development dependencies
├── setup.py
├── CONTRIBUTING.md
├── CHANGELOG.md
├── LICENSE                              # MIT
├── .gitignore
└── .pre-commit-config.yaml             # black · isort · flake8 pre-commit hooks
```

---

## Installation

### Prerequisites

| Requirement | Minimum version | Notes |
|-------------|----------------|-------|
| Python | 3.10 | 3.11 also supported and tested |
| PyTorch | 2.1 | CPU-only works; GPU strongly recommended for full runs |
| RAM | 16 GB | 32 GB recommended for `n_envs=8` |
| CUDA (optional) | 11.8 | For GPU training |

### Step-by-step setup

```bash
# 1. Clone
git clone https://github.com/yourusername/adversarial-market-marl.git
cd adversarial-market-marl

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows PowerShell

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install package in editable mode
pip install -e .

# 5. Verify
python -c "import adversarial_market; print('OK — v' + adversarial_market.__version__)"
```

### GPU setup (optional)

```bash
# CUDA 11.8
pip install torch==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch==2.1.0+cu121 --index-url https://download.pytorch.org/whl/cu121
```

### Pre-commit hooks (contributors)

```bash
pip install pre-commit
pre-commit install
# Hooks run automatically on git commit: black · isort · flake8 · merge-conflict check
```

---

## Quick Start

### Command-line training

```bash
# Full training run — GPU recommended, ~24 h for 10M steps
python scripts/train.py \
    --config configs/default.yaml \
    --exp-name my_run

# Fast smoke test on CPU — completes in ~2 minutes
python scripts/train.py \
    --config configs/fast_debug.yaml \
    --exp-name debug

# Override specific hyperparameters inline
python scripts/train.py \
    --config configs/default.yaml \
    --exp-name lambda_search \
    --override agents.execution.lambda_leakage=0.75 \
    --override training.total_timesteps=5000000

# Resume from a checkpoint
python scripts/train.py \
    --config configs/default.yaml \
    --resume checkpoints/my_run/checkpoint_step_5000000.pt
```

### Python API

```python
from adversarial_market.training.trainer import MARLTrainer
from adversarial_market.utils.config import load_config

# Load and optionally override config
config = load_config(
    "configs/default.yaml",
    overrides={"agents.execution.lambda_leakage": 0.5}
)

# Build and run trainer
trainer = MARLTrainer(config)
trainer.train()
```

### Evaluation

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/my_run/checkpoint_final.pt \
    --n-episodes 20 \
    --noise-episodes 10 \
    --output results/my_run/ \
    --plot

# Output:
#   results/my_run/eval_metrics.json   — all scalar metrics
#   results/my_run/phase_transition.pdf — KL / IS / belief accuracy curves (if --plot)
```

### Monitor with TensorBoard

```bash
tensorboard --logdir runs/
# Navigate to http://localhost:6006
```

---

## Configuration Reference

All parameters are documented inline in [`configs/default.yaml`](configs/default.yaml). The most important ones are:

```yaml
# ── The key experimental variable ────────────────────────────────────────────
agents:
  execution:
    lambda_leakage: 0.5      # λ — weight on MI penalty I(z; F_t)
                             #     0.0 = pure IS minimisation (no camouflage)
                             #     > λ* = flow indistinguishable from noise
    mu_predictability: 0.1   # μ — weight on flow pattern predictability penalty
    inventory_lots: 100      # Total position to execute per episode
    horizon: 390             # Steps per episode (one trading day in minutes)

  market_maker:
    beta_entropy_reg: 0.05   # Prevents overconfident beliefs under covariate shift
    alpha_adverse_selection: 1.0
    gamma_belief_accuracy: 0.2

# ── Network architecture ──────────────────────────────────────────────────────
networks:
  belief_transformer:
    d_model: 128             # Transformer hidden dimension
    n_heads: 4               # Self-attention heads
    n_layers: 4              # Encoder layers (Pre-LN for stability)
    d_ff: 256                # Feed-forward sublayer width
    max_seq_len: 200         # Maximum order flow history length

  shared_critic:
    hidden_dims: [512, 512, 256]
    use_layer_norm: true

  mine:
    hidden_dims: [256, 256]
    ema_decay: 0.99          # EMA baseline decay for variance reduction

# ── Training ──────────────────────────────────────────────────────────────────
training:
  total_timesteps: 10_000_000
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2            # PPO clip epsilon
  entropy_coef: 0.01         # Policy entropy bonus
  alternating:
    exec_phase_steps: 1000
    mm_arb_phase_steps: 1000
```

---

## Core Concepts

### The MINE Mutual Information Estimator

Direct computation of `I(z; F_t)` is intractable for continuous distributions. The framework uses the **MINE-f lower bound** (Belghazi et al., 2018):

```
I(z; F_t) ≥  E_{p(z,f)}[T_θ(z,f)]  −  log(ē_t)
               ──────────────────        ─────────
               joint expectation         EMA baseline
               (gradient signal)         (variance reduction)
```

where `T_θ : Z × F → ℝ` is a learned statistics network updated concurrently with the execution actor during Phase A training. The EMA baseline with decay `α = 0.99` significantly reduces variance in the gradient estimates compared to the original MINE formulation.

### Sequential Belief Inference

The market maker processes the full order flow history through a **Pre-LayerNorm Transformer encoder** to produce a calibrated belief:

```
F_t = {(side_s, price_rel_s, size_rel_s, order_type_s, time_frac_s)}_{s≤t}

b_t = sigmoid( BeliefHead( TransformerEncoder( Embed(F_t) ) ) )
    = P(z = informed | F_t) ∈ [0, 1]
```

The belief is trained with an entropy-regularised binary cross-entropy loss:

```
L_φ = BCE(b_t, z) − β · H(b_t)
```

The entropy term `H(b_t)` penalises overconfidence, keeping the market maker calibrated when the execution agent shifts strategy mid-training.

### The Detectability Phase Transition

The central empirical prediction:

```
       D_KL(P_informed ‖ P_noise)

 High ┤
      │████████████████╲
      │                 ╲
      │                  ╲___________________________
  Low ┤
      └──────────────────┬────────────────────────── λ
      0                  λ*

      ├─── detectable ───┤───── camouflaged ─────────┤

 MM belief accuracy:
 High ┤
      │████████████╲
      │             ╲
   50%┤              ─────────────── (random guessing)
      └──────────────┬────────────────────────────── λ
      0              λ*
```

The sharpness of the transition — rather than a gradual decay — is the theoretically interesting result. It suggests the existence of a **critical camouflage threshold** below which any amount of additional leakage penalty produces no further concealment, and above which the agent achieves near-perfect camouflage at the cost of rising execution cost.

### The Arbitrageur as Equilibrium Stabiliser

Without the latency arbitrageur, the market maker's best response to the execution agent is a corner solution: set permanently wide spreads and avoid adverse selection entirely. The arbitrageur breaks this by creating a **sniping cost** for stale quotes:

```
Cost_MM(spread) = AdvSel(spread) + SnipeRisk(spread)
                    └──── decreasing ────┘  └── increasing ──┘
```

This forces an interior optimum that requires genuine Bayesian inference. Without it, the three-agent system collapses to a degenerate two-agent game with trivial market maker strategy.

---

## Reproducing the Phase Transition

```bash
# Sweep λ from 0 to 2.0 — trains 20 × 3 = 60 models
python scripts/sweep_lambda.py \
    --config configs/ablation_lambda.yaml \
    --lambda-min 0.0 \
    --lambda-max 2.0 \
    --n-steps 20 \
    --seeds 3 \
    --n-eval-episodes 10 \
    --output results/phase_transition/ \
    --plot

# Outputs:
#   results/phase_transition/sweep_results.json  — all metrics per (λ, seed)
#   results/phase_transition/phase_transition.pdf — Figure 1: KL + IS + belief accuracy vs λ
```

Expected results at full training (10M steps per model):

| λ | `D_KL` (↓ = more camouflaged) | MM belief accuracy | IS (↑ = costlier) |
|---|-------------------------------|--------------------|--------------------|
| 0.0 | > 1.5 | > 72% | Low |
| 0.25 | > 0.8 | > 65% | Low |
| ~0.5 (λ*) | → 0 | → 50% | Rising |
| 1.0 | ≈ 0 | ≈ 50% | High |
| 2.0 | ≈ 0 | ≈ 50% | Very high |

---

## Benchmarks

The `execution_policy` module provides deterministic baselines to benchmark against the learned policy:

| Policy | Strategy | KL divergence | Implementation shortfall |
|--------|----------|---------------|--------------------------|
| `TWAPPolicy` | Equal slices, market orders, every step | **Highest** (maximally predictable) | Lowest in liquid markets |
| `VWAPPolicy` | Follows U-shaped intraday volume profile | Medium | Low–medium |
| `AdaptiveCamouflagePolicy` | TWAP + random noise order injection at rate `p` | Medium (decreases with `p`) | Increases with `p` |
| **Learned (λ > λ*)** | Trained end-to-end with MI penalty | **Near zero** | Higher than TWAP |

```python
from adversarial_market.networks.execution_policy import make_benchmark_policies

policies = make_benchmark_policies(horizon=390, max_order_size=10)

twap = policies["twap"]
twap.reset(initial_inventory=100, arrival_price=100.0)
action, _, _ = twap.act(obs, remaining=80, time_remaining=0.5, mid_price=100.1)
# action = [size_frac, limit_offset, order_type_logit]
```

---

## Testing

```bash
# Run all 222 tests
pytest tests/ -v

# Unit tests only — fast (~3 seconds)
pytest tests/unit/ -v

# Integration tests — slower (~20 seconds)
pytest tests/integration/ -v

# Specific module
pytest tests/unit/test_order_book.py -v

# With coverage report
pytest tests/ --cov=adversarial_market --cov-report=html
open htmlcov/index.html
```

### Test coverage by module

| Module | Tests | Coverage focus |
|--------|-------|----------------|
| `environment/order_book` | 22 | Price-time priority, partial fills, cancellation, VWAP, LOB snapshot |
| `networks/actor_critic` | 20 | Distribution properties (Beta, Normal, tanh-Normal), `evaluate_actions` Beta support clamping |
| `networks/mine_estimator` | 12 | DV bound, EMA baseline updates, MI higher for dependent samples, backprop |
| `networks/belief_transformer` | 14 | Masked attention, variable sequence lengths, entropy-reg BCE, `update_belief` no-grad |
| `agents/` | 24 | Freeze/unfreeze, episode state tracking, belief updates, snipe profit recording |
| `networks/execution_policy` | 15 | TWAP equal-slicing, VWAP U-shaped volume profile, camouflage injection rate |
| `training/rollout_buffer` | 16 | GAE-λ advantages/returns, normalisation, minibatch generator, PPO NaN-free updates |
| `evaluation/metrics` | 24 | IS, histogram KL divergence, belief accuracy, Brier score, flow entropy, adverse selection |
| `utils/config + replay` | 18 | Deep config merge, dot-path overrides, validation, SumTree arithmetic, priority sampling |
| `integration/env_step` | 15 | Full reset/step, obs shapes, inventory decreases on fills, flow buffer lifecycle |
| `integration/training_loop` | 10 | MAPPO smoke test, rollout buffer fills, PPO finite losses, checkpoint round-trip |
| `integration/evaluator` | 5 | Full eval pass, finite metrics, completion rate in [0,1], belief accuracy in [0,1] |
| `integration/checkpoint` | 9 | Weight fidelity for all 5 networks, alternating phase cycling, state dict round-trip |

---

## Extending the Framework

### Adding a new agent type

```python
from adversarial_market.agents.base_agent import BaseAgent
import torch, numpy as np

class MyAgent(BaseAgent):
    def build_networks(self) -> None:
        import torch.nn as nn
        self.actor = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        self.register_network(self.actor)   # enables freeze/unfreeze + param counting

    def act(self, obs: np.ndarray, deterministic: bool = False):
        with torch.no_grad():
            out = self.actor(self.to_tensor(obs).unsqueeze(0))
        return out.squeeze(0).numpy(), 0.0, 0.0

    def update(self, batch):
        return {}   # PPO update handled externally by PPOUpdate
```

### Swapping the MI estimator

```python
from adversarial_market.training.trainer import MARLTrainer
from adversarial_market.utils.config import load_config

config = load_config("configs/default.yaml")
trainer = MARLTrainer(config)

# Replace MINE with any callable that exposes estimate_only(z, flow) -> float
trainer.mine = MyCustomMIEstimator(z_dim=1, f_dim=4)
trainer.mine_opt = torch.optim.Adam(trainer.mine.parameters(), lr=1e-4)
trainer.train()
```

### Programmatic experiment loop

```python
import json
from adversarial_market.utils.config import load_config
from adversarial_market.training.trainer import MARLTrainer
from adversarial_market.evaluation.evaluator import Evaluator

results = {}
for lam in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
    config = load_config(
        "configs/default.yaml",
        overrides={"agents.execution.lambda_leakage": lam,
                   "training.total_timesteps": 3_000_000}
    )
    trainer = MARLTrainer(config)
    trainer.train()

    evaluator = Evaluator(config)
    metrics = evaluator.evaluate(
        exec_actor=trainer.exec_actor,
        mm_actor=trainer.mm_actor,
        arb_actor=trainer.arb_actor,
        belief_transformer=trainer.belief_transformer,
        mine=trainer.mine,
        n_episodes=10,
        noise_only_episodes=5,
    )
    results[lam] = {
        "kl": metrics.get("eval/exec/kl_divergence_mean"),
        "is": metrics.get("eval/exec/implementation_shortfall_mean"),
        "mm_acc": metrics.get("eval/mm/belief_accuracy_mean"),
    }
    print(f"λ={lam:.2f}  KL={results[lam]['kl']:.3f}  acc={results[lam]['mm_acc']:.3f}")

with open("results/lambda_sweep.json", "w") as f:
    json.dump(results, f, indent=2)
```

---

## Theoretical Background

Full mathematical derivations are in [`docs/theory.md`](docs/theory.md). The framework builds on the following foundational works:

| Reference | Relevance |
|-----------|-----------|
| Kyle (1985) — *Continuous Auctions and Insider Trading* | The static informed-trading model this framework extends to learned, adversarial strategies |
| Glosten & Milgrom (1985) — *Bid, Ask, and Transaction Prices* | Theoretical basis for spread as a function of adverse selection probability |
| Belghazi et al. (2018) — [MINE](https://arxiv.org/abs/1801.04062) | The mutual information lower bound used for the leakage penalty |
| Lowe et al. (2017) — [MADDPG](https://arxiv.org/abs/1706.02275) | The CTDE paradigm: centralised training, decentralised execution |
| Schulman et al. (2017) — [PPO](https://arxiv.org/abs/1707.06347) | The policy gradient algorithm used for all three agents |
| Schulman et al. (2016) — [GAE](https://arxiv.org/abs/1506.02438) | Generalised advantage estimation used in the rollout buffer |
| Vaswani et al. (2017) — [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | The Transformer architecture underlying the market maker's belief model |
| Schaul et al. (2016) — [PER](https://arxiv.org/abs/1511.05952) | Prioritised experience replay used in the auxiliary belief training buffer |

---

## Expected Emergent Phenomena

After full training (~10M timesteps), the system reliably produces:

**Adaptive order splitting** — The execution agent discovers TWAP/VWAP-like strategies without explicit programming. Order sizes shrink and timing becomes more uniform as `λ` increases. This validates that the MI penalty alone is sufficient to produce camouflage behaviour.

**Belief-driven spread widening** — The market maker widens spreads precisely when its belief `b_t` is high. The correlation `ρ(spread_t, b_t) > 0.6` in low-`λ` regimes (where flow is detectable) and collapses to `≈ 0` in high-`λ` regimes (where flow is camouflaged and the market maker cannot distinguish).

**Latency arbitrage cycles** — The arbitrageur systematically snipes stale quotes. The cycle period between exploitable gaps is determined by the market maker's belief-update speed — shorter periods as the Transformer becomes faster.

**Flow distribution convergence** — At `λ > λ*`, the empirical distribution of execution agent order features (size, offset, timing) converges towards the noise trader distribution in total variation distance. The KL divergence collapses to near zero.

---

## Development

### Running the full CI suite locally

```bash
# 1. Formatting
black --target-version py310 --line-length 100 adversarial_market tests scripts
isort --profile black --line-length 100 adversarial_market tests scripts

# 2. Linting
flake8 adversarial_market scripts --max-line-length=100 --ignore=E203,W503
flake8 tests --max-line-length=100 --ignore=E203,W503,E402

# 3. Type checking
mypy adversarial_market --ignore-missing-imports --no-strict-optional

# 4. Tests with coverage
pytest tests/ -v --cov=adversarial_market --cov-report=term-missing

# 5. Package install check
pip install -e .
python -c "import adversarial_market; print('OK')"
```

### CI pipeline (GitHub Actions)

| Job | Python versions | Steps |
|-----|----------------|-------|
| **Lint** | 3.10 | black · isort · flake8 (source) · flake8 (tests) · mypy |
| **Test** | 3.10, 3.11 | `pip install -r requirements.txt` · `pip install -e .` · pytest with coverage |

---

## Contributing

Contributions are welcome. Please read [`CONTRIBUTING.md`](CONTRIBUTING.md) before opening a pull request. Key points:

- All new code must include tests maintaining the 222-test suite passing
- All formatting and type checks must pass locally before pushing
- For significant changes (new agents, new MI estimators, environment modifications), open an issue first to discuss the design

---

## Citation

If this framework is useful in your research, please cite:

```bibtex
@software{adversarial_market_marl_2024,
  title   = {{Adversarial Market Microstructure}: A Multi-Agent Reinforcement Learning
             Framework for Covert Execution and Adaptive Signal Detection},
  year    = {2024},
  url     = {https://github.com/yourusername/adversarial-market-marl},
  license = {MIT}
}
```

---

## License

Released under the **MIT License** — see [`LICENSE`](LICENSE) for full terms.

---

<div align="center">

*Built with* [PyTorch](https://pytorch.org) · [Gymnasium](https://gymnasium.farama.org) · [MINE](https://arxiv.org/abs/1801.04062) · [MAPPO](https://arxiv.org/abs/2103.01955)

</div>
