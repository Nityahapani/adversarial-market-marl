# adversarial-market-marl

**A multi-agent reinforcement learning framework for adversarial market microstructure with endogenous information leakage.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?style=flat-square)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green?style=flat-square)](https://github.com/yourusername/adversarial-market-marl/blob/main/LICENSE)
[![Tests](https://img.shields.io/badge/tests-222%20passing-brightgreen?style=flat-square)](https://github.com/yourusername/adversarial-market-marl/actions)

---

## Overview

This framework models financial markets as **adversarial information ecosystems** rather than price-discovery mechanisms. Three co-evolving agents contest informational advantage through a realistic limit order book:

| Agent | Role | Objective |
|-------|------|-----------|
| **Execution agent** | Institutional trader | Minimise implementation shortfall and information leakage |
| **Market maker** | Adaptive liquidity provider | Maximise spread capture; infer informed vs noise flow |
| **Arbitrageur** | Latency exploiter | Profit from belief-update lags |

The central question the framework is designed to answer:

> *To what extent can an informed trader optimally execute large orders while minimising the statistical detectability of their informational advantage?*

---

## Key Features

- **Information leakage as a reward signal** — mutual information $I(z; F_t)$ estimated live by a MINE-f neural network and penalised directly in the execution agent's reward
- **Sequential belief inference** — Pre-LayerNorm Transformer encoder tracks $b_t = P(\text{informed} \mid \mathcal{F}_t)$ in real time
- **Endogenous price formation** — no exogenous price path; all price discovery emerges from agent interaction
- **Detectability phase transition** — infrastructure to locate the critical $\lambda^*$ where informed flow becomes statistically indistinguishable from noise
- **222 tests, full CI** — black, isort, flake8, mypy, pytest on Python 3.10 and 3.11

---

## Quick Install

```bash
git clone https://github.com/Nityahapani/adversarial-market-marl.git
cd adversarial-market-marl
pip install -r requirements.txt
pip install -e .
```

See [Installation](installation.md) for GPU setup and dependency details.

---

## Quick Start

```bash
# Smoke test — runs in ~2 minutes on CPU
python scripts/train.py --config configs/fast_debug.yaml --exp-name debug

# Full training run
python scripts/train.py --config configs/default.yaml --exp-name my_run
```

See [Quick Start](quickstart.md) for Python API usage, evaluation, and the phase transition sweep.
