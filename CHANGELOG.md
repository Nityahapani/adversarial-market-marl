# Changelog

All notable changes to this project will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2025-01-01

Initial release of adversarial-market-marl.

### Added

**Core environment**
- LOBEnvironment: Gymnasium-compatible multi-agent limit order book with
  price-time priority matching, endogenous price formation, Poisson noise
  traders, and an Ornstein-Uhlenbeck fundamental value process
- OrderBook: matching engine supporting limit/market orders, partial fills,
  cancellations, VWAP, and normalised LOB snapshots
- MarketState and AgentState containers tracking inventory, PnL, and fills

**Agents**
- ExecutionAgent: minimises implementation shortfall and MI leakage;
  Beta and Bernoulli action distributions
- MarketMakerAgent: sequential belief inference via embedded Transformer;
  belief-conditioned spread decisions
- ArbitrageAgent: quote staleness detection and snipe profit tracking
- BaseAgent: abstract interface with freeze/unfreeze, param counting, save/load

**Networks**
- BeliefTransformer: Pre-LN Transformer encoder outputting calibrated belief
  b_t in [0,1] with entropy-regularised BCE training
- MINEEstimator: MINE-f lower bound with EMA variance-reduction baseline
- PredictabilityPenalty: GRU-based auxiliary flow predictability estimator
- ExecutionActor, MarketMakerActor, ArbitrageActor, SharedCritic
- Deterministic baselines: TWAPPolicy, VWAPPolicy, AdaptiveCamouflagePolicy

**Training**
- Full MAPPO loop with centralised training and decentralised execution
- AlternatingOptimizer: Phase A/B scheduler for adversarial co-evolution
- RolloutBuffer: GAE-lambda multi-agent buffer with typed fields
- PPOUpdate: clipped surrogate objective with optional VF clipping and
  KL early stopping

**Evaluation**
- Evaluator: informed and noise-only episode runs for KL divergence estimation
- Metrics: IS, KL divergence, belief accuracy, Brier score, flow entropy,
  adverse selection cost, spread-toxicity correlation
- Visualizer: phase transition plot, belief trajectory, training curves,
  flow distribution heatmap

**Tooling**
- YAML config system with deep merge and dot-path CLI overrides
- train.py, evaluate.py, sweep_lambda.py scripts
- TensorBoard and optional W&B logging
- Prioritised replay buffer (SumTree, O(log N))
- 222 tests across 13 files; CI on Python 3.10 and 3.11
- JOSS paper (paper.md, paper.bib)
