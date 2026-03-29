title: 'adversarial-market-marl: A Multi-Agent Reinforcement Learning
  Framework for Adversarial Market Microstructure with Endogenous
  Information Leakage'
tags:
  - Python
  - reinforcement learning
  - market microstructure
  - multi-agent systems
  - information theory
  - limit order book
  - finance
authors:
  - name: Nitya Hapani
    orcid: 0009-0007-8422-7163
    corresponding: true
    affiliation: 1
affiliations:
  - name: The Galaxy School, India
    index: 1
date: 29 March 2026
bibliography: paper.bib
---

# Summary

Financial markets are conventionally modeled as price-discovery mechanisms:
agents interact through a limit order book (LOB) and prices converge toward
an equilibrium reflecting fundamental value. This framing, however, ignores
a strategically important dimension of market interaction — the continuous
contest over *information*. An institutional trader executing a large order
holds private information about its intent; a market maker providing
liquidity must infer, in real time, whether incoming order flow originates
from an informed or uninformed source; and fast-moving arbitrageurs exploit
the lag between information arrival and market maker belief updating.

`adversarial-market-marl` is a Python framework that operationalises this
information-theoretic view of market microstructure. It models the LOB as an
**adversarial information channel** [@shannon1948] and implements three
co-evolving agents: an execution agent that minimises both implementation
shortfall [@almgren2001] and the mutual information between its latent intent
and its observable order flow; a market maker with an embedded Transformer
[@vaswani2017] that performs sequential Bayesian inference on order flow
history; and a latency arbitrageur that exploits delays between information
arrival and quote adjustment. All three agents are trained end-to-end using Multi-Agent Proximal Policy
Optimisation (MAPPO) [@schulman2017; @lowe2017] with a centralised critic and
decentralised execution (CTDE), and mutual information is estimated at runtime
using the MINE-f neural estimator [@belghazi2018].

# Statement of Need

Existing computational tools for LOB simulation and market microstructure
research address price impact modelling and optimal execution in isolation,
or provide simulation environments without the adversarial multi-agent
training infrastructure needed to study information dynamics. The Kyle (1985)
model [@kyle1985] and its extensions provide analytical solutions under
restrictive assumptions — linear strategies, known distributional forms, and
static equilibrium — that preclude the study of learned, adaptive behaviour.
Empirical market microstructure research relies on historical data, which
cannot be used to run controlled experiments on how strategic obfuscation
affects detectability.

`adversarial-market-marl` fills this gap by providing:

1. A realistic price-time priority LOB environment compatible with the
   Gymnasium interface [@towers2023], with endogenous price formation,
   Poisson noise trader arrivals, and an Ornstein-Uhlenbeck fundamental
   value process — no exogenous price path is imposed.

2. A training framework in which the **mutual information between an
   agent's private type and its observable order flow is an explicit
   optimisation variable**, estimated by a concurrently trained neural
   network. This is, to our knowledge, the first LOB simulation framework
   to treat information leakage as a first-class reward signal rather than
   an emergent post-hoc observation.

3. Infrastructure for characterising the **detectability phase transition**:
   a conjectured critical obfuscation threshold $\lambda^*$ beyond which
   informed order flow becomes statistically indistinguishable from noise,
   $D_\mathrm{KL}(P_\mathrm{informed} \| P_\mathrm{noise}) \to 0$. The
   included sweep script and evaluation metrics allow systematic empirical
   investigation of this transition.

The framework is designed for researchers in market microstructure, financial
economics, and multi-agent reinforcement learning who need a principled
computational tool for studying the interplay between strategic execution,
adverse selection, and information revelation.

# State of the Field

Several tools exist for LOB simulation and reinforcement learning in financial
markets. ABIDES [@vyetrenko2020] and its Gymnasium extension ABIDES-Gym
[@amrouni2021] provide high-fidelity, event-driven market simulations suited
to testing execution algorithms against background noise, but do not include
adversarial MARL training infrastructure or information-theoretic objectives.
@spooner2020 and @karpe2020 study reinforcement learning for market making,
demonstrating that learned policies outperform heuristic rules, but neither
treats information leakage as an optimisation variable nor studies the
statistical detectability of informed trading as a research question.

`adversarial-market-marl` differs from these tools in two fundamental ways.
The information-theoretic reward — a live MINE-f estimate of $I(z; F_t)$
updated concurrently with policy training — requires tight coupling between
the environment step loop, the MI estimator, and the PPO gradient
computation that cannot be achieved by wrapping existing simulators. And the
alternating optimisation scheduler for stable adversarial co-evolution is a
training-level concern absent from single-agent or cooperative MARL
environments.

# Software Design

The package is organised into six submodules reflecting the separation of
concerns in the MARL system.

`adversarial_market.environment` implements the LOB matching engine
(`OrderBook`) and the Gymnasium-compatible multi-agent environment
(`LOBEnvironment`). The matching engine enforces price-time priority,
supports partial fills and cancellations, and exposes a normalised LOB
snapshot as a numpy array at each step. The environment manages Poisson noise
trader arrivals, the fundamental value process, and the per-agent observation
and action spaces without imposing an exogenous price path.

`adversarial_market.networks` contains `ExecutionActor` (Beta and Bernoulli
distributions over order size and type), `BeliefTransformer` (Pre-LayerNorm
Transformer encoder [@vaswani2017] outputting belief scalar $b_t \in [0,1]$),
`MINEEstimator` (MINE-f lower bound [@belghazi2018]), and `SharedCritic`
(layer-normalised MLP for centralised training).

`adversarial_market.training` implements the full MAPPO loop. The
`AlternatingOptimizer` schedules Phase A (execution agent trains, MM and Arb
frozen) and Phase B (MM and Arb train, execution agent frozen), which is
necessary for stable adversarial co-evolution. The `RolloutBuffer` implements
Generalised Advantage Estimation [@schulman2016] for all three agents, and
`PPOUpdate` performs the clipped surrogate objective update with value
function loss and policy entropy regularisation.

`adversarial_market.evaluation` provides the `Evaluator` class, which runs
separate informed and noise-only episodes to compute the KL divergence
between flow distributions, implementation shortfall, market maker belief
accuracy, Brier score, and flow entropy.

All hyperparameters are externalised to YAML files with a dot-path command-line
override mechanism designed to support systematic ablation studies and sweeps.

# Mathematics

The execution agent's reward is the Lagrangian relaxation of an
information-constrained execution problem. The unconstrained problem is:

$$
\min_{\pi_E} \; \mathbb{E}\bigl[\mathrm{IS}(\pi_E)\bigr]
\quad \text{subject to} \quad I(z;\, F_t^{\pi_E}) \leq \varepsilon
$$

where $\mathrm{IS}(\pi_E)$ is the implementation shortfall and $\varepsilon$
is a detectability budget. The Lagrangian relaxation yields:

$$
r_E = -\mathrm{IS}(\pi_E) - \lambda \cdot \hat{I}(z;\, F_t) - \mu \cdot \mathrm{Pred}(F_t)
$$

where $\hat{I}$ is the MINE-f lower bound on mutual information, $\lambda$
is the dual variable (the shadow price of information leakage), and
$\mathrm{Pred}(F_t)$ is an auxiliary flow predictability penalty. As $\lambda$
increases, the Pareto frontier between execution cost and detectability is
traced.

The market maker's belief is trained with an entropy-regularised binary
cross-entropy loss:

$$
\mathcal{L}_\phi = \mathrm{BCE}(b_t,\, z) - \beta\, H(b_t)
$$

where $H(b_t) = -b_t \log b_t - (1-b_t)\log(1-b_t)$ is the binary entropy.
The entropy term prevents the belief model from becoming overconfident when
the execution agent's strategy shifts during co-evolution.

The central empirical hypothesis is that there exists a critical $\lambda^*$
such that for $\lambda \geq \lambda^*$:

$$
D_\mathrm{KL}\!\bigl(P_\mathrm{informed}^{(\lambda)} \| P_\mathrm{noise}\bigr) \approx 0
$$

implying that informed order flow has become statistically indistinguishable
from noise trader flow. The included sweep tooling is designed to locate
$\lambda^*$ empirically.

# Acknowledgements

We acknowledge the open-source communities behind PyTorch [@pytorch2019]
and Gymnasium [@towers2023], on which this framework depends.

# References
