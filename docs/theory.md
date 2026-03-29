# Theoretical Foundations

## 1. Market Microstructure as an Information Channel

We model the limit order book as a **noisy channel** in the sense of Shannon (1948). The execution agent acts as an encoder, transforming its private information (latent type $z \in \{0,1\}$) into an observable signal (order flow $F_t$). The market maker acts as a decoder, performing sequential inference on $F_t$ to recover $z$.

### 1.1 The Kyle (1985) Baseline

Kyle's continuous auction model establishes that an informed trader optimally spreads their trades across time to minimise price impact. In equilibrium:
- The market maker's pricing rule is linear in order flow
- The informed trader's strategy is linear in their private signal
- Information is incorporated gradually into prices

**Our departure:** Kyle's model is static and assumes closed-form optimal strategies. We replace both with learned policies in a dynamic adversarial game.

---

## 2. Information-Theoretic Objective

### 2.1 Mutual Information Penalty

The execution agent's reward is augmented with an information-theoretic penalty:

$$r_E = -\text{IS}(\pi_E) - \lambda \cdot \hat{I}(z; F_t) - \mu \cdot \text{Pred}(F_t)$$

where:
- $\text{IS}$ = implementation shortfall (execution cost)
- $\hat{I}(z; F_t)$ = estimated mutual information between the agent's type $z$ and its observable order flow $F_t$
- $\text{Pred}(F_t)$ = predictability penalty on flow patterns
- $\lambda, \mu$ = trade-off weights

This creates a direct tension: executing the order (reducing IS) requires predictable, detectable flow; minimising leakage requires camouflaged, noisy flow.

### 2.2 MINE Lower Bound

Direct MI computation is intractable for continuous distributions. We use the **MINE** (Mutual Information Neural Estimation) lower bound:

$$I(X; Y) \geq \mathbb{E}_{p(x,y)}[T_\theta(x,y)] - \log\left(\mathbb{E}_{p(x)p(y)}[e^{T_\theta(x,y)}]\right)$$

where $T_\theta: \mathcal{X} \times \mathcal{Y} \to \mathbb{R}$ is a learned statistics network. The bound is tight as $T_\theta$ approaches the log density ratio.

We use the **MINE-f variant** with an exponential moving average baseline for variance reduction:

$$\hat{I}_\text{MINE-f} = \mathbb{E}_{p(x,y)}[T_\theta] - \frac{\mathbb{E}_{p(x)p(y)}[e^{T_\theta}]}{\bar{e}_t}$$

where $\bar{e}_t = \alpha \bar{e}_{t-1} + (1-\alpha) \mathbb{E}[e^{T_\theta}]$.

---

## 3. Sequential Belief Inference

### 3.1 Market Maker's Problem

The market maker performs Bayesian sequential inference:

$$b_t = P(z = 1 \mid \mathcal{F}_t)$$

where $\mathcal{F}_t = \{(\text{side}_s, p_s, q_s, \tau_s)\}_{s \leq t}$ is the full order flow history.

By Bayes' rule:

$$b_t = \frac{f(F_t \mid z=1) \cdot b_{t-1}}{f(F_t \mid z=1) \cdot b_{t-1} + f(F_t \mid z=0) \cdot (1-b_{t-1})}$$

Rather than assuming parametric forms for $f(\cdot \mid z)$, we learn the full mapping $b_t = g_\phi(\mathcal{F}_t)$ end-to-end using a Transformer.

### 3.2 Belief Loss

The market maker's belief model is trained with a regularised binary cross-entropy:

$$\mathcal{L}_\phi = \text{BCE}(b_t, z) - \beta \cdot H(b_t)$$

The entropy term $H(b_t) = -b_t \log b_t - (1-b_t)\log(1-b_t)$ penalises overconfidence, preventing the market maker from becoming brittle when the execution agent's strategy shifts.

---

## 4. Adversarial Equilibrium

### 4.1 Game Formulation

Define a three-player partially observable stochastic game (POSG):

$$\mathcal{G} = \langle \mathcal{S}, \{\mathcal{O}_i\}, \{\mathcal{A}_i\}, P, \{R_i\}, \gamma, T \rangle$$

where:
- $\mathcal{S}$ = global LOB state
- $\mathcal{O}_i$ = agent $i$'s partial observation
- $\mathcal{A}_i$ = agent $i$'s action space
- $P$ = LOB transition dynamics (price-time priority matching)
- $R_i$ = agent $i$'s reward function
- $T$ = episode horizon

A Nash equilibrium $(\pi_E^*, \pi_M^*, \pi_A^*)$ satisfies:

$$V_i(\pi_i^*, \pi_{-i}^*) \geq V_i(\pi_i, \pi_{-i}^*) \quad \forall \pi_i, \forall i$$

### 4.2 CTDE Framework

We use **Centralized Training with Decentralized Execution (CTDE)**:
- During training: shared critic $V(s, a_1, a_2, a_3)$ sees the full global state
- During execution: each policy $\pi_i(a_i \mid o_i)$ uses only its own observation

This is theoretically grounded in the centralised critic not needing to be deployed, so information sharing during training is permissible.

### 4.3 Alternating Optimisation

Simultaneous gradient updates in adversarial games are known to cause oscillation and divergence (simultaneous descent can cycle around Nash points). We use **alternating optimisation**:

**Phase A** (steps $t \in [kT, kT + T_E)$):
  - Update $\pi_E$ with $\pi_M, \pi_A$ frozen
  - Update $T_\theta$ (MINE estimator)

**Phase B** (steps $t \in [kT + T_E, (k+1)T)$):
  - Update $\pi_M, \pi_A$ with $\pi_E$ frozen
  - Update $g_\phi$ (belief transformer)

This mirrors the training of GANs and has convergence guarantees under mild smoothness conditions.

---

## 5. Detectability Phase Transition

### 5.1 Formal Statement

**Conjecture (Detectability Phase Transition):** There exists a critical $\lambda^* > 0$ such that:

$$\lim_{\lambda \to \lambda^*} D_\text{KL}(P_\text{informed}^{(\lambda)} \| P_\text{noise}) = 0$$

where $P_\text{informed}^{(\lambda)}$ is the order flow distribution induced by the execution agent trained with leakage penalty $\lambda$.

Below $\lambda^*$: informed flow is statistically detectable ($D_\text{KL} > 0$, market maker has accuracy $> 0.5$).
Above $\lambda^*$: informed flow is indistinguishable from noise ($D_\text{KL} \approx 0$, market maker accuracy $\approx 0.5$).

### 5.2 The Cost-Detectability Trade-off

The trade-off is sharp: execution cost (IS) is monotonically increasing in $\lambda$, while detectability is monotonically decreasing. The optimal $\lambda$ depends on the agent's private valuation of information concealment vs execution quality.

$$\lambda^* = \arg\min_\lambda \{ \text{IS}(\lambda) + c \cdot D_\text{KL}(\lambda) \}$$

for some cost weight $c > 0$ that encodes the regulatory or strategic penalty for detection.

---

## 6. The Latency Arbitrageur's Role

The arbitrageur introduces a **second dimension of competition**: time, in addition to information. It exploits the lag between:
1. New information arriving in the order flow
2. The market maker's belief update $b_t$
3. Quote adjustment by the market maker

Without the arbitrageur, the market maker could adopt a static "always wide spread" strategy that is immune to adverse selection. The arbitrageur punishes excessively wide spreads by sniping them, forcing the market maker to solve a joint problem:

$$\min_{\text{spread}} \; \text{AdvSel}(\text{spread}) + \text{Snipe Risk}(\text{spread})$$

which admits an interior optimum requiring genuine inference rather than a corner solution.
