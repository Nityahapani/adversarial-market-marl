# Expected Results and Emergent Phenomena

## 1. Detectability Phase Transition

The central empirical result. As $\lambda$ (leakage penalty) increases:

| Metric | Low λ (λ < λ*) | High λ (λ > λ*) |
|---|---|---|
| $D_\text{KL}$(informed \|\| noise) | High (> 1.0) | Near zero (< 0.05) |
| MM belief accuracy | > 70% | ≈ 50% (random guess) |
| Implementation shortfall | Low | High (execution cost rises) |
| Flow entropy | Low (predictable) | High (uniform, camouflaged) |
| Average spread | Tight | Wide (MM uncertain) |

**Expected shape:** KL divergence decreases sharply at $\lambda^* \approx 0.5$–$1.0$ (model-dependent), with IS rising smoothly from zero.

---

## 2. Emergent Order Splitting

Without explicitly programming TWAP/VWAP, the execution agent should discover order splitting strategies through the reward signal alone. Evidence:

- Order sizes should decrease with $\lambda$ (smaller orders = less detectable)
- Timing should become more uniform with $\lambda$ (less burst trading)
- Limit order fraction should increase with $\lambda$ (passive orders look like noise)

---

## 3. Spread Widening Under Toxicity

The market maker should learn to widen spreads when $b_t$ is high. Quantified by the **spread-belief correlation**:

$$\rho(\text{spread}_t, b_t) > 0$$

A well-trained market maker should show $\rho > 0.6$ in low-$\lambda$ regimes (where informed flow is detectable) and $\rho \approx 0$ in high-$\lambda$ regimes (where flow is camouflaged and the market maker cannot distinguish).

---

## 4. Latency Arbitrage Cycles

The arbitrageur should exhibit cyclical exploitation patterns:
1. Execution agent submits large order → price movement
2. Market maker belief updates with delay
3. Arbitrageur snipes stale quotes before spread adjustment
4. Market maker tightens, cycle resets

This cycle period should decrease as the market maker's belief transformer improves (faster belief updates = shorter exploitation window).

---

## 5. Breakdown of Detectability

The phase transition should manifest as a **discontinuous drop** in belief accuracy at $\lambda^*$:

```
Belief Accuracy
    1.0 |████████████\
        |             \
    0.7 |              \_____________________
        |
    0.5 |─────────────────────────────────── (random)
        └──────────────────────────────────→ λ
        0    0.5   1.0   1.5   2.0
                    λ*
```

The sharp transition (as opposed to gradual) is the key theoretical prediction: it suggests the existence of a **critical camouflage threshold** analogous to phase transitions in statistical mechanics.

---

## 6. Training Stability Indicators

Healthy training should show:

- **Execution agent entropy:** Initially high (random policy), then decreasing as it learns patterns, then increasing again as it discovers camouflage strategies
- **MM belief accuracy:** Should initially increase (learning to classify), then decrease (execution agent adapts)
- **Alternating phase losses:** Should decrease within each phase, with a small jump at phase switches (non-stationarity)
- **MINE estimate:** Should track actual MI — rising when flow is predictable, falling when camouflage kicks in

---

## 7. Benchmark Comparisons

### vs. TWAP Baseline
A simple TWAP strategy trades fixed equal slices every period. Expected:
- Lower IS than trained execution agent at low $\lambda$
- Higher IS than trained agent at high $\lambda$ (TWAP is trivially detectable)

### vs. Kyle Model Prediction
The Kyle (1985) model predicts a linear information revelation schedule. The trained agent should match Kyle at $\lambda = 0$ (no penalty) and deviate increasingly as $\lambda$ rises.

### vs. No-Adversary Baseline
Training execution agent alone (no MM inference, no arb) should produce lower IS but higher KL — confirming the adversarial pressure is necessary for camouflage emergence.
