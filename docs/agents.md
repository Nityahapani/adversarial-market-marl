# Agent Design

## Execution Agent

The execution agent represents an institutional trader with a large position to execute over a finite horizon. It is the **signal encoder** — it transforms its private type into observable order flow while trying to minimise the information content of that flow.

### Observation Space

| Feature | Description | Normalisation |
|---|---|---|
| LOB snapshot (80-dim) | Bid/ask prices and sizes at 20 depth levels | Prices: rel. to mid; sizes: max-norm |
| `exec_remaining` | Remaining inventory fraction | `/ initial_inventory` |
| `time_remaining` | Fraction of horizon left | `step / max_steps` |
| `completion_frac` | Fraction of order executed | `1 - remaining / initial` |
| `mid_price` | Current mid price | `/ 100.0` |
| `spread` | Current bid-ask spread | `/ mid_price` |
| `realized_vol` | Recent price volatility | Annualised |

### Action Space

| Dimension | Range | Meaning |
|---|---|---|
| `size_frac` | [0, 1] | Fraction of `max_order_size` to trade this step |
| `limit_offset` | [-5, 5] | Price offset from mid in ticks (negative = passive) |
| `order_type_logit` | [-1, 1] | > 0 → limit; ≤ 0 → market order |

### Reward

$$r_E = -\text{IS} - \lambda \cdot \hat{I}(z; F_t) - \mu \cdot \text{Predictability}(F_t) + r_\text{terminal}$$

**Implementation Shortfall (IS):** Arrival price × total quantity − Σ(fill_price × fill_qty). Measures how much worse than arrival price the execution achieved.

**MI Penalty:** MINE lower bound on I(z; F_t), estimated from the execution agent's flow feature buffer. Injected into the reward during training updates.

**Terminal Penalty:** Large penalty proportional to unexecuted inventory at horizon end, ensuring the agent does not trivially minimise MI by never trading.

---

## Market Maker Agent

The market maker is the **signal decoder** — it provides liquidity and simultaneously performs real-time inference on the informational content of incoming order flow.

### Belief Model

The core innovation is an embedded **Transformer encoder** that processes the order flow sequence and outputs a calibrated belief:

$$b_t = P(z = \text{informed} \mid \mathcal{F}_t) \in [0, 1]$$

This belief is:
1. Directly appended to the market maker's observation before the policy network
2. Used to adjust spread and depth (wider spread when $b_t$ is high)
3. Tracked as the primary detectability metric

### Observation Space

| Feature | Description |
|---|---|
| LOB snapshot (80-dim) | Same as execution agent |
| Flow history (60-dim) | Last 20 fills encoded as (side, price_rel, size_rel) |
| `mm_inventory` | Current inventory / max inventory |
| `mid_price` | Current mid / 100 |
| `spread` | Current spread / mid |
| `belief` | Current b_t from transformer |
| `time_remaining` | Step fraction |

### Action Space

| Dimension | Range | Meaning |
|---|---|---|
| `bid_offset` | [-spread_max, 0] | Bid price below mid (in ticks) |
| `ask_offset` | [0, spread_max] | Ask price above mid (in ticks) |
| `bid_size_frac` | [0, 1] | Quote size on bid side |
| `ask_size_frac` | [0, 1] | Quote size on ask side |

### Reward

$$r_M = \text{PnL} - \alpha \cdot \text{AdvSel} - \beta \cdot H(b_t) + \gamma \cdot \text{Accuracy}(b_t)$$

The entropy regularisation $-\beta H(b_t)$ is crucial: it prevents the market maker from becoming overconfident, which would make the policy brittle when the execution agent's strategy shifts.

---

## Latency Arbitrageur

The arbitrageur is the **temporal exploiter** — it acts faster than the market maker can update its quotes, extracting profit from belief-update lag.

### Role in System Dynamics

Without the arbitrageur, the market maker could adopt a trivially safe strategy (always widen spreads). The arbitrageur makes this costly:

- Wide spreads → stale quotes → sniping opportunities
- This forces the market maker to solve a genuine inference problem

### Observation Space

| Feature | Description |
|---|---|
| LOB snapshot (80-dim) | Current book state |
| `quote_staleness` | Proxy for how stale MM quotes are vs fair value |
| `arb_inventory` | Current position / max position |
| `mid_price` | Current mid / 100 |
| `time_remaining` | Step fraction |

### Action Space

Scalar $a \in [-1, 1]$:
- $a > 0.33$: lift best ask (buy aggressively)
- $a < -0.33$: hit best bid (sell aggressively)
- otherwise: do nothing

### Reward

$$r_A = \text{PnL}_\text{snipe} + \delta \cdot \text{BeliefLag}$$

The belief lag term rewards acting before the market maker's spread adjustment catches up to the new information — incentivising the arbitrageur to be maximally fast.
