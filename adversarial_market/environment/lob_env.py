"""
Limit Order Book multi-agent environment.

Wraps the order book matching engine into a Gymnasium-compatible
multi-agent environment. Agents act sequentially within each step;
noise traders fire between agent actions to maintain realistic flow.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from adversarial_market.environment.market_state import AgentState, MarketState
from adversarial_market.environment.order import Fill, Order, OrderType, Side
from adversarial_market.environment.order_book import OrderBook

# Agent IDs
EXEC_ID = 0
MM_ID = 1
ARB_ID = 2
NOISE_ID = 3


class LOBEnvironment(gym.Env):
    """
    Multi-agent limit order book environment.

    Observation spaces (per agent):
        Execution agent  : [lob_snapshot | own_state | time_features]
        Market maker     : [lob_snapshot | flow_history | own_state | belief]
        Arbitrageur      : [lob_snapshot | quote_staleness | own_state]

    Action spaces:
        Execution agent  : Box([order_size, limit_offset, order_type_logit])
        Market maker     : Box([bid_offset_ticks, ask_offset_ticks, bid_size, ask_size])
        Arbitrageur      : Discrete (do nothing | hit bid | lift ask)

    The environment produces observations, rewards, and done flags for
    all three agents. The trainer coordinates the action loop.
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.cfg = config
        env_cfg = config["environment"]
        agent_cfg = config["agents"]

        self.tick_size: float = env_cfg["tick_size"]
        self.lot_size: int = env_cfg["lot_size"]
        self.max_steps: int = env_cfg["max_steps_per_episode"]
        self.initial_mid: float = env_cfg["initial_mid_price"]
        self.fundamental_value: float = env_cfg["fundamental_value"]
        self.fundamental_vol: float = env_cfg["fundamental_vol"]
        self.noise_arrival_rate: float = env_cfg["noise_trader_arrival_rate"]
        self.noise_size_mean: float = env_cfg["noise_order_size_mean"]
        self.noise_size_std: float = env_cfg["noise_order_size_std"]
        self.lob_depth: int = env_cfg["n_price_levels"]
        self.latency: Dict[str, float] = env_cfg["latency_ms"]

        # Execution agent config
        exec_cfg = agent_cfg["execution"]
        self.exec_inventory: int = exec_cfg["inventory_lots"]
        self.exec_horizon: int = exec_cfg["horizon"]
        self.exec_max_order: int = exec_cfg["max_order_size"]
        self.lambda_leakage: float = exec_cfg["lambda_leakage"]
        self.mu_predictability: float = exec_cfg["mu_predictability"]
        self.terminal_penalty: float = exec_cfg["terminal_inventory_penalty"]

        # Market maker config
        mm_cfg = agent_cfg["market_maker"]
        self.mm_max_spread: int = mm_cfg["max_spread_ticks"]
        self.mm_max_inv: int = mm_cfg["max_inventory_lots"]
        self.mm_max_quote: int = mm_cfg["max_quote_size_lots"]

        # Arb config
        arb_cfg = agent_cfg["arbitrageur"]
        self.arb_max_pos: int = arb_cfg["max_position_lots"]

        # Spaces
        lob_dim = self.lob_depth * 4
        self.observation_space = self._build_obs_spaces(lob_dim)  # type: ignore[assignment]
        self.action_space = self._build_action_spaces()  # type: ignore[assignment]

        # Runtime state (reset on each episode)
        # Initialized here; fully reset on every reset() call
        self.book: OrderBook = OrderBook(tick_size=self.tick_size, lot_size=self.lot_size)
        self.state: MarketState = MarketState(
            step=0,
            max_steps=self.max_steps,
            mid_price=self.initial_mid,
            exec_initial_inventory=self.exec_inventory,
            exec_remaining_inventory=self.exec_inventory,
        )
        self._step: int = 0
        self._rng = np.random.default_rng()

        # Flow history for MI estimation (execution agent actions only)
        self._exec_flow_history: List[np.ndarray] = []
        # Track order flow feature vectors for MINE
        self._flow_buffer: List[np.ndarray] = []

        # TWAP benchmark (for implementation shortfall)
        self._twap_prices: List[float] = []
        self._arrival_price: float = 0.0

    # ── Gymnasium interface ────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.book = OrderBook(tick_size=self.tick_size, lot_size=self.lot_size)
        self._step = 0
        self._exec_flow_history = []
        self._flow_buffer = []
        self._twap_prices = []

        # Initialise market state
        self.state = MarketState(
            step=0,
            max_steps=self.max_steps,
            mid_price=self.initial_mid,
            exec_initial_inventory=self.exec_inventory,
            exec_remaining_inventory=self.exec_inventory,
        )
        self.state.agent_states = {
            EXEC_ID: AgentState(EXEC_ID),
            MM_ID: AgentState(MM_ID),
            ARB_ID: AgentState(ARB_ID),
        }

        self._arrival_price = self.initial_mid

        # Seed the book with initial quotes from a simple market-maker stub
        self._seed_initial_quotes()
        self._update_state()

        obs = self._get_observations()
        return obs, {}

    def step(
        self,
        actions: Dict[str, np.ndarray],
    ) -> Any:  # type: ignore[override]
        """
        Process one environment step.

        Args:
            actions: {"execution": array, "market_maker": array, "arbitrageur": array}

        Returns:
            obs, rewards, terminated, truncated, info (all dicts keyed by agent name)
        """
        assert self.book is not None and self.state is not None

        self._step += 1
        self.book.step()

        # Fire noise traders first (background liquidity)
        self._fire_noise_traders()

        # Process MM quotes (they set the passive side before active traders)
        mm_fills = self._process_mm_action(actions.get("market_maker", np.zeros(4)))

        # Process execution agent order
        exec_fills = self._process_exec_action(actions.get("execution", np.zeros(3)))

        # Process arbitrageur (reacts to stale quotes after agents have acted)
        arb_fills = self._process_arb_action(actions.get("arbitrageur", np.zeros(1)))

        # Update fundamental value (mean-reverting Ornstein-Uhlenbeck)
        self._update_fundamental()

        # Update market state
        self._update_state()
        self._twap_prices.append(self.state.mid_price)

        # Compute rewards
        rewards = self._compute_rewards(exec_fills, mm_fills, arb_fills)

        # Check termination
        done = self._step >= self.max_steps or self.state.exec_remaining_inventory <= 0
        terminated = {"execution": done, "market_maker": done, "arbitrageur": done}
        truncated = {"execution": False, "market_maker": False, "arbitrageur": False}

        if done:
            rewards["execution"] += self._terminal_exec_reward()

        obs = self._get_observations()
        info = self._get_info(exec_fills, mm_fills, arb_fills)

        return obs, rewards, terminated, truncated, info

    # ── Action processing ─────────────────────────────────────────────────

    def _process_exec_action(self, action: np.ndarray) -> List:
        """
        Execution agent action: [size_frac, limit_offset_ticks, order_type_logit]
          size_frac ∈ [0,1]: fraction of max_order_size to trade
          limit_offset_ticks ∈ [-5,5]: offset from mid (negative = aggressive)
          order_type_logit: >0 → LIMIT, <=0 → MARKET
        """
        size = max(1, int(action[0] * self.exec_max_order))
        size = min(size, self.state.exec_remaining_inventory)
        if size <= 0:
            return []

        use_limit = action[2] > 0
        mid = self.state.mid_price

        if use_limit:
            offset = float(np.clip(action[1], -5, 5)) * self.tick_size
            # Execution agent is buying (positive inventory = buy order)
            price = mid + offset
            order_type = OrderType.LIMIT
        else:
            price = mid  # market order — price irrelevant
            order_type = OrderType.MARKET

        # Execution agent always buys (positive inventory target)
        order = Order(
            order_id=0,
            agent_id=EXEC_ID,
            side=Side.BID,
            order_type=order_type,
            price=price,
            size=size,
            timestamp=self._step,
        )
        fills = self.book.submit(order)

        for fill in fills:
            self.state.agent_states[EXEC_ID].update_fill(fill.price, fill.quantity, Side.BID)
            self.state.exec_remaining_inventory -= fill.quantity

        # Record flow feature for MI estimation
        flow_feat = np.array(
            [
                float(size) / self.exec_max_order,
                float(action[1]),
                float(use_limit),
                float(self._step) / self.max_steps,
            ],
            dtype=np.float32,
        )
        self._flow_buffer.append(flow_feat)
        if len(self._flow_buffer) > 500:
            self._flow_buffer.pop(0)

        return fills

    def _process_mm_action(self, action: np.ndarray) -> List:
        """
        Market maker action: [bid_offset, ask_offset, bid_size_frac, ask_size_frac]
        Cancels all previous MM quotes and posts new ones.
        """
        mid = self.state.mid_price
        bid_offset = float(np.clip(action[0], -self.mm_max_spread, 0)) * self.tick_size
        ask_offset = float(np.clip(action[1], 0, self.mm_max_spread)) * self.tick_size
        bid_size = max(1, int(action[2] * self.mm_max_quote))
        ask_size = max(1, int(action[3] * self.mm_max_quote))

        # Check inventory limits before quoting
        mm_inv = self.state.agent_states[MM_ID].inventory
        if mm_inv >= self.mm_max_inv:
            bid_size = 0
        if mm_inv <= -self.mm_max_inv:
            ask_size = 0

        fills: List[Fill] = []
        if bid_size > 0:
            bid = Order(
                order_id=0,
                agent_id=MM_ID,
                side=Side.BID,
                order_type=OrderType.LIMIT,
                price=mid + bid_offset,
                size=bid_size,
                timestamp=self._step,
            )
            fills += self.book.submit(bid)

        if ask_size > 0:
            ask = Order(
                order_id=0,
                agent_id=MM_ID,
                side=Side.ASK,
                order_type=OrderType.LIMIT,
                price=mid + ask_offset,
                size=ask_size,
                timestamp=self._step,
            )
            fills += self.book.submit(ask)

        for fill in fills:
            side = fill.aggressive_side
            passive_side = Side.ASK if side == Side.BID else Side.BID
            self.state.agent_states[MM_ID].update_fill(fill.price, fill.quantity, passive_side)

        return fills

    def _process_arb_action(self, action: np.ndarray) -> List:
        """
        Arbitrageur action: scalar in [-1, 1]
          < -0.33: hit best bid (sell aggressively)
          > +0.33: lift best ask (buy aggressively)
          else: do nothing
        """
        a = float(action[0])
        fills: List[Fill] = []
        arb_inv = self.state.agent_states[ARB_ID].inventory

        if a > 0.33 and arb_inv < self.arb_max_pos:
            # Lift ask
            ask = self.book.best_ask
            if ask is not None:
                order = Order(
                    order_id=0,
                    agent_id=ARB_ID,
                    side=Side.BID,
                    order_type=OrderType.MARKET,
                    price=ask,
                    size=1,
                    timestamp=self._step,
                )
                fills = self.book.submit(order)
                for f in fills:
                    self.state.agent_states[ARB_ID].update_fill(f.price, f.quantity, Side.BID)

        elif a < -0.33 and arb_inv > -self.arb_max_pos:
            # Hit bid
            bid = self.book.best_bid
            if bid is not None:
                order = Order(
                    order_id=0,
                    agent_id=ARB_ID,
                    side=Side.ASK,
                    order_type=OrderType.MARKET,
                    price=bid,
                    size=1,
                    timestamp=self._step,
                )
                fills = self.book.submit(order)
                for f in fills:
                    self.state.agent_states[ARB_ID].update_fill(f.price, f.quantity, Side.ASK)

        return fills

    def _fire_noise_traders(self) -> None:
        """Poisson-arrival noise traders provide background liquidity and flow."""
        n_orders = self._rng.poisson(self.noise_arrival_rate / self.max_steps * 10)
        mid = self.state.mid_price if self.state else self.initial_mid

        for _ in range(n_orders):
            side = Side.BID if self._rng.random() < 0.5 else Side.ASK
            size = max(1, int(self._rng.normal(self.noise_size_mean, self.noise_size_std)))
            price_offset = self._rng.uniform(-3, 3) * self.tick_size
            price = mid + price_offset

            order = Order(
                order_id=0,
                agent_id=NOISE_ID,
                side=side,
                order_type=OrderType.LIMIT,
                price=price,
                size=size,
                timestamp=self._step,
            )
            self.book.submit(order)

    # ── Reward computation ────────────────────────────────────────────────

    def _compute_rewards(
        self, exec_fills: List[Fill], mm_fills: List[Fill], arb_fills: List[Fill]
    ) -> Dict[str, float]:
        mid = self.state.mid_price

        # ── Execution agent ──
        exec_is = self._implementation_shortfall(exec_fills)
        exec_reward = -exec_is
        # MI penalty is computed externally by MINE estimator and injected
        # during the training update; here we return the base IS reward.

        # ── Market maker ──
        mm_pnl = sum(
            f.price * f.quantity * (1 if f.aggressive_side == Side.BID else -1) for f in mm_fills
        )
        # Adverse selection: post-fill price movement against MM position
        mm_inv = self.state.agent_states[MM_ID].inventory
        adv_sel = (
            mm_inv
            * (mid - self.state.mid_price)
            * self.cfg["agents"]["market_maker"]["alpha_adverse_selection"]
        )
        mm_reward = mm_pnl - adv_sel

        # ── Arbitrageur ──
        arb_pnl = sum(
            f.price * f.quantity * (1 if f.aggressive_side == Side.ASK else -1) for f in arb_fills
        )
        # Mark arb position to market
        arb_inv = self.state.agent_states[ARB_ID].inventory
        arb_mtm = arb_inv * mid
        arb_reward = arb_pnl + arb_mtm * 0.01  # small fraction of unrealised

        return {
            "execution": float(exec_reward),
            "market_maker": float(mm_reward),
            "arbitrageur": float(arb_reward),
        }

    def _implementation_shortfall(self, fills: List[Fill]) -> float:
        """IS = arrival_price * total_qty - sum(fill_price * fill_qty)"""
        if not fills:
            return 0.0
        total_cost = sum(f.price * f.quantity for f in fills)
        total_qty = sum(f.quantity for f in fills)
        return total_cost - self._arrival_price * total_qty

    def _terminal_exec_reward(self) -> float:
        remaining = self.state.exec_remaining_inventory
        if remaining <= 0:
            return 0.0
        return -self.terminal_penalty * remaining

    # ── State updates ──────────────────────────────────────────────────────

    def _update_state(self) -> None:
        mid = self.book.mid_price
        if mid is not None:
            self.state.mid_price = mid
        self.state.best_bid = self.book.best_bid
        self.state.best_ask = self.book.best_ask
        self.state.spread = self.book.spread
        self.state.lob_snapshot = self.book.get_snapshot(self.lob_depth)
        self.state.recent_fills = self.book.get_recent_fills(50)
        self.state.step = self._step
        self.state.update_price_history()

        for agent_state in self.state.agent_states.values():
            agent_state.mark_to_market(self.state.mid_price)

    def _update_fundamental(self) -> None:
        """Ornstein-Uhlenbeck mean-reversion for fundamental value."""
        dt = 1.0 / self.max_steps
        theta = 0.1  # mean-reversion speed
        sigma = self.fundamental_vol * np.sqrt(dt)
        dF = theta * (self.fundamental_value - self.state.mid_price) * dt
        dF += sigma * self._rng.normal()
        # Nudge best ask/bid slightly toward fundamental (simplified)
        self.fundamental_value += dF

    def _seed_initial_quotes(self) -> None:
        """Post initial two-sided quotes to give the book starting liquidity."""
        mid = self.initial_mid
        for i in range(1, 6):
            for size in [5, 3, 2]:
                bid = Order(
                    order_id=0,
                    agent_id=MM_ID,
                    side=Side.BID,
                    order_type=OrderType.LIMIT,
                    price=mid - i * self.tick_size,
                    size=size,
                    timestamp=0,
                )
                ask = Order(
                    order_id=0,
                    agent_id=MM_ID,
                    side=Side.ASK,
                    order_type=OrderType.LIMIT,
                    price=mid + i * self.tick_size,
                    size=size,
                    timestamp=0,
                )
                self.book.submit(bid)
                self.book.submit(ask)

    # ── Observation builders ───────────────────────────────────────────────

    def _get_observations(self) -> Dict[str, np.ndarray]:  # type: ignore[return]
        s = self.state
        lob = s.lob_snapshot

        exec_obs = np.concatenate(
            [
                lob,
                [
                    s.exec_remaining_inventory / max(s.exec_initial_inventory, 1),
                    s.time_remaining_frac,
                    s.exec_completion_frac,
                    s.mid_price / 100.0,
                    s.spread / s.mid_price if s.spread else 0.02,
                    s.realized_volatility,
                ],
            ]
        ).astype(np.float32)

        # MM observation includes flow history features
        flow_feats = self._encode_recent_flow(n=20)
        mm_obs = np.concatenate(
            [
                lob,
                flow_feats,
                [
                    s.agent_states[MM_ID].inventory / self.mm_max_inv,
                    s.mid_price / 100.0,
                    s.spread / s.mid_price if s.spread else 0.02,
                    s.mm_belief,
                    s.time_remaining_frac,
                ],
            ]
        ).astype(np.float32)

        # Arb observation: LOB + staleness signal + position
        quote_staleness = self._quote_staleness()
        arb_obs = np.concatenate(
            [
                lob,
                [
                    quote_staleness,
                    s.agent_states[ARB_ID].inventory / self.arb_max_pos,
                    s.mid_price / 100.0,
                    s.time_remaining_frac,
                ],
            ]
        ).astype(np.float32)

        return {
            "execution": exec_obs,
            "market_maker": mm_obs,
            "arbitrageur": arb_obs,
        }

    def _encode_recent_flow(self, n: int = 20) -> np.ndarray:
        """Encode last n fills as a flat feature vector for MM observation."""
        fills = self.book.get_recent_fills(n)
        feat = np.zeros(n * 3, dtype=np.float32)
        for i, f in enumerate(fills[-n:]):
            feat[3 * i] = float(f.aggressive_side)
            feat[3 * i + 1] = f.price / self.state.mid_price - 1.0
            feat[3 * i + 2] = float(f.quantity) / self.mm_max_quote
        return feat  # type: ignore[return-value]

    def _quote_staleness(self) -> float:
        """Proxy for how stale MM quotes are vs current mid."""
        bb, ba = self.book.best_bid, self.book.best_ask
        if bb is None or ba is None:
            return 0.0
        fair_spread = self.tick_size * 2
        current_spread = ba - bb
        return float(np.clip((current_spread - fair_spread) / fair_spread, 0, 5))

    def get_flow_buffer(self) -> np.ndarray:
        """Return raw flow feature buffer for external MINE estimation."""
        if not self._flow_buffer:
            return np.empty((0, 4), dtype=np.float32)
        return np.array(self._flow_buffer, dtype=np.float32)

    # ── Space builders ─────────────────────────────────────────────────────

    def _build_obs_spaces(self, lob_dim: int) -> Dict[str, Any]:
        return {
            "execution": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(lob_dim + 6,),
                dtype=np.float32,
            ),
            "market_maker": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(lob_dim + 20 * 3 + 5,),
                dtype=np.float32,
            ),
            "arbitrageur": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(lob_dim + 4,),
                dtype=np.float32,
            ),
        }

    def _build_action_spaces(self) -> Dict[str, Any]:
        return {
            "execution": spaces.Box(
                low=np.array([0.0, -5.0, -1.0]),
                high=np.array([1.0, 5.0, 1.0]),
                dtype=np.float32,
            ),
            "market_maker": spaces.Box(
                low=np.array([-float(self.mm_max_spread), 0.0, 0.0, 0.0]),
                high=np.array([0.0, float(self.mm_max_spread), 1.0, 1.0]),
                dtype=np.float32,
            ),
            "arbitrageur": spaces.Box(
                low=np.array([-1.0]),
                high=np.array([1.0]),
                dtype=np.float32,
            ),
        }

    def _get_info(
        self, exec_fills: List[Fill], mm_fills: List[Fill], arb_fills: List[Fill]
    ) -> Dict[str, Any]:
        return {
            "execution": {
                "fills": len(exec_fills),
                "remaining": self.state.exec_remaining_inventory,
                "mid_price": self.state.mid_price,
            },
            "market_maker": {
                "fills": len(mm_fills),
                "inventory": self.state.agent_states[MM_ID].inventory,
                "belief": self.state.mm_belief,
            },
            "arbitrageur": {
                "fills": len(arb_fills),
                "inventory": self.state.agent_states[ARB_ID].inventory,
            },
        }
