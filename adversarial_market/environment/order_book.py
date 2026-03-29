"""
Limit order book with price-time priority matching.

Maintains separate bid and ask sides as sorted price-level queues.
Supports market orders, limit orders, and partial fills.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional

import numpy as np

from adversarial_market.environment.order import Fill, Order, OrderType, Side


class PriceLevel:
    """All resting limit orders at a single price, FIFO ordered."""

    def __init__(self, price: float) -> None:
        self.price = price
        self.orders: deque[Order] = deque()
        self.total_size: int = 0

    def add(self, order: Order) -> None:
        self.orders.append(order)
        self.total_size += order.remaining

    def remove_filled(self) -> None:
        while self.orders and not self.orders[0].is_active:
            self.orders.popleft()

    def is_empty(self) -> bool:
        return self.total_size == 0 or not self.orders

    def __repr__(self) -> str:
        return f"PriceLevel(price={self.price:.2f}, size={self.total_size})"


class OrderBook:
    """
    Central limit order book with price-time priority matching.

    Price levels:
        - Bids sorted descending (best bid = highest price)
        - Asks sorted ascending  (best ask = lowest price)

    Matching algorithm:
        For each incoming order, walk through resting levels on the opposite
        side in price-time priority until the order is filled or no more
        matching quotes exist. Limit orders that are not fully matched rest
        at their price level.
    """

    def __init__(self, tick_size: float = 0.01, lot_size: int = 100) -> None:
        self.tick_size = tick_size
        self.lot_size = lot_size
        # price → PriceLevel
        self._bids: Dict[float, PriceLevel] = {}
        self._asks: Dict[float, PriceLevel] = {}
        self._order_map: Dict[int, Order] = {}
        self._next_order_id: int = 0
        self._fill_history: List[Fill] = []
        self._step: int = 0

    # ── Public API ────────────────────────────────────────────────────────

    def submit(self, order: Order) -> List[Fill]:
        """
        Submit an order. Returns list of Fill events generated.
        Market orders fill immediately or are cancelled if no liquidity.
        Limit orders rest if not immediately matchable.
        """
        order.order_id = self._next_order_id
        self._next_order_id += 1
        order.timestamp = self._step
        self._order_map[order.order_id] = order

        fills = self._match(order)
        if order.is_active and order.order_type == OrderType.LIMIT:
            self._add_to_book(order)
        elif order.is_active and order.order_type == OrderType.MARKET:
            # Market order: cancel any unfilled remainder
            order.cancel()

        self._fill_history.extend(fills)
        return fills

    def cancel(self, order_id: int) -> bool:
        """Cancel a resting order. Returns True if found and cancelled."""
        order = self._order_map.get(order_id)
        if order is None or not order.is_active:
            return False
        order.cancel()
        side_book = self._bids if order.side == Side.BID else self._asks
        level = side_book.get(order.price)
        if level:
            level.total_size -= order.remaining
            level.remove_filled()
        return True

    def step(self) -> None:
        """Advance internal clock by one step."""
        self._step += 1

    # ── Book state queries ────────────────────────────────────────────────

    @property
    def best_bid(self) -> Optional[float]:
        bids = [p for p, lv in self._bids.items() if not lv.is_empty()]
        return max(bids) if bids else None

    @property
    def best_ask(self) -> Optional[float]:
        asks = [p for p, lv in self._asks.items() if not lv.is_empty()]
        return min(asks) if asks else None

    @property
    def mid_price(self) -> Optional[float]:
        bb, ba = self.best_bid, self.best_ask
        if bb is not None and ba is not None:
            return (bb + ba) / 2.0
        return None

    @property
    def spread(self) -> Optional[float]:
        bb, ba = self.best_bid, self.best_ask
        if bb is not None and ba is not None:
            return ba - bb
        return None

    def get_snapshot(self, depth: int = 10) -> np.ndarray:
        """
        Returns LOB snapshot as numpy array of shape (depth * 4,):
            [bid_price_1, bid_size_1, ..., bid_price_depth, bid_size_depth,
             ask_price_1, ask_size_1, ..., ask_price_depth, ask_size_depth]
        Prices normalised by current mid; sizes normalised by max observable.
        """
        mid = self.mid_price or 100.0

        bid_levels = sorted(
            [(p, lv.total_size) for p, lv in self._bids.items() if not lv.is_empty()],
            reverse=True,
        )[:depth]
        ask_levels = sorted(
            [(p, lv.total_size) for p, lv in self._asks.items() if not lv.is_empty()]
        )[:depth]

        snap = np.zeros(depth * 4, dtype=np.float32)
        for i, (p, s) in enumerate(bid_levels):
            snap[2 * i] = (p - mid) / mid
            snap[2 * i + 1] = float(s)
        offset = depth * 2
        for i, (p, s) in enumerate(ask_levels):
            snap[offset + 2 * i] = (p - mid) / mid
            snap[offset + 2 * i + 1] = float(s)

        # Normalise sizes
        max_size = snap[1::2].max()
        if max_size > 0:
            snap[1::2] /= max_size

        return snap

    def get_recent_fills(self, n: int = 20) -> List[Fill]:
        return self._fill_history[-n:]

    def get_vwap(self, n_fills: int = 50) -> Optional[float]:
        fills = self._fill_history[-n_fills:]
        if not fills:
            return None
        total_val = sum(f.price * f.quantity for f in fills)
        total_qty = sum(f.quantity for f in fills)
        return total_val / total_qty if total_qty > 0 else None

    def total_bid_depth(self) -> int:
        return sum(lv.total_size for lv in self._bids.values() if not lv.is_empty())

    def total_ask_depth(self) -> int:
        return sum(lv.total_size for lv in self._asks.values() if not lv.is_empty())

    # ── Internal matching ─────────────────────────────────────────────────

    def _match(self, order: Order) -> List[Fill]:
        fills: List[Fill] = []
        if order.side == Side.BID:
            opposite = self._asks
            price_ok = (
                lambda resting_price: resting_price <= order.price
                or order.order_type == OrderType.MARKET
            )
            levels = sorted(opposite.keys())
        else:
            opposite = self._bids
            price_ok = (
                lambda resting_price: resting_price >= order.price
                or order.order_type == OrderType.MARKET
            )
            levels = sorted(opposite.keys(), reverse=True)

        for level_price in levels:
            if order.remaining == 0:
                break
            if not price_ok(level_price):
                break
            level = opposite[level_price]
            level.remove_filled()
            if level.is_empty():
                continue

            for resting in list(level.orders):
                if order.remaining == 0:
                    break
                if not resting.is_active:
                    continue

                fill_qty = min(order.remaining, resting.remaining)
                fill_price = resting.price  # passive side sets the price

                order.apply_fill(fill_price, fill_qty)
                resting.apply_fill(fill_price, fill_qty)
                level.total_size -= fill_qty

                fill = Fill(
                    aggressive_order_id=order.order_id,
                    passive_order_id=resting.order_id,
                    price=fill_price,
                    quantity=fill_qty,
                    timestamp=self._step,
                    aggressive_side=order.side,
                )
                fills.append(fill)

            level.remove_filled()
            if level.is_empty():
                del opposite[level_price]

        return fills

    def _add_to_book(self, order: Order) -> None:
        side_book = self._bids if order.side == Side.BID else self._asks
        price = self._round_price(order.price)
        if price not in side_book:
            side_book[price] = PriceLevel(price)
        side_book[price].add(order)

    def _round_price(self, price: float) -> float:
        return round(round(price / self.tick_size) * self.tick_size, 10)

    def __repr__(self) -> str:
        return (
            f"OrderBook(best_bid={self.best_bid}, best_ask={self.best_ask}, "
            f"spread={self.spread}, step={self._step})"
        )
