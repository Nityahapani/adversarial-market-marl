"""
Order data structures for the limit order book environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional


class Side(IntEnum):
    BID = 1  # buy
    ASK = -1  # sell


class OrderType(IntEnum):
    MARKET = 0
    LIMIT = 1


class OrderStatus(IntEnum):
    OPEN = 0
    FILLED = 1
    PARTIALLY_FILLED = 2
    CANCELLED = 3


@dataclass
class Order:
    """
    Represents a single order submitted to the LOB.

    Attributes:
        order_id:     Unique identifier.
        agent_id:     Which agent submitted this order (0=exec, 1=mm, 2=arb, 3=noise).
        side:         BID (buy) or ASK (sell).
        order_type:   MARKET or LIMIT.
        price:        Limit price (ignored for market orders).
        size:         Original order quantity in lots.
        remaining:    Unfilled quantity in lots.
        timestamp:    Submission time (env step counter).
        status:       Current order status.
        fills:        List of (price, quantity) tuples from partial/full fills.
    """

    order_id: int
    agent_id: int
    side: Side
    order_type: OrderType
    price: float
    size: int
    remaining: int = field(init=False)
    timestamp: int = 0
    status: OrderStatus = field(default=OrderStatus.OPEN, init=False)
    fills: list[tuple[float, int]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.remaining = self.size

    @property
    def is_active(self) -> bool:
        return self.status in (OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED)

    @property
    def filled_quantity(self) -> int:
        return self.size - self.remaining

    @property
    def avg_fill_price(self) -> Optional[float]:
        if not self.fills:
            return None
        total_value = sum(p * q for p, q in self.fills)
        total_qty = sum(q for _, q in self.fills)
        return total_value / total_qty if total_qty > 0 else None

    def apply_fill(self, fill_price: float, fill_qty: int) -> None:
        """Update order state after a fill event."""
        assert fill_qty <= self.remaining, "Fill exceeds remaining quantity"
        self.fills.append((fill_price, fill_qty))
        self.remaining -= fill_qty
        if self.remaining == 0:
            self.status = OrderStatus.FILLED
        else:
            self.status = OrderStatus.PARTIALLY_FILLED

    def cancel(self) -> None:
        self.status = OrderStatus.CANCELLED

    def __repr__(self) -> str:
        return (
            f"Order(id={self.order_id}, agent={self.agent_id}, "
            f"side={self.side.name}, type={self.order_type.name}, "
            f"price={self.price:.2f}, size={self.size}, remaining={self.remaining}, "
            f"status={self.status.name})"
        )


@dataclass
class Fill:
    """Record of a matched trade."""

    aggressive_order_id: int
    passive_order_id: int
    price: float
    quantity: int
    timestamp: int
    aggressive_side: Side
