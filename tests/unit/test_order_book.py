"""
Unit tests for the limit order book matching engine.

Tests price-time priority matching, partial fills, market orders,
spread computation, and snapshot generation.
"""

import numpy as np
import pytest

from adversarial_market.environment.order import Order, OrderStatus, OrderType, Side
from adversarial_market.environment.order_book import OrderBook

# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def empty_book():
    return OrderBook(tick_size=0.01, lot_size=100)


@pytest.fixture
def book_with_quotes():
    """Book pre-seeded with symmetric quotes around 100.00."""
    book = OrderBook(tick_size=0.01, lot_size=100)
    for i in range(1, 4):
        bid = Order(
            order_id=0,
            agent_id=1,
            side=Side.BID,
            order_type=OrderType.LIMIT,
            price=100.0 - i * 0.01,
            size=10,
        )
        ask = Order(
            order_id=0,
            agent_id=1,
            side=Side.ASK,
            order_type=OrderType.LIMIT,
            price=100.0 + i * 0.01,
            size=10,
        )
        book.submit(bid)
        book.submit(ask)
    return book


def make_limit(book, agent_id, side, price, size):
    order = Order(
        order_id=0, agent_id=agent_id, side=side, order_type=OrderType.LIMIT, price=price, size=size
    )
    return book.submit(order), order


def make_market(book, agent_id, side, size):
    order = Order(
        order_id=0, agent_id=agent_id, side=side, order_type=OrderType.MARKET, price=0.0, size=size
    )
    return book.submit(order), order


# ── Basic state ────────────────────────────────────────────────────────────


class TestBasicState:
    def test_empty_book_has_no_quotes(self, empty_book):
        assert empty_book.best_bid is None
        assert empty_book.best_ask is None
        assert empty_book.mid_price is None
        assert empty_book.spread is None

    def test_single_bid_rests(self, empty_book):
        _, order = make_limit(empty_book, 1, Side.BID, 99.95, 5)
        assert empty_book.best_bid == pytest.approx(99.95)
        assert empty_book.best_ask is None
        assert order.status == OrderStatus.OPEN

    def test_single_ask_rests(self, empty_book):
        _, order = make_limit(empty_book, 1, Side.ASK, 100.05, 5)
        assert empty_book.best_ask == pytest.approx(100.05)
        assert order.status == OrderStatus.OPEN

    def test_mid_price_and_spread(self, book_with_quotes):
        assert book_with_quotes.best_bid == pytest.approx(99.99)
        assert book_with_quotes.best_ask == pytest.approx(100.01)
        assert book_with_quotes.mid_price == pytest.approx(100.00)
        assert book_with_quotes.spread == pytest.approx(0.02)


# ── Matching ───────────────────────────────────────────────────────────────


class TestMatching:
    def test_limit_order_crosses_best_ask(self, book_with_quotes):
        """Aggressive limit buy at ask price should fill immediately."""
        fills, order = make_limit(book_with_quotes, 0, Side.BID, 100.01, 5)
        assert len(fills) == 1
        assert fills[0].price == pytest.approx(100.01)
        assert fills[0].quantity == 5
        assert order.status == OrderStatus.FILLED
        assert order.remaining == 0

    def test_limit_order_does_not_cross_rests(self, book_with_quotes):
        """Passive limit buy below best ask should rest."""
        fills, order = make_limit(book_with_quotes, 0, Side.BID, 99.95, 5)
        assert fills == []
        assert order.status == OrderStatus.OPEN
        assert order.remaining == 5

    def test_market_buy_fills_at_best_ask(self, book_with_quotes):
        fills, order = make_market(book_with_quotes, 0, Side.BID, 5)
        assert len(fills) >= 1
        assert fills[0].price == pytest.approx(100.01)
        assert order.status == OrderStatus.FILLED

    def test_market_sell_fills_at_best_bid(self, book_with_quotes):
        fills, order = make_market(book_with_quotes, 0, Side.ASK, 5)
        assert len(fills) >= 1
        assert fills[0].price == pytest.approx(99.99)
        assert order.status == OrderStatus.FILLED

    def test_partial_fill(self, book_with_quotes):
        """Buy 15 lots when best ask only has 10 → partial fill."""
        fills, order = make_market(book_with_quotes, 0, Side.BID, 15)
        total_filled = sum(f.quantity for f in fills)
        # Should fill at least 10 (first level) and try second level
        assert total_filled >= 10
        assert order.status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED)

    def test_market_order_cancelled_if_no_liquidity(self, empty_book):
        fills, order = make_market(empty_book, 0, Side.BID, 5)
        assert fills == []
        assert order.status == OrderStatus.CANCELLED

    def test_fill_price_is_passive_side_price(self, empty_book):
        """Price should be set by the resting (passive) order."""
        make_limit(empty_book, 1, Side.ASK, 100.05, 10)
        fills, _ = make_market(empty_book, 0, Side.BID, 5)
        assert fills[0].price == pytest.approx(100.05)

    def test_price_time_priority_fifo(self, empty_book):
        """Two orders at same price — earlier order should fill first."""
        _, o1 = make_limit(empty_book, 1, Side.ASK, 100.01, 5)
        _, o2 = make_limit(empty_book, 2, Side.ASK, 100.01, 5)
        fills, _ = make_market(empty_book, 0, Side.BID, 5)
        assert fills[0].passive_order_id == o1.order_id

    def test_better_price_fills_first(self, empty_book):
        """Ask at 100.01 should fill before ask at 100.05."""
        _, o_cheap = make_limit(empty_book, 1, Side.ASK, 100.01, 5)
        _, o_expensive = make_limit(empty_book, 2, Side.ASK, 100.05, 5)
        fills, _ = make_market(empty_book, 0, Side.BID, 5)
        assert fills[0].passive_order_id == o_cheap.order_id


# ── Cancellation ───────────────────────────────────────────────────────────


class TestCancellation:
    def test_cancel_resting_order(self, book_with_quotes):
        _, order = make_limit(book_with_quotes, 0, Side.BID, 99.90, 5)
        assert order.is_active
        result = book_with_quotes.cancel(order.order_id)
        assert result is True
        assert order.status == OrderStatus.CANCELLED

    def test_cancel_nonexistent_order(self, empty_book):
        result = empty_book.cancel(9999)
        assert result is False

    def test_cancelled_order_not_fillable(self, empty_book):
        _, ask = make_limit(empty_book, 1, Side.ASK, 100.01, 10)
        empty_book.cancel(ask.order_id)
        fills, _ = make_market(empty_book, 0, Side.BID, 5)
        assert fills == []


# ── Snapshot ───────────────────────────────────────────────────────────────


class TestSnapshot:
    def test_snapshot_shape(self, book_with_quotes):
        snap = book_with_quotes.get_snapshot(depth=10)
        assert snap.shape == (40,)  # depth * 4

    def test_snapshot_normalised(self, book_with_quotes):
        snap = book_with_quotes.get_snapshot(depth=10)
        # Sizes should be normalised to [0, 1]
        sizes = snap[1::2]
        assert sizes.max() <= 1.0 + 1e-6
        assert sizes.min() >= 0.0

    def test_snapshot_prices_relative_to_mid(self, book_with_quotes):
        snap = book_with_quotes.get_snapshot(depth=5)
        # Bid prices (first half) should be <= 0 (below mid)
        bid_prices = snap[:10:2]
        non_zero = bid_prices[bid_prices != 0]
        if len(non_zero):
            assert (non_zero <= 0).all()

    def test_empty_depth_returns_zeros(self, empty_book):
        snap = empty_book.get_snapshot(depth=5)
        assert np.allclose(snap, 0.0)


# ── VWAP ───────────────────────────────────────────────────────────────────


class TestVWAP:
    def test_vwap_no_fills_returns_none(self, empty_book):
        assert empty_book.get_vwap() is None

    def test_vwap_single_fill(self, book_with_quotes):
        make_market(book_with_quotes, 0, Side.BID, 5)
        vwap = book_with_quotes.get_vwap()
        assert vwap is not None
        assert 99.0 < vwap < 101.0

    def test_vwap_weighted_correctly(self, empty_book):
        """Manual VWAP check: 5@100 and 10@101 → VWAP = (500+1010)/15 = 100.67"""
        make_limit(empty_book, 1, Side.ASK, 100.0, 5)
        make_limit(empty_book, 1, Side.ASK, 101.0, 10)
        make_market(empty_book, 0, Side.BID, 15)
        vwap = empty_book.get_vwap()
        expected = (5 * 100.0 + 10 * 101.0) / 15
        assert vwap == pytest.approx(expected, rel=1e-4)
